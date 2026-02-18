# Analytical Reachability via Closed-Form Trajectory Equations

**Document Version:** 2.0  
**Last Updated:** February 15, 2026  
**Purpose:** Replace simulation-based reachability with exact analytical trajectory computation for PBRS subgoal selection

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Physics Recurrence and Closed-Form Derivation](#2-physics-recurrence-and-closed-form-derivation)
3. [Two-Phase Jump Trajectory](#3-two-phase-jump-trajectory)
4. [Horizontal Axis with Speed Cap](#4-horizontal-axis-with-speed-cap)
5. [Affine Velocity Decomposition](#5-affine-velocity-decomposition)
6. [Trajectory Catalog Precomputation](#6-trajectory-catalog-precomputation)
7. [Scatter-and-Collect Reachability](#7-scatter-and-collect-reachability)
8. [State-Specific Trajectory Sets](#8-state-specific-trajectory-sets)
9. [Translation Invariance](#9-translation-invariance)
10. [Runtime Integration with Level Geometry](#10-runtime-integration-with-level-geometry)
11. [PBRS Subgoal Algorithm](#11-pbrs-subgoal-algorithm)
12. [Data Structures](#12-data-structures)
13. [Computational Complexity](#13-computational-complexity)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Physics Constants Reference](#15-physics-constants-reference)

## 1. Executive Summary

### The Breakthrough

The N++ physics engine applies drag, gravity, and acceleration in a fixed order each frame with constant coefficients. This creates a **linear recurrence relation** on velocity, which has an exact closed-form solution involving geometric series. Since position is the cumulative sum of velocity, it also has a closed-form solution. No frame-by-frame simulation is required.

This means: given the ninja's current state (position, velocity, contact type, surface normal, jump frames remaining), we can compute the **exact position and velocity at any future frame n** with O(1) arithmetic — two evaluations of dⁿ (one power operation) and a handful of multiplications and additions.

### The Affine Property

A deeper consequence of the closed-form solution: the endpoint position of any trajectory is **affine in the initial velocity**. For a fixed control sequence (jump hold duration, horizontal inputs), the landing position is:

```
px_land = Cx · vx₀ + Bx        (Cx, Bx are constants depending only on control sequence)
py_land = Cy · vy₀ + By        (Cy, By are constants depending only on control sequence)
```

This eliminates the need for velocity-binned hull catalogs entirely. We precompute a single set of trajectory entries (one per control sequence), each storing its affine coefficients (Cx, Bx, Cy, By). At runtime, evaluating where each trajectory lands for any continuous velocity is one multiply-add per trajectory — no binning, no interpolation, no quantization error.

### Architecture Overview

```
PRECOMPUTATION (once, offline, level-independent):
  For each control sequence (j, h₁, h₂):
    Solve landing frame analytically
    Compute affine coefficients (Cx, Bx, Cy, By)
    Compute speed cap threshold per trajectory
    Store as trajectory catalog entry (~414 entries)

PER-LEVEL SETUP (once per level, already exists):
  Extract surface points with spatial hash
  Compute A* distances from each surface to switch/exit

RUNTIME (per frame, per environment, during training):
  1. Read ninja state (position, velocity, contact type, surface normal)
  2. For each trajectory in catalog (~414 entries):
     a. Compute landing point: land_x = ninja_x + Bx + vx₀·Cx (one mul-add)
     b. Look up nearby surfaces in spatial hash (O(1) amortized)
  3. Among all reachable surfaces, find closest to goal (A* distance)
  4. Φ(s) = −distance(subgoal, goal) / normalization
  5. PBRS reward: F_t = γΦ(s') − Φ(s)
```

**Runtime cost:** ~2–5μs per environment. At 128 environments: ~0.5ms per training step.

---

## 2. Physics Recurrence and Closed-Form Derivation

### Integration Order

From `sim_mechanics_doc.md`, the per-frame physics integration order is:

```
1. Apply drag:     v ← v × d
2. Apply gravity:  v_y ← v_y + g       (or accel for horizontal: v_x ← v_x + a)
3. Update position: p ← p + v
4. Collision resolution (ignored for empty-map computation)
```

Where:
- `d` = DRAG_REGULAR = 0.9933221725495059
- `g` = GRAVITY_JUMP (0.01111...) during jump hold, GRAVITY_FALL (0.06666...) after release
- `a` = h × AIR_ACCEL = h × 0.04444... for horizontal input h ∈ {−1, 0, +1}

### Velocity Recurrence

The drag-then-force order gives the recurrence:

```
v(n) = d · v(n−1) + c
```

Where `c` is the constant force term (gravity for vertical, acceleration for horizontal). This is a **first-order linear recurrence with constant coefficients**.

### Closed-Form Velocity

**Derivation:** Unroll the recurrence:

```
v(1) = d·v₀ + c
v(2) = d·v(1) + c = d²·v₀ + d·c + c
v(3) = d·v(2) + c = d³·v₀ + d²·c + d·c + c
...
v(n) = dⁿ·v₀ + c·Σᵢ₌₀ⁿ⁻¹ dⁱ
     = dⁿ·v₀ + c·(dⁿ − 1)/(d − 1)
```

**Final velocity formula:**

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   v(n) = dⁿ · v₀  +  c · (dⁿ − 1) / (d − 1)      │
│                                                     │
│   where:  d = DRAG_REGULAR                          │
│           c = force constant (gravity or h·accel)   │
│           v₀ = initial velocity on this axis        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Terminal velocity** (n → ∞): v_∞ = c / (1 − d). This is the velocity at which drag exactly cancels the applied force.

For vertical free-fall: v_∞ = GRAVITY_FALL / (1 − DRAG_REGULAR) = 0.06666... / 0.006678... ≈ **9.98 px/frame** (well above the 6.0 death threshold — terminal velocity is always lethal from sufficient height).

For horizontal air control: v_∞ = AIR_ACCEL / (1 − DRAG_REGULAR) = 0.04444... / 0.006678... ≈ **6.65 px/frame** (above MAX_HOR_SPEED cap of 3.333, so the cap will always bind before terminal velocity).

### Closed-Form Position

Position is the cumulative sum of post-update velocities:

```
p(n) = p₀ + Σₖ₌₁ⁿ v(k)
```

Substituting the velocity formula and evaluating each sum:

```
Σₖ₌₁ⁿ dᵏ = d·(dⁿ − 1)/(d − 1)

Σₖ₌₁ⁿ (dᵏ − 1)/(d − 1) = [d·(dⁿ − 1)/(d − 1) − n] / (d − 1)
```

**Final position formula:**

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│                    d·(dⁿ − 1)         c     ⎡ d·(dⁿ − 1)      ⎤    │
│   p(n) = p₀ + v₀·─────────── + ───────── · ⎢ ─────────── − n  ⎥    │
│                      d − 1       (d − 1)    ⎣    d − 1         ⎦    │
│                                                                      │
│   Equivalently, defining:                                            │
│     S₁(n) = d·(dⁿ − 1)/(d − 1)           (geometric sum of dᵏ)     │
│     S₂(n) = (S₁(n) − n)/(d − 1)          (double geometric sum)    │
│                                                                      │
│   p(n) = p₀ + v₀·S₁(n) + c·S₂(n)                                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Helper Functions

Define these as reusable building blocks:

```python
import math

D = 0.9933221725495059  # DRAG_REGULAR
D_MINUS_1 = D - 1.0     # ≈ -0.006678 (negative, since d < 1)

def S1(n: int) -> float:
    """Geometric sum: Σₖ₌₁ⁿ dᵏ = d·(dⁿ − 1)/(d − 1)"""
    return D * (D**n - 1.0) / D_MINUS_1

def S2(n: int) -> float:
    """Double geometric sum: Σₖ₌₁ⁿ Σⱼ₌₁ᵏ dʲ  (cumulative position contribution)"""
    return (S1(n) - n) / D_MINUS_1

def velocity_at(v0: float, c: float, n: int) -> float:
    """Velocity at frame n given initial velocity v0 and constant force c."""
    dn = D**n
    return dn * v0 + c * (dn - 1.0) / D_MINUS_1

def position_at(p0: float, v0: float, c: float, n: int) -> float:
    """Position at frame n given initial position p0, velocity v0, force c."""
    return p0 + v0 * S1(n) + c * S2(n)
```

### Verification

These formulas must be verified against the simulator output before any further work. Run the simulator headless for a known initial state (e.g., ninja at (500, 300), vx=0, vy=0, jumping on flat ground, no horizontal input, empty map). Log (x, y, vx, vy) at every frame. Compare against the closed-form predictions. They should match to floating-point precision (< 1e-10 error) for the duration before any collision occurs.

**Test cases:**

| Test | Initial State | Force | Expected Match |
|------|--------------|-------|----------------|
| Free fall, no input | vy=0, g=GRAVITY_FALL | c=0.0666... | Exact |
| Jump hold, no input | vy=−2.0, g=GRAVITY_JUMP | c=0.0111... | Exact |
| Horizontal accel from rest | vx=0, h=+1 | c=0.0444... | Exact until speed cap |
| Horizontal at max speed | vx=3.333, h=+1 | c=0.0444... | Exact (cap should maintain) |

**Critical**: If the simulator applies drag as `v *= d` using IEEE 754 double-precision, and we compute `dⁿ` using `math.pow(d, n)` or `d**n`, the results will agree. If the simulator uses a slightly different drag pathway (e.g., separate x/y drag applications with intermediate rounding), there may be divergence. The verification step catches this.

---

## 3. Two-Phase Jump Trajectory

A jump from a grounded state (floor or slope) consists of two phases with different gravity constants.

### Phase 1: Jump Hold (frames 0 to j)

During jump hold, gravity = GRAVITY_JUMP = 0.01111... (reduced gravity, allowing the ninja to gain height).

**Initial conditions (flat ground jump):**

```
vx₀ = current horizontal velocity (from running, etc.)
vy₀ = JUMP_FLAT_GROUND_Y = −2.0  (upward impulse)
```

**At end of jump hold (frame j):**

```
vx(j) = D^j · vx₀ + (h · AIR_ACCEL) · (D^j − 1) / (D − 1)
vy(j) = D^j · vy₀ + GRAVITY_JUMP  · (D^j − 1) / (D − 1)

px(j) = px₀ + vx₀ · S1(j) + (h · AIR_ACCEL) · S2(j)
py(j) = py₀ + vy₀ · S1(j) + GRAVITY_JUMP  · S2(j)
```

Where h ∈ {−1, 0, +1} is the horizontal input during the jump.

### Phase 2: Post-Release Fall (frames j to j+k)

At frame j, gravity switches to GRAVITY_FALL = 0.06666... The initial conditions for phase 2 are the terminal state of phase 1.

**At frame j+k (k frames after release):**

```
vx(j+k) = D^k · vx(j) + (h₂ · AIR_ACCEL) · (D^k − 1) / (D − 1)
vy(j+k) = D^k · vy(j) + GRAVITY_FALL   · (D^k − 1) / (D − 1)

px(j+k) = px(j) + vx(j) · S1(k) + (h₂ · AIR_ACCEL) · S2(k)
py(j+k) = py(j) + vy(j) · S1(k) + GRAVITY_FALL   · S2(k)
```

Where h₂ ∈ {−1, 0, +1} is the horizontal input after jump release (may differ from h during hold).

### Combined Trajectory Function

```python
def jump_trajectory(
    px0: float, py0: float,       # initial position
    vx0: float, vy0_impulse: float,  # initial velocity (vy0 = jump impulse)
    h1: int,                       # horizontal input during jump hold {-1, 0, +1}
    j: int,                        # jump hold duration in frames [0, 45]
    h2: int,                       # horizontal input after release {-1, 0, +1}
    k: int,                        # frames after release
) -> tuple[float, float, float, float]:
    """
    Returns (px, py, vx, vy) at frame j+k of a two-phase jump trajectory.
    No speed cap applied — see Section 4 for capped variant.
    """
    a1 = h1 * AIR_ACCEL
    a2 = h2 * AIR_ACCEL
    
    # Phase 1: jump hold (j frames, GRAVITY_JUMP)
    vx_j = velocity_at(vx0, a1, j)
    vy_j = velocity_at(vy0_impulse, GRAVITY_JUMP, j)
    px_j = position_at(px0, vx0, a1, j)
    py_j = position_at(py0, vy0_impulse, GRAVITY_JUMP, j)
    
    # Phase 2: post-release fall (k frames, GRAVITY_FALL)
    vx_jk = velocity_at(vx_j, a2, k)
    vy_jk = velocity_at(vy_j, GRAVITY_FALL, k)
    px_jk = position_at(px_j, vx_j, a2, k)
    py_jk = position_at(py_j, vy_j, GRAVITY_FALL, k)
    
    return px_jk, py_jk, vx_jk, vy_jk
```

### Finding the Apex (Maximum Height)

The apex occurs when vy = 0 during phase 1 (while GRAVITY_JUMP is active). Solve:

```
vy(n) = Dⁿ · vy₀ + GRAVITY_JUMP · (Dⁿ − 1)/(D − 1) = 0
```

Let G = GRAVITY_JUMP / (D − 1). Then:

```
Dⁿ · vy₀ + G · (Dⁿ − 1) = 0
Dⁿ · (vy₀ + G) = G
Dⁿ = G / (vy₀ + G)
n_apex = ln(G / (vy₀ + G)) / ln(D)
```

For vy₀ = −2.0: G = 0.01111... / (−0.006678...) ≈ −1.6639. Then:

```
n_apex = ln(−1.6639 / (−2.0 + (−1.6639))) / ln(0.9933...)
       = ln(−1.6639 / −3.6639) / ln(0.9933...)
       = ln(0.4541) / (−0.006700...)
       ≈ (−0.7882) / (−0.006700)
       ≈ 117.6 frames
```

This means with a full 45-frame jump hold, the ninja is still ascending at frame 45 (hasn't reached apex yet). The apex occurs during phase 2 at:

```
n_apex_phase2 = solve vy(j) · D^k + GRAVITY_FALL · (D^k − 1)/(D − 1) = 0
```

Same formula structure with phase 2 initial conditions. This tells us the total arc shape.

### Slope Jump Variants

For slope jumps, replace the initial impulse `vy₀ = −2.0` with the slope-specific impulse from `sim_mechanics_doc.md`:

| Slope Condition | vx impulse | vy impulse |
|----------------|------------|------------|
| Flat ground | 0 (vx preserved) | −2.0 |
| Downhill with input | 2/3 × slope_x | 2 × slope_y |
| Uphill perpendicular | 2/3 × slope_x | 2 × slope_y (+ speed reset) |
| Default uphill | 0 (vx preserved) | −1.4 |
| Wall regular | ±1.0 × wall_normal | −1.4 |
| Wall slide | ±0.667 × wall_normal | −1.0 |

Each variant uses the same two-phase formula structure with different (vx₀, vy₀). The affine decomposition (Section 5) applies identically — only the impulse constants change.

### Solving the Landing Frame

The ninja "lands" when py(j+k) ≥ py₀ (position returns to or below the launch height, on an empty flat map). This requires solving:

```
vy₀ · S1(j) + GRAVITY_JUMP · S2(j) + vy(j) · S1(k) + GRAVITY_FALL · S2(k) ≥ 0
```

This is transcendental in k (contains D^k terms). Solve via **Newton's method**:

```python
def solve_landing_frame(vy0, j, tol=0.01) -> int | None:
    """
    Find the frame k (in phase 2) where py returns to launch height.
    Returns integer frame count, or None if no landing within 500 frames.
    """
    # Phase 1 terminal state
    vy_j = velocity_at(vy0, GRAVITY_JUMP, j)
    py_j = position_at(0.0, vy0, GRAVITY_JUMP, j)  # py₀ = 0
    
    # Phase 2: find k where py_j + vy_j·S1(k) + GRAVITY_FALL·S2(k) ≥ 0
    # f(k) = py_j + vy_j·S1(k) + GRAVITY_FALL·S2(k)
    # f'(k) = vy(j+k) = velocity_at(vy_j, GRAVITY_FALL, k)
    
    # Initial guess: simple ballistic estimate (ignoring drag)
    discriminant = vy_j**2 - 2 * GRAVITY_FALL * py_j
    if discriminant < 0:
        return None
    k_guess = (-vy_j + math.sqrt(discriminant)) / GRAVITY_FALL
    k = max(1.0, k_guess)
    
    for _ in range(20):  # Newton iterations
        f = py_j + vy_j * S1(int(k)) + GRAVITY_FALL * S2(int(k))
        f_prime = velocity_at(vy_j, GRAVITY_FALL, int(k))
        if abs(f_prime) < 1e-12:
            break
        k_new = k - f / f_prime
        if abs(k_new - k) < tol:
            k = k_new
            break
        k = max(1.0, k_new)
    
    k_int = int(math.ceil(k))
    return k_int if k_int < 500 else None  # 500 frame safety bound
```

**Key observation for Section 5:** The landing frame k depends only on vy₀ and j — it is independent of vx₀ and horizontal input. This means the y-axis landing time is determined entirely by the vertical trajectory, and the x-axis landing position is a function of that fixed k. This is what makes the affine decomposition clean.

---

## 4. Horizontal Axis with Speed Cap

### The Cap Problem

MAX_HOR_SPEED = 3.333 px/frame. The uncapped horizontal terminal velocity is AIR_ACCEL / (1 − D) ≈ 6.65, so the cap binds during any sustained horizontal input.

### Finding the Cap Frame

The frame at which |vx| first reaches MAX_HOR_SPEED:

```
|vx(n_cap)| = MAX_HOR_SPEED
|Dⁿ · vx₀ + a · (Dⁿ − 1)/(D − 1)| = MAX_HOR_SPEED
```

Let A = a / (D − 1). Solving:

```
Dⁿ · (vx₀ + A) − A = ±MAX_HOR_SPEED
Dⁿ = (±MAX_HOR_SPEED + A) / (vx₀ + A)
n_cap = ln((±MAX_HOR_SPEED + A) / (vx₀ + A)) / ln(D)
```

Use the sign matching the direction of acceleration. If the result is negative or complex, the cap is never reached (e.g., decelerating from below cap). If n_cap < 0, the initial velocity already exceeds the cap.

### Capped Regime Behavior

Once the cap is reached, the velocity is clamped each frame:

```
v_capped = D · MAX_HOR_SPEED + a
```

In practice, `D · 3.333 + 0.04444 ≈ 3.356` which gets clamped back to 3.333. The position gain per frame during capped regime is simply MAX_HOR_SPEED.

**Regime summary for horizontal axis:**

```
Regime 1 (uncapped, frames 0 to n_cap−1):
  vx(n) = Dⁿ · vx₀ + a · (Dⁿ − 1)/(D − 1)
  px(n) = px₀ + vx₀ · S1(n) + a · S2(n)

Regime 2 (capped, frames n_cap onward):
  vx(n) = sign(a) · MAX_HOR_SPEED  (constant)
  px(n) = px(n_cap) + sign(a) · MAX_HOR_SPEED · (n − n_cap)
```

### Per-Trajectory Cap Threshold

For the parametric approach (Section 5), the speed cap introduces a piecewise nonlinearity. We handle this by precomputing a **cap threshold** for each trajectory entry — the initial velocity vx₀ at which the cap first binds during that trajectory's arc:

```python
def compute_cap_threshold(h1: int, j: int, h2: int, k_land: int) -> float | None:
    """
    For a trajectory (h1, j, h2, k_land), find the vx₀ at which
    |vx| first reaches MAX_HOR_SPEED at any frame during the arc.
    Returns None if the cap never binds for any vx₀ in [-MAX, +MAX].
    """
    a1 = h1 * AIR_ACCEL
    # At frame n during phase 1: vx(n) = D^n · vx₀ + a1 · (D^n - 1)/(D-1)
    # Cap binds when: |D^n · vx₀ + a1 · (D^n - 1)/(D-1)| = MAX_HOR_SPEED
    # Worst case is at the end of phase 1 (frame j) or during phase 2
    # Solve for vx₀:
    #   D^j · vx₀ + a1 · (D^j - 1)/(D-1) = ±MAX_HOR_SPEED
    #   vx₀ = (±MAX_HOR_SPEED - a1 · (D^j - 1)/(D-1)) / D^j
    
    dj = D**j
    bias = a1 * (dj - 1.0) / D_MINUS_1
    
    # Threshold: the vx₀ at which phase 1 endpoint hits the cap
    if abs(a1) < 1e-12:
        # No horizontal accel — cap only binds if |vx₀| > MAX_HOR_SPEED already
        return MAX_HOR_SPEED  # effectively never binds within valid range
    
    # In the acceleration direction
    threshold = (MAX_HOR_SPEED * (1 if a1 > 0 else -1) - bias) / dj
    return threshold
```

At runtime, each trajectory entry carries this threshold. A single comparison per trajectory determines whether to use the uncapped affine formula or the capped variant.

---

## 5. Affine Velocity Decomposition

### The Core Mathematical Property

This is the key insight that eliminates velocity binning entirely.

For a fixed control sequence (j, h₁, h₂) with landing frame k, expand the x-position formula:

**Phase 1 contribution to px:**

```
px(j) = vx₀ · S1(j) + (h₁ · AIR_ACCEL) · S2(j)
         ╰── scales with vx₀    ╰── constant (no vx₀)
```

**Phase 1 terminal velocity (input to phase 2):**

```
vx(j) = D^j · vx₀ + (h₁ · AIR_ACCEL) · (D^j − 1) / (D − 1)
         ╰── scales with vx₀    ╰── constant
```

**Phase 2 contribution to px:**

```
px(j+k) − px(j) = vx(j) · S1(k) + (h₂ · AIR_ACCEL) · S2(k)
```

Substituting vx(j):

```
= [D^j · vx₀ + (h₁·A·(D^j−1))/(D−1)] · S1(k) + (h₂·A) · S2(k)
= D^j · S1(k) · vx₀  +  [(h₁·A·(D^j−1))/(D−1)] · S1(k) + (h₂·A) · S2(k)
   ╰── scales with vx₀     ╰── all constant terms
```

**Combining both phases:**

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   px(j+k) = Cx · vx₀  +  Bx                                        │
│                                                                      │
│   where:                                                             │
│     Cx = S1(j) + D^j · S1(k)                                        │
│     Bx = (h₁ · AIR_ACCEL) · S2(j)                                   │
│        + [(h₁ · AIR_ACCEL · (D^j − 1)) / (D − 1)] · S1(k)          │
│        + (h₂ · AIR_ACCEL) · S2(k)                                   │
│                                                                      │
│   Cx depends only on j, k (frame counts)                             │
│   Bx depends only on j, k, h₁, h₂ (control sequence)               │
│   Neither depends on vx₀.                                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**The same property holds for the y-axis:**

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   py(j+k) = Cy · vy₀  +  By                                         │
│                                                                      │
│   where:                                                             │
│     Cy = S1(j) + D^j · S1(k)        (same form as Cx!)              │
│     By = GRAVITY_JUMP · S2(j)                                        │
│        + [GRAVITY_JUMP · (D^j − 1) / (D − 1)] · S1(k)              │
│        + GRAVITY_FALL · S2(k)                                        │
│                                                                      │
│   Cy depends only on j, k                                            │
│   By depends only on j, k (gravity constants are fixed)              │
│   Neither depends on vy₀.                                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Crucial Observation: Cx = Cy

The x-coefficient and y-coefficient have **identical form**: `S1(j) + D^j · S1(k)`. This is because drag is applied identically to both axes. The coefficient depends only on the total frame structure (j frames of phase 1, k frames of phase 2), not on the forces applied. We can simply call this `C`:

```
C(j, k) = S1(j) + D^j · S1(k)
```

Every trajectory with the same (j, k) shares the same C value. The only difference between x and y is the bias term (Bx vs By), which encodes the different forces (AIR_ACCEL vs gravity).

### What This Means

For any continuous vx₀ (not binned, not discretized), the landing x-position of trajectory (j, h₁, h₂) is:

```
landing_x = ninja_x + C(j, k) · vx₀ + Bx(j, k, h₁, h₂)
```

This is **one multiply and one add**. No hull lookup, no interpolation, no convex hull containment test.

For slope jumps where vy₀ also varies (depends on slope angle):

```
landing_x = ninja_x + C(j, k) · vx₀ + Bx(j, k, h₁, h₂)
landing_y = ninja_y + C(j, k) · vy₀ + By(j, k)
```

Two multiply-adds. Still O(1) per trajectory, exact for any continuous (vx₀, vy₀).

---

## 6. Trajectory Catalog Precomputation

### Catalog Structure

For each control sequence (j, h₁, h₂), precompute and store:

```python
@dataclass
class TrajectoryEntry:
    # Control parameters
    j: int                # jump hold duration [0, 45]
    h1: int               # horizontal input during hold {-1, 0, +1}
    h2: int               # horizontal input after release {-1, 0, +1}
    
    # Solved landing frame (computed at vx₀=0 for flat ground)
    k_land: int           # frames after release until landing
    total_frames: int     # j + k_land (total arc duration)
    
    # Affine coefficients
    C: float              # velocity coefficient: S1(j) + D^j · S1(k_land)
    Bx: float             # x-axis bias (depends on h1, h2)
    By: float             # y-axis bias (depends on gravity constants)
    
    # Speed cap handling
    cap_threshold: float  # |vx₀| above which speed cap binds during this arc
    Bx_capped: float      # x-axis bias when speed cap is active
    capped_linear_frames: int  # frames spent at MAX_HOR_SPEED in capped regime
    
    # Landing velocity (for downstream subgoal evaluation)
    landing_vy: float     # vy at landing (for death check: must be < MAX_SURVIVABLE_IMPACT)
    C_vx_land: float      # landing vx = C_vx_land · vx₀ + Bx_vx_land
    Bx_vx_land: float
```

### Enumeration

Sweep all control sequences:

```python
def build_trajectory_catalog(vy0_impulse: float = JUMP_FLAT_GROUND_Y) -> list[TrajectoryEntry]:
    """
    Build the trajectory catalog for a given jump impulse.
    Returns ~414 entries for flat ground jumps.
    Additional catalogs for slope/wall jump impulses.
    """
    catalog = []
    
    for j in range(0, MAX_JUMP_DURATION + 1):        # 0..45
        for h1 in [-1, 0, 1]:                        # input during hold
            for h2 in [-1, 0, 1]:                     # input after release
                
                # Solve landing frame (y-axis only, independent of vx₀)
                k_land = solve_landing_frame(vy0_impulse, j)
                if k_land is None:
                    continue  # trajectory never lands (shouldn't happen in practice)
                
                # Compute affine coefficients
                dj = D ** j
                dk = D ** k_land
                
                C = S1(j) + dj * S1(k_land)
                
                a1 = h1 * AIR_ACCEL
                a2 = h2 * AIR_ACCEL
                
                # Bx: all x-position terms that don't involve vx₀
                Bx_phase1 = a1 * S2(j)
                vx_j_bias = a1 * (dj - 1.0) / D_MINUS_1  # constant part of vx(j)
                Bx_phase2 = vx_j_bias * S1(k_land) + a2 * S2(k_land)
                Bx = Bx_phase1 + Bx_phase2
                
                # By: all y-position terms that don't involve vy₀
                By_phase1 = GRAVITY_JUMP * S2(j)
                vy_j_bias = GRAVITY_JUMP * (dj - 1.0) / D_MINUS_1
                By_phase2 = vy_j_bias * S1(k_land) + GRAVITY_FALL * S2(k_land)
                By = By_phase1 + By_phase2
                
                # Landing velocity (vertical — for death check)
                vy_j = velocity_at(vy0_impulse, GRAVITY_JUMP, j)
                landing_vy = velocity_at(vy_j, GRAVITY_FALL, k_land)
                
                # Skip trajectories that result in lethal landing
                if landing_vy > MAX_SURVIVABLE_IMPACT:
                    continue
                
                # Speed cap threshold
                cap_threshold = compute_cap_threshold(h1, j, h2, k_land)
                
                # Capped variant coefficients (for when |vx₀| > cap_threshold)
                # In capped regime, horizontal displacement = 
                #   uncapped portion + MAX_HOR_SPEED × capped_frames
                # This is trajectory-specific and precomputed
                Bx_capped, capped_frames = compute_capped_coefficients(
                    h1, j, h2, k_land, cap_threshold
                )
                
                # Landing vx coefficients (affine in vx₀)
                C_vx_land = dj * dk  # D^(j+k) coefficient on vx₀
                Bx_vx_land = vx_j_bias * dk + a2 * (dk - 1.0) / D_MINUS_1
                
                catalog.append(TrajectoryEntry(
                    j=j, h1=h1, h2=h2,
                    k_land=k_land, total_frames=j + k_land,
                    C=C, Bx=Bx, By=By,
                    cap_threshold=cap_threshold,
                    Bx_capped=Bx_capped, capped_linear_frames=capped_frames,
                    landing_vy=landing_vy,
                    C_vx_land=C_vx_land, Bx_vx_land=Bx_vx_land,
                ))
    
    return catalog
```

### Catalog Size

- Jump hold durations: 46 (0–45)
- Horizontal inputs during hold: 3
- Horizontal inputs after release: 3
- Total control sequences: 46 × 3 × 3 = **414**
- Some pruned (lethal landing, no landing): typically ~380 remain

Per entry storage: ~10 floats × 8 bytes = 80 bytes.
**Total catalog size: ~33 KB** (level-independent, computed once ever).

### No-Jump Trajectories (Walk/Fall Off Edge)

In addition to the jump catalog, include trajectories where the ninja does **not jump** and instead walks off a platform edge. These have no vertical impulse (vy₀ = 0 at departure, or vy₀ = small positive from edge departure). They are represented as:

```
j = 0 (no jump hold), vy₀ = 0 (no impulse), h1/h2 = horizontal input
```

These use GRAVITY_FALL for the entire arc (no phase 1). The affine decomposition applies identically:

```
landing_x = C_fall(k) · vx₀ + Bx_fall(k, h)
landing_y = C_fall(k) · 0 + By_fall(k)     ← vy₀ = 0, so only bias term
```

These are included as additional catalog entries with `j = 0` and a separate vy₀ = 0 (rather than the jump impulse).

---

## 7. Scatter-and-Collect Reachability

### The Core Algorithm

Instead of building a convex hull and testing surface points against it, we **scatter** landing points from the trajectory catalog onto the level and **collect** nearby surface points. This inverts the hull-containment approach and is naturally suited to the affine parameterization.

```python
def find_reachable_surfaces(
    ninja_x: float, ninja_y: float,
    vx0: float, vy0_impulse: float,
    catalog: list[TrajectoryEntry],
    surface_hash: SpatialHash,
    match_radius: float = 12.0,   # half a tile — how close a surface must be to a landing
) -> set[SurfacePoint]:
    """
    Find all surface points reachable from the ninja's current state
    via any single trajectory in the catalog.
    
    Runtime: ~414 entries × (1 mul-add + 1 hash lookup) ≈ 2–5μs
    """
    reachable = set()
    
    for traj in catalog:
        # Affine evaluation — O(1) per trajectory
        if abs(vx0) < traj.cap_threshold:
            # Uncapped regime: exact affine formula
            land_x = ninja_x + traj.C * vx0 + traj.Bx
        else:
            # Capped regime: use precomputed capped bias
            land_x = ninja_x + traj.Bx_capped + (
                math.copysign(MAX_HOR_SPEED, vx0) * traj.capped_linear_frames
            )
        
        land_y = ninja_y + traj.C * vy0_impulse + traj.By
        
        # Spatial hash lookup for nearby surfaces — O(1) amortized
        nearby = surface_hash.query_radius(land_x, land_y, match_radius)
        reachable.update(nearby)
    
    return reachable
```

### Why Scatter-and-Collect Beats Hull Containment

| Property | Hull Containment (v1.0) | Scatter-and-Collect (v2.0) |
|----------|------------------------|---------------------------|
| Velocity handling | Discrete bins (quantization error) | Continuous (exact affine) |
| Runtime per env | ~1μs (precomputed hull) or ~1ms (recomputed hull) | ~2–5μs (always) |
| Precomputed storage | ~200 KB (409 hulls) | ~33 KB (414 trajectory entries) |
| Accuracy | ±1 tile at 15 bins | Exact (within match_radius) |
| Hull topology changes | Must handle vertex membership changes with velocity | Not applicable — no hull computed |
| Speed cap handling | Baked into each bin's hull | Per-trajectory comparison (1 branch) |
| Extensibility | Multiplicative: bins × slopes × ... | Additive: separate catalog per jump type |

### Match Radius Selection

The `match_radius` parameter controls how close a surface point must be to a trajectory landing point to be considered reachable. This replaces the hull polygon's containment boundary.

**Recommended: 12px (half a tile).** This is slightly larger than the NINJA_RADIUS (10px), accounting for:
- The ninja's circular collision shape (can contact a surface when center is up to 10px away)
- Small position errors from the empty-map assumption (real trajectories may land slightly differently due to intermediate collisions)
- Surface sampling resolution (~6px from VALUE_PRECOMPUTATION_SPEC §3)

**Can be tuned per-level** if needed. Tighter match_radius (6px) gives more accurate reachability at the cost of missing some barely-reachable surfaces. Wider (24px) captures more surfaces but may include unreachable ones.

### Mid-Arc Sampling

The basic scatter-and-collect tests only landing points. To also detect surfaces the ninja passes through mid-arc (relevant for switch/entity collection during a jump), add intermediate arc samples:

```python
def find_reachable_surfaces_with_arc_samples(
    ninja_x, ninja_y, vx0, vy0_impulse,
    catalog, surface_hash,
    match_radius=12.0,
    sample_interval=10,     # sample every N frames along arc
) -> set[SurfacePoint]:
    """Extended version that samples along arcs, not just endpoints."""
    reachable = set()
    
    for traj in catalog:
        # Sample along the arc at regular intervals
        total = traj.total_frames
        for frame in range(0, total + 1, sample_interval):
            # Determine which phase this frame is in
            if frame <= traj.j:
                # Phase 1 (jump hold)
                px, py, _, _ = jump_trajectory(
                    ninja_x, ninja_y, vx0, vy0_impulse,
                    traj.h1, frame, traj.h2, 0
                )
            else:
                # Phase 2 (falling)
                k = frame - traj.j
                px, py, _, _ = jump_trajectory(
                    ninja_x, ninja_y, vx0, vy0_impulse,
                    traj.h1, traj.j, traj.h2, k
                )
            
            nearby = surface_hash.query_radius(px, py, match_radius)
            reachable.update(nearby)
        
        # Always include the landing point
        land_x = ninja_x + traj.C * vx0 + traj.Bx
        land_y = ninja_y + traj.C * vy0_impulse + traj.By
        nearby = surface_hash.query_radius(land_x, land_y, match_radius)
        reachable.update(nearby)
    
    return reachable
```

**Cost of mid-arc sampling**: Each intermediate sample requires a full `jump_trajectory()` call (~10 FLOPs). At 10-frame intervals on a ~150-frame arc, that's ~15 samples per trajectory × 414 trajectories × 10 FLOPs ≈ 62,000 FLOPs. At ~1 GFLOP/s per core, this is ~62μs — acceptable but noticeably more than the landing-only 2–5μs.

**Recommendation**: Use landing-only scatter-and-collect for normal PBRS evaluation. Use mid-arc sampling only when checking if the goal entity (switch/exit) is reachable by flying through it, which is a special-case check rather than the common path.

---

## 8. State-Specific Trajectory Sets

### Floor States (Flat Ground)

**Jump impulse**: vx₀ = current vx (preserved), vy₀ = −2.0 (fixed).

**Catalog**: The primary catalog from Section 6. Since vy₀ is fixed, the y-axis coefficients (C, By) are constants — only the x-axis varies with vx₀ via the affine formula.

**No-jump case**: Include fall-off-edge entries where the ninja walks off a platform without jumping. These have vy₀ = 0 and use GRAVITY_FALL for the entire arc.

### Floor States (Slopes)

**Jump impulse**: Both vx₀ and vy₀ vary depending on slope normal and movement direction (see slope jump variants table in Section 3).

**Catalog**: Separate catalog entries per slope-jump type. The affine decomposition becomes **bilinear** in (vx₀, vy₀):

```
landing_x = C · vx₀ + Bx
landing_y = C · vy₀ + By
```

Where vx₀ and vy₀ are both determined by the slope normal and the ninja's pre-jump velocity. At runtime:

```python
# Slope jump: compute impulse from slope normal
vx_impulse = compute_slope_vx_impulse(slope_normal, ninja_vx, input_dir)
vy_impulse = compute_slope_vy_impulse(slope_normal, ninja_vy, input_dir)

# Use same catalog, different initial velocities
for traj in slope_catalog:
    land_x = ninja_x + traj.C * vx_impulse + traj.Bx
    land_y = ninja_y + traj.C * vy_impulse + traj.By
    ...
```

The catalog entries are identical in structure — only the impulse values change. No separate precomputation needed per slope angle.

### Wall States

**Jump impulse**: Wall jumps **replace** the current velocity entirely. vx₀ and vy₀ are fixed by the jump type and wall normal:

| Wall Jump Type | vx₀ | vy₀ |
|---------------|-----|-----|
| Regular, right wall (normal = −1) | −1.0 | −1.4 |
| Regular, left wall (normal = +1) | +1.0 | −1.4 |
| Slide, right wall | −0.667 | −1.0 |
| Slide, left wall | +0.667 | −1.0 |

Since both vx₀ and vy₀ are fixed constants (not variable), every trajectory entry reduces to a single precomputed landing point — no affine evaluation needed at runtime, just direct coordinate lookup.

```python
# Wall jump: all landing points are fully precomputed constants
for traj in wall_catalog[wall_normal][jump_type]:
    land_x = ninja_x + traj.precomputed_dx  # C · vx₀ + Bx, but vx₀ is constant
    land_y = ninja_y + traj.precomputed_dy
    ...
```

**Storage**: 4 wall types × ~414 entries × 2 floats (dx, dy) × 8 bytes ≈ **26 KB**.

### Airborne States

The ninja cannot jump while airborne. Available controls:
- Horizontal input h ∈ {−1, 0, +1}
- Jump release timing (if still holding)

**Falling only (jump released/exhausted):** Single-phase trajectory with GRAVITY_FALL. Parameterized by (vx, vy) — both are free variables.

```
Catalog: 3 entries (one per horizontal input h)
Landing position:
  land_x = ninja_x + C_fall(k) · vx + Bx_fall(k, h)
  land_y = ninja_y + C_fall(k) · vy + By_fall(k)
```

Where k is the landing frame, solved from the y-axis equation with the current vy.

**Complication**: The landing frame k depends on vy (how long until the ninja returns to a surface level). Unlike grounded jumps where vy₀ is a fixed impulse, airborne vy is a free variable. This means k varies with vy, and therefore C and B vary with vy.

**Solution**: For airborne states, solve the landing frame at runtime (one Newton solve per vy, ~5 iterations × ~10 FLOPs = ~50 FLOPs), then evaluate C(k) and B(k) analytically. Total cost per airborne state: ~50 + 3 × 10 = ~80 FLOPs. Still sub-microsecond.

**Jump still held:** Two-phase trajectory where the ninja chooses when to release (0 to jump_remaining frames). Sweep release_at in ~10 values, then for each, solve as a fixed-impulse trajectory with the velocity at the release frame as initial conditions. This is ~30 trajectory evaluations (10 release times × 3 h values) with per-evaluation cost ~80 FLOPs. Total: ~2,400 FLOPs, still under 5μs.

---

## 9. Translation Invariance

### Key Property

The N++ physics has **no position-dependent forces**. Gravity, drag, friction, and acceleration are all constants — they don't change based on where the ninja is on the map. The only position-dependent interactions are tile collisions, which are handled separately at runtime.

**Consequence**: The affine coefficients (C, Bx, By) computed at the origin apply everywhere. The trajectory catalog is **level-independent and position-independent**. At runtime, the ninja's position is simply added to each landing point:

```
landing_x = ninja_x + C · vx₀ + Bx      ← ninja_x is the only position-dependent term
landing_y = ninja_y + C · vy₀ + By      ← ninja_y is the only position-dependent term
```

This is the same translation-invariance property from v1.0, but now it applies to individual trajectory landing points rather than hull vertices.

### What Varies, What Doesn't

| Property | Position-Invariant? | Velocity-Parameterized? |
|----------|-------------------|------------------------|
| Trajectory shape (given same velocity) | Yes | Affine in (vx₀, vy₀) |
| Jump impulse on flat ground | Yes | vy₀ = −2.0 (constant) |
| Jump impulse on slopes | Yes | vy₀ = f(slope_normal) — evaluated at runtime |
| Wall jump impulse | Yes | vx₀, vy₀ fixed per wall type |
| Air control effect | Yes | Encoded in Bx per h₁, h₂ |
| Speed cap behavior | Yes | Per-trajectory threshold comparison |
| Landing frame | Position-invariant | vy₀-dependent (solved at runtime for airborne states) |

---

## 10. Runtime Integration with Level Geometry

### Existing Infrastructure

From the project documentation, the codebase already has:

1. **Tile grid**: 42×23 cells, each a known tile type. Collision segments stored per cell.
2. **Surface lookup table**: From `VALUE_PRECOMPUTATION_SPEC.md` §3, surfaces are extracted and sampled at regular intervals. Each surface point has (x, y, normal, type).
3. **A\* pathfinding**: From `OBSERVATION_SPACE_DOCUMENTATION.md`, A\*-derived path features already exist in the observation space.

### Runtime Algorithm (Per Frame, Per Environment)

```
1. READ NINJA STATE
   position (px, py), velocity (vx, vy), contact type, surface normal,
   movement state, jump frames remaining

2. SELECT TRAJECTORY CATALOG
   - Floor (flat): primary catalog, vx₀ = vx, vy₀ = -2.0
   - Floor (slope): primary catalog, vx₀/vy₀ from slope impulse formula
   - Wall: wall catalog (precomputed constants, no affine eval)
   - Airborne (falling): solve landing frame at runtime, evaluate 3 trajectories
   - Airborne (holding jump): sweep release times, evaluate ~30 trajectories

3. SCATTER LANDING POINTS
   For each trajectory entry in selected catalog:
     land_x = ninja_x + C · vx₀ + Bx    (one mul-add, or comparison + branch for cap)
     land_y = ninja_y + C · vy₀ + By    (one mul-add)
     ↓
4. COLLECT NEARBY SURFACES
     nearby = surface_hash.query_radius(land_x, land_y, match_radius)
     Add to reachable set

5. SELECT SUBGOAL
   Among reachable surface points, find the one minimizing A* distance to goal
   If goal entity itself is within match_radius of any landing point: goal is directly reachable

6. COMPUTE Φ
   Φ(s) = −a_star_distance(subgoal, goal) / normalization_constant
   If goal directly reachable: Φ(s) = −frame_cost / normalization

7. PBRS REWARD
   F_t = γ · Φ(s') − Φ(s)
```

### Handling Collisions (Empty-Map vs Real Level)

The trajectory catalog assumes empty-map physics (no tile collisions during the arc). In real levels, walls and floors can block trajectories, making some landing points unreachable in practice.

**Conservative approach (recommended):** Use the unclipped landing points. A surface flagged as "reachable" is actually "reachable on an empty map" — an upper bound. This occasionally selects a subgoal behind a wall, but:

- The A\* distance metric for subgoal ranking already accounts for walls (A\* paths go around obstacles). A surface behind a wall will have a larger A\* distance to the goal than a surface on the correct side, so it will rarely be selected as the best subgoal.
- The agent's policy network handles actual navigation (collision avoidance via spatial context features in the observation space).
- PBRS only needs to be directionally correct, not geometrically exact. A slightly overoptimistic potential doesn't break the PBRS guarantee (Ng et al. 1999) — it just weakens the gradient.

**Line-of-sight filter (v2 refinement):** For each landing point, cast a ray from the ninja's position to the landing position through the tile grid. If the ray intersects any solid tile, discard the trajectory as blocked. This is a grid traversal (DDA algorithm, O(tiles_crossed) ≈ O(20)), adding ~8,000 FLOPs per frame (414 trajectories × ~20 FLOPs per ray). Total: ~8μs additional overhead.

---

## 11. PBRS Subgoal Algorithm

### Complete Algorithm

```python
def compute_phi(
    ninja_state: NinjaState,
    goal_pos: tuple[float, float],
    level_surfaces: SpatialHash,
    catalog: TrajectoryCatalog,
    a_star_distances: dict[int, float],
    match_radius: float = 12.0,
    normalization: float = None,    # max A* distance in level, computed once
) -> float:
    """
    Compute PBRS potential Φ(s) using analytical reachability.
    
    Returns: float in [-1, 0], where 0 = at goal.
    """
    # Terminal states
    if ninja_state.is_dead or ninja_state.is_timeout or ninja_state.is_success:
        return 0.0
    
    # Determine initial velocities for trajectory evaluation
    vx0, vy0, traj_set = select_trajectory_set(ninja_state, catalog)
    
    # Check if goal is directly reachable by any trajectory
    goal_dx, goal_dy = goal_pos[0] - ninja_state.x, goal_pos[1] - ninja_state.y
    for traj in traj_set:
        if abs(vx0) < traj.cap_threshold:
            land_dx = traj.C * vx0 + traj.Bx
        else:
            land_dx = traj.Bx_capped + math.copysign(
                MAX_HOR_SPEED * traj.capped_linear_frames, vx0
            )
        land_dy = traj.C * vy0 + traj.By
        
        if (abs(land_dx - goal_dx) < match_radius and 
            abs(land_dy - goal_dy) < match_radius):
            # Goal directly reachable — return high potential scaled by frame cost
            return -traj.total_frames / normalization
    
    # Scatter landing points and collect reachable surfaces
    reachable = set()
    for traj in traj_set:
        if abs(vx0) < traj.cap_threshold:
            land_x = ninja_state.x + traj.C * vx0 + traj.Bx
        else:
            land_x = ninja_state.x + traj.Bx_capped + math.copysign(
                MAX_HOR_SPEED * traj.capped_linear_frames, vx0
            )
        land_y = ninja_state.y + traj.C * vy0 + traj.By
        
        nearby = level_surfaces.query_radius(land_x, land_y, match_radius)
        reachable.update(nearby)
    
    if not reachable:
        # Fallback: use nearest surface below (airborne with no reachable surface)
        nearest = level_surfaces.nearest_below(ninja_state.x, ninja_state.y)
        if nearest is not None:
            return -a_star_distances.get(nearest.id, normalization) / normalization
        return -1.0  # worst case
    
    # Select best subgoal: reachable surface closest to goal by A* distance
    best_subgoal = min(
        reachable,
        key=lambda s: a_star_distances.get(s.id, float('inf'))
    )
    
    subgoal_dist = a_star_distances.get(best_subgoal.id, normalization)
    return -subgoal_dist / normalization


def select_trajectory_set(
    state: NinjaState, catalog: TrajectoryCatalog
) -> tuple[float, float, list[TrajectoryEntry]]:
    """
    Select the appropriate trajectory set and initial velocities
    based on the ninja's current contact state.
    
    Returns: (vx0, vy0, trajectory_entries)
    """
    if state.contact == FLOOR:
        if is_flat_ground(state.floor_normal):
            return (state.vx, JUMP_FLAT_GROUND_Y, catalog.floor_flat)
        else:
            vx_imp, vy_imp = compute_slope_impulse(
                state.floor_normal, state.vx, state.input_h
            )
            return (vx_imp, vy_imp, catalog.floor_flat)  # same catalog, different impulse
    
    elif state.contact == WALL:
        normal_dir = 1 if state.wall_normal_x > 0 else -1
        jump_type = "slide" if state.movement_state == WALL_SLIDING else "regular"
        # Wall jumps have fixed velocities — use precomputed landing points
        return (None, None, catalog.wall[normal_dir][jump_type])
    
    elif state.contact == AIRBORNE:
        if state.jump_held and state.jump_remaining > 0:
            return (state.vx, state.vy, catalog.airborne_holding)
        else:
            return (state.vx, state.vy, catalog.airborne_falling)
```

### Two-Phase Reward Machine Integration

```
Phase 0 (switch not collected):
  goal = exit_switch_position
  a_star_distances = a_star_switch (precomputed per level)
  Φ(s, u0) = compute_phi(state, switch_pos, ...)
  Φ_phase = 0.0

Phase 1 (switch collected, exit active):
  goal = exit_door_position
  a_star_distances = a_star_exit (precomputed per level)
  Φ(s, u1) = compute_phi(state, exit_pos, ...)
  Φ_phase = C_PHASE (0.5 recommended)

PBRS reward:
  F_t = γ · (Φ(s', u') + Φ_phase') − (Φ(s, u) + Φ_phase)
```

### Monotonicity Guarantee

The algorithm guarantees that if the agent transitions from surface A to a reachable surface B that is closer to the goal (by A\* distance), then Φ(B) > Φ(A). This provides a monotonically improving potential along any goal-approaching trajectory.

The guarantee holds because:
- The scatter-and-collect identifies physically reachable surfaces (via analytically computed landing points)
- A\* distance to goal is a consistent heuristic (never overestimates on the graph)
- The subgoal at B is at least as close to the goal as B itself (B's own reachable set may contain surfaces even closer to the goal)

---

## 12. Data Structures

### Trajectory Catalog

```python
@dataclass
class TrajectoryCatalog:
    # Floor/slope jump catalog (~380 entries after pruning lethal landings)
    floor_flat: list[TrajectoryEntry]
    
    # Wall jump catalogs (precomputed constants, no affine eval needed)
    # Keyed by (wall_normal_direction, jump_type)
    wall: dict[tuple[int, str], list[WallTrajectoryEntry]]
    
    # Airborne catalogs
    airborne_falling: list[AirborneTrajectoryEntry]    # 3 entries (h = -1, 0, +1)
    airborne_holding: list[AirborneTrajectoryEntry]    # ~30 entries (release × h)

@dataclass
class TrajectoryEntry:
    """Single precomputed trajectory with affine coefficients."""
    j: int              # jump hold duration
    h1: int             # horizontal input during hold
    h2: int             # horizontal input after release
    k_land: int         # landing frame in phase 2
    total_frames: int   # j + k_land
    C: float            # affine velocity coefficient
    Bx: float           # x-axis position bias
    By: float           # y-axis position bias
    cap_threshold: float  # |vx₀| above which speed cap binds
    Bx_capped: float    # x-bias in capped regime
    capped_linear_frames: int
    landing_vy: float   # for death check
    C_vx_land: float    # landing vx affine coefficient
    Bx_vx_land: float   # landing vx bias

@dataclass
class WallTrajectoryEntry:
    """Wall jump trajectory — fully precomputed, no velocity parameterization."""
    j: int
    h1: int
    h2: int
    total_frames: int
    dx: float           # precomputed landing offset x
    dy: float           # precomputed landing offset y
    landing_vy: float

@dataclass
class AirborneTrajectoryEntry:
    """Airborne trajectory — runtime landing frame solve required."""
    h: int              # horizontal input
    release_at: int     # frames to continue holding jump (0 if already released)
    # Coefficients for post-release phase
    C_post: float       # S1(k) for post-release — but k is runtime-dependent
    # These are evaluated at runtime since k depends on current vy
```

### Level Surface Data (Already Exists)

```python
@dataclass
class LevelSurfaces:
    surface_points: list[SurfacePoint]  # ~3000 points with (x, y, normal, type)
    spatial_hash: SpatialHash            # grid-based spatial index
    a_star_switch: dict[int, float]      # surface_id → A* distance to switch
    a_star_exit: dict[int, float]        # surface_id → A* distance to exit
```

### Serialization

```
Trajectory catalog: trajectory_catalog.npz
  floor_entries: structured array (380 × TrajectoryEntry fields)
  wall_entries: structured array (4 types × ~380 × 2 floats)
  metadata: {n_entries, max_total_frames, constants used}
  
Total size: ~33 KB (one file, computed once, level-independent)
```

---

## 13. Computational Complexity

### Precomputation (Once, Offline, Level-Independent)

```
Trajectory catalog:
  414 control sequences × 1 landing frame solve (Newton, ~100 FLOPs)
  + 414 × ~20 FLOPs (coefficient computation)
  = ~50,000 FLOPs
  → <1 millisecond

Wall catalogs:
  4 types × 414 sequences × ~20 FLOPs = ~33,000 FLOPs
  → <1 millisecond

TOTAL PRECOMPUTATION: <10 milliseconds (computed once ever)
```

### Per-Level Setup (Once Per Level)

```
Surface extraction: already exists in codebase
A* distance computation: already exists in codebase
Spatial hash construction: O(n_surfaces) ≈ O(3000) → negligible
```

### Runtime (Per Frame, Per Environment)

```
Floor state (typical):
  Trajectory evaluation:  380 entries × 3 FLOPs (mul, add, comparison) = ~1,140 FLOPs
  Spatial hash lookups:   380 × ~5 FLOPs (hash + compare) = ~1,900 FLOPs
  Subgoal selection:      ~50 reachable × 1 comparison = ~50 FLOPs
  Total: ~3,100 FLOPs → ~2–3μs

Wall state:
  Landing lookup:         380 entries × 2 FLOPs (add dx, dy) = ~760 FLOPs
  Spatial hash lookups:   380 × ~5 FLOPs = ~1,900 FLOPs
  Total: ~2,700 FLOPs → ~2μs

Airborne (falling):
  Landing frame solve:    1 Newton solve × ~100 FLOPs = ~100 FLOPs
  Trajectory evaluation:  3 entries × 10 FLOPs = ~30 FLOPs
  Spatial hash lookups:   3 × ~5 FLOPs = ~15 FLOPs
  Total: ~150 FLOPs → <1μs

Airborne (holding jump):
  Landing frame solves:   ~10 release times × ~100 FLOPs = ~1,000 FLOPs
  Trajectory evaluation:  30 entries × 10 FLOPs = ~300 FLOPs
  Spatial hash lookups:   30 × ~5 FLOPs = ~150 FLOPs
  Total: ~1,450 FLOPs → ~1–2μs

AT 128 ENVIRONMENTS:
  Average ~3μs × 128 = ~0.4ms per training step
  Compared to typical step time of ~10–50ms → <4% overhead
```

### Comparison with v1.0 (Binned Hull Catalog)

| Metric | v1.0 (Binned Hulls) | v2.0 (Parametric Scatter) |
|--------|---------------------|--------------------------|
| Precomputation | ~2 seconds | ~10 milliseconds |
| Storage | ~200 KB | ~33 KB |
| Runtime per env | ~1μs (surface) to ~1ms (hull recompute) | ~2–5μs (all states) |
| At 128 envs | ~128ms (if hull recompute) | ~0.5ms |
| Velocity accuracy | ±0.238 px/frame (15 bins) | Exact (continuous) |
| Speed cap handling | Baked per bin | Per-trajectory threshold |

---

## 14. Implementation Roadmap

### Phase 1: Verify Closed-Form Equations (Priority: Critical)

**Goal**: Confirm the closed-form expressions match the simulator exactly.

**Tasks**:
1. Implement `velocity_at()`, `position_at()`, `S1()`, `S2()` helper functions
2. Write a test harness that:
   - Places ninja on flat ground in empty map
   - Executes a jump with known parameters (h=0, j=45)
   - Logs (x, y, vx, vy) at every frame from the simulator
   - Compares against closed-form predictions
3. Test cases:
   - Free fall from rest (no jump)
   - Full jump hold, no horizontal input
   - Full jump hold, h=+1
   - Partial jump hold (j=20), h=−1
   - Wall jump (regular and slide)
   - Starting with vx = MAX_HOR_SPEED (cap behavior)
4. Acceptance criteria: max error < 1e-8 across all frames and all test cases

**File**: `nclone/nclone/physics/analytical_trajectory.py`

**Estimated effort**: 1–2 days

### Phase 2: Affine Coefficient Computation and Catalog Generation

**Goal**: Build the trajectory catalog with affine coefficients and speed cap thresholds.

**Tasks**:
1. Implement `solve_landing_frame()` using Newton's method
2. Implement affine coefficient computation (C, Bx, By derivation from Section 5)
3. Implement speed cap threshold computation per trajectory
4. Build full catalog: floor, wall (precomputed constants), airborne templates
5. Serialize to `.npz`
6. **Verification**: For each catalog entry, compare `C · vx₀ + Bx` against `jump_trajectory()` output for several vx₀ values. Must match to float precision.

**File**: `nclone/nclone/physics/trajectory_catalog.py`

**Estimated effort**: 2–3 days

### Phase 3: Scatter-and-Collect Runtime Engine

**Goal**: Implement the runtime reachability evaluation using scatter-and-collect.

**Tasks**:
1. Implement `find_reachable_surfaces()` — the core scatter-and-collect loop
2. Implement `select_trajectory_set()` — state classification
3. Implement airborne landing frame solver (runtime Newton solve)
4. Integrate with existing spatial hash for surface point lookup
5. Benchmark per-environment cost (target: < 5μs average, < 10μs P99)
6. Test at 128 parallel environments (target: < 1ms per training step)

**File**: `nclone/nclone/physics/reachability_scatter.py`

**Estimated effort**: 2–3 days

### Phase 4: PBRS Integration

**Goal**: Wire the reachability system into the training reward pipeline.

**Tasks**:
1. Implement `compute_phi()` — full PBRS potential with two-phase reward machine
2. Implement normalization (per-level max A\* distance)
3. Integrate with FiGAR-PPO reward computation pipeline
4. Add C_PHASE transition handling at switch collection
5. Verify PBRS constraint: Φ(terminal) = 0 for all terminal states

**File**: `nclone/nclone/physics/reachability_pbrs.py`
**Integration**: `nclone/nclone/gym_environment/reward.py`

**Estimated effort**: 1–2 days

### Phase 5: Testing and Validation

**Goal**: Verify the full pipeline produces correct, smooth PBRS potentials.

**Tasks**:
1. **Affine accuracy test**: For 100 random (vx₀, state) pairs, compare scatter-and-collect reachable set against brute-force simulation (run actual trajectories). Sets should match within match_radius tolerance.
2. **Monotonicity test**: Along known optimal paths, verify Φ increases monotonically toward goal.
3. **Smoothness test**: Plot Φ across the level as a 2D heatmap; verify no discontinuities except at phase transition.
4. **Gradient direction test**: At each surface, verify at least one action produces positive F_t.
5. **Training comparison**: Run training episodes, compare learning curves against baseline.
6. **Visualization**: Scatter plot of landing points from a ninja position overlaid on actual level geometry.

**Estimated effort**: 2–3 days

### Total Estimated Effort: 8–13 days

---

## 15. Physics Constants Reference

All values from `sim_mechanics_doc.md` and `VALUE_PRECOMPUTATION_SPEC.md`, used in the closed-form equations.

```python
# === Drag ===
DRAG_REGULAR            = 0.9933221725495059   # Applied to both axes every frame
D = DRAG_REGULAR
D_MINUS_1 = D - 1.0                            # ≈ -0.006678 (used in all formulas)
LN_D = math.log(D)                             # ≈ -0.006700 (used in frame solvers)

# === Gravity ===
GRAVITY_JUMP            = 0.01111111111111111  # During active jump hold (reduced)
GRAVITY_FALL            = 0.06666666666666665  # When falling or jump released

# === Acceleration ===
AIR_ACCEL               = 0.04444444444444444  # Horizontal air control per frame
GROUND_ACCEL            = 0.06666666666666665  # Horizontal ground control per frame

# === Speed Limits ===
MAX_HOR_SPEED           = 3.333333333333333    # Horizontal velocity cap
MAX_JUMP_DURATION       = 45                    # Maximum jump hold frames

# === Jump Impulses ===
JUMP_FLAT_GROUND_Y      = -2.0                 # Flat ground jump vy
JUMP_WALL_REGULAR_X     = 1.0                  # Regular wall jump vx (× wall_normal)
JUMP_WALL_REGULAR_Y     = -1.4                 # Regular wall jump vy
JUMP_WALL_SLIDE_X       = 0.6667               # Wall-slide jump vx (× wall_normal)
JUMP_WALL_SLIDE_Y       = -1.0                 # Wall-slide jump vy
JUMP_SLOPE_DOWNHILL_X   = 0.6667               # Slope jump (× slope_x)
JUMP_SLOPE_DOWNHILL_Y   = 2.0                  # Slope jump (× slope_y)
JUMP_SLOPE_UPHILL_FORWARD_Y = -1.4             # Default uphill jump vy

# === Death ===
MAX_SURVIVABLE_IMPACT   = 6.0                  # Impact velocity death threshold

# === Level ===
NINJA_RADIUS            = 10                    # px
TILE_PIXEL_SIZE         = 24                    # px
MAP_TILE_WIDTH          = 42                    # tiles
MAP_TILE_HEIGHT         = 23                    # tiles
FULL_MAP_WIDTH_PX       = 1056                  # px
FULL_MAP_HEIGHT_PX      = 600                   # px

# === Derived Constants (precompute once) ===
# Terminal velocity (vertical, falling)
V_TERMINAL_Y = GRAVITY_FALL / (1.0 - D)        # ≈ 9.98 px/frame (always lethal)
# Terminal velocity (horizontal, with air accel)
V_TERMINAL_X = AIR_ACCEL / (1.0 - D)           # ≈ 6.65 px/frame (above cap)
# Force-over-drag constants (used in closed-form)
G_JUMP_OVER_DM1 = GRAVITY_JUMP / D_MINUS_1     # ≈ -1.664
G_FALL_OVER_DM1 = GRAVITY_FALL / D_MINUS_1     # ≈ -9.982
A_OVER_DM1 = AIR_ACCEL / D_MINUS_1             # ≈ -6.655
```

---

## Appendix A: Full Derivation Verification

### Step-by-Step Manual Verification

For a flat ground jump with vx₀=0, vy₀=−2.0, h=0 (no horizontal input), j=45 (full hold):

**Frame 1 (manual):**
```
vy = (-2.0) × 0.9933... + 0.01111... = -1.97555... (simulator should match)
py = 0 + (-1.97555...) = -1.97555...
```

**Frame 1 (formula):**
```
vy(1) = D¹ × (-2.0) + 0.01111... × (D¹ - 1)/(D - 1)
      = 0.9933... × (-2.0) + 0.01111... × (-0.006678...) / (-0.006678...)
      = -1.98664... + 0.01111...
      = -1.97553...  ✓ (matches to float precision)
```

**Frame 45 (formula):**
```
D⁴⁵ = 0.9933...⁴⁵ = 0.7399... (precomputable)
vy(45) = 0.7399... × (-2.0) + 0.01111... × (0.7399... - 1.0) / (-0.006678...)
       = -1.4798... + 0.01111... × (-0.2601...) / (-0.006678...)
       = -1.4798... + 0.4327...
       = -1.0471...

py(45) = (-2.0) × S1(45) + 0.01111... × S2(45)
```

The implementation must produce these exact values. Any deviation indicates a bug in the integration order assumption.

### Integration Order Sensitivity

**CRITICAL**: The closed-form derivation assumes drag is applied **before** the force term. If the simulator applies them in the opposite order (force then drag), the recurrence becomes:

```
v(n) = d × (v(n-1) + c) = d×v(n-1) + d×c
```

Which gives: `v(n) = dⁿ·v₀ + d·c·(dⁿ - 1)/(d - 1)`

The extra factor of `d` on the force term changes all predictions. **Verify the integration order in the actual simulator code before proceeding.** The order documented in `sim_mechanics_doc.md` is: drag → gravity → position, which matches our derivation.

### Affine Property Verification

After verifying the base equations, explicitly verify the affine decomposition:

```python
# For a given trajectory (j=30, h1=1, h2=0):
entry = catalog.get_entry(j=30, h1=1, h2=0)

# Test at multiple vx₀ values
for vx0 in [-3.0, -1.5, 0.0, 1.5, 3.0]:
    # Affine prediction
    predicted_x = entry.C * vx0 + entry.Bx
    
    # Full trajectory computation
    actual_x, _, _, _ = jump_trajectory(0, 0, vx0, -2.0, 1, 30, 0, entry.k_land)
    
    assert abs(predicted_x - actual_x) < 1e-10, f"Affine mismatch at vx₀={vx0}"
```

This must pass for every catalog entry. Any failure indicates an error in the coefficient derivation.

---

## Appendix B: Comparison with Simulation-Based Precomputation

| Property | Analytical Parametric (This Spec v2.0) | Simulation (VALUE_PRECOMPUTATION_SPEC) |
|----------|---------------------------------------|---------------------------------------|
| Precomputation time | ~10ms (level-independent) | ~1 minute per level |
| Storage per level | ~33 KB (shared) + level surfaces | ~5 MB per level |
| Runtime Φ evaluation | ~2–5μs all states | ~1μs surface, ~10ms airborne |
| At 128 envs per step | ~0.5ms | ~1.3s (airborne-heavy) |
| Accuracy | Exact velocity, approximate collision | Exact including collisions |
| Collision handling | Conservative (overestimates reachability) | Exact (physics sim includes tiles) |
| Velocity sensitivity | Continuous exact (affine) | Discrete (velocity bins on graph) |
| Entity interaction | Not modeled | Partially modeled (launch pads, doors) |
| Implementation complexity | Moderate (closed-form + spatial hash) | High (full physics sim + Dijkstra + arc cache) |

The analytical parametric approach is a **lightweight, high-performance** PBRS engine suitable for 128+ parallel environments. It trades collision-awareness for speed and continuous velocity accuracy. For levels where tile occlusion is the primary navigation challenge (narrow corridors, mandatory detours), the full simulation-based graph from VALUE_PRECOMPUTATION_SPEC may still be needed as an augmentation.

**Recommended integration**: Use the analytical parametric PBRS as the primary reward shaping signal. Monitor training progress per level. For levels where the agent stalls (indicating the conservative reachability overestimate is misleading), augment with simulation-based collision-aware distances.
