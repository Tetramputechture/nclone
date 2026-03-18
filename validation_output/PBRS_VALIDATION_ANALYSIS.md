# PBRS Mine Avoidance Validation — Analysis Summary

**Date:** 2025-03-17  
**Value function:** `value_functions/basics_surface_pbrs.npz`  
**Level:** `test-single-level/basics` (hairpin corridor)

## Level Geometry Context

The "basics" level is a **hairpin / U-shaped corridor**:

- **Lower corridor:** Ninja spawns at the right end, runs **left** toward the wall.
- **Upper corridor:** Long horizontal platform with floor at ~y=86; runs from left-center to the far-right wall cap.
- **Transition zone:** 72px vertical gap on the left where the ninja wall-jumps from lower to upper corridor.
- **Optimal route:** Spawn → left → wall-jump (past Mine 1) → land on upper corridor → right (jump Mine 2) → switch → slight left to exit door.

**Three toggle mines (all type 1, initially deadly):**

| Mine | Position | Role |
|------|----------|------|
| **Mine 1** | (96, 108) | Transition-zone hazard — on the left wall of the middle platform; agent navigates past during wall-jump ascent. |
| **Mine 2** | (480, 96) | **Mid-corridor jump hazard** — on the floor of the upper corridor; agent must jump over it while traversing right toward the switch. This is the mine the agent walks into in the screenshot. |
| **Mine 3** | (1032, 84) | **Braking test** — on the left wall of the rightmost wall cap. Exit switch and door are to its left; agent must brake and reverse before overshooting. **Not jump-over-able** (wall cap). |

---

## Executive Summary

The validation confirms that the precomputed PBRS value function correctly encodes mine avoidance (exclusion zones, inflated costs, boundary pinning), but two issues explain why the agent walks into **Mine 2** as seen in the screenshot:

1. **Gravity well at Mine 2 (x=480):** A secondary local minimum in the d* profile at x=450 (30px from mine) creates an attractive gradient that draws the agent toward the danger zone instead of encouraging a jump.
2. **Net-positive death return:** PBRS terminal gain (~+0.84) plus R_DEATH (-0.5) yields weakly-negative or net-positive mine-death return (+0.33 to +0.40), slowing credit assignment for mine avoidance.

---

## Root Cause Confirmation

### Root Cause 1: Secondary Local Minimum ("Gravity Well") — CONFIRMED

**Mine 2 (480, 96)** shows the classic W-shaped profile:

| x   | d*   | phi   | Description                    |
|-----|------|-------|--------------------------------|
| 396 | 166.7 | -0.42 | First local minimum           |
| 432 | 180.8 | -0.45 | Local maximum (inflation hill) |
| **450** | **169.1** | **-0.42** | **Gravity well (30px from mine)** |
| 465 | 207.3 | -0.52 | Mine boundary cliff            |

The gradient from x=438→450 is **+0.0236 to +0.0100 phi/step** — positive, so the agent is encouraged to move toward the mine. The gradient in the last 60px before the exclusion boundary is **+0.000132 phi/px** (attractive), contradicting the intended repulsive behaviour.

**Mine 3 (1032, 84)** has a different geometry: the gravity well at x=936 is 96px from the mine. The gradient near the exclusion boundary is **repulsive** (-0.006 phi/px), which is correct. However, the d* profile is non-monotonic because the switch zone (x≈960–984) creates a valley (d*=0); this is expected and not harmful.

**Mine 1 (96, 108):** No gravity well; d* is monotonically increasing on approach. The mine sits 22px below the floor and is not reachable from the floor surface.

---

### Root Cause 2: Inflation Beta Too Weak — CONFIRMED

With `mine_inflation_beta=50`:

- At clearance 7px (near exclusion): penalty ≈ 1.02 → arc cost 2.02× base
- The inflation creates an 8% bump (d* 167→181) followed by a 7% valley (181→169)
- Cross-gap arcs from x=450 have only modest cost inflation, so they undercut walking around the mine

**Validation recommendation:** Increase `mine_inflation_beta` to 200+ to sustain a repulsive gradient all the way to the exclusion boundary.

---

### Root Cause 3: PBRS Terminal Convention + Sparse Reward Balance — CONFIRMED

**Mine 2 walk trajectory:**

- Walk total reward: **+0.4024** (net positive)
- With R_DEATH=-0.5, net episode return for dying ≈ +0.40 − 0.5 = **−0.10** (slightly negative)
- PBRS terminal gain dominates: the agent does not experience mine death as strongly negative
- Jump trajectory yields **+0.75** → jump is preferred, but the agent must explore to discover it

**Mine 3 walk trajectory:**

- Walk total reward: **−0.4880** (net negative) — here the gradient is repulsive and the value function correctly penalises walking into the mine

The difference between Mine 2 and Mine 3 reinforces that the gravity well at Mine 2 is the main problem: it makes a walk-into-mine trajectory appear profitable until the agent is very close.

---

## FiGAR Mine Interrupt Analysis

| Mine | Gravity well dist | FiGAR (40px) | Verdict |
|------|-------------------|--------------|---------|
| Mine 2 | 30px | Interrupts at 40px | **PASS** — FiGAR fires before agent reaches gravity well |
| Mine 3 | 96px | Interrupts at 40px | **WARN** — Agent reaches gravity well long before FiGAR; interrupt may be too late |

For Mine 2, the 40px FiGAR threshold is sufficient: the agent is interrupted before entering the danger pocket. For Mine 3, the gravity well is far from the mine (96px); increasing FiGAR to ~100px would interrupt before the agent enters that valley, though the gradient there is already repulsive.

---

## Concrete Training Recommendations

### Priority 1: Recompute value function with higher `mine_inflation_beta`

```bash
python -m nclone.value_function.precompute_cli \
    --map-path test-single-level/basics \
    --output-dir value_functions \
    --mine-inflation-beta 200
```

Re-run the validation script to confirm the gravity well at Mine 2 is eliminated.

### Priority 2: Adjust R_DEATH or PBRS scaling for mine deaths

- **Option A:** Increase R_DEATH to −2.0 or −3.0 so mine deaths are clearly net-negative.
- **Option B:** Use `pbrs_gamma < 1.0` (e.g. 0.995) to reduce terminal PBRS gain.
- **Option C:** Add a mine-specific death penalty (e.g. R_DEATH_MINE = −2.0) when `death_cause == 'mine'`.

Mitigate risk of passive behaviour (e.g. excessive NOOP) by tuning HWM_ALPHA or exploration.

### Priority 3: Investigate Mine 3 cross-gap arcs

Mine 3 has **0 cross-gap arcs**. Possible causes:

- Mine is at the right edge; no surface exists to the right to jump to
- The level geometry makes jumping over it infeasible
- Arc enumeration may have a bug for edge-of-level mines

Run `visualize_mine_arcs` to confirm whether jump trajectories exist. If they do not, the agent must route around the mine; the value function should guide that (e.g. via wall surfaces or an alternate path).

### Priority 4: FiGAR threshold

- Mine 2: 40px is adequate.
- Mine 3: Consider raising `figar_mine_interrupt_threshold` to ~100px if the agent tends to enter the gravity well region before discovering a route around.

---

## Validation Verdict

**Overall: PASS** (with caveats)

- Exclusion zones: **PASS** for all mines
- Cross-gap arcs: **PASS** for Mines 1 and 2; **WARN** for Mine 3 (0 arcs)
- Gravity well: **WARN** at Mine 2 (x=450)
- Reward balance: **WARN** — walk-into-mine can be net-positive at Mine 2

The PBRS value function is structurally correct. The observed failures are driven by:

1. A gravity well at Mine 2 that attracts the agent
2. Insufficient penalty for mine death relative to PBRS terminal gain
3. Mine 3’s geometry (no cross-gap arcs) — may require level-specific handling

Implementing the above recommendations should improve mine avoidance and training stability.
