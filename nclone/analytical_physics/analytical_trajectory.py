"""
Closed-form trajectory equations for N++ physics.

Derives exact position and velocity at any future frame n using the linear
recurrence relation of the drag-force-position integration loop.

CRITICAL INTEGRATION ORDER (verified from nsim.py):
  Per frame: integrate() -> collision -> think()

  integrate():
    v_x *= drag          (drag applied first)
    v_y *= drag
    v_y += gravity       (gravity applied to vertical BEFORE position update)
    p_x += v_x           (position update)
    p_y += v_y

  think():
    v_x += h * AIR_ACCEL (horizontal accel applied AFTER position update)

Consequence:
  - Vertical: v_y(n) = d^n * vy0 + g * (d^n - 1)/(d - 1)
              p_y(n) = py0 + vy0 * S1(n) + g * S2(n)       [standard formula]
  - Horizontal: v_x(n) has the same recurrence formula BUT position uses the
    post-drag/pre-accel velocity. This shifts the position by -a*n vs. the
    spec's formula:
              p_x(n) = px0 + vx0 * S1(n) + a * Sx2(n)      [corrected formula]
              where Sx2(n) = S2(n) - n
"""

import math
from typing import Optional

from nclone.constants import (
    DRAG_REGULAR,
    GRAVITY_JUMP,
    GRAVITY_FALL,
    AIR_ACCEL,
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
    MAX_SURVIVABLE_IMPACT,
    JUMP_FLOOR_Y,
)

# Precomputed constants
D = DRAG_REGULAR  # ~0.9933
D_MINUS_1 = D - 1.0  # ~-0.006678
LN_D = math.log(D)  # ~-0.006700

# Precompute powers of D for common frame counts (0..500)
_MAX_CACHED_FRAME = 500
_D_POWERS = [D**n for n in range(_MAX_CACHED_FRAME + 1)]


def _dn(n: int) -> float:
    """Return D^n, using cache for small n."""
    if 0 <= n <= _MAX_CACHED_FRAME:
        return _D_POWERS[n]
    return D**n


# ---------------------------------------------------------------------------
# Helper sums
# ---------------------------------------------------------------------------


def S1(n: int) -> float:
    """Geometric sum: sum_{k=1}^{n} d^k = d * (d^n - 1) / (d - 1).

    This is the cumulative drag coefficient applied to initial velocity
    across n frames of the integrate() loop.
    """
    if n <= 0:
        return 0.0
    return D * (_dn(n) - 1.0) / D_MINUS_1


def S2(n: int) -> float:
    """Double geometric sum used for vertical position (gravity in integrate).

    S2(n) = (S1(n) - n) / (d - 1)

    This accumulates the contribution of a constant force (gravity) applied
    inside integrate() BEFORE position update.
    """
    if n <= 0:
        return 0.0
    return (S1(n) - n) / D_MINUS_1


def Sx2(n: int) -> float:
    """Corrected double sum for horizontal position (accel in think).

    Sx2(n) = S2(n) - n

    Accounts for horizontal acceleration being applied in think() AFTER
    position update. The position contribution of acceleration is delayed
    by one frame relative to the vertical axis.
    """
    if n <= 0:
        return 0.0
    return S2(n) - n


# ---------------------------------------------------------------------------
# Core equations
# ---------------------------------------------------------------------------


def velocity_at(v0: float, c: float, n: int) -> float:
    """Velocity at frame n given initial velocity v0 and constant force c.

    v(n) = d^n * v0 + c * (d^n - 1) / (d - 1)

    This formula applies identically to both axes since the velocity
    recurrence v(n) = d*v(n-1) + c has the same form regardless of
    whether c is applied in integrate() or think().

    Args:
        v0: Initial velocity.
        c: Constant force per frame (gravity for vertical, h*AIR_ACCEL for horizontal).
        n: Number of frames.
    """
    dn = _dn(n)
    return dn * v0 + c * (dn - 1.0) / D_MINUS_1


def position_y_at(p0: float, v0: float, g: float, n: int) -> float:
    """Vertical position at frame n.

    p_y(n) = p0 + v0 * S1(n) + g * S2(n)

    Gravity is applied inside integrate() before position update, so the
    standard S2 sum applies.

    Args:
        p0: Initial y position.
        v0: Initial y velocity.
        g: Gravity constant (GRAVITY_JUMP or GRAVITY_FALL).
        n: Number of frames.
    """
    return p0 + v0 * S1(n) + g * S2(n)


def position_x_at(p0: float, v0: float, a: float, n: int) -> float:
    """Horizontal position at frame n (corrected for think-after-integrate).

    p_x(n) = p0 + v0 * S1(n) + a * Sx2(n)

    Horizontal acceleration is applied in think() AFTER position update,
    so we use the corrected Sx2 = S2 - n.

    Args:
        p0: Initial x position.
        v0: Initial x velocity.
        a: Horizontal acceleration (h * AIR_ACCEL).
        n: Number of frames.
    """
    return p0 + v0 * S1(n) + a * Sx2(n)


# ---------------------------------------------------------------------------
# Two-phase jump trajectory
# ---------------------------------------------------------------------------


def jump_trajectory(
    px0: float,
    py0: float,
    vx0: float,
    vy0: float,
    h1: int,
    j: int,
    h2: int,
    k: int,
) -> tuple[float, float, float, float]:
    """Compute position and velocity at frame j+k of a two-phase jump.

    Phase 1 (frames 0..j): Jump hold with GRAVITY_JUMP, horizontal input h1.
    Phase 2 (frames j..j+k): Post-release with GRAVITY_FALL, horizontal input h2.

    No speed cap applied -- see find_cap_frame() for cap detection.

    Args:
        px0, py0: Initial position (at time of jump).
        vx0: Initial horizontal velocity (preserved from pre-jump state).
        vy0: Initial vertical velocity (jump impulse, e.g. JUMP_FLOOR_Y = -2.0).
        h1: Horizontal input during jump hold {-1, 0, +1}.
        j: Jump hold duration in frames [0, MAX_JUMP_DURATION].
        h2: Horizontal input after release {-1, 0, +1}.
        k: Frames after release.

    Returns:
        (px, py, vx, vy) at frame j+k.
    """
    a1 = h1 * AIR_ACCEL
    a2 = h2 * AIR_ACCEL

    # Phase 1: jump hold (j frames, GRAVITY_JUMP)
    vx_j = velocity_at(vx0, a1, j)
    vy_j = velocity_at(vy0, GRAVITY_JUMP, j)
    px_j = position_x_at(px0, vx0, a1, j)
    py_j = position_y_at(py0, vy0, GRAVITY_JUMP, j)

    # Phase 2: post-release fall (k frames, GRAVITY_FALL)
    vx_jk = velocity_at(vx_j, a2, k)
    vy_jk = velocity_at(vy_j, GRAVITY_FALL, k)
    px_jk = position_x_at(px_j, vx_j, a2, k)
    py_jk = position_y_at(py_j, vy_j, GRAVITY_FALL, k)

    return px_jk, py_jk, vx_jk, vy_jk


# ---------------------------------------------------------------------------
# Landing frame solver
# ---------------------------------------------------------------------------


def solve_landing_frame(
    vy0: float,
    j: int,
    tol: float = 0.01,
    max_frames: int = 500,
) -> Optional[int]:
    """Find the frame k (in phase 2) where p_y returns to launch height.

    Uses Newton's method to solve:
        py(j) + vy(j)*S1(k) + GRAVITY_FALL*S2(k) = 0

    where py(j) and vy(j) are the vertical state at the end of jump hold.

    Args:
        vy0: Initial vertical velocity (jump impulse).
        j: Jump hold duration in frames.
        tol: Convergence tolerance for Newton iterations.
        max_frames: Safety bound on landing frame.

    Returns:
        Integer frame count k, or None if no landing within max_frames.
    """
    # Phase 1 terminal state (vertical only, position relative to launch = 0)
    vy_j = velocity_at(vy0, GRAVITY_JUMP, j)
    py_j = position_y_at(0.0, vy0, GRAVITY_JUMP, j)

    # If ninja is already below launch height after phase 1, landing is immediate
    if py_j >= 0 and j > 0:
        return 0

    # Initial guess from simple ballistic estimate (ignoring drag)
    # 0 = py_j + vy_j * k + 0.5 * GRAVITY_FALL * k^2
    discriminant = vy_j**2 - 2.0 * GRAVITY_FALL * py_j
    if discriminant < 0:
        return None
    k_guess = (-vy_j + math.sqrt(discriminant)) / GRAVITY_FALL
    k = max(1.0, k_guess)

    # Newton iterations
    for _ in range(30):
        ki = max(1, int(round(k)))
        if ki > max_frames:
            return None

        # f(k) = py_j + vy_j * S1(k) + GRAVITY_FALL * S2(k)
        f_val = py_j + vy_j * S1(ki) + GRAVITY_FALL * S2(ki)

        # f'(k) ≈ vy at frame k (velocity is the derivative of position)
        f_prime = velocity_at(vy_j, GRAVITY_FALL, ki)

        if abs(f_prime) < 1e-12:
            break

        k_new = k - f_val / f_prime
        if abs(k_new - k) < tol:
            k = k_new
            break
        k = max(1.0, k_new)

    k_int = int(math.ceil(max(1.0, k)))

    # Verify that we actually crossed the launch height
    if k_int > max_frames:
        return None

    f_check = py_j + vy_j * S1(k_int) + GRAVITY_FALL * S2(k_int)
    if f_check < -1.0:
        # Haven't landed yet, try incrementing
        for extra in range(1, 50):
            k_try = k_int + extra
            if k_try > max_frames:
                return None
            f_try = py_j + vy_j * S1(k_try) + GRAVITY_FALL * S2(k_try)
            if f_try >= 0:
                return k_try
        return None

    return k_int


# ---------------------------------------------------------------------------
# Speed cap frame solver
# ---------------------------------------------------------------------------


def find_cap_frame(vx0: float, a: float, n_max: int) -> Optional[int]:
    """Find the first frame at which |vx| reaches MAX_HOR_SPEED.

    Solves: |D^n * vx0 + a * (D^n - 1)/(D-1)| = MAX_HOR_SPEED

    Args:
        vx0: Initial horizontal velocity.
        a: Horizontal acceleration (h * AIR_ACCEL).
        n_max: Maximum frame to check.

    Returns:
        Frame number where cap first binds, or None if never reached.
    """
    if abs(a) < 1e-12:
        # No acceleration -- cap only binds if already over
        if abs(vx0) >= MAX_HOR_SPEED:
            return 0
        return None

    A = a / D_MINUS_1  # Terminal velocity (uncapped)

    # Solve D^n * (vx0 + A) - A = sign(a) * MAX_HOR_SPEED
    target = math.copysign(MAX_HOR_SPEED, a)
    numerator = target + A
    denominator = vx0 + A

    if abs(denominator) < 1e-12:
        return None

    ratio = numerator / denominator
    if ratio <= 0:
        return None

    n_cap = math.log(ratio) / LN_D

    if n_cap < 0:
        # Already past cap at frame 0
        return 0
    if n_cap > n_max:
        return None

    return int(math.ceil(n_cap))


# ---------------------------------------------------------------------------
# Single-phase fall trajectory (for airborne states)
# ---------------------------------------------------------------------------


def fall_trajectory(
    px0: float,
    py0: float,
    vx0: float,
    vy0: float,
    h: int,
    k: int,
) -> tuple[float, float, float, float]:
    """Compute position and velocity for a pure falling trajectory.

    Single phase with GRAVITY_FALL. Used for airborne states where
    jump is already released or exhausted.

    Args:
        px0, py0: Initial position.
        vx0, vy0: Initial velocity.
        h: Horizontal input {-1, 0, +1}.
        k: Number of frames.

    Returns:
        (px, py, vx, vy) at frame k.
    """
    a = h * AIR_ACCEL
    vx_k = velocity_at(vx0, a, k)
    vy_k = velocity_at(vy0, GRAVITY_FALL, k)
    px_k = position_x_at(px0, vx0, a, k)
    py_k = position_y_at(py0, vy0, GRAVITY_FALL, k)
    return px_k, py_k, vx_k, vy_k


def solve_fall_landing_frame(
    vy0: float,
    target_dy: float = 0.0,
    max_frames: int = 500,
) -> Optional[int]:
    """Find frame k where vertical displacement reaches target_dy during a fall.

    Solves: vy0 * S1(k) + GRAVITY_FALL * S2(k) >= target_dy

    Used for airborne states where vy0 is a free variable and we need
    to find when the ninja reaches a surface below.

    Args:
        vy0: Current vertical velocity.
        target_dy: Vertical displacement target (positive = downward).
        max_frames: Safety bound.

    Returns:
        Integer frame count, or None if target not reached within max_frames.
    """
    # Simple ballistic estimate for initial guess
    if GRAVITY_FALL < 1e-12:
        return None

    discriminant = vy0**2 + 2.0 * GRAVITY_FALL * target_dy
    if discriminant < 0 and vy0 < 0:
        # Going up, need to reach a point below -- estimate time to apex + fall
        k_apex = max(1, int(-vy0 / GRAVITY_FALL))
        k = k_apex * 2
    else:
        k_guess = (-vy0 + math.sqrt(max(0.0, discriminant))) / GRAVITY_FALL
        k = max(1, int(k_guess))

    # Scan for the crossing frame
    for ki in range(max(1, k - 10), min(k + 100, max_frames + 1)):
        dy = vy0 * S1(ki) + GRAVITY_FALL * S2(ki)
        if dy >= target_dy:
            return ki

    return None
