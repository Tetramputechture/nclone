"""Reflection-equivariant quantization of the ninja's position onto a grid.

Every ego-centric observation (the crops, the raycast tile march, the scandot
column fan, the per-episode visit grid) locates the ninja with
``floor(px / cell)``. That is where reflection equivariance breaks, and the
reason is a small theorem rather than an oversight:

    A cell index ``f`` is equivariant iff ``f(W - x) = N - 1 - f(x)`` for the
    ``N``-cell grid spanning ``W``. Substituting ``h = f + 1/2`` turns this into
    ``h(x) + h(W - x) = N``, i.e. ``h - N/2`` must be ODD. An odd function
    vanishes at the origin, but ``h - N/2`` takes values in ``Z + 1/2`` when
    ``N`` is even -- which never contains 0. So for an even ``N`` (ours are 176,
    88 and 44) **no pure function of position can be equivariant.**

The failure set is exactly ``x/cell in Z`` -- the ninja sitting precisely on a
cell boundary. That is not exotic: spawns are grid-aligned, and so is a ninja
standing still on a tile edge. Each such frame shifts every ninja-centred window
by one cell in the mirrored level.

The escape is to stop insisting on a *pure function of position*. Break the tie
on a quantity that is itself reflection-ODD -- the ninja's horizontal velocity,
falling back to the bearing to the active objective and then to the floor
normal. Then the mirrored level's tie resolves the other way, which is exactly
what equivariance requires:

    x on a boundary, d > 0  ->  cell = x/c
    mirrored: x' = W - x, d' = -d < 0  ->  cell' = x'/c - 1 = N - 1 - x/c   ✓

A residual tie survives when every odd quantity is zero at once (at rest, on
flat ground, with the objective exactly overhead). That set is genuinely
measure-zero rather than "every grid-aligned frame", and the harness reports it
instead of hiding it.
"""

from __future__ import annotations

import math

__all__ = ["reflection_odd_direction", "quantize_cell"]


def reflection_odd_direction(
    vx: float = 0.0,
    goal_dx: float = 0.0,
    floor_normal_x: float = 0.0,
) -> int:
    """A ``+-1`` sign that flips under reflection, for tie-breaking.

    Cascades through the reflection-odd quantities the environment has, in
    decreasing order of how meaningful they are as "which way am I facing":
    travel direction, then the bearing to the active objective, then the slope
    the ninja is standing on. Returns ``+1`` when all three vanish.
    """
    for value in (vx, goal_dx, floor_normal_x):
        if value > 0.0:
            return 1
        if value < 0.0:
            return -1
    return 1


def quantize_cell(px: float, cell_px: float, odd_dir: int = 1) -> int:
    """``floor(px / cell_px)``, with the exact-boundary tie broken on ``odd_dir``.

    Away from a boundary this is plain ``floor`` and ``odd_dir`` is ignored, so
    it is a drop-in replacement. Exactly on a boundary a negative ``odd_dir``
    selects the cell to the LEFT, which is what makes the assignment commute
    with reflection (see the module docstring).

    The equality test is exact rather than epsilon-based on purpose: sim
    positions are exact binary fractions and ``px / cell_px`` is correctly
    rounded, so a true integer quotient compares equal and a near-miss does not.
    """
    q = px / cell_px
    i = math.floor(q)
    if odd_dir < 0 and q == i:
        i -= 1
    return int(i)
