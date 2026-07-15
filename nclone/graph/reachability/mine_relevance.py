"""Cull mines that cannot affect player-reachable space.

On mine-dense levels (e.g. ``mines-extremely-difficult`` has 13,413 mines) the
overwhelming majority of mines are sealed inside dense mine-fields or in pockets
the player can never enter. Such mines never change reachability and can never
be touched, so they are pure waste in every per-step O(num_mines) operation
(state signature, entity-dict build, hazard map, overlays) and pure noise in the
model's observation.

A mine is *relevant* iff its kill-zone borders the player-reachable free space.
We flood-fill the free region from the ninja spawn on a coarse grid: the player
fits through a 24px centre-to-centre gap but not 23px, so two mines act as a
solid barrier once they are <=23px apart. Modelling each deadly mine as a
``passability_radius`` (=12px, so 2*12 = 24px is the passable threshold)
exclusion disk and flood-filling the complement reproduces that rule. Mines that
border the reachable region are kept; the rest are discounted.

This is reachability-preserving by construction (dropping a mine that borders no
reachable cell cannot reconnect any reachable space — verified: the spawn-
reachable region is byte-identical with the relevant-only subset). Death
correctness is unaffected regardless: the native physics simulator retains every
mine, so this only shrinks the Python-side obs/graph/reward view.
"""

from collections import deque
from typing import Iterable, Set, Tuple

# N++ playfield: 44x25 tiles of 24px (42x23 playable + 1-tile border).
DEFAULT_LEVEL_WIDTH = 44 * 24
DEFAULT_LEVEL_HEIGHT = 25 * 24


def compute_relevant_mine_keys(
    mine_positions: Iterable[Tuple[float, float]],
    deadly_positions: Iterable[Tuple[float, float]],
    spawn_xy: Tuple[float, float],
    *,
    grid: int = 6,
    passability_radius: float = 12.0,
    width: int = DEFAULT_LEVEL_WIDTH,
    height: int = DEFAULT_LEVEL_HEIGHT,
) -> Set[Tuple[int, int]]:
    """Return the set of ``(round(x), round(y))`` keys for mines that border the
    player-reachable free space (and are therefore relevant to reachability /
    death / observation). Mines absent from the returned set can be discounted.

    Args:
        mine_positions: all candidate mine world positions (deadly + safe).
        deadly_positions: positions that currently block the player (the
            blockers used to carve the free space). Safe mines do not block.
        spawn_xy: ninja spawn world position (flood-fill seed).
        grid: rasterisation cell size in px (must be < passability gap to
            resolve the 24px rule; 6px is well within it).
        passability_radius: per-mine exclusion radius; ``2*passability_radius``
            is the minimum passable gap (24px by default).
    """
    deadly = [(float(x), float(y)) for (x, y) in deadly_positions]
    candidates = [(float(x), float(y)) for (x, y) in mine_positions]

    gw = max(1, width // grid)
    gh = max(1, height // grid)
    R = float(passability_radius)
    R2 = R * R

    # Mark cells whose centre lies inside a deadly mine's exclusion disk.
    blocked = bytearray(gw * gh)  # 0 = free, 1 = blocked
    rc = int(R // grid) + 1
    half = grid * 0.5
    for mx, my in deadly:
        cx, cy = int(mx // grid), int(my // grid)
        for ix in range(max(0, cx - rc), min(gw, cx + rc + 1)):
            ccx = ix * grid + half
            ddx = ccx - mx
            for iy in range(max(0, cy - rc), min(gh, cy + rc + 1)):
                ccy = iy * grid + half
                if ddx * ddx + (ccy - my) ** 2 < R2:
                    blocked[ix * gh + iy] = 1

    # Seed the flood from the spawn cell, snapping to the nearest free cell if
    # the spawn itself is (spuriously) inside an exclusion disk.
    sx, sy = int(spawn_xy[0] // grid), int(spawn_xy[1] // grid)
    sx = min(max(sx, 0), gw - 1)
    sy = min(max(sy, 0), gh - 1)
    if blocked[sx * gh + sy]:
        snapped = None
        for rad in range(1, 9):
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < gw and 0 <= ny < gh and not blocked[nx * gh + ny]:
                        snapped = (nx, ny)
                        break
                if snapped:
                    break
            if snapped:
                break
        if snapped is None:
            # Spawn fully enclosed (degenerate) — keep every mine to stay safe.
            return {(round(x), round(y)) for x, y in candidates}
        sx, sy = snapped

    # 4-connected flood fill over free cells from the spawn.
    reachable = bytearray(gw * gh)
    dq = deque([(sx, sy)])
    reachable[sx * gh + sy] = 1
    while dq:
        x, y = dq.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < gw and 0 <= ny < gh:
                idx = nx * gh + ny
                if not blocked[idx] and not reachable[idx]:
                    reachable[idx] = 1
                    dq.append((nx, ny))

    # A mine is relevant iff a reachable free cell lies within (R + grid) of it
    # — i.e. its exclusion disk borders space the player can actually occupy.
    rr = R + grid
    rr2 = rr * rr
    rcc = int(rr // grid) + 1
    relevant: Set[Tuple[int, int]] = set()
    for mx, my in candidates:
        cx, cy = int(mx // grid), int(my // grid)
        hit = False
        for ix in range(max(0, cx - rcc), min(gw, cx + rcc + 1)):
            ccx = ix * grid + half
            ddx = ccx - mx
            for iy in range(max(0, cy - rcc), min(gh, cy + rcc + 1)):
                if reachable[ix * gh + iy]:
                    ccy = iy * grid + half
                    if ddx * ddx + (ccy - my) ** 2 < rr2:
                        hit = True
                        break
            if hit:
                break
        if hit:
            relevant.add((round(mx), round(my)))
    return relevant
