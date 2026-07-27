"""v26 W4 — topological region decomposition for maze exploration.

CHAMBER_MAZE T3 sits at 0.71 success over 3027-frame episodes and is the only
bucket in v25 that times out at all. Every intrinsic currently available (RND,
NovelD, E3B, count-based visitation, frontier) is **state** novelty, and state
novelty saturates in a maze: by the time it matters the agent has stood in every
corridor, so every cell is stale and the bonus is uniformly near zero. What the
agent does not know is the *order* — which room to enter next.

Region novelty attacks that directly. Rather than counting 12 px cells, this
partitions the reachable space into rooms and rewards the first entry to each
room per episode. A room stays novel until it is entered even after the corridor
leading to it has been walked a hundred times, so the signal survives exactly
where cell novelty dies.

Decomposition
-------------
Connected components of the raw reachable set are useless here — an N++ level is
almost always one component, which is why "components, already computed" is not
enough on its own. The rooms only appear once the corridors are cut:

1. **Erode** the reachable mask (a cell survives iff it and its 4-neighbours are
   all reachable). Passages one cell wide vanish; chambers do not.
2. **Label** the connected components of the eroded mask — those are the rooms.
3. **Absorb** every remaining reachable cell (the corridors, and the one-cell rim
   the erosion shaved off each room) into the nearest room by multi-source BFS
   over the *un-eroded* mask, so the partition covers all reachable space and
   corridor cells are attributed to the room they lead into.

The result is a per-level constant, computed once and cached with the level.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np

# A cell survives erosion iff it and its 4-neighbours are reachable. One pass
# severs 1-cell-wide passages, which at the 12 px visitation resolution is a
# half-tile corridor. More passes would start dissolving small chambers into
# nothing, which reads as "this level has no rooms" — the failure mode the
# fallback below exists to catch.
_EROSION_PASSES = 1
# Rooms smaller than this after erosion are erosion noise, not chambers, and
# would inflate the region count with cells the agent cannot meaningfully
# "enter". They are dropped and absorbed into a real neighbour instead.
_MIN_ROOM_CELLS = 4


def _erode(mask: np.ndarray, passes: int = _EROSION_PASSES) -> np.ndarray:
    """4-connected binary erosion. Out-of-bounds counts as unreachable."""
    out = mask
    for _ in range(max(0, passes)):
        padded = np.zeros(
            (out.shape[0] + 2, out.shape[1] + 2), dtype=bool
        )
        padded[1:-1, 1:-1] = out
        out = (
            out
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return out


def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """4-connected component labelling. Returns ``(labels, n)``; 0 = background."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    n = 0
    w, h = mask.shape
    for sx in range(w):
        for sy in range(h):
            if not mask[sx, sy] or labels[sx, sy]:
                continue
            n += 1
            labels[sx, sy] = n
            q = deque([(sx, sy)])
            while q:
                x, y = q.popleft()
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if (
                        0 <= nx < w
                        and 0 <= ny < h
                        and mask[nx, ny]
                        and not labels[nx, ny]
                    ):
                        labels[nx, ny] = n
                        q.append((nx, ny))
    return labels, n


def _absorb(labels: np.ndarray, mask: np.ndarray) -> None:
    """Multi-source BFS assigning every reachable unlabelled cell to the nearest
    labelled one, in place. Travel is restricted to ``mask``, so a corridor is
    attributed to the room it actually connects to rather than the one that
    happens to be closest in a straight line."""
    w, h = mask.shape
    q = deque(
        (int(x), int(y)) for x, y in np.argwhere(labels > 0)
    )
    while q:
        x, y = q.popleft()
        lab = labels[x, y]
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (
                0 <= nx < w
                and 0 <= ny < h
                and mask[nx, ny]
                and labels[nx, ny] == 0
            ):
                labels[nx, ny] = lab
                q.append((nx, ny))


def label_regions(
    reachable: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], int]:
    """Partition a reachable mask into rooms.

    Args:
        reachable: ``(W, H)`` bool array on the 12 px visitation grid, or None.

    Returns:
        ``(labels, n_regions)`` where ``labels`` is ``(W, H) int32`` with 0 for
        unreachable cells and ``1..n_regions`` for room ids, or ``(None, 0)``
        when no mask was supplied or nothing is reachable.
    """
    if reachable is None:
        return None, 0
    mask = np.asarray(reachable, dtype=bool)
    if not mask.any():
        return None, 0

    eroded = _erode(mask)
    labels, n = _label_components(eroded)

    if n > 1:
        # Drop erosion noise, then renumber so ids stay dense (1..n).
        counts = np.bincount(labels.reshape(-1), minlength=n + 1)
        keep = [i for i in range(1, n + 1) if counts[i] >= _MIN_ROOM_CELLS]
        if keep and len(keep) < n:
            remap = np.zeros(n + 1, dtype=np.int32)
            for new_id, old_id in enumerate(keep, start=1):
                remap[old_id] = new_id
            labels = remap[labels]
            n = len(keep)
        elif not keep:
            n = 0
            labels[:] = 0

    if n == 0:
        # A level with no chamber wider than one cell (a pure corridor loop, or
        # a mask too thin to survive erosion). Fall back to the raw components
        # so the partition is still defined — it just has fewer, larger rooms.
        labels, n = _label_components(mask)
        if n == 0:
            return None, 0

    _absorb(labels, mask)
    return labels, n
