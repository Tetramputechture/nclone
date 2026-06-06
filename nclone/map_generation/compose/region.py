"""Region grid: partition the playable interior into an R x C grid of cells.

The playable interior is the 42x23 canvas minus the 1-tile solid border, i.e.
tiles x in [1, 40], y in [1, 21]. The default 2x3 grid yields ~20x10 cells.
Optional seeded jitter perturbs the internal split lines so layouts vary while
staying deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ...constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT

# Interior playfield (inside the 1-tile border): x in [1, W-2], y in [1, H-2].
INTERIOR = (1, 1, MAP_TILE_WIDTH - 2, MAP_TILE_HEIGHT - 2)  # (x0, y0, x1, y1) inclusive


@dataclass(frozen=True)
class Region:
    """An axis-aligned tile rectangle (a single grid cell)."""

    x: int  # left tile (inclusive)
    y: int  # top tile (inclusive)
    w: int  # width in tiles
    h: int  # height in tiles
    row: int  # grid row index
    col: int  # grid col index

    @property
    def x1(self) -> int:
        """Right tile (inclusive)."""
        return self.x + self.w - 1

    @property
    def y1(self) -> int:
        """Bottom tile (inclusive)."""
        return self.y + self.h - 1

    @property
    def floor_row(self) -> int:
        """Default standing row for a floor-style component (bottom of the cell)."""
        return self.y1

    def contains(self, gx: int, gy: int) -> bool:
        return self.x <= gx <= self.x1 and self.y <= gy <= self.y1

    def clamp(self, gx: int, gy: int) -> Tuple[int, int]:
        return (
            max(self.x, min(gx, self.x1)),
            max(self.y, min(gy, self.y1)),
        )


def _split(length: int, n: int, start: int, jitter_rng) -> List[Tuple[int, int]]:
    """Split ``length`` tiles starting at ``start`` into ``n`` (offset, size) spans.

    Even split, then optional +-1..2 jitter on the internal boundaries (seeded),
    never producing a span smaller than 3 tiles.
    """
    base = length // n
    rem = length - base * n
    sizes = [base + (1 if i < rem else 0) for i in range(n)]
    if jitter_rng is not None and n > 1:
        for i in range(n - 1):
            delta = jitter_rng.randint(-2, 2)
            # keep both neighbours >= 3
            if sizes[i] + delta >= 3 and sizes[i + 1] - delta >= 3:
                sizes[i] += delta
                sizes[i + 1] -= delta
    spans = []
    cur = start
    for s in sizes:
        spans.append((cur, s))
        cur += s
    return spans


def partition(
    rows: int = 2,
    cols: int = 3,
    jitter_rng=None,
    interior: Optional[Tuple[int, int, int, int]] = None,
) -> List[List[Region]]:
    """Partition the interior into a ``rows`` x ``cols`` grid of :class:`Region`.

    Returns ``grid[row][col]``. ``jitter_rng`` (a seeded ``random.Random``) enables
    deterministic size jitter; pass ``None`` for an even split.
    """
    x0, y0, x1, y1 = interior if interior is not None else INTERIOR
    total_w = x1 - x0 + 1
    total_h = y1 - y0 + 1
    col_spans = _split(total_w, cols, x0, jitter_rng)
    row_spans = _split(total_h, rows, y0, jitter_rng)

    grid: List[List[Region]] = []
    for r, (ry, rh) in enumerate(row_spans):
        row_cells: List[Region] = []
        for c, (cx, cw) in enumerate(col_spans):
            row_cells.append(Region(x=cx, y=ry, w=cw, h=rh, row=r, col=c))
        grid.append(row_cells)
    return grid
