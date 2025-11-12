"""
Spatial segment index for fast collision detection queries.

This module provides a two-level spatial index structure that enables
fast segment queries by filtering based on bounding boxes before
performing expensive collision calculations.
"""

import math
from typing import Dict, List, Tuple


class SpatialSegmentIndex:
    """Spatial index for fast segment queries.

    Two-level structure:
    1. Coarse grid (24x24 pixel cells) for fast spatial lookup
    2. Per-cell metadata (bounds, segment count) for early rejection

    Optimized for static tile data - built once per level load.
    """

    def __init__(self):
        """Initialize empty spatial index."""
        self.cell_data: Dict[Tuple[int, int], Dict] = {}
        self.level_hash: str = None
        self._bounds_cache: Dict[Tuple, Tuple] = {}  # Cache computed query bounds

    def build(
        self,
        segment_dic: Dict[Tuple[int, int], List],
        level_hash: str = None,
        use_cache: bool = True,
    ):
        """Build spatial index from segment dictionary.

        Args:
            segment_dic: Dictionary mapping (xcell, ycell) -> list of segments
            level_hash: Optional hash to identify this level (for cache invalidation)
            use_cache: If True, try to load from/save to persistent cache
        """
        self.level_hash = level_hash

        # Try loading from persistent cache if available
        if use_cache and level_hash:
            try:
                from .persistent_collision_cache import PersistentCollisionCache

                # Try to load from cache
                def builder():
                    return self._build_index(segment_dic)

                cached_data = PersistentCollisionCache.get_or_build(
                    level_hash, builder, data_type="spatial_index"
                )

                if cached_data:
                    self.cell_data = cached_data["cell_data"]
                    self._bounds_cache = cached_data.get("bounds_cache", {})
                    return

            except Exception:
                # Cache error - fall through to direct build
                pass

        # Direct build (no cache or cache failed)
        self._build_index(segment_dic)

    def _build_index(self, segment_dic: Dict[Tuple[int, int], List]) -> dict:
        """Internal method to build index data structures.

        Args:
            segment_dic: Dictionary mapping (xcell, ycell) -> list of segments

        Returns:
            Dictionary with cell_data and bounds_cache for caching
        """
        self.cell_data.clear()
        self._bounds_cache.clear()

        # Build per-cell metadata
        for (xcell, ycell), segments in segment_dic.items():
            if not segments:
                continue

            # Compute cell bounding box from all segments
            min_x = float("inf")
            min_y = float("inf")
            max_x = float("-inf")
            max_y = float("-inf")

            active_segments = []
            for segment in segments:
                if segment.active:
                    active_segments.append(segment)
                    seg_bounds = segment.get_bounds()
                    min_x = min(min_x, seg_bounds[0])
                    min_y = min(min_y, seg_bounds[1])
                    max_x = max(max_x, seg_bounds[2])
                    max_y = max(max_y, seg_bounds[3])

            if active_segments:
                self.cell_data[(xcell, ycell)] = {
                    "segments": active_segments,
                    "bounds": (min_x, min_y, max_x, max_y),
                    "segment_count": len(active_segments),
                }

        # Return data for caching
        return {"cell_data": self.cell_data, "bounds_cache": self._bounds_cache}

    def query_region(self, x1: float, y1: float, x2: float, y2: float) -> List:
        """Query segments in a rectangular region.

        Optimized version of gather_segments_from_region that uses
        spatial indexing and bounding box filtering.

        Args:
            x1, y1, x2, y2: Query rectangle bounds

        Returns:
            List of active segments intersecting the query region
        """
        # Compute bounds and cell ranges
        min_x = min(x1, x2)
        min_y = min(y1, y2)
        max_x = max(x1, x2)
        max_y = max(y1, y2)

        # Convert bounds to cell index ranges and clamp to map grid (44x25)
        min_cell_x = max(0, min(43, math.floor(min_x / 24)))
        max_cell_x = max(0, min(43, math.floor(max_x / 24)))
        min_cell_y = max(0, min(24, math.floor(min_y / 24)))
        max_cell_y = max(0, min(24, math.floor(max_y / 24)))

        # Collect segments from relevant cells with bounding box filtering
        segments = []
        query_bounds = (min_x, min_y, max_x, max_y)

        for xcell in range(min_cell_x, max_cell_x + 1):
            for ycell in range(min_cell_y, max_cell_y + 1):
                cell_key = (xcell, ycell)

                if cell_key not in self.cell_data:
                    continue

                cell_info = self.cell_data[cell_key]

                # Fast rejection: check if cell bounds intersect query bounds
                cell_bounds = cell_info["bounds"]
                if not self._bounds_intersect(query_bounds, cell_bounds):
                    continue

                # Add all segments from this cell
                # (fine-grained filtering happens in collision detection)
                segments.extend(cell_info["segments"])

        return segments

    def query_point(self, x: float, y: float, radius: float) -> List:
        """Query segments near a point (with radius).

        Convenience method for circular queries.

        Args:
            x, y: Point coordinates
            radius: Search radius

        Returns:
            List of active segments within radius of point
        """
        return self.query_region(x - radius, y - radius, x + radius, y + radius)

    @staticmethod
    def _bounds_intersect(bounds1: Tuple, bounds2: Tuple) -> bool:
        """Check if two bounding boxes intersect.

        Args:
            bounds1, bounds2: Tuples of (min_x, min_y, max_x, max_y)

        Returns:
            True if bounds overlap, False otherwise
        """
        min_x1, min_y1, max_x1, max_y1 = bounds1
        min_x2, min_y2, max_x2, max_y2 = bounds2
        return not (
            max_x1 < min_x2 or min_x1 > max_x2 or max_y1 < min_y2 or min_y1 > max_y2
        )

    def get_stats(self) -> Dict:
        """Get statistics about the spatial index.

        Returns:
            Dictionary with index statistics
        """
        total_segments = sum(cell["segment_count"] for cell in self.cell_data.values())
        non_empty_cells = len(self.cell_data)

        return {
            "total_cells": non_empty_cells,
            "total_segments": total_segments,
            "avg_segments_per_cell": total_segments / non_empty_cells
            if non_empty_cells > 0
            else 0,
            "level_hash": self.level_hash,
        }

    def __len__(self) -> int:
        """Return number of non-empty cells in the index."""
        return len(self.cell_data)
