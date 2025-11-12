"""
Tile segment cache for pre-computed collision segments.

This module provides a cache of segment templates per tile type, eliminating
repeated segment creation and reducing memory allocations. Segments are stored
as templates with relative coordinates and instantiated with position offsets.
"""

from typing import Dict, List
from ..entities import GridSegmentLinear, GridSegmentCircular
from ..tile_definitions import (
    TILE_SEGMENT_DIAG_MAP,
    TILE_SEGMENT_CIRCULAR_MAP,
)
from ..constants.physics_constants import TILE_PIXEL_SIZE


class SegmentTemplate:
    """Template for a segment with relative coordinates.

    Stores segment definition relative to tile origin (0, 0).
    Can be instantiated multiple times with different position offsets.
    """

    def __init__(self, segment_type: str, **params):
        """Initialize segment template.

        Args:
            segment_type: 'linear' or 'circular'
            **params: Segment-specific parameters
        """
        self.segment_type = segment_type
        self.params = params

    def instantiate(self, tile_x: int, tile_y: int):
        """Create actual segment instance at given tile position.

        Args:
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate

        Returns:
            GridSegmentLinear or GridSegmentCircular instance
        """
        xtl = tile_x * TILE_PIXEL_SIZE
        ytl = tile_y * TILE_PIXEL_SIZE

        if self.segment_type == "linear":
            x1_rel, y1_rel = self.params["p1"]
            x2_rel, y2_rel = self.params["p2"]
            return GridSegmentLinear(
                (xtl + x1_rel, ytl + y1_rel),
                (xtl + x2_rel, ytl + y2_rel),
                oriented=self.params.get("oriented", True),
            )
        elif self.segment_type == "circular":
            x_center_rel, y_center_rel = self.params["center"]
            return GridSegmentCircular(
                (xtl + x_center_rel, ytl + y_center_rel),
                self.params["quadrant"],
                self.params["convex"],
            )
        else:
            raise ValueError(f"Unknown segment type: {self.segment_type}")


class TileSegmentCache:
    """Cache of pre-computed segment templates per tile type.

    Provides fast lookup of segment templates for each tile type (0-37).
    Templates are shared across all instances of the same tile type.
    """

    # Class-level cache (shared across all instances)
    _template_cache: Dict[int, List[SegmentTemplate]] = {}
    _initialized = False

    @classmethod
    def initialize(cls):
        """Initialize the template cache with all tile types.

        Called lazily on first access. Pre-computes segment templates
        for tile types 0-33. Tile types 34-37 are glitched/unused and
        treated as empty (no collision).
        """
        if cls._initialized:
            return

        # Build templates for each tile type (0-33 are valid, 34-37 are glitched/empty)
        for tile_id in range(38):
            templates = []

            # Skip glitched tiles (34-37) - treat as empty
            if tile_id >= 34:
                cls._template_cache[tile_id] = templates
                continue

            # Diagonal segments
            if tile_id in TILE_SEGMENT_DIAG_MAP:
                p1, p2 = TILE_SEGMENT_DIAG_MAP[tile_id]
                templates.append(SegmentTemplate("linear", p1=p1, p2=p2, oriented=True))

            # Circular segments
            if tile_id in TILE_SEGMENT_CIRCULAR_MAP:
                center, quadrant, convex = TILE_SEGMENT_CIRCULAR_MAP[tile_id]
                templates.append(
                    SegmentTemplate(
                        "circular", center=center, quadrant=quadrant, convex=convex
                    )
                )

            cls._template_cache[tile_id] = templates

        cls._initialized = True

    @classmethod
    def get_templates(cls, tile_id: int) -> List[SegmentTemplate]:
        """Get segment templates for a tile type.

        Args:
            tile_id: Tile type ID (0-37)

        Returns:
            List of SegmentTemplate objects for this tile type
        """
        if not cls._initialized:
            cls.initialize()

        return cls._template_cache.get(tile_id, [])

    @classmethod
    def instantiate_segments(cls, tile_id: int, tile_x: int, tile_y: int) -> List:
        """Create segment instances for a tile at given position.

        Args:
            tile_id: Tile type ID (0-37)
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate

        Returns:
            List of instantiated segment objects
        """
        templates = cls.get_templates(tile_id)
        return [template.instantiate(tile_x, tile_y) for template in templates]

    @classmethod
    def clear(cls):
        """Clear the template cache (for testing/cleanup)."""
        cls._template_cache.clear()
        cls._initialized = False

    @classmethod
    def get_segment_count(cls, tile_id: int) -> int:
        """Get number of segments for a tile type.

        Args:
            tile_id: Tile type ID (0-37)

        Returns:
            Number of non-orthogonal segments for this tile type.
            Returns 0 for glitched tiles (34-37).
        """
        if not cls._initialized:
            cls.initialize()

        # Glitched tiles have no segments
        if tile_id >= 34:
            return 0

        return len(cls._template_cache.get(tile_id, []))
