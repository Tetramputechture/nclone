"""
OpenCV-based flood fill approximator using existing tile and entity renderers.

This module leverages the existing tile renderer and entity renderer to create
pixel-perfect collision masks, then uses OpenCV's optimized flood fill and
morphological operations to handle ninja radius and entity interactions.
"""

import time
import numpy as np
import cv2
import pygame
from typing import Set, Tuple, Dict, Optional, List, Any

from .reachability_types import ReachabilityApproximation
from ..common import CELL_SIZE, GRID_WIDTH, GRID_HEIGHT
from ...constants.physics_constants import NINJA_RADIUS
from ...tile_renderer import TileRenderer
from ...entity_renderer import EntityRenderer


class OpenCVFloodFill:
    """
    OpenCV-based flood fill approximator using existing renderers.

    This class leverages the existing tile and entity renderers to create
    accurate collision masks, then uses OpenCV's optimized algorithms for:
    - Morphological operations (ninja radius handling)
    - Flood fill (reachability analysis)
    - Image processing (entity state rendering)
    """

    def __init__(self, debug: bool = False, render_scale: float = 0.25):
        """
        Initialize OpenCV flood fill approximator.

        Args:
            debug: Enable debug output and visualization
            render_scale: Scale factor for rendering (1.0 = full resolution, 0.5 = half, etc.)
        """
        self.debug = debug
        self.render_scale = render_scale
        self.ninja_radius = NINJA_RADIUS

        # Rendering components
        self.tile_renderer = None
        self.entity_renderer = None
        self.mock_screen = None

        # Cache for rendered masks
        self.collision_mask_cache: Dict[str, np.ndarray] = {}

        # OpenCV kernel for morphological operations
        self.ninja_kernel = self._create_ninja_kernel()

    def quick_check(
        self,
        ninja_pos: Tuple[int, int],
        level_data,
        switch_states: Dict[str, bool],
        entities: Optional[List[Any]] = None,
    ) -> ReachabilityApproximation:
        """
        Perform OpenCV-based flood fill analysis.

        Args:
            ninja_pos: Ninja position in pixels (x, y)
            level_data: Level tile data
            switch_states: Current switch states
            entities: List of entities in the level

        Returns:
            Reachability approximation with high accuracy
        """
        start_time = time.perf_counter()

        # Create collision mask using renderers
        t1 = time.perf_counter()
        collision_mask = self._create_collision_mask(
            level_data, switch_states, entities
        )
        t2 = time.perf_counter()

        # Apply ninja radius using morphological operations
        traversable_mask = self._apply_ninja_radius(collision_mask)
        t3 = time.perf_counter()

        # Perform OpenCV flood fill
        reachable_mask = self._opencv_flood_fill(ninja_pos, traversable_mask)
        t4 = time.perf_counter()

        # Convert result to position set
        reachable_positions = self._mask_to_positions(reachable_mask)
        t5 = time.perf_counter()

        if self.debug:
            print(f"DEBUG: Collision mask creation: {(t2 - t1) * 1000:.3f}ms")
            print(f"DEBUG: Ninja radius morphology: {(t3 - t2) * 1000:.3f}ms")
            print(f"DEBUG: OpenCV flood fill: {(t4 - t3) * 1000:.3f}ms")
            print(f"DEBUG: Position conversion: {(t5 - t4) * 1000:.3f}ms")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if self.debug:
            print(f"DEBUG: OpenCV flood fill completed in {elapsed_ms:.3f}ms")
            print(f"DEBUG: Found {len(reachable_positions)} reachable positions")
            print(f"DEBUG: Collision mask shape: {collision_mask.shape}")

            # Save debug images if requested
            self._save_debug_images(
                collision_mask, traversable_mask, reachable_mask, ninja_pos
            )

        return ReachabilityApproximation(
            reachable_positions=reachable_positions,
            confidence=0.95,  # High confidence due to renderer accuracy
            computation_time_ms=elapsed_ms,
            method="opencv_flood_fill",
            tier_used=0,  # Tier 0 = Ultra-accurate
        )

    def _create_collision_mask(
        self,
        level_data,
        switch_states: Dict[str, bool],
        entities: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Create collision mask using existing tile and entity renderers.

        Args:
            level_data: Level tile data
            switch_states: Current switch states
            entities: List of entities

        Returns:
            Binary collision mask (True = solid/blocked, False = traversable)
        """
        # Calculate render dimensions using full pixel resolution
        # Full level is 42x23 tiles at 24 pixels per tile = 1008x552 pixels
        full_pixel_width = GRID_WIDTH * CELL_SIZE  # 42 * 24 = 1008
        full_pixel_height = GRID_HEIGHT * CELL_SIZE  # 23 * 24 = 552
        render_width = int(full_pixel_width * self.render_scale)
        render_height = int(full_pixel_height * self.render_scale)

        # Create mock simulation and screen for rendering
        mock_sim = self._create_mock_sim(level_data, switch_states, entities)
        mock_screen = pygame.Surface((render_width, render_height))

        # Create tile renderer
        tile_renderer = TileRenderer(mock_sim, mock_screen, self.render_scale)

        # Render tiles to get collision mask (use white for solid tiles)
        tile_surface = tile_renderer.draw_tiles(init=True, tile_color="FFFFFF")

        # Convert pygame surface to numpy array
        tile_array = pygame.surfarray.array3d(tile_surface)
        tile_array = np.transpose(tile_array, (1, 0, 2))  # Correct orientation

        # Convert to grayscale and create binary mask
        tile_gray = cv2.cvtColor(tile_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        collision_mask = tile_gray > 128  # White pixels = solid

        # Add entity collisions if entities provided
        if entities:
            entity_mask = self._render_entity_collisions(
                mock_sim, mock_screen, entities, switch_states
            )
            collision_mask = np.logical_or(collision_mask, entity_mask)

        return collision_mask

    def _render_entity_collisions(
        self, mock_sim, mock_screen, entities: List[Any], switch_states: Dict[str, bool]
    ) -> np.ndarray:
        """
        Render entity collisions using entity renderer.

        Args:
            mock_sim: Mock simulation object
            mock_screen: Mock pygame screen
            entities: List of entities
            switch_states: Current switch states

        Returns:
            Binary mask of entity collisions
        """
        # Create entity renderer
        entity_renderer = EntityRenderer(
            mock_sim,
            mock_screen,
            self.render_scale,
            mock_screen.get_width(),
            mock_screen.get_height(),
        )

        # Update mock sim with entities in their current states
        mock_sim.entities = []
        for entity in entities:
            # Only add solid entities based on switch states
            if self._is_entity_solid_for_rendering(entity, switch_states):
                mock_sim.entities.append(entity)

        # Render entities
        entity_surface = entity_renderer.draw_entities(init=True)

        # Convert to collision mask
        entity_array = pygame.surfarray.array3d(entity_surface)
        entity_array = np.transpose(entity_array, (1, 0, 2))
        entity_gray = cv2.cvtColor(entity_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Any non-transparent pixel is a collision
        entity_mask = entity_gray > 0

        return entity_mask

    def _is_entity_solid_for_rendering(
        self, entity: Any, switch_states: Dict[str, bool]
    ) -> bool:
        """
        Determine if entity should be rendered as solid based on current states.

        Args:
            entity: Entity object
            switch_states: Current switch states

        Returns:
            True if entity should block movement
        """
        entity_type = getattr(entity, "type", None)

        # One-way platforms are always rendered (direction handled by renderer)
        if entity_type == 11:  # ONE_WAY
            return True

        # Doors depend on switch states
        if entity_type == 5:  # REGULAR_DOOR
            switch_id = getattr(entity, "switch_id", None)
            if switch_id and switch_id in switch_states:
                # Door is solid when closed (switch inactive)
                return not switch_states[switch_id]
            return True  # Default to solid

        # Locked doors depend on their activation state
        if entity_type == 6:  # LOCKED_DOOR
            # Only consider door parts (not switch parts) for collision
            is_door_part = getattr(entity, "is_door_part", True)
            if is_door_part:
                # Door is solid when closed
                return getattr(entity, "closed", True)
            else:
                # Switch parts don't block movement
                return False

        # Trap doors depend on their activation state
        if entity_type == 8:  # TRAP_DOOR
            # Only consider door parts (not switch parts) for collision
            is_door_part = getattr(entity, "is_door_part", True)
            if is_door_part:
                # Door is solid when closed
                return getattr(entity, "closed", False)  # Trap doors start open
            else:
                # Switch parts don't block movement
                return False

        # Other entities generally don't block movement for reachability
        return False

    def _apply_ninja_radius(self, collision_mask: np.ndarray) -> np.ndarray:
        """
        Apply ninja radius using morphological operations.

        Args:
            collision_mask: Binary collision mask

        Returns:
            Traversable mask accounting for ninja radius
        """
        # Dilate obstacles by ninja radius to account for ninja size
        dilated_obstacles = cv2.dilate(
            collision_mask.astype(np.uint8), self.ninja_kernel, iterations=1
        )

        # Traversable areas are where dilated obstacles are False
        traversable_mask = dilated_obstacles == 0

        return traversable_mask

    def _opencv_flood_fill(
        self, ninja_pos: Tuple[int, int], traversable_mask: np.ndarray
    ) -> np.ndarray:
        """
        Perform OpenCV flood fill from ninja position.

        Args:
            ninja_pos: Ninja position in pixels (x, y)
            traversable_mask: Binary traversable mask

        Returns:
            Binary mask of reachable areas
        """
        # Apply level padding offset (-1 tile = -24 pixels in each direction)
        # The ninja position includes the 1-tile border, but our rendered mask doesn't
        TILE_SIZE = 24  # TILE_PIXEL_SIZE
        adjusted_x = ninja_pos[0] - TILE_SIZE
        adjusted_y = ninja_pos[1] - TILE_SIZE

        # Convert ninja position to mask coordinates
        mask_x = int(adjusted_x * self.render_scale)
        mask_y = int(adjusted_y * self.render_scale)

        # Ensure position is within bounds
        height, width = traversable_mask.shape
        mask_x = max(0, min(mask_x, width - 1))
        mask_y = max(0, min(mask_y, height - 1))

        # Check if starting position is traversable
        if not traversable_mask[mask_y, mask_x]:
            if self.debug:
                print(f"DEBUG: Starting position ({mask_x}, {mask_y}) not traversable")
            # Find nearest traversable position
            mask_x, mask_y = self._find_nearest_traversable(
                mask_x, mask_y, traversable_mask
            )

        # Create flood fill mask (must be 2 pixels larger than input)
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

        # Convert traversable mask to uint8 for OpenCV
        traversable_uint8 = traversable_mask.astype(np.uint8) * 255

        # Perform flood fill
        cv2.floodFill(
            traversable_uint8,
            flood_mask,
            (mask_x, mask_y),
            255,  # Fill value
            loDiff=0,
            upDiff=0,
            flags=cv2.FLOODFILL_FIXED_RANGE,
        )

        # Extract reachable mask
        reachable_mask = traversable_uint8 == 255

        return reachable_mask

    def _find_nearest_traversable(
        self, start_x: int, start_y: int, traversable_mask: np.ndarray
    ) -> Tuple[int, int]:
        """
        Find nearest traversable position using BFS.

        Args:
            start_x: Starting x coordinate
            start_y: Starting y coordinate
            traversable_mask: Binary traversable mask

        Returns:
            Coordinates of nearest traversable position
        """
        height, width = traversable_mask.shape
        visited = np.zeros_like(traversable_mask, dtype=bool)
        queue = [(start_x, start_y)]

        while queue:
            x, y = queue.pop(0)

            if visited[y, x]:
                continue
            visited[y, x] = True

            if traversable_mask[y, x]:
                return x, y

            # Check 8-connected neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                        queue.append((nx, ny))

        # If no traversable position found, return original
        return start_x, start_y

    def _mask_to_positions(self, reachable_mask: np.ndarray) -> Set[Tuple[int, int]]:
        """
        Convert reachable mask to set of tile center positions.

        Args:
            reachable_mask: Binary mask of reachable areas

        Returns:
            Set of reachable tile center positions
        """
        # Find all True positions in the mask
        y_coords, x_coords = np.where(reachable_mask)

        if len(y_coords) == 0:
            return set()

        # Vectorized operations for speed
        # Scale back to original coordinates
        orig_x = (x_coords / self.render_scale).astype(int)
        orig_y = (y_coords / self.render_scale).astype(int)

        # Convert to tile coordinates
        tile_x = orig_x // CELL_SIZE
        tile_y = orig_y // CELL_SIZE

        # Calculate tile center positions
        center_x = tile_x * CELL_SIZE + CELL_SIZE // 2
        center_y = tile_y * CELL_SIZE + CELL_SIZE // 2

        # Filter within level bounds
        level_width_px = GRID_WIDTH * CELL_SIZE  # 42 * 24 = 1008
        level_height_px = GRID_HEIGHT * CELL_SIZE  # 23 * 24 = 552

        valid_mask = (
            (center_x >= 0)
            & (center_x < level_width_px)
            & (center_y >= 0)
            & (center_y < level_height_px)
        )

        valid_center_x = center_x[valid_mask]
        valid_center_y = center_y[valid_mask]

        # Convert to set of tuples (using unique to avoid duplicates)
        positions = set(zip(valid_center_x.tolist(), valid_center_y.tolist()))

        if self.debug:
            print(
                f"DEBUG: Generated {len(positions)} tile center positions from {len(y_coords)} pixels"
            )

        return positions

    def _create_ninja_kernel(self) -> np.ndarray:
        """
        Create morphological kernel for ninja radius.

        Returns:
            Circular kernel for ninja radius
        """
        # Create circular kernel with ninja radius
        kernel_size = int(self.ninja_radius * 2 * self.render_scale) + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)

        center = kernel_size // 2
        radius = self.ninja_radius * self.render_scale

        for y in range(kernel_size):
            for x in range(kernel_size):
                distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
                if distance <= radius:
                    kernel[y, x] = 1

        return kernel

    def _create_mock_sim(self, level_data, switch_states: Dict[str, bool], entities):
        """
        Create mock simulation object for rendering.

        Args:
            level_data: Level tile data
            switch_states: Current switch states
            entities: List of entities

        Returns:
            Mock simulation object
        """

        class MockNinja:
            def __init__(self, x=0, y=0):
                self.xpos = x
                self.ypos = y
                self.bones = []  # Empty bones for animation

        class MockSimConfig:
            def __init__(self):
                self.enable_anim = False  # Disable animation for rendering

        class MockSim:
            def __init__(self):
                self.tile_dic = {}
                self.entities = entities or []
                self.switch_states = switch_states
                self.ninja = MockNinja()  # Add mock ninja
                self.sim_config = MockSimConfig()  # Add mock sim config

        mock_sim = MockSim()
        mock_sim.entity_dic = {}  # Add entity_dic for compatibility

        # Convert level data to tile dictionary
        if hasattr(level_data, "tiles"):
            tiles = level_data.tiles
        elif hasattr(level_data, "tile_data"):
            tiles = level_data.tile_data
        elif isinstance(level_data, (list, np.ndarray)):
            tiles = level_data
        else:
            tiles = getattr(level_data, "data", level_data)

        # Convert to numpy array if needed
        if not isinstance(tiles, np.ndarray):
            tiles = np.array(tiles)

        # Ensure proper shape and create tile dictionary
        if tiles.shape == (42, 23):
            tiles = tiles.T  # Transpose to (23, 42) = (height, width)

        height, width = tiles.shape
        for y in range(height):
            for x in range(width):
                tile_val = tiles[y, x]
                if tile_val != 0:
                    mock_sim.tile_dic[(x, y)] = tile_val

        return mock_sim

    def _save_debug_images(
        self,
        collision_mask: np.ndarray,
        traversable_mask: np.ndarray,
        reachable_mask: np.ndarray,
        ninja_pos: Tuple[int, int],
    ):
        """
        Save debug images for visualization.

        Args:
            collision_mask: Original collision mask
            traversable_mask: Mask after ninja radius application
            reachable_mask: Final reachable areas mask
            ninja_pos: Ninja starting position
        """
        import os

        debug_dir = "/tmp/opencv_flood_fill_debug"
        os.makedirs(debug_dir, exist_ok=True)

        # Save collision mask
        cv2.imwrite(
            f"{debug_dir}/collision_mask.png",
            (collision_mask * 255).astype(np.uint8),
        )

        # Save traversable mask
        cv2.imwrite(
            f"{debug_dir}/traversable_mask.png",
            (traversable_mask * 255).astype(np.uint8),
        )

        # Save reachable mask with ninja position marked
        reachable_debug = (reachable_mask * 255).astype(np.uint8)

        # Apply level padding offset for ninja position
        TILE_SIZE = 24  # TILE_PIXEL_SIZE
        adjusted_x = ninja_pos[0] - TILE_SIZE
        adjusted_y = ninja_pos[1] - TILE_SIZE
        ninja_x = int(adjusted_x * self.render_scale)
        ninja_y = int(adjusted_y * self.render_scale)

        # Mark ninja position in red with proper radius scaling (10px ninja radius)
        ninja_radius = max(
            2, int(10 * self.render_scale)
        )  # Scale 10px ninja radius with render scale
        if (
            0 <= ninja_x < reachable_debug.shape[1]
            and 0 <= ninja_y < reachable_debug.shape[0]
        ):
            cv2.circle(reachable_debug, (ninja_x, ninja_y), ninja_radius, 128, -1)

        cv2.imwrite(f"{debug_dir}/reachable_mask.png", reachable_debug)

        if self.debug:
            print(f"DEBUG: Saved debug images to {debug_dir}")
