import os
import random
import math
import logging
from typing import Optional, Dict, Any
from .nsim import Simulator
from .map_generation.map_generator import generate_map
from .sim_config import SimConfig
import numpy as np
from typing import List
from .constants.physics_constants import (
    MAX_HOR_SPEED,
    TOGGLE_MINE_RADII,
    GRAVITY_JUMP,
    GRAVITY_FALL,
    DRAG_REGULAR,
    DRAG_SLOW,
    FRICTION_GROUND,
    FRICTION_GROUND_SLOW,
    GROUND_ACCEL,
    AIR_ACCEL,
    LEVEL_HEIGHT_PX,
)
from . import render_utils

logger = logging.getLogger(__name__)


class NPlayHeadless:
    """
    This class is used to run the simulation in headless mode,
    while supporting rendering a frame to a NumPy array. Has manual
    control over the simulation and rendering, and is intended to be
    used for training machine learning agents.

    Has a simple API for interacting with the simulation (moving horizontally
    or jumping, loading a map, resetting and ticking the simulation).
    """

    def __init__(
        self,
        render_mode: str = "grayscale_array",
        enable_animation: bool = False,
        enable_logging: bool = False,
        enable_debug_overlay: bool = False,
        seed: Optional[int] = None,
        enable_rendering: bool = True,
        debug=False,
    ):
        """
        Initialize the simulation and renderer, as well as the headless pygame
        interface and display.

        Note: Rendering automatically uses grayscale in grayscale_array mode (headless) for performance.
        RGB is only used in 'human' mode for visual testing.

        Args:
            enable_rendering: If False, skip pygame initialization entirely for maximum performance.
                Only use when visual observations are not needed (graph+state+reachability sufficient).
        """
        self.render_mode = render_mode
        self.enable_rendering = enable_rendering or render_mode == "human"
        self.debug = debug
        self.sim = Simulator(
            SimConfig(
                enable_anim=enable_animation, log_data=enable_logging, debug=debug
            )
        )

        # Only initialize rendering if needed
        if self.enable_rendering:
            import pygame
            from .nsim_renderer import NSimRenderer

            # OPTIMIZATION: Always use grayscale in headless mode (grayscale_array)
            # RGB only used for human viewing (render_mode="human")
            # This eliminates expensive RGB->grayscale conversion (~30% speedup)
            use_grayscale = render_mode == "grayscale_array"

            self.sim_renderer = NSimRenderer(
                self.sim, render_mode, enable_debug_overlay, grayscale=use_grayscale
            )
            self.current_map_data = None
            self.clock = pygame.time.Clock()

            # init pygame
            pygame.init()
            pygame.display.init()
            if self.render_mode == "grayscale_array":
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            else:
                print("Setting up pygame display")
                pygame.display.set_mode((render_utils.SRCWIDTH, render_utils.SRCHEIGHT))

            # Pre-allocate buffer for surface to array conversion
            self._render_buffer = np.empty(
                (render_utils.SRCWIDTH, render_utils.SRCHEIGHT, 3), dtype=np.uint8
            )
        else:
            # Rendering disabled - skip pygame initialization entirely
            self.sim_renderer = None
            self.current_map_data = None
            self.clock = None
            self._render_buffer = None

        self.enable_debug_overlay = enable_debug_overlay
        self.seed = seed
        self.rng = random.Random(seed)

        self.current_tick = -1
        self.last_rendered_tick = -1
        self.cached_render_surface = None
        self.cached_render_buffer = None

    def _perform_grayscale_conversion(self, surface: Any) -> np.ndarray:
        """
        Converts a Pygame surface to a grayscaled NumPy array (H, W, 1).
        OPTIMIZED: If surface is already grayscale (8-bit), use fast path.
        """
        import pygame

        if surface.get_bytesize() == 1:
            # Grayscale surface - use fast pixels2d (no RGB conversion needed!)
            try:
                # Get direct reference to pixel data (W, H)
                referenced_array_wh = pygame.surfarray.pixels2d(surface)
                # Transpose to (H, W) and add channel dimension
                grayscale_hw = np.transpose(referenced_array_wh, (1, 0))
                # Copy and add channel dimension (H, W, 1)
                final_gray_output_hw1 = np.array(
                    grayscale_hw, copy=True, dtype=np.uint8
                )[..., np.newaxis]
                del referenced_array_wh  # Unlock surface
                return final_gray_output_hw1
            except Exception:
                # Fallback to array2d if pixels2d fails
                array_wh = pygame.surfarray.array2d(surface)
                grayscale_hw = np.transpose(array_wh, (1, 0))
                return grayscale_hw.astype(np.uint8)[..., np.newaxis]

        # RGB surface - perform grayscale conversion
        try:
            # Attempt to get a referenced array (W, H, C)
            referenced_array_whc = pygame.surfarray.pixels3d(surface)
            # Transpose W, H, C to H, W, C
            transposed_array_hwc = np.transpose(referenced_array_whc, (1, 0, 2))
            # Grayscale it (H, W, 1)
            # Inline frame_to_grayscale logic
            processed_frame = transposed_array_hwc[..., :3]
            grayscale = (
                0.2989 * processed_frame[..., 0]
                + 0.5870 * processed_frame[..., 1]
                + 0.1140 * processed_frame[..., 2]
            )
            final_gray_output_hw1 = grayscale[..., np.newaxis].astype(np.uint8)
            del referenced_array_whc  # Unlock surface
        except pygame.error:  # If pixels3d fails
            # Fallback to original method using pre-allocated self._render_buffer (W,H,3)
            pygame.pixelcopy.surface_to_array(self._render_buffer, surface)
            # Transpose W, H, C to H, W, C
            transposed_render_buffer_hwc = np.transpose(self._render_buffer, (1, 0, 2))
            # Grayscale it (H, W, 1)
            # Inline frame_to_grayscale logic
            processed_frame = transposed_render_buffer_hwc[..., :3]
            grayscale = (
                0.2989 * processed_frame[..., 0]
                + 0.5870 * processed_frame[..., 1]
                + 0.1140 * processed_frame[..., 2]
            )
            final_gray_output_hw1 = grayscale[..., np.newaxis].astype(np.uint8)
        return final_gray_output_hw1

    def load_map_from_map_data(self, map_data: List[int]):
        """
        Load a map from a list of integers.

        NOTE: When using curriculum learning, the caller should call
        clear_graph_caches_for_curriculum_load() on the environment AFTER
        this method to ensure all graph and path caches are invalidated.
        This prevents "graph pollution" where goal positions from previous
        levels persist in SubprocVecEnv scenarios.
        """
        self.sim.load(map_data)
        self.current_map_data = map_data

        # Set flag to indicate a new map was loaded (can be checked by environment)
        self._map_just_loaded = True

        # Clear pathfinding visualization cache when map changes
        if hasattr(self, "sim_renderer") and hasattr(
            self.sim_renderer, "debug_overlay_renderer"
        ):
            from nclone.cache_management import (
                clear_pathfinding_caches,
                clear_debug_overlay_caches,
            )

            clear_pathfinding_caches(self.sim_renderer.debug_overlay_renderer)
            # Also clear debug overlay caches to prevent stale visualizations
            clear_debug_overlay_caches(self.sim_renderer.debug_overlay_renderer)

        # Clear render caches to force fresh rendering of new map
        if hasattr(self, "cached_render_surface"):
            self.cached_render_surface = None
        if hasattr(self, "cached_render_buffer"):
            self.cached_render_buffer = None

    def load_map(self, map_path: str):
        """
        Load a map from a file.
        """
        with open(map_path, "rb") as map_file:
            map_data = [int(b) for b in map_file.read()]
        self.load_map_from_map_data(map_data)

    def load_random_map(
        self, map_type: Optional[str] = "SIMPLE_HORIZONTAL_NO_BACKTRACK"
    ):
        """
        Generate a random map and load it into the simulator.
        """
        # Get the map data
        map_data = generate_map(level_type=map_type, seed=self.seed).map_data()
        self.load_map_from_map_data(map_data)

    def load_random_official_map(self):
        """
        Load a random official map from the maps/official folder.
        """
        base_map_path = os.path.join(os.path.dirname(__file__), "maps", "official")

        # Get all subfolders in maps/official
        subfolders = [
            f
            for f in os.listdir(base_map_path)
            if os.path.isdir(os.path.join(base_map_path, f))
        ]

        # Choose random subfolder
        subfolder = self.rng.choice(subfolders)
        subfolder_path = os.path.join(base_map_path, subfolder)

        # Choose random map from subfolder
        map_file = self.rng.choice(os.listdir(subfolder_path))
        map_path = os.path.join(subfolder_path, map_file)

        self.load_map(map_path)
        return os.path.join(subfolder, map_file)

    def reset(self):
        """
        Reset the simulation to the initial state, including rendering caches and ticks.
        """
        self.sim.reset()

        if self.enable_rendering:
            # Clear rendering caches
            from nclone.cache_management import (
                clear_render_caches,
                clear_pathfinding_caches,
            )

            clear_render_caches(self)
            # Clear pathfinding cache when ninja resets (position changes)
            if hasattr(self, "sim_renderer") and hasattr(
                self.sim_renderer, "debug_overlay_renderer"
            ):
                clear_pathfinding_caches(self.sim_renderer.debug_overlay_renderer)

    def tick(self, horizontal_input: int, jump_input: int):
        """
        Tick the simulation with the given horizontal and jump inputs.
        """
        self.current_tick += 1
        self.sim.tick(horizontal_input, jump_input)
        if self.enable_rendering:
            self.cached_render_surface = None
            self.cached_render_buffer = None
            self.clock.tick(120)

    def render(self, debug_info: Optional[dict] = None):
        """
        Render the current frame.
        If render_mode is 'grayscale_array', returns a grayscaled HxWx1 NumPy array.
        If render_mode is 'human', returns the Pygame surface.
        Uses caching for performance.

        Args:
            debug_info: A dictionary containing debug information to be displayed on the screen.
        """
        # Return None if rendering is disabled (visual observations not needed)
        if not self.enable_rendering or self.sim_renderer is None:
            return None

        if self.current_tick == self.last_rendered_tick:
            if (
                self.render_mode == "human" and self.cached_render_surface is not None
            ):  # If human mode, return surface directly
                if debug_info is not None:
                    # This will redraw the game and the overlay if new debug_info is provided
                    surface = self.sim_renderer.draw(self.sim.frame <= 1, debug_info)
                    self.cached_render_surface = surface
                    return self.cached_render_surface
                return self.cached_render_surface

            # If not human mode, or surface cache miss, check buffer cache
            if self.cached_render_buffer is not None:  # This is H, W, 1 grayscale
                return self.cached_render_buffer
            elif (
                self.cached_render_surface is not None
            ):  # Mode might have switched from human to grayscale_array, or first grayscale_array render
                # Generate grayscaled buffer from self.cached_render_surface
                gray_array_hw1 = self._perform_grayscale_conversion(
                    self.cached_render_surface
                )
                self.cached_render_buffer = (
                    gray_array_hw1.copy()
                )  # Cache the new grayscale buffer
                return self.cached_render_buffer

        init = self.sim.frame <= 1
        surface = self.sim_renderer.draw(init, debug_info)

        self.cached_render_surface = surface  # Always cache the raw surface for potential mode switch or human display
        self.last_rendered_tick = self.current_tick

        if self.render_mode == "human":
            # For human mode, the surface is already updated and displayed by NSimRenderer.draw
            return self.cached_render_surface

        final_gray_output_hw1 = self._perform_grayscale_conversion(surface)
        self.cached_render_buffer = (
            final_gray_output_hw1.copy()
        )  # Cache the (H,W,1) grayscale buffer
        return final_gray_output_hw1

    def ninja_has_won(self):
        return self.sim.ninja.has_won()

    def ninja_has_died(self):
        return self.sim.ninja.has_died()

    def ninja_death_cause(self):
        """Get the cause of ninja's death if available.

        Returns:
            String indicating death cause ("mine", "impact", "hazard", etc.) or None
        """
        return getattr(self.sim.ninja, "death_cause", None)

    def ninja_position(self):
        return self.sim.ninja.xpos, self.sim.ninja.ypos

    def ninja_velocity(self):
        return self.sim.ninja.xspeed, self.sim.ninja.yspeed

    def ninja_velocity_old(self):
        """Get ninja's previous frame velocities for impact risk calculation."""
        return self.sim.ninja.xspeed_old, self.sim.ninja.yspeed_old

    def ninja_last_jump_was_buffered(self) -> bool:
        """Check if the last jump executed was via buffer (frame-perfect execution).

        Returns True if the last jump executed used a buffer (floor_buffer, wall_buffer, etc.)
        for frame-perfect timing.
        """
        return self.sim.ninja.last_jump_was_buffered

    def get_action_mask(self) -> list:
        """Get mask of valid actions for current ninja state.

        Currently only masks actions with useless jump component.
        Horizontal movement is NOT masked due to wall interaction complexity.

        Returns:
            List of 6 bools (True = action is valid, False = action is masked/invalid)
        """
        return self.sim.ninja.get_valid_action_mask()

    def ninja_airborn_old(self):
        """Get ninja's previous frame airborne state for impact risk calculation."""
        return self.sim.ninja.airborn_old

    def ninja_floor_normal(self):
        """Get normalized floor normal vector for impact risk calculation."""
        return self.sim.ninja.floor_normalized_x, self.sim.ninja.floor_normalized_y

    def ninja_ceiling_normal(self):
        """Get normalized ceiling normal vector for impact risk calculation."""
        return self.sim.ninja.ceiling_normalized_x, self.sim.ninja.ceiling_normalized_y

    def _ensure_entities_loaded(self) -> bool:
        """
        Verify that entities are loaded and ready to query.

        Returns:
            True if entities are loaded, False otherwise.

        If entities are not loaded, this logs detailed diagnostics to help
        identify the cause of the issue.
        """
        # Check if entity_dic exists and has exit switch (type 3)
        if not hasattr(self.sim, "entity_dic"):
            logger.error("ENTITY_LOAD_CHECK: sim.entity_dic does not exist!")
            return False

        exit_entities = self.sim.entity_dic.get(3, [])
        if len(exit_entities) == 0:
            # No exit switch - this could be normal for empty maps or a loading issue
            entity_keys = list(self.sim.entity_dic.keys())
            entity_counts = {k: len(v) for k, v in self.sim.entity_dic.items()}

            # If entity_dic is completely empty, that's a loading problem
            if len(entity_keys) == 0:
                logger.error(
                    f"ENTITY_LOAD_CHECK: entity_dic is empty! "
                    f"Frame: {self.sim.frame}, map_data length: {len(self.sim.map_data) if self.sim.map_data else 0}"
                )
                return False
            else:
                # Some entities exist but no exit switch - might be intentional (test maps)
                logger.debug(
                    f"ENTITY_LOAD_CHECK: No exit switch (type 3). "
                    f"Available types: {entity_keys}, counts: {entity_counts}"
                )
                return True  # Entities are loaded, just no exit switch

        return True

    def _reload_entities_if_needed(self):
        """
        Force reload entities if they appear to be missing.

        This is a recovery mechanism for curriculum learning where entity loading
        might fail silently.
        """
        if not self._ensure_entities_loaded():
            logger.warning(
                "Entities not loaded - attempting to reload via map_loader.load_map_entities()"
            )
            try:
                self.sim.map_loader.load_map_entities()
                if self._ensure_entities_loaded():
                    logger.info("Entity reload successful!")
                else:
                    logger.error("Entity reload failed - entities still not loaded")
            except Exception as e:
                logger.error(f"Entity reload raised exception: {e}")

    def verify_entities_loaded(self) -> bool:
        """
        Public method to verify entities are properly loaded after a map load.

        Should be called by curriculum environments after load_map_from_map_data()
        to ensure entities are available for position queries.

        Returns:
            True if entities are loaded (exit switch available), False otherwise.
        """
        exit_switch = self._sim_exit_switch()
        if exit_switch is None:
            # Log detailed diagnostics
            entity_keys = list(self.sim.entity_dic.keys())
            entity_counts = {k: len(v) for k, v in self.sim.entity_dic.items()}
            logger.warning(
                f"verify_entities_loaded: No exit switch found! "
                f"entity_dic keys: {entity_keys}, counts: {entity_counts}, "
                f"frame: {self.sim.frame}"
            )
            return False

        # Verify switch has valid position
        if exit_switch.xpos == 0 and exit_switch.ypos == 0:
            logger.warning(
                "verify_entities_loaded: Exit switch has (0, 0) position! "
                "This may indicate incomplete entity loading."
            )
            return False

        return True

    def _sim_exit_switch(self):
        """Get exit switch entity (type 4, stored in entity_dic[3]).
        
        Note: entity_dic[3] contains both EntityExit (exit doors) and EntityExitSwitch
        (exit switches). We must filter by type to ensure we get the switch, not the door.
        """
        # Safely check if entity type 3 exists and has entries
        exit_entities = self.sim.entity_dic.get(3, [])
        if len(exit_entities) == 0:
            return None
        
        # Find the actual EntityExitSwitch (not EntityExit) by checking type name
        # Iterate in reverse to prioritize more recently added entities
        for entity in reversed(exit_entities):
            if type(entity).__name__ == "EntityExitSwitch":
                return entity
        
        # No exit switch found in the list
        return None

    def exit_switch_activated(self):
        exit_switch = self._sim_exit_switch()
        if exit_switch is None:
            return True  # No exit switch means it's "activated" (for empty maps)
        return not exit_switch.active

    def exit_switch_position(self):
        """
        Get exit switch position.

        Returns:
            Tuple (x, y) of exit switch position, or (0, 0) if no exit switch.

        IMPORTANT: Returns (0, 0) if entities are not loaded. Callers should
        validate the returned position is not (0, 0) before caching it.
        """
        exit_switch = self._sim_exit_switch()
        if exit_switch is None:
            # Check if this is expected (no exit switch) vs unexpected (entities not loaded)
            self._ensure_entities_loaded()
            return (0, 0)
        return exit_switch.xpos, exit_switch.ypos

    def _sim_exit_door(self):
        # We want to get the .parent attribute of the exit switch.
        exit_switch = self._sim_exit_switch()
        if exit_switch is None:
            return None
        return exit_switch.parent

    def exit_door_position(self):
        """
        Get exit door position.

        Returns:
            Tuple (x, y) of exit door position, or (0, 0) if no exit door.

        IMPORTANT: Returns (0, 0) if entities are not loaded. Callers should
        validate the returned position is not (0, 0) before caching it.
        """
        exit_door = self._sim_exit_door()
        if exit_door is None:
            # Check if this is expected (no exit door) vs unexpected (entities not loaded)
            self._ensure_entities_loaded()
            return (0, 0)
        return exit_door.xpos, exit_door.ypos

    def mines(self):
        """Return list of active toggled mines."""
        toggle_mines = self.sim.entity_dic.get(1, [])
        return [
            entity for entity in toggle_mines if entity.active and entity.state == 0
        ]

    def get_all_mine_data_for_visualization(self) -> List[Dict[str, Any]]:
        """Get all mine data for route visualization.

        Returns data for both entity types 1 (untoggled mines) and 21 (toggled mines).
        This method centralizes mine extraction logic to avoid duplication.

        Returns:
            List of mine dictionaries with keys: x, y, state, radius
        """
        mines = []
        entity_dic = self.sim.entity_dic

        # Process entity type 1 (untoggled mines - start in untoggled state)
        if 1 in entity_dic:
            toggle_mines = entity_dic[1]
            for mine in toggle_mines:
                if hasattr(mine, "xpos") and hasattr(mine, "ypos"):
                    state = getattr(
                        mine, "state", 1
                    )  # Type 1 starts untoggled (state 1)
                    radius = TOGGLE_MINE_RADII.get(state, 4.0)

                    mines.append(
                        {
                            "x": float(mine.xpos),
                            "y": float(mine.ypos),
                            "state": int(state),
                            "radius": float(radius),
                        }
                    )

        # Process entity type 21 (toggled mines - start in toggled state)
        if 21 in entity_dic:
            toggled_mines = entity_dic[21]
            for mine in toggled_mines:
                if hasattr(mine, "xpos") and hasattr(mine, "ypos"):
                    state = getattr(
                        mine, "state", 0
                    )  # Type 21 starts toggled (state 0)
                    radius = TOGGLE_MINE_RADII.get(state, 4.0)

                    mines.append(
                        {
                            "x": float(mine.xpos),
                            "y": float(mine.ypos),
                            "state": int(state),
                            "radius": float(radius),
                        }
                    )

        return mines

    def locked_doors(self):
        """Return locked door entities (type 6). Includes their switch coordinates."""
        return list(self.sim.entity_dic.get(6, []))

    def locked_door_switches(self):
        """Return locked door switch entities (type 7)."""
        return list(self.sim.entity_dic.get(7, []))

    def get_tile_data(self):
        """Get tile data from simulator."""
        return self.sim.tile_dic

    def exit(self):
        if self.enable_rendering:
            import pygame

            pygame.quit()

    def get_ninja_terminal_impact(self):
        return self.sim.ninja.terminal_impact

    def get_ninja_state(self):
        """Get ninja state as a list of floats with fixed length, all normalized between -1 and 1.

        Returns:
            List of GAME_STATE_CHANNELS floats representing ninja state:
            - Core movement state (8 features)
            - Input and buffer state (5 features)
            - Surface contact information (6 features)
            - Additional physics state (7 features)
            - Temporal features (6 features)
            - Physics features (8 features)
        """
        ninja = self.sim.ninja
        state = []

        # === Core Movement State (8 features) ===

        # 1. Velocity magnitude (normalized by max possible speed)
        velocity_magnitude = (ninja.xspeed**2 + ninja.yspeed**2) ** 0.5
        max_velocity = MAX_HOR_SPEED * 2  # Account for vertical velocity
        velocity_mag_norm = (
            min(velocity_magnitude / max_velocity, 1.0) * 2 - 1
        )  # [-1, 1]
        state.append(velocity_mag_norm)

        # 2-3. Velocity direction (normalized, handling zero velocity)
        if velocity_magnitude > 1e-6:
            velocity_dir_x = ninja.xspeed / velocity_magnitude  # Already [-1, 1]
            velocity_dir_y = ninja.yspeed / velocity_magnitude
        else:
            velocity_dir_x = 0.0
            velocity_dir_y = 0.0
        state.extend([velocity_dir_x, velocity_dir_y])

        # 4-7. Movement state categories (compressed from 10 states to 4 categories)
        ground_movement = (
            1.0 if ninja.state in [0, 1, 2] else -1.0
        )  # Immobile, Running, Ground sliding
        air_movement = 1.0 if ninja.state in [3, 4] else -1.0  # Jumping, Falling
        wall_interaction = 1.0 if ninja.state == 5 else -1.0  # Wall sliding
        special_states = (
            1.0 if ninja.state in [6, 7, 8, 9] else -1.0
        )  # Dead, Awaiting death, Celebrating, Disabled
        state.extend([ground_movement, air_movement, wall_interaction, special_states])

        # 8. Airborne status
        airborne_status = 1.0 if ninja.airborn else -1.0
        state.append(airborne_status)

        # === Input and Buffer State (5 features) ===

        # 9. Current horizontal input (normalized to [-1, 1])
        horizontal_input = float(ninja.hor_input)  # Already -1, 0, or 1
        state.append(horizontal_input)

        # 10. Current jump input
        jump_input = 1.0 if ninja.jump_input else -1.0
        state.append(jump_input)

        # 11-13. Buffer states (normalized to [-1, 1])
        buffer_window_size = 5.0
        jump_buffer_norm = (max(ninja.jump_buffer, 0) / buffer_window_size) * 2 - 1
        floor_buffer_norm = (max(ninja.floor_buffer, 0) / buffer_window_size) * 2 - 1
        wall_buffer_norm = (max(ninja.wall_buffer, 0) / buffer_window_size) * 2 - 1
        state.extend([jump_buffer_norm, floor_buffer_norm, wall_buffer_norm])

        # === Surface Contact Information (6 features) ===

        # 14-16. Contact strength (normalized)
        floor_contact = (min(ninja.floor_count, 1) * 2) - 1  # Convert 0,1 to -1,1
        wall_contact = (min(ninja.wall_count, 1) * 2) - 1
        ceiling_contact = (min(ninja.ceiling_count, 1) * 2) - 1
        state.extend([floor_contact, wall_contact, ceiling_contact])

        # 17. Floor normal strength
        floor_normal_strength = (
            ninja.floor_normalized_x**2 + ninja.floor_normalized_y**2
        ) ** 0.5
        floor_normal_strength = (floor_normal_strength * 2) - 1  # Normalize to [-1, 1]
        state.append(floor_normal_strength)

        # 18. Wall normal direction (if wall contact exists)
        if ninja.wall_count > 0 and hasattr(ninja, "wall_normal"):
            # wall_normal is a scalar indicating direction (-1 or 1)
            wall_direction = float(ninja.wall_normal)  # Already [-1, 1]
        else:
            wall_direction = 0.0
        state.append(wall_direction)

        # 19. Surface slope (floor normal y component indicates slope)
        surface_slope = ninja.floor_normalized_y  # Already [-1, 1]
        state.append(surface_slope)

        # === Additional Physics State (5 features) ===

        # 20. Applied gravity (normalized between GRAVITY_JUMP and GRAVITY_FALL)
        # GRAVITY_JUMP < GRAVITY_FALL, so normalize: -1 = GRAVITY_JUMP, 1 = GRAVITY_FALL
        gravity_range = GRAVITY_FALL - GRAVITY_JUMP
        if gravity_range > 1e-6:
            gravity_norm = (
                ninja.applied_gravity - GRAVITY_JUMP
            ) / gravity_range * 2 - 1  # [-1, 1]
        else:
            gravity_norm = 0.0
        state.append(gravity_norm)

        # 21. Walled status (boolean to -1/1)
        walled_status = 1.0 if ninja.walled else -1.0
        state.append(walled_status)

        # 22. Floor normal x-component (full x-component, already [-1, 1])
        floor_normal_x = ninja.floor_normalized_x
        state.append(floor_normal_x)

        # 23-24. Ceiling normal vector (full ceiling normal, already [-1, 1])
        ceiling_normal_x = ninja.ceiling_normalized_x
        ceiling_normal_y = ninja.ceiling_normalized_y
        state.extend([ceiling_normal_x, ceiling_normal_y])

        # 25. Applied drag (normalized between DRAG_SLOW and DRAG_REGULAR)
        # DRAG_SLOW < DRAG_REGULAR, so normalize: -1 = DRAG_SLOW, 1 = DRAG_REGULAR
        drag_range = DRAG_REGULAR - DRAG_SLOW
        if drag_range > 1e-6:
            drag_norm = (ninja.applied_drag - DRAG_SLOW) / drag_range * 2 - 1  # [-1, 1]
        else:
            drag_norm = 0.0
        state.append(drag_norm)

        # 26. Applied friction (normalized between FRICTION_GROUND_SLOW and FRICTION_GROUND)
        # FRICTION_GROUND_SLOW < FRICTION_GROUND, so normalize: -1 = FRICTION_GROUND_SLOW, 1 = FRICTION_GROUND
        friction_range = FRICTION_GROUND - FRICTION_GROUND_SLOW
        if friction_range > 1e-6:
            friction_norm = (
                ninja.applied_friction - FRICTION_GROUND_SLOW
            ) / friction_range * 2 - 1  # [-1, 1]
        else:
            friction_norm = 0.0
        state.append(friction_norm)

        # === Temporal Features (6 features) ===
        # 27-32. Temporal dynamics from ninja.get_temporal_features()
        temporal_features = ninja.get_temporal_features()
        state.extend(temporal_features)

        # === Enhanced Physics Features (8 features) ===

        # 33. Kinetic energy (normalized, always computable from current velocity)
        kinetic_energy = 0.5 * (ninja.xspeed**2 + ninja.yspeed**2)
        kinetic_energy_norm = min(kinetic_energy / (MAX_HOR_SPEED**2), 1.0) * 2 - 1
        state.append(kinetic_energy_norm)

        # 34. Potential energy (height-based, using current position)
        potential_energy_norm = (ninja.ypos / LEVEL_HEIGHT_PX) * 2 - 1
        state.append(potential_energy_norm)

        # 35. Applied force magnitude (from current physics constants)
        force_magnitude = math.sqrt(
            ninja.applied_gravity**2
            + (GROUND_ACCEL if not ninja.airborn else AIR_ACCEL) ** 2
        )
        force_magnitude_norm = min(force_magnitude / 0.1, 1.0) * 2 - 1
        state.append(force_magnitude_norm)

        # 36. Energy change rate (using already-tracked xspeed_old, yspeed_old)
        prev_kinetic = 0.5 * (ninja.xspeed_old**2 + ninja.yspeed_old**2)
        energy_change_rate = (kinetic_energy - prev_kinetic) / max(
            kinetic_energy + 0.01, 0.01
        )
        energy_change_norm = max(-1.0, min(1.0, energy_change_rate))
        state.append(energy_change_norm)

        # 37. Contact strength (normalized floor contact count)
        floor_contact_strength = min(ninja.floor_count / 5.0, 1.0) * 2 - 1
        state.append(floor_contact_strength)

        # 38. Wall contact strength (normalized wall contact count)
        wall_contact_strength = min(ninja.wall_count / 3.0, 1.0) * 2 - 1
        state.append(wall_contact_strength)

        # 39. Surface slope angle (from existing floor_normalized_x, floor_normalized_y)
        surface_slope = (
            math.atan2(ninja.floor_normalized_y, ninja.floor_normalized_x) / math.pi
        )
        state.append(surface_slope)

        # 40. Wall interaction strength (from existing wall_normal, already computed)
        wall_interaction = ninja.wall_normal if ninja.walled else 0.0
        state.append(wall_interaction)

        return state
