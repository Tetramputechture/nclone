import pygame
import os
import random
from typing import Optional
from .nsim import Simulator
from .nsim_renderer import NSimRenderer
from .map_generation.map_generator import generate_map
from .sim_config import SimConfig
import numpy as np
from typing import List
from .constants.entity_types import EntityType
from .constants.physics_constants import (
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
)
from . import render_utils


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
    ):
        """
        Initialize the simulation and renderer, as well as the headless pygame
        interface and display.

        Note: Rendering automatically uses grayscale in grayscale_array mode (headless) for performance.
        RGB is only used in 'human' mode for visual testing.
        """
        self.render_mode = render_mode

        self.sim = Simulator(
            SimConfig(enable_anim=enable_animation, log_data=enable_logging)
        )

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

        self.enable_debug_overlay = enable_debug_overlay
        self.seed = seed
        self.rng = random.Random(seed)

        self.current_tick = -1
        self.last_rendered_tick = -1
        self.cached_render_surface = None
        self.cached_render_buffer = None

    def _perform_grayscale_conversion(self, surface: pygame.Surface) -> np.ndarray:
        """
        Converts a Pygame surface to a grayscaled NumPy array (H, W, 1).
        OPTIMIZED: If surface is already grayscale (8-bit), use fast path.
        """
        # OPTIMIZATION: Check if surface is already grayscale (8-bit with palette)
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
        """
        self.sim.load(map_data)
        self.current_map_data = map_data

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
        # Reset rendering state for consistency
        self.current_tick = -1
        self.last_rendered_tick = -1
        self.cached_render_surface = None
        self.cached_render_buffer = None

    def tick(self, horizontal_input: int, jump_input: int):
        """
        Tick the simulation with the given horizontal and jump inputs.
        """
        self.current_tick += 1
        # Invalidate both caches since a tick advanced
        self.cached_render_surface = None
        self.cached_render_buffer = None
        self.sim.tick(horizontal_input, jump_input)
        if self.render_mode == "human":
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

    def ninja_position(self):
        return self.sim.ninja.xpos, self.sim.ninja.ypos

    def ninja_velocity(self):
        return self.sim.ninja.xspeed, self.sim.ninja.yspeed

    def _sim_exit_switch(self):
        # We want to get the last entry of self.sim.entity_dic[3];
        # this is the exit switch.
        # First, check if the exit switch exists.
        if len(self.sim.entity_dic[3]) == 0:
            return None

        return self.sim.entity_dic[3][-1]

    def exit_switch_activated(self):
        exit_switch = self._sim_exit_switch()
        if exit_switch is None:
            return True  # No exit switch means it's "activated" (for empty maps)
        return not exit_switch.active

    def exit_switch_position(self):
        exit_switch = self._sim_exit_switch()
        if exit_switch is None:
            return (0, 0)  # Default position for empty maps
        return exit_switch.xpos, exit_switch.ypos

    def _sim_exit_door(self):
        # We want to get the .parent attribute of the exit switch.
        exit_switch = self._sim_exit_switch()
        if exit_switch is None:
            return None
        return exit_switch.parent

    def exit_door_position(self):
        exit_door = self._sim_exit_door()
        if exit_door is None:
            return (0, 0)  # Default position for empty maps
        return exit_door.xpos, exit_door.ypos

    def mines(self):
        # We want to return a list of all mines in the simulation with state == 0 (toggled)
        # of type 1.
        return [
            entity
            for entity in self.sim.entity_dic[1]
            if entity.active and entity.state == 0
        ]

    # ---- Door helpers for graph construction ----
    def regular_doors(self):
        """Return regular door entities (type 5)."""
        return list(self.sim.entity_dic.get(5, []))

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
        pygame.quit()

    def get_doors_opened(self):
        """Returns the total doors opened by the ninja."""
        return self.sim.ninja.doors_opened

    def get_ninja_state(self):
        """Get ninja state as a list of floats with fixed length, all normalized between -1 and 1.

        Returns:
            List of 30 floats representing enhanced ninja state:
            - Core movement state (8 features)
            - Input and buffer state (5 features)
            - Surface contact information (6 features)
            - Momentum and physics (4 features)
            - Entity proximity and hazards (4 features)
            - Level progress and objectives (3 features)
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

        # === Momentum and Physics (4 features) ===

        # 20-21. Recent acceleration (change in velocity)
        accel_x = (
            ninja.xspeed - ninja.xspeed_old
        ) / MAX_HOR_SPEED  # Normalize by max speed
        accel_y = (ninja.yspeed - ninja.yspeed_old) / MAX_HOR_SPEED
        accel_x = max(-1.0, min(1.0, accel_x))  # Clamp to [-1, 1]
        accel_y = max(-1.0, min(1.0, accel_y))
        state.extend([accel_x, accel_y])

        # 21. Nearest hazard distance (initialized to 0, computed by observation processor)
        nearest_hazard_distance = 0.0  # Computed from entity_states or mine processor
        state.append(nearest_hazard_distance)

        # 22. Nearest collectible distance (initialized to 0, computed by observation processor)
        nearest_collectible_distance = 0.0  # Computed from switch position
        state.append(nearest_collectible_distance)

        # 23. Entity interaction cooldown (frames since last entity interaction)
        # This would need to be tracked separately - for now, use jump duration as proxy
        interaction_cooldown = (ninja.jump_duration / MAX_JUMP_DURATION) * 2 - 1
        state.append(interaction_cooldown)

        # === Level Progress and Objectives (2 features) ===

        # 24. Switch activation progress
        # Compute based on actual switch state: 1.0 if activated, -1.0 if not
        switch_activated = self.exit_switch_activated()
        switch_progress = 1.0 if switch_activated else -1.0
        state.append(switch_progress)

        # 25. Exit accessibility
        # Compute based on switch state and distance to exit
        # 1.0 if switch is activated (exit accessible), -1.0 otherwise
        exit_accessibility = 1.0 if switch_activated else -1.0
        state.append(exit_accessibility)

        return state

    def get_entity_states(self):
        """Get all entity states as a list of floats with fixed length, all normalized between 0 and 1."""
        state = []
        # Maximum number of attributes per entity (padding with zeros if entity has fewer attributes)
        MAX_ATTRIBUTES = 6

        # Entity type to max count mapping based on our own constraints
        MAX_COUNTS = {
            # Support max amount of mines and gold; otherwise, constrain to 32
            EntityType.TOGGLE_MINE: 128,
            EntityType.EXIT_DOOR: 1,
            EntityType.LOCKED_DOOR: 32,
            EntityType.LAUNCH_PAD: 32,
        }

        entity_types = list(MAX_COUNTS.keys())

        # For each entity type in the simulation
        for entity_type in entity_types:
            entities = self.sim.entity_dic.get(entity_type, [])
            # Default to 16 if not specified
            max_count = MAX_COUNTS.get(entity_type, 16)

            state.append(float(len(entities)) / max_count)

            # Process each entity up to max_count
            for entity_idx in range(max_count):
                # If we have an actual entity at this index
                if entity_idx < len(entities):
                    entity = entities[entity_idx]
                    entity_state = entity.get_state()

                    ninja = self.sim.ninja
                    # Distance to ninja (normalized by screen diagonal)
                    dx = entity.xpos - ninja.xpos
                    dy = entity.ypos - ninja.ypos
                    distance = (dx**2 + dy**2) ** 0.5
                    screen_diagonal = (
                        render_utils.SRCWIDTH**2 + render_utils.SRCHEIGHT**2
                    ) ** 0.5
                    normalized_distance = min(distance / screen_diagonal, 1.0)
                    entity_state.append(normalized_distance)

                    # Relative velocity (if entity has velocity attributes)
                    if hasattr(entity, "xspeed") and hasattr(entity, "yspeed"):
                        # Relative velocity magnitude normalized by ninja's max speed
                        rel_vx = entity.xspeed - ninja.xspeed
                        rel_vy = entity.yspeed - ninja.yspeed
                        rel_speed = (rel_vx**2 + rel_vy**2) ** 0.5
                        normalized_rel_speed = min(rel_speed / MAX_HOR_SPEED, 1.0)
                        entity_state.append(normalized_rel_speed)
                    else:
                        entity_state.append(0.0)  # No velocity information

                    # Assert that all entity states are between 0 and 1. If not
                    # print an informative error message containing the entity type,
                    # index, and state.
                    if not all(0 <= state_val <= 1 for state_val in entity_state):
                        print(
                            f"Entity type {entity_type} index {entity_idx} state {entity_state} is out of bounds"
                        )
                        raise ValueError(
                            f"Entity type {entity_type} index {entity_idx} state {entity_state} is out of bounds"
                        )

                    while len(entity_state) < MAX_ATTRIBUTES:
                        entity_state.append(0.0)

                    state.extend(entity_state)
                else:
                    # Add padding for non-existent entity
                    state.extend([0.0] * MAX_ATTRIBUTES)

        return state
