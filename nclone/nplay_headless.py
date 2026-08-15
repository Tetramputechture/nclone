import os
import random
import math
import logging
from typing import Optional, Dict, Any, List, Sequence, Tuple
from .nsim import Simulator
from .sim_config import SimConfig
import numpy as np
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
        panel_width: int = 0,
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
        # The ninja's bone skeleton is not purely cosmetic: every hitscan kill
        # test (`FUN_1050B070`) intersects its beam against 11 limbs, so a beam
        # can pass within the ninja's radius and miss between them. Without bones
        # `check_lineseg_vs_ninja` falls back to the circle prefilter alone, which
        # is strictly MORE lethal than the game.
        #
        # This flag is no longer what decides that. `MapLoader.load_map_entities`
        # turns the skeleton on for any map carrying a beam-capable entity
        # regardless, which is every map where it changes an outcome; `enable_anim`
        # now only forces it on for the *renderer*, which wants limbs everywhere.
        # `NCLONE_ENABLE_ANIM=1` still does that from the environment.
        if os.environ.get("NCLONE_ENABLE_ANIM") == "1":
            enable_animation = True
        self.sim = Simulator(
            SimConfig(
                enable_anim=enable_animation, log_data=enable_logging, debug=debug
            )
        )

        # Only initialize rendering if needed
        if self.enable_rendering:
            import pygame
            from .nsim_renderer import NSimRenderer

            # Only human mode ever puts pixels on a screen. Every array mode
            # draws into an OFFSCREEN surface (see NSimRenderer) and render()
            # reads the pixels straight back off it, so the display subsystem is
            # dead weight there. Pin the dummy driver before `pygame.init()`
            # brings the display up, or a headless caller (lattice precompute,
            # dataset builds, tests) opens a real window and steals the host's
            # focus.
            #
            # setdefault, and only while the display is still down: an explicit
            # driver choice by the caller wins, and a process that already has a
            # window (the dev tool) must not have it pulled out from under it by
            # an array-mode sim built alongside it.
            if self.render_mode != "human" and not pygame.display.get_init():
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

            # Initialise pygame BEFORE anything opens a window. NSimRenderer
            # calls `display.set_mode` in its constructor for human mode, and
            # that used to happen before `pygame.init()` had run.
            pygame.init()
            pygame.display.init()

            # OPTIMIZATION: Use grayscale surface only for grayscale_array mode
            # RGB is used for human viewing (render_mode="human") and rgb_array (video recording)
            # Grayscale mode eliminates expensive RGB->grayscale conversion (~30% speedup)
            use_grayscale = render_mode == "grayscale_array"

            self.sim_renderer = NSimRenderer(
                self.sim,
                render_mode,
                enable_debug_overlay,
                grayscale=use_grayscale,
                panel_width=panel_width,
            )
            self.current_map_data = None
            self.clock = pygame.time.Clock()

            # No `set_mode` here for ANY array mode. rgb_array used to open a
            # window "for the display flip", but there is no flip on that path:
            # NSimRenderer only flips in human mode, and render() converts the
            # offscreen surface it returns. The window was pure cost -- an empty
            # 1056x600 frame that grabbed focus on every headless run.
            #
            # Human mode deliberately does NOT re-set the mode either.
            # NSimRenderer has already opened the window at this exact size, with
            # RESIZABLE, and holds the surface it drew into. Calling `set_mode` a
            # second time recreates the window: it drops the RESIZABLE flag,
            # leaves `NSimRenderer.screen` pointing at a surface that is no
            # longer the display, and resets SDL's cursor state -- which is why
            # the mouse pointer vanished over the dev tool window.

            if self.render_mode == "human":
                # Be explicit rather than trusting whatever state the window
                # creation path left behind. The dev panel is mouse-driven, so a
                # hidden cursor makes it unusable.
                pygame.mouse.set_visible(True)

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

    def _perform_rgb_conversion(self, surface: Any) -> np.ndarray:
        """
        Converts a Pygame surface to an RGB NumPy array (H, W, 3).
        Used for rgb_array render mode (video recording).
        """
        import pygame

        try:
            # Get RGB array from surface (W, H, 3)
            rgb_array_whc = pygame.surfarray.array3d(surface)
            # Transpose to (H, W, 3)
            rgb_array_hwc = np.transpose(rgb_array_whc, (1, 0, 2))
            return rgb_array_hwc.astype(np.uint8)
        except pygame.error:
            # Fallback: use pixelcopy
            rgb_buffer = np.empty(
                (surface.get_width(), surface.get_height(), 3), dtype=np.uint8
            )
            pygame.pixelcopy.surface_to_array(rgb_buffer, surface)
            # Transpose to (H, W, 3)
            rgb_array_hwc = np.transpose(rgb_buffer, (1, 0, 2))
            return rgb_array_hwc

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

    def fast_reset(self):
        """Fast reset for same-level training.

        Uses Simulator.fast_reset() to reset entity states in-place without
        recreating objects. Significantly faster than reset() for same-level
        training scenarios.

        CRITICAL: Must be called ONLY when the map has not changed since the last
        reset. For new maps, use reset() instead. Entities are reset to their
        ORIGINAL positions.
        """
        self.sim.fast_reset()

        if self.enable_rendering:
            # Clear rendering caches (same as reset())
            # Render caches must be cleared because entity positions may change
            # (even if they reset to original positions, they may have moved during episode)
            self.cached_render_surface = None
            self.cached_render_buffer = None

            # Clear pathfinding cache when ninja resets (position changes)
            if hasattr(self, "sim_renderer") and hasattr(
                self.sim_renderer, "debug_overlay_renderer"
            ):
                from nclone.cache_management import clear_pathfinding_caches

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
            if self.cached_render_buffer is not None:
                # Cached buffer can be grayscale (H, W, 1) or RGB (H, W, 3)
                return self.cached_render_buffer
            elif (
                self.cached_render_surface is not None
            ):  # Mode might have switched, or first array render
                # Generate appropriate buffer from self.cached_render_surface
                if self.render_mode == "rgb_array":
                    rgb_array_hw3 = self._perform_rgb_conversion(
                        self.cached_render_surface
                    )
                    self.cached_render_buffer = rgb_array_hw3.copy()
                    return self.cached_render_buffer
                else:  # grayscale_array mode
                    gray_array_hw1 = self._perform_grayscale_conversion(
                        self.cached_render_surface
                    )
                    self.cached_render_buffer = gray_array_hw1.copy()
                    return self.cached_render_buffer

        init = self.sim.frame <= 1
        surface = self.sim_renderer.draw(init, debug_info)

        self.cached_render_surface = surface  # Always cache the raw surface for potential mode switch or human display
        self.last_rendered_tick = self.current_tick

        if self.render_mode == "human":
            # For human mode, the surface is already updated and displayed by NSimRenderer.draw
            return self.cached_render_surface
        elif self.render_mode == "rgb_array":
            # For rgb_array mode, convert surface to RGB numpy array (H, W, 3)
            rgb_output_hw3 = self._perform_rgb_conversion(surface)
            self.cached_render_buffer = rgb_output_hw3.copy()  # Cache the RGB array
            return rgb_output_hw3

        # For grayscale_array mode, convert to grayscale
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

    def get_state_snapshot(self):
        """Capture complete simulation state for replay-free checkpoint restore.

        Returns a backend-tagged, picklable blob ``("python", state)``. The tag
        lets :meth:`set_state_snapshot` reject a snapshot taken under a different
        physics backend.
        """
        return ("python", self.sim.get_full_state())

    def set_state_snapshot(self, blob):
        """Restore a snapshot from :meth:`get_state_snapshot` (no action replay)."""
        backend, state = blob
        if backend != "python":
            raise ValueError(
                f"Cannot restore a {backend!r} snapshot into the python backend"
            )
        self.sim.restore_full_state(state)

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
        """Check if exit switch has been activated (collected).

        Returns:
            True if switch is activated/collected (active=False), False if not collected (active=True)
        """
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

    def get_mine_entities(self) -> Tuple[List, List]:
        """Return mine entity objects directly for efficient spatial context computation.

        Returns raw entity objects instead of processed dictionaries for performance.
        This avoids unnecessary dictionary creation and type filtering overhead.

        Returns:
            Tuple of (toggle_mines, toggled_mines) entity object lists
        """
        entity_dic = self.sim.entity_dic
        return (entity_dic.get(1, []), entity_dic.get(21, []))

    def get_drone_positions(
        self,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Return ``(zap_xy, mini_xy)`` — live positions of the patrol drones.

        Backend-uniform accessor mirrored by
        ``NativePlayHeadless.get_drone_positions``. Split by class because zap
        (type 14, radius 7.5) and mini (type 26, radius 4) drones have different
        contact-death radii. The chaser (15) is excluded: its movement depends
        on the ninja, so it is not forward-rollable and needs separate handling.
        """
        entity_dic = self.sim.entity_dic
        return (
            [(float(d.xpos), float(d.ypos)) for d in entity_dic.get(14, [])],
            [(float(d.xpos), float(d.ypos)) for d in entity_dic.get(26, [])],
        )

    def get_predicted_drone_positions(
        self, horizons: Sequence[int]
    ) -> List[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        """Patrol-drone positions ``horizons`` frames from now, one entry per horizon.

        Each entry has the same ``(zap_xy, mini_xy)`` shape as
        :meth:`get_drone_positions`, so callers treat "now" and "later"
        identically. ``horizons`` must be sorted ascending.

        The forecast is exact — see ``EntityDroneBase.predict_positions``.
        Mirrored by ``NativePlayHeadless.get_predicted_drone_positions``.
        """
        entity_dic = self.sim.entity_dic
        horizons = list(horizons)
        out = [([], []) for _ in horizons]
        for type_id, slot in ((14, 0), (26, 1)):
            for drone in entity_dic.get(type_id, []):
                for entry, xy in zip(out, drone.predict_positions(horizons)):
                    entry[slot].append(xy)
        return out

    def get_gold_state(self) -> List[Tuple[float, float, bool]]:
        """Return ``[(x, y, active), ...]`` for every gold entity (type 2).

        Backend-uniform accessor mirrored by ``NativePlayHeadless.get_gold_state``.
        The python sim's gold objects mutate in place, so ``active`` reflects live
        collection; positions are static, so the list length is the original gold
        count (collected gold stays in the list with ``active=False``).
        """
        return [
            (float(g.xpos), float(g.ypos), bool(getattr(g, "active", True)))
            for g in self.sim.entity_dic.get(2, [])
        ]

    def get_goal_positions_for_features(self) -> Dict[str, Any]:
        """Return pre-computed goal positions for reachability features (O(1) lookups).

        This method provides direct entity_dic access to avoid O(n) entity iteration
        in feature computation code. Uses existing helper methods that access entity_dic
        directly by type.

        Returns:
            Dictionary containing:
            - switch_pos: Exit switch position tuple (x, y)
            - exit_pos: Exit door position tuple (x, y)
            - switch_activated: Boolean, True if switch collected
            - locked_doors: List of locked door entities
            - locked_door_switches: List of locked door switch entities
            - mine_entities: Tuple of (toggle_mines, toggled_mines) for mine counting
        """
        return {
            "switch_pos": self.exit_switch_position(),
            "exit_pos": self.exit_door_position(),
            "switch_activated": self.exit_switch_activated(),
            "locked_doors": self.locked_doors(),
            "locked_door_switches": self.locked_door_switches(),
            "mine_entities": self.get_mine_entities(),  # O(1) entity_dic access
        }

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

        Returns 24 features (time_remaining is appended separately in base_environment.py):
            - Velocity (2): velocity_x, velocity_y
            - Movement state categories (4): ground, air, wall, special
            - Airborne status (1)
            - Buffers (4): jump_buffer, floor_buffer, wall_buffer, jump_duration_remaining
            - Surface contact (3): floor, wall, ceiling
            - Surface normals (3): floor_normal_x, floor_normal_y, wall_normal
            - Physics parameters (3): applied_gravity, applied_drag, applied_friction
            - Walled status (1)
            - Position (2): position_x, position_y (normalized)

        REMOVED from v1 (41 dims): velocity_magnitude, velocity_direction, floor_normal_strength,
        surface_slope, ceiling_normal, temporal_features (6), kinetic_energy, potential_energy,
        force_magnitude, energy_change_rate, contact_strength (2), surface_slope_angle,
        wall_interaction_strength (14 features removed)

        ADDED: jump_duration_remaining (MDP critical), position_x, position_y
        """
        ninja = self.sim.ninja
        from .constants.physics_constants import MAX_JUMP_DURATION
        from .gym_environment.constants import LEVEL_WIDTH, LEVEL_HEIGHT

        state = []

        # === Velocity (2 features) ===
        # 0-1. Direct velocity components (most information-dense features)
        velocity_x = np.clip(ninja.xspeed / MAX_HOR_SPEED, -1, 1)
        velocity_y = np.clip(ninja.yspeed / MAX_HOR_SPEED, -1, 1)
        state.extend([velocity_x, velocity_y])

        # === Movement State Categories (4 features) ===
        # 2-5. Movement state categories (binary classification of ninja state)
        ground_movement = (
            1.0 if ninja.state in [0, 1, 2] else -1.0
        )  # Immobile, Running, Ground sliding
        air_movement = 1.0 if ninja.state in [3, 4] else -1.0  # Jumping, Falling
        wall_interaction = 1.0 if ninja.state == 5 else -1.0  # Wall sliding
        special_states = (
            1.0 if ninja.state in [6, 7, 8, 9] else -1.0
        )  # Dead, Awaiting death, Celebrating, Disabled
        state.extend([ground_movement, air_movement, wall_interaction, special_states])

        # 6. Airborne status
        airborne_status = 1.0 if ninja.airborn else -1.0
        state.append(airborne_status)

        # === Buffer State (4 features) ===
        # 7-9. Buffer states (normalized to [-1, 1]).
        # Map (buffer + 1) over (window + 1) so INACTIVE (-1) -> -1.0 and every
        # ACTIVE value 0..window lands in (-1, 1] distinctly — with the previous
        # max(b, 0)/window form, "just armed" (b=0, ACTIVE) aliased "inactive"
        # (b=-1) at exactly -1.0, a one-frame ambiguity in a frame-perfect domain.
        buffer_window_size = 5.0
        jump_buffer_norm = (
            (max(ninja.jump_buffer, -1) + 1) / (buffer_window_size + 1)
        ) * 2 - 1
        floor_buffer_norm = (
            (max(ninja.floor_buffer, -1) + 1) / (buffer_window_size + 1)
        ) * 2 - 1
        wall_buffer_norm = (
            (max(ninja.wall_buffer, -1) + 1) / (buffer_window_size + 1)
        ) * 2 - 1
        state.extend([jump_buffer_norm, floor_buffer_norm, wall_buffer_norm])

        # 10. Jump duration remaining (MDP critical)
        # When jumping (state 3), track remaining frames until forced fall
        # Normalized to [0, 1]: 1.0 = full jump available, 0.0 = jump ending
        if ninja.state == 3:  # Jumping state
            jump_duration_remaining = (
                max(0, MAX_JUMP_DURATION - ninja.jump_duration) / MAX_JUMP_DURATION
            )
        else:
            jump_duration_remaining = 0.0
        state.append(jump_duration_remaining)

        # === Surface Contact Information (3 features) ===
        # 11-13. Contact states (binary). Wall contact gates on the live
        # ``walled`` flag: ``wall_count`` is never incremented during simulation
        # (tile wall contacts fold into ceiling_count in post_collision), so
        # gating on it left this feature permanently -1.
        floor_contact = (min(ninja.floor_count, 1) * 2) - 1  # Convert 0,1 to -1,1
        wall_contact = 1.0 if ninja.walled else -1.0
        ceiling_contact = (min(ninja.ceiling_count, 1) * 2) - 1
        state.extend([floor_contact, wall_contact, ceiling_contact])

        # === Surface Normals (3 features) ===
        # 14-15. Floor normal (full vector, already [-1, 1])
        floor_normal_x = ninja.floor_normalized_x
        floor_normal_y = ninja.floor_normalized_y
        state.extend([floor_normal_x, floor_normal_y])

        # 18. Wall normal direction (scalar, -1 or 1). Gate on ``walled`` (not
        # the never-incremented ``wall_count``): post_collision recomputes
        # ``wall_normal`` whenever walled is true, so it is always current here.
        if ninja.walled and hasattr(ninja, "wall_normal"):
            wall_normal = float(ninja.wall_normal)  # Already [-1, 1]
        else:
            wall_normal = 0.0
        state.append(wall_normal)

        # === Walled Status (1 feature) ===
        # 19. Walled status (boolean to -1/1)
        walled_status = 1.0 if ninja.walled else -1.0
        state.append(walled_status)

        # === Physics Parameters (3 features) ===
        # 20. Applied gravity (normalized between GRAVITY_JUMP and GRAVITY_FALL)
        gravity_range = GRAVITY_FALL - GRAVITY_JUMP
        if gravity_range > 1e-6:
            gravity_norm = (
                ninja.applied_gravity - GRAVITY_JUMP
            ) / gravity_range * 2 - 1  # [-1, 1]
        else:
            gravity_norm = 0.0
        state.append(gravity_norm)

        # 21. Applied drag (normalized between DRAG_SLOW and DRAG_REGULAR)
        drag_range = DRAG_REGULAR - DRAG_SLOW
        if drag_range > 1e-6:
            drag_norm = (ninja.applied_drag - DRAG_SLOW) / drag_range * 2 - 1  # [-1, 1]
        else:
            drag_norm = 0.0
        state.append(drag_norm)

        # 22. Applied friction (normalized between FRICTION_GROUND_SLOW and FRICTION_GROUND)
        friction_range = FRICTION_GROUND - FRICTION_GROUND_SLOW
        if friction_range > 1e-6:
            friction_norm = (
                ninja.applied_friction - FRICTION_GROUND_SLOW
            ) / friction_range * 2 - 1  # [-1, 1]
        else:
            friction_norm = 0.0
        state.append(friction_norm)

        # === Position (2 features) ===
        # 23-24. Position normalized to [0, 1]
        position_x = ninja.xpos / LEVEL_WIDTH
        position_y = ninja.ypos / LEVEL_HEIGHT
        state.extend([position_x, position_y])

        # Note: time_remaining (25th feature) is appended in base_environment.py
        # to create the final 24-dimensional game state vector

        return state
