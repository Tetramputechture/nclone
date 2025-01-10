import pygame
import os
import random
from typing import Optional
from nsim import Simulator
from nsim_renderer import NSimRenderer
from map_generation.map_generator import generate_map
from sim_config import SimConfig
import numpy as np
from typing import List
import math
from entities import (EntityToggleMine, EntityGold, EntityExit, EntityExitSwitch,
                      EntityDoorRegular, EntityDoorLocked, EntityDoorTrap, EntityLaunchPad,
                      EntityOneWayPlatform, EntityDroneZap, EntityBounceBlock, EntityThwump,
                      EntityBoostPad, EntityDeathBall, EntityMiniDrone, EntityShoveThwump)

SRCWIDTH = 1056
SRCHEIGHT = 600


class NPlayHeadless:
    """
    This class is used to run the simulation in headless mode,
    while supporting rendering a frame to a NumPy array. Has manual
    control over the simulation and rendering, and is intended to be
    used for training machine learning agents.

    Has a simple API for interacting with the simulation (moving horizontally
    or jumping, loading a map, resetting and ticking the simulation).
    """

    def __init__(self,
                 render_mode: str = 'rgb_array',
                 enable_animation: bool = False,
                 enable_logging: bool = False,
                 enable_debug_overlay: bool = False,
                 seed: Optional[int] = None):
        """
        Initialize the simulation and renderer, as well as the headless pygame
        interface and display.
        """
        self.render_mode = render_mode

        self.sim = Simulator(
            SimConfig(enable_anim=enable_animation, log_data=enable_logging))
        self.sim_renderer = NSimRenderer(
            self.sim, render_mode, enable_debug_overlay)
        self.current_map_data = None
        self.clock = pygame.time.Clock()

        # init pygame
        pygame.init()
        pygame.display.init()
        if self.render_mode == 'rgb_array':
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            print('Setting up pygame display')
            pygame.display.set_mode((SRCWIDTH, SRCHEIGHT))

        # Pre-allocate buffer for surface to array conversion
        self._render_buffer = np.empty(
            (SRCWIDTH, SRCHEIGHT, 3), dtype=np.uint8)

        self.enable_debug_overlay = enable_debug_overlay
        self.seed = seed
        self.rng = random.Random(seed)

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

    def load_random_map(self, map_type: Optional[str] = "SIMPLE_HORIZONTAL_NO_BACKTRACK"):
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
        base_map_path = os.path.join(
            os.path.dirname(__file__), 'maps', 'official')
        map_file = self.rng.choice(os.listdir(base_map_path))
        map_path = os.path.join(base_map_path, map_file)
        self.load_map(map_path)
        return map_file

    def reset(self):
        """ 
        Reset the simulation to the initial state.
        """
        self.sim.reset()

    def tick(self, horizontal_input: int, jump_input: int):
        """
        Tick the simulation with the given horizontal and jump inputs.
        """
        self.sim.tick(horizontal_input, jump_input)
        if self.render_mode == 'human':
            self.clock.tick(120)

    def render(self, debug_info: Optional[dict] = None):
        """
        Render the current frame to a NumPy array.
        Uses a pre-allocated buffer and direct pixel access for better performance.

        Args:
            debug_info: A dictionary containing debug information to be displayed on the screen.
        """
        init = self.sim.frame <= 1
        surface = self.sim_renderer.draw(init, debug_info)

        # if self.render_mode == 'rgb_array':
        # Use our pre-allocated buffer
        pygame.pixelcopy.surface_to_array(
            self._render_buffer, surface)
        return self._render_buffer  # Already in correct shape (H, W, 3)
        # else:
        #     return surface

    def render_collision_map(self):
        """Render the collision map to a NumPy array."""
        init = self.sim.frame <= 1
        surface = self.sim_renderer.draw_collision_map(init)
        imgdata = pygame.surfarray.array3d(surface)
        imgdata = imgdata.swapaxes(0, 1)
        return imgdata

    def ninja_has_won(self):
        return self.sim.ninja.has_won()

    def ninja_has_died(self):
        return self.sim.ninja.has_died()

    def ninja_position(self):
        return self.sim.ninja.xpos, self.sim.ninja.ypos

    def ninja_velocity(self):
        return self.sim.ninja.xspeed, self.sim.ninja.yspeed

    def ninja_is_in_air(self):
        return self.sim.ninja.airborn

    def ninja_is_walled(self):
        return self.sim.ninja.walled

    def _sim_exit_switch(self):
        # We want to get the last entry of self.sim.entity_dic[3];
        # this is the exit switch.
        # First, check if the exit switch exists.
        if len(self.sim.entity_dic[3]) == 0:
            return None

        return self.sim.entity_dic[3][-1]

    def exit_switch_activated(self):
        return not self._sim_exit_switch().active

    def exit_switch_position(self):
        return self._sim_exit_switch().xpos, self._sim_exit_switch().ypos

    def _sim_exit_door(self):
        # We want to get the .parent attribute of the exit switch.
        exit_switch = self._sim_exit_switch()
        if exit_switch is None:
            return None
        return exit_switch.parent

    def exit_door_position(self):
        return self._sim_exit_door().xpos, self._sim_exit_door().ypos

    def mines(self):
        # We want to return a list of all mines in the simulation with state == 0 (toggled)
        # of type 1.
        return [entity for entity in self.sim.entity_dic[1] if entity.active and entity.state == 0]

    def get_tile_data(self):
        """Get tile data from simulator."""
        return self.sim.tile_dic

    def get_segment_data(self):
        """Get segment data from simulator."""
        return self.sim.segment_dic

    def get_grid_edges(self):
        """Get horizontal and vertical grid edges."""
        return {
            'horizontal': self.sim.hor_grid_edge_dic,
            'vertical': self.sim.ver_grid_edge_dic
        }

    def get_segment_edges(self):
        """Get horizontal and vertical segment edges."""
        return {
            'horizontal': self.sim.hor_segment_dic,
            'vertical': self.sim.ver_segment_dic
        }

    def exit(self):
        pygame.quit()

    def get_state_vector(self, only_exit_and_switch: bool = False):
        """
        Get a complete state representation of the game environment as a vector of float values.
        This includes ninja state, entity states, and environment geometry.

        Returns:
            numpy.ndarray: A 1D array containing all state information, with fixed length.
            All values are normalized between 0 and 1.
        """
        state = []

        # Add ninja state (10 values)
        ninja_state = self.get_ninja_state()
        state.extend(ninja_state)

        # Add entity states with fixed size per entity type
        entity_states = self.get_entity_states(only_exit_and_switch)
        state.extend(entity_states)

        # Add environment geometry (fixed size)
        # Lets avoid this for now
        # geometry_state = self._get_geometry_state()
        # state.extend(geometry_state)

        return np.array(state, dtype=np.float32)

    def get_gold_collected(self):
        """Returns the total gold collected by the ninja."""
        return self.sim.ninja.gold_collected

    def get_total_gold_available(self):
        """Returns count of all gold (entity type 2) in the map."""
        return sum(1 for entity in self.sim.entity_dic[2])

    def get_ninja_state(self):
        """Get ninja state information as a 10-element list of floats, all normalized between 0 and 1.

        Returns:
            numpy.ndarray: A 1D array containing:
            - X Position normalized
            - Y position normalized
            - X speed normalized
            - Y speed normalized
            - Airborn boolean
            - Walled boolean
            - Jump duration normalized
            - Applied gravity normalized
            - Applied drag normalized
            - Applied friction normalized
        """
        ninja = self.sim.ninja

        state = [
            ninja.xpos / SRCWIDTH,  # Position normalized
            ninja.ypos / SRCHEIGHT,
            (ninja.xspeed / ninja.MAX_HOR_SPEED + 1) / \
            2,  # Speed normalized to [0,1]
            (ninja.yspeed / ninja.MAX_HOR_SPEED + 1) / 2,
            float(ninja.airborn),  # Boolean already 0 or 1
            float(ninja.walled),
            ninja.jump_duration / ninja.MAX_JUMP_DURATION,  # Already normalized
            # Physics parameters normalized based on their typical ranges
            (ninja.applied_gravity - ninja.GRAVITY_JUMP) / \
            (ninja.GRAVITY_FALL - ninja.GRAVITY_JUMP),
            (ninja.applied_drag - ninja.DRAG_SLOW) / \
            (ninja.DRAG_REGULAR - ninja.DRAG_SLOW),
            (ninja.applied_friction - ninja.FRICTION_WALL) / \
            (ninja.FRICTION_GROUND - ninja.FRICTION_WALL)
        ]
        return state

    def get_entity_states(self, only_one_exit_and_switch: bool = False):
        """Get all entity states as a list of floats with fixed length, all normalized between 0 and 1.

        Args:
            only_exit_and_switch: If True, only include exit and switch entities. This useful for
            training a model on simple levels without entities. Exit is entity type 3, and switch is
            entity type 4.
        """
        state = []

        # If we are only interested in the exit and switch, we can reduce the number of attributes
        # And only return:
        # [switch_active, exit_active]
        if only_one_exit_and_switch:
            return [float(self._sim_exit_switch().active), float(self._sim_exit_door().active)]

        # Maximum number of attributes per entity (padding with zeros if entity has fewer attributes)
        MAX_ATTRIBUTES = 8

        # Entity type to max count mapping based on MAX_COUNT_PER_LEVEL constants
        MAX_COUNTS = {
            EntityToggleMine.ENTITY_TYPE: EntityToggleMine.MAX_COUNT_PER_LEVEL,
            EntityGold.ENTITY_TYPE: EntityGold.MAX_COUNT_PER_LEVEL,
            EntityExit.ENTITY_TYPE: EntityExit.MAX_COUNT_PER_LEVEL,
            EntityDoorRegular.ENTITY_TYPE: EntityDoorRegular.MAX_COUNT_PER_LEVEL,
            EntityDoorLocked.ENTITY_TYPE: EntityDoorLocked.MAX_COUNT_PER_LEVEL,
            EntityDoorTrap.ENTITY_TYPE: EntityDoorTrap.MAX_COUNT_PER_LEVEL,
            EntityLaunchPad.ENTITY_TYPE: EntityLaunchPad.MAX_COUNT_PER_LEVEL,
            EntityOneWayPlatform.ENTITY_TYPE: EntityOneWayPlatform.MAX_COUNT_PER_LEVEL,
            EntityDroneZap.ENTITY_TYPE: EntityDroneZap.MAX_COUNT_PER_LEVEL,
            EntityBounceBlock.ENTITY_TYPE: EntityBounceBlock.MAX_COUNT_PER_LEVEL,
            EntityThwump.ENTITY_TYPE: EntityThwump.MAX_COUNT_PER_LEVEL,
            EntityBoostPad.ENTITY_TYPE: EntityBoostPad.MAX_COUNT_PER_LEVEL,
            EntityDeathBall.ENTITY_TYPE: EntityDeathBall.MAX_COUNT_PER_LEVEL,
            EntityMiniDrone.ENTITY_TYPE: EntityMiniDrone.MAX_COUNT_PER_LEVEL,
            EntityShoveThwump.ENTITY_TYPE: EntityShoveThwump.MAX_COUNT_PER_LEVEL
        }

        exit_entity_type = EntityExit.ENTITY_TYPE
        switch_entity_type = EntityExitSwitch.ENTITY_TYPE

        entity_types = [
            exit_entity_type, switch_entity_type] if only_one_exit_and_switch else range(1, 29)

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

                    # Assert that all entity states are between 0 and 1. If not
                    # print an informative error message containing the entity type,
                    # index, and state.
                    if not all(0 <= state <= 1 for state in entity_state):
                        print(
                            f"Entity type {entity_type} index {entity_idx} state {entity_state} is out of bounds")
                        raise ValueError(
                            f"Entity type {entity_type} index {entity_idx} state {entity_state} is out of bounds")

                    while len(entity_state) < MAX_ATTRIBUTES:
                        entity_state.append(0.0)

                    state.extend(entity_state)
                else:
                    # Add padding for non-existent entity
                    state.extend([0.0] * MAX_ATTRIBUTES)

        return state

    def _get_geometry_state(self):
        """Get environment geometry state as a list of floats, all normalized between 0 and 1."""
        state = []

        # Add tile data - normalize by max tile ID (37)
        # Fixed size: 44 * 25 = 1100 tiles
        for x in range(44):
            for y in range(25):
                tile_id = self.sim.tile_dic.get((x, y), 0)
                state.append(float(tile_id) / 37.0)

        # Grid edges and segments can be -1, 0, or 1, so normalize to [0,1]
        # Add horizontal grid edges (88 * 51 = 4488 edges)
        for x in range(88):
            for y in range(51):
                edge = self.sim.hor_grid_edge_dic.get((x, y), 0)
                # Normalize from [-1,1] to [0,1]
                state.append((float(edge) + 1) / 2)

        # Add vertical grid edges (89 * 50 = 4450 edges)
        for x in range(89):
            for y in range(50):
                edge = self.sim.ver_grid_edge_dic.get((x, y), 0)
                # Normalize from [-1,1] to [0,1]
                state.append((float(edge) + 1) / 2)

        # Add horizontal segments (88 * 51 = 4488 segments)
        for x in range(88):
            for y in range(51):
                segment = self.sim.hor_segment_dic.get((x, y), 0)
                # Normalize from [-1,1] to [0,1]
                state.append((float(segment) + 1) / 2)

        # Add vertical segments (89 * 50 = 4450 segments)
        for x in range(89):
            for y in range(50):
                segment = self.sim.ver_segment_dic.get((x, y), 0)
                # Normalize from [-1,1] to [0,1]
                state.append((float(segment) + 1) / 2)

        # Assert that all states are between 0 and 1
        if not all(0 <= s <= 1 for s in state):
            invalid_states = [(i, s)
                              for i, s in enumerate(state) if not 0 <= s <= 1]
            print(f"Invalid states found at indices: {invalid_states}")
            raise ValueError("Some geometry states are out of bounds [0,1]")

        return state
