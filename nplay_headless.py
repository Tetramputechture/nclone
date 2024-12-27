import pygame
import os
from typing import Optional
from nsim import Simulator
from nsim_renderer import NSimRenderer
from map import MapGenerator
from sim_config import SimConfig
import numpy as np
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

    def __init__(self, render_mode: str = 'rgb_array'):
        """
        Initialize the simulation and renderer, as well as the headless pygame
        interface and display.
        """
        self.render_mode = render_mode

        self.sim = Simulator(SimConfig())
        self.sim_renderer = NSimRenderer(self.sim, render_mode)
        self.current_map_data = None

        # Create a new map generator
        self.map_gen = MapGenerator()
        self.clock = pygame.time.Clock()

        # init pygame
        pygame.init()

        if self.render_mode == 'rgb_array':
            # set pygame env to headless
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            print('Setting up pygame display')
            pygame.display.set_mode((SRCWIDTH, SRCHEIGHT))

        # init display
        pygame.display.init()

    def load_map(self, map_path: str):
        """
        Load a map from a file.
        """
        with open(map_path, "rb") as map_file:
            mapdata = [int(b) for b in map_file.read()]
        self.sim.load(mapdata)
        self.current_map_data = mapdata

    def load_random_map(self, seed: Optional[int] = None, map_type: Optional[str] = "SIMPLE_HORIZONTAL_NO_BACKTRACK"):
        """
        Load a random map from the map_data folder.
        """
        # Generate a random level (optionally with a seed for reproducibility)
        self.map_gen.generate_random_map(map_type, seed=seed)

        # Get the map data
        map_data = self.map_gen.generate()
        self.sim.load(map_data)
        self.current_map_data = map_data

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
            print('NO')
            pygame.display.update()
            pygame.event.pump()
            self.clock.tick(60)

    def render(self):
        """
        Render the current frame to a NumPy array.
        """
        init = self.sim.frame <= 1
        surface = self.sim_renderer.draw(init)
        # if self.render_mode == 'rgb_array':
        imgdata = pygame.surfarray.array3d(surface)
        imgdata = imgdata.swapaxes(0, 1)
        return imgdata
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

        # Add ninja state (13 values)
        ninja_state = self.get_ninja_state()
        state.extend(ninja_state)

        # Add entity states with fixed size per entity type
        entity_states = self.get_entity_states(only_exit_and_switch)
        state.extend(entity_states)

        # Add environment geometry (fixed size)
        geometry_state = self._get_geometry_state()
        state.extend(geometry_state)

        return np.array(state, dtype=np.float32)

    def get_ninja_state(self):
        """Get ninja state information as a 12-element list of floats, all normalized between 0 and 1.

        Returns:
            numpy.ndarray: A 1D array containing:
            - X Position normalized
            - Y position normalized
            - X speed normalized
            - Y speed normalized
            - Airborn boolean
            - Walled boolean
            - Jump duration normalized
            - Facing normalized
            - Tilt angle normalized
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
            (float(ninja.facing) + 1) / 2,  # Facing (-1 or 1) normalized
            (ninja.tilt + math.pi) / (2 * math.pi),  # Tilt angle normalized
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

            if only_one_exit_and_switch:
                max_count = 1

            # Add count of this entity type (normalize by max count for this type)
            # We only need to do this if we are not only getting the exit and switch
            if not only_one_exit_and_switch:
                state.append(float(len(entities)) / max_count)

            # Process each entity up to max_count
            for entity_idx in range(max_count):
                # If we have an actual entity at this index
                if entity_idx < len(entities):
                    entity = entities[entity_idx]
                    entity_state = entity.get_state(
                        minimal_state=only_one_exit_and_switch)

                    # Assert that all entity states are between 0 and 1. If not
                    # print an informative error message containing the entity type,
                    # index, and state.
                    if not all(0 <= state <= 1 for state in entity_state):
                        print(
                            f"Entity type {entity_type} index {entity_idx} state {entity_state} is out of bounds")
                        raise ValueError(
                            f"Entity type {entity_type} index {entity_idx} state {entity_state} is out of bounds")

                    # Pad remaining attributes to reach MAX_ATTRIBUTES if needed
                    if not only_one_exit_and_switch:
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
