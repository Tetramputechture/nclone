import argparse
import array
import math
from itertools import product
import os.path
import struct
import copy
import random
from ninja import Ninja
from sim_config import SimConfig
from entities import *


class Simulator:
    """Main class that handles ninjas, entities and tile geometry for simulation."""

    # This is a dictionary mapping every tile id to the grid edges it contains.
    # The first 6 values represent horizontal half-tile edges, from left to right then top to bottom.
    # The last 6 values represent vertical half-tile edges, from top to bottom then left to right.
    # 1 if there is a grid edge, 0 otherwise.
    TILE_GRID_EDGE_MAP = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],  # 0-1: Empty and full tiles
                          # 2-5: Half tiles
                          2: [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], 3: [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
                          4: [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1], 5: [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
                          # 6-9: 45 degree slopes
                          6: [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], 7: [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                          8: [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], 9: [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
                          # 10-13: Quarter moons
                          10: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], 11: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                          12: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], 13: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                          # 14-17: Quarter pipes
                          14: [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], 15: [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                          16: [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], 17: [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
                          # 18-21: Short mild slopes
                          18: [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], 19: [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                          20: [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1], 21: [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                          # 22-25: Raised mild slopes
                          22: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], 23: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                          24: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], 25: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                          # 26-29: Short steep slopes
                          26: [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0], 27: [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
                          28: [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1], 29: [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
                          # 30-33: Raised steep slopes
                          30: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], 31: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                          32: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], 33: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                          # 34-37: Glitched tiles
                          34: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 35: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          36: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 37: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]}

    # This is a dictionary mapping every tile id to the orthogonal linear segments it contains,
    # same order as grid edges.
    # 0 if no segment, -1 if normal facing left or up, 1 if normal right or down.
    TILE_SEGMENT_ORTHO_MAP = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [-1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1],  # 0-1: Empty and full tiles
                              # 2-5: Half tiles
                              2: [-1, -1, 1, 1, 0, 0, -1, 0, 0, 0, 1, 0], 3: [0, -1, 0, 0, 0, 1, 0, 0, -1, -1, 1, 1],
                              4: [0, 0, -1, -1, 1, 1, 0, -1, 0, 0, 0, 1], 5: [-1, 0, 0, 0, 1, 0, -1, -1, 1, 1, 0, 0],
                              # 6-9: 45 degree slopes
                              6: [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0], 7: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                              8: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], 9: [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                              # 10-13: Quarter moons
                              10: [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0], 11: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                              12: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], 13: [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                              # 14-17: Quarter pipes
                              14: [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0], 15: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                              16: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], 17: [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                              # 18-21: Short mild slopes
                              18: [-1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], 19: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              20: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], 21: [0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 0],
                              # 22-25: Raised mild slopes
                              22: [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 1, 0], 23: [-1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1],
                              24: [0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 1, 1], 25: [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 1],
                              # 26-29: Short steep slopes
                              26: [-1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0], 27: [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                              28: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1], 29: [0, 0, 0, 0, 1, 0, -1, -1, 0, 0, 0, 0],
                              # 30-33: Raised steep slopes
                              30: [-1, -1, 0, 0, 1, 0, -1, -1, 0, 0, 0, 0], 31: [-1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
                              32: [0, -1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], 33: [-1, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                              # 34-37: Glitched tiles
                              34: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 35: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                              36: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 37: [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0]}

    # This is a dictionary mapping every tile id to the diagonal linear segment it contains.
    # Segments are defined by two sets of point that need to be added to the position inside the grid.
    TILE_SEGMENT_DIAG_MAP = {6: ((0, 24), (24, 0)), 7: ((0, 0), (24, 24)),
                             8: ((24, 0), (0, 24)), 9: ((24, 24), (0, 0)),
                             18: ((0, 12), (24, 0)), 19: ((0, 0), (24, 12)),
                             20: ((24, 12), (0, 24)), 21: ((24, 24), (0, 12)),
                             22: ((0, 24), (24, 12)), 23: ((0, 12), (24, 24)),
                             24: ((24, 0), (0, 12)), 25: ((24, 12), (0, 0)),
                             26: ((0, 24), (12, 0)), 27: ((12, 0), (24, 24)),
                             28: ((24, 0), (12, 24)), 29: ((12, 24), (0, 0)),
                             30: ((12, 24), (24, 0)), 31: ((0, 0), (12, 24)),
                             32: ((12, 0), (0, 24)), 33: ((24, 24), (12, 0))}

    # This is a dictionary mapping every tile id to the circular segment it contains.
    # Segments defined by their center point and the quadrant.
    TILE_SEGMENT_CIRCULAR_MAP = {10: ((0, 0), (1, 1), True), 11: ((24, 0), (-1, 1), True),
                                 12: ((24, 24), (-1, -1), True), 13: ((0, 24), (1, -1), True),
                                 14: ((24, 24), (-1, -1), False), 15: ((0, 24), (1, -1), False),
                                 16: ((0, 0), (1, 1), False), 17: ((24, 0), (-1, 1), False)}

    def __init__(self, sc: SimConfig):
        self.frame = 0
        self.collisionlog = []
        self.gold_collected = 0

        self.sim_config = sc
        self.ninja = None
        self.tile_dic = {}
        self.segment_dic = {}
        self.grid_entity = {}
        self.entity_dic = {}
        self.hor_grid_edge_dic = {}
        self.ver_grid_edge_dic = {}
        self.hor_segment_dic = {}
        self.ver_segment_dic = {}
        self.map_data = None

    def load(self, map_data):
        """From the given map data, initiate the level geometry, the entities and the ninja."""
        self.map_data = map_data
        self.reset_map_tile_data()
        self.load_map_tiles()
        self.reset()

    def load_from_created(self, created_map):
        """Load a map that was manually constructed using the Map class."""
        self.load(created_map.map_data())

    def reset(self):
        """Reset the simulation to the initial state. Keeps the current map tiles, and resets the ninja,
        entities and the collision log."""
        self.frame = 0
        self.collisionlog = []
        self.gold_collected = 0
        self.ninja = None
        self.reset_map_entity_data()
        self.load_map_entities()

    def reset_map_entity_data(self):
        """Reset the map entity data. This is used when a new map is loaded or when the map is reset."""
        self.grid_entity = {}
        for x in range(44):
            for y in range(25):
                self.grid_entity[(x, y)] = []
        self.entity_dic = {i: [] for i in range(1, 29)}

    def reset_map_tile_data(self):
        """Reset the map cell data. This is used when a new map is loaded."""
        self.segment_dic = {}
        for x in range(45):
            for y in range(26):
                self.segment_dic[(x, y)] = []
        for x in range(88):
            for y in range(51):
                self.hor_grid_edge_dic[(x, y)] = 1 if y in (0, 50) else 0
        for x in range(89):
            for y in range(50):
                self.ver_grid_edge_dic[(x, y)] = 1 if x in (0, 88) else 0
        for x in range(88):
            for y in range(51):
                self.hor_segment_dic[(x, y)] = 0
        for x in range(89):
            for y in range(50):
                self.ver_segment_dic[(x, y)] = 0

    def load_map_tiles(self):
        """Load the map tiles into the simulation. These shouldn't change during the simulation,
        only when a new map is loaded."""
        # extract tile data from map data
        tile_data = self.map_data[184:1150]

        # map each tile to its cell
        for x in range(42):
            for y in range(23):
                self.tile_dic[(x+1, y+1)] = tile_data[x + y*42]

        # Set our outer edges to tile type 1 (full tile)
        for x in range(44):
            self.tile_dic[(x, 0)] = 1
            self.tile_dic[(x, 24)] = 1
        for y in range(25):
            self.tile_dic[(0, y)] = 1
            self.tile_dic[(43, y)] = 1

        # This loops makes the inventory of grid edges and orthogonal linear segments,
        # and initiates non-orthogonal linear segments and circular segments.
        for coord, tile_id in self.tile_dic.items():
            xcoord, ycoord = coord
            # Assign every grid edge and orthogonal linear segment to the dictionaries.
            if tile_id in self.TILE_GRID_EDGE_MAP and tile_id in self.TILE_SEGMENT_ORTHO_MAP:
                grid_edge_list = self.TILE_GRID_EDGE_MAP[tile_id]
                segment_ortho_list = self.TILE_SEGMENT_ORTHO_MAP[tile_id]
                for y in range(3):
                    for x in range(2):
                        self.hor_grid_edge_dic[(2*xcoord + x, 2*ycoord + y)] = (
                            (self.hor_grid_edge_dic[(2*xcoord + x, 2*ycoord + y)] + grid_edge_list[2*y + x]) % 2)
                        self.hor_segment_dic[(
                            2*xcoord + x, 2*ycoord + y)] += segment_ortho_list[2*y + x]
                for x in range(3):
                    for y in range(2):
                        self.ver_grid_edge_dic[(2*xcoord + x, 2*ycoord + y)] = (
                            (self.ver_grid_edge_dic[(2*xcoord + x, 2*ycoord + y)] + grid_edge_list[2*x + y + 6]) % 2)
                        self.ver_segment_dic[(
                            2*xcoord + x, 2*ycoord + y)] += segment_ortho_list[2*x + y + 6]

            # Initiate non-orthogonal linear and circular segments.
            xtl = xcoord * 24
            ytl = ycoord * 24
            if tile_id in self.TILE_SEGMENT_DIAG_MAP:
                ((x1, y1), (x2, y2)) = self.TILE_SEGMENT_DIAG_MAP[tile_id]
                self.segment_dic[coord].append(
                    GridSegmentLinear((xtl+x1, ytl+y1), (xtl+x2, ytl+y2)))
            if tile_id in self.TILE_SEGMENT_CIRCULAR_MAP:
                ((x, y), quadrant,
                 convex) = self.TILE_SEGMENT_CIRCULAR_MAP[tile_id]
                self.segment_dic[coord].append(
                    GridSegmentCircular((xtl+x, ytl+y), quadrant, convex))

        # Initiate segments from the dictionaries of orthogonal linear segments.
        # Note that two segments of the same position but opposite orientation cancel each other,
        # and no segment is initiated.
        for coord, state in self.hor_segment_dic.items():
            if state:
                xcoord, ycoord = coord
                cell = (math.floor(xcoord/2),
                        math.floor((ycoord - 0.1*state) / 2))
                point1 = (12*xcoord, 12*ycoord)
                point2 = (12*xcoord+12, 12*ycoord)
                if state == -1:
                    point1, point2 = point2, point1
                self.segment_dic[cell].append(
                    GridSegmentLinear(point1, point2))
        for coord, state in self.ver_segment_dic.items():
            if state:
                xcoord, ycoord = coord
                cell = (math.floor((xcoord - 0.1*state) / 2),
                        math.floor(ycoord/2))
                point1 = (12*xcoord, 12*ycoord+12)
                point2 = (12*xcoord, 12*ycoord)
                if state == -1:
                    point1, point2 = point2, point1
                self.segment_dic[cell].append(
                    GridSegmentLinear(point1, point2))

    def load_map_entities(self):
        """Load the map entities into the simulation. These should change during the simulation,
        and are reset when a new map is loaded."""

        # initiate player 1 instance of Ninja at spawn coordinates
        self.ninja = Ninja(self, ninja_anim_mode=(
            self.sim_config.enable_anim and not self.sim_config.basic_sim))

        # initiate each entity (other than ninjas)
        index = 1230
        exit_door_count = self.map_data[1156]
        Entity.entity_counts = [0] * 40
        while (index < len(self.map_data)):
            entity_type = self.map_data[index]
            xcoord = self.map_data[index+1]
            ycoord = self.map_data[index+2]
            orientation = self.map_data[index+3]
            mode = self.map_data[index+4]
            if entity_type == 1:
                entity = EntityToggleMine(entity_type, self, xcoord, ycoord, 0)
            elif entity_type == 2:
                entity = EntityGold(entity_type, self, xcoord, ycoord)
            elif entity_type == 3:
                parent = EntityExit(entity_type, self, xcoord, ycoord)
                self.entity_dic[entity_type].append(parent)
                child_xcoord = self.map_data[index + 5*exit_door_count + 1]
                child_ycoord = self.map_data[index + 5*exit_door_count + 2]
                entity = EntityExitSwitch(
                    4, self, child_xcoord, child_ycoord, parent)
            elif entity_type == 5:
                entity = EntityDoorRegular(
                    entity_type, self, xcoord, ycoord, orientation, xcoord, ycoord)
            elif entity_type == 6:
                switch_xcoord = self.map_data[index + 6]
                switch_ycoord = self.map_data[index + 7]
                entity = EntityDoorLocked(
                    entity_type, self, xcoord, ycoord, orientation, switch_xcoord, switch_ycoord)
            elif entity_type == 8:
                switch_xcoord = self.map_data[index + 6]
                switch_ycoord = self.map_data[index + 7]
                entity = EntityDoorTrap(
                    entity_type, self, xcoord, ycoord, orientation, switch_xcoord, switch_ycoord)
            elif entity_type == 10:
                entity = EntityLaunchPad(
                    entity_type, self, xcoord, ycoord, orientation)
            elif entity_type == 11:
                entity = EntityOneWayPlatform(
                    entity_type, self, xcoord, ycoord, orientation)
            elif entity_type == 14 and not self.sim_config.basic_sim:
                entity = EntityDroneZap(
                    entity_type, self, xcoord, ycoord, orientation, mode)
            # elif type == 15 and not ARGUMENTS.basic_sim:
            #    entity = EntityDroneChaser(type, self, xcoord, ycoord, orientation, mode)
            elif entity_type == 17:
                entity = EntityBounceBlock(entity_type, self, xcoord, ycoord)
            elif entity_type == 20:
                entity = EntityThwump(
                    entity_type, self, xcoord, ycoord, orientation)
            elif entity_type == 21:
                entity = EntityToggleMine(entity_type, self, xcoord, ycoord, 1)
            # elif entity_type == 23 and not ARGUMENTS.basic_sim:
            #    entity = EntityLaser(entity_type, self, xcoord, ycoord, orientation, mode)
            elif entity_type == 24:
                entity = EntityBoostPad(entity_type, self, xcoord, ycoord)
            elif entity_type == 25 and not self.sim_config.basic_sim:
                entity = EntityDeathBall(entity_type, self, xcoord, ycoord)
            elif entity_type == 26 and not self.sim_config.basic_sim:
                entity = EntityMiniDrone(
                    entity_type, self, xcoord, ycoord, orientation, mode)
            elif entity_type == 28:
                entity = EntityShoveThwump(entity_type, self, xcoord, ycoord)
            else:
                entity = None
            if entity:
                self.entity_dic[entity_type].append(entity)
                self.grid_entity[entity.cell].append(entity)
            index += 5

        for entity_list in self.entity_dic.values():
            for entity in entity_list:
                entity.log_position()

    def tick(self, hor_input, jump_input):
        """Gets called every frame to update the whole physics simulation."""
        # Increment the current frame
        self.frame += 1

        # Store inputs as ninja variables
        self.ninja.hor_input = hor_input
        self.ninja.jump_input = jump_input

        # Cache active entities to avoid repeated filtering
        active_movable_entities = []
        active_thinkable_entities = []

        # Single pass to categorize entities
        for entity_list in self.entity_dic.values():
            for entity in entity_list:
                if not entity.active:
                    continue
                if entity.is_movable:
                    active_movable_entities.append(entity)
                if entity.is_thinkable:
                    active_thinkable_entities.append(entity)

        # Move all movable entities
        for entity in active_movable_entities:
            entity.move()

        # Make all thinkable entities think
        for entity in active_thinkable_entities:
            entity.think()

        if self.ninja.state != 9:
            # if dead, apply physics to ragdoll instead.
            ninja = self.ninja if self.ninja.state != 6 else self.ninja.ragdoll
            ninja.integrate()  # Do preliminary speed and position updates.
            ninja.pre_collision()  # Do pre collision calculations.

            # Cache collision results
            for _ in range(4):
                # Handle PHYSICAL collisions with entities.
                ninja.collide_vs_objects()
                # Handle physical collisions with tiles.
                ninja.collide_vs_tiles()

            ninja.post_collision()  # Do post collision calculations.
            self.ninja.think()  # Make ninja think
            if self.sim_config.enable_anim:
                self.ninja.update_graphics()  # Update limbs of ninja

        if self.ninja.state == 6 and self.sim_config.enable_anim:  # Placeholder because no ragdoll!
            self.ninja.anim_frame = 105
            self.ninja.anim_state = 7
            self.ninja.calc_ninja_position()

        if self.sim_config.log_data:
            # Update all the logs for debugging purposes and for tracing the route.
            self.ninja.log()

            # Batch entity position logging
            for entity in active_movable_entities:
                entity.log_position()

        # Clear physics caches periodically
        if self.frame % 100 == 0:  # Clear caches every 100 frames
            from physics import clear_caches
            clear_caches()
