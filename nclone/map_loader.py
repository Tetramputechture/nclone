import math

from .ninja import Ninja
# Import base classes from entities.py module
from .entities import GridSegmentLinear, GridSegmentCircular, Entity
# Import entity classes from entity_classes package
from .entity_classes.entity_toggle_mine import EntityToggleMine
from .entity_classes.entity_gold import EntityGold
from .entity_classes.entity_exit import EntityExit
from .entity_classes.entity_exit_switch import EntityExitSwitch
from .entity_classes.entity_door_regular import EntityDoorRegular
from .entity_classes.entity_door_locked import EntityDoorLocked
from .entity_classes.entity_door_trap import EntityDoorTrap
from .entity_classes.entity_launch_pad import EntityLaunchPad
from .entity_classes.entity_one_way_platform import EntityOneWayPlatform
from .entity_classes.entity_drone_zap import EntityDroneZap
from .entity_classes.entity_bounce_block import EntityBounceBlock
from .entity_classes.entity_thwump import EntityThwump
from .entity_classes.entity_boost_pad import EntityBoostPad
from .entity_classes.entity_death_ball import EntityDeathBall
from .entity_classes.entity_mini_drone import EntityMiniDrone
from .entity_classes.entity_shove_thwump import EntityShoveThwump
from .tile_definitions import (
    TILE_GRID_EDGE_MAP, TILE_SEGMENT_ORTHO_MAP, TILE_SEGMENT_DIAG_MAP, TILE_SEGMENT_CIRCULAR_MAP
)


class MapLoader:
    """Handles loading of map tiles and entities for the simulation."""

    def __init__(self, simulator):
        """Initializes the MapLoader with a reference to the main simulator."""
        self.sim = simulator

    def load_map_tiles(self):
        """Load the map tiles into the simulation. These shouldn't change during the simulation,
        only when a new map is loaded."""
        # extract tile data from map data
        tile_data = self.sim.map_data[184:1150]

        # map each tile to its cell
        for x_coord_tile in range(42):
            for y_coord_tile in range(23):
                self.sim.tile_dic[(x_coord_tile + 1, y_coord_tile + 1)] = tile_data[x_coord_tile + y_coord_tile * 42]

        # Set our outer edges to tile type 1 (full tile)
        for x_coord_tile in range(44):
            self.sim.tile_dic[(x_coord_tile, 0)] = 1
            self.sim.tile_dic[(x_coord_tile, 24)] = 1
        for y_coord_tile in range(25):
            self.sim.tile_dic[(0, y_coord_tile)] = 1
            self.sim.tile_dic[(43, y_coord_tile)] = 1

        # This loops makes the inventory of grid edges and orthogonal linear segments,
        # and initiates non-orthogonal linear segments and circular segments.
        for coord, tile_id in self.sim.tile_dic.items():
            xcoord, ycoord = coord
            # Assign every grid edge and orthogonal linear segment to the dictionaries.
            if tile_id in TILE_GRID_EDGE_MAP and tile_id in TILE_SEGMENT_ORTHO_MAP:
                grid_edge_list = TILE_GRID_EDGE_MAP[tile_id]
                segment_ortho_list = TILE_SEGMENT_ORTHO_MAP[tile_id]
                for y_loop_idx in range(3):
                    for x_loop_idx in range(2):
                        self.sim.hor_grid_edge_dic[(2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)] = \
                            (self.sim.hor_grid_edge_dic[(2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)] + grid_edge_list[2 * y_loop_idx + x_loop_idx]) % 2
                        self.sim.hor_segment_dic[(2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)] += segment_ortho_list[2 * y_loop_idx + x_loop_idx]
                for x_loop_idx in range(3):
                    for y_loop_idx in range(2):
                        self.sim.ver_grid_edge_dic[(2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)] = \
                            (self.sim.ver_grid_edge_dic[(2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)] + grid_edge_list[2 * x_loop_idx + y_loop_idx + 6]) % 2
                        self.sim.ver_segment_dic[(2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx)] += segment_ortho_list[2 * x_loop_idx + y_loop_idx + 6]

            # Initiate non-orthogonal linear and circular segments.
            xtl = xcoord * 24
            ytl = ycoord * 24
            if tile_id in TILE_SEGMENT_DIAG_MAP:
                ((x1, y1), (x2, y2)) = TILE_SEGMENT_DIAG_MAP[tile_id]
                self.sim.segment_dic[coord].append(
                    GridSegmentLinear((xtl + x1, ytl + y1), (xtl + x2, ytl + y2)))
            if tile_id in TILE_SEGMENT_CIRCULAR_MAP:
                ((x_center, y_center), quadrant,
                 convex) = TILE_SEGMENT_CIRCULAR_MAP[tile_id]
                self.sim.segment_dic[coord].append(
                    GridSegmentCircular((xtl + x_center, ytl + y_center), quadrant, convex))

        # Initiate segments from the dictionaries of orthogonal linear segments.
        # Note that two segments of the same position but opposite orientation cancel each other,
        # and no segment is initiated.
        for coord, state in self.sim.hor_segment_dic.items():
            if state:
                xcoord, ycoord = coord
                cell = (math.floor(xcoord / 2),
                        math.floor((ycoord - 0.1 * state) / 2))
                point1 = (12 * xcoord, 12 * ycoord)
                point2 = (12 * xcoord + 12, 12 * ycoord)
                if state == -1:
                    point1, point2 = point2, point1
                self.sim.segment_dic[cell].append(
                    GridSegmentLinear(point1, point2))
        for coord, state in self.sim.ver_segment_dic.items():
            if state:
                xcoord, ycoord = coord
                cell = (math.floor((xcoord - 0.1 * state) / 2),
                        math.floor(ycoord / 2))
                point1 = (12 * xcoord, 12 * ycoord + 12)
                point2 = (12 * xcoord, 12 * ycoord)
                if state == -1:
                    point1, point2 = point2, point1
                self.sim.segment_dic[cell].append(
                    GridSegmentLinear(point1, point2))

    def load_map_entities(self):
        """Load the map entities into the simulation. These should change during the simulation,
        and are reset when a new map is loaded."""

        # initiate player 1 instance of Ninja at spawn coordinates
        self.sim.ninja = Ninja(self.sim, ninja_anim_mode=(
            self.sim.sim_config.enable_anim and not self.sim.sim_config.basic_sim))

        if self.sim.map_data[1233] not in [-1, 1]:
            self.sim.map_data[1233] = -1

        # initiate each entity (other than ninjas)
        index = 1230
        exit_door_count = self.sim.map_data[1156]
        Entity.entity_counts = [0] * 40 # Reset global entity counts
        while (index < len(self.sim.map_data)):
            entity_type = self.sim.map_data[index]
            xcoord = self.sim.map_data[index + 1]
            ycoord = self.sim.map_data[index + 2]
            orientation = self.sim.map_data[index + 3]
            mode = self.sim.map_data[index + 4]
            entity = None # Initialize entity to None

            if entity_type == 1:
                entity = EntityToggleMine(entity_type, self.sim, xcoord, ycoord, 0)
            elif entity_type == 2:
                entity = EntityGold(entity_type, self.sim, xcoord, ycoord)
            elif entity_type == 3:
                parent = EntityExit(entity_type, self.sim, xcoord, ycoord)
                self.sim.entity_dic[entity_type].append(parent)
                child_xcoord = self.sim.map_data[index + 5 * exit_door_count + 1]
                child_ycoord = self.sim.map_data[index + 5 * exit_door_count + 2]
                entity = EntityExitSwitch(
                    4, self.sim, child_xcoord, child_ycoord, parent)
            elif entity_type == 5:
                entity = EntityDoorRegular(
                    entity_type, self.sim, xcoord, ycoord, orientation, xcoord, ycoord)
            elif entity_type == 6:
                switch_xcoord = self.sim.map_data[index + 6]
                switch_ycoord = self.sim.map_data[index + 7]
                entity = EntityDoorLocked(
                    entity_type, self.sim, xcoord, ycoord, orientation, switch_xcoord, switch_ycoord)
            elif entity_type == 8:
                switch_xcoord = self.sim.map_data[index + 6]
                switch_ycoord = self.sim.map_data[index + 7]
                entity = EntityDoorTrap(
                    entity_type, self.sim, xcoord, ycoord, orientation, switch_xcoord, switch_ycoord)
            elif entity_type == 10:
                entity = EntityLaunchPad(
                    entity_type, self.sim, xcoord, ycoord, orientation)
            elif entity_type == 11:
                entity = EntityOneWayPlatform(
                    entity_type, self.sim, xcoord, ycoord, orientation)
            elif entity_type == 14 and not self.sim.sim_config.basic_sim:
                entity = EntityDroneZap(
                    entity_type, self.sim, xcoord, ycoord, orientation, mode)
            # elif type == 15 and not ARGUMENTS.basic_sim: # Placeholder for EntityDroneChaser
            #    entity = EntityDroneChaser(type, self.sim, xcoord, ycoord, orientation, mode)
            elif entity_type == 17:
                entity = EntityBounceBlock(entity_type, self.sim, xcoord, ycoord)
            elif entity_type == 20:
                entity = EntityThwump(
                    entity_type, self.sim, xcoord, ycoord, orientation)
            elif entity_type == 21:
                entity = EntityToggleMine(entity_type, self.sim, xcoord, ycoord, 1) # Active mine
            # elif entity_type == 23 and not self.sim.sim_config.basic_sim: # Placeholder for EntityLaser
            #     entity = EntityLaser(
            #         entity_type, self.sim, xcoord, ycoord, orientation, mode)
            elif entity_type == 24:
                entity = EntityBoostPad(entity_type, self.sim, xcoord, ycoord)
            elif entity_type == 25 and not self.sim.sim_config.basic_sim:
                entity = EntityDeathBall(entity_type, self.sim, xcoord, ycoord)
            elif entity_type == 26 and not self.sim.sim_config.basic_sim:
                entity = EntityMiniDrone(
                    entity_type, self.sim, xcoord, ycoord, orientation, mode)
            elif entity_type == 28:
                entity = EntityShoveThwump(entity_type, self.sim, xcoord, ycoord)
            
            if entity:
                # It's possible that the entity type does not yet exist in entity_dic if it's the first of its kind
                if entity_type not in self.sim.entity_dic:
                    self.sim.entity_dic[entity_type] = [] # Initialize list for new entity type
                self.sim.entity_dic[entity_type].append(entity)
                self.sim.grid_entity[entity.cell].append(entity)
            index += 5

        for entity_list in self.sim.entity_dic.values():
            for entity_instance in entity_list:
                entity_instance.log_position() 