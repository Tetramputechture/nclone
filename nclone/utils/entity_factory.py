"""
Entity factory utilities for creating entities based on type identifiers.

This module provides utilities for mapping entity type integers to their corresponding
entity classes and creating entity instances with the appropriate parameters.
"""

# Import entity classes
import os

from ..entity_classes.entity_toggle_mine import EntityToggleMine
from ..entity_classes.entity_gold import EntityGold
from ..entity_classes.entity_exit import EntityExit
from ..entity_classes.entity_exit_switch import EntityExitSwitch
from ..entity_classes.entity_door_regular import EntityDoorRegular
from ..entity_classes.entity_door_locked import EntityDoorLocked
from ..entity_classes.entity_door_trap import EntityDoorTrap
from ..entity_classes.entity_launch_pad import EntityLaunchPad
from ..entity_classes.entity_one_way_platform import EntityOneWayPlatform
from ..entity_classes.entity_drone_zap import EntityDroneZap
from ..entity_classes.entity_bounce_block import EntityBounceBlock
from ..entity_classes.entity_thwump import EntityThwump
from ..entity_classes.entity_boost_pad import EntityBoostPad
from ..entity_classes.entity_death_ball import EntityDeathBall
from ..entity_classes.entity_mini_drone import EntityMiniDrone
from ..entity_classes.entity_shove_thwump import EntityShoveThwump
from ..entity_classes.entity_drone_chaingun import EntityDroneChaingun
from ..entity_classes.entity_drone_laser import EntityDroneLaser
from ..entity_classes.entity_drone_chaser import EntityDroneChaser
from ..entity_classes.entity_floor_guard import EntityFloorGuard
from ..entity_classes.entity_rocket import EntityRocket
from ..entity_classes.entity_turret import EntityTurret
from ..entity_classes.entity_evil_ninja import EntityEvilNinja
from ..entity_classes.entity_dual_laser import EntityDualLaser
from ..entity_classes.entity_eye_bat import EntityEyeBat
from ..constants.entity_types import EntityType


def _disabled_entity_types():
    """Entity type ids to skip at load, from ``NCLONE_DISABLE_ENTITY_TYPES``.

    A comma-separated list, e.g. ``NCLONE_DISABLE_ENTITY_TYPES=19,23``. Maps load
    as if those types were absent.

    This exists because of how the attract-demo corpus validates hazards. None of
    the reverse-engineered types has a physical collision (spec §6.6), so they
    cannot bend the ninja's trajectory -- an over-lethal one shows up ONLY as a
    drop in the win rate, and since several hazards usually share a map, a single
    bad entity drags down every type it co-occurs with. Bisecting that needs the
    ability to enable exactly one at a time. It is off unless the variable is set.
    """
    raw = os.environ.get("NCLONE_DISABLE_ENTITY_TYPES", "")
    out = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return out


def get_entity_class_for_type(entity_type, sim_config=None):
    """
    Returns the appropriate entity class for the given entity type integer.

    Args:
        entity_type (int): The entity type identifier
        sim_config: Optional simulation configuration object for basic_sim checks

    Returns:
        type: The entity class that should be instantiated for this type,
              or None if the entity type is not supported or should be skipped
    """
    if entity_type in _disabled_entity_types():
        return None

    # Handle entities that require basic_sim check
    if sim_config and entity_type in [
        EntityType.DRONE_ZAP,
        EntityType.MINI_DRONE,
        EntityType.DEATH_BALL,
    ]:
        if sim_config.basic_sim:
            return None

    # Map entity types to their corresponding classes
    entity_class_map = {
        EntityType.TOGGLE_MINE: EntityToggleMine,
        EntityType.GOLD: EntityGold,
        EntityType.EXIT_DOOR: EntityExit,
        EntityType.REGULAR_DOOR: EntityDoorRegular,
        EntityType.LOCKED_DOOR: EntityDoorLocked,
        EntityType.TRAP_DOOR: EntityDoorTrap,
        EntityType.LAUNCH_PAD: EntityLaunchPad,
        EntityType.ONE_WAY: EntityOneWayPlatform,
        EntityType.DRONE_ZAP: EntityDroneZap,
        EntityType.BOUNCE_BLOCK: EntityBounceBlock,
        EntityType.THWUMP: EntityThwump,
        EntityType.TOGGLE_MINE_TOGGLED: EntityToggleMine,
        EntityType.BOOST_PAD: EntityBoostPad,
        EntityType.DEATH_BALL: EntityDeathBall,
        EntityType.MINI_DRONE: EntityMiniDrone,
        EntityType.SHWUMP: EntityShoveThwump,
        # Reverse-engineered from npp.dll; see docs/ENTITY_INTEGRATION_SPEC.md.
        EntityType.DRONE_CHAINGUN: EntityDroneChaingun,
        EntityType.DRONE_LASER: EntityDroneLaser,
        EntityType.DRONE_CHASER: EntityDroneChaser,
        EntityType.FLOOR_GUARD: EntityFloorGuard,
        EntityType.ROCKET: EntityRocket,
        EntityType.TURRET: EntityTurret,
        EntityType.EVIL_NINJA: EntityEvilNinja,
        EntityType.DUAL_LASER: EntityDualLaser,
        EntityType.EYE_BAT: EntityEyeBat,
        # Types 7 and 9 are deliberately absent: they are not entities, they are
        # the switch halves of the locked and trap doors, and the door entity owns
        # both positions. The real game hard-errors if asked to build one.
        #
        # Type 29 (PlayerControlledRocket) is also absent: it is spawned at
        # runtime, not map-placed. Confirmed empirically -- it appears zero times
        # in the entity streams of 8000 decoded attract maps. The class exists in
        # entity_pc_rocket.py for completeness.
    }

    return entity_class_map.get(entity_type)


# old logic:
#         while (index < len(self.sim.map_data)):
# entity_type = self.sim.map_data[index]
# xcoord = self.sim.map_data[index + 1]
# ycoord = self.sim.map_data[index + 2]
# orientation = self.sim.map_data[index + 3]
# mode = self.sim.map_data[index + 4]
# entity = None # Initialize entity to None

# if entity_type == 1:
#     entity = EntityToggleMine(entity_type, self.sim, xcoord, ycoord, 0)
# elif entity_type == 2:
#     entity = EntityGold(entity_type, self.sim, xcoord, ycoord)
# elif entity_type == 3:
#     parent = EntityExit(entity_type, self.sim, xcoord, ycoord)
#     self.sim.entity_dic[entity_type].append(parent)
#     child_xcoord = self.sim.map_data[index + 5 * exit_door_count + 1]
#     child_ycoord = self.sim.map_data[index + 5 * exit_door_count + 2]
#     entity = EntityExitSwitch(
#         4, self.sim, child_xcoord, child_ycoord, parent)
# elif entity_type == 5:
#     entity = EntityDoorRegular(
#         entity_type, self.sim, xcoord, ycoord, orientation, xcoord, ycoord)
# elif entity_type == 6:
#     switch_xcoord = self.sim.map_data[index + 6]
#     switch_ycoord = self.sim.map_data[index + 7]
#     entity = EntityDoorLocked(
#         entity_type, self.sim, xcoord, ycoord, orientation, switch_xcoord, switch_ycoord)
# elif entity_type == 8:
#     switch_xcoord = self.sim.map_data[index + 6]
#     switch_ycoord = self.sim.map_data[index + 7]
#     entity = EntityDoorTrap(
#         entity_type, self.sim, xcoord, ycoord, orientation, switch_xcoord, switch_ycoord)
# elif entity_type == 10:
#     entity = EntityLaunchPad(
#         entity_type, self.sim, xcoord, ycoord, orientation)
# elif entity_type == 11:
#     entity = EntityOneWayPlatform(
#         entity_type, self.sim, xcoord, ycoord, orientation)
# elif entity_type == 14 and not self.sim.sim_config.basic_sim:
#     entity = EntityDroneZap(
#         entity_type, self.sim, xcoord, ycoord, orientation, mode)
# # elif type == 15 and not ARGUMENTS.basic_sim: # Placeholder for EntityDroneChaser
# #    entity = EntityDroneChaser(type, self.sim, xcoord, ycoord, orientation, mode)
# elif entity_type == 17:
#     entity = EntityBounceBlock(entity_type, self.sim, xcoord, ycoord)
# elif entity_type == 20:
#     entity = EntityThwump(
#         entity_type, self.sim, xcoord, ycoord, orientation)
# elif entity_type == 21:
#     entity = EntityToggleMine(entity_type, self.sim, xcoord, ycoord, 1) # Active mine
# # elif entity_type == 23 and not self.sim.sim_config.basic_sim: # Placeholder for EntityLaser
# #     entity = EntityLaser(
# #         entity_type, self.sim, xcoord, ycoord, orientation, mode)
# elif entity_type == 24:
#     entity = EntityBoostPad(entity_type, self.sim, xcoord, ycoord)
# elif entity_type == 25 and not self.sim.sim_config.basic_sim:
#     entity = EntityDeathBall(entity_type, self.sim, xcoord, ycoord)
# elif entity_type == 26 and not self.sim.sim_config.basic_sim:
#     entity = EntityMiniDrone(
#         entity_type, self.sim, xcoord, ycoord, orientation, mode)
# elif entity_type == 28:
#     entity = EntityShoveThwump(entity_type, self.sim, xcoord, ycoord)

# if entity:
#     # It's possible that the entity type does not yet exist in entity_dic if it's the first of its kind
#     if entity_type not in self.sim.entity_dic:
#         self.sim.entity_dic[entity_type] = [] # Initialize list for new entity type
#     self.sim.entity_dic[entity_type].append(entity)
#     self.sim.grid_entity[entity.cell].append(entity)
# index += 5


def create_entity_instance(
    entity_type,
    sim,
    xcoord,
    ycoord,
    orientation,
    mode,
    map_data=None,
    index=None,
    exit_door_count=None,
):
    """
    Creates an entity instance based on the entity type and parameters.

    Args:
        entity_type (int): The entity type identifier
        sim: The simulation object
        xcoord (int): X coordinate
        ycoord (int): Y coordinate
        orientation (int): Entity orientation
        mode (int): Entity mode
        map_data: Optional map data array for entities requiring additional coordinates
        index (int): Optional current index in map data for entities needing switch coordinates
        exit_door_count (int): Optional number of exit doors for exit door entities

    Returns:
        Entity or None: The created entity instance, or None if no entity should be created
    """
    entity_class = get_entity_class_for_type(
        entity_type, sim.sim_config if hasattr(sim, "sim_config") else None
    )
    if not entity_class:
        return None

    entity = None

    if entity_type == EntityType.TOGGLE_MINE:
        entity = entity_class(entity_type, sim, xcoord, ycoord, 0)
    elif entity_type == EntityType.GOLD:
        entity = entity_class(entity_type, sim, xcoord, ycoord)
    elif entity_type == EntityType.EXIT_DOOR:
        # Special case: creates parent entity and adds child entity separately
        if map_data is None or index is None or exit_door_count is None:
            raise ValueError(
                "EXIT_DOOR entities require map_data, index, and exit_door_count parameters"
            )
        parent = entity_class(entity_type, sim, xcoord, ycoord)
        sim.entity_dic[entity_type].append(parent)
        child_xcoord = map_data[index + 5 * exit_door_count + 1]
        child_ycoord = map_data[index + 5 * exit_door_count + 2]
        entity = EntityExitSwitch(
            EntityType.EXIT_SWITCH, sim, child_xcoord, child_ycoord, parent
        )
    elif entity_type == EntityType.REGULAR_DOOR:
        entity = entity_class(
            entity_type, sim, xcoord, ycoord, orientation, xcoord, ycoord
        )
    elif entity_type in [EntityType.LOCKED_DOOR, EntityType.TRAP_DOOR]:
        if map_data is None or index is None:
            raise ValueError(
                "LOCKED_DOOR and TRAP_DOOR entities require map_data and index parameters"
            )
        # Switch coordinates are at index+6 and index+7 in the map data
        # The format varies by file version:
        # 9-byte format:  [type, x, y, orientation, mode, switch_x, switch_y, 0, 0]
        # 10-byte format: [type, x, y, orientation, mode, ???, switch_x, switch_y, ???, 0]
        # For compatibility with both formats, read from index+6 and index+7
        switch_xcoord = map_data[index + 6]
        switch_ycoord = map_data[index + 7]
        entity = entity_class(
            entity_type, sim, xcoord, ycoord, orientation, switch_xcoord, switch_ycoord
        )
    elif entity_type in [
        EntityType.LAUNCH_PAD,
        EntityType.ONE_WAY,
        EntityType.THWUMP,
        EntityType.TURRET,
    ]:
        entity = entity_class(entity_type, sim, xcoord, ycoord, orientation)
    elif entity_type in [
        EntityType.DRONE_ZAP,
        EntityType.MINI_DRONE,
        EntityType.DRONE_CHAINGUN,
        EntityType.DRONE_LASER,
        EntityType.DRONE_CHASER,
        EntityType.DUAL_LASER,
    ]:
        entity = entity_class(entity_type, sim, xcoord, ycoord, orientation, mode)
    elif entity_type in [
        EntityType.BOUNCE_BLOCK,
        EntityType.BOOST_PAD,
        EntityType.DEATH_BALL,
        EntityType.SHWUMP,
        EntityType.FLOOR_GUARD,
        EntityType.EYE_BAT,
        EntityType.ROCKET,
        EntityType.EVIL_NINJA,
    ]:
        entity = entity_class(entity_type, sim, xcoord, ycoord)
    elif entity_type == EntityType.TOGGLE_MINE_TOGGLED:
        entity = entity_class(entity_type, sim, xcoord, ycoord, 1)  # Active mine

    return entity
