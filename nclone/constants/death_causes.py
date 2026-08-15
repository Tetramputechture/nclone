# Canonical death-cause taxonomy.
#
# Every way the ninja can die, as the string both engines write to
# `Ninja.death_cause` and the env forwards as `info["death_cause"]`.
#
# WHY THIS FILE EXISTS. The two engines used to disagree. The pure-Python sim
# named each drone class (`drone_zap` / `mini_drone` / `drone_chaser`) and each
# dual-laser mode (`laser_spinner` / `laser_surface`); `nclone-native` collapsed
# the first three to `"drone"` and the last two to `"laser"`. Training runs on
# native, so the collapse erased exactly the distinctions that matter for
# diagnosis: a patrol-drone death is a policy failure against a threat the
# forecast channels show exactly, while a chaser death is a reactive-threat
# problem with a different fix. Downstream consumers must never have to know
# which backend produced a string, so the strings are pinned here and asserted
# against both engines by `tests/test_death_cause_parity.py`.
#
# Adding a cause: add the constant, put it in exactly one class in
# `DEATH_CAUSE_CLASS`, map it in `DEATH_CAUSE_TO_ENTITY_TYPE` if an entity is
# responsible, and emit the identical literal from BOTH engines.

from .entity_types import EntityType

# --- Environment / self-inflicted ------------------------------------------

#: Landed (or hit a ceiling) above `MAX_SURVIVABLE_IMPACT`.
TERMINAL_IMPACT = "terminal_impact"
#: Squeezed below `MIN_SURVIVABLE_CRUSHING`. Almost always a traversal entity --
#: a thwump, a shove thwump, a bounce block or a door closing on the ninja --
#: but the ninja detects it geometrically and cannot name the culprit, so this
#: stays one cause rather than being split per entity.
CRUSH = "crush"

# --- Static hazards ---------------------------------------------------------

MINE = "mine"

# --- Contact-lethal patrols and homers ---------------------------------------

DRONE_ZAP = "drone_zap"
MINI_DRONE = "mini_drone"
DRONE_CHASER = "drone_chaser"
DEATH_BALL = "death_ball"
EYE_BAT = "eye_bat"
EVIL_NINJA = "evil_ninja"
FLOOR_GUARD = "floor_guard"

# --- Beams (hitscan; the emitter's own body is harmless) ---------------------

TURRET = "turret"
LASER_SPINNER = "laser_spinner"
LASER_SURFACE = "laser_surface"
#: Both shooter drones (12, 13) discharge through the shared base class and the
#: reference engine gives them one string. Kept as one cause for parity; the
#: emitters are told apart by entity type, not by cause.
DRONE_SHOT = "drone_shot"

# --- Projectiles -------------------------------------------------------------

ROCKET = "rocket"
PC_ROCKET = "pc_rocket"

# --- Traversal bodies that are also lethal -----------------------------------

THWUMP = "thwump"
SHOVE_THWUMP = "shove_thwump"


#: Stable order for reports, TensorBoard rows and fixed-width metric blocks.
#: APPEND-ONLY -- a reordering silently re-labels every historical run.
ALL_DEATH_CAUSES = (
    TERMINAL_IMPACT,
    CRUSH,
    MINE,
    DRONE_ZAP,
    MINI_DRONE,
    DRONE_CHASER,
    DEATH_BALL,
    EYE_BAT,
    EVIL_NINJA,
    FLOOR_GUARD,
    TURRET,
    LASER_SPINNER,
    LASER_SURFACE,
    DRONE_SHOT,
    ROCKET,
    PC_ROCKET,
    THWUMP,
    SHOVE_THWUMP,
)

#: Reported when a death arrives with no cause string. A non-zero rate here is a
#: taxonomy gap -- a kill site that forgot its `cause=` -- not a category.
UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Threat super-classes
#
# The policy does not need to know that a thing is a rocket. It needs to know
# how the threat behaves, and that is a four-way split, not a twenty-way one.
# These are the classes the observation's fixed-width threat block is indexed by
# (see `gym_environment/constants.py`), so this mapping and that block's width
# move together.
# ---------------------------------------------------------------------------

#: Lethal on contact with a body that moves on its own. Avoid by not being there.
CLASS_CONTACT = "contact"
#: Lethal on contact with a body that moves TOWARD the ninja. Avoid by denying
#: approach -- position alone is not enough, because the threat follows.
CLASS_HOMING = "homing"
#: Lethal along a line, instantaneously. Avoid by breaking line of sight or by
#: crossing between sweeps; distance is nearly irrelevant.
CLASS_BEAM = "beam"
#: Lethal via a travelling body launched after a telegraph. Avoid by reading the
#: telegraph and moving before it resolves.
CLASS_PROJECTILE = "projectile"
#: Not a threat entity -- the level's own physics.
CLASS_ENVIRONMENT = "environment"

THREAT_CLASSES = (CLASS_CONTACT, CLASS_HOMING, CLASS_BEAM, CLASS_PROJECTILE)

DEATH_CAUSE_CLASS = {
    TERMINAL_IMPACT: CLASS_ENVIRONMENT,
    CRUSH: CLASS_ENVIRONMENT,
    # A mine never moves, so it is neither contact-patrol nor homing. It is
    # already fully covered by `hazard_map`, and it is deliberately NOT folded
    # into the threat block: a mine death and a drone death have different
    # causes and different fixes.
    MINE: CLASS_ENVIRONMENT,
    DRONE_ZAP: CLASS_CONTACT,
    MINI_DRONE: CLASS_CONTACT,
    # The floor guard charges along a floor once it latches, and until it
    # latches it is inert. It tracks the ninja to latch, but not afterwards --
    # contact, not homing.
    FLOOR_GUARD: CLASS_CONTACT,
    DRONE_CHASER: CLASS_HOMING,
    DEATH_BALL: CLASS_HOMING,
    EYE_BAT: CLASS_HOMING,
    # The evil ninja follows the ninja's own recorded path, which is homing in
    # effect even though the mechanism is a tape replay.
    EVIL_NINJA: CLASS_HOMING,
    TURRET: CLASS_BEAM,
    LASER_SPINNER: CLASS_BEAM,
    LASER_SURFACE: CLASS_BEAM,
    DRONE_SHOT: CLASS_BEAM,
    ROCKET: CLASS_PROJECTILE,
    PC_ROCKET: CLASS_PROJECTILE,
    # Thwumps charge along one axis and crush; the lethal face arrives on a
    # trajectory, which is what contact means here.
    THWUMP: CLASS_CONTACT,
    SHOVE_THWUMP: CLASS_CONTACT,
}

#: Which entity type is responsible, where one is. Absent for the environmental
#: causes. `DRONE_SHOT` is ambiguous by design (see its constant) and maps to the
#: chaingun drone only as the more common of the two emitters; use it for
#: grouping, never for attribution.
DEATH_CAUSE_TO_ENTITY_TYPE = {
    MINE: EntityType.TOGGLE_MINE,
    DRONE_ZAP: EntityType.DRONE_ZAP,
    MINI_DRONE: EntityType.MINI_DRONE,
    DRONE_CHASER: EntityType.DRONE_CHASER,
    DEATH_BALL: EntityType.DEATH_BALL,
    EYE_BAT: EntityType.EYE_BAT,
    EVIL_NINJA: EntityType.EVIL_NINJA,
    FLOOR_GUARD: EntityType.FLOOR_GUARD,
    TURRET: EntityType.TURRET,
    LASER_SPINNER: EntityType.DUAL_LASER,
    LASER_SURFACE: EntityType.DUAL_LASER,
    DRONE_SHOT: EntityType.DRONE_CHAINGUN,
    ROCKET: EntityType.ROCKET,
    PC_ROCKET: EntityType.PLAYER_CONTROLLED_ROCKET,
    THWUMP: EntityType.THWUMP,
    SHOVE_THWUMP: EntityType.SHWUMP,
}


def classify_death_cause(cause) -> str:
    """Map a raw ``info["death_cause"]`` to its threat class.

    Unknown and missing causes classify as :data:`CLASS_ENVIRONMENT` so a
    taxonomy gap never inflates a threat-class rate. Track :data:`UNKNOWN`
    separately to see the gap itself.
    """
    if not cause:
        return CLASS_ENVIRONMENT
    return DEATH_CAUSE_CLASS.get(str(cause).lower(), CLASS_ENVIRONMENT)


def normalize_death_cause(cause) -> str:
    """Map a raw ``info["death_cause"]`` to a known cause or :data:`UNKNOWN`.

    Never raises: an unrecognised string becomes :data:`UNKNOWN` rather than
    creating an unbounded set of TensorBoard rows from a corrupt value.
    """
    if not cause:
        return UNKNOWN
    lowered = str(cause).lower()
    return lowered if lowered in _KNOWN else UNKNOWN


_KNOWN = frozenset(ALL_DEATH_CAUSES)
