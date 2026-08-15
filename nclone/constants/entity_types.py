# All entity types in the game.
#
# Authoritative source: the switch in `Simulator::createEntity` (0x1051A400) in
# `npp.dll`, cross-referenced against each class's constructor and against the
# `count(type) = u16 at sim + 0x0E + 2*(type-1)` layout formula -- three
# independent confirmations. See `docs/ENTITY_INTEGRATION_SPEC.md` §2.
#
# Two things the table below encodes that are easy to get wrong:
#   * Types 7 and 9 are NOT entities. They are the switch halves of the locked
#     and trap doors, and the door entity owns both positions. The real game
#     hard-errors if asked to construct one, so a loader that creates anything
#     for 7 or 9 has a bug. They are named here only so the 5-byte map stride
#     stays legible.
#   * Types 1 and 21 are the SAME class, differing by constructor argument.


class EntityType:
    """All entity types in the game"""

    NINJA = 0
    TOGGLE_MINE = 1
    GOLD = 2
    EXIT_DOOR = 3
    EXIT_SWITCH = 4
    REGULAR_DOOR = 5
    LOCKED_DOOR = 6
    LOCKED_DOOR_SWITCH = 7  # not an entity; owned by LOCKED_DOOR
    TRAP_DOOR = 8
    TRAP_DOOR_SWITCH = 9  # not an entity; owned by TRAP_DOOR
    LAUNCH_PAD = 10
    ONE_WAY = 11
    DRONE_CHAINGUN = 12
    DRONE_LASER = 13
    DRONE_ZAP = 14
    DRONE_CHASER = 15
    FLOOR_GUARD = 16
    BOUNCE_BLOCK = 17
    ROCKET = 18
    TURRET = 19
    THWUMP = 20
    TOGGLE_MINE_TOGGLED = 21
    EVIL_NINJA = 22
    DUAL_LASER = 23
    BOOST_PAD = 24
    #: `EntityBat` in the binary -- this is the death ball, not a separate bat.
    DEATH_BALL = 25
    MINI_DRONE = 26
    EYE_BAT = 27
    SHWUMP = 28
    #: Spawned rather than map-placed: its factory call takes no position
    #: arguments, unlike every other case (spec §2).
    PLAYER_CONTROLLED_ROCKET = 29


#: Types the parser must never instantiate -- the door entity owns them.
PHANTOM_ENTITY_TYPES = frozenset(
    {EntityType.LOCKED_DOOR_SWITCH, EntityType.TRAP_DOOR_SWITCH}
)

#: Highest type id, used to size per-type tables. 29 exists even though it is not
#: map-placed.
MAX_ENTITY_TYPE = EntityType.PLAYER_CONTROLLED_ROCKET


# ---------------------------------------------------------------------------
# Entity profiles
#
# Two different questions get asked about "which entities are allowed", and
# conflating them is what let the training manifest go stale for a whole entity
# reverse-engineering effort:
#
#   1. CAPABILITY -- "can the simulator run this?"  Tracks the engines, changes
#      when entity work lands.  -> NATIVE_IMPLEMENTED_TYPES
#   2. CONTRACT -- "is this generator allowed to emit this?"  A declaration a
#      procedural generator makes about its own output, and deliberately much
#      narrower than what the sim can run.  -> MINIMAL_TRAINING_SUBSET
#
# Keep them separate. Widening (1) must never silently widen (2).
# ---------------------------------------------------------------------------

#: Every type `nclone-native` constructs (`map/loader.rs` dispatch, verified
#: against `types.rs::EntityType::from_id`). Phantom switch halves are excluded
#: because no entity is ever built for them, and 29 because it is spawned at
#: runtime rather than placed by the map.
NATIVE_IMPLEMENTED_TYPES = frozenset(
    set(range(1, 29)) - set(PHANTOM_ENTITY_TYPES)
)

#: The original five-type training subset. Predates native entity parity; kept
#: as a *named* profile so datasets generated against it stay reproducible and
#: so procedural generators keep a tight contract to declare against.
MINIMAL_TRAINING_SUBSET = frozenset(
    {
        EntityType.TOGGLE_MINE,
        EntityType.GOLD,
        EntityType.EXIT_DOOR,
        EntityType.EXIT_SWITCH,
        EntityType.TOGGLE_MINE_TOGGLED,
    }
)

#: Hazards with no physical collision: they can kill the ninja but never bend
#: its trajectory (spec §6.6). Deleting one can therefore only make a level
#: *easier*, so a winning demo stays winning -- which is what makes
#: hazard-stripping a sound way to bring an out-of-scope human level into scope.
#: Verify with a demo replay anyway; this is a claim about the physics, and the
#: replay is what actually holds us to it.
PURE_HAZARD_TYPES = frozenset(
    {
        EntityType.DRONE_CHAINGUN,
        EntityType.DRONE_LASER,
        EntityType.DRONE_ZAP,
        EntityType.DRONE_CHASER,
        EntityType.FLOOR_GUARD,
        EntityType.ROCKET,
        EntityType.TURRET,
        EntityType.EVIL_NINJA,
        EntityType.DUAL_LASER,
        EntityType.DEATH_BALL,
        EntityType.MINI_DRONE,
        EntityType.EYE_BAT,
    }
)

#: Types that gate, carry, propel or physically block the ninja. Removing one
#: changes the route, so a demo recorded against it will not replay -- these can
#: only be *supported*, never stripped.
TRAVERSAL_TYPES = frozenset(
    {
        EntityType.REGULAR_DOOR,
        EntityType.LOCKED_DOOR,
        EntityType.TRAP_DOOR,
        EntityType.LAUNCH_PAD,
        EntityType.ONE_WAY,
        EntityType.BOUNCE_BLOCK,
        EntityType.THWUMP,
        EntityType.BOOST_PAD,
        EntityType.SHWUMP,
    }
)

# ---------------------------------------------------------------------------
# Training-scope profiles (docs/ENTITY_INTEGRATION_PLAN.md §9)
#
# The full-entity rollout is phased by WHAT INFRASTRUCTURE an entity needs, not
# by how many maps it appears on -- the per-entity coverage curve is flat (no
# type unlocks more than ~6 points of corpus), so an entity-at-a-time rollout
# front-loads the expensive graph work and buys almost nothing per step.
#
# The split that does carve the corpus is PURE_HAZARD_TYPES vs TRAVERSAL_TYPES:
# pure hazards never bend the ninja's trajectory, so the reachability graph, the
# cost-to-go field and the d* demo generator are ALREADY CORRECT on maps that
# only add hazards. Admitting all of them costs observation work and no planner
# work, and moves S* scope from 14.2% to 43.3%.
#
# Percentages below are of the 2382-map S* family (the difficulty-ordered target
# set), measured over nclone/maps/official.
# ---------------------------------------------------------------------------

#: Phase 1 -- hazards. Observation work only; the planner is untouched.
PHASE1_HAZARD_SCOPE = MINIMAL_TRAINING_SUBSET | PURE_HAZARD_TYPES  # 43.3% S*

#: Phase 2 -- the two traversal types the reachability graph ALREADY models
#: (`graph/reachability/entity_mask.py`). Observation channels only.
PHASE2_SCOPE = PHASE1_HAZARD_SCOPE | {
    EntityType.LOCKED_DOOR,
    EntityType.LOCKED_DOOR_SWITCH,
    EntityType.ONE_WAY,
}  # 63.5% S*

#: Phase 3 -- the first traversal types that need NEW graph work. This is the
#: only phase that changes the cost-to-go field, and therefore the only one that
#: can invalidate PBRS, the demo corpus and the reverse curriculum at once.
PHASE3_SCOPE = PHASE2_SCOPE | {
    EntityType.BOUNCE_BLOCK,
    EntityType.TRAP_DOOR,
    EntityType.TRAP_DOOR_SWITCH,
    EntityType.LAUNCH_PAD,
}  # 90.5% S*

#: Phase 4 -- the tail. Everything the simulators implement.
PHASE4_SCOPE = NATIVE_IMPLEMENTED_TYPES | set(PHANTOM_ENTITY_TYPES)  # 100% S*

#: Selectable by name from CLIs and configs, so a dataset can record *which*
#: profile produced it instead of pinning a bare set literal.
#:
#: `native` and `full` are the same set today (entity parity is complete) and are
#: both kept: `native` is the CAPABILITY question ("what can the sim run"), which
#: tracks the engines, while `full` is a CONTRACT a caller declares. They will
#: diverge again the moment an entity is added to one engine before the other.
ENTITY_PROFILES = {
    "minimal": MINIMAL_TRAINING_SUBSET,
    "native": NATIVE_IMPLEMENTED_TYPES,
    "phase1": PHASE1_HAZARD_SCOPE,
    "phase2": PHASE2_SCOPE,
    "phase3": PHASE3_SCOPE,
    "phase4": PHASE4_SCOPE,
    "full": PHASE4_SCOPE,
}


def map_in_scope(entity_types, profile) -> bool:
    """Is a map playable under ``profile``, given the entity types it contains?

    ``entity_types`` is any iterable of raw type ids as they appear in the map
    file -- INCLUDING the phantom switch halves (7, 9), which every scope that
    admits their parent door admits alongside it. Type 0 (the ninja spawn) is
    always in scope and is ignored.

    ``profile`` is a name from :data:`ENTITY_PROFILES` or a set of type ids.
    """
    scope = ENTITY_PROFILES[profile] if isinstance(profile, str) else set(profile)
    return set(entity_types) - {EntityType.NINJA} <= set(scope)
