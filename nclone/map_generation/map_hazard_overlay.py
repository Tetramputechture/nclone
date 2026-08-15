"""Dress an ARBITRARY base generator's level with hazards of one class.

This is `map_mine_overlay` generalised past mines, and it exists for the same
reason that one did. `mine_path` was a bespoke layout generator, and the result
was a mine *crowd* capped at 4 regardless of mine count, because every mine sat
on one corridor floor row at fixed spacing -- four tiers of ramping produced a
distribution that never once resembled the official one. Dressing someone else's
level instead makes the whole 23-generator registry reusable and keeps morphology
(slopes, shafts, mazes, platforms, loops) orthogonal to hazard content.

    base generator (any)  ->  route the CLEAN base (plan_dstar_guided_fast)
                          ->  place hazards of class C at count N
                          ->  replay the base tape on a fresh sim
                          ->  classify the outcome

WHAT IS GENUINELY DIFFERENT FROM MINES, AND IT IS THE WHOLE DESIGN.

**The planner does not model hazards.** At phase 1 that is deliberate: pure
hazards never bend the ninja's trajectory, so the reachability graph and the
cost-to-go field are *already correct* on a hazard-dressed level. Two consequences
follow and both are load-bearing:

1. Routing certifies **geometric solvability only**. A d* route can pass straight
   through a drone's patrol path and report `solved`. So routing the dressed level
   would tell us nothing the clean base did not -- which is why the base is routed
   ONCE and the tape is reused down the whole count ladder, instead of paying 2-5 s
   per rung the way the mine overlay must.
2. The **replay becomes the discriminator**, not a belt-and-braces check. Three
   outcomes, and conflating them biases the bucket:

   | outcome | meaning | action |
   |---|---|---|
   | route solved, replay survives | the naive route is safe | keep; ship the route as BC data |
   | route solved, replay dies | geometry fine, needs TIMING | keep at T3/T4, ship no demo |
   | route unsolved | geometry broken | thin and retry |
   | route timeout | says nothing about the level | do NOT thin -- log and retry |

   The second row is the interesting one: those levels are precisely the ones that
   teach hazard timing, and discarding them would bias the bucket toward hazards
   that never actually threaten the route. `TARGET_LETHAL_FRACTION` sets their
   share per tier deliberately rather than letting it fall out.

   The timeout-vs-unsolved distinction is inherited from `map_mine_overlay` and is
   there for a reason: *a gate that rejects everything is not evidence that the
   thing being gated is impossible.*

SOLID classes (thwump, shove thwump) are the exception to (1): their bodies DO
change geometry, so `HazardSpec.is_solid` forces a re-route of the dressed level.

PLACEMENT IS THE ACTUAL WORK. A mine is placeable anywhere; a hazard is not.
Nearly every class has a mounting or line-of-sight precondition, and a hazard that
fails it is either inert or invisible. The phantom-mine incident is the precedent:
all PLATFORMING / MINE_NAVIGATION mines were buried in solid rock while poisoning
60-72% of standable ground. **A hazard placed without its precondition is worse
than no hazard: it costs observation bandwidth and teaches nothing.**

Every number in `HAZARD_SPECS` below is measured off the official phase-1 corpus
(`npp-rl/scripts/hazard_census.py`, 1542 maps), not chosen. Two of those
measurements contradict what intuition would have produced:

* **Density is FLAT across the game's own difficulty ordering** -- patrol drone
  median count is 14/12/14/14/14 across difficulty quintiles, rocket is 1 in every
  one. So the tiers ramp the BASE level's morphology and barely move the count.
  Ramping count from zero is what leaves a T4 easier than official's T1.
* **Most hazards are NOT route-adjacent** -- only 15-37% sit within three tiles of
  the route. Mounting every hazard on the route corridor, which is the obvious
  reading of "place it where it threatens the ninja", would build a bucket far
  harsher than the target distribution. `near_route_fraction` targets the measured
  value instead.

THE ONE EXCEPTION TO "DO NOT RAMP COUNT" IS T1, and it matters: for a mechanic the
agent has never encountered, tier 1 must be a *single isolated instance in generous
geometry*. That is the `shaft_t2_one_mine` precedent, and it is the difference
between a learnable on-ramp and a bucket that reads 0.000 forever.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .map import Map
from .map_mine_overlay import (
    ENT_START,
    ENT_STRIDE,
    EXIT_DOOR_COUNT_IDX,
    EXIT_DOOR_TYPE,
    EXIT_SWITCH_TYPE,
    GRID_W,
    GRID_H,
    PX_PER_TILE,
    SPAWN_X_IDX,
    SPAWN_Y_IDX,
    TILE_START,
    _plan_with_hard_timeout,
)

logger = logging.getLogger(__name__)

EMPTY_TILE = 0
#: Tiles that are fully solid. Half tiles (2-5) are only partially solid, and
#: requiring a fully EMPTY tile is the conservative choice for placement -- it is
#: what keeps hazards out of slope geometry, which is the phantom-mine failure.
SOLID_TILES = frozenset({1, 2, 3, 4, 5})


# ---------------------------------------------------------------------------
# Hazard classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HazardSpec:
    """How one hazard class is placed, and what the official corpus says it
    should look like when it is.

    Every field carrying a number has a measurement behind it; see the census
    table in each entry's comment. `count_*` are over levels that CONTAIN the
    class -- a zero-inflated mean over the whole corpus is not a tier density.
    """

    name: str
    #: `(entity_type, weight, grid_px, offset_x, offset_y)`.
    #:
    #: The grid is PER TYPE, not per class, and getting that wrong is silent: a
    #: zap drone steps between 24 px cell centres (`DRONE_GRID_SIZE`) while a mini
    #: drone steps on a 12 px grid (`MINI_DRONE_GRID_SIZE`), so they land on
    #: different lattices even though both are "patrol drones". Confirmed against
    #: every instance in the corpus -- zap sits at (x, y) = (12, 12) mod 24 in
    #: 5292/5292, mini at (6|18, 6|18) mod 24 in 5230/5230. A drone off its own
    #: lattice starts mid-step between cells and its entire patrol is
    #: phase-shifted relative to anything the forecast channels predict.
    types: Tuple[Tuple[int, int, float, float, float], ...]
    #: Orientation bytes to draw from, uniformly. Empty means the entity ignores
    #: orientation (`utils.entity_factory.create_entity_instance` passes it only
    #: for LAUNCH_PAD / ONE_WAY / THWUMP / TURRET and the drone family).
    orientations: Tuple[int, ...]
    #: Mode bytes to draw from, with weights. Drone modes are
    #: 0/1 = follow wall CW/CCW, 2/3 = wander CW/CCW.
    modes: Tuple[Tuple[int, int], ...]
    #: Fraction of instances with a solid tile within one tile below.
    supported_fraction: float
    #: Fraction that should sit within `CORRIDOR_PX` of the routed path.
    near_route_fraction: float
    #: Count band over levels containing the class: (p25, p50, p75, p90).
    count_p25: int
    count_p50: int
    count_p75: int
    count_p90: int
    #: Largest number of the class within `CROWD_RADIUS_PX` of one another.
    crowd_p50: int
    crowd_p90: int
    #: Body blocks the ninja, so it changes GEOMETRY and the dressed level must be
    #: re-routed rather than reusing the clean base's tape.
    is_solid: bool = False
    #: Needs a clear travel lane ahead of its mount (thwump family).
    lane_tiles: int = 0
    #: Needs unobstructed line of sight to route cells (turret, rocket, beams).
    needs_los: bool = False
    #: HARD requirement for ground support, not a distributional target.
    #:
    #: `supported_fraction` steers a MIX and is allowed to miss; this rejects the
    #: candidate outright. A floor guard charges ALONG A FLOOR -- airborne it has
    #: no mechanic at all, and is pure observation bandwidth attached to nothing.
    #: Measured before this existed: only 32% of FLOOR_GUARD/T1 guards were
    #: grounded, against a 0.83 target that was itself wrong (the census's
    #: support probe reads slopes and half-tiles as unsupported, so official's
    #: real figure is nearer 1.0 -- 691/767 instances sit exactly on the floor
    #: line at y mod 24 == 18).
    requires_support: bool = False
    #: True when the numbers above are NOT measured from official -- the type is
    #: out of phase-1 ingest scope, so the corpus holds no instances to measure.
    #: Flagged rather than silently presented as calibration.
    uncalibrated: bool = False


#: Measured by `npp-rl/scripts/hazard_census.py` over the 1542-level phase-1
#: official corpus, plus a direct scan of every instance's (x mod 24, y mod 24),
#: orientation and mode bytes. Numbers in the comments are the raw readings.
HAZARD_SPECS: Dict[str, HazardSpec] = {
    # 444 levels (28.8%), count p25/50/75/90 = 8/14/28/52, crowd 6/18, near 29.5%.
    # zap sits at cell centre (12,12) in 5292/5292 instances; mini sits on the
    # 12 px half-lattice at (6|18, 6|18) in all 5230. Orientation is uniform over
    # {0,2,4,6} for both -- the drone base does `turn(orientation // 2)`, so the
    # four diagonals collapse onto the same headings and official never uses them.
    # zap mode is uniform 0-3; mini skews to wall-follow (0: 2411, 1: 1546).
    "patrol_drone": HazardSpec(
        name="patrol_drone",
        types=((14, 50, 24.0, 12.0, 12.0), (26, 50, 12.0, 6.0, 6.0)),
        orientations=(0, 2, 4, 6),
        modes=((0, 30), (1, 25), (2, 25), (3, 20)),
        supported_fraction=0.27,
        near_route_fraction=0.295,
        count_p25=8, count_p50=14, count_p75=28, count_p90=52,
        crowd_p50=6, crowd_p90=18,
    ),
    # 26 levels (1.7%), count 2/4/5/7, crowd 1/2, near 11.7%. Cell centre in all
    # 137 instances; mode skews hard to wander (2: 74, 3: 54).
    "chaser_drone": HazardSpec(
        name="chaser_drone",
        types=((15, 100, 24.0, 12.0, 12.0),),
        orientations=(0, 2, 4, 6),
        modes=((2, 55), (3, 40), (0, 3), (1, 2)),
        supported_fraction=0.18,
        near_route_fraction=0.117,
        count_p25=2, count_p50=4, count_p75=5, count_p90=7,
        crowd_p50=1, crowd_p90=2,
    ),
    # 205 levels (13.3%), count 1/2/5/8, crowd 1/3, near 36.6%. y mod 24 == 18 in
    # 691/767 -- it sits ON the floor line -- and supported is 0.83, by far the
    # highest of any class. Orientation and mode are 0 in every instance; the
    # entity takes neither (created as `entity_class(type, sim, x, y)`).
    "floor_guard": HazardSpec(
        name="floor_guard",
        types=((16, 100, 12.0, 0.0, 6.0),),
        orientations=(),
        modes=(),
        # 1.0, not the census's 0.83. A floor guard needs a floor: the 0.83 is a
        # measurement artefact of probing official geometry with SOLID_TILES
        # only (slopes and half-tiles read as unsupported), and 691/767 official
        # instances sit exactly on the floor line. Paired with requires_support,
        # which makes it a precondition rather than a target that can be missed.
        supported_fraction=1.0,
        requires_support=True,
        near_route_fraction=0.366,
        count_p25=1, count_p50=2, count_p75=5, count_p90=8,
        crowd_p50=1, crowd_p90=3,
    ),
    # 138 levels (8.9%), count 1/1/2/4, crowd 1/2, near 20.8%. A SINGLE-INSTANCE
    # hazard: p50 = 1 and crowd = 1, so there is no density ladder to walk here
    # and a "turret density" tier would be meaningless. Hitscan, so it needs LOS
    # to the route or it is pure decoration.
    "turret": HazardSpec(
        name="turret",
        types=((19, 100, 12.0, 0.0, 0.0),),
        orientations=(0,),
        modes=(),
        supported_fraction=0.29,
        near_route_fraction=0.208,
        count_p25=1, count_p50=1, count_p75=2, count_p90=4,
        crowd_p50=1, crowd_p90=2,
        needs_los=True,
    ),
    # 134 levels (8.7%), count 2/2/4/8, crowd 1/4, near 14.8%. Orientation IS
    # used (the sim passes it) and official spreads over all eight, though the
    # four axis-aligned dominate 484/540. Mode is 0 or 1.
    "dual_laser": HazardSpec(
        name="dual_laser",
        types=((23, 100, 12.0, 0.0, 0.0),),
        orientations=(0, 2, 4, 6, 1, 3, 5, 7),
        modes=((0, 60), (1, 40)),
        supported_fraction=0.41,
        near_route_fraction=0.148,
        count_p25=2, count_p50=2, count_p75=4, count_p90=8,
        crowd_p50=1, crowd_p90=4,
        needs_los=True,
    ),
    # 295 levels (19.1%), count 1/1/2/3, crowd 1/1, near 30.5%. The second
    # single-instance class, and the most common hazard after patrol drones by
    # level count. Homing, so LOS at spawn is what makes it engage at all.
    "rocket": HazardSpec(
        name="rocket",
        types=((18, 100, 12.0, 0.0, 0.0),),
        orientations=(),
        modes=(),
        supported_fraction=0.21,
        near_route_fraction=0.305,
        count_p25=1, count_p50=1, count_p75=2, count_p90=3,
        crowd_p50=1, crowd_p90=1,
        needs_los=True,
    ),
    # 37 levels (2.4%), count 1/2/3/4, crowd 1/2, near 20.7%. Open volume: it
    # bounces around a chamber, so a corridor mount makes it a wall ornament.
    "death_ball": HazardSpec(
        name="death_ball",
        types=((25, 100, 12.0, 0.0, 0.0),),
        orientations=(),
        modes=(),
        supported_fraction=0.41,
        near_route_fraction=0.207,
        count_p25=1, count_p50=2, count_p75=3, count_p90=4,
        crowd_p50=1, crowd_p90=2,
    ),
    # ZERO instances in the phase-1 official corpus, so nothing here is measured
    # -- these mirror death_ball, which is the closest behavioural analogue (open
    # volume, ninja-seeking). Do not read the numbers as calibration.
    "eye_bat": HazardSpec(
        name="eye_bat",
        types=((27, 100, 12.0, 0.0, 0.0),),
        orientations=(),
        modes=(),
        supported_fraction=0.30,
        near_route_fraction=0.25,
        count_p25=1, count_p50=2, count_p75=3, count_p90=4,
        crowd_p50=1, crowd_p90=2,
        uncalibrated=True,
    ),
    # Thwumps are OUT OF PHASE-1 INGEST SCOPE (`PHASE1_HAZARD_SCOPE` excludes
    # type 20 and 28), so the official corpus holds no instances to calibrate
    # against -- these are reasoned, not measured. They are also the ONLY classes
    # here whose bodies are SOLID, which means two things: the dressed level must
    # be re-routed rather than replayed against the base tape, and they are the
    # only source of a non-zero `solid_map` anywhere in gamma-1 (the official
    # corpus cannot produce one at phase 1).
    "thwump": HazardSpec(
        name="thwump",
        types=((20, 100, 24.0, 12.0, 12.0),),
        orientations=(0, 2, 4, 6),
        modes=(),
        supported_fraction=0.60,
        near_route_fraction=0.35,
        count_p25=1, count_p50=2, count_p75=3, count_p90=5,
        crowd_p50=1, crowd_p90=2,
        is_solid=True,
        lane_tiles=3,
        uncalibrated=True,
    ),
    "shove_thwump": HazardSpec(
        name="shove_thwump",
        types=((28, 100, 24.0, 12.0, 12.0),),
        orientations=(),
        modes=(),
        supported_fraction=0.60,
        near_route_fraction=0.35,
        count_p25=1, count_p50=2, count_p75=3, count_p90=4,
        crowd_p50=1, crowd_p90=2,
        is_solid=True,
        lane_tiles=2,
        uncalibrated=True,
    ),
}


class MapHazardOverlay(Map):
    """Generate a base level from another generator, then overlay one hazard class.

    `HAZARD_CLASSES` may name several: the mixed bucket (`HAZARD_FIELD`) draws
    2-3 of them, because official levels do not serve one hazard at a time and a
    policy that handles each class in isolation has not been shown to handle them
    together. Only 16.0% of official phase-1 levels carry two or more classes, so
    the mix is deliberately shallow.
    """

    # ------------------------------------------------------------------
    # Config (GeneratorFactory upper-cases every dataclass field onto these)
    # ------------------------------------------------------------------
    #: Which `HAZARD_SPECS` entries this preset draws from.
    HAZARD_CLASSES: Sequence[str] = ("patrol_drone",)
    #: How many distinct classes one level carries. >1 is the HAZARD_FIELD mix.
    CLASSES_PER_LEVEL: int = 1

    #: Base morphology. Same registry the mine overlay uses, and deliberately so
    #: -- hazard content is orthogonal to level shape, so variety is free.
    #: `horizontal_corridor` is absent for the reason `map_mine_overlay` records:
    #: it costs 380-480 s per level inside the LatticeDistanceSource precompute,
    #: all of it outside the search.
    BASE_SPECS: Sequence[str] = (
        "jump_platforms:platforms_t2",
        "single_chamber:flat_run_t2",
        "platform_stairs:stairs_t2",
        "slopes:hill_symmetric_t2",
        "multi_jump:staircase_t2",
        "drop_landing:drop_t2",
        "open_field:open_t2",
        "round_corner:corner_t2",
        "loop_corridor:loop_t2",
        "wall_jump:shaft_medium_t2",
        "precision:ledges_t2",
        "curves:two_curves_t2",
        "vertical_corridor:shaft_t2",
        "basic_jump:single_step_up_t2",
    )

    #: Count band, as a multiplier on the class's measured p50. 1.0 reproduces
    #: official's median. The tiers move this BARELY -- see the module docstring
    #: on why ramping count is the wrong axis.
    COUNT_SCALE: float = 1.0
    #: T1 override: a single isolated instance, whatever the class's p50 says.
    #: This is the on-ramp, and it is the difference between a learnable bucket
    #: and one that reads 0.000 forever.
    SINGLE_INSTANCE: bool = False

    #: Share of emitted levels where the naive route DIES -- the levels that
    #: teach timing. Rises with tier. They ship no demo (the route is not a
    #: winning one), so a bucket made entirely of them cannot be BC-seeded.
    TARGET_LETHAL_FRACTION: float = 0.25

    #: Corridor half-width, px, for the near-route test. 72 px = 3 tiles, the band
    #: a jump arc sweeps, and the same one the census measured against.
    CORRIDOR_PX: float = 72.0
    #: Two hazards inside this radius count as one crowd. 144 px = 6 tiles.
    CROWD_RADIUS_PX: float = 144.0

    #: Ceiling on hazards per 1000 REACHABLE lattice cells, per class.
    #:
    #: Official's own density is 0.91/1k at the median and 17.3 at p90 (measured
    #: over 400 packed official levels). The default sits between: dense enough
    #: that a tier can still ramp toward official's hard tail, bounded enough
    #: that a small chamber cannot become 100x official -- which is what DRONE
    #: T4 measured at (p90 = 117.9) before this existed. `GeneratorFactory`
    #: upper-cases config fields onto the generator, so a preset can raise it
    #: per tier; `_hazard_tiers` does exactly that.
    DENSITY_CAP_PER_1K: float = 4.0

    #: Keep hazards off the spawn and off the objectives themselves.
    SPAWN_CLEARANCE_PX: float = 48.0
    OBJECTIVE_CLEARANCE_PX: float = 36.0

    #: Descended only when the level fails to certify. Multiplies the target
    #: count, so it is a genuine thinning rather than a re-roll.
    COUNT_LADDER: Sequence[float] = (1.0, 0.7, 0.45, 0.25, 0.1)
    #: Full-count layout re-draws tried BEFORE any thinning. Which hazards sit on
    #: the route -- not how many there are -- is what separates a safe level from
    #: a lethal one, so re-drawing preserves the count band where thinning
    #: destroys it (measured: patrol drones fell to a median of 3 against
    #: official's 14 when thinning was the only lever).
    LAYOUT_RETRIES: int = 6
    MIN_ACCEPT: int = 1
    MAX_BASE_RETRY: int = 12

    #: Physics gate. Same knobs as the mine overlay; the cap is the real cost
    #: control because it is FAILURES that run to the expansion limit.
    ROUTE_MACRO: int = 6
    ROUTE_H_WEIGHT: float = 2.0
    #: A healthy base routes in 2-5 s; the cost of a pathological one is 380-480 s
    #: inside the lattice precompute, which no in-process cap can interrupt (hence
    #: the child process in `_plan_with_hard_timeout`). 12 s is the mine overlay's
    #: 10 s plus headroom for the extra replay this generator does per rung.
    ROUTE_TIME_CAP_S: float = 12.0
    ROUTE_MAX_EXPANSIONS: int = 60000
    ROUTE_MAX_FRAMES: int = 2600
    REQUIRE_ROUTE_CERTIFICATE: bool = True

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self._base_spec: Optional[str] = None
        self._achieved: Dict[str, int] = {}
        self._achieved_scale: float = 0.0
        #: "safe" (the naive route survives) or "lethal" (it dies to a hazard).
        #: The dataset builder reads this to decide whether to ship a demo, and
        #: the bucket's split of the two is a statistic worth logging.
        self.hazard_outcome: Optional[str] = None
        #: Winning macro tape, and ONLY set when `hazard_outcome == "safe"`. A
        #: lethal level's route is not a winning route, and shipping it as BC
        #: data would teach the policy to walk into the hazard.
        self.route_actions: Optional[List[int]] = None
        self.route_frames: Optional[int] = None
        #: Populated for telemetry: how the certification actually went.
        self.certify_log: List[str] = []

    # ------------------------------------------------------------------
    # map_data helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tile_at(md: Sequence[int], px: float, py: float) -> Optional[int]:
        """Tile id under a pixel position, or None outside the grid.

        Entity/graph pixel coordinates are offset ONE TILE from the tile grid's
        origin -- verified in `map_mine_overlay`, where gold lands in a solid tile
        0.70% of the time with this offset against 29-46% for every alternative.
        """
        tx = int((px - PX_PER_TILE) // PX_PER_TILE)
        ty = int((py - PX_PER_TILE) // PX_PER_TILE)
        if not (0 <= tx < GRID_W and 0 <= ty < GRID_H):
            return None
        return md[TILE_START + ty * GRID_W + tx]

    @classmethod
    def _is_free(cls, md: Sequence[int], px: float, py: float) -> bool:
        return cls._tile_at(md, px, py) == EMPTY_TILE

    @staticmethod
    def _iter_entities(md: Sequence[int]):
        i, n = ENT_START, len(md)
        while i + ENT_STRIDE <= n:
            t = md[i]
            if t == 0:
                break
            yield i, md[i:i + ENT_STRIDE]
            i += ENT_STRIDE

    @classmethod
    def _objective_positions(cls, md: Sequence[int]) -> List[Tuple[float, float]]:
        return [(e[1] * 6.0, e[2] * 6.0)
                for _i, e in cls._iter_entities(md)
                if e[0] in (EXIT_SWITCH_TYPE, EXIT_DOOR_TYPE)]

    # ------------------------------------------------------------------
    # Preconditions
    # ------------------------------------------------------------------
    def _has_support(self, md: Sequence[int], px: float, py: float,
                     probe_px: float = 24.0) -> bool:
        """Is there a solid tile within `probe_px` (one tile) straight below?

        The bound is inclusive of `probe_px` and NOTHING beyond it. The first
        version incremented before testing its own loop condition and so probed
        one extra step, which made the generator's idea of "supported" (30 px)
        disagree with the census's (24 px) -- and a support statistic that is not
        the one official was measured with is not a calibration.
        """
        d = 6.0
        while d <= probe_px + 1e-9:
            t = self._tile_at(md, px, py + d)
            # SOLID_TILES, not "any non-empty tile". The census measures official
            # support with the fully-solid set, and calibrating against a wider
            # definition would be calibrating against a different statistic --
            # slope and curve tiles would count here and not there.
            if t is not None and t in SOLID_TILES:
                return True
            d += 6.0
        return False

    def _seat_on_surface(self, md: Sequence[int], px: float, py: float,
                         lattice: float, offset: float,
                         max_drop_px: float = 120.0
                         ) -> Optional[Tuple[float, float]]:
        """Drop straight down onto the lowest FREE cell of the type's OWN lattice
        that still sits above the first surface below `(px, py)`.

        This exists because the reachability graph cannot supply ground support on
        its own. Its nodes are overwhelmingly airborne traversal states -- measured
        here, 3 of 450 zap-lattice cells and 31 of 1749 mini-lattice cells are
        grounded, ~2% against official's 27% for patrol drones and 83% for floor
        guards. The mine overlay hit the identical wall (1.9% supported against a
        33.9% target) and solved it the same way: drop the entity onto the floor
        it is already above.

        Two things this must not do, both of which the first version did and both
        of which read as "supported = 0.00" rather than as an error:

        * land INSIDE the surface. `_is_free` then rejects the candidate, so
          every would-be grounded hazard is silently filtered out and the emitted
          field is entirely airborne -- the phantom-mine failure, inverted.
        * ignore the type's lattice OFFSET. Snapping to a bare multiple of the
          grid puts a zap drone at y = 0 mod 24 when its cell centres are at 12,
          and a drone off its own lattice starts mid-step between cells.
        """
        grid = max(6.0, float(lattice))
        d = 6.0
        surface = None
        while d <= max_drop_px:
            t = self._tile_at(md, px, py + d)
            if t is None:
                return None
            # SOLID_TILES, matching `_has_support`. Seating onto a slope or curve
            # tile instead would put the entity somewhere `_has_support` then
            # calls UNSUPPORTED, so the grounded pool would fill with cells that
            # do not count as grounded by the statistic being calibrated.
            if t in SOLID_TILES:
                surface = py + d
                break
            if t != EMPTY_TILE:
                # Partially-solid tile: not a seat, and not passable either.
                return None
            d += 6.0
        if surface is None:
            return None
        # Highest lattice value strictly below the surface, on the type's phase.
        seat = math.floor((surface - 1e-6 - offset) / grid) * grid + offset
        if seat <= py or not self._is_free(md, px, seat):
            return None
        return (px, seat)

    def _clear_lane(self, md: Sequence[int], px: float, py: float,
                    orientation: int, tiles: int) -> bool:
        """Free travel lane of `tiles` ahead, along `orientation`.

        A thwump mounted against a wall it immediately charges into never moves,
        which is the solid-body equivalent of a mine buried in rock: it costs
        observation bandwidth and teaches nothing.
        """
        dx, dy = _ORIENT_VECTOR.get(orientation, (1.0, 0.0))
        for k in range(1, tiles + 1):
            if not self._is_free(md, px + dx * k * PX_PER_TILE,
                                 py + dy * k * PX_PER_TILE):
                return False
        return True

    def _has_los(self, md: Sequence[int], px: float, py: float,
                 targets: Sequence[Tuple[float, float]], need: int = 1) -> bool:
        """Unobstructed sight line to at least `need` of `targets`.

        Turrets are hitscan and rockets home, so both are pure decoration without
        it. Sampled every 6 px along the segment, which is the sim's own
        sub-tile resolution.
        """
        hits = 0
        for tx, ty in targets:
            dist = math.hypot(tx - px, ty - py)
            if dist < 1e-6:
                continue
            steps = max(1, int(dist // 6.0))
            blocked = False
            for k in range(1, steps):
                t = k / steps
                if not self._is_free(md, px + (tx - px) * t, py + (ty - py) * t):
                    blocked = True
                    break
            if not blocked:
                hits += 1
                if hits >= need:
                    return True
        return False

    # ------------------------------------------------------------------
    # Anchors
    # ------------------------------------------------------------------
    def _reachable_nodes(self, md: Sequence[int]) -> List[Tuple[float, float]]:
        """Positions the ninja can actually occupy.

        Anchoring on the reachability graph rather than on any free tile keeps
        hazards near surfaces the agent will traverse, so they are threats on the
        route rather than decoration in an unreachable void. The graph's node
        keys are in the SAME pixel frame as entity coordinates and the spawn, so
        no conversion is needed.

        Computed once per base and snapped per TYPE by `_anchors_for` -- the
        reachability build is the expensive half and the snap is arithmetic.
        """
        from .solvability import _build_level_data, _reachable_set

        try:
            level_data = _build_level_data(list(md))
            reachable, _adj = _reachable_set(level_data)
        except Exception as exc:  # noqa: BLE001
            logger.debug("hazard_overlay: reachability failed: %s", exc)
            return []
        return [(float(n[0]), float(n[1])) for n in reachable
                if isinstance(n, tuple) and len(n) >= 2]

    def _anchors_for(self, md: Sequence[int],
                     nodes: Sequence[Tuple[float, float]],
                     grid: float, ox: float, oy: float
                     ) -> Tuple[List[Tuple[float, float]],
                                List[Tuple[float, float]]]:
        """Free lattice cells for one entity type, split `(grounded, airborne)`.

        Snapping a reachable node to a lattice can land it INSIDE a solid tile --
        reachability nodes hug surfaces, and the nearest lattice cell to a node
        standing on a floor is frequently the floor. Filtering to free cells HERE
        rather than at sampling time is what makes the grounded/airborne split
        meaningful: an earlier version filtered afterwards and thereby discarded
        every grounded candidate, emitting a 100% airborne field against
        official's 27-83% supported. That is the phantom-mine failure with the
        sign flipped, and it is just as invisible.

        The grounded pool is then EXTENDED by dropping every snapped cell onto
        the surface beneath it, because the graph alone cannot supply one: only
        ~2% of its cells are grounded, against 27% for patrol drones and 83% for
        floor guards. Doing the drop here rather than per draw is what makes the
        pool diverse -- hundreds of airborne cells in one column seat onto the
        same floor cell, so seating at sample time spends the whole attempt
        budget rediscovering a handful of positions.
        """
        snapped: Set[Tuple[float, float]] = set()
        for nx, ny in nodes:
            snapped.add((round((nx - ox) / grid) * grid + ox,
                         round((ny - oy) / grid) * grid + oy))
        grounded: Set[Tuple[float, float]] = set()
        airborne: List[Tuple[float, float]] = []
        for px, py in sorted(snapped):
            if self._is_free(md, px, py):
                if self._has_support(md, px, py):
                    grounded.add((px, py))
                else:
                    airborne.append((px, py))
            seat = self._seat_on_surface(md, px, py, grid, oy)
            if seat is not None and self._has_support(md, seat[0], seat[1]):
                # Confirmed, not assumed: the lattice phase can put the seat more
                # than one tile above the surface it was computed from, and a
                # cell that `_has_support` calls airborne must not enter the
                # grounded pool -- that is the statistic being calibrated.
                grounded.add(seat)
        return sorted(grounded), airborne

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _target_count(self, rng: random.Random, spec: HazardSpec,
                      n_nodes: Optional[int] = None) -> int:
        """Target instance count, capped by the level's own REACHABLE AREA.

        THE COUNT BANDS ARE CONDITIONAL AND THE CAP IS WHAT MAKES THEM SAFE.
        `count_p25..p75` are measured over official levels that CONTAIN the class
        (patrol drone p50 = 14). Applying a conditional count to EVERY generated
        level, in a chamber that is not the size of the official level it was
        measured from, does not reproduce official density -- it multiplies it.

        Measured consequence before this cap existed, in hazards per 1000
        reachable cells (median): OFFICIAL 0.91, DRONE T2 **4.07** (4.5x),
        T3 6.24 (6.9x), T4 6.98 (7.7x) with a p90 of **117.9**. Every other
        bucket sat at 0.4-1.2, i.e. at or below official -- the explosion is
        specific to the classes with large conditional counts, which is exactly
        what a conditional-count-applied-unconditionally bug looks like.

        So the count band still sets the SHAPE and the tier still ramps it, but
        a level can never carry more hazards than its standable space supports.
        """
        if self.SINGLE_INSTANCE:
            return 1
        # Draw between p25 and p75 rather than fixing p50: the official spread is
        # part of the distribution and a bucket at exactly the median is not the
        # same distribution as one whose median is the median.
        lo, hi = spec.count_p25, spec.count_p75
        base = rng.randint(min(lo, hi), max(lo, hi))
        want = max(1, int(round(base * float(self.COUNT_SCALE))))
        if n_nodes and self.DENSITY_CAP_PER_1K > 0:
            cap = int(round(self.DENSITY_CAP_PER_1K * float(n_nodes) / 1000.0))
            want = max(1, min(want, cap))
        return want

    @staticmethod
    def _standable_cells(nodes) -> int:
        """Distinct 12 px CELLS the reachability graph covers.

        THE UNIT MATTERS, AND GETTING IT WRONG MAKES THE CAP INERT. The graph
        carries several NODES per cell -- measured 8,540 nodes over 2,196 cells
        and 15,368 over 3,848, about 3.9x -- because its nodes are traversal
        STATES, not positions. `DENSITY_CAP_PER_1K` is calibrated against
        official in 12 px cells, so passing raw node counts inflates the ceiling
        ~4x and it stops binding: a cap of 2.0 became 17-31 hazards against a
        draw band of 8-28, i.e. no constraint at all.

        That is exactly what happened on the first attempt here, and an 8-seed
        spot check was too underpowered to notice -- the regenerated DRONE T2
        came out at 5.21/1k, WORSE than the 4.07 the cap was meant to fix.
        """
        return len({(int(x // 12.0), int(y // 12.0)) for x, y in nodes})

    def _sample_class(
        self,
        rng: random.Random,
        md: Sequence[int],
        spec: HazardSpec,
        nodes: Sequence[Tuple[float, float]],
        route_pts: Sequence[Tuple[float, float]],
        n_want: int,
    ) -> List[Tuple[float, float, int, int, int]]:
        """`(px, py, type, orientation, mode)` placements honouring the spec.

        Two mixes are steered on the ACCEPTED placements rather than on the
        draws, which is not the same thing and matters here: classes with a hard
        precondition (LOS for turrets and rockets, a clear lane for thwumps)
        reject candidates at very different rates in different parts of a level,
        so a 29% draw rate produced a 6% accept rate the first time. Each
        acceptance re-reads what is still needed.

        * **near-route.** The census says only 15-37% of official hazards sit
          within three tiles of the route -- placing them all on the corridor,
          the obvious reading of "put it where it threatens the ninja", builds a
          bucket far harsher than the distribution it imitates.
        * **ground support.** 27% for patrol drones, 83% for floor guards, which
          barely move at all: a floor guard needs a floor to charge along.
        """
        spawn = (md[SPAWN_X_IDX] * 6.0, md[SPAWN_Y_IDX] * 6.0)
        objectives = self._objective_positions(md)
        sp_c2 = float(self.SPAWN_CLEARANCE_PX) ** 2
        ob_c2 = float(self.OBJECTIVE_CLEARANCE_PX) ** 2
        corridor2 = float(self.CORRIDOR_PX) ** 2
        crowd_r2 = float(self.CROWD_RADIUS_PX) ** 2

        def near_route(px: float, py: float) -> bool:
            return any((px - rx) ** 2 + (py - ry) ** 2 <= corridor2
                       for rx, ry in route_pts)

        # Anchors per TYPE, split on the FULL cross product of the two mixes we
        # steer. Steering them independently does not work: they fight. Grounded
        # cells sit on floors and the route runs along floors, so "grounded" and
        # "near the route" are strongly correlated -- asking for grounded-and-far
        # by rejecting mismatches burned the attempt budget and emitted 0.02
        # support against a 0.27 target. Selecting from the joint pool directly,
        # with an explicit relaxation order when a cell of the pool does not
        # exist on this level, is what makes both marginals reachable.
        pools: Dict[int, Dict[Tuple[bool, bool], List[Tuple[float, float]]]] = {}
        for etype, _w, grid, ox, oy in spec.types:
            grounded, airborne = self._anchors_for(md, nodes, grid, ox, oy)
            if not (grounded or airborne):
                continue
            split: Dict[Tuple[bool, bool], List[Tuple[float, float]]] = {}
            for is_ground, cells in ((True, grounded), (False, airborne)):
                for px, py in cells:
                    is_near = bool(route_pts) and near_route(px, py)
                    split.setdefault((is_ground, is_near), []).append((px, py))
            pools[etype] = split
        if not pools:
            return []

        def pick(split, want_ground: bool, want_near: bool):
            """Nearest available pool. Relax NEAR before GROUND: the near-route
            fraction is a distributional target, while ground support is a
            precondition for the entity to function at all (a floor guard needs
            a floor to charge along)."""
            for key in ((want_ground, want_near),
                        (want_ground, not want_near),
                        (not want_ground, want_near),
                        (not want_ground, not want_near)):
                if split.get(key):
                    return split[key]
            return None

        types = [t for t, _w, _g, _x, _y in spec.types if t in pools]
        tweights = [w for t, w, _g, _x, _y in spec.types if t in pools]
        modes = [m for m, _w in spec.modes]
        mweights = [w for _m, w in spec.modes]

        out: List[Tuple[float, float, int, int, int]] = []
        used: Set[Tuple[float, float]] = set()
        n_near = 0
        n_ground = 0
        # Generous attempt budget: most rejections are cheap predicate failures,
        # and a class with a hard precondition can reject a long run of
        # candidates before finding a legal mount.
        for _ in range(max(400, n_want * 120)):
            if len(out) >= n_want:
                break
            etype = rng.choices(types, weights=tweights, k=1)[0]
            # Steer toward the deficit, with randomized rounding. A bare
            # `n_near < f * (len(out)+1)` makes the FIRST placement always want
            # near-route, which for a single-instance class like rocket (official
            # p50 = 1) turns a 30% target into ~100%. The `rng.random()` term
            # keeps the marginal correct at every count, including one.
            want_near = (n_near + rng.random()) < (
                spec.near_route_fraction * (len(out) + 1))
            # Steered on ACCEPTANCES, exactly like `want_near` above and for the
            # same reason. A per-attempt Bernoulli under-delivers whenever the
            # two pools accept at different rates, and they do: the grounded pool
            # is one or two orders of magnitude smaller (35 and 76 cells against
            # 447 and 1737 on a measured base), so it dedups out far faster and a
            # 0.27 draw rate emitted 0.11.
            want_ground = (n_ground + rng.random()) < (
                spec.supported_fraction * (len(out) + 1))
            pool = pick(pools[etype], want_ground, want_near)
            if pool is None:
                continue
            px, py = pool[rng.randrange(len(pool))]

            if (px, py) in used:
                continue
            if (px - spawn[0]) ** 2 + (py - spawn[1]) ** 2 < sp_c2:
                continue
            if any((px - ox) ** 2 + (py - oy) ** 2 < ob_c2 for ox, oy in objectives):
                continue
            if spec.needs_los and route_pts and not self._has_los(
                    md, px, py, route_pts, need=1):
                continue
            # Hard mounting precondition. Unlike `supported_fraction`, which
            # steers a mix and is allowed to miss, this REJECTS: a floor guard
            # charges along a floor and airborne it has no mechanic at all.
            # Measured before this existed: 32% grounded at FLOOR_GUARD/T1.
            if spec.requires_support and not self._has_support(md, px, py):
                continue

            orientation = rng.choice(spec.orientations) if spec.orientations else 0
            if spec.lane_tiles:
                lanes = list(spec.orientations) or [0, 2, 4, 6]
                rng.shuffle(lanes)
                orientation = next(
                    (o for o in lanes
                     if self._clear_lane(md, px, py, o, spec.lane_tiles)),
                    None,
                )
                if orientation is None:
                    continue
            mode = rng.choices(modes, weights=mweights, k=1)[0] if modes else 0

            # Crowd budget. Official's crowd p90 is the ceiling; exceeding it
            # produces the blob artefact the mine overlay had to fix (the right
            # COUNT squeezed into the wrong area).
            crowd = sum(1 for ox, oy, _t, _o, _m in out
                        if (px - ox) ** 2 + (py - oy) ** 2 < crowd_r2)
            if crowd >= spec.crowd_p90:
                continue

            used.add((px, py))
            out.append((px, py, etype, orientation, mode))
            n_near += int(bool(route_pts) and near_route(px, py))
            n_ground += int(self._has_support(md, px, py))
        return out

    # ------------------------------------------------------------------
    # Base construction and routing
    # ------------------------------------------------------------------
    def _build_base(self, spec: str, seed: int) -> Optional[List[int]]:
        from .generator_factory import GeneratorFactory

        gen_type, preset = spec.split(":", 1)
        try:
            gen = GeneratorFactory.create_from_preset(gen_type, preset, seed)
            gen.generate(seed=seed)
            md = list(gen.map_data())
        except Exception as exc:  # noqa: BLE001
            logger.debug("hazard_overlay: base %s failed: %s", spec, exc)
            return None
        if md[EXIT_DOOR_COUNT_IDX] == 0:
            return None
        if md[SPAWN_X_IDX] == 0 and md[SPAWN_Y_IDX] == 0:
            return None
        return md

    def _to_tmp_map(self, md: Sequence[int]) -> str:
        import os
        import tempfile

        from .compose.mapio import save_map

        blob = list(md)
        if blob[1233] == -1:
            # map_data carries the ninja orientation as -1, out of byte range;
            # the on-disk encoding is 255.
            blob[1233] = 255
        fd, path = tempfile.mkstemp(suffix=".map")
        os.close(fd)
        save_map(blob, path, fmt="binary")
        return path

    def _route(self, md: Sequence[int]) -> Optional[dict]:
        """Plan a winning route. Returns None for BOTH timeout and no-route --
        `certify_log` records which, because they must not be conflated: thinning
        on a timeout is exactly how a density generator biases itself sparse."""
        import os

        path = None
        try:
            path = self._to_tmp_map(md)
            res = _plan_with_hard_timeout(
                path,
                macro=int(self.ROUTE_MACRO),
                h_weight=float(self.ROUTE_H_WEIGHT),
                max_expansions=int(self.ROUTE_MAX_EXPANSIONS),
                max_frames=int(self.ROUTE_MAX_FRAMES),
                timeout_s=float(self.ROUTE_TIME_CAP_S),
            )
        except Exception as exc:  # noqa: BLE001
            self.certify_log.append(f"route_exception:{type(exc).__name__}")
            return None
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        if res is None:
            self.certify_log.append("route_timeout")
            logger.warning(
                "hazard_overlay: planner TIMED OUT after %.0fs -- this says "
                "nothing about whether the level is winnable, so the count "
                "ladder must NOT thin on it", float(self.ROUTE_TIME_CAP_S))
            return None
        if not res.get("solved"):
            if res.get("reason") == "exception":
                self.certify_log.append("route_crash")
                logger.warning("hazard_overlay: planner raised: %s",
                               res.get("error"))
            else:
                self.certify_log.append("route_unsolved")
            return None

        tape = list(res.get("frame_tape") or [])
        if not tape:
            # `frame_tape` is only attached when the planner took a clinch leg.
            # Treating its absence as "no route" silently threw away solved bases
            # in the mine overlay -- rebuild it from the macro actions, which are
            # always returned.
            from ..value_function.tools.dstar_guided_spike import _tape_from_actions

            tape = list(_tape_from_actions(res.get("actions") or [],
                                           int(self.ROUTE_MACRO)))
        if not tape:
            self.certify_log.append("route_no_tape")
            return None
        res["frame_tape"] = tape
        return res

    def _replay(self, md: Sequence[int], tape
                ) -> Tuple[Optional[bool], Optional[List[Tuple[float, float]]]]:
        """Replay `tape` on a fresh sim. Returns `(won, path_points)`.

        `won is None` means the replay itself failed to run, which is a different
        thing from losing and must not be read as "the hazard killed it".
        """
        import os

        from ..value_function.tools.dstar_guided_spike import _replay_frames_native

        path = None
        try:
            path = self._to_tmp_map(md)
            pre, won, _gold, _gt, _died = _replay_frames_native(path, tape)
        except Exception as exc:  # noqa: BLE001
            logger.debug("hazard_overlay: replay failed: %s", exc)
            return None, None
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        return bool(won), [(float(s[0]), float(s[1])) for s in pre]

    # ------------------------------------------------------------------
    def _assert_spawn_clearance(
        self, md: Sequence[int],
        placements: Sequence[Tuple[float, float, int, int, int]]) -> None:
        """Hard post-condition: no hazard within `SPAWN_CLEARANCE_PX` of spawn.

        `_sample_class` already rejects such candidates, but that is a filter
        inside the sampler and nothing downstream re-checks it -- a new placement
        path, a relaxed pool or a future thinning rung could reintroduce a
        point-blank hazard and the only symptom would be an unlearnable bucket
        that reads 0.000 forever, which is indistinguishable from an encoding
        bug. Checking at the single commit point makes it structural.

        A drone, rocket or turret spawned on top of the ninja is the worst case:
        the episode is decided before the policy observes anything, so the level
        teaches nothing and still counts as a death. This RAISES rather than
        filtering -- a violation means the sampler's guard is broken, and
        silently dropping the hazard would hide that.
        """
        sx, sy = md[SPAWN_X_IDX] * 6.0, md[SPAWN_Y_IDX] * 6.0
        limit2 = float(self.SPAWN_CLEARANCE_PX) ** 2
        bad = [(px, py, etype, math.hypot(px - sx, py - sy))
               for px, py, etype, _o, _m in placements
               if (px - sx) ** 2 + (py - sy) ** 2 < limit2]
        if bad:
            detail = ", ".join(
                f"type {t} at ({px:.1f},{py:.1f}) d={d:.1f}px"
                for px, py, t, d in bad[:5])
            raise AssertionError(
                f"{len(bad)} hazard(s) inside the {self.SPAWN_CLEARANCE_PX:.0f}px "
                f"spawn clearance at ({sx:.1f},{sy:.1f}): {detail}"
            )

    def _adopt(self, md: Sequence[int],
               placements: Sequence[Tuple[float, float, int, int, int]]) -> None:
        self._assert_spawn_clearance(md, placements)
        base = Map.from_map_data(list(md))
        self.tile_data = base.tile_data
        self.entity_data = base.entity_data
        self.entity_counts = base.entity_counts
        self.ninja_spawn_x = base.ninja_spawn_x
        self.ninja_spawn_y = base.ninja_spawn_y
        self.ninja_orientation = base.ninja_orientation
        for px, py, etype, orientation, mode in placements:
            # add_entity takes GRID coordinates and scales by 4 to map units, so
            # a pixel position divides by 24. Sub-tile values are intended: the
            # 12 px lattice is half-tile and official uses it.
            self.add_entity(etype, px / PX_PER_TILE, py / PX_PER_TILE,
                            orientation, mode)

    @staticmethod
    def _dress(md: Sequence[int],
               placements: Sequence[Tuple[float, float, int, int, int]]
               ) -> List[int]:
        trial = Map.from_map_data(list(md))
        for px, py, etype, orientation, mode in placements:
            trial.add_entity(etype, px / PX_PER_TILE, py / PX_PER_TILE,
                             orientation, mode)
        return list(trial.map_data())

    # ------------------------------------------------------------------
    def generate(self, seed: Optional[int] = None) -> "MapHazardOverlay":
        if seed is not None:
            self.rng = random.Random(seed)
        rng = self.rng
        from .solvability import is_map_solvable

        base_seed = seed if seed is not None else rng.randrange(1 << 30)
        specs = list(self.BASE_SPECS)
        classes = [HAZARD_SPECS[c] for c in self.HAZARD_CLASSES
                   if c in HAZARD_SPECS]
        if not classes:
            raise ValueError(
                f"hazard_overlay: no known class in {list(self.HAZARD_CLASSES)}; "
                f"known: {sorted(HAZARD_SPECS)}"
            )
        want_lethal = rng.random() < float(self.TARGET_LETHAL_FRACTION)
        any_solid = any(c.is_solid for c in classes)

        for attempt in range(int(self.MAX_BASE_RETRY)):
            s = base_seed + attempt * 1_000_000
            spec_name = specs[(base_seed + attempt) % len(specs)]
            md = self._build_base(spec_name, s)
            if md is None:
                continue
            if not is_map_solvable(md):
                # A base that is already unsolvable can never be rescued by
                # adding hazards, and would burn the whole ladder proving it.
                continue

            # THE CLEAN BASE IS ROUTED ONCE. Pure hazards do not bend the ninja's
            # trajectory, so the reachability graph and the cost-to-go field are
            # already correct on the dressed level and re-routing every rung
            # would pay 2-5 s to rediscover the same tape. Solid classes are the
            # exception and re-route below.
            route = None
            route_pts: List[Tuple[float, float]] = []
            if self.REQUIRE_ROUTE_CERTIFICATE:
                route = self._route(md)
                if route is None:
                    continue
                won, pts = self._replay(md, route["frame_tape"])
                if not won or not pts:
                    self.certify_log.append("base_tape_does_not_replay")
                    continue
                route_pts = pts

            arng = random.Random(s)
            chosen = list(classes)
            if len(chosen) > int(self.CLASSES_PER_LEVEL):
                chosen = arng.sample(chosen, int(self.CLASSES_PER_LEVEL))

            nodes = self._reachable_nodes(md)
            if not nodes:
                continue

            # Candidate layouts, in the order they are tried. FULL-COUNT RE-DRAWS
            # COME FIRST, and that ordering is the whole point: thinning is the
            # only lever the mine overlay had, but here most rungs are descended
            # to satisfy the *role* (safe vs lethal), not because the level is
            # unroutable. Thinning to make a level safe collapses the count --
            # measured, patrol drones emitted a median of 3 against official's 14
            # -- when the real difference between a safe and a lethal level is
            # WHICH hazards sit on the route, not how many there are. So re-draw
            # the layout at full count first, and thin only if no draw works.
            def _layouts():
                # PASS 1: every full-count re-draw. Thinning is tried only after
                # all of these fail, because the ladder's job is to rescue an
                # over-dense level and NOT to satisfy the safe/lethal role --
                # descending it for the role is what collapsed the count (patrol
                # drones fell to a median of 4 against official's 14).
                drawn = []
                for _draw in range(max(1, int(self.LAYOUT_RETRIES))):
                    full: List[Tuple[float, float, int, int, int]] = []
                    counts: Dict[str, int] = {}
                    for cls_spec in chosen:
                        got = self._sample_class(
                            arng, md, cls_spec, nodes, route_pts,
                            self._target_count(
                                arng, cls_spec,
                                self._standable_cells(nodes)))
                        counts[cls_spec.name] = len(got)
                        full.extend(got)
                    if len(full) < int(self.MIN_ACCEPT):
                        continue
                    drawn.append((full, counts))
                    yield full, counts, 1.0
                # PASS 2: thin the FIRST draw down the ladder. Sample once and
                # thin that same set -- re-rolling per rung made the mine
                # overlay's count non-monotone (measured 37/22/12/42/27).
                if not drawn:
                    return
                full, counts = drawn[0]
                order = list(range(len(full)))
                arng.shuffle(order)
                for scale in self.COUNT_LADDER[1:]:
                    keep = max(0, int(round(len(full) * scale)))
                    thinned = [full[i] for i in sorted(order[:keep])]
                    if len(thinned) >= int(self.MIN_ACCEPT):
                        yield thinned, counts, scale

            fallback = None
            per_class: Dict[str, int] = {}
            for placements, per_class, scale in _layouts():
                trial = self._dress(md, placements)

                if not self.REQUIRE_ROUTE_CERTIFICATE:
                    outcome, actions, frames = "safe", None, None
                elif any_solid:
                    # Solid bodies CHANGE GEOMETRY, so the base tape is not a
                    # route on this level and the dressed level must be planned
                    # from scratch. A failure here is a genuine "geometry
                    # broken" and the ladder should thin.
                    dressed = self._route(trial)
                    if dressed is None:
                        continue
                    won, _pts = self._replay(trial, dressed["frame_tape"])
                    if won is None:
                        continue
                    outcome = "safe" if won else "lethal"
                    actions = list(dressed.get("actions") or [])
                    frames = dressed.get("frames")
                else:
                    # Pure hazards: geometry is unchanged, so the base tape IS a
                    # geometric route. Whether it SURVIVES is the discriminator.
                    won, _pts = self._replay(trial, route["frame_tape"])
                    if won is None:
                        # The replay could not be run at all -- not evidence
                        # that the hazards killed the route.
                        self.certify_log.append("replay_error")
                        continue
                    outcome = "safe" if won else "lethal"
                    actions = list(route.get("actions") or [])
                    frames = route.get("frames")

                if want_lethal != (outcome == "lethal"):
                    # Right level, wrong role. Hold it as the fallback and keep
                    # descending -- thinning tends to convert lethal -> safe, so
                    # a wanted-safe draw usually resolves one or two rungs down.
                    # Never DISCARD it: rejecting every off-role level would bias
                    # the bucket toward hazards that never threaten the route,
                    # which is the exact failure this whole design avoids.
                    # Keep the FIRST, which is the full-count draw: if no layout
                    # ever lands on the wanted role, the emitted level should at
                    # least carry the right hazard COUNT.
                    if fallback is None:
                        fallback = (placements, outcome, actions, frames, scale,
                                    dict(per_class))
                    continue

                self._commit(md, placements, outcome, actions, frames,
                             spec_name, scale, per_class)
                return self

            if fallback is not None:
                placements, outcome, actions, frames, scale, counts = fallback
                self.certify_log.append(f"role_fallback:{outcome}")
                self._commit(md, placements, outcome, actions, frames,
                             spec_name, scale, counts)
                return self

        raise RuntimeError(
            f"hazard_overlay: no certified level after {self.MAX_BASE_RETRY} base "
            f"attempts (seed={base_seed}, classes={list(self.HAZARD_CLASSES)}); "
            f"certify_log={self.certify_log[-8:]}"
        )

    def _commit(self, md, placements, outcome, actions, frames,
                spec_name, scale, per_class) -> None:
        del per_class  # counted from what was ADOPTED, not from the draw
        self._adopt(md, placements)
        self._base_spec = spec_name
        self._achieved_scale = float(scale)
        # Recount from `placements`. Reporting the draw instead was wrong by up
        # to the whole ladder: a thinned level reported 5 hazards and emitted 1,
        # so any audit reading `_achieved` would certify a density the emitted
        # bucket does not have -- which is exactly the mistake this generator
        # exists to stop making.
        by_type: Dict[int, str] = {
            t: spec.name
            for spec in HAZARD_SPECS.values()
            for t, *_rest in spec.types
        }
        counts: Dict[str, int] = {}
        for _px, _py, etype, _o, _m in placements:
            key = by_type.get(etype, f"type_{etype}")
            counts[key] = counts.get(key, 0) + 1
        self._achieved = counts
        self.hazard_outcome = outcome
        if outcome == "safe":
            self.route_actions = list(actions or []) or None
            self.route_frames = frames
        else:
            # A lethal level's route is not a winning route. Shipping it as BC
            # data would teach the policy to walk into the hazard.
            self.route_actions = None
            self.route_frames = None


#: `physics.map_orientation_to_vector`, inlined so placement does not import the
#: simulator. 0 is (+1, 0) -- NOT "up", which is the misconception that made
#: `map_augmentation.transforms._ORIENT_FLIP` wrong for the axis-aligned cases.
_DIAG = math.sqrt(2) / 2
_ORIENT_VECTOR: Dict[int, Tuple[float, float]] = {
    0: (1.0, 0.0),
    1: (_DIAG, _DIAG),
    2: (0.0, 1.0),
    3: (-_DIAG, _DIAG),
    4: (-1.0, 0.0),
    5: (-_DIAG, -_DIAG),
    6: (0.0, -1.0),
    7: (_DIAG, -_DIAG),
}
