"""Cavern/chamber generator: multi-area levels with rock-tunnel connections.

The gap this closes: our procgen corpus is dominated by *single-area* levels (one
corridor, one chamber, one shaft), while OFFICIAL N++ levels are typically several
distinct **areas** the ninja traverses between on the way to the switch and the
door. This generator produces that macro-structure directly.

Pipeline
--------
1. **Partition.** A jittered BSP splits the playable rect into leaves; each leaf
   gets one chamber rectangle inset inside it with a 1-2 tile random margin. The
   margins guarantee >=2 tiles of solid rock between any two chambers, which is
   exactly the width the corridor router needs to squeeze between them.
2. **Carve.** Each chamber is carved with one of four styles -- ``box``,
   ``terraced`` (stepped floor + 45 degree ramps), ``pillared`` (floor pillars),
   ``cavern`` (cellular-automata rock hanging from the ceiling/walls). Every
   style upholds one invariant: a **walk band** of 2 empty rows sits above the
   floor across the chamber's full width, so a chamber is always traversable
   left-to-right at floor level.
3. **Connect.** Chamber centres are linked by a random spanning tree (guaranteed
   connectivity) plus ``EXTRA_EDGES`` extra links, which are what create cycles
   -- i.e. genuinely *nonlinear* topologies with more than one viable route.
4. **Route.** Each link is carved by a BFS over 2x2 blocks of *solid rock only*
   (chamber interiors are hard obstacles). Because every carved passage is 2
   tiles wide, horizontal legs are 2-tall corridors with rock underfoot and
   vertical legs are 2-wide shafts with rock on both faces -- the canonical
   wall-jumpable width (see ``map_loop_corridor``: 3-wide shafts are not
   chainable and leave levels unsolvable). Ports always attach at a chamber's
   *floor* level on its left/right edge, so all climbing happens in shafts that
   are wall-jumpable by construction.
5. **Smooth.** A tile-variety pass rewrites exposed rock into 45 degree bevels,
   quarter moons, quarter pipes and half tiles. This is the point of the whole
   exercise for geometry variety: it only ever *removes* material (every
   replacement's solid region is a subset of the full tile), so it can never
   sever a route. Corridor cells and their surrounding walls are excluded so
   shaft faces stay flat and wall-jumpable.
6. **Place.** Spawn / exit switch / exit door are placed on chamber floors, and
   which chambers they land in is chosen from graph distance to produce either a
   **linear** route (spawn -> switch -> door strictly outbound) or a
   **nonlinear** one (switch at the far end, door back near spawn, forcing an
   out-and-back through the chamber graph).

7. **Populate.** Toggle mines stuck to floors, ceilings and wall faces; optional
   traversal furniture (one-way platforms, bounce blocks, launch pads, boost
   pads); and shaped gold clusters. Anything that can obstruct is proved harmless
   by a flood fill that treats it as a solid wall -- see ``_drop_severing``.

Entity scope, and why it is split three ways
--------------------------------------------
* **Mines and traversal furniture are placed HERE**, because placement depends on
  geometry this generator alone knows -- which cells are chamber floor, which are
  a wall-jump shaft face, which ledge is a corridor mouth.
* **Pure hazards (drones, floor guards, turrets, rockets, death balls, thwumps)
  are NOT.** ``map_hazard_overlay`` already dresses an arbitrary base level with
  them, using per-class mounting preconditions and orientation/mode distributions
  measured off the official corpus, and it certifies the result by replaying a
  routed tape. Reimplementing that here would be strictly worse. Use the
  ``hazard_overlay:cavern_hazard_t*`` presets, which pin this generator as the
  base.
* **Traversal types move the agent's training scope**, unlike hazards. Per
  ``constants.entity_types``, ONE_WAY is phase 2 -- the reachability graph
  already models it, so the cost-to-go field stays correct -- but BOUNCE_BLOCK
  and LAUNCH_PAD are phase 3, the step that changes that field and can invalidate
  PBRS, the demo corpus and the reverse curriculum at once. All of it is off by
  default so a preset has to opt in deliberately.

Config-driven class attributes (patched by ``GeneratorFactory`` from
``CavernChambersConfig``): MIN/MAX_CHAMBERS, MIN/MAX_CHAMBER_WIDTH,
MIN/MAX_CHAMBER_HEIGHT, FILL_BIAS, EXTRA_EDGES, STYLE_WEIGHTS, MAX_FLOOR_STEP,
CAVERN_FILL, WALL_ROUGHNESS, SMOOTHING_RATE, FLAT_SMOOTHING_RATE,
NONLINEAR_PROB, LAYOUT_ATTEMPTS, MIN/MAX_MINES, ARMED_MINE_PROB,
MINE_SURFACE_WEIGHTS, MINE_MIN_SPACING, MINE_IN_CORRIDORS, MINE_GOAL_CLEARANCE,
MIN/MAX_ONE_WAYS, MIN/MAX_BOUNCE_BLOCKS, MIN/MAX_LAUNCH_PADS,
MIN/MAX_BOOST_PADS, TRAVERSAL_MIN_SPACING, MIN/MAX_GOLD_CLUSTERS,
MIN/MAX_GOLD_PER_CLUSTER, GOLD_SHAPE_WEIGHTS, GOLD_HALF_PITCH_PROB.

Measured against the 3.5k-level official corpus, the default tiers land at a
tile-histogram entropy of ~1.1-1.2 bits (official: 1.34), a shaped-tile share of
~2.5% (official: 4.1%) and 30 of the 34 usable tile types -- versus the pure
0/1 palette the older chamber generators emit. The four types not reachable are
the steep *ceiling* pairs (26/27/30/31), which have no continuous two-tile
mirror in the collision model.
"""

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


EMPTY = 0
SOLID = 1

# Rock reserved between a BSP leaf's edge and its chamber. Two on each side
# leaves a 4-tile rock band between neighbouring chambers, which is exactly wide
# enough for a 2-wide corridor to run down the middle with a tile of rock either
# side -- so connections read as tunnels instead of dissolving the wall between
# two chambers into one undifferentiated cave.
_LEAF_MARGIN = 2

# ---------------------------------------------------------------------------
# Tile palettes. Every entry was read off the AUTHORITATIVE collision predicate
# (``graph_builder._check_subnode_validity_simple``), not the prose in
# ``tile_definitions``, which is loose about which side of a diagonal is solid.
#
# Keys name the corner of the tile that is EMPTY after the rewrite.
# ---------------------------------------------------------------------------

# Convex bevel: a 45 degree cut across the tile. 8/9 are *floor* ramps (solid is
# bottom-anchored, so their top edge is a walkable slope); 6/7 are ceiling bevels.
_BEVEL_45 = {"TL": 8, "TR": 9, "BL": 7, "BR": 6}
# Quarter moons: the same corner, rounded instead of cut.
_BEVEL_MOON = {"TL": 12, "TR": 13, "BL": 11, "BR": 10}
# Quarter pipes: a deep concave scoop that leaves only a thin L of rock hugging
# the two solid neighbours. Keyed by the empty corner, so the value is the pipe
# whose rock clings to the *opposite* corner.
_SCOOP_PIPE = {"TL": 16, "TR": 17, "BL": 15, "BR": 14}
# Half tiles, keyed by the single side that is empty.
_HALF = {"U": 4, "D": 2, "L": 3, "R": 5}

# --- Ramps. "up"/"down" are as read left-to-right. -------------------------
#
# Every pair below was verified by sampling the collision predicate across the
# ramp's columns and checking the solid surface is continuous and spans exactly
# the intended rise -- the prose tile names do not distinguish which side of a
# diagonal is solid, and half of them are ceiling tiles that would leave a floor
# hanging in mid-air.
#
# Floor ramps are bottom-anchored (you walk on their top edge):
_FLOOR_RAMP_45 = {"up": 8, "down": 9}  # 1 column, 1 row
_FLOOR_RAMP_MILD = {"up": (20, 24), "down": (25, 21)}  # 2 columns, 1 row
_FLOOR_RAMP_STEEP = {"up": (28, 32), "down": (29, 33)}  # 1 column, 2 rows (upper, lower)
# Ceiling ramps are top-anchored (they hang; you pass underneath):
_CEIL_RAMP_45 = {"up": 6, "down": 7}
_CEIL_RAMP_MILD = {"up": (22, 18), "down": (19, 23)}


class _Chamber:
    """One carved area. ``(x1, y1)``-``(x2, y2)`` is the inclusive interior rect."""

    __slots__ = ("x1", "y1", "x2", "y2", "style", "floor")

    def __init__(self, x1: int, y1: int, x2: int, y2: int, style: str):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.style = style
        # column -> lowest EMPTY row in that column (the tile below it is rock)
        self.floor: Dict[int, int] = {x: y2 for x in range(x1, x2 + 1)}

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class MapCavernChambers(Map):
    """Multi-chamber cavern levels with linear and nonlinear completion routes."""

    # --- macro layout -------------------------------------------------------
    MIN_CHAMBERS = 3
    MAX_CHAMBERS = 5
    MIN_CHAMBER_WIDTH = 7
    # The BSP leaf is the real cap on chamber size; these ceilings only bite when
    # a tier wants deliberately small chambers inside big leaves.
    MAX_CHAMBER_WIDTH = 30
    MIN_CHAMBER_HEIGHT = 4
    MAX_CHAMBER_HEIGHT = 18
    # How strongly chamber size is pushed toward the largest rect its leaf can
    # hold. 1.0 always maxes out the leaf (open, airy levels); 0.0 samples the
    # full [MIN, MAX] range (cramped levels with lots of rock). This is the
    # solid-fraction knob -- the official corpus spans ~0.12 to ~0.85 solid, so
    # the tiers deliberately spread across it.
    FILL_BIAS = 0.6
    # Extra chamber links beyond the spanning tree. >0 creates cycles, which is
    # what makes a route genuinely nonlinear (more than one way around).
    EXTRA_EDGES = 1

    # --- chamber interiors --------------------------------------------------
    # Relative weights for (box, terraced, pillared, cavern).
    STYLE_WEIGHTS = (2, 3, 2, 3)
    # Largest single change in floor height between adjacent columns of a
    # terraced chamber. 1 becomes a 45 degree ramp; 2 is a step the ninja hops.
    # Above 2 the step stops being reliably climbable on the way back.
    MAX_FLOOR_STEP = 2
    # Initial solid probability for the cellular-automata rock in cavern chambers.
    CAVERN_FILL = 0.45

    # Rock added back onto chamber ceilings/side walls, plus free-standing
    # boulders. Roughness is what creates *corners*, and corners are what the
    # smoothing pass turns into 45 degree tiles and curves -- a perfectly
    # rectangular chamber can only ever produce flat half tiles.
    WALL_ROUGHNESS = 0.25

    # --- finishing ----------------------------------------------------------
    # Fraction of exposed rock CORNERS rewritten into a bevel/curve. High by
    # design: in the official corpus the 45 degree tiles (6-9) are the most
    # common shaped tiles by a factor of two over every other family.
    SMOOTHING_RATE = 0.85
    # Fraction of *flat* exposed rock faces thinned to a half tile. Deliberately
    # low -- long runs of half tiles read as procedural noise, and official
    # levels use roughly half as many half tiles as 45s.
    FLAT_SMOOTHING_RATE = 0.12
    # Probability that a seed uses the out-and-back (nonlinear) goal layout.
    NONLINEAR_PROB = 0.5

    # --- mines --------------------------------------------------------------
    # Toggle mines stuck to floors, ceilings and wall faces. Off by default: a
    # preset opts in, so the geometry-only tiers stay geometry-only.
    #
    # For scale: in the official corpus 97% of levels carry toggle mines, at a
    # median of 42 per level (p25 22, p90 118). Our procgen has historically
    # shipped 0-4, which is the bulk of the entity-density half of the
    # procgen-vs-human gap -- so these counts are allowed to go high.
    MIN_MINES = 0
    MAX_MINES = 0
    # Share placed as type 21 (spawns ARMED/deadly) rather than type 1 (spawns
    # untoggled and arms once touched). 0.26 is the official corpus's ratio.
    ARMED_MINE_PROB = 0.26
    # Relative weights for (floor, ceiling, wall) surfaces.
    MINE_SURFACE_WEIGHTS = (3, 1, 1)
    # Minimum tile separation between two mines. This is the main thing keeping
    # a mined floor hoppable: a lone mine with clear tiles either side is a
    # tap-hop, a contiguous row of them is a wall.
    MINE_MIN_SPACING = 3
    # Mines inside the connecting corridors. OFF by default -- a corridor is a
    # choke point with no way around, and a mine in a 2-wide vertical shaft is
    # undodgeable mid-wall-jump (the failure that made LOOP_CORRIDOR unsolvable
    # at cw=3). When enabled, only HORIZONTAL corridor runs are ever used.
    MINE_IN_CORRIDORS = False
    # Tiles kept clear around the switch and the door, so both stay touchable.
    MINE_GOAL_CLEARANCE = 2

    # --- traversal furniture -------------------------------------------------
    # One-way platforms, bounce blocks, launch pads and boost pads. All OFF by
    # default. These are TRAVERSAL types, not hazards: they change where the
    # ninja can go, and per ``constants.entity_types`` bounce blocks and launch
    # pads sit in phase 3 -- the scope step that changes the cost-to-go field and
    # can invalidate PBRS, the demo corpus and the reverse curriculum together.
    # Turning them on is a training-scope decision, not a level-variety one.
    #
    # Corpus reference (600 official levels): one-way 21/level on the 156 levels
    # that use them, bounce block 13/level on 72, launch pad 3/level on 53,
    # boost pad 3.6/level on 35.
    MIN_ONE_WAYS = 0
    MAX_ONE_WAYS = 0
    MIN_BOUNCE_BLOCKS = 0
    MAX_BOUNCE_BLOCKS = 0
    MIN_LAUNCH_PADS = 0
    MAX_LAUNCH_PADS = 0
    MIN_BOOST_PADS = 0
    MAX_BOOST_PADS = 0
    TRAVERSAL_MIN_SPACING = 3

    # --- mobile / ranged hazards ---------------------------------------------
    # Patrol drones (zap 14 + mini 26) and emplacements (turret 19, rocket 18).
    # These are PURE HAZARDS: per ``constants.entity_types`` they never bend the
    # ninja's trajectory, so the reachability graph and the cost-to-go field stay
    # correct and they need no connectivity proof -- only a guarantee that they
    # cannot shoot the spawn before the agent has moved.
    #
    # Caps rather than densities, because both classes are naturally sparse in
    # the corpus: rocket runs 1/1/2/3 per level at p25/50/75/90 with a crowd of 1,
    # turret 1/1/2/4 with a crowd of 1-2. Patrol drones are the exception --
    # official runs 8/14/28/52 -- so MAX_DRONES here is deliberately far below
    # corpus density; raise it if the caverns should feel like official swarms.
    MIN_DRONES = 0
    MAX_DRONES = 0
    MIN_EMPLACEMENTS = 0
    MAX_EMPLACEMENTS = 0
    #: Relative weights for (turret, rocket) within the emplacement budget.
    EMPLACEMENT_WEIGHTS = (1, 1)
    #: Tiles of clearance from the ninja spawn. Ranged classes additionally must
    #: have no unobstructed line to the spawn cell at all.
    HAZARD_SPAWN_CLEARANCE = 4
    #: Open cells a turret/rocket must see down some axis, or it is decoration.
    HAZARD_MIN_SIGHTLINE = 4

    # --- gold ----------------------------------------------------------------
    # Gold is laid down in CLUSTERS with a deliberate shape, because that is what
    # the official corpus is: every one of its 3568 levels has gold, 93% of it
    # sits in clusters, and the counts are 24 gold in 5 clusters at the median
    # (p90: 51 gold in 11 clusters). Clusters are also what makes the collection
    # objective tractable -- the planner orders CLUSTERS exactly and sweeps
    # within them, which only works if gold actually comes in blobs.
    MIN_GOLD_CLUSTERS = 0
    MAX_GOLD_CLUSTERS = 0
    MIN_GOLD_PER_CLUSTER = 3
    MAX_GOLD_PER_CLUSTER = 12
    # Relative weights for (grid, triangle, ring, arc, blob).
    GOLD_SHAPE_WEIGHTS = (3, 2, 2, 2, 2)
    # Share of clusters laid on the HALF-tile (12px) lattice rather than the
    # full-tile one. Measured: 12px is the dominant intra-cluster
    # nearest-neighbour distance in the official corpus (76% of pairs) and gold
    # has a 6px radius, so a 12px pitch packs them exactly touching.
    GOLD_HALF_PITCH_PROB = 0.75

    # Whole-layout retries before falling back to a guaranteed-simple layout.
    LAYOUT_ATTEMPTS = 12

    # Corridors are 2 tiles wide/tall: the wall-jumpable shaft width, and enough
    # headroom to run through. Not configurable -- the router's block model and
    # the solvability guarantees both assume it.
    _CORRIDOR = 2

    # ------------------------------------------------------------------ setup
    def _fill_solid(self) -> None:
        self.set_tiles_bulk([SOLID] * (MAP_TILE_WIDTH * MAP_TILE_HEIGHT))

    # -------------------------------------------------------------- partition
    def _partition(self) -> List[Tuple[int, int, int, int]]:
        """Jittered BSP of the playable rect into leaf regions.

        A leaf must be able to hold the smallest chamber plus a 1-tile margin on
        every side, so the split is only taken when *both* halves stay legal.
        """
        min_leaf_w = self.MIN_CHAMBER_WIDTH + 2 * _LEAF_MARGIN
        min_leaf_h = self.MIN_CHAMBER_HEIGHT + 2 * _LEAF_MARGIN
        target = self.rng.randint(self.MIN_CHAMBERS, self.MAX_CHAMBERS)

        leaves = [(1, 1, MAP_TILE_WIDTH - 2, MAP_TILE_HEIGHT - 2)]
        while len(leaves) < target:
            # Bias toward splitting the biggest leaf so chambers stay comparable.
            order = sorted(
                range(len(leaves)),
                key=lambda i: -(
                    (leaves[i][2] - leaves[i][0] + 1) * (leaves[i][3] - leaves[i][1] + 1)
                ),
            )
            for idx in order:
                x1, y1, x2, y2 = leaves[idx]
                w, h = x2 - x1 + 1, y2 - y1 + 1
                can_v = w >= 2 * min_leaf_w
                can_h = h >= 2 * min_leaf_h
                if not (can_v or can_h):
                    continue
                if can_v and can_h:
                    vertical = self.rng.random() < (w / float(w + h))
                else:
                    vertical = can_v
                if vertical:
                    cut = self.rng.randint(x1 + min_leaf_w - 1, x2 - min_leaf_w)
                    leaves[idx] = (x1, y1, cut, y2)
                    leaves.append((cut + 1, y1, x2, y2))
                else:
                    cut = self.rng.randint(y1 + min_leaf_h - 1, y2 - min_leaf_h)
                    leaves[idx] = (x1, y1, x2, cut)
                    leaves.append((x1, cut + 1, x2, y2))
                break
            else:
                break  # nothing left to split
        return leaves

    def _leaf_fraction(self, lo: int, hi: int, frac_lo: float) -> int:
        """Take ``uniform(frac_lo, 1.0)`` of ``hi``, clamped into ``[lo, hi]``."""
        if hi <= lo:
            return lo
        return max(lo, min(hi, int(round(hi * self.rng.uniform(frac_lo, 1.0)))))

    def _chambers_from_leaves(
        self, leaves: Sequence[Tuple[int, int, int, int]]
    ) -> List[_Chamber]:
        """Inset one chamber rect inside each leaf, with a >=1 tile rock margin.

        Two chambers in adjacent leaves therefore always have >=2 rock tiles
        between them -- the corridor router's minimum clearance.
        """
        styles = ("box", "terraced", "pillared", "cavern")
        chambers: List[_Chamber] = []
        # One roominess draw per map, so a seed is uniformly airy or uniformly
        # cramped rather than a mix -- which is how the official corpus's solid
        # fraction is distributed (bimodal per level, not per chamber).
        bias = max(0.0, min(1.0, float(self.FILL_BIAS)))
        frac_lo = 0.30 + 0.60 * bias
        frac_lo = self.rng.uniform(max(0.25, frac_lo - 0.18), min(1.0, frac_lo + 0.12))

        for lx1, ly1, lx2, ly2 in leaves:
            lw, lh = lx2 - lx1 + 1, ly2 - ly1 + 1
            m = _LEAF_MARGIN
            max_w = min(self.MAX_CHAMBER_WIDTH, lw - 2 * m)
            max_h = min(self.MAX_CHAMBER_HEIGHT, lh - 2 * m)
            if max_w < self.MIN_CHAMBER_WIDTH or max_h < self.MIN_CHAMBER_HEIGHT:
                continue
            # Size as a FRACTION of the leaf, so the knob controls the map's
            # solid fraction directly rather than through the chamber count.
            w = self._leaf_fraction(self.MIN_CHAMBER_WIDTH, max_w, frac_lo)
            h = self._leaf_fraction(self.MIN_CHAMBER_HEIGHT, max_h, frac_lo)
            x1 = self.rng.randint(lx1 + m, lx2 - m - w + 1)
            y1 = self.rng.randint(ly1 + m, ly2 - m - h + 1)
            style = self.rng.choices(styles, weights=list(self.STYLE_WEIGHTS))[0]
            chambers.append(_Chamber(x1, y1, x1 + w - 1, y1 + h - 1, style))
        return chambers

    # ----------------------------------------------------------- chamber carve
    def _carve_chamber(self, c: _Chamber) -> None:
        self.set_empty_rectangle(c.x1, c.y1, c.x2, c.y2)
        if c.style == "terraced":
            self._lay_floor(c)
        elif c.style == "pillared":
            self._shape_pillared(c)
        elif c.style == "cavern":
            self._lay_floor(c)
            self._shape_cavern(c)
        # "box" keeps the flat floor the constructor already recorded.
        if c.style in ("terraced", "cavern") and self.rng.random() < 0.7:
            self._lay_ceiling(c)
        self._rough(c)

    def _rough(self, c: _Chamber) -> None:
        """Jag the chamber's ceiling and side walls, and drop in boulders.

        Everything here adds rock, and everything stays at least two rows above
        the floor, so the walk band that makes a chamber crossable is untouched.
        The point is corner *density*: each bump and boulder contributes up to
        four corners for the smoothing pass to bevel, which is what moves the
        tile histogram from "rectangles plus half tiles" toward the official
        corpus's 45-degree-dominated mix.
        """
        rate = float(self.WALL_ROUGHNESS)
        if rate <= 0.0:
            return

        def band_top(x: int) -> int:
            """Lowest row roughness may touch in column ``x`` (walk band + head)."""
            return c.floor[x] - 2

        # Side walls: vertical streaks of rock pushed 1-2 tiles inward. Streaks
        # rather than per-row coin flips, so a wall reads as a buttress instead
        # of static.
        for x_wall, inward in ((c.x1, 1), (c.x2, -1)):
            y = c.y1
            while y <= band_top(x_wall):
                if self.rng.random() < rate:
                    depth = 1 if self.rng.random() < 0.7 else 2
                    for k in range(self.rng.randint(2, 4)):  # streak height
                        for j in range(depth):
                            x = x_wall + inward * j
                            if c.x1 <= x <= c.x2 and y + k <= band_top(x):
                                self.set_tile(x, y + k, SOLID)
                    y += 3
                y += 1

        # Ceiling: stalactite clusters, tapering away from their centre column.
        x = c.x1
        while x <= c.x2:
            if self.rng.random() < rate * 0.7:
                span = self.rng.randint(1, 3)
                drop = self.rng.randint(1, 3)
                for k in range(span):
                    cx = x + k
                    if cx > c.x2:
                        break
                    y = c.y1
                    while y <= band_top(cx) and self.get_tile(cx, y) != EMPTY:
                        y += 1  # first air row under the existing ceiling
                    taper = max(1, drop - abs(k - span // 2))
                    for j in range(taper):
                        if y + j <= band_top(cx):
                            self.set_tile(cx, y + j, SOLID)
                x += span + self.rng.randint(1, 3)
            else:
                x += 1

        # Free-standing ledges: short horizontal rock shelves in the chamber's
        # air. Wide enough to read (and land) as platforms rather than specks.
        area = (c.x2 - c.x1 + 1) * (c.y2 - c.y1 + 1)
        for _ in range(int(area * rate / 45)):
            bw = self.rng.randint(2, 4)
            bx = self.rng.randint(c.x1 + 1, max(c.x1 + 1, c.x2 - bw))
            by = self.rng.randint(c.y1 + 1, max(c.y1 + 1, band_top(bx) - 1))
            for k in range(bw):
                x = bx + k
                if c.x1 <= x <= c.x2 and c.y1 < by <= band_top(x):
                    self.set_tile(x, by, SOLID)

    # ``surf`` below is the row of the TOPMOST SOLID tile in a column; the
    # walkable empty cell is ``surf - 1``, which is what ``_Chamber.floor``
    # stores.

    def _fill_below(self, c: _Chamber, x: int, from_row: int) -> None:
        for y in range(from_row, c.y2 + 1):
            self.set_tile(x, y, SOLID)

    def _lay_floor(self, c: _Chamber) -> None:
        """Build a stepped floor whose height changes are real ramp geometry.

        Flat runs of 2-5 columns are separated by a transition drawn from the
        verified ramp palette (45 degree, mild 2-column, steep 2-row) or by a
        bare cliff the ninja hops. ``lo`` keeps three empty rows above the floor
        so every ramp's tiles fit inside the chamber; the first and last columns
        are forced flat so the left/right corridor ports dock on level ground.
        """
        lo, hi = c.y1 + 3, c.y2
        if lo > hi:
            return  # too short to terrace: keep the flat box floor
        step_cap = max(1, int(self.MAX_FLOOR_STEP))
        surf = self.rng.randint(lo, hi)
        x = c.x1

        def flat(col: int) -> None:
            self.set_tile(col, surf, SOLID)
            self._fill_below(c, col, surf)
            c.floor[col] = surf - 1

        while x <= c.x2 - 1:
            run = min(self.rng.randint(2, 5), c.x2 - x)
            for _ in range(run):
                flat(x)
                x += 1
            if x > c.x2 - 1:
                break
            budget = c.x2 - 1 - x  # columns a transition may consume
            surf = self._floor_transition(c, x, surf, lo, hi, budget, step_cap)
            x += self._last_transition_cols
        # Tail: the final column(s) stay flat so the right-hand port docks level.
        while x <= c.x2:
            flat(x)
            x += 1

    def _floor_transition(
        self,
        c: _Chamber,
        x: int,
        surf: int,
        lo: int,
        hi: int,
        budget: int,
        step_cap: int,
    ) -> int:
        """Draw one height change at column ``x``. Returns the new ``surf``."""
        options: List[Tuple[str, int, int]] = []  # (kind, delta, columns used)
        if surf + 1 <= hi:
            options += [("cliff", 1, 0), ("r45", 1, 1)]
            if budget >= 2:
                options.append(("mild", 1, 2))
        if surf - 1 >= lo:
            options += [("cliff", -1, 0), ("r45", -1, 1)]
            if budget >= 2:
                options.append(("mild", -1, 2))
        if step_cap >= 2:
            if surf + 2 <= hi:
                options += [("cliff", 2, 0), ("steep", 2, 1)]
            if surf - 2 >= lo:
                options += [("cliff", -2, 0), ("steep", -2, 1)]
        if not options:
            self._last_transition_cols = 0
            return surf

        kind, delta, cols = self.rng.choice(options)
        self._last_transition_cols = cols
        direction = "down" if delta > 0 else "up"

        if kind == "r45":
            row = surf if direction == "down" else surf - 1
            self.set_tile(x, row, _FLOOR_RAMP_45[direction])
            self._fill_below(c, x, row + 1)
            c.floor[x] = row - 1
            self._ramp_columns.add((x, c.floor[x]))
        elif kind == "mild":
            row = surf if direction == "down" else surf - 1
            left, right = _FLOOR_RAMP_MILD[direction]
            self.set_tile(x, row, left)
            self.set_tile(x + 1, row, right)
            for dx in (0, 1):
                self._fill_below(c, x + dx, row + 1)
                c.floor[x + dx] = row - 1
                self._ramp_columns.add((x + dx, row - 1))
        elif kind == "steep":
            upper = surf if direction == "down" else surf - 2
            top_tile, bot_tile = _FLOOR_RAMP_STEEP[direction]
            self.set_tile(x, upper, top_tile)
            self.set_tile(x, upper + 1, bot_tile)
            self._fill_below(c, x, upper + 2)
            c.floor[x] = upper - 1
            self._ramp_columns.add((x, c.floor[x]))
        return surf + delta

    def _lay_ceiling(self, c: _Chamber) -> None:
        """Drop the ceiling into the chamber in ramped steps.

        ``r`` is the LOWEST SOLID row of the ceiling, starting at the rock row
        just above the rect. It never comes within three rows of the floor, so
        the walk band and its headroom survive whatever the profile does.
        """
        r = c.y1 - 1
        x = c.x1

        def head_limit(col: int) -> int:
            return c.floor[col] - 3

        while x <= c.x2:
            run = min(self.rng.randint(2, 6), c.x2 - x + 1)
            for _ in range(run):
                for y in range(c.y1, min(r, head_limit(x)) + 1):
                    self.set_tile(x, y, SOLID)
                x += 1
            if x > c.x2 - 1:
                break
            budget = c.x2 - x
            options: List[Tuple[str, int, int]] = []
            if r + 1 <= head_limit(x) and r + 1 <= c.y2:
                options += [("step", 1, 0), ("r45", 1, 1)]
                if budget >= 2 and r + 1 <= head_limit(x + 1):
                    options.append(("mild", 1, 2))
            if r - 1 >= c.y1 - 1:
                options += [("step", -1, 0), ("r45", -1, 1)]
                if budget >= 2:
                    options.append(("mild", -1, 2))
            if not options:
                break
            kind, delta, cols = self.rng.choice(options)
            direction = "down" if delta > 0 else "up"
            if kind == "r45":
                row = r + 1 if direction == "down" else r
                if c.y1 <= row <= head_limit(x):
                    self.set_tile(x, row, _CEIL_RAMP_45[direction])
                    for y in range(c.y1, row):
                        self.set_tile(x, y, SOLID)
            elif kind == "mild":
                row = r + 1 if direction == "down" else r
                left, right = _CEIL_RAMP_MILD[direction]
                if c.y1 <= row <= min(head_limit(x), head_limit(x + 1)):
                    self.set_tile(x, row, left)
                    self.set_tile(x + 1, row, right)
                    for dx in (0, 1):
                        for y in range(c.y1, row):
                            self.set_tile(x + dx, y, SOLID)
            r += delta
            x += cols

    def _shape_pillared(self, c: _Chamber) -> None:
        """Stand 1-2 wide pillars on the floor, <=3 tall, with >=3 tile gaps.

        Short enough to hop, spaced enough to run between: variety without
        blocking the walk band's traversability at ground level... the pillar
        occupies the band, so gaps are what keep the chamber crossable.
        """
        span = c.x2 - c.x1 + 1
        head = c.y2 - c.y1  # rows of air above the floor
        if span < 8 or head < 3:
            return
        x = c.x1 + 2
        while x <= c.x2 - 3:
            if self.rng.random() < 0.45:
                pw = self.rng.randint(1, 2)
                ph = self.rng.randint(1, min(3, head - 1))
                for px in range(x, min(x + pw, c.x2)):
                    for py in range(c.y2 - ph + 1, c.y2 + 1):
                        self.set_tile(px, py, SOLID)
                    c.floor[px] = c.y2 - ph
                x += pw + self.rng.randint(3, 5)
            else:
                x += self.rng.randint(2, 4)

    def _shape_cavern(self, c: _Chamber) -> None:
        """Cellular-automata rock clinging to the chamber's ceiling and walls.

        The CA runs only above the walk band, and is seeded with a probability
        that decays away from the rect border, so blobs hang from the ceiling and
        cling to the side walls instead of filling the middle. The two rows above
        the floor are force-cleared afterwards, so the chamber stays crossable
        whatever the automaton does.
        """
        top = c.y1
        # Never within two rows of the (possibly terraced) floor: that is the
        # walk band, and it is what keeps the chamber crossable at ground level.
        bot = min(c.floor[x] for x in range(c.x1, c.x2 + 1)) - 2
        if bot < top:
            return
        w = c.x2 - c.x1 + 1
        h = bot - top + 1
        grid = [[False] * w for _ in range(h)]
        for j in range(h):
            for i in range(w):
                # Distance to the nearest rect edge, in tiles.
                d = min(i, w - 1 - i, j, h - 1 - j)
                p = self.CAVERN_FILL * (0.45 ** d) + 0.06
                grid[j][i] = self.rng.random() < p
        for _ in range(4):
            nxt = [[False] * w for _ in range(h)]
            for j in range(h):
                for i in range(w):
                    n = 0
                    for dj in (-1, 0, 1):
                        for di in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            jj, ii = j + dj, i + di
                            # Out-of-rect counts as rock: the surrounding map is
                            # solid, so blobs grow inward off the walls.
                            if not (0 <= jj < h and 0 <= ii < w) or grid[jj][ii]:
                                n += 1
                    nxt[j][i] = n >= 5 if grid[j][i] else n >= 6
            grid = nxt
        for j in range(h):
            for i in range(w):
                if grid[j][i]:
                    self.set_tile(c.x1 + i, top + j, SOLID)

    # ------------------------------------------------------------ chamber graph
    def _build_graph(self, chambers: Sequence[_Chamber]) -> List[Tuple[int, int]]:
        """Random-weight spanning tree over chamber centres + ``EXTRA_EDGES``.

        The spanning tree is what guarantees every chamber is reachable; the
        extra edges are what make the topology cyclic (nonlinear routing).
        """
        n = len(chambers)
        pairs = []
        for a in range(n):
            for b in range(a + 1, n):
                d = abs(chambers[a].cx - chambers[b].cx) + abs(
                    chambers[a].cy - chambers[b].cy
                )
                # Jitter so the tree is not always the same nearest-neighbour chain.
                pairs.append((d * self.rng.uniform(0.75, 1.35), a, b))
        pairs.sort()

        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        tree: List[Tuple[int, int]] = []
        rest: List[Tuple[int, int]] = []
        for _d, a, b in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                tree.append((a, b))
            else:
                rest.append((a, b))
        extra = rest[: max(0, int(self.EXTRA_EDGES))]
        return tree + extra

    # ------------------------------------------------------------------ router
    def _block_free(
        self, bx: int, by: int, chambers: Sequence[_Chamber]
    ) -> bool:
        """Can the 2x2 corridor block anchored at ``(bx, by)`` be carved here?

        Blocks must stay off the 1-tile map border (so shafts keep outer walls)
        and must never overlap a chamber interior -- chambers own their space,
        and a corridor cutting through one would destroy the flat shaft faces the
        wall-jump relies on.
        """
        if bx < 1 or by < 1:
            return False
        if bx + 1 > MAP_TILE_WIDTH - 2 or by + 1 > MAP_TILE_HEIGHT - 2:
            return False
        for c in chambers:
            if not (bx + 1 < c.x1 or c.x2 < bx or by + 1 < c.y1 or c.y2 < by):
                return False
        return True

    def _ports(
        self, c: _Chamber, chambers: Sequence[_Chamber]
    ) -> List[Tuple[int, int]]:
        """Corridor block anchors that dock against this chamber at floor level.

        Docking at the floor is the invariant that keeps every level solvable:
        a corridor leaves and enters a chamber on flat ground, so all height
        change happens inside 2-wide rock shafts, which are wall-jumpable both
        ways.
        """
        out = []
        left_y = c.floor[c.x1]
        right_y = c.floor[c.x2]
        for anchor in ((c.x1 - 2, left_y - 1), (c.x2 + 1, right_y - 1)):
            if self._block_free(anchor[0], anchor[1], chambers):
                out.append(anchor)
        return out

    def _route(
        self,
        starts: Sequence[Tuple[int, int]],
        goals: Sequence[Tuple[int, int]],
        chambers: Sequence[_Chamber],
    ) -> Optional[List[Tuple[int, int]]]:
        """BFS a 2x2-block path through solid rock. Returns block anchors."""
        if not starts or not goals:
            return None
        goal_set = set(goals)
        seen = {s: None for s in starts}
        queue = deque(starts)
        while queue:
            cur = queue.popleft()
            if cur in goal_set:
                path = []
                node = cur
                while node is not None:
                    path.append(node)
                    node = seen[node]
                path.reverse()
                return path
            cx, cy = cur
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cx + dx, cy + dy)
                if nxt in seen:
                    continue
                if not self._block_free(nxt[0], nxt[1], chambers):
                    continue
                seen[nxt] = cur
                queue.append(nxt)
        return None

    def _carve_path(self, path: Sequence[Tuple[int, int]]) -> None:
        """Carve the 2x2 blocks along ``path``, tagging each cell's run direction.

        The tag is what later lets mines go in horizontal runs but never in
        vertical ones. A cell touched by both a horizontal and a vertical block
        is a corner, and is tagged vertical -- the conservative reading, since a
        corner is where a climb starts.
        """
        for idx, (bx, by) in enumerate(path):
            vertical = False
            for other in (path[idx - 1] if idx else None, path[idx + 1] if idx + 1 < len(path) else None):
                if other is not None and other[0] == bx and other[1] != by:
                    vertical = True
            for y in range(by, by + self._CORRIDOR):
                for x in range(bx, bx + self._CORRIDOR):
                    self.set_tile(x, y, EMPTY)
                    self._corridor_cells.add((x, y))
                    if vertical:
                        self._shaft_cells.add((x, y))

    def _connect(
        self, chambers: Sequence[_Chamber], edges: Sequence[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Carve every edge that routes; return the edges that actually landed."""
        ports = [self._ports(c, chambers) for c in chambers]
        carved: List[Tuple[int, int]] = []
        for a, b in edges:
            path = self._route(ports[a], ports[b], chambers)
            if path is None:
                continue
            self._carve_path(path)
            carved.append((a, b))
        return carved

    # ---------------------------------------------------------------- smoothing
    def _is_open(self, x: int, y: int) -> bool:
        """Treat out-of-bounds as rock so the map border never gets bevelled."""
        if not (0 <= x < MAP_TILE_WIDTH and 0 <= y < MAP_TILE_HEIGHT):
            return False
        return self.get_tile(x, y) == EMPTY

    def _smooth(self) -> None:
        """Rewrite exposed rock into slopes, curves and half tiles.

        Safety rests on two properties:

        * every replacement's solid region is a strict subset of the full tile,
          so the pass only ever *opens* space and can never sever a route;
        * the palette is anchor-correct. A tile with air above is only ever
          replaced by a bottom-anchored tile and one with air below by a
          top-anchored one, so a floor keeps its floor and a ceiling keeps its
          ceiling -- what changes is the shape of the surface, not whether there
          is one.

        Two exclusions keep it from making levels *worse* rather than unsolvable:
        the rock columns flanking a corridor are frozen so shaft faces stay flat
        and wall-jumpable, and the ground under the three goal entities is frozen
        so they sit on level footing.
        """
        corner_rate = float(self.SMOOTHING_RATE)
        flat_rate = float(self.FLAT_SMOOTHING_RATE)
        if corner_rate <= 0.0 and flat_rate <= 0.0:
            return
        # Freeze only the two rock columns FLANKING a corridor -- those are the
        # vertical faces a shaft climb wall-jumps off, and a bevel there puts a
        # notch in the face. Rock directly above a corridor is just its ceiling
        # and is free to be shaped (that only adds headroom); rock directly below
        # is its floor, already covered by the ``supports_floor`` rule.
        frozen = set()
        for x, y in self._corridor_cells:
            for dy in (-1, 0, 1):
                frozen.add((x - 1, y + dy))
                frozen.add((x + 1, y + dy))
        for gx, gy in self._goal_cells or ():
            for dx in (-1, 0, 1):
                frozen.add((gx + dx, gy + 1))

        # Snapshot: decisions all read the ORIGINAL 0/1 mask, so a rewrite never
        # cascades into its neighbour's classification.
        original = list(self.tile_data)

        def open_at(x: int, y: int) -> bool:
            if not (0 <= x < MAP_TILE_WIDTH and 0 <= y < MAP_TILE_HEIGHT):
                return False
            return original[x + y * MAP_TILE_WIDTH] == EMPTY

        for y in range(1, MAP_TILE_HEIGHT - 1):
            for x in range(1, MAP_TILE_WIDTH - 1):
                if original[x + y * MAP_TILE_WIDTH] != SOLID:
                    continue
                if (x, y) in frozen:
                    continue
                u, d = open_at(x, y - 1), open_at(x, y + 1)
                l, r = open_at(x - 1, y), open_at(x + 1, y)
                if u + d + l + r == 0:
                    continue  # buried rock: nothing to shape

                corner = None
                if u and l and not d and not r:
                    corner = "TL"
                elif u and r and not d and not l:
                    corner = "TR"
                elif d and l and not u and not r:
                    corner = "BL"
                elif d and r and not u and not l:
                    corner = "BR"

                if corner is not None:
                    if self.rng.random() >= corner_rate:
                        continue
                    roll = self.rng.random()
                    if roll < 0.62:
                        tile = _BEVEL_45[corner]
                    elif roll < 0.87:
                        tile = _BEVEL_MOON[corner]
                    else:
                        tile = _SCOOP_PIPE[corner]
                    self.set_tile(x, y, tile)
                    continue

                if self.rng.random() >= flat_rate:
                    continue
                if u and not d and not l and not r:
                    self.set_tile(x, y, _HALF["U"])
                elif d and not u and not l and not r:
                    self.set_tile(x, y, _HALF["D"])
                elif l and not u and not d and not r:
                    self.set_tile(x, y, _HALF["L"])
                elif r and not u and not d and not l:
                    self.set_tile(x, y, _HALF["R"])

    # ------------------------------------------------------------------ entities
    def _graph_distances(
        self, n: int, edges: Sequence[Tuple[int, int]], src: int
    ) -> List[int]:
        adj: List[List[int]] = [[] for _ in range(n)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if dist[nxt] < 0:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        return dist

    def _floor_spots(self, c: _Chamber) -> List[Tuple[int, int]]:
        """Standable cells: empty floor cell, empty headroom, flat rock underfoot.

        Ramp columns are excluded -- a slope is fine to run over but a poor place
        to spawn (the ninja slides off) or to park a switch the agent must touch.
        """
        spots = []
        for x in range(c.x1, c.x2 + 1):
            fy = c.floor[x]
            if fy - 1 < c.y1 or (x, fy) in self._ramp_columns:
                continue
            if self.get_tile(x, fy) != EMPTY or self.get_tile(x, fy - 1) != EMPTY:
                continue
            if self.get_tile(x, fy + 1) != SOLID:
                continue
            spots.append((x, fy))
        return spots

    def _place_goals(
        self, chambers: Sequence[_Chamber], edges: Sequence[Tuple[int, int]]
    ) -> bool:
        """Pick spawn/switch/door chambers, then standable cells inside them."""
        n = len(chambers)
        spots = [self._floor_spots(c) for c in chambers]
        usable = [i for i in range(n) if spots[i]]
        if not usable:
            return False

        # Spawn in the most peripheral usable chamber, so route length is maximal.
        best_src, best_ecc = usable[0], -1
        for i in usable:
            dist = self._graph_distances(n, edges, i)
            ecc = max(d for d in dist if d >= 0)
            if ecc > best_ecc:
                best_src, best_ecc = i, ecc
        src = best_src
        dist = self._graph_distances(n, edges, src)
        reachable = [i for i in usable if dist[i] >= 0]
        far = max(reachable, key=lambda i: (dist[i], self.rng.random()))

        nonlinear = self.rng.random() < float(self.NONLINEAR_PROB)
        if nonlinear:
            # Out-and-back: switch at the far end, door back beside the spawn.
            # The agent must cross the whole graph and then cross it again.
            switch_c = far
            near = [i for i in reachable if i != switch_c]
            door_c = min(near, key=lambda i: (dist[i], self.rng.random())) if near else far
        else:
            # Outbound: door at the far end, switch strictly closer, so the two
            # legs chain instead of doubling back. Picking the *second* farthest
            # (rather than a random mid chamber) matters when the graph is a
            # 3-node star, where "strictly between" is empty and the switch would
            # otherwise land in the door's own chamber.
            door_c = far
            mid = [i for i in reachable if i != door_c]
            switch_c = max(mid, key=lambda i: (dist[i], self.rng.random())) if mid else far

        taken: List[Tuple[int, int]] = []

        def pick(ci: int) -> Optional[Tuple[int, int]]:
            cands = [
                s
                for s in spots[ci]
                if all(abs(s[0] - t[0]) >= 3 or abs(s[1] - t[1]) >= 2 for t in taken)
            ]
            if not cands:
                cands = spots[ci]
            if not cands:
                return None
            choice = self.rng.choice(cands)
            taken.append(choice)
            return choice

        ninja = pick(src)
        switch = pick(switch_c)
        door = pick(door_c)
        if ninja is None or switch is None or door is None:
            return False

        self.set_ninja_spawn(ninja[0], ninja[1], 1 if switch[0] >= ninja[0] else -1)
        self.add_entity(3, door[0], door[1], 0, 0, switch[0], switch[1])
        self._goal_cells = (ninja, switch, door)
        return True

    # -------------------------------------------------------------------- mines
    #
    # Placement coordinates follow the border-corrected convention (see
    # ``map_mine_path`` / ``map_loop_corridor``): the sim wraps the map in a
    # 1-tile border and toggle mines take NO entity offset, so a grid coordinate
    # ``g`` handed to ``add_entity`` lands at pixel ``g * 24``. For the EMPTY cell
    # ``(cx, cy)`` that a mine is stuck to, that makes
    #   cx + 1.5 -> the cell's horizontal centre     cy + 1.5 -> vertical centre
    #   cx + 1.0 -> its left edge                    cy + 1.0 -> its top edge
    #   cx + 2.0 -> its right edge                   cy + 2.0 -> its bottom edge
    # so a floor mine is (centre, bottom edge), a ceiling mine (centre, top
    # edge), and a wall mine (the wall's edge, centre). Passing raw tile
    # coordinates instead floats the mine a full 48px off its surface.

    def _mine_surface(self, x: int, y: int) -> Optional[Tuple[str, float, float]]:
        """Classify the surface at empty cell ``(x, y)``, if it can hold a mine.

        Requires a FULL solid tile (type 1) as the anchor -- a mine fused into a
        slope or curve reads wrong and sits off the visible surface -- and
        requires the clearance a ninja needs to get past it: two empty rows above
        a floor mine to hop it, two empty rows below a ceiling mine to run under
        it, two empty columns beside a wall mine to squeeze past it.
        """
        options: List[Tuple[str, float, float]] = []
        weights: List[int] = []
        w_floor, w_ceiling, w_wall = self.MINE_SURFACE_WEIGHTS

        if w_floor > 0 and self.get_tile(x, y + 1) == SOLID and self._is_open(x, y - 1):
            # A run-up on at least one side is what makes the hop possible.
            if self._is_open(x - 1, y) or self._is_open(x + 1, y):
                options.append(("floor", x + 1.5, y + 2.0))
                weights.append(w_floor)
        if (
            w_ceiling > 0
            and self.get_tile(x, y - 1) == SOLID
            and self._is_open(x, y + 1)
            and self._is_open(x, y + 2)
        ):
            options.append(("ceiling", x + 1.5, y + 1.0))
            weights.append(w_ceiling)
        if w_wall > 0:
            if (
                self.get_tile(x - 1, y) == SOLID
                and self._is_open(x + 1, y)
                and self._is_open(x + 2, y)
            ):
                options.append(("wall_left", x + 1.0, y + 1.5))
                weights.append(w_wall)
            elif (
                self.get_tile(x + 1, y) == SOLID
                and self._is_open(x - 1, y)
                and self._is_open(x - 2, y)
            ):
                options.append(("wall_right", x + 2.0, y + 1.5))
                weights.append(w_wall)

        if not options:
            return None
        return self.rng.choices(options, weights=weights)[0]

    def _mine_cells(self, chambers: Sequence[_Chamber]) -> List[Tuple[int, int]]:
        """Cells eligible to carry a mine, before per-mine spacing is applied."""
        cells: List[Tuple[int, int]] = []
        for c in chambers:
            for y in range(c.y1, c.y2 + 1):
                for x in range(c.x1, c.x2 + 1):
                    if self.get_tile(x, y) == EMPTY:
                        cells.append((x, y))
        if self.MINE_IN_CORRIDORS:
            # Horizontal runs only: a shaft cell is a wall-jump face, and a mine
            # there cannot be dodged mid-climb.
            cells += [c for c in self._corridor_cells if c not in self._shaft_cells]
            near_shaft = set()
            for x, y in self._shaft_cells:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        near_shaft.add((x + dx, y + dy))
            cells = [c for c in cells if c not in near_shaft]
        else:
            # Keep a one-tile apron around every corridor mouth clear, so a mine
            # never sits in the one cell a route has to pass through.
            blocked = set()
            for x, y in self._corridor_cells:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        blocked.add((x + dx, y + dy))
            cells = [c for c in cells if c not in blocked]
        return cells

    def _place_mines(
        self, chambers: Sequence[_Chamber], occupied: Set[Tuple[int, int]]
    ) -> List[Tuple[Tuple[int, int], bool, int, float, float, int]]:
        """Stick toggle mines to floors, ceilings and walls.

        Runs after smoothing, so it sees the final geometry and can insist on a
        full solid anchor tile. Returns placement records rather than emitting,
        so mines and traversal furniture share one connectivity proof.
        """
        target = self.rng.randint(int(self.MIN_MINES), int(self.MAX_MINES))
        if target <= 0 or not self._goal_cells:
            return []
        # The authoritative spawn-clearance radius, and never less than a tile
        # either side of the spawn.
        from .solvability import MINE_SPAWN_CLEARANCE_PX

        clearance_px = max(float(MINE_SPAWN_CLEARANCE_PX), 2.0 * 24.0)
        ninja, switch, door = self._goal_cells
        # set_ninja_spawn applies NINJA_SPAWN_OFFSET_UNITS, putting the spawn at
        # its cell's CENTRE in the bordered pixel frame.
        spawn_px = ((ninja[0] + 1.5) * 24.0, (ninja[1] + 1.5) * 24.0)
        goal_clear = int(self.MINE_GOAL_CLEARANCE)
        spacing = int(self.MINE_MIN_SPACING)

        cells = [c for c in self._mine_cells(chambers) if c not in occupied]
        self.rng.shuffle(cells)

        placed: List[Tuple[int, int]] = []
        records = []
        for cx, cy in cells:
            if len(records) >= target:
                break
            if any(
                abs(cx - gx) <= goal_clear and abs(cy - gy) <= goal_clear
                for gx, gy in (ninja, switch, door)
            ):
                continue
            if any(
                abs(cx - px) < spacing and abs(cy - py) < spacing for px, py in placed
            ):
                continue
            surface = self._mine_surface(cx, cy)
            if surface is None:
                continue
            _kind, gx, gy = surface
            dx, dy = gx * 24.0 - spawn_px[0], gy * 24.0 - spawn_px[1]
            if dx * dx + dy * dy < clearance_px * clearance_px:
                continue
            mine_type = 21 if self.rng.random() < float(self.ARMED_MINE_PROB) else 1
            placed.append((cx, cy))
            # A mine is a hazard, not geometry: it never blocks the flood fill's
            # topology, but it is still proved harmless below on the pessimistic
            # mines-as-walls model.
            records.append(((cx, cy), True, mine_type, gx, gy, 0))
        return records

    # ------------------------------------------------------- traversal furniture
    #
    # Everything here is a TRAVERSAL type (entity_types.TRAVERSAL_TYPES): unlike a
    # pure hazard, it changes where the ninja can go. Two consequences drive the
    # design:
    #
    # * They can BLOCK. A one-way's whole point is being passable one way only,
    #   and a bounce block is an 18px solid in a 24px cell. Both are proved
    #   harmless by the same pessimistic flood fill the mines use -- a one-way is
    #   strictly more permeable than a wall, so a route that survives with its
    #   cell walled off survives with the real entity there.
    # * They move the agent's SCOPE. Per `constants.entity_types`, ONE_WAY is
    #   phase 2 (the reachability graph already models it in
    #   `graph/reachability/entity_mask.py`), but BOUNCE_BLOCK and LAUNCH_PAD are
    #   phase 3 -- the phase that changes the cost-to-go field, and so the one
    #   that can invalidate PBRS, the demo corpus and the reverse curriculum at
    #   once. They are OFF by default for that reason, not because they misbehave.
    #
    # Placements follow the official corpus's measured tile alignment: one-ways
    # and launch pads flush on a tile boundary and horizontally centred, bounce
    # and boost pads on the cell centre.

    def _one_way_slots(self, cells, chambers):
        """Boundaries between two empty cells that can carry a floating platform.

        Orientation 6 (normal up) is the corpus majority at 51%, and the only
        orientation that cannot become a one-way *valve* across a corridor: you
        pass up through it and land on top, and in an open chamber you can always
        walk off its side. It still blocks DOWNWARD travel, which is exactly what
        the connectivity proof is for.
        """
        out = []
        for cx, cy in cells:
            # Keep two clear columns at each chamber edge so a platform can never
            # span a whole row -- there is always a way around its side.
            if not any(
                c.x1 + 2 <= cx <= c.x2 - 2 and c.y1 <= cy <= c.y2 for c in chambers
            ):
                continue
            if not (
                self._is_open(cx, cy)
                and self._is_open(cx, cy + 1)
                and self._is_open(cx, cy - 1)
            ):
                continue
            # Block the cell BELOW: that is the passage the platform closes to a
            # falling ninja.
            out.append(((cx, cy + 1), True, 11, cx + 1.5, cy + 2.0, 6))
        return out

    def _bounce_slots(self, cells, chambers):
        """Cells that can hold a free-standing bounce block.

        Wants air under it (a springboard, not a doorstop) and air above it, so
        the block reads as something to bounce off rather than a plug.
        """
        out = []
        for cx, cy in cells:
            if not self._is_open(cx, cy):
                continue
            if not (self._is_open(cx, cy - 1) and self._is_open(cx, cy + 1)):
                continue
            if not any(
                c.x1 + 1 <= cx <= c.x2 - 1 and c.y1 <= cy <= c.y2 for c in chambers
            ):
                continue
            out.append(((cx, cy), True, 17, cx + 1.5, cy + 1.5, 0))
        return out

    def _pad_slots(self, cells, etype: int):
        """Floor-mounted, non-blocking pads (launch pad 10 / boost pad 24).

        Both need a full solid tile under them and clear air above -- a launch
        pad firing into a ceiling 24px up is a trap, not a shortcut.
        """
        out = []
        for cx, cy in cells:
            if self.get_tile(cx, cy) != EMPTY or self.get_tile(cx, cy + 1) != SOLID:
                continue
            if not (self._is_open(cx, cy - 1) and self._is_open(cx, cy - 2)):
                continue
            if etype == 10:
                if not self._is_open(cx, cy - 3):
                    continue  # launch pads need real headroom to fire into
                out.append(((cx, cy), False, 10, cx + 1.5, cy + 2.0, 6))
            else:
                out.append(((cx, cy), False, 24, cx + 1.5, cy + 1.5, 0))
        return out

    def _place_traversal(self, chambers, occupied):
        """Place one-ways, bounce blocks, launch pads and boost pads."""
        # Pickiest first. A launch pad needs a solid floor AND three rows of
        # headroom to fire into; a one-way needs only air. Placing the permissive
        # types first lets them eat every cell the picky ones could have used, so
        # a fully-furnished preset would quietly ship zero pads.
        wanted = (
            (self._pad_slots, self.MIN_LAUNCH_PADS, self.MAX_LAUNCH_PADS, 10),
            (self._pad_slots, self.MIN_BOOST_PADS, self.MAX_BOOST_PADS, 24),
            (self._bounce_slots, self.MIN_BOUNCE_BLOCKS, self.MAX_BOUNCE_BLOCKS, None),
            (self._one_way_slots, self.MIN_ONE_WAYS, self.MAX_ONE_WAYS, None),
        )
        if not any(int(hi) > 0 for _fn, _lo, hi, _arg in wanted):
            return []

        ninja, switch, door = self._goal_cells
        goal_clear = int(self.MINE_GOAL_CLEARANCE)
        spacing = max(2, int(self.TRAVERSAL_MIN_SPACING))
        base_cells = self._mine_cells(chambers)  # same corridor-apron exclusions

        records = []
        for slot_fn, lo, hi, arg in wanted:
            target = self.rng.randint(int(lo), int(hi))
            if target <= 0:
                continue
            cells = [c for c in base_cells if c not in occupied]
            self.rng.shuffle(cells)
            slots = slot_fn(cells, arg) if arg is not None else slot_fn(cells, chambers)
            taken = 0
            for cell, blocks, etype, gx, gy, orient in slots:
                if taken >= target:
                    break
                cx, cy = cell
                if any(
                    abs(cx - ex) <= goal_clear and abs(cy - ey) <= goal_clear
                    for ex, ey in (ninja, switch, door)
                ):
                    continue
                # Spacing is per-TYPE: two one-ways in a row read as one long
                # platform, but a boost pad next to a one-way is a combo, not
                # clutter. Cross-type crowding is handled by ``occupied``.
                if any(
                    abs(cx - px) < spacing and abs(cy - py) < spacing
                    for (px, py), _b, t, _gx, _gy, _o in records
                    if t == etype
                ):
                    continue
                records.append((cell, blocks, etype, gx, gy, orient))
                occupied.add(cell)
                occupied.add((cx, cy - 1))
                occupied.add((cx, cy + 1))
                taken += 1
        return records

    # ------------------------------------------------------------ mobile hazards

    def _ray(self, cell: Tuple[int, int], step: Tuple[int, int], limit: int = 40):
        """Open cells from ``cell`` along ``step`` until rock stops the ray."""
        out = []
        x, y = cell
        for _ in range(limit):
            x, y = x + step[0], y + step[1]
            if not self._is_open(x, y):
                break
            out.append((x, y))
        return out

    def _snap(self, center_px: float, grid: float, offset: float) -> float:
        """Snap a pixel position onto an entity type's own placement lattice.

        A cell centre sits exactly BETWEEN two points of the 12px lattices, so
        rounding resolves every tie the same way and the class only ever occupies
        half its lattice -- mini drones would land on x%24==6 and never 18, where
        official uses both. Break the tie randomly instead.
        """
        if grid <= 0:
            return center_px
        k = (center_px - offset) / grid
        low = int(k // 1)
        frac = k - low
        if abs(frac - 0.5) < 1e-9:
            low += self.rng.choice((0, 1))
        else:
            low = round(k)
        return low * grid + offset

    def _emit_on_lattice(self, etype, grid, ox, oy, cell, orient=0, mode=0) -> None:
        """Place ``etype`` at ``cell``, snapped to the lattice its class uses.

        The lattice is PER ENTITY TYPE and getting it wrong is silent but total:
        a zap drone steps between 24px cell centres while a mini drone steps on a
        12px grid, so a mini placed on the zap lattice starts mid-step and its
        whole patrol is phase-shifted out of sync with anything the forecast
        observation channels predict. Values come straight from
        ``map_hazard_overlay.HAZARD_SPECS``, which measured them over every
        instance in the official corpus.
        """
        cx, cy = cell
        px = self._snap((cx + 1.5) * 24.0, grid, ox)
        py = self._snap((cy + 1.5) * 24.0, grid, oy)
        self.add_entity(etype, px / 24.0, py / 24.0, orient, mode)

    def _place_hazards(self, chambers, occupied: Set[Tuple[int, int]]) -> None:
        """Patrol drones in open volume, turrets/rockets on walls with sightlines."""
        n_drones = self.rng.randint(int(self.MIN_DRONES), int(self.MAX_DRONES))
        n_empl = self.rng.randint(int(self.MIN_EMPLACEMENTS), int(self.MAX_EMPLACEMENTS))
        if (n_drones <= 0 and n_empl <= 0) or not self._goal_cells:
            return
        # Imported here rather than at module scope: map_hazard_overlay pulls in
        # the planner via map_mine_overlay, which is far too heavy to load on
        # every `import map_cavern_chambers`.
        from .map_hazard_overlay import HAZARD_SPECS

        ninja = self._goal_cells[0]
        clear = int(self.HAZARD_SPAWN_CLEARANCE)
        sight = int(self.HAZARD_MIN_SIGHTLINE)
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))

        reachable = self._reachable_cells(ninja)
        cells = [c for c in self._mine_cells(chambers) if c not in occupied]
        self.rng.shuffle(cells)

        def far_from_spawn(cell):
            return max(abs(cell[0] - ninja[0]), abs(cell[1] - ninja[1])) >= clear

        # --- patrol drones: open volume, room to actually patrol --------------
        spec = HAZARD_SPECS["patrol_drone"]
        dtypes = [(t, w, g, ox, oy) for t, w, g, ox, oy in spec.types]
        modes = [m for m, _w in spec.modes]
        mweights = [w for _m, w in spec.modes]
        placed_h = []
        for cell in cells:
            if len(placed_h) >= n_drones:
                break
            if cell not in reachable or not far_from_spawn(cell):
                continue
            # Needs somewhere to go: a drone boxed into a one-cell pocket has no
            # patrol, and its wall-follow mode degenerates.
            if sum(1 for s in steps if self._is_open(*(cell[0] + s[0], cell[1] + s[1]))) < 3:
                continue
            if any(max(abs(cell[0] - p[0]), abs(cell[1] - p[1])) < 4 for p in placed_h):
                continue
            etype, _w, grid, ox, oy = self.rng.choices(
                dtypes, weights=[w for _t, w, _g, _ox, _oy in dtypes]
            )[0]
            self._emit_on_lattice(
                etype, grid, ox, oy, cell,
                orient=self.rng.choice(spec.orientations),
                mode=self.rng.choices(modes, weights=mweights)[0],
            )
            placed_h.append(cell)
            occupied.add(cell)

        # --- emplacements: mounted, with a sightline, but not onto the spawn ---
        placed_e = []
        for cell in cells:
            if len(placed_e) >= n_empl:
                break
            if cell in occupied or cell not in reachable or not far_from_spawn(cell):
                continue
            # Mounted against something -- official turrets/rockets sit on
            # surfaces, and a free-floating one reads as a bug.
            if not any(
                self.get_tile(cell[0] + s[0], cell[1] + s[1]) == SOLID for s in steps
            ):
                continue
            rays = {s: self._ray(cell, s) for s in steps}
            if max(len(r) for r in rays.values()) < sight:
                continue
            # A free shot at the spawn is an unavoidable death before the agent
            # has taken an action. The overlay can afford to allow this because
            # it replays a routed tape and classifies the outcome; a runtime
            # generator with no such gate cannot.
            if any(ninja in r for r in rays.values()):
                continue
            if any(max(abs(cell[0] - p[0]), abs(cell[1] - p[1])) < 5 for p in placed_e):
                continue
            name = self.rng.choices(
                ("turret", "rocket"), weights=list(self.EMPLACEMENT_WEIGHTS)
            )[0]
            espec = HAZARD_SPECS[name]
            etype, _w, grid, ox, oy = espec.types[0]
            self._emit_on_lattice(
                etype, grid, ox, oy, cell,
                orient=self.rng.choice(espec.orientations) if espec.orientations else 0,
            )
            placed_e.append(cell)
            occupied.add(cell)

        self.hazard_cells = [(c, "drone") for c in placed_h] + [
            (c, "emplacement") for c in placed_e
        ]

    # --------------------------------------------------------------------- gold
    #
    # Unlike every other entity here, gold takes GOLD_OFFSET_UNITS in
    # ``add_entity``, which lands an INTEGER grid coordinate on the CENTRE of its
    # cell -- so a cluster is laid out directly in grid units, and a half-integer
    # step is the 12px pitch the corpus favours.

    @staticmethod
    def _shape_offsets(rng, shape: str, n: int) -> List[Tuple[int, int]]:
        """Lattice offsets for one cluster shape, at most ``n`` points.

        Offsets are in LATTICE units; the caller scales them by the pitch. Shapes
        are deduped and truncated rather than forced to an exact count, so a ring
        that collapses onto itself degrades to a smaller ring instead of a blob.
        """
        pts: List[Tuple[int, int]] = []
        if shape == "grid":
            # Squarish factorisation, so 12 reads as 4x3 rather than 12x1.
            w = max(1, int(round(n ** 0.5)))
            if rng.random() < 0.35:
                w = max(1, min(n, w + rng.choice((-1, 1))))
            h = -(-n // w)
            pts = [(x, y) for y in range(h) for x in range(w)]
        elif shape == "triangle":
            row, y = 1, 0
            while len(pts) < n:
                for x in range(row):
                    pts.append((x - row // 2, y))
                row += 1
                y += 1
        elif shape == "ring":
            import math

            k = max(3, n)
            r = max(1, int(round(k / (2 * math.pi))) + 1)
            for i in range(k):
                a = 2 * math.pi * i / k
                pts.append((int(round(r * math.cos(a))), int(round(r * math.sin(a)))))
        elif shape == "arc":
            # A shallow parabola, or a straight run when the curvature is 0.
            curve = rng.choice((-1, 0, 0, 1))
            horizontal = rng.random() < 0.5
            for i in range(n):
                t = i - n // 2
                bend = (curve * t * t) // max(2, n // 2)
                pts.append((i, bend) if horizontal else (bend, i))
        else:  # "blob": a random connected growth
            pts = [(0, 0)]
            frontier = [(0, 0)]
            while len(pts) < n and frontier:
                bx, by = rng.choice(frontier)
                dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
                cand = (bx + dx, by + dy)
                if cand not in pts:
                    pts.append(cand)
                    frontier.append(cand)

        seen = set()
        out = []
        for p in pts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out[:n]

    def _gold_footprint(self, gx: float, gy: float) -> List[Tuple[int, int]]:
        """Generator cells a gold's 6px disc overlaps.

        A half-pitch coordinate sits on a tile boundary, so its disc straddles
        two cells (or four, on a corner). All of them have to be open, or the
        gold is half-buried in rock.
        """
        cells = set()
        for ox in (0.25, 0.75):
            for oy in (0.25, 0.75):
                cells.add((int(gx + ox), int(gy + oy)))
        return sorted(cells)

    def _place_gold(self, chambers, occupied: Set[Tuple[int, int]]) -> int:
        """Scatter shaped gold clusters through the reachable chambers.

        Gold is the one entity here that cannot obstruct anything, so it needs no
        connectivity proof -- but it does need to be REACHABLE, since gold walled
        into a pocket is pure observation noise. Every cell of every cluster is
        required to be in the spawn-connected region.
        """
        clusters = self.rng.randint(
            int(self.MIN_GOLD_CLUSTERS), int(self.MAX_GOLD_CLUSTERS)
        )
        if clusters <= 0 or not self._goal_cells:
            return 0

        shapes = ("grid", "triangle", "ring", "arc", "blob")
        reachable = self._reachable_cells(self._goal_cells[0])
        anchors = [c for c in self._mine_cells(chambers) if c in reachable]
        self.rng.shuffle(anchors)

        placed = 0
        used: Set[Tuple[int, int]] = set()
        for cx, cy in anchors:
            if placed >= clusters:
                break
            n = self.rng.randint(
                int(self.MIN_GOLD_PER_CLUSTER), int(self.MAX_GOLD_PER_CLUSTER)
            )
            shape = self.rng.choices(shapes, weights=list(self.GOLD_SHAPE_WEIGHTS))[0]
            pitch = 0.5 if self.rng.random() < float(self.GOLD_HALF_PITCH_PROB) else 1.0
            offsets = self._shape_offsets(self.rng, shape, n)

            coords = []
            ok = True
            for dx, dy in offsets:
                gx, gy = cx + dx * pitch, cy + dy * pitch
                for cell in self._gold_footprint(gx, gy):
                    if (
                        cell not in reachable
                        or cell in occupied
                        or cell in used
                        or self.get_tile(*cell) != EMPTY
                    ):
                        ok = False
                        break
                if not ok:
                    break
                coords.append((gx, gy))
            # All-or-nothing: a half-placed ring is not a ring, and a cluster
            # clipped by a wall is exactly the "scattered" regime the corpus
            # says official levels do not have.
            if not ok or len(coords) < int(self.MIN_GOLD_PER_CLUSTER):
                continue
            for gx, gy in coords:
                self.add_entity(2, gx, gy)
                for cell in self._gold_footprint(gx, gy):
                    used.add(cell)
                    # Keep a 1-cell skirt so the next cluster reads as separate.
                    for ddx in (-1, 0, 1):
                        for ddy in (-1, 0, 1):
                            used.add((cell[0] + ddx, cell[1] + ddy))
            self.gold_clusters.append((shape, len(coords), (cx, cy)))
            placed += 1
        return placed

    def _reachable_cells(self, start: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Empty cells 4-connected to ``start``."""
        if self.get_tile(*start) != EMPTY:
            return set()
        seen = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if nxt not in seen and self._is_open(*nxt):
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def _drop_severing(self, records):
        """Remove placements whose cells are load-bearing for the route.

        Treating a mine or a one-way as a solid wall is pessimistic -- the ninja
        can hop the one and pass up through the other -- but that is exactly what
        makes it a *guarantee*: if the three goals still connect with every such
        cell blocked, no arrangement of toggle states or approach angles can seal
        the level. Cheap enough to run at generation time (one flood fill, plus
        one per offender), unlike the forked d* route certificate the overlay
        generators use.
        """
        if not records:
            return records
        goals = list(self._goal_cells)

        def blocking(rs):
            return {cell for cell, blocks, _t, _gx, _gy, _o in rs if blocks}

        if self._cells_connected(goals, blocking(records)):
            return records
        # Drop candidates one at a time, most recently added first, until the
        # route reopens. Ordering by recency keeps the earlier (better spread)
        # placements and sheds the ones that closed the last gap.
        keep = list(records)
        while keep:
            keep.pop()
            if self._cells_connected(goals, blocking(keep)):
                break
        return keep

    def _populate(self, chambers: Sequence[_Chamber]) -> None:
        """Traversal furniture, then mines, then one connectivity proof over both."""
        if not self._goal_cells:
            return
        occupied: Set[Tuple[int, int]] = set()
        records = self._place_traversal(chambers, occupied)
        records += self._place_mines(chambers, occupied)
        records = self._drop_severing(records)
        for _cell, _blocks, etype, gx, gy, orient in records:
            self.add_entity(etype, gx, gy, orient)
        self.mine_cells = [
            cell for cell, _b, t, _gx, _gy, _o in records if t in (1, 21)
        ]
        self.entity_cells = [
            (cell, t) for cell, _b, t, _gx, _gy, _o in records if t not in (1, 21)
        ]
        # Gold last: it never obstructs, so it sits outside the connectivity
        # proof, but it must not land on top of anything already placed.
        for cell, _b, _t, _gx, _gy, _o in records:
            occupied.add(cell)
        # Hazards before gold: both are non-blocking, but a drone parked inside a
        # gold cluster is worse than a gold cluster missing one cell.
        self._place_hazards(chambers, occupied)
        self._place_gold(chambers, occupied)

    # ---------------------------------------------------------------- validation
    def _cells_connected(
        self,
        cells: Sequence[Tuple[int, int]],
        blocked: Optional[Set[Tuple[int, int]]] = None,
    ) -> bool:
        """4-connected flood fill over empty tiles from the first cell.

        A coarse necessary condition (it ignores gravity), used to reject
        layouts whose corridor routing failed outright, and -- with ``blocked``
        -- to prove a set of mines cannot seal the route. Physics-level
        verification is ``solvability.is_map_solvable``, run by the dataset
        pipeline.
        """
        start = cells[0]
        if self.get_tile(*start) != EMPTY:
            return False
        blocked = blocked or set()
        seen = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if nxt in seen or nxt in blocked or not self._is_open(*nxt):
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return all(c in seen for c in cells[1:])

    # -------------------------------------------------------------------- build
    def _attempt(self) -> bool:
        self.reset()
        self._fill_solid()
        self._corridor_cells: Set[Tuple[int, int]] = set()
        self._shaft_cells: Set[Tuple[int, int]] = set()
        self._ramp_columns: Set[Tuple[int, int]] = set()
        self._last_transition_cols = 0
        self._goal_cells = None
        self.chambers = []
        self.mine_cells: List[Tuple[int, int]] = []
        self.entity_cells: List[Tuple[Tuple[int, int], int]] = []
        self.gold_clusters: List[Tuple[str, int, Tuple[int, int]]] = []
        self.hazard_cells: List[Tuple[Tuple[int, int], str]] = []

        chambers = self._chambers_from_leaves(self._partition())
        if len(chambers) < 2:
            return False
        for c in chambers:
            self._carve_chamber(c)

        edges = self._connect(chambers, self._build_graph(chambers))
        if len(edges) < len(chambers) - 1:
            return False  # a link failed to route; the graph may be split

        if not self._place_goals(chambers, edges):
            return False
        if not self._cells_connected(list(self._goal_cells)):
            return False

        # Smoothing before mines so it sees the finished 0/1 mask (and so the
        # goal cells it might touch are already fixed -- it only opens space
        # anyway); mines after it, so they anchor to the final surfaces.
        self._smooth()
        self._populate(chambers)
        self.chambers = chambers
        self.chamber_edges = edges
        return True

    def _fallback(self) -> None:
        """Two boxes on one floor line joined by a straight corridor.

        Only reached if every randomized layout failed, which should not happen
        for sane configs -- but a generator that can raise inside a training
        reset is a worse failure than a plain level.
        """
        self.reset()
        self._fill_solid()
        self._corridor_cells = set()
        self._shaft_cells = set()
        self._ramp_columns = set()
        self.mine_cells = []
        self.entity_cells = []
        self.gold_clusters = []
        self.hazard_cells = []
        mid = MAP_TILE_HEIGHT // 2
        left = _Chamber(3, mid - 3, 13, mid + 2, "box")
        right = _Chamber(28, mid - 3, 38, mid + 2, "box")
        for c in (left, right):
            self.set_empty_rectangle(c.x1, c.y1, c.x2, c.y2)
        for x in range(14, 28):
            for y in (mid + 1, mid + 2):
                self.set_tile(x, y, EMPTY)
                self._corridor_cells.add((x, y))
        self.set_ninja_spawn(5, mid + 2, 1)
        self.add_entity(3, 36, mid + 2, 0, 0, 30, mid + 2)
        self.chambers = [left, right]
        self.chamber_edges = [(0, 1)]

    def generate(self, seed: Optional[int] = None) -> "MapCavernChambers":
        if seed is not None:
            self.rng.seed(seed)
        for _ in range(max(1, int(self.LAYOUT_ATTEMPTS))):
            if self._attempt():
                return self
        self._fallback()
        return self
