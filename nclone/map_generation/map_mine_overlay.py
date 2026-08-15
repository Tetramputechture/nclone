"""Dress an ARBITRARY base generator's level with an official-like minefield.

WHY THIS GENERATOR EXISTS. Measured 2026-08-05 across the shipped corpus and the
official S* tabs, mine geometry in training data does not resemble mine geometry
in the levels we are scored on:

    corpus                  med live mines   crowd p50 / p90 / max
    pool/MINE_NAVIGATION          1              1 /  4 /   8
    pool/DENSE_COMPOSITE          3              1 /  3 /   9
    pool/PLATFORMING             55             16 / 16 /  16   (uniform border row)
    official/S*                  36              8 / 21 / 123

"crowd" = other mines inside ``MINE_PROXIMITY_RANGE_PX`` (96). Two consequences:
the agent is scored on a distribution it never trains on, and the 4-slot
``mine_proximity`` observation is contested for 83.2% of official mines (hiding a
median of 5 more) versus 8-15% of procgen ones.

The existing generators cannot close that gap by retuning. ``mine_path`` puts
every mine on ONE corridor floor row at ``min_mine_spacing`` = 2 tiles (48 px) --
unchanged across all four tiers -- which caps crowd at 4 REGARDLESS of mine count.
So this generator does not add another bespoke minefield layout; it adds mines to
whatever any other generator built, which makes the whole registry reusable and
keeps morphological variety (slopes, shafts, mazes, platforms, loops) orthogonal
to hazard density.

A second measured fact drives the tier design: official mine density is FLAT
across the game's own difficulty ordering (crowd p50 = 8 in every quintile, from
the easiest fifth to the hardest). Mine density is a constant background property
of authored levels, not a difficulty axis. So the tiers here ramp the BASE level's
morphology and only mildly widen the density band -- they do not ramp density from
near-zero, which is what makes the current curriculum's T4 easier than official's
T1.

SOLVABILITY IS CERTIFIED BY ROUTING THE FINISHED LEVEL. Mines are overlaid at full
density, then ``plan_dstar_guided_fast`` must actually win the level as emitted;
the density ladder is descended only when it cannot. The returned tape is then
replayed on a fresh sim as an independent check, and kept (``route_actions``) so
every emitted level ships a validated route -- free BC data.

A NOTE ON THE DESIGN THIS REPLACED, because the history is instructive. An earlier
version planned on the MINE-FREE base and carved a clearance tube around the winning
trajectory, placing mines only outside it -- winnable by geometry rather than by
search. That existed because overlaid levels appeared unroutable (a 22-rung trace
certified nothing at all: routed=0, gate2_fail=22). The real cause was a
save/restore bug in the native sim: a snapshot of ANY level holding an armed mine
failed validation, so restore mutated nothing and returned false, and since callers
ignore that bool the incremental search silently degenerated into one drifting
trajectory. ``mine_overlay`` places ~20% armed mines, so it tripped this on nearly
every candidate. With that fixed, full-density overlaid levels route directly in
2-5 s (measured 5/5), so the tube was removed -- along with the artefact it caused:
carving a mine-free corridor pushed mines off the walkable floor, leaving 13.6% of
them ground-supported against official's 33.9%.

The lesson worth keeping: a gate that rejects everything is not evidence that the
thing being gated is impossible. Check the gate.

TWO FAILURE MODES THAT ARE NOT THE SAME, and the ladder must not conflate them: a
level that is genuinely unroutable should thin, whereas a planner TIMEOUT says
nothing about the level and thinning on it is exactly how a density generator
biases itself sparse. ``_route_level`` logs which occurred.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .map import Map

logger = logging.getLogger(__name__)


def _route_worker(map_path: str, macro: int, h_weight: float,
                  max_expansions: int, max_frames: int, q) -> None:
    """Child-process entry point for one base plan. Puts a result dict or None.

    THE FIRST TWO LINES ARE LOAD-BEARING WHEN THE PARENT HAS A WINDOW. Under
    ``fork`` this child inherits the parent's SDL state, including a live X11
    connection when it is called from the interactive dev tool. ``setdefault``
    was a no-op there -- the dev tool has already set ``SDL_VIDEODRIVER`` -- so
    the child kept the inherited display and tearing it down on exit took the
    PARENT'S WINDOW with it. From the user's side that is "pressing Next
    generator closed the tool", with no exception anywhere to catch.
    """
    import os as _os

    # FORCE, not setdefault: the point is to override whatever the parent chose.
    _os.environ["SDL_VIDEODRIVER"] = "dummy"
    try:
        import pygame as _pygame

        # Drop the inherited display before anything can touch it. Harmless when
        # there is none (the dataset builder, tests) -- `quit` on an
        # uninitialised display is a no-op.
        if _pygame.display.get_init():
            _pygame.display.quit()
    except Exception:  # noqa: BLE001 - pygame is optional for this path
        pass
    try:
        from ..value_function.tools.fast_route_solver import plan_dstar_guided_fast

        res = plan_dstar_guided_fast(map_path, macro=macro, h_weight=h_weight,
                                     max_expansions=max_expansions,
                                     max_frames=max_frames)
        if not res.get("solved"):
            q.put({"solved": False, "reason": "unsolved"})
            return
        # Ship only what the parent needs; the full result carries unpicklable
        # solver state.
        q.put({"solved": True, "actions": list(res.get("actions") or []),
               "frames": res.get("frames"),
               "frame_tape": list(res.get("frame_tape") or [])})
    except Exception as exc:
        # Distinguish a CRASH from an honest "no route". Collapsing both to None
        # hid a planner exception behind a 0-second "no route" on mine_maze bases
        # and made an immediate throw look like a search result. Anything that
        # takes ~0 s to "fail" is a throw, and the reason has to survive the
        # process boundary to be diagnosable at all.
        import traceback as _tb

        q.put({"solved": False, "reason": "exception",
               "error": f"{type(exc).__name__}: {exc}"[:200],
               "traceback": _tb.format_exc()[-1500:]})


def _plan_with_hard_timeout(map_path: str, *, macro: int, h_weight: float,
                            max_expansions: int, max_frames: int,
                            timeout_s: float):
    """Run one base plan in a child process and KILL it if it overruns.

    A process boundary is the only cap that actually holds here. The cost lives in
    ``LatticeDistanceSource``'s native precompute, where a SIGALRM handler cannot
    run between bytecodes -- measured, a base ran 386 s under a 10 s in-process
    cap -- and ``max_expansions`` does not bind it either (60000 -> 8000 moved
    352 s to 348 s) because the search itself is 0.0 s.

    The trigger is the generated LEVEL, not the generator: long thin flat corridors
    blow up the surface graph, and several flat-run presets can produce one. So this
    cannot be fixed by excluding specs -- it needs a real timeout.

    Returns the result dict, or None on timeout/failure (caller treats both as
    "this base is not usable", which is correct -- there are 15+ others).
    """
    import multiprocessing as mp

    # `fork`, deliberately, and NOT `spawn` even though this child inherits the
    # parent's SDL state. Spawn re-imports the parent's `__main__`, so any entry
    # point that is not import-safe makes the child die on startup -- and the
    # failure surfaces here as `None`, i.e. "no route", which is indistinguishable
    # from a genuinely unroutable level. Measured: switching to spawn turned a
    # working route into a 0.2 s silent failure. A silent wrong answer is worse
    # than the crash it was meant to prevent.
    #
    # The inherited-display hazard is handled inside `_route_worker` instead
    # (force the dummy driver, drop the inherited display), and interactive
    # callers should set `REQUIRE_ROUTE_CERTIFICATE = False` so this is never
    # reached from an event loop at all.
    try:
        ctx = mp.get_context("fork")
    except (ValueError, RuntimeError):
        return None
    q = ctx.Queue()
    # daemon=False so this still works when the dataset builder already has us
    # inside a worker process.
    p = ctx.Process(target=_route_worker,
                    args=(map_path, macro, h_weight, max_expansions, max_frames, q),
                    daemon=False)
    p.start()
    try:
        p.join(float(timeout_s))
        if p.is_alive():
            p.terminate()
            p.join(5.0)
            if p.is_alive():
                p.kill()
                p.join(5.0)
            logger.debug("mine_overlay: base plan killed after %.0fs", timeout_s)
            return None
        try:
            return q.get_nowait()
        except Exception:
            return None
    finally:
        if p.is_alive():
            p.kill()


PX_PER_TILE = 24.0
# Half tiles (2-5) are only partially solid; requiring a fully EMPTY tile is the
# conservative choice for placement and avoids mines fused into slope geometry.
EMPTY_TILE = 0
TOGGLE_MINE = 1
ARMED_MINE = 21
MINE_TYPES = frozenset({TOGGLE_MINE, ARMED_MINE})

# map_data layout (see Map.from_map_data)
TILE_START, TILE_END = 184, 1150
ENT_START, ENT_STRIDE = 1235, 5
EXIT_DOOR_COUNT_IDX = 1156
SPAWN_X_IDX, SPAWN_Y_IDX = 1231, 1232
GRID_W, GRID_H = 42, 23

EXIT_SWITCH_TYPE, EXIT_DOOR_TYPE = 4, 3


class MapMineOverlay(Map):
    """Generate a base level from another generator, then overlay a minefield."""

    # ------------------------------------------------------------------
    # Config (GeneratorFactory upper-cases every dataclass field onto these)
    # ------------------------------------------------------------------
    # "generator:preset" specs sampled per level. Deliberately broad -- hazard
    # density is orthogonal to morphology, so every traversable base shape is
    # fair game and variety is free.
    # Prefer the TIERED presets: they are the ones the curriculum is built from
    # and are sized for the trainer, unlike the legacy medium/complex presets.
    BASE_SPECS: Sequence[str] = (
        "jump_platforms:platforms_t2",
        # horizontal_corridor is deliberately ABSENT. It is a thin flat corridor,
        # so once the clearance tube is carved there is almost no off-path floor
        # left and it yielded 0-1 mines at EVERY density (rejected as TOO_FEW).
        # It also costs ~380-480 s per level -- all of it inside the
        # LatticeDistanceSource precompute, measured spec-wide across fresh
        # seeds, not in the search. Zero yield at ~400 s a try.
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
        # mine_maze is deliberately NOT a base here: it already ships its own
        # mine columns, so overlaying more would double-dose the density. It
        # enters MINE_FIELD directly as its own bucket entry instead.
    )
    # Placement lattice. 12 px is official's median mine-to-mine spacing and the
    # native pitch of the compose editor's cluster helpers.
    LATTICE_PX: float = 12.0
    # Official mix is 79.5% toggle / 20.5% armed-at-spawn.
    ARMED_FRACTION: float = 0.205
    MIN_CLUSTERS: int = 3
    MAX_CLUSTERS: int = 9
    # A 24 px run yields crowd 8 (four neighbours each side inside 96 px), which
    # is official's median exactly; 12 px yields 16 and supplies the upper tail.
    RUN_PITCH_CHOICES_PX: Sequence[float] = (12.0, 24.0, 24.0, 36.0)
    RUN_LEN_MIN: int = 3
    RUN_LEN_MAX: int = 14
    PATCH_RADIUS_MIN_PX: float = 12.0
    PATCH_RADIUS_MAX_PX: float = 42.0
    PATCH_PITCH_CHOICES_PX: Sequence[float] = (12.0, 24.0)
    SCATTER_MIN: int = 0
    SCATTER_MAX: int = 6
    # Relative cluster-kind weights: (run, patch, scatter).
    CLUSTER_WEIGHTS: Sequence[float] = (0.55, 0.30, 0.15)
    # Tried in order; the first that keeps a mine-free route wins. A level that
    # only survives at a trivial density is rejected outright (MIN_ACCEPT_MINES)
    # so the ladder cannot quietly degrade the bucket into sparse levels.
    # Mine count alone does NOT control the statistic we care about. Compact
    # bases (single_chamber, slopes) packed 38-48 mines into a small walkable
    # region and landed at crowd p50 35-40 against official's 8 -- every mine
    # inside every other's 96 px window, i.e. one blob in an otherwise bare
    # level. Official spreads a similar COUNT over a much larger area. So the
    # ladder is walked until the level's own crowd lands in band, which is what
    # makes it a calibration rather than a guess.
    # Mine MOUNTING, measured over 107k mines in the official S* tabs:
    #   33.9% have a solid tile within one tile below ("supported")
    #   27.9% never land at all, median drop 36 px
    # Reachability-graph nodes are overwhelmingly airborne, so uniform anchoring
    # gave 1.9% supported / 56.3% never-landing -- floating minefields official
    # never shows. Bias cluster anchors so the emitted mix matches.
    SUPPORT_PROBE_PX: float = 24.0
    SEAT_MAX_DROP_PX: float = 120.0
    SUPPORTED_ANCHOR_FRACTION: float = 0.34
    MINE_PROXIMITY_RANGE_PX: float = 96.0
    # Upper bound only. Official p50 is 8 and p90 is 20; allowing up to 20 keeps
    # levels that are harder than official's median without admitting blobs.
    CROWD_P50_MAX: float = 20.0
    DENSITY_LADDER: Sequence[float] = (1.0, 0.75, 0.5, 0.3, 0.15)
    MIN_ACCEPT_MINES: int = 8
    MAX_BASE_RETRY: int = 12
    # Physics gate. A SOLVED route returns in ~0-1 s; it is the FAILURES that are
    # slow (the search runs to the expansion cap), so the cap is the real cost
    # knob. 60k keeps a rejected candidate near ~10 s instead of ~50 s while
    # still clearing every base level tested.
    ROUTE_MACRO: int = 6
    ROUTE_H_WEIGHT: float = 2.0
    # BEST-EFFORT ONLY -- this cap does NOT reliably fire, so do not rely on it
    # to bound generation time. Measured: horizontal_corridor ran 386 s to a
    # successful route under a 10 s cap, because plan_dstar_guided_fast blocks
    # inside long native calls and Python signal handlers only run between
    # bytecode instructions. An earlier check appeared to confirm the cap, but it
    # only exercised bases returning in 0.5-2.6 s, which cannot distinguish a
    # working timer from a dead one. ROUTE_MAX_EXPANSIONS is the bound the
    # planner actually honours; this is a backstop for the Python-side phases.
    ROUTE_TIME_CAP_S: float = 10.0
    ROUTE_MAX_EXPANSIONS: int = 60000
    ROUTE_MAX_FRAMES: int = 2600
    REQUIRE_ROUTE_CERTIFICATE: bool = True
    # Keep mines off the spawn and off the objectives themselves. The stock gate
    # enforces the spawn radius too; filtering here avoids burning ladder steps.
    SPAWN_CLEARANCE_PX: float = 30.0
    OBJECTIVE_CLEARANCE_PX: float = 18.0

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self._base_spec: Optional[str] = None
        self._achieved_mines: int = 0
        self._achieved_density: float = 0.0
        # Winning macro-action tape from the physics gate (None if the gate is
        # disabled). Kept so callers can emit BC demos without re-solving.
        self.route_actions: Optional[List[int]] = None
        self.route_frames: Optional[int] = None

    # ------------------------------------------------------------------
    # map_data helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tile_at(md: Sequence[int], px: float, py: float) -> Optional[int]:
        """Tile id under a pixel position, or None outside the grid.

        Entity/graph pixel coordinates are offset ONE TILE from the tile grid's
        origin. Verified rather than assumed: gold must sit in free space, and
        across 200 official maps it lands in a solid tile 0.70% of the time with
        this offset versus 29-46% under every other ordering/offset tried.
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
        out = []
        for _i, e in cls._iter_entities(md):
            if e[0] in (EXIT_SWITCH_TYPE, EXIT_DOOR_TYPE):
                out.append((e[1] * 6.0, e[2] * 6.0))
        return out

    @staticmethod
    def _worst_case(md: Sequence[int]) -> List[int]:
        """Copy of ``md`` with EVERY mine rewritten to permanently-armed type 21.

        This is the whole solvability argument. ``is_map_solvable`` only blocks on
        mines that are deadly NOW, and type-1 toggle mines start safe, so the
        stock gate would certify a wall of them. Validating this copy instead
        proves a route exists that never enters any mine's kill radius.
        """
        out = list(md)
        i, n = ENT_START, len(out)
        while i + ENT_STRIDE <= n:
            t = out[i]
            if t == 0:
                break
            if t in MINE_TYPES:
                out[i] = ARMED_MINE
            i += ENT_STRIDE
        return out

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------
    def _reachable_anchors(self, md: Sequence[int]) -> List[Tuple[float, float]]:
        """Lattice-snapped positions the ninja can actually occupy.

        Anchoring on the reachability graph (rather than on any free tile) keeps
        every cluster near a surface the agent will traverse, so the mines are
        hazards on the route rather than decoration in an unreachable void. The
        graph's node keys are in the SAME pixel frame as entity coordinates and
        the spawn, so no conversion is needed here.
        """
        from .solvability import _build_level_data, _reachable_set

        try:
            level_data = _build_level_data(list(md))
            reachable, _adj = _reachable_set(level_data)
        except Exception as exc:
            logger.debug("mine_overlay: reachability failed: %s", exc)
            return []
        step = float(self.LATTICE_PX)
        snapped: Set[Tuple[float, float]] = set()
        for node in reachable:
            if not (isinstance(node, tuple) and len(node) >= 2):
                continue
            nx = round(float(node[0]) / step) * step
            ny = round(float(node[1]) / step) * step
            snapped.add((nx, ny))
        return sorted(snapped)

    def _has_support(self, md: Sequence[int], x_px: float, y_px: float) -> bool:
        """Is there a solid tile within ``SUPPORT_PROBE_PX`` straight below?

        Official mines are a MIX: 33.9% sit on a surface, 27.9% never land at all
        (measured over 107k mines in the S* tabs). Reachability-graph nodes are
        overwhelmingly airborne traversal states, so anchoring uniformly on them
        produced 1.9% supported / 56.3% never-landing -- a level of pure floating
        minefields that official never shows. This predicate is what lets
        ``_sample_mines`` hit the official mix on purpose.
        """
        d = 0.0
        while d <= float(self.SUPPORT_PROBE_PX):
            d += 6.0
            t = self._tile_at(md, x_px, y_px + d)
            if t is not None and t != EMPTY_TILE:
                return True
        return False

    def _cluster_positions(
        self, rng: random.Random, anchor: Tuple[float, float], kind: str
    ) -> List[Tuple[float, float]]:
        ax, ay = anchor
        if kind == "run":
            pitch = rng.choice(list(self.RUN_PITCH_CHOICES_PX))
            n = rng.randint(int(self.RUN_LEN_MIN), int(self.RUN_LEN_MAX))
            # Runs follow a surface, so they are horizontal; a vertical variant
            # would mostly land inside geometry.
            x0 = ax - (n // 2) * pitch
            return [(x0 + k * pitch, ay) for k in range(n)]
        if kind == "patch":
            pitch = rng.choice(list(self.PATCH_PITCH_CHOICES_PX))
            r = rng.uniform(float(self.PATCH_RADIUS_MIN_PX),
                            float(self.PATCH_RADIUS_MAX_PX))
            out = []
            steps = int(r // pitch) + 1
            for iy in range(-steps, steps + 1):
                for ix in range(-steps, steps + 1):
                    px, py = ax + ix * pitch, ay + iy * pitch
                    if (px - ax) ** 2 + (py - ay) ** 2 <= r * r:
                        out.append((px, py))
            return out
        return [anchor]

    def _sample_mines(
        self,
        rng: random.Random,
        md: Sequence[int],
        anchors: Sequence[Tuple[float, float]],
        density: float,
    ) -> List[Tuple[float, float, int]]:
        if not anchors:
            return []
        spawn = (md[SPAWN_X_IDX] * 6.0, md[SPAWN_Y_IDX] * 6.0)
        objectives = self._objective_positions(md)
        sp_c2 = float(self.SPAWN_CLEARANCE_PX) ** 2
        ob_c2 = float(self.OBJECTIVE_CLEARANCE_PX) ** 2

        kinds = ("run", "patch", "scatter")
        weights = list(self.CLUSTER_WEIGHTS)
        n_clusters = rng.randint(int(self.MIN_CLUSTERS), int(self.MAX_CLUSTERS))
        n_clusters = max(1, int(round(n_clusters * density)))

        # Split anchors by ground support so the emitted mix can be steered to
        # official's (see SUPPORTED_ANCHOR_FRACTION). Uniform sampling over the
        # reachability graph lands almost everything in mid-air.
        grounded = [a for a in anchors if self._has_support(md, a[0], a[1])]
        airborne = [a for a in anchors if a not in set(grounded)] or list(anchors)
        p_ground = float(self.SUPPORTED_ANCHOR_FRACTION)

        def pick_anchor() -> Tuple[float, float]:
            pool = grounded if (grounded and rng.random() < p_ground) else airborne
            return pool[rng.randrange(len(pool))]

        chosen: Set[Tuple[float, float]] = set()
        for _ in range(n_clusters):
            kind = rng.choices(kinds, weights=weights, k=1)[0]
            for pos in self._cluster_positions(rng, pick_anchor(), kind):
                chosen.add(pos)
        for _ in range(rng.randint(int(self.SCATTER_MIN), int(self.SCATTER_MAX))):
            chosen.add(pick_anchor())

        out: List[Tuple[float, float, int]] = []
        for px, py in sorted(chosen):
            # Seat a share of the field on the surface below it. Biasing the
            # cluster ANCHOR is not enough on its own -- _cluster_positions then
            # scatters members up to PATCH_RADIUS_MAX_PX away, back into the air
            # (measured: anchor bias alone moved support only 0.019 -> 0.076
            # against official's 0.339). Dropping the mine onto the floor it is
            # already above is what actually reproduces official's mounting.
            if not self._has_support(md, px, py) and rng.random() < float(
                    self.SUPPORTED_ANCHOR_FRACTION):
                seated = self._seat_on_surface(md, px, py)
                if seated is not None:
                    px, py = seated
            if not self._is_free(md, px, py):
                continue
            if (px - spawn[0]) ** 2 + (py - spawn[1]) ** 2 < sp_c2:
                continue
            if any((px - ox) ** 2 + (py - oy) ** 2 < ob_c2 for ox, oy in objectives):
                continue
            mtype = ARMED_MINE if rng.random() < float(self.ARMED_FRACTION) else TOGGLE_MINE
            out.append((px, py, mtype))
        return out

    def _seat_on_surface(self, md: Sequence[int], px: float,
                         py: float) -> Optional[Tuple[float, float]]:
        """Drop a mine straight down onto the first surface beneath it.

        Returns the seated position, or None if nothing solid lies within
        ``SEAT_MAX_DROP_PX`` (leave those in the air -- official has 27.9% that
        never land, so seating everything would overshoot in the other
        direction).
        """
        d = 0.0
        while d <= float(self.SEAT_MAX_DROP_PX):
            d += 6.0
            t = self._tile_at(md, px, py + d)
            if t is None:
                return None
            if t != EMPTY_TILE:
                # Sit just above the surface, on the lattice.
                step = float(self.LATTICE_PX)
                return (px, round((py + d - step) / step) * step)
        return None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _build_base(self, spec: str, seed: int) -> Optional[List[int]]:
        from .generator_factory import GeneratorFactory

        gen_type, preset = spec.split(":", 1)
        try:
            gen = GeneratorFactory.create_from_preset(gen_type, preset, seed)
            gen.generate(seed=seed)
            md = list(gen.map_data())
        except Exception as exc:
            logger.debug("mine_overlay: base %s failed: %s", spec, exc)
            return None
        # Same structural gate the dataset builder applies before solvability.
        if md[EXIT_DOOR_COUNT_IDX] == 0:
            return None
        if md[SPAWN_X_IDX] == 0 and md[SPAWN_Y_IDX] == 0:
            return None
        return md

    def _to_tmp_map(self, md: Sequence[int]) -> str:
        """Write ``md`` to a temp binary .map and return the path.

        ``map_data`` carries the ninja orientation as -1, which is out of byte
        range and makes ``save_map`` refuse; the on-disk encoding for it is 255,
        so convert rather than dropping to a JSON the sim cannot load.
        """
        import os
        import tempfile

        from .compose.mapio import save_map

        blob = list(md)
        if blob[1233] == -1:
            blob[1233] = 255
        fd, path = tempfile.mkstemp(suffix=".map")
        os.close(fd)
        save_map(blob, path, fmt="binary")
        return path

    def _route_level(self, md: Sequence[int]) -> Optional[dict]:
        """Plan a winning route on a FINISHED level (mines included).

        Runs in a child process with a hard kill (see ``_plan_with_hard_timeout``):
        a healthy level routes in ~2-5 s even at full mine density, but a
        pathological one spends 380-480 s in the lattice precompute and no
        in-process cap can stop it.

        Returns None for BOTH "no route exists" and "timed out", which are not the
        same thing -- a timeout that silently thinned density is exactly how the
        old design biased itself sparse -- so each is logged distinctly.
        """
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
        except Exception as exc:
            logger.debug("mine_overlay: base route failed: %s", exc)
            return None
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        if res is None:
            # Killed by the wall-clock cap. This says NOTHING about whether the
            # level is winnable, so thinning density on it would be the sparse
            # bias all over again -- surface it rather than let it pass as a
            # quiet "too dense".
            logger.warning("mine_overlay: planner TIMED OUT after %.0fs -- density "
                           "ladder will thin on a level that may be perfectly "
                           "routable", float(self.ROUTE_TIME_CAP_S))
            return None
        if not res.get("solved"):
            # A planner CRASH is a defect to fix, not a level to skip quietly --
            # surface it at a level the caller will actually see.
            if res.get("reason") == "exception":
                logger.warning("mine_overlay: planner raised on level: %s",
                               res.get("error"))
                logger.debug("mine_overlay: planner traceback:\n%s",
                             res.get("traceback"))
            else:
                logger.debug("mine_overlay: level genuinely unroutable at this "
                             "density; thinning")
            return None
        # ``frame_tape`` is NOT always present: plan_dstar_guided_fast only
        # attaches it when it took a clinch leg. Treating its absence as "no
        # route" silently threw away solved bases -- measured, jump_platforms
        # solved in 0.7 s with no tape key and was discarded. Rebuild it from the
        # macro actions, which are always returned.
        tape = list(res.get("frame_tape") or [])
        if not tape:
            from ..value_function.tools.dstar_guided_spike import _tape_from_actions

            tape = list(_tape_from_actions(res.get("actions") or [],
                                           int(self.ROUTE_MACRO)))
        if not tape:
            return None
        res["frame_tape"] = tape
        return res

    def _path_points(self, md: Sequence[int], tape) -> Optional[List[Tuple[float, float]]]:
        """Ninja centre position entering every frame of ``tape`` on the base."""
        import os

        from ..value_function.tools.dstar_guided_spike import _replay_frames_native

        path = None
        try:
            path = self._to_tmp_map(md)
            pre, won, _gold, _gt, died = _replay_frames_native(path, tape)
        except Exception as exc:
            logger.debug("mine_overlay: base replay failed: %s", exc)
            return None
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        if not won or died is not None:
            return None
        return [(float(s[0]), float(s[1])) for s in pre]

    @classmethod
    def _base_mine_px(cls, md: Sequence[int]) -> List[Tuple[float, float, int]]:
        """Mines ALREADY present in the base level, in pixels.

        Bases are not always mine-free: ``mine_maze`` builds its own columns, and
        several composite presets carry mines too. The crowd budget is a property
        of the FINISHED level, so those have to be counted -- measuring only the
        overlay would let a dense base blow straight through CROWD_P50_MAX.
        """
        return [(e[1] * 6.0, e[2] * 6.0, e[0])
                for _i, e in cls._iter_entities(md) if e[0] in MINE_TYPES]

    def _crowd_p50(self, mines: Sequence[Tuple[float, float, int]]) -> float:
        """Median count of OTHER mines inside ``MINE_PROXIMITY_RANGE_PX``.

        This is the statistic the agent actually experiences through the 4-slot
        ``mine_proximity`` observation, and the one official S* levels are
        measured on (p50 8, p90 20). Targeting it directly is what stops compact
        bases from being certified as blobs.
        """
        if not mines:
            return 0.0
        r2 = float(self.MINE_PROXIMITY_RANGE_PX) ** 2
        counts = []
        for px, py, _ in mines:
            counts.append(sum(1 for qx, qy, _ in mines
                              if (px - qx) ** 2 + (py - qy) ** 2 < r2) - 1)
        counts.sort()
        return float(counts[len(counts) // 2])

    def _replay_certificate(self, md: Sequence[int], tape) -> bool:
        """Confirm the base tape still wins WITH the minefield installed.

        A replay is O(frames) with no search, so it costs orders of magnitude
        less than a fresh plan. It is also strictly stronger evidence than the
        worst-case reachability gate, which passed levels the physics could not
        actually thread.
        """
        pts = self._path_points(md, tape)
        return pts is not None

    def _adopt(self, md: Sequence[int], mines: Sequence[Tuple[float, float, int]]) -> None:
        """Install a base map_data plus the overlay onto self."""
        base = Map.from_map_data(list(md))
        self.tile_data = base.tile_data
        self.entity_data = base.entity_data
        self.entity_counts = base.entity_counts
        self.ninja_spawn_x = base.ninja_spawn_x
        self.ninja_spawn_y = base.ninja_spawn_y
        self.ninja_orientation = base.ninja_orientation
        for px, py, mtype in mines:
            # add_entity takes GRID coordinates and scales by 4 to map units
            # (x6 -> pixels), so a pixel position divides by 24. Sub-tile values
            # are intended: the 12 px lattice is half-tile.
            self.add_entity(mtype, px / PX_PER_TILE, py / PX_PER_TILE)

    def generate(self, seed: Optional[int] = None) -> "MapMineOverlay":
        if seed is not None:
            self.rng = random.Random(seed)
        rng = self.rng
        from .solvability import is_map_solvable

        base_seed = seed if seed is not None else rng.randrange(1 << 30)
        specs = list(self.BASE_SPECS)

        for attempt in range(int(self.MAX_BASE_RETRY)):
            s = base_seed + attempt * 1_000_000
            spec = specs[(base_seed + attempt) % len(specs)]
            md = self._build_base(spec, s)
            if md is None:
                continue
            # A base that is already unsolvable can never be rescued by adding
            # hazards, and would burn the whole density ladder proving it.
            if not is_map_solvable(md):
                continue
            anchors = self._reachable_anchors(md)
            if not anchors:
                continue

            base_mines = self._base_mine_px(md)
            arng = random.Random(s)
            # The ladder de-escalates only if a rung that fails is retried with
            # FEWER mines. Re-rolling a fresh layout per rung off a shared RNG
            # made mine count non-monotone (measured 37/22/12/42/27 down the
            # rungs), so sample once and thin that same set.
            full = self._sample_mines(arng, md, anchors, 1.0)
            # Thin by a RANDOM subset, not a prefix: _sample_mines returns
            # position-sorted mines, so full[:keep] would keep only the top-left
            # corner of the field and leave the rest of the level bare.
            order = list(range(len(full)))
            arng.shuffle(order)
            for density in self.DENSITY_LADDER:
                if density >= 1.0:
                    mines = list(full)
                else:
                    keep = max(0, int(round(len(full) * density)))
                    mines = [full[i] for i in sorted(order[:keep])]
                if len(mines) < int(self.MIN_ACCEPT_MINES):
                    continue
                # Reject blobs: keep descending the ladder until this level's own
                # crowd is official-like. Without this, compact bases certified at
                # crowd 35-40 (measured) -- the right mine COUNT squeezed into the
                # wrong area. The last rung is allowed through regardless so a
                # small base still yields a level rather than nothing.
                # Counted over base + overlay: a mine-bearing base (mine_maze)
                # would otherwise smuggle its own density past the budget.
                if (self._crowd_p50(base_mines + mines) > float(self.CROWD_P50_MAX)
                        and density > float(self.DENSITY_LADDER[-1])):
                    continue
                trial_map = Map.from_map_data(list(md))
                for px, py, mtype in mines:
                    trial_map.add_entity(mtype, px / PX_PER_TILE, py / PX_PER_TILE)
                trial = trial_map.map_data()

                # THE CERTIFICATE: route the FINISHED level, mines and all.
                #
                # This replaced a clearance-tube design that planned on the
                # mine-free base and then carved a corridor around the winning
                # trajectory. That existed only because overlaid levels appeared
                # unroutable -- which was the save/restore bug (a snapshot of any
                # level holding an ARMED mine failed validation and restored
                # NOTHING, so the search degenerated into one drifting
                # trajectory). With that fixed, full-density overlaid levels route
                # directly in 2-5 s, measured 5/5. Dropping the tube also removes
                # the distributional artefact it caused: carving a mine-free
                # corridor pushed mines off the walkable floor, leaving 13.6% of
                # them ground-supported against official's 33.9%.
                route = None
                if self.REQUIRE_ROUTE_CERTIFICATE:
                    route = self._route_level(trial)
                    if route is None:
                        # Descend the ladder. Now that the planner is correct a
                        # failure here is meaningful -- but a TIMEOUT still is
                        # not, so _route_level logs which one happened rather
                        # than silently thinning.
                        continue
                    # Belt and braces: the returned tape must actually win on a
                    # fresh sim. Cheap (no search) and independent of the
                    # planner's own bookkeeping.
                    if not self._replay_certificate(trial, route["frame_tape"]):
                        logger.warning("mine_overlay: planner claimed a route that "
                                       "does not replay; rejecting level")
                        continue
                self._adopt(md, mines)
                self._base_spec = spec
                self._achieved_mines = len(mines)
                self._achieved_density = density
                if route is not None:
                    self.route_actions = list(route.get("actions") or [])
                    self.route_frames = route.get("frames")
                return self

        raise RuntimeError(
            f"mine_overlay: no solvable minefield after {self.MAX_BASE_RETRY} base "
            f"attempts (seed={base_seed}); loosen DENSITY_LADDER or MIN_ACCEPT_MINES"
        )
