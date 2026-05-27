#!/usr/bin/env python3
"""Lattice audit: measure soundness / completeness / optimality of surface-graph d*.

The surface-graph backward-Dijkstra ``d*`` is the *macro* planner for large /
open maps (it computes cost-to-go over a fixed velocity-augmented lattice in one
polynomial pass — unlike the forward spike, which times out on open levels).
This tool is the measurement harness that judges whether that ``d*`` is actually
sound, complete, and optimal-on-lattice, **without** needing the spike as an
oracle (the spike can't run on open maps).

For a map it:
  1. Precomputes the surface graph in-process (extract -> build -> enumerate ->
     backward Dijkstra to switch and exit, with shortest-path trees).  No arc
     cache / waypoints (we only need routes), so RSS stays low.
  2. Extracts the spawn route: phase-0 (spawn -> switch) then phase-1 (switch
     landing -> exit), via the Dijkstra trees.
  3. Compiles the route to a per-frame input tape and **replays it continuously
     on the real native sim from spawn** (real velocity carries through arc
     boundaries — no per-arc bin-center teleport).
  4. Reports:
       * Reachability   — is spawn d* finite for both phases?
       * Self-consistency (optimality proxy) — d*_frames(spawn) vs the actual
         replayed frame count.  On no-mine maps edge weights ARE frames, so a
         large gap means the lattice route is non-executable / bin-aliased.
       * Executability  — does the replay WIN?  min clearance to deadly mines.
       * Per-arc drift  — distance between each arc's expected landing and the
         replay's actual position at that frame (localizes snap-through).
       * Clip telemetry — % of enumerated landings at a velocity-bin extreme.

On no-mine maps the self-consistency check is exact; on mine maps the current
``d*`` is mine-inflated (not frames), so the gap is confounded until the
``d*_frames`` / ``d*_potential`` split lands — this is reported explicitly.

Usage::

    python -m nclone.value_function.tools.lattice_audit \
        --map "nclone/maps/official/SL/000 arc" --workers 4
    python -m nclone.value_function.tools.lattice_audit \
        --map nclone/test_maps/minefield --workers 4 --spike
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import nclone_native as Nnative
from ...constants.physics_constants import MAX_JUMP_DURATION
from ..surface_extraction import extract_surfaces, SURFACE_FLOOR
from ..surface_graph import (
    build_graph,
    compute_cost_to_go_with_tree,
    compute_two_phase_cost_to_go,
    _build_reverse_adjacency,
    find_goal_nodes,
    node_id_to_parts,
)
from ..edge_enumeration import enumerate_edges, populate_graph_edges
from ..edge_risk import (
    apply_risk_weighting,
    DEFAULT_W_RISK,
    DEFAULT_W_LOOP,
    DEFAULT_RISK_CAP,
    DEFAULT_JUMP_TAX,
)
from ..route_extraction import (
    find_nearest_start_node,
    extract_and_simplify_route,
    ArcTarget,
)
from .physics_route_spike import (
    ACT,
    _load_map_data,
    _goal_positions,
    _switch_done,
    _MINE_STATE_DEADLY,
    _MINE_KILL_PX,
)


# ---------------------------------------------------------------------------
# Tape compilation
# ---------------------------------------------------------------------------
def _arc_to_tape(arc: ArcTarget) -> List[Tuple[int, int]]:
    """Compile one ArcTarget into a per-frame (h_input, jump_input) tape.

    For jumps the button is held ``jump_hold_frames`` (capped at
    MAX_JUMP_DURATION, matching enumeration) with ``h_input_during``, then
    released for the remaining frames with the post-release input.

    The post-release input is read from ``arc.h_input_after`` *if present*; the
    surface-graph edge does not store it today (only the primary during-hold
    input is kept), so we fall back to ``h_input_during``.  That fallback is
    precisely the snap-through bug this harness measures — once the edge stores
    ``h_after`` the same code path replays faithfully.
    """
    n = int(round(arc.arc_cost_frames))
    if n <= 0:
        return []
    h_d = int(arc.h_input_during)
    h_a = int(getattr(arc, "h_input_after", h_d))
    jh = int(arc.jump_hold_frames)
    if jh > 0:
        hold = min(jh, MAX_JUMP_DURATION, n)
        return [(h_d, 1)] * hold + [(h_a, 0)] * max(0, n - hold)
    return [(h_d, 0)] * n


@dataclass
class _Segment:
    tape: List[Tuple[int, int]]
    expect_x: float
    expect_y: float
    is_jump: bool
    cumulative_cost: float  # d* of this arc's destination (frames-or-inflated)


def _route_to_segments(route: List[ArcTarget]) -> List[_Segment]:
    segs: List[_Segment] = []
    for arc in route:
        segs.append(
            _Segment(
                tape=_arc_to_tape(arc),
                expect_x=arc.landing_x,
                expect_y=arc.landing_y,
                is_jump=arc.jump_hold_frames > 0,
                cumulative_cost=arc.cumulative_cost,
            )
        )
    return segs


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
@dataclass
class AuditResult:
    name: str
    n_points: int
    n_nodes: int
    n_edges: int
    pct_clipped: float
    has_mines: bool
    n_deadly: int
    # reachability
    spawn_d_switch: float
    spawn_d_exit_from_switch: float
    lattice_route_frames: float   # d* prediction (sum of arc costs along route)
    reachable: bool
    # executability
    replay_won: bool
    replay_died: bool
    replay_frames: int
    switch_frame: Optional[int]
    min_mine_clearance: float
    dist_switch_end: float
    dist_exit_end: float
    # closed-loop (position-tracking) executability
    cl_won: bool
    cl_died: bool
    cl_frames: int
    cl_switch_frame: Optional[int]
    cl_min_clear: float
    cl_reached: int
    # closed-loop + micro-search executability
    ms_won: bool
    ms_died: bool
    ms_frames: int
    ms_switch_frame: Optional[int]
    ms_min_clear: float
    ms_reached: int
    # self-consistency
    max_arc_drift: float
    mean_arc_drift: float
    n_arcs: int
    enum_time_s: float
    # optional spike oracle
    spike_frames: Optional[int] = None
    spike_solved: Optional[bool] = None


class LatticeAudit:
    def __init__(self, map_path: Optional[str] = None, workers: int = 4,
                 mine_inflation: bool = True,
                 map_data: Optional[List[int]] = None, name: Optional[str] = None,
                 use_native: bool = True,
                 w_risk: float = DEFAULT_W_RISK,
                 w_loop: float = DEFAULT_W_LOOP,
                 risk_cap: float = DEFAULT_RISK_CAP,
                 jump_tax: float = DEFAULT_JUMP_TAX,
                 momentum_model: str = "legacy"):
        self.map_path = map_path
        self.use_native = use_native
        # Velocity representation: "legacy" world-vx/vy bins (default) or
        # "tangential" along-surface speed + curved/slope node extraction
        # (see surface_extraction / edge_enumeration).  Tangential currently
        # runs the single-process Python enumeration (no native kernel yet).
        self.momentum_model = str(momentum_model)
        # Soft, bounded completion>loops>speed reweighting of the d* edge costs
        # (see edge_risk.apply_risk_weighting).  w_risk=w_loop=jump_tax=0 =>
        # pure frames.  jump_tax is an additive per-jump cost that discourages
        # wall-climb spam in favour of momentum-carrying routes.
        self.w_risk = float(w_risk)
        self.w_loop = float(w_loop)
        self.risk_cap = float(risk_cap)
        self.jump_tax = float(jump_tax)
        if map_data is not None:
            self.map_data = list(map_data)
            self.name = name or "inline"
        else:
            self.name = name or os.path.basename(map_path.rstrip("/"))
            self.map_data = _load_map_data(map_path)
        self.sim = Nnative.Simulator.load(self.map_data)
        self.workers = workers
        self.mine_inflation = mine_inflation
        self.spawn = self.sim.get_ninja_pos()
        self.switch, self.exit, mines = _goal_positions(self.sim)
        self.deadly = [(x, y) for (x, y, s) in mines if s == _MINE_STATE_DEADLY]
        self.all_mines = [(x, y) for (x, y, _s) in mines]

    def _cache_path(self) -> str:
        import hashlib
        from ..surface_graph import velocity_bin_centers
        from ..surface_extraction import SURFACE_FLOOR, SURFACE_WALL
        # Bin schedule affects every edge, so it must key the cache.
        sched = (velocity_bin_centers(SURFACE_FLOOR).tobytes()
                 + velocity_bin_centers(SURFACE_WALL).tobytes())
        # ``riskv`` bumps whenever the edge_risk reweighting FORMULA changes (the
        # post-enumeration transform), even though the enumerated edge set
        # (``v14-wallchain``) is unchanged.  ``jt`` is the additive jump tax.
        payload = (np.asarray(self.map_data, dtype=np.int64).tobytes()
                   + f"|infl={self.mine_inflation}|native={self.use_native}"
                     f"|wr={self.w_risk}|wl={self.w_loop}|rc={self.risk_cap}"
                     f"|jt={self.jump_tax}|riskv=2"
                     f"|mm={self.momentum_model}|mmv=1"
                     "|v14-wallchain".encode()
                   + sched)
        h = hashlib.md5(payload).hexdigest()[:12]
        # Persistent/shared cache location override so an offline corpus job and
        # the training workers read/write the same precomputed d* (keyed by the
        # map_data hash + planner params above).  Default stays /tmp.
        d = os.environ.get("NCLONE_LATTICE_CACHE_DIR", "/tmp/lattice_audit_cache")
        os.makedirs(d, exist_ok=True)
        # Filename keyed by the hash ALONE (not ``self.name``): the offline corpus
        # job and the training workers use different human-readable names for the
        # same level, so a name prefix would silently break cache sharing.  The
        # hash already uniquely keys (map_data + planner params + bin schedule).
        return os.path.join(d, f"{h}.pkl")

    def _append_cdj_trick_edges(self, graph):
        """Enumerate CDJ trick edges and append them to ``graph`` so the global
        d* can route THROUGH them (composing tricks anywhere / multiple times).

        Trick edges with a landing node become real graph edges (``src -> dst``,
        cost = tape frames) with the verbatim tape stored in
        ``graph.trick_tapes[edge_index]``.  Trick edges whose trajectory reaches a
        goal mid-air (no landing) are returned as ``(switch_air, exit_air)`` seeds
        for ``compute_two_phase_cost_to_go``.  Additive: a map with no firing
        corner gets no edges and the pipeline is unchanged.
        """
        from ..cdj_edges import enumerate_cdj_edges
        sw_air: dict = {}
        ex_air: dict = {}
        try:
            tedges = enumerate_cdj_edges(graph, self.sim, self.switch, self.exit)
        except Exception:
            return sw_air, ex_air
        if not tedges:
            return sw_air, ex_air
        graph.trick_tapes = dict(graph.trick_tapes or {})
        new_src, new_dst, new_w = [], [], []
        base = len(graph.edges_src)
        for te in tedges:
            if te.dst_node is not None and te.dst_node != te.src_node:
                idx = base + len(new_src)
                new_src.append(int(te.src_node))
                new_dst.append(int(te.dst_node))
                new_w.append(float(te.cost_frames))
                graph.trick_tapes[idx] = list(te.full_tape)
            if te.switch_contact_frame is not None:
                s = int(te.src_node)
                sw_air[s] = min(sw_air.get(s, te.switch_contact_frame),
                                int(te.switch_contact_frame))
            if te.exit_contact_frame is not None:
                s = int(te.src_node)
                ex_air[s] = min(ex_air.get(s, te.exit_contact_frame),
                                int(te.exit_contact_frame))
        if new_src:
            n = len(new_src)
            graph.edges_src = np.concatenate(
                [graph.edges_src, np.array(new_src, np.int32)])
            graph.edges_dst = np.concatenate(
                [graph.edges_dst, np.array(new_dst, np.int32)])
            graph.edges_weight = np.concatenate(
                [graph.edges_weight, np.array(new_w, np.float32)])
            if graph.edges_jump_hold is not None:
                graph.edges_jump_hold = np.concatenate(
                    [graph.edges_jump_hold, np.zeros(n, np.int8)])
            if graph.edges_h_input is not None:
                graph.edges_h_input = np.concatenate(
                    [graph.edges_h_input, np.zeros(n, np.int8)])
            if graph.edges_h_after is not None:
                graph.edges_h_after = np.concatenate(
                    [graph.edges_h_after, np.zeros(n, np.int8)])
        return sw_air, ex_air

    def precompute(self, use_cache: bool = True) -> dict:
        import pickle
        cp = self._cache_path()
        if use_cache and os.path.exists(cp):
            with open(cp, "rb") as f:
                pc = pickle.load(f)
            pc["cached"] = True
            return pc
        t0 = time.time()
        # surface extraction wants mine positions for segment splitting
        from ..physics_step_wrapper import PhysicsStepWrapper
        phys = PhysicsStepWrapper(map_data=self.map_data)
        td = phys.tile_dic
        surface_points = extract_surfaces(td, mine_positions=self.all_mines,
                                          momentum_model=self.momentum_model)
        graph = build_graph(surface_points)
        # Tag the graph so route_extraction reconstructs launch/landing velocity
        # in the matching frame (tangential => via per-point tangent).
        graph.momentum_model = self.momentum_model

        mine_pos = (
            np.asarray(self.all_mines, dtype=np.float32)
            if (self.all_mines and self.mine_inflation) else None
        )
        # Goal positions for air-reachable-goal detection during enumeration:
        # index 0 = switch, 1 = exit, each (x, y, contact_range).  Lets arcs that
        # pass within contact range of a switch/exit NOT adjacent to any surface
        # node seed the backward Dijkstra (valid + common on open levels).
        from ...constants.physics_constants import (
            NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS,
        )
        goal_positions = [
            (self.switch[0], self.switch[1], NINJA_RADIUS + EXIT_SWITCH_RADIUS),
            (self.exit[0], self.exit[1], NINJA_RADIUS + EXIT_DOOR_RADIUS),
        ]
        stats: dict = {}
        edges, _wp = enumerate_edges(
            graph, self.map_data,
            num_workers=self.workers,
            collect_waypoints=False,
            mine_positions=mine_pos,
            stats_out=stats,
            goal_positions=goal_positions,
            use_native=self.use_native,
            momentum_model=self.momentum_model,
        )
        populate_graph_edges(graph, edges)
        enum_time = time.time() - t0

        gc = stats.get("goal_contacts", {})
        sw_air = dict(gc.get(0, {}))
        ex_air = dict(gc.get(1, {}))
        # Append CDJ trick edges so the global d* can compose tricks anywhere
        # (e.g. an air switch / high ledge reachable only via one or more CDJs).
        tsw, tex = self._append_cdj_trick_edges(graph)
        for n, f in tsw.items():
            sw_air[n] = min(sw_air.get(n, f), f)
        for n, f in tex.items():
            ex_air[n] = min(ex_air.get(n, f), f)
        # Completion>loops>speed: soft, bounded risk/loop reweight of the edge
        # costs BEFORE Dijkstra so d* descends toward safe, executable routes.
        # No-op when w_risk==w_loop==0 (pure frames).  Applied after trick edges
        # so CDJ arcs carry their (high) execution risk too.
        risk_stats = apply_risk_weighting(
            graph, self.deadly,
            w_risk=self.w_risk, w_loop=self.w_loop, risk_cap=self.risk_cap,
            jump_tax=self.jump_tax, goal_xy=self.exit,
        )
        rev_adj = _build_reverse_adjacency(graph)
        # Jointly optimal switch-then-exit cost-to-go (d_comb) plus the phase-1
        # exit tree for post-switch route extraction.  Air-contact seeds make
        # air-reachable switches/exits work.
        (d_comb, pred_comb, best_comb,
         d_ex, pred_ex, best_ex) = compute_two_phase_cost_to_go(
            graph, self.switch[0], self.switch[1],
            self.exit[0], self.exit[1], rev_adj=rev_adj,
            switch_air_contacts=sw_air, exit_air_contacts=ex_air)
        info_air = {"n_switch_air": len(sw_air), "n_exit_air": len(ex_air)}

        pc = {
            "graph": graph, "edges": edges, "stats": stats,
            "d_comb": d_comb, "pred_comb": pred_comb, "best_comb": best_comb,
            "d_ex": d_ex, "pred_ex": pred_ex, "best_ex": best_ex,
            "enum_time": enum_time, "cached": False, "air": info_air,
            "risk": risk_stats,
        }
        try:
            import pickle
            with open(self._cache_path(), "wb") as f:
                pickle.dump(pc, f)
        except Exception:
            pass
        return pc

    def _spawn_node(self, graph, d_star) -> Optional[int]:
        sx, sy = self.spawn
        # spawn is grounded at rest
        return find_nearest_start_node(graph, sx, sy, 0.0, 0.0, 1, d_star)

    def build_route(self, pc: dict) -> Tuple[List[ArcTarget], List[ArcTarget], dict]:
        """Extract the jointly-optimal route across the phase boundary.

        Phase 0 follows the combined tree (``d_comb``) from spawn until a
        switch-contact node (a phase-0 seed: ``pred_comb == -1``); phase 1 then
        follows the exit tree (``d_ex``) from that switch node to the exit.
        """
        graph = pc["graph"]
        d_comb, pred_comb, best_comb = pc["d_comb"], pc["pred_comb"], pc["best_comb"]
        info: dict = {}
        start = self._spawn_node(graph, d_comb)
        info["spawn_node"] = start
        if start is None or not np.isfinite(d_comb[start]):
            info["spawn_d_switch"] = float("inf")
            info["spawn_d_exit_from_switch"] = float("inf")
            return [], [], info
        # d_comb[start] = jointly optimal spawn -> switch -> exit cost.
        info["spawn_d_combined"] = float(d_comb[start])

        phase0 = extract_and_simplify_route(
            graph, d_comb, pred_comb, best_comb, start, dedup_loops=True)

        # The combined tree terminates at the optimal switch-contact node s*
        # (pred_comb == -1).  Walk predecessors to recover it, then route the
        # exit phase from there along the exit tree.
        s_star = start
        guard = 0
        while pred_comb[s_star] != -1 and guard < 100000:
            s_star = int(pred_comb[s_star])
            guard += 1
        info["switch_node"] = s_star
        info["spawn_d_switch"] = float(d_comb[start] - pc["d_ex"][s_star]) \
            if np.isfinite(pc["d_ex"][s_star]) else float(d_comb[start])
        info["spawn_d_exit_from_switch"] = float(pc["d_ex"][s_star])
        if not np.isfinite(pc["d_ex"][s_star]):
            return phase0, [], info
        phase1 = extract_and_simplify_route(
            graph, pc["d_ex"], pc["pred_ex"], pc["best_ex"], s_star,
            dedup_loops=True)
        return phase0, phase1, info

    def _settle(self, max_frames: int = 20) -> int:
        """Let the ninja fall to a grounded rest state before executing the route.

        The spawn (and any airborne route start) sits slightly off the surface
        point the lattice launches arcs from; edge enumeration injects an
        airborne state exactly at the surface point and relies on the first
        tick's floor contact to arm the jump.  Replaying from the raw airborne
        spawn therefore mis-fires the first jump.  A real agent simply falls to
        the floor first, so we reproduce that: tick with no input until grounded
        (state in {Immobile, Running, GroundSlide}).  Returns frames spent.
        """
        # Note: the spawn reports state 0 (Immobile) even while a few px above
        # the floor, so we must tick first and require a *stable* grounded state
        # (grounded contact AND vy ~= 0), not just the state code.
        for f in range(max_frames):
            self.sim.tick(0, 0)
            _vx, vy = self.sim.get_ninja_velocity()
            if self.sim.get_ninja_state_code() in (0, 1, 2) and abs(vy) < 0.01:
                return f + 1
        return max_frames

    def replay(self, segments: List[_Segment]) -> dict:
        """Replay the concatenated segment tapes continuously from spawn.

        Settles to grounded first (see :meth:`_settle`), then executes the
        route's per-frame tape with the real ``tick`` (full physics/entities).
        """
        self.sim.fast_reset()
        settle_frames = self._settle()
        # boundary frame index -> (expect_x, expect_y) for drift measurement
        boundaries: List[Tuple[int, float, float]] = []
        flat: List[Tuple[int, int]] = []
        for seg in segments:
            flat.extend(seg.tape)
            boundaries.append((len(flat), seg.expect_x, seg.expect_y))

        traj: List[Tuple[float, float]] = [self.sim.get_ninja_pos()]
        min_clear = float("inf")
        won = died = False
        switch_frame = None
        actual_at: dict = {}  # frame index -> (x, y)
        boundary_frames = {b[0] for b in boundaries}
        for f, (h, j) in enumerate(flat, start=1):
            self.sim.tick(h, j)
            x, y = self.sim.get_ninja_pos()
            traj.append((x, y))
            for (mx, my) in self.deadly:
                c = math.hypot(x - mx, y - my)
                if c < min_clear:
                    min_clear = c
            if switch_frame is None and _switch_done(self.sim):
                switch_frame = f
            if f in boundary_frames:
                actual_at[f] = (x, y)
            if self.sim.has_won():
                won = True
            if self.sim.has_died():
                died = True
            if won or died:
                break

        self._last_boundaries = boundaries
        self._last_actual_at = actual_at
        # per-arc drift over boundaries we actually reached
        drifts = []
        for (bf, ex, ey) in boundaries:
            if bf in actual_at:
                ax, ay = actual_at[bf]
                drifts.append(math.hypot(ax - ex, ay - ey))
        fx, fy = traj[-1]
        return {
            "won": won, "died": died, "frames": len(traj) - 1,
            "switch_frame": switch_frame, "min_clear": min_clear,
            "traj": traj, "drifts": drifts, "settle_frames": settle_frames,
            "dist_switch_end": math.hypot(fx - self.switch[0], fy - self.switch[1]),
            "dist_exit_end": math.hypot(fx - self.exit[0], fy - self.exit[1]),
        }

    def replay_closed_loop(self, route: List[ArcTarget],
                           max_frames: int = 2000,
                           advance_px: float = 12.0) -> dict:
        """Track the route by POSITION (closed-loop), not by fixed frame budgets.

        Each arc applies its action pattern — jump held for ``jump_hold_frames``
        with ``h_input_during``, then released with ``h_input_after`` — until the
        ninja reaches the *next* arc's launch point (within ``advance_px``), then
        advances.  This absorbs the run-arc d* frame under-count (Finding 5):
        a run that the lattice costs at 36f but physically takes 65f is simply
        run until the ninja *reaches* the target, instead of cut off at 36f.
        A generous per-arc cap prevents infinite loops on a stuck arc.

        Uses the real ``tick`` (full physics + entities), so switch/exit/mine are
        all live.  Returns win/death/min-clearance + how many arcs were reached.
        """
        self.sim.fast_reset()
        settle_frames = self._settle()
        if not route:
            return {"won": False, "died": False, "frames": 0,
                    "switch_frame": None, "min_clear": float("inf"),
                    "n_reached": 0, "n_arcs": 0, "stuck_arc": None}
        _GROUNDED = (0, 1, 2)
        arc_idx = 0
        frames_into_arc = 0
        jumped = False          # has THIS jump arc fired its jump yet?
        jhr = 0                 # remaining jump-hold frames (after firing)
        min_clear = float("inf")
        won = died = False
        switch_frame = None
        n_reached = 0
        stuck_arc = None
        f = 0
        while f < max_frames and arc_idx < len(route):
            arc = route[arc_idx]
            # Advance when near the next arc's launch point.
            if arc_idx + 1 < len(route):
                na = route[arc_idx + 1]
                x, y = self.sim.get_ninja_pos()
                if math.hypot(x - na.launch_x, y - na.launch_y) < advance_px:
                    arc_idx += 1
                    n_reached += 1
                    frames_into_arc = 0
                    jumped = False
                    jhr = 0
                    continue
            # Action selection.  A jump arc must fire from a GROUNDED state
            # (mirrors a real player: land, then jump); firing mid-air wastes the
            # input.  Until grounded we steer with h_input_during (no jump); once
            # grounded we hold the jump for jump_hold_frames, then fall with
            # h_input_after.  Run arcs (jump_hold==0) just apply the run input.
            is_jump = arc.jump_hold_frames > 0
            grounded = self.sim.get_ninja_state_code() in _GROUNDED
            if is_jump and not jumped:
                if grounded:
                    h, j = int(arc.h_input_during), 1
                    jumped = True
                    jhr = int(arc.jump_hold_frames) - 1
                else:
                    h, j = int(arc.h_input_during), 0  # approach/settle, no jump
            elif is_jump and jhr > 0:
                h, j = int(arc.h_input_during), 1
                jhr -= 1
            elif is_jump:
                h, j = int(arc.h_input_after), 0  # post-release fall
            else:
                h, j = int(arc.h_input_during), 0  # run
            self.sim.tick(h, j)
            f += 1
            frames_into_arc += 1
            jhr = max(0, jhr - 1)
            x, y = self.sim.get_ninja_pos()
            for (mx, my) in self.deadly:
                c = math.hypot(x - mx, y - my)
                if c < min_clear:
                    min_clear = c
            if switch_frame is None and _switch_done(self.sim):
                switch_frame = f
            if self.sim.has_won():
                won = True
                break
            if self.sim.has_died():
                died = True
                break
            # Per-arc cap: if we vastly overshoot the expected frames without
            # reaching the next launch point, the arc is stuck — advance or stop.
            if frames_into_arc > max(int(arc.arc_cost_frames * 3.0), 90):
                if arc_idx + 1 < len(route):
                    arc_idx += 1
                    frames_into_arc = 0
                    jumped = False
                    jhr = 0
                else:
                    stuck_arc = arc_idx
                    break
        fx, fy = self.sim.get_ninja_pos()
        return {
            "won": won, "died": died, "frames": f,
            "switch_frame": switch_frame, "min_clear": min_clear,
            "n_reached": n_reached, "n_arcs": len(route),
            "stuck_arc": stuck_arc, "settle_frames": settle_frames,
            "dist_switch_end": math.hypot(fx - self.switch[0], fy - self.switch[1]),
            "dist_exit_end": math.hypot(fx - self.exit[0], fy - self.exit[1]),
        }

    # ------------------------------------------------------------------
    # Closed-loop + micro-search executor (prototype for npp-rl port)
    # ------------------------------------------------------------------
    def _sim_jump_candidate(self, runup: int, hold: int, h_during: int,
                            h_after: int, target: Tuple[float, float],
                            fall_cap: int = 200):
        """From the CURRENT ninja state, simulate runup→hold→fall with the
        ENTITY-SAFE tick (tick_physics_mines_only: real trajectory + mine death,
        no switch/door/mine mutation), so candidates can be tried back-to-back
        via save/restore_ninja_state.

        Returns ``(min_dist_to_target_along_traj, landing_xy_or_None, died, seq)``.
        Scoring by the *closest approach along the whole trajectory* (not just
        the landing) lets goal arcs find a jump that PASSES THROUGH the switch /
        exit entity (contact is positional and can happen mid-air).
        """
        seq: List[Tuple[int, int]] = (
            [(h_during, 0)] * runup + [(h_during, 1)] * hold
        )
        min_d = float("inf")

        def _upd():
            nonlocal min_d
            x, y = self.sim.get_ninja_pos()
            d = math.hypot(x - target[0], y - target[1])
            if d < min_d:
                min_d = d

        for (h, j) in seq:
            self.sim.tick_physics_mines_only(h, j)
            _upd()
            if self.sim.get_ninja_state_code() in (6, 7):
                return min_d, None, True, seq
        was_air = False
        for _ in range(fall_cap):
            self.sim.tick_physics_mines_only(h_after, 0)
            seq.append((h_after, 0))
            _upd()
            sc = self.sim.get_ninja_state_code()
            if sc in (6, 7):
                return min_d, None, True, seq
            if sc in (3, 4):
                was_air = True
            elif was_air and sc in (0, 1, 2, 5):
                return min_d, self.sim.get_ninja_pos(), False, seq
        return min_d, None, False, seq

    def _search_jump(self, target: Tuple[float, float], h_during: int,
                     nominal_hold: int):
        """Search run-up × hold × post-release variations for the jump whose
        trajectory comes closest to ``target``, excluding lethal candidates.
        Run-up builds the launch velocity the lattice node assumes (the missing
        ingredient for position-only tracking).  Returns (best_seq_or_None, d)."""
        snap = self.sim.save_ninja_state()
        holds = sorted({max(1, min(MAX_JUMP_DURATION, nominal_hold + d))
                        for d in (-10, -5, -2, 0, 2, 5, 10)})
        best_seq = None
        best_d = float("inf")
        for runup in (0, 3, 6, 10, 14, 20):
            for hold in holds:
                for h_after in (-1, 0, 1):
                    self.sim.restore_ninja_state(snap)
                    min_d, _land, died, seq = self._sim_jump_candidate(
                        runup, hold, h_during, h_after, target)
                    if died:
                        continue  # never pick a lethal jump
                    if min_d < best_d:
                        best_d = min_d
                        best_seq = seq
        self.sim.restore_ninja_state(snap)
        return best_seq, best_d

    def replay_micro_search(self, route: List[ArcTarget], n_phase0: int,
                            max_frames: int = 2500,
                            advance_px: float = 12.0) -> dict:
        """Closed-loop + micro-search: position-track runs; for each jump, search
        local (run-up, hold, post-release) variations to reach the target.

        The target depends on the arc's role: the last phase-0 arc targets the
        SWITCH entity (so contact actually activates it), the final phase-1 arc
        targets the EXIT entity, and every other arc targets the next arc's
        launch node.  Applying a candidate stops early on win / switch activation
        so we don't overshoot a goal we've already reached.
        """
        from ...constants.physics_constants import (
            EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS, NINJA_RADIUS,
        )
        self.sim.fast_reset()
        settle_frames = self._settle()
        min_clear = float("inf")
        won = died = False
        switch_frame = None
        n_reached = 0
        f = 0
        # Tape starts from spawn: the settle (fall-to-floor) frames first, then
        # the executed actions, so it replays standalone on a fresh sim.
        tape: List[Tuple[int, int]] = [(0, 0)] * settle_frames

        def _track(h, j):
            nonlocal f, won, died, switch_frame, min_clear
            self.sim.tick(h, j)
            tape.append((int(h), int(j)))
            f += 1
            x, y = self.sim.get_ninja_pos()
            for (mx, my) in self.deadly:
                c = math.hypot(x - mx, y - my)
                if c < min_clear:
                    min_clear = c
            if switch_frame is None and _switch_done(self.sim):
                switch_frame = f
            if self.sim.has_won():
                won = True
            if self.sim.has_died():
                died = True

        n = len(route)
        for arc_idx, arc in enumerate(route):
            if won or died or f >= max_frames:
                break

            # TRICK arc (CDJ): the recorded tape was captured from the ninja at
            # rest AGAINST the corner wall (the trick's launch state), and the
            # CDJ is frame-perfect — it only reproduces when the ninja starts in
            # that exact state (a few px off the wall and the buffer-intersection
            # mis-fires).  So we drive the ninja firmly into the corner (in the
            # tape's own approach direction) until the wall stops it, settle to
            # rest, THEN replay the tape verbatim.  Stopping merely "within 6px of
            # launch_x" was too loose — it could halt just short of the wall, in
            # the trick's dead zone, and the climb failed.
            if getattr(arc, "trick_tape", None):
                # Approach direction = the tape's horizontal drive into the wall.
                adir = 0
                for (h, _j) in arc.trick_tape:
                    if h:
                        adir = 1 if h > 0 else -1
                        break
                # Drive into the corner until grounded at/over the launch point.
                cap = 80
                while cap > 0 and not (won or died) and f < max_frames:
                    x, _y = self.sim.get_ninja_pos()
                    grounded = self.sim.get_ninja_state_code() in (0, 1, 2)
                    at_corner = (
                        (adir > 0 and x >= arc.launch_x - 0.5)
                        or (adir < 0 and x <= arc.launch_x + 0.5)
                        or adir == 0
                    )
                    if grounded and at_corner:
                        break
                    hdir = adir if adir else (
                        1 if arc.launch_x > x + 1 else (-1 if arc.launch_x < x - 1 else 0))
                    _track(hdir, 0)
                    cap -= 1
                # Settle to rest against the wall (the recorded launch state).
                cap = 30
                while (cap > 0 and not (won or died) and f < max_frames
                       and abs(self.sim.get_ninja_velocity()[0]) > 0.05):
                    _track(0, 0)
                    cap -= 1
                for (h, j) in arc.trick_tape:
                    if won or died or f >= max_frames:
                        break
                    _track(h, j)
                n_reached += 1
                continue

            # Pick the target + arrival tolerance by the arc's role.
            is_switch_arc = (arc_idx == n_phase0 - 1)
            is_exit_arc = (arc_idx == n - 1)
            switch_just_activated = switch_frame is not None
            if is_exit_arc:
                # Target the arc's geometry-aware LANDING (the lattice routed it
                # into the exit's chamber); the progress-guarded finisher closes
                # the last few px to the exit entity.  Targeting the entity
                # directly makes the drop-jump search wander out of the chamber.
                target = (arc.landing_x, arc.landing_y)
                tol = advance_px
            elif is_switch_arc and not switch_just_activated:
                target = self.switch
                tol = NINJA_RADIUS + EXIT_SWITCH_RADIUS
            elif arc_idx + 1 < n:
                nxt = route[arc_idx + 1]
                target = (nxt.launch_x, nxt.launch_y)
                tol = advance_px
            else:
                target = (arc.landing_x, arc.landing_y)
                tol = advance_px

            if arc.jump_hold_frames > 0:
                approach_cap = 60
                while (self.sim.get_ninja_state_code() not in (0, 1, 2)
                       and approach_cap > 0 and not (won or died)):
                    _track(int(arc.h_input_during), 0)
                    approach_cap -= 1
                if won or died:
                    break
                best_seq, _bd = self._search_jump(
                    target, int(arc.h_input_during), int(arc.jump_hold_frames))
                if best_seq is None:
                    # No SAFE jump reached the target (every candidate was lethal).
                    # Don't apply a known-lethal default sequence — skip the jump
                    # (stay grounded / survive) and let the next arc + finisher
                    # try.  Converts a guaranteed death into a recoverable stall.
                    best_seq = []
                for (h, j) in best_seq:
                    if won or died or f >= max_frames:
                        break
                    _track(h, j)
                    # Early-stop once a goal arc has scored its contact.
                    if is_switch_arc and switch_frame is not None:
                        break
                n_reached += 1
            else:
                # Run arc: drive toward the target until within tolerance.
                cap = max(int(arc.arc_cost_frames * 3), 120)
                steps = 0
                while not (won or died) and f < max_frames and steps < cap:
                    x, y = self.sim.get_ninja_pos()
                    if math.hypot(x - target[0], y - target[1]) < tol:
                        break
                    # Mine-safe run: the dominant execution-death mode is the
                    # open-loop run drifting into a deadly mine.  Peek one tick
                    # with the entity-safe tick (ninja-only save/restore leaves
                    # mine state pristine) and stop the run before a lethal step,
                    # converting a drift-death into a survivable stop.
                    if self.deadly:
                        snap = self.sim.save_ninja_state()
                        self.sim.tick_physics_mines_only(int(arc.h_input_during), 0)
                        lethal = self.sim.get_ninja_state_code() in (6, 7)
                        self.sim.restore_ninja_state(snap)
                        if lethal:
                            break
                    _track(int(arc.h_input_during), 0)
                    steps += 1
                    if is_switch_arc and switch_frame is not None:
                        break
                n_reached += 1

        # Goal finisher: arcs exhausted but not won (the ninja reached the exit
        # NODE but not the exit ENTITY).  Greedily close the gap, PROGRESS-GUARDED
        # so it never wanders away from a good position: each burst must reduce
        # the distance to the exit or we stop.  Prefer a RUN when the exit is at
        # similar height (a jump would launch up and overshoot); jump-search only
        # when the exit is meaningfully above.  Only runs after switch collection.
        exit_tol = NINJA_RADIUS + EXIT_DOOR_RADIUS
        ex, ey = self.exit
        prev_d = math.hypot(self.sim.get_ninja_pos()[0] - ex,
                            self.sim.get_ninja_pos()[1] - ey)
        stall = 0
        while (not won and not died and switch_frame is not None
               and f < max_frames and stall < 3):
            x, y = self.sim.get_ninja_pos()
            if math.hypot(x - ex, y - ey) < exit_tol:
                break
            # Ground first, steering horizontally toward the exit.
            approach_cap = 40
            while (self.sim.get_ninja_state_code() not in (0, 1, 2)
                   and approach_cap > 0 and not (won or died)):
                xx, _yy = self.sim.get_ninja_pos()
                hx = 1 if ex > xx + 2 else (-1 if ex < xx - 2 else 0)
                _track(hx, 0)
                approach_cap -= 1
            if won or died:
                break
            x, y = self.sim.get_ninja_pos()
            h_dir = 1 if ex >= x else -1
            if ey < y - 20:
                # Exit is above: micro-search a jump toward it.
                best_seq, _bd = self._search_jump((ex, ey), h_dir, 12)
                if best_seq is None:
                    break
                for (h, j) in best_seq:
                    if won or died or f >= max_frames:
                        break
                    _track(h, j)
            else:
                # Exit at similar/lower height: run toward it.
                steps = 0
                while (not (won or died) and f < max_frames and steps < 40
                       and abs(self.sim.get_ninja_pos()[0] - ex) > 2):
                    _track(h_dir, 0)
                    steps += 1
            new_d = math.hypot(self.sim.get_ninja_pos()[0] - ex,
                               self.sim.get_ninja_pos()[1] - ey)
            stall = stall + 1 if new_d >= prev_d - 2.0 else 0
            prev_d = new_d

        fx, fy = self.sim.get_ninja_pos()
        return {
            "won": won, "died": died, "frames": f,
            "switch_frame": switch_frame, "min_clear": min_clear,
            "n_reached": n_reached, "n_arcs": len(route),
            "dist_exit_end": math.hypot(fx - self.exit[0], fy - self.exit[1]),
            "tape": tape,
        }

    def replay_dstar(self, pc: dict, max_frames: int = 2500,
                     advance_px: float = 12.0) -> dict:
        """Closed-loop d*-FOLLOWING executor.

        `replay`/`replay_closed_loop`/`replay_micro_search` all replay a FIXED
        precomputed arc sequence and accumulate drift.  This instead re-anchors
        to the ninja's ACTUAL nearest reachable node every decision point and
        takes the d*-greedy edge from the precomputed tree, so drift only changes
        which node we snap to next — it self-corrects.  It is the hand-coded form
        of the value-greedy policy an RL agent learns on the same d* field.
        Falls through gracefully on no-route levels (nearest node is None).
        """
        from ...constants.physics_constants import (
            EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS, NINJA_RADIUS,
        )
        graph = pc["graph"]
        trees = {0: (pc["d_comb"], pc["pred_comb"], pc["best_comb"]),
                 1: (pc["d_ex"], pc["pred_ex"], pc["best_ex"])}
        self.sim.fast_reset()
        settle = self._settle()
        f = 0
        won = died = False
        switch_frame = None
        min_clear = float("inf")
        tape: List[Tuple[int, int]] = [(0, 0)] * settle

        def _track(h, j):
            nonlocal f, won, died, switch_frame, min_clear
            self.sim.tick(int(h), int(j))
            tape.append((int(h), int(j)))
            f += 1
            x, y = self.sim.get_ninja_pos()
            for (mx, my) in self.deadly:
                c = math.hypot(x - mx, y - my)
                if c < min_clear:
                    min_clear = c
            if switch_frame is None and _switch_done(self.sim):
                switch_frame = f
            if self.sim.has_won():
                won = True
            if self.sim.has_died():
                died = True

        def _run_to(target, h_pref, tol, cap):
            """Mine-safe run toward target until within tol (or stuck/cap)."""
            steps = 0
            while not (won or died) and f < max_frames and steps < cap:
                x, y = self.sim.get_ninja_pos()
                if math.hypot(x - target[0], y - target[1]) < tol:
                    return
                hx = h_pref or (1 if target[0] > x + 1 else
                                (-1 if target[0] < x - 1 else 0))
                if self.deadly:
                    snap = self.sim.save_ninja_state()
                    self.sim.tick_physics_mines_only(int(hx), 0)
                    lethal = self.sim.get_ninja_state_code() in (6, 7)
                    self.sim.restore_ninja_state(snap)
                    if lethal:
                        return
                _track(hx, 0)
                steps += 1

        GROUNDED = (0, 1, 2)
        n_reached = 0
        stuck = 0
        prev = None
        while not (won or died) and f < max_frames and stuck < 10:
            sc = self.sim.get_ninja_state_code()
            x, y = self.sim.get_ninja_pos()
            vx, vy = self.sim.get_ninja_velocity()
            if sc in (3, 4):       # airborne: fall to a surface before deciding
                _track(0, 0)
                continue
            phase = 1 if switch_frame is not None else 0
            d_star, pred, best = trees[phase]
            node = find_nearest_start_node(graph, x, y, vx, vy, sc, d_star)
            if node is None or not np.isfinite(d_star[node]):
                stuck += 1
                _track(0, 0)
                continue
            if pred[node] == -1:                # at the phase-goal node
                if phase == 0:
                    _run_to(self.switch, 0, NINJA_RADIUS + EXIT_SWITCH_RADIUS, 60)
                    if switch_frame is None:
                        bs, _ = self._search_jump(self.switch, 0, 12)
                        for (h, j) in (bs or []):
                            if won or died or switch_frame is not None:
                                break
                            _track(h, j)
                    stuck += 1
                    continue
                _run_to(self.exit, 0, NINJA_RADIUS + EXIT_DOOR_RADIUS, 80)
                break
            sub = extract_and_simplify_route(graph, d_star, pred, best, node)
            if not sub:
                stuck += 1
                _track(0, 0)
                continue
            arc = sub[0]
            target = (arc.landing_x, arc.landing_y)
            if getattr(arc, "trick_tape", None):
                for (h, j) in arc.trick_tape:
                    if won or died or f >= max_frames:
                        break
                    _track(h, j)
            elif arc.jump_hold_frames > 0:
                cap = 40
                while (self.sim.get_ninja_state_code() not in GROUNDED
                       and cap > 0 and not (won or died)):
                    _track(int(arc.h_input_during), 0)
                    cap -= 1
                bs, _ = self._search_jump(target, int(arc.h_input_during),
                                          int(arc.jump_hold_frames))
                for (h, j) in (bs or []):
                    if won or died or f >= max_frames:
                        break
                    _track(h, j)
                if bs is None:
                    _track(0, 0)
            else:
                _run_to(target, int(arc.h_input_during), advance_px,
                        max(int(arc.arc_cost_frames * 3), 120))
            n_reached += 1
            cur = (round(x), round(y), phase)
            stuck = stuck + 1 if cur == prev else 0
            prev = cur

        fx, fy = self.sim.get_ninja_pos()
        return {
            "won": won, "died": died, "frames": f,
            "switch_frame": switch_frame, "min_clear": min_clear,
            "n_reached": n_reached, "n_arcs": n_reached,
            "dist_exit_end": math.hypot(fx - self.exit[0], fy - self.exit[1]),
            "tape": tape, "via": "dstar",
        }

    def _eval_tape(self, tape: List[Tuple[int, int]], via: str = "") -> dict:
        """Replay a per-frame tape on the native sim and report the result dict."""
        self.sim.fast_reset()
        min_clear = float("inf")
        switch_frame = None
        won = died = False
        f = 0
        for (h, j) in tape:
            self.sim.tick(h, j)
            f += 1
            x, y = self.sim.get_ninja_pos()
            for (mx, my) in self.deadly:
                c = math.hypot(x - mx, y - my)
                if c < min_clear:
                    min_clear = c
            if switch_frame is None and _switch_done(self.sim):
                switch_frame = f
            if self.sim.has_won():
                won = True
                break
            if self.sim.has_died():
                died = True
                break
        return {"won": won, "died": died, "frames": f,
                "switch_frame": switch_frame, "min_clear": min_clear,
                "n_reached": 0, "n_arcs": 0, "tape": tape[:f], "via": via}

    def _forward_micro_search(self, prefix_tape: List[Tuple[int, int]],
                              max_expansions: int = 20000, macro: int = 4,
                              max_macros: int = 120, w: float = 1.25
                              ) -> Optional[List[Tuple[int, int]]]:
        """Mine-aware forward A* on the native sim, seeded by ``prefix_tape``.

        Explores macro-actions from the state the prefix leaves the ninja in
        (e.g. a CDJ apex) toward the switch then the exit, pruning branches that
        die (real deadly-mine contact under ``tick``).  Returns the per-frame
        traverse AFTER the prefix (truncated at win), or None if no win is found.
        This is the local forward search that completes a trick the lattice can't
        route past — bounded by ``max_expansions`` so it stays tractable.
        """
        import heapq
        from ...constants.physics_constants import MAX_HOR_SPEED

        sim = self.sim
        SW, EX = self.switch, self.exit

        def replay(actions: List[int]) -> None:
            sim.fast_reset()
            for h, j in prefix_tape:
                sim.tick(h, j)
                if sim.has_died() or sim.has_won():
                    return
            for a in actions:
                h, j = ACT[a]
                for _ in range(macro):
                    sim.tick(h, j)
                    if sim.has_died() or sim.has_won():
                        return

        def state() -> dict:
            x, y = sim.get_ninja_pos()
            vx, vy = sim.get_ninja_velocity()
            return {"x": x, "y": y, "vx": vx, "vy": vy,
                    "dead": sim.has_died(), "won": sim.has_won(),
                    "sw": _switch_done(sim)}

        def heur(s: dict) -> float:
            gx, gy = (EX if s["sw"] else SW)
            d = math.hypot(s["x"] - gx, s["y"] - gy)
            if not s["sw"]:
                d += math.hypot(SW[0] - EX[0], SW[1] - EX[1])
            return d / MAX_HOR_SPEED

        def key(s: dict) -> Tuple:
            return (round(s["x"] / 10.0), round(s["y"] / 10.0),
                    round(s["vx"] / 0.3), round(s["vy"] / 0.3), s["sw"])

        replay([])
        root = state()
        if root["won"]:
            return []
        if root["dead"]:
            return None
        frontier = [(w * heur(root), 0.0, 0, [], root)]
        best = {key(root): 0.0}
        cnt = 1
        exp = 0
        sol = None
        while frontier and exp < max_expansions:
            _f, g, _, acts, s = heapq.heappop(frontier)
            if g > best.get(key(s), float("inf")):
                continue
            exp += 1
            if s["won"]:
                sol = acts
                break
            if len(acts) >= max_macros:
                continue
            for a in ACT:
                ca = acts + [a]
                replay(ca)
                cs = state()
                if cs["dead"]:
                    continue
                cg = g + macro
                k = key(cs)
                if cg < best.get(k, float("inf")):
                    best[k] = cg
                    if cs["won"]:
                        sol = ca
                        break
                    heapq.heappush(
                        frontier, (cg + w * heur(cs), cg, cnt, ca, cs))
                    cnt += 1
            if sol is not None:
                break
        if sol is None:
            return None
        # Expand the winning macro-action list to a per-frame traverse.
        sim.fast_reset()
        for h, j in prefix_tape:
            sim.tick(h, j)
        traverse: List[Tuple[int, int]] = []
        for a in sol:
            hh, jj = ACT[a]
            for _ in range(macro):
                sim.tick(hh, jj)
                traverse.append((hh, jj))
                if sim.has_won() or sim.has_died():
                    return traverse
        return traverse

    def try_cdj_completion(self, holds: Tuple[int, ...] = (74, 80, 68),
                           dirs: Tuple[int, ...] = (1, -1),
                           max_expansions: int = 20000) -> Optional[dict]:
        """CDJ trick action + forward micro-search, for switches the lattice can't
        reach (air goal cleared only by a Corner Double Jump).

        Attempts the adaptive closed-loop CDJ (``cdj_primitive.execute_cdj``) off a
        corner near spawn on each side; when it finds a live corner and climbs,
        completes the spawn→apex→switch→exit traverse with the mine-aware forward
        search.  Returns a winning result dict, or None if no corner CDJ completes.

        Runs the CDJ on the NATIVE sim (``self.sim``) — the native module now
        exposes the ``airborn`` / ``walled`` / ``wall_buffer`` signals corner
        detection needs, so no Python ``nsim`` is spun up and the whole trick path
        stays on the fast engine.
        """
        from ..cdj_primitive import execute_cdj

        for d in dirs:
            for hold in holds:
                self.sim.fast_reset()
                ex = execute_cdj(self.sim, approach_dir=int(d),
                                 hold_frames=int(hold))
                # Only pursue a real corner trick that gained meaningful height.
                if (not ex.corner_found) or ex.result.died \
                        or ex.result.climb_px < 80.0:
                    continue
                traverse = self._forward_micro_search(
                    ex.tape, max_expansions=max_expansions)
                if traverse is not None:
                    return self._eval_tape(ex.tape + traverse, via="cdj")
        return None

    def _switch_has_surface_node(self) -> bool:
        """Cheap (~0.02s, no edge enumeration) check: does the switch have a
        walkable surface contact node?  If not it is an air goal the lattice can
        only reach via deferred air-contact arcs — so prefer the CDJ trick and
        skip the expensive mine-aware enumeration that would fail anyway."""
        try:
            from ..physics_step_wrapper import PhysicsStepWrapper
            from ..surface_graph import find_goal_nodes
            phys = PhysicsStepWrapper(map_data=self.map_data)
            sp = extract_surfaces(phys.tile_dic, mine_positions=self.all_mines)
            g = build_graph(sp)
            return len(find_goal_nodes(
                g, self.switch[0], self.switch[1], "switch")) > 0
        except Exception:
            return True  # conservative: keep the lattice path on any failure

    def plan_tape(self, use_cache: bool = True, allow_cdj: bool = True) -> dict:
        """Plan the lattice route and micro-search it into a winning input tape.

        Returns the micro-search result dict (including ``tape`` — the per-frame
        ``(h_input, jump_input)`` sequence).  The tape replays bit-identically on
        the Python ``nsim`` engine (native↔Python is 0px), so it both drives the
        completion videos and is consumable by ``open_loop_replay``.

        When the switch is air-reachable only via a Corner Double Jump (no surface
        contact node) the expensive lattice enumeration is skipped and the CDJ
        trick action + forward micro-search (:meth:`try_cdj_completion`) is tried
        first — this is what unlocks levels like ``di-required``.  The CDJ is also
        the fallback when a normal lattice route fails to win.
        """
        res = {"won": False, "tape": [], "n_arcs": 0, "frames": 0,
               "switch_frame": None, "min_clear": float("inf")}

        # Air switch (no surface node): the ~100s mine-aware enumeration is doomed
        # here, so try the CDJ trick first and only fall back to the lattice if it
        # finds no completing corner.
        if allow_cdj and not self._switch_has_surface_node():
            cdj = self.try_cdj_completion()
            if cdj is not None and cdj.get("won"):
                return cdj

        try:
            pc = self.precompute(use_cache=use_cache)
            # Closed-loop d*-following first (drift-robust); only fall back to the
            # fixed-route micro-search executor if it doesn't win.
            res = self.replay_dstar(pc)
            if not res.get("won"):
                phase0, phase1, _info = self.build_route(pc)
                route = phase0 + phase1
                if route:
                    ms = self.replay_micro_search(route, len(phase0))
                    if ms.get("won"):
                        res = ms
        except ValueError:
            # Switch/exit unreachable on the lattice — fall through to the trick.
            pass
        if res.get("won") or not allow_cdj:
            return res
        cdj = self.try_cdj_completion()
        return cdj if (cdj is not None and cdj.get("won")) else res

    def run(self, spike: bool = False, verbose: bool = False,
            use_cache: bool = True) -> AuditResult:
        pc = self.precompute(use_cache=use_cache)
        graph = pc["graph"]
        phase0, phase1, info = self.build_route(pc)
        route = phase0 + phase1
        segs = _route_to_segments(phase0) + _route_to_segments(phase1)
        # lattice route frames = sum of arc costs (frames-or-inflated)
        route_cost = sum(
            int(round(a.arc_cost_frames)) for a in (phase0 + phase1)
        )
        rep = self.replay(segs) if segs else {
            "won": False, "died": False, "frames": 0, "switch_frame": None,
            "min_clear": float("inf"), "drifts": [],
            "dist_switch_end": float("inf"), "dist_exit_end": float("inf"),
        }
        drifts = rep["drifts"]

        # Closed-loop (position-tracking) replay of the same route.
        cl = self.replay_closed_loop(route) if route else {
            "won": False, "died": False, "frames": 0, "switch_frame": None,
            "min_clear": float("inf"), "n_reached": 0, "n_arcs": 0,
        }
        # Closed-loop + micro-search replay (velocity-matching per jump).
        ms = self.replay_micro_search(route, len(phase0)) if route else {
            "won": False, "died": False, "frames": 0, "switch_frame": None,
            "min_clear": float("inf"), "n_reached": 0, "n_arcs": 0,
        }

        if verbose and segs:
            boundaries = getattr(self, "_last_boundaries", [])
            actual_at = getattr(self, "_last_actual_at", {})
            n0 = len(phase0)
            print("  per-arc breakdown (expected landing vs continuous-replay actual):")
            print(f"    spawn=({self.spawn[0]:.1f},{self.spawn[1]:.1f})  "
                  f"spawn_node={info.get('spawn_node')}")
            for i, (arc, (bf, ex, ey)) in enumerate(zip(route, boundaries)):
                phase = 0 if i < n0 else 1
                kind = (f"JUMP hold={arc.jump_hold_frames} h={arc.h_input_during}"
                        f"/{arc.h_input_after}" if arc.jump_hold_frames > 0
                        else f"RUN h={arc.h_input_during}")
                act = actual_at.get(bf)
                if act is not None:
                    drift = math.hypot(act[0] - ex, act[1] - ey)
                    actstr = f"actual=({act[0]:.1f},{act[1]:.1f}) drift={drift:.1f}px"
                else:
                    actstr = "actual=<not reached / died>"
                print(f"    p{phase} arc{i}: {kind:<26} cost={arc.arc_cost_frames:.0f}f "
                      f"launch=({arc.launch_x:.1f},{arc.launch_y:.1f}) "
                      f"-> expect=({ex:.1f},{ey:.1f}) {actstr}")
        spike_frames = spike_solved = None
        if spike:
            from .physics_route_spike import PhysicsRouteSpike
            sp = PhysicsRouteSpike(self.map_path, h_weight=1.25)
            sres = sp.plan()
            spike_solved = sres["solved"]
            spike_frames = sres["frames"] if sres["solved"] else None

        reachable = (
            np.isfinite(info.get("spawn_d_switch", float("inf")))
            and np.isfinite(info.get("spawn_d_exit_from_switch", float("inf")))
        )
        return AuditResult(
            name=self.name,
            n_points=graph.n_points,
            n_nodes=graph.n_nodes,
            n_edges=len(pc["edges"]),
            pct_clipped=float(pc["stats"].get("pct_clipped", 0.0)),
            has_mines=bool(self.all_mines),
            n_deadly=len(self.deadly),
            spawn_d_switch=info.get("spawn_d_switch", float("inf")),
            spawn_d_exit_from_switch=info.get("spawn_d_exit_from_switch", float("inf")),
            lattice_route_frames=float(route_cost),
            reachable=reachable,
            replay_won=rep["won"],
            replay_died=rep["died"],
            replay_frames=rep["frames"],
            switch_frame=rep["switch_frame"],
            min_mine_clearance=rep["min_clear"],
            dist_switch_end=rep.get("dist_switch_end", float("inf")),
            dist_exit_end=rep.get("dist_exit_end", float("inf")),
            cl_won=cl["won"],
            cl_died=cl["died"],
            cl_frames=cl["frames"],
            cl_switch_frame=cl["switch_frame"],
            cl_min_clear=cl["min_clear"],
            cl_reached=cl["n_reached"],
            ms_won=ms["won"],
            ms_died=ms["died"],
            ms_frames=ms["frames"],
            ms_switch_frame=ms["switch_frame"],
            ms_min_clear=ms["min_clear"],
            ms_reached=ms["n_reached"],
            max_arc_drift=float(max(drifts)) if drifts else 0.0,
            mean_arc_drift=float(np.mean(drifts)) if drifts else 0.0,
            n_arcs=len(phase0) + len(phase1),
            enum_time_s=pc["enum_time"],
            spike_frames=spike_frames,
            spike_solved=spike_solved,
        )


def _print_report(r: AuditResult) -> None:
    mine_note = (
        f"{r.n_deadly} deadly mines — pruned by REAL-death tick "
        f"(tick_physics_mines_only); edge weights are frames (no CHOMP inflation "
        f"without waypoints), so d* is sound + frame-exact here" if r.has_mines
        else "no mines — edge weights ARE frames, self-consistency is exact"
    )
    print(f"\n=== Lattice audit: {r.name} ===")
    print(f"  graph: {r.n_points:,} pts, {r.n_nodes:,} nodes, {r.n_edges:,} edges "
          f"| enum {r.enum_time_s:.1f}s | clip {r.pct_clipped:.1f}%")
    print(f"  mines: {mine_note}")
    print(f"  REACHABILITY: spawn->switch d*={r.spawn_d_switch:.1f}  "
          f"switch->exit d*={r.spawn_d_exit_from_switch:.1f}  "
          f"=> {'REACHABLE' if r.reachable else 'UNREACHABLE (false-negative?)'}")
    print(f"  lattice route prediction: {r.lattice_route_frames:.0f}f over {r.n_arcs} arcs")
    verdict = ("WON" if r.replay_won else
               ("DIED" if r.replay_died else "DID-NOT-FINISH"))
    reached = "yes" if r.switch_frame is not None else "NO"
    print(f"  EXECUTABILITY (continuous open-loop replay): {verdict} "
          f"@ {r.replay_frames}f  reached_switch={reached}(@{r.switch_frame})  "
          f"min_mine_clear={r.min_mine_clearance:.1f}px "
          f"(kill={_MINE_KILL_PX:.0f})")
    if not r.replay_won:
        print(f"    end-of-tape distance to goals: switch={r.dist_switch_end:.1f}px "
              f"exit={r.dist_exit_end:.1f}px  (small => macro route correct, "
              f"micro-execution miss; large => wrong macro route)")
    if r.replay_won and not r.has_mines:
        gap = r.replay_frames - r.lattice_route_frames
        print(f"  SELF-CONSISTENCY: replay {r.replay_frames}f vs d* {r.lattice_route_frames:.0f}f "
              f"=> gap {gap:+.0f}f ({100*gap/max(1,r.lattice_route_frames):+.0f}%)")
    print(f"  ARC DRIFT (open-loop snap-through localizer): max {r.max_arc_drift:.1f}px  "
          f"mean {r.mean_arc_drift:.1f}px")
    cl_verdict = ("WON" if r.cl_won else ("DIED" if r.cl_died else "DID-NOT-FINISH"))
    cl_reached = "yes" if r.cl_switch_frame is not None else "NO"
    print(f"  EXECUTABILITY (CLOSED-LOOP position tracking): {cl_verdict} "
          f"@ {r.cl_frames}f  reached_switch={cl_reached}(@{r.cl_switch_frame})  "
          f"arcs_reached={r.cl_reached}/{r.n_arcs}  "
          f"min_mine_clear={r.cl_min_clear:.1f}px")
    ms_verdict = ("WON" if r.ms_won else ("DIED" if r.ms_died else "DID-NOT-FINISH"))
    ms_reached = "yes" if r.ms_switch_frame is not None else "NO"
    print(f"  EXECUTABILITY (CLOSED-LOOP + MICRO-SEARCH): {ms_verdict} "
          f"@ {r.ms_frames}f  reached_switch={ms_reached}(@{r.ms_switch_frame})  "
          f"arcs_reached={r.ms_reached}/{r.n_arcs}  "
          f"min_mine_clear={r.ms_min_clear:.1f}px")
    if r.spike_solved is not None:
        if r.spike_solved:
            print(f"  SPIKE ORACLE: {r.spike_frames}f "
                  f"(lattice replay {r.replay_frames}f, "
                  f"{'OK' if r.replay_won else 'lattice FAILED'})")
        else:
            print(f"  SPIKE ORACLE: no route found (timed out / capped)")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--spike", action="store_true",
                   help="also run the forward spike as an absolute oracle "
                        "(only feasible on small levels)")
    p.add_argument("--no-mine-inflation", action="store_true",
                   help="disable mine cost inflation during enumeration")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print per-arc expected-vs-actual breakdown")
    p.add_argument("--no-cache", action="store_true",
                   help="force re-precompute (ignore /tmp cache)")
    args = p.parse_args()
    audit = LatticeAudit(args.map, workers=args.workers,
                         mine_inflation=not args.no_mine_inflation)
    r = audit.run(spike=args.spike, verbose=args.verbose,
                  use_cache=not args.no_cache)
    _print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
