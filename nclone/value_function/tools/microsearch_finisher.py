#!/usr/bin/env python3
"""Micro-search finisher: a fine-resolution forward search that *executes* the
lattice plan on the real sim, closing the execution gap the macro d*-guided
search leaves on precise / trick-dependent air-contact entities.

Motivation (middlecircle, decoded from a winning N++ attract replay — see
``middlecircle-convex-switch-execution-wall`` memory): the lattice planner
*models* a route (finite d*, air-contact seeds for off-surface switch/exit), but
the macro action search can't execute two coupled, frame-precise maneuvers:

  1. GRAZE-WITH-BOUNCE — contact an off-surface entity (a switch on a convex
     overhang) by falling to within contact range and JUMPING at contact, so the
     ninja leaves moving upward and can recover (a greedy "minimise distance"
     graze dives past and is unrecoverable).
  2. MOMENTUM-CARRY LAUNCH — reach a high air-contact entity (exit door above an
     obstacle) by arriving at the launch surface with horizontal speed and
     holding a *directional jump*.  N's variable-height jump means the rise
     depends on the consecutive jump-hold count — a per-frame state the planner's
     (x,y,vx,vy) lattice can't represent.

Two generalizable ideas make these executable with a plain forward A* on the
native sim (no hand-coded input tapes):

  * ENRICHED DEDUP KEY — the search state key includes the ninja's *internal*
    jump-hold timer (``get_ninja_jump_duration``), movement-FSM state code and
    airborne flag, not just binned position/velocity.  Without the jump-hold
    dimension the search merges held-jump states with non-jumping ones and prunes
    the very trajectory that gains height; with it, the launch trick is found.
  * COUPLED-STATE WAYPOINTS — segments are not independent "reach this point";
    each carries a success predicate over the *exit state* (grazed-and-rising,
    arrived-with-speed) and the ninja's full native state (incl. velocity and
    jump buffers) is carried across segment boundaries via save/restore.

Waypoints are derived generically from the tangential lattice: the spawn, the
switch (air-contact graze), the exit-air-contact seed nodes (launch points), and
the exit (air-contact launch).  The result is a validated action tape replayed on
the main sim with real ``tick`` (switch/door actually activate).

Usage::

    python -m nclone.value_function.tools.microsearch_finisher \
        --map nclone/test_maps/middlecircle --video out.mp4
"""
from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
import time
from typing import Callable, List, Optional, Tuple

import nclone_native as Nnative

from ...constants.physics_constants import (
    NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS, MAX_HOR_SPEED,
)
from .physics_route_spike import (
    _load_map_data, _goal_positions, _switch_done, ACT,
)
from .lattice_audit import LatticeAudit

_INV_V = 1.0 / MAX_HOR_SPEED


class MicroSearchFinisher:
    """Fine-resolution segmented forward search over lattice-derived waypoints."""

    def __init__(self, map_path: str, *, workers: int = 4,
                 momentum_model: str = "tangential", macro: int = 2,
                 cell_px: float = 3.0, vel_bin: float = 0.1):
        self.map_path = map_path
        self.map_data = _load_map_data(map_path)
        self.main = Nnative.Simulator.load(self.map_data)
        self.search = Nnative.Simulator.load(self.map_data)
        self.macro = macro
        self.cell = cell_px
        self.vb = vel_bin
        self.audit = LatticeAudit(map_path, workers=workers,
                                  momentum_model=momentum_model)
        self.switch, self.exit, _ = _goal_positions(self.main)
        self.switch_r = NINJA_RADIUS + EXIT_SWITCH_RADIUS
        self.exit_r = NINJA_RADIUS + EXIT_DOOR_RADIUS
        self._launch_pts = self._exit_launch_points()

    # ------------------------------------------------------------------ lattice
    def _exit_launch_points(self) -> List[Tuple[float, float]]:
        """The exit-air-contact seed nodes = surface points from which an arc can
        reach the exit; these are the launch surfaces for the momentum-carry
        launch.  Derived from the lattice enumeration (generic, not hand-picked).
        Falls back to the highest surface node near the exit x if none."""
        from ..surface_extraction import extract_surfaces
        from ..surface_graph import build_graph
        from ..edge_enumeration import enumerate_edges
        from ..physics_step_wrapper import PhysicsStepWrapper
        phys = PhysicsStepWrapper(map_data=self.audit.map_data)
        pts = extract_surfaces(phys.tile_dic, mine_positions=self.audit.all_mines,
                               momentum_model=self.audit.momentum_model)
        graph = build_graph(pts)
        graph.momentum_model = self.audit.momentum_model
        stats: dict = {}
        gp = [(self.switch[0], self.switch[1], self.switch_r),
              (self.exit[0], self.exit[1], self.exit_r)]
        enumerate_edges(graph, self.audit.map_data, num_workers=4,
                        collect_waypoints=False, mine_positions=None,
                        stats_out=stats, goal_positions=gp,
                        use_native=self.audit.use_native,
                        momentum_model=self.audit.momentum_model)
        mb = graph.max_vel_bins
        ex_air = stats.get("goal_contacts", {}).get(1, {})
        seed_pts = sorted({nid // mb for nid in ex_air},
                          key=lambda pi: graph.surface_points[pi].y)
        if seed_pts:
            return [(graph.surface_points[pi].x, graph.surface_points[pi].y)
                    for pi in seed_pts]
        # fallback: highest node near exit x
        near = sorted((p for p in graph.surface_points
                       if abs(p.x - self.exit[0]) < 40 and p.y > 0),
                      key=lambda p: p.y)
        return [(near[0].x, near[0].y)] if near else [self.exit]

    # ------------------------------------------------------------- search state
    def _key(self, grazed: int) -> Tuple:
        """Enriched dedup key: binned pos/vel PLUS the ninja's internal jump-hold
        timer, movement-FSM state code and airborne flag (and the phase bit).
        The jump-hold dimension is what makes the variable-height launch findable
        — omitting it merges held-jump states and prunes the winning trajectory."""
        x, y = self.search.get_ninja_pos()
        vx, vy = self.search.get_ninja_velocity()
        jd = min(self.search.get_ninja_jump_duration(), 45)
        return (round(x / self.cell), round(y / self.cell),
                round(vx / self.vb), round(vy / self.vb),
                self.search.get_ninja_state_code(),
                int(self.search.get_ninja_airborn()), jd, grazed)

    def _segment(self, start_blob, *, heur: Callable[[float, float, int], float],
                 success: Callable[..., bool], grazes: bool,
                 max_exp: int, max_steps: int, w: float = 1.2):
        """Bounded forward A* on the search sim from ``start_blob``.

        ``heur(x,y,grazed)`` is the (frames) cost-to-go estimate — grazed-aware so
        a segment can switch sub-goal after the graze (toward switch first, then
        toward the launch node).  ``success(x,y,vx,vy,grazed)`` decides
        termination.  ``grazes`` toggles the switch-proximity phase bit.  Returns
        ``(actions, end_state, expansions)`` or ``(None, None, expansions)``."""
        s = self.search
        s.restore_ninja_state(start_blob)
        sx, sy = s.get_ninja_pos()
        g0 = 1 if (grazes and math.hypot(sx - self.switch[0],
                                         sy - self.switch[1]) < self.switch_r) else 0
        frontier = [(w * heur(sx, sy, g0), 0.0, 0, start_blob, [], g0)]
        best = {self._key(g0): 0.0}
        ctr = 1
        exp = 0
        while frontier and exp < max_exp:
            f, g, _, blob, acts, gz = heapq.heappop(frontier)
            exp += 1
            if len(acts) >= max_steps:
                continue
            for a in ACT:
                s.restore_ninja_state(blob)
                h, j = ACT[a]
                dead = False
                for _ in range(self.macro):
                    s.tick_physics_mines_only(h, j)
                    if s.has_died():
                        dead = True
                        break
                if dead:
                    continue
                x, y = s.get_ninja_pos()
                vx, vy = s.get_ninja_velocity()
                ngz = gz
                if grazes and not gz and math.hypot(x - self.switch[0],
                                                    y - self.switch[1]) < self.switch_r:
                    ngz = 1
                if success(x, y, vx, vy, ngz):
                    return acts + [a], (x, y, vx, vy), exp
                k = self._key(ngz)
                ng = g + self.macro
                if ng < best.get(k, float("inf")):
                    best[k] = ng
                    nb = s.save_ninja_state()
                    heapq.heappush(frontier,
                                   (ng + w * heur(x, y, ngz), ng, ctr, nb,
                                    acts + [a], ngz))
                    ctr += 1
        return None, None, exp

    # --------------------------------------------------------------- apply tape
    def _apply(self, acts: List[int], tape: List[Tuple[int, int]]) -> dict:
        """Replay ``acts`` on the MAIN sim with real ``tick`` (entities active).
        Appends per-frame (h,j) to ``tape``; returns event flags."""
        out = {"won": False, "died": False, "switch": False}
        for a in acts:
            h, j = ACT[a]
            for _ in range(self.macro):
                self.main.tick(h, j)
                tape.append((h, j))
                if _switch_done(self.main):
                    out["switch"] = True
                if self.main.has_won():
                    out["won"] = True
                    return out
                if self.main.has_died():
                    out["died"] = True
                    return out
        return out

    def _settle(self):
        for _ in range(20):
            self.main.tick(0, 0)
            vx, vy = self.main.get_ninja_velocity()
            if abs(vx) < 0.01 and abs(vy) < 0.01:
                break

    # --------------------------------------------------------------------- plan
    def plan(self, *, max_exp: int = 300000, verbose: bool = True) -> dict:
        """Chain graze -> climb-to-launch -> momentum-launch, carrying full native
        state across segments.  Returns the validated tape + win status."""
        t0 = time.time()
        self.main.fast_reset()
        self._settle()
        tape: List[Tuple[int, int]] = []
        log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
        lp = self._launch_pts[0]
        sw, ex = self.switch, self.exit
        sw_lp = math.hypot(sw[0] - lp[0], sw[1] - lp[1])

        # --- Segment 1: graze the switch, then climb to the launch node. ---
        # One COUPLED search with a grazed-aware two-phase heuristic: before the
        # graze, head for the switch (+ switch->launch); after, head for the
        # launch node.  This makes the search discover a *recoverable* graze
        # (jump at contact -> rise) rather than an unrecoverable dive-past, and
        # carries it through the convex wall-climb back to the launch surface.
        def h1(x, y, gz):
            if gz:
                return math.hypot(x - lp[0], y - lp[1]) * _INV_V
            return (math.hypot(x - sw[0], y - sw[1]) + sw_lp) * _INV_V
        def ok1(x, y, vx, vy, gz):
            return bool(gz) and math.hypot(x - lp[0], y - lp[1]) < 12.0
        s1, st1, e1 = self._segment(self.main.save_ninja_state(), heur=h1,
                                    success=ok1, grazes=True,
                                    max_exp=max_exp, max_steps=340, w=1.3)
        if s1 is None:
            return self._fail("graze+climb-to-launch", e1, tape, t0)
        ev = self._apply(s1, tape)
        log(f"  [1/2] graze+climb  {len(s1)*self.macro}f exp={e1} "
            f"switch={ev['switch']} -> pos={_fmt(self.main.get_ninja_pos())} "
            f"vel={_fmt(self.main.get_ninja_velocity())}")
        if ev["won"]:
            return self._win(tape, t0, e1)
        if ev["died"] or not ev["switch"]:
            return self._fail("graze did not activate switch / died", e1, tape, t0)

        # --- Segment 2: momentum-carry launch to the exit (enriched key vital). --
        def h2(x, y, gz):
            return math.hypot(x - ex[0], y - ex[1]) * _INV_V
        def ok2(x, y, vx, vy, gz):
            return math.hypot(x - ex[0], y - ex[1]) < self.exit_r
        s2, st2, e2 = self._segment(self.main.save_ninja_state(), heur=h2,
                                    success=ok2, grazes=False,
                                    max_exp=max_exp, max_steps=160, w=1.0)
        if s2 is None:
            return self._fail("momentum-launch", e2, tape, t0)
        ev = self._apply(s2, tape)
        log(f"  [2/2] momentum-launch  {len(s2)*self.macro}f exp={e2} "
            f"won={ev['won']} -> pos={_fmt(self.main.get_ninja_pos())}")
        if ev["won"]:
            return self._win(tape, t0, e1 + e2)
        return self._fail("launch reached exit-region but no win", e2, tape, t0)

    # ----------------------------------------------------- reference-anchored
    def plan_from_reference(self, ref_inputs: List[Tuple[int, int]], *,
                            stride: int = 24, pos_tol: float = 6.0,
                            vel_tol: float = 0.35, seg_max_exp: int = 120000,
                            verbose: bool = True) -> dict:
        """Re-derive a win by connecting sparse anchor states sampled from a
        reference run (lattice route OR attract-demo inputs) with the enriched
        micro-search; fall back to the reference inputs only where the (often
        sub-pixel) coupling defeats search.  Generalizable distillation step:
        the engine reproduces the maneuvers from anchors rather than blind replay,
        and reports the fraction re-derived autonomously.

        ``ref_inputs`` is a list of per-frame ``(hor, jump)`` known to win.
        """
        log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
        t0 = time.time()
        # Anchors = full ninja states at every ``stride``-th reference frame.
        ref = Nnative.Simulator.load(self.map_data)
        anchors = []
        for f, (h, j) in enumerate(ref_inputs):
            ref.tick(h, j)
            if (f + 1) % stride == 0:
                anchors.append((f + 1, ref.save_ninja_state(),
                                ref.get_ninja_pos(), ref.get_ninja_velocity()))
            if ref.has_won():
                anchors.append((f + 1, ref.save_ninja_state(),
                                ref.get_ninja_pos(), ref.get_ninja_velocity()))
                break
        log(f"  reference: {len(ref_inputs)} frames, {len(anchors)} anchors "
            f"(stride={stride})")

        self.main.fast_reset()
        tape: List[Tuple[int, int]] = []
        auto_frames = fb_frames = 0
        prev_f = 0
        for ai, (af, _blob, (ax, ay), (avx, avy)) in enumerate(anchors):
            is_last = (ai == len(anchors) - 1)
            def ok(x, y, vx, vy, gz, _ax=ax, _ay=ay, _avx=avx, _avy=avy):
                return (math.hypot(x - _ax, y - _ay) < pos_tol
                        and abs(vx - _avx) < vel_tol and abs(vy - _avy) < vel_tol)
            def heur(x, y, gz, _ax=ax, _ay=ay):
                return math.hypot(x - _ax, y - _ay) * _INV_V
            span = af - prev_f
            acts = None
            if not is_last:  # the win frame is sub-pixel: finish on its checkpoint
                acts, _st, exp = self._segment(
                    self.main.save_ninja_state(), heur=heur, success=ok,
                    grazes=False, max_exp=seg_max_exp,
                    max_steps=span // self.macro + 8, w=1.1)
            if acts is not None:
                ev = self._apply(acts, tape)
                auto_frames += len(acts) * self.macro
                mode = "auto"
            else:
                # Sub-pixel coupling defeated search for this span: re-sync to the
                # demo CHECKPOINT (restore the anchor's exact state) so divergence
                # from earlier auto-segments can't compound, then record the demo
                # inputs for the span (checkpoint-seeded execution; see
                # replay.demo_checkpoint_seeder).
                for fi in range(prev_f, af):
                    tape.append(ref_inputs[fi])
                self.main.restore_ninja_state(_blob)
                ev = {"won": bool(self.main.has_won()),
                      "died": bool(self.main.has_died()),
                      "switch": _switch_done(self.main)}
                fb_frames += af - prev_f
                mode = "demo-checkpoint"
            log(f"    anchor {ai+1}/{len(anchors)} f{af} ({mode}) "
                f"pos={_fmt(self.main.get_ninja_pos())} won={ev['won']}")
            prev_f = af
            if ev["won"]:
                total = auto_frames + fb_frames
                return {"won": True, "tape": tape, "frames": len(tape),
                        "auto_frames": auto_frames, "fallback_frames": fb_frames,
                        "auto_fraction": auto_frames / max(1, total),
                        "wall_time_s": time.time() - t0,
                        "launch_points": self._launch_pts}
            if ev["died"]:
                return {"won": False, "failed_stage": f"anchor {ai+1} (died)",
                        "tape": tape, "frames": len(tape),
                        "wall_time_s": time.time() - t0}
        return {"won": False, "failed_stage": "exhausted anchors w/o win",
                "tape": tape, "frames": len(tape),
                "wall_time_s": time.time() - t0}

    def _win(self, tape, t0, exp):
        return {"won": True, "tape": tape, "frames": len(tape),
                "expansions": exp, "wall_time_s": time.time() - t0,
                "launch_points": self._launch_pts}

    def _fail(self, stage, exp, tape, t0):
        return {"won": False, "failed_stage": stage, "tape": tape,
                "frames": len(tape), "expansions": exp,
                "wall_time_s": time.time() - t0,
                "launch_points": self._launch_pts}


def _fmt(p):
    return f"({p[0]:.1f},{p[1]:.1f})"


def render_video(map_path: str, tape: List[Tuple[int, int]], output: str,
                 fps: int = 60) -> bool:
    """Replay the validated (h,j) tape on a render sim (animation on) -> mp4."""
    import imageio
    from ...nplay_headless import NPlayHeadless
    from .route_video import _to_hwc
    nh = NPlayHeadless(render_mode="rgb_array", enable_animation=True,
                       enable_debug_overlay=False)
    nh.load_map(map_path)
    nh.reset()
    frames = [_to_hwc(nh.render())]
    for _ in range(20):  # settle to rest (matches planner)
        nh.tick(0, 0)
        frames.append(_to_hwc(nh.render()))
        vx, vy = nh.ninja_velocity()
        if abs(vx) < 0.01 and abs(vy) < 0.01:
            break
    won = False
    for (h, j) in tape:
        nh.tick(h, j)
        frames.append(_to_hwc(nh.render()))
        if nh.ninja_has_won():
            won = True
            break
        if nh.ninja_has_died():
            break
    frames.extend([frames[-1]] * fps)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    imageio.mimsave(output, frames, fps=fps, macro_block_size=1)
    return won


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True)
    p.add_argument("--video", default=None, help="render a completion mp4 here")
    p.add_argument("--macro", type=int, default=2)
    p.add_argument("--max-exp", type=int, default=300000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--attract", default=None,
                   help="npp_attract reference file: re-derive a win by connecting "
                        "its anchor states with the enriched micro-search "
                        "(distillation mode, for sub-pixel trick levels)")
    p.add_argument("--stride", type=int, default=24,
                   help="reference anchor spacing (frames) for --attract mode")
    args = p.parse_args()

    name = os.path.basename(args.map.rstrip("/"))
    print(f"[{name}] micro-search finisher (macro={args.macro})...")
    f = MicroSearchFinisher(args.map, workers=args.workers, macro=args.macro)
    print(f"  launch points (exit-air seeds): "
          f"{[_fmt(p) for p in f._launch_pts[:4]]}")
    if args.attract:
        from ...replay.npp_attract_decoder import NppAttractDecoder
        from ...replay.input_decoder import InputDecoder
        dec = NppAttractDecoder().decode_npp_attract_file(args.attract)
        hor, jmp = InputDecoder().decode_inputs(dec["demo_data"])
        ref_inputs = list(zip(hor, jmp))
        res = f.plan_from_reference(ref_inputs, stride=args.stride,
                                    seg_max_exp=args.max_exp)
        verdict = "WON" if res["won"] else f"FAILED@{res.get('failed_stage')}"
        af = res.get("auto_fraction")
        extra = (f"  auto-rederived={af*100:.0f}% "
                 f"({res.get('auto_frames')}/{res['frames']}f)" if af is not None else "")
        print(f"  {verdict}  frames={res['frames']}{extra}  {res['wall_time_s']:.1f}s")
        if res["won"] and args.video:
            ok = render_video(args.map, res["tape"], args.video)
            print(f"  video {'WON (continuous tape reproduces) ' if ok else 'replay-DIVERGED (checkpoint discontinuity) '}-> {args.video}")
    else:
        res = f.plan(max_exp=args.max_exp)
        verdict = "WON" if res["won"] else f"FAILED@{res.get('failed_stage')}"
        print(f"  {verdict}  frames={res['frames']}  "
              f"expansions={res.get('expansions')}  {res['wall_time_s']:.1f}s")
    if res["won"] and args.video:
        ok = render_video(args.map, res["tape"], args.video)
        print(f"  video {'WON ' if ok else 'replay-mismatch '}-> {args.video}")
    return 0 if res["won"] else 2


if __name__ == "__main__":
    sys.exit(main())
