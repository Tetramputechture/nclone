"""Animate the WHOLE lattice planner operation, start to completion.

Tells the planner's story as a single mp4, layered over the rendered level:

  Phase 1  d* field      — backward-Dijkstra cost-to-go painted on every surface
                           node (green = near goal, red = far): the gradient the
                           planner descends.  Curved/slope nodes appear here too
                           under the tangential model.
  Phase 2  search        — the d*-guided FORWARD search expanding (the ``explored``
                           frontier from dstar_guided_spike), coloured by
                           cost-to-spawn: "traversing the graph".
  Phase 3  route         — the chosen path traced arc-by-arc spawn -> switch ->
                           exit (runs straight, jumps as arcs): "finding the
                           correct path".
  Phase 4  completion    — the winning physics trajectory drawn to the goal.

Everything is world-space (1056x600), matching the headless render 1:1.

Usage:
    .venv/bin/python -m nclone.value_function.tools.planner_viz \
        --map "nclone/maps/official/SL/250 pipeworks" \
        --output planner.mp4 --momentum-model tangential --h-weight 1.0
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

import numpy as np

from ...nplay_headless import NPlayHeadless
from ..route_extraction import extract_and_simplify_route
from .lattice_audit import LatticeAudit
from .route_video import _to_hwc

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - PIL is a render-only dep
    Image = ImageDraw = None

_MINE_KILL_PX = 14.0


def _per_point_min_dstar(d_star: np.ndarray, n_pts: int, max_bins: int) -> np.ndarray:
    """Min finite d* over a point's velocity bins (the velocity-agnostic potential)."""
    out = np.full(n_pts, np.inf, dtype=np.float64)
    for i in range(n_pts):
        block = d_star[i * max_bins:(i + 1) * max_bins]
        fin = block[np.isfinite(block)]
        if fin.size:
            out[i] = float(fin.min())
    return out


def _heat(t: float) -> Tuple[int, int, int]:
    """t in [0,1] -> green (0, near goal) to red (1, far)."""
    t = max(0.0, min(1.0, t))
    return (int(255 * t), int(200 * (1.0 - t)), 60)


def _blend(base: "Image.Image", overlay: "Image.Image") -> "Image.Image":
    return Image.alpha_composite(base, overlay)


def _new_overlay(size) -> Tuple["Image.Image", "ImageDraw.ImageDraw"]:
    ov = Image.new("RGBA", size, (0, 0, 0, 0))
    return ov, ImageDraw.Draw(ov)


def _dot(d, x, y, r, color):
    d.ellipse([x - r, y - r, x + r, y + r], fill=color)


def _arc_points(arc, n: int = 10) -> List[Tuple[float, float]]:
    """Sample an arc for drawing: a quadratic bezier bowed by jump hold for
    jumps, a straight segment for runs."""
    x0, y0 = arc.launch_x, arc.launch_y
    x1, y1 = arc.landing_x, arc.landing_y
    if arc.jump_hold_frames <= 0:
        return [(x0, y0), (x1, y1)]
    # control point lifted above the chord midpoint, scaled by hold duration
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    lift = min(60.0, 6.0 + 1.5 * arc.jump_hold_frames)
    cx, cy = mx, my - lift
    pts = []
    for i in range(n + 1):
        s = i / n
        u = 1.0 - s
        px = u * u * x0 + 2 * u * s * cx + s * s * x1
        py = u * u * y0 + 2 * u * s * cy + s * s * y1
        pts.append((px, py))
    return pts


def _draw_static(d, switch, exit_, spawn, deadly):
    """Persistent markers: spawn / switch / exit / mine kill-zones."""
    for (mx, my) in deadly:
        d.ellipse([mx - _MINE_KILL_PX, my - _MINE_KILL_PX,
                   mx + _MINE_KILL_PX, my + _MINE_KILL_PX],
                  fill=(255, 0, 0, 55), outline=(255, 70, 70, 160))
    if spawn:
        _dot(d, spawn[0], spawn[1], 5, (255, 255, 255, 255))
    if switch:
        _dot(d, switch[0], switch[1], 6, (255, 220, 0, 255))
    if exit_:
        _dot(d, exit_[0], exit_[1], 6, (0, 230, 120, 255))


def _hud(d, text, size):
    d.rectangle([0, 0, size[0], 16], fill=(0, 0, 0, 150))
    d.text((6, 3), text, fill=(255, 255, 255, 255))


def visualize_planner(map_path: str, output: str, *, momentum_model: str = "tangential",
                      h_weight: float = 1.0, fps: int = 30, hold: int = 45,
                      run_spike: bool = True, workers: int = 4) -> dict:
    """Render the full planner animation to ``output`` (mp4). Returns a stats dict."""
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required for planner_viz")
    import imageio

    audit = LatticeAudit(map_path, workers=workers, momentum_model=momentum_model)
    pc = audit.precompute(use_cache=True)
    graph, d_comb = pc["graph"], pc["d_comb"]
    spawn_node = audit._spawn_node(graph, d_comb)
    route = (extract_and_simplify_route(graph, d_comb, pc["pred_comb"], pc["best_comb"],
                                        spawn_node, dedup_loops=True)
             if spawn_node is not None else [])

    spike = {}
    if run_spike:
        from .dstar_guided_spike import plan_dstar_guided
        spike = plan_dstar_guided(map_path, h_weight=h_weight, momentum_model=momentum_model)

    # Base level frame (ninja at spawn).
    nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    nh.load_map(map_path)
    nh.reset()
    base_rgb = _to_hwc(nh.render())
    base = Image.fromarray(base_rgb).convert("RGBA")
    size = base.size

    # Per-point d* heatmap data.
    pts = graph.surface_points
    xy = np.array([(p.x, p.y) for p in pts], dtype=np.float64)
    dmin = _per_point_min_dstar(d_comb, len(pts), graph.max_vel_bins)
    finite = np.isfinite(dmin)
    dmax = float(dmin[finite].max()) if finite.any() else 1.0

    switch, exit_, spawn = audit.switch, audit.exit, audit.spawn
    deadly = audit.deadly
    frames: List[np.ndarray] = []

    def emit(overlay, text):
        comp = _blend(base, overlay)
        d2 = ImageDraw.Draw(comp)
        _hud(d2, text, size)
        frames.append(np.asarray(comp.convert("RGB")))

    # ---- Phase 1: d* field reveals over the lattice nodes ----
    for k in range(hold):
        frac = (k + 1) / hold
        ov, d = _new_overlay(size)
        shown = int(frac * len(pts))
        for i in range(shown):
            if finite[i]:
                _dot(d, xy[i, 0], xy[i, 1], 2, (*_heat(dmin[i] / dmax), 200))
        _draw_static(d, switch, exit_, spawn, deadly)
        emit(ov, f"[1/4] d* cost-to-go field  ({momentum_model})  nodes={len(pts)}")

    # Persistent faint d* layer for later phases.
    base_d, dd = _new_overlay(size)
    for i in range(len(pts)):
        if finite[i]:
            _dot(dd, xy[i, 0], xy[i, 1], 2, (*_heat(dmin[i] / dmax), 70))
    base = _blend(base, base_d)

    # ---- Phase 2: forward search expansion ("traversing the graph") ----
    explored = spike.get("explored") or []
    if explored:
        ex = np.array(explored, dtype=np.float64)  # (N, 3) = x, y, g
        gmax = float(ex[:, 2].max()) if len(ex) else 1.0
        n_steps = min(hold, len(ex))
        for k in range(n_steps):
            upto = int((k + 1) / n_steps * len(ex))
            ov, d = _new_overlay(size)
            for j in range(upto):
                g = ex[j, 2] / gmax if gmax else 0.0
                c = (int(120 + 135 * g), int(160 * (1 - g) + 60), 255, 160)
                _dot(d, ex[j, 0], ex[j, 1], 2, c)
            _draw_static(d, switch, exit_, spawn, deadly)
            emit(ov, f"[2/4] d*-guided search expanding  explored={upto}/{len(ex)}")

    # ---- Phase 3: chosen route traced arc-by-arc ----
    if route:
        # precompute polylines per arc
        arc_polys = [(_arc_points(a), a.jump_hold_frames > 0) for a in route]
        n_steps = max(len(route), 1)
        for k in range(n_steps):
            ov, d = _new_overlay(size)
            for idx in range(k + 1):
                poly, is_jump = arc_polys[idx]
                col = (255, 180, 0, 255) if is_jump else (0, 230, 255, 255)
                scr = [(p[0], p[1]) for p in poly]
                if len(scr) >= 2:
                    d.line(scr, fill=col, width=2)
            # highlight current arc launch
            la = route[k]
            _dot(d, la.launch_x, la.launch_y, 3, (255, 255, 255, 255))
            _draw_static(d, switch, exit_, spawn, deadly)
            emit(ov, f"[3/4] route  arc {k+1}/{len(route)}  "
                     f"(amber=jump cyan=run)")
        # hold the full route a beat
        for _ in range(max(1, hold // 3)):
            ov, d = _new_overlay(size)
            for (poly, is_jump) in arc_polys:
                col = (255, 180, 0, 255) if is_jump else (0, 230, 255, 255)
                d.line([(p[0], p[1]) for p in poly], fill=col, width=2)
            _draw_static(d, switch, exit_, spawn, deadly)
            emit(ov, f"[3/4] route complete  arcs={len(route)}")

    # ---- Phase 4: winning physics trajectory to completion ----
    traj = spike.get("trajectory") or []
    solved = bool(spike.get("solved"))
    if traj:
        tr = np.array(traj, dtype=np.float64)
        n_steps = min(2 * hold, len(tr))
        for k in range(n_steps):
            upto = int((k + 1) / n_steps * len(tr))
            ov, d = _new_overlay(size)
            if upto >= 2:
                d.line([(tr[j, 0], tr[j, 1]) for j in range(upto)],
                       fill=(255, 255, 0, 230), width=2)
            _dot(d, tr[upto - 1, 0], tr[upto - 1, 1], 4, (255, 255, 255, 255))
            _draw_static(d, switch, exit_, spawn, deadly)
            tag = "WON" if solved else "no-win"
            emit(ov, f"[4/4] completion  frame {upto}/{len(tr)}  "
                     f"{tag}  frames={spike.get('frames')}")

    imageio.mimsave(output, frames, fps=fps, macro_block_size=1)
    return {
        "output": output, "n_frames": len(frames), "n_route_arcs": len(route),
        "n_explored": len(explored), "solved": solved,
        "spike_frames": spike.get("frames"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True)
    p.add_argument("--output", required=True, help="mp4 path")
    p.add_argument("--momentum-model", choices=["legacy", "tangential"], default="tangential")
    p.add_argument("--h-weight", type=float, default=1.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--hold", type=int, default=45, help="frames per phase segment")
    p.add_argument("--no-spike", action="store_true",
                   help="skip the forward-search + trajectory phases (d*+route only)")
    args = p.parse_args()
    res = visualize_planner(args.map, args.output, momentum_model=args.momentum_model,
                            h_weight=args.h_weight, fps=args.fps, hold=args.hold,
                            run_spike=not args.no_spike)
    print(f"wrote {res['output']}  frames={res['n_frames']} route_arcs={res['n_route_arcs']} "
          f"explored={res['n_explored']} solved={res['solved']} spike_frames={res['spike_frames']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
