#!/usr/bin/env python3
"""Render the lattice ``d*`` path (the planner's route edges) as a PNG per level.

For each planner-supported official level this draws the jointly-optimal
spawn -> switch -> exit route from the backward-Dijkstra ``d*`` tree (loop-deduped,
risk-weighted: completion>loops>speed) over the rendered level.  This is the
*lattice path* — the sequence of arc edges, NOT a frame-exact executed route.

Each arc is one edge drawn launch -> landing (phase-0 = spawn->switch in cyan,
phase-1 = switch->exit in lime; jumps thicker), with a dot at every lattice node.
Deadly-mine kill-zones, spawn / switch / exit markers, and a status HUD are
overlaid.  World pixels map 1:1 to frame pixels (see ``route_video``).

Subset: only the ``S*`` official folders, and only levels whose entities are a
subset of {toggle mine, gold, exit door, exit switch} with exactly one switch +
one exit (the same filter as ``benchmark_filter``).

Usage::

    .venv/bin/python -m nclone.value_function.tools.render_lattice_paths \
        --official-dir nclone/maps/official --out-dir /tmp/lattice_paths \
        [--limit-per-folder N] [--processes K]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from typing import List, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless rendering (WSL/CI)

import numpy as np
from PIL import Image, ImageDraw

from ...nplay_headless import NPlayHeadless
from .lattice_audit import LatticeAudit
from .route_video import _to_hwc
from .physics_route_spike import _MINE_KILL_PX

# Entity-type keys the planner supports (mirrors benchmark_filter.ALLOWED).
ALLOWED = {1, 2, 3, 21}  # mine, gold, exit(door+switch), toggled-mine

_S_FOLDER_PREFIX = "S"  # the "S*" episode folders (S, S2, SI, SL, SL2, SS, SS2)

# Colours (RGBA).
_C_PHASE0 = (0, 220, 255)     # spawn -> switch
_C_PHASE1 = (140, 255, 60)    # switch -> exit
_C_NODE = (255, 255, 255)
_C_SPAWN = (255, 230, 0)
_C_SWITCH = (255, 150, 0)
_C_EXIT = (80, 255, 120)


def _supported(nh: NPlayHeadless) -> Tuple[bool, str]:
    """Filter: entities subset of ALLOWED with exactly one exit + one switch."""
    ed = nh.sim.entity_dic
    keys = {k for k, v in ed.items() if v}
    if keys - ALLOWED:
        return False, "unsupported_entity"
    ex = ed.get(3, [])
    n_door = sum(1 for e in ex if type(e).__name__ == "EntityExit")
    n_switch = sum(1 for e in ex if type(e).__name__ == "EntityExitSwitch")
    if n_door == 1 and n_switch == 1:
        return True, "ok"
    return False, "bad_goal"


def _seg(d, a, b, color, width):
    d.line([(a[0], a[1]), (b[0], b[1])], fill=color, width=width)


def _draw_route(frame_hwc, audit, phase0, phase1, info, status) -> np.ndarray:
    img = Image.fromarray(frame_hwc).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    # Deadly-mine kill zones.
    for (mx, my) in audit.deadly:
        d.ellipse([mx - _MINE_KILL_PX, my - _MINE_KILL_PX,
                   mx + _MINE_KILL_PX, my + _MINE_KILL_PX],
                  fill=(255, 0, 0, 30), outline=(255, 60, 60, 160), width=1)

    def _draw_arcs(arcs, color):
        for arc in arcs:
            w = 3 if arc.jump_hold_frames > 0 else 2
            _seg(d, (arc.launch_x, arc.launch_y),
                 (arc.landing_x, arc.landing_y), color + (235,), w)
            for (px, py) in ((arc.launch_x, arc.launch_y),
                             (arc.landing_x, arc.landing_y)):
                d.ellipse([px - 2, py - 2, px + 2, py + 2],
                          fill=_C_NODE + (220,))

    # spawn -> first launch connector (the approach), then the arc edges.
    if phase0:
        _seg(d, audit.spawn, (phase0[0].launch_x, phase0[0].launch_y),
             _C_PHASE0 + (120,), 1)
    _draw_arcs(phase0, _C_PHASE0)
    _draw_arcs(phase1, _C_PHASE1)

    # Entity markers.
    sx, sy = audit.spawn
    d.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], outline=_C_SPAWN + (255,), width=2)
    wx, wy = audit.switch
    d.ellipse([wx - 5, wy - 5, wx + 5, wy + 5], outline=_C_SWITCH + (255,), width=2)
    ex_, ey_ = audit.exit
    d.rectangle([ex_ - 5, ey_ - 5, ex_ + 5, ey_ + 5], outline=_C_EXIT + (255,), width=2)

    img = Image.alpha_composite(img, ov)
    d2 = ImageDraw.Draw(img)
    d2.rectangle([0, 0, 430, 30], fill=(0, 0, 0, 160))
    d2.text((5, 3), f"{audit.name}", fill=(255, 255, 255, 255))
    d2.text((5, 16), status, fill=(180, 230, 255, 255))
    return np.asarray(img.convert("RGB"))


def render_one(map_path: str, out_dir: str,
               w_risk: Optional[float] = None) -> dict:
    """Render one level's lattice path; returns a small result dict.

    Skips levels that fail the planner-support filter.  Reachability is reported
    in the HUD + result (``reachable`` = spawn route reaches switch AND exit).
    """
    name = os.path.basename(map_path.rstrip("/"))
    try:
        nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        nh.load_map(map_path)
        ok, why = _supported(nh)
        if not ok:
            return {"name": name, "status": "skip", "reason": why}

        from ..edge_risk import DEFAULT_W_RISK
        audit = LatticeAudit(
            map_path, w_risk=DEFAULT_W_RISK if w_risk is None else w_risk)
        pc = audit.precompute(use_cache=True)
        phase0, phase1, info = audit.build_route(pc)

        d_sw = info.get("spawn_d_switch", float("inf"))
        d_ex = info.get("spawn_d_exit_from_switch", float("inf"))
        reachable = np.isfinite(d_sw) and np.isfinite(d_ex)
        n_arcs = len(phase0) + len(phase1)
        if reachable:
            status = (f"REACHABLE  arcs={n_arcs} (p0={len(phase0)} p1={len(phase1)})"
                      f"  d*={d_sw + d_ex:.0f}f")
        elif np.isfinite(d_sw):
            status = f"PARTIAL: switch reachable, exit NOT  (p0={len(phase0)})"
        else:
            status = "NO ROUTE (spawn cannot reach switch on the lattice)"

        raw = _to_hwc(nh.render())
        img = _draw_route(raw, audit, phase0, phase1, info, status)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}.png")
        Image.fromarray(img).save(out_path)
        return {"name": name, "status": "ok", "reachable": bool(reachable),
                "n_arcs": n_arcs, "out": out_path}
    except Exception as e:  # noqa: BLE001
        return {"name": name, "status": "error", "reason": f"{type(e).__name__}: {e}"}


def _worker(args):
    map_path, out_dir, w_risk = args
    return render_one(map_path, out_dir, w_risk=w_risk)


def _gather_s_maps(official_dir: str, limit_per_folder: Optional[int]) -> List[str]:
    folders = sorted(
        d for d in glob.glob(os.path.join(official_dir, _S_FOLDER_PREFIX + "*"))
        if os.path.isdir(d)
    )
    maps: List[str] = []
    for folder in folders:
        fmaps = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                       if os.path.isfile(f))
        if limit_per_folder is not None:
            fmaps = fmaps[:limit_per_folder]
        maps.extend(fmaps)
    return maps


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--official-dir", default="nclone/maps/official")
    p.add_argument("--out-dir", default="/tmp/lattice_paths")
    p.add_argument("--limit-per-folder", type=int, default=None,
                   help="render only the first N levels per S* folder (smoke)")
    p.add_argument("--processes", type=int, default=1,
                   help="parallel render processes (default 1; d* cache is shared)")
    p.add_argument("--w-risk", type=float, default=None)
    args = p.parse_args()

    maps = _gather_s_maps(args.official_dir, args.limit_per_folder)
    print(f"S* folders: scanning {len(maps)} candidate levels -> {args.out_dir}")

    results = []
    if args.processes <= 1:
        for i, m in enumerate(maps):
            r = render_one(m, args.out_dir, w_risk=args.w_risk)
            results.append(r)
            _progress(i + 1, len(maps), r)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.processes) as ex:
            futs = [ex.submit(_worker, (m, args.out_dir, args.w_risk)) for m in maps]
            for i, f in enumerate(as_completed(futs)):
                r = f.result()
                results.append(r)
                _progress(i + 1, len(maps), r)

    supported = [r for r in results if r["status"] in ("ok",)]
    reachable = [r for r in supported if r.get("reachable")]
    skipped = [r for r in results if r["status"] == "skip"]
    errored = [r for r in results if r["status"] == "error"]
    print(f"\n=== lattice path render ===\n"
          f"  candidates={len(maps)}  supported(rendered)={len(supported)}  "
          f"reachable={len(reachable)}  skipped(unsupported)={len(skipped)}  "
          f"errors={len(errored)}\n"
          f"  images -> {args.out_dir}")
    if supported:
        print(f"  d* route coverage (reachable / rendered): "
              f"{100.0*len(reachable)/len(supported):.1f}%")
    for r in errored[:15]:
        print(f"    ERROR {r['name']}: {r['reason']}")
    return 0


def _progress(done, total, r):
    tag = {"ok": ("R" if r.get("reachable") else "u"), "skip": "·",
           "error": "E"}.get(r["status"], "?")
    extra = ""
    if r["status"] == "ok":
        extra = f"arcs={r['n_arcs']} reachable={r.get('reachable')}"
    elif r["status"] != "ok":
        extra = r.get("reason", "")
    print(f"  [{done}/{total}] {tag} {r['name']}  {extra}")


if __name__ == "__main__":
    sys.exit(main())
