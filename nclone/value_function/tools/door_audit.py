"""How much does the door-blind free mask actually cost?

``free_space_distance._build`` historically read exactly entity types 4
(switch), 3 (exit) and 1/21 (mines).  Doors appeared nowhere, so every flood —
planner ``d*`` guidance, the ``cost_to_go_map`` observation, and the free-space
PBRS potential — routed straight through a shut locked door.

Only type 6 is a genuine defect, and that is worth stating precisely because the
other two look like the same bug and are not:

* type 5 REGULAR opens on 10px proximity to its own switch and re-closes 5
  frames later.  It yields to any ninja that walks up to it, so for ROUTING it
  is permanently open.  Correctly absent.
* type 8 TRAP starts OPEN and shuts only if the ninja touches its switch.  An
  avoidable, self-inflicted barrier; optimistically open.  Correctly absent.
* type 6 LOCKED starts SHUT and opens only when its switch is touched.  A real
  barrier — and a PHASE-DEPENDENT one.

What this measures, per level with at least one locked door, is the gap between
the two extreme topologies (``doors="open"`` = today, ``doors="closed"`` = the
phase-0 truth):

``mispriced``   fraction of free cells whose cost-to-go the open mask makes
                strictly cheaper than the closed one — cells whose gradient
                currently points at a shut door.
``phantom``     fraction of free cells that are finite under ``open`` and
                ``inf`` under ``closed``: reachable ONLY by walking through a
                shut door.  This is the mis-routing in its purest form.
``gated``       whether the goal is unreachable at phase 0 at all.  This is the
                number that decides whether phase-aware legs are MANDATORY or
                merely nicer: on a gated level ``doors="closed"`` alone would
                destroy the gradient, so flipping the default is not an option
                and the door_dependency legs have to be wired through.

Usage::

    python -m nclone.value_function.tools.door_audit \\
        --official-root nclone/maps/official --workers 6 \\
        --out /tmp/door_audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

#: Entity types whose absence from the free mask is a deliberate optimistic
#: choice rather than a defect; reported for context only.
_BENIGN_DOOR_TYPES = (5, 8)


def _scan_one(path: str) -> Optional[dict]:
    """Count doors on one level without building anything expensive."""
    try:
        import nclone_native
        with open(path, "rb") as f:
            md = [int(b) for b in f.read()]
        sim = nclone_native.Simulator.load(list(md))
        counts: Dict[int, int] = {}
        for e in sim.get_state().entities:
            t = int(e.entity_type)
            if t in (5, 6, 8):
                counts[t] = counts.get(t, 0) + 1
        return {"path": path, "name": os.path.basename(path),
                "folder": os.path.basename(os.path.dirname(path)),
                "n_regular": counts.get(5, 0), "n_locked": counts.get(6, 0),
                "n_trap": counts.get(8, 0)}
    except Exception as exc:
        return {"path": path, "name": os.path.basename(path),
                "error": f"{type(exc).__name__}: {exc}"}


def _measure_one(args) -> dict:
    """Build both door topologies for one level and diff them."""
    path, physics = args
    rec: Dict[str, object] = {"name": os.path.basename(path),
                              "folder": os.path.basename(os.path.dirname(path))}
    t0 = time.time()
    try:
        import nclone_native
        from ..free_space_distance import FreeSpaceDistanceSource, _CELL, _W, _H

        with open(path, "rb") as f:
            md = [int(b) for b in f.read()]
        sim = nclone_native.Simulator.load(list(md))
        spawn = sim.get_ninja_pos()

        fields = {}
        for mode in ("open", "closed"):
            fs = FreeSpaceDistanceSource(md, mine_hard=True,
                                         physics_cost=physics, doors=mode)
            if not fs.available or fs._free is None:
                rec["error"] = f"{mode}: unavailable"
                return rec
            fields[mode] = fs
        rec["n_locked"] = len(fields["open"].locked_doors)
        free = fields["open"]._free
        n_free = int(free.sum())
        rec["n_free_cells"] = n_free
        if n_free == 0:
            rec["error"] = "no free cells"
            return rec

        # The pre-switch leg is what the agent descends first, so the switch
        # field is the one whose mis-routing actually reaches the policy.
        for leg in ("switch", "exit"):
            o = getattr(fields["open"], f"_d_{leg}")
            c = getattr(fields["closed"], f"_d_{leg}")
            fin_o = np.isfinite(o) & free
            fin_c = np.isfinite(c) & free
            phantom = fin_o & ~fin_c
            both = fin_o & fin_c
            # Strictly cheaper through the door, beyond float noise.  Both arms
            # run the same flood on the same grid, so an untouched cell comes
            # back bit-identical.  The difference is taken only where BOTH are
            # finite: elsewhere it is inf-inf = nan, which is not a comparison.
            mis = np.zeros_like(both)
            mis[both] = (c[both] - o[both]) > 1e-6
            rec[leg] = {
                "frac_phantom": float(phantom.sum()) / n_free,
                "frac_mispriced": float(mis.sum()) / n_free,
                "mispriced_rel_p50": (
                    float(np.median((c[mis] - o[mis]) / np.maximum(o[mis], 1e-9)))
                    if mis.any() else None),
                "coverage_open": float(fin_o.sum()) / n_free,
                "coverage_closed": float(fin_c.sum()) / n_free,
            }

        # Is the goal reachable AT ALL with every door shut?  If not, the level
        # genuinely needs the phase legs and "just close the doors" is not a
        # available fix.
        sx, sy = int(spawn[0] / _CELL), int(spawn[1] / _CELL)
        sx, sy = min(max(sx, 0), _W - 1), min(max(sy, 0), _H - 1)
        d_open = fields["open"]._d_switch
        d_closed = fields["closed"]._d_switch
        rec["spawn_finite_open"] = bool(np.isfinite(d_open[sx, sy]))
        rec["spawn_finite_closed"] = bool(np.isfinite(d_closed[sx, sy]))
        rec["gated"] = bool(rec["spawn_finite_open"] and not rec["spawn_finite_closed"])
        rec["build_s"] = time.time() - t0
        return rec
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["build_s"] = time.time() - t0
        return rec


def _agg(vals) -> dict:
    v = [float(x) for x in vals if x is not None]
    if not v:
        return {"n": 0}
    s = sorted(v)
    return {"n": len(v), "mean": sum(v) / len(v),
            "p50": s[len(s) // 2], "p90": s[int(0.9 * (len(s) - 1))],
            "max": s[-1]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official-root", default="nclone/maps/official")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-physics", action="store_true",
                    help="geometric cost model (faster; ordering only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    paths: List[str] = []
    for folder in sorted(os.listdir(args.official_root)):
        d = os.path.join(args.official_root, folder)
        if os.path.isdir(d):
            paths.extend(os.path.join(d, f) for f in sorted(os.listdir(d)))
    print(f"[doors] scanning {len(paths)} official maps")

    t0 = time.time()
    scans: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_scan_one, paths, chunksize=16):
            if r:
                scans.append(r)
    ok = [s for s in scans if "error" not in s]
    with_locked = [s for s in ok if s["n_locked"] > 0]
    census = {
        "n_maps": len(scans),
        "n_scan_errors": len(scans) - len(ok),
        "n_with_locked_door": len(with_locked),
        "n_with_regular_door": sum(1 for s in ok if s["n_regular"] > 0),
        "n_with_trap_door": sum(1 for s in ok if s["n_trap"] > 0),
        "locked_per_level": _agg([s["n_locked"] for s in with_locked]),
    }
    print(json.dumps(census, indent=2))

    targets = [s["path"] for s in with_locked]
    if args.limit:
        targets = targets[:args.limit]
    print(f"[doors] measuring {len(targets)} locked-door levels "
          f"({time.time()-t0:.0f}s scanning)")

    records: List[dict] = []
    work = [(p, not args.no_physics) for p in targets]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_measure_one, w) for w in work]
        for i, fut in enumerate(as_completed(futs)):
            records.append(fut.result())
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(futs)} ({time.time()-t0:.0f}s)")

    good = [r for r in records if "error" not in r]
    summary = {
        "census": census,
        "n_measured": len(records),
        "n_errors": len(records) - len(good),
        "n_gated_at_phase0": sum(1 for r in good if r.get("gated")),
        "frac_gated_at_phase0": (sum(1 for r in good if r.get("gated")) / len(good)
                                 if good else None),
        "build_s": _agg([r.get("build_s") for r in good]),
    }
    for leg in ("switch", "exit"):
        legs = [r[leg] for r in good if isinstance(r.get(leg), dict)]
        summary[leg] = {
            "frac_phantom": _agg([x.get("frac_phantom") for x in legs]),
            "frac_mispriced": _agg([x.get("frac_mispriced") for x in legs]),
            "mispriced_rel_p50": _agg([x.get("mispriced_rel_p50") for x in legs]),
            "n_levels_any_phantom": sum(
                1 for x in legs if (x.get("frac_phantom") or 0) > 0),
            "n_levels_any_mispriced": sum(
                1 for x in legs if (x.get("frac_mispriced") or 0) > 0),
        }
    summary["elapsed_s"] = time.time() - t0
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "records": records,
                       "census_rows": with_locked}, f, indent=2)
        print(f"[doors] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
