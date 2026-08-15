"""Phase-split the lattice precompute to find where the time actually goes.

`lattice-precompute-pathological-levels` recorded 380-480s on long flat corridors
with "search itself is 0.0s", and the root cause inside `audit.precompute()`
explicitly NOT diagnosed. Tuning budgets is useless here -- max_expansions
60000->8000 moved 352s to 348s -- so measure the phases before touching anything.

use_cache=False throughout: a repeat measurement of the SAME level reads ~0.3s
off a cache hit, which is how this cost hides.

Usage:  python profile_lattice.py <n_levels>
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402


def profile(map_path):
    from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
    from nclone.value_function.surface_extraction import extract_surfaces
    from nclone.value_function.surface_graph import build_graph
    from nclone.value_function.tools.lattice_audit import LatticeAudit
    from nclone.value_function.tools.physics_route_spike import PhysicsRouteSpike

    t = {}
    t0 = time.time()
    map_data = PhysicsRouteSpike(str(map_path)).map_data
    audit = LatticeAudit(map_data=list(map_data), name=map_path.name)
    t["load"] = time.time() - t0

    t0 = time.time()
    td = PhysicsStepWrapper(map_data=audit.map_data).tile_dic
    t["phys"] = time.time() - t0

    t0 = time.time()
    pts = extract_surfaces(
        td,
        mine_positions=audit.all_mines,
        momentum_model=audit.momentum_model,
        keep_mine_geometry=audit.keep_mine_geometry,
    )
    t["surfaces"] = time.time() - t0

    t0 = time.time()
    graph = build_graph(pts)
    t["build_graph"] = time.time() - t0

    # The rest of precompute(): edge enumeration + the backward Dijkstras.
    t0 = time.time()
    audit.precompute(use_cache=False)
    t["rest_of_precompute"] = time.time() - t0

    return t, len(pts), graph.n_nodes


def main():
    root = Path("nclone/maps/official")
    maps = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix == "" and not p.name.startswith(".")
    )[: int(sys.argv[1])]

    rows = []
    for m in maps:
        try:
            t, n_pts, n_nodes = profile(m)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {m.name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        total = sum(t.values())
        rows.append((total, n_nodes, n_pts, m.name, t))
        print(f"  {total:7.2f}s nodes={n_nodes:6d} {m.name[:32]}", flush=True)

    if not rows:
        print("no levels profiled")
        return
    rows.sort(reverse=True)
    print(f"\n{'total':>8} {'nodes':>7} {'points':>7}  phases")
    for total, n_nodes, n_pts, name, t in rows[:15]:
        ph = " ".join(f"{k}={v:.2f}" for k, v in t.items())
        print(f"{total:8.2f} {n_nodes:7d} {n_pts:7d}  {name[:32]:32s} {ph}")

    nodes = np.array([r[1] for r in rows], dtype=float)
    totals = np.array([r[0] for r in rows], dtype=float)
    print(f"\nlevels={len(rows)}")
    print(
        f"nodes:  p50={np.percentile(nodes,50):.0f} "
        f"p95={np.percentile(nodes,95):.0f} max={nodes.max():.0f}"
    )
    print(
        f"time:   p50={np.percentile(totals,50):.2f}s "
        f"p95={np.percentile(totals,95):.2f}s max={totals.max():.2f}s  "
        f"sum={totals.sum():.0f}s"
    )
    # The question that decides the fix: is cost driven by node count?
    if len(rows) > 3:
        print(f"corr(nodes, time) = {np.corrcoef(nodes, totals)[0,1]:.3f}")


if __name__ == "__main__":
    main()
