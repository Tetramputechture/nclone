"""Run the lattice planner+executor over the planner-supported official levels
and record route + completion stats.  Sequential + fault-tolerant + incremental
JSON (resumable, monitorable).  Completion = the planned tape wins on the native
sim (for these mine/gold/switch/door levels the native sim models every relevant
entity, so this equals true completion).

Usage:
    .venv/bin/python -m nclone.value_function.tools.benchmark_run \
        VALID_LIST OUT_JSON [N_SAMPLE] [WORKERS]

N_SAMPLE: if given, evenly sample N maps across the list (pipeline test /
representative estimate); omit to run the full list.
"""
import json
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from nclone.value_function.tools.lattice_audit import LatticeAudit


def categorize(r):
    """Failure category for diagnosis."""
    if not r.get("ok"):
        return "error"
    if r.get("won"):
        return "won"
    if r.get("died"):
        return "died"
    if (r.get("n_arcs") or 0) == 0:
        return "no_route"          # lattice couldn't connect spawn->switch->exit
    if r.get("switch_frame") is None:
        return "stall_pre_switch"  # route built, executor stalled before the switch
    return "stall_post_switch"     # hit switch, couldn't finish to the exit


def run_one(path, workers):
    t = time.time()
    try:
        a = LatticeAudit(map_path=path, workers=workers, use_native=True)
        res = a.plan_tape(use_cache=True, allow_cdj=True)
        r = {
            "map": path, "ok": True,
            "won": bool(res.get("won")), "died": bool(res.get("died")),
            "frames": res.get("frames"), "switch_frame": res.get("switch_frame"),
            "min_clear": res.get("min_clear"), "n_arcs": res.get("n_arcs"),
            "n_reached": res.get("n_reached"), "dist_exit_end": res.get("dist_exit_end"),
            "tape_len": len(res.get("tape") or []), "time": time.time() - t,
        }
    except Exception as e:
        r = {"map": path, "ok": False, "error": f"{type(e).__name__}: {e}",
             "time": time.time() - t}
    r["category"] = categorize(r)
    return r


def main(valid_list, out_json, n_sample=None, workers=12):
    with open(valid_list) as f:
        maps = [ln.strip() for ln in f if ln.strip()]
    if n_sample and n_sample < len(maps):
        stride = max(1, len(maps) // n_sample)
        maps = maps[::stride][:n_sample]
    print(f"running planner+executor on {len(maps)} maps (workers={workers})")

    results = []
    won = err = 0
    t0 = time.time()
    for i, p in enumerate(maps):
        r = run_one(p, workers)
        results.append(r)
        if r.get("won"):
            won += 1
        if not r.get("ok"):
            err += 1
        with open(out_json, "w") as f:
            json.dump(results, f)
        n = i + 1
        tag = "WON " if r.get("won") else ("ERR " if not r.get("ok") else "miss")
        print(f"[{n:4d}/{len(maps)}] {tag} t={r.get('time',0):5.1f}s  "
              f"won={won}/{n} ({100.0*won/n:.0f}%)  err={err}  "
              f"{os.path.basename(p)}", flush=True)

    dt = time.time() - t0
    from collections import Counter
    cats = Counter(r["category"] for r in results)
    print(f"\n=== {won}/{len(maps)} completable ({100.0*won/len(maps):.1f}%), "
          f"{err} errors, {dt:.0f}s total, {dt/max(1,len(maps)):.1f}s/map ===")
    print("failure breakdown:")
    for c in ("won", "no_route", "stall_pre_switch", "stall_post_switch",
              "died", "error"):
        if cats.get(c):
            print(f"  {c:18s}: {cats[c]:4d} ({100.0*cats[c]/len(maps):.1f}%)")
    print(f"results -> {out_json}")


if __name__ == "__main__":
    valid_list = sys.argv[1] if len(sys.argv) > 1 else "/tmp/valid_maps.txt"
    out_json = sys.argv[2] if len(sys.argv) > 2 else "/tmp/benchmark_results.json"
    n_sample = int(sys.argv[3]) if len(sys.argv) > 3 else None
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    main(valid_list, out_json, n_sample, workers)
