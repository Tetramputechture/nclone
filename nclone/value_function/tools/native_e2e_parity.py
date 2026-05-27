"""End-to-end LatticeAudit parity: ``use_native`` True vs False.

Runs the full precompute (edge enumeration -> backward Dijkstra d* -> route
extraction) both ways and asserts the d* arrays (``d_comb``, ``d_ex``), the
extracted phase-0/phase-1 routes (every ``ArcTarget`` field incl. the trick
tape), and the route ``info`` dict are byte-identical.  This is the d* + winning
tape half of the native-kernel validation gate.

Usage:
    .venv/bin/python -m nclone.value_function.tools.native_e2e_parity \
        nclone/test_maps/simple-walk nclone/test_maps/di-required ...

Exit code 0 iff every map PASSES.  Each map runs precompute twice (legacy +
native), so a large map is ~6 min; prefer small/medium maps when iterating.
"""
import sys
import time

import numpy as np

from .lattice_audit import LatticeAudit


def arc_key(a):
    return (
        int(a.launch_segment_id),
        round(float(a.launch_x), 6), round(float(a.launch_y), 6),
        round(float(a.launch_vx), 6),
        round(float(a.surface_normal_x), 6), round(float(a.surface_normal_y), 6),
        int(a.jump_hold_frames), int(a.h_input_during), int(a.h_input_after),
        round(float(a.landing_x), 6), round(float(a.landing_y), 6),
        round(float(a.landing_vx), 6),
        round(float(a.arc_cost_frames), 6), round(float(a.cumulative_cost), 6),
        tuple(tuple(t) for t in (a.trick_tape or [])),
    )


def d_equal(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False, f"shape {a.shape} != {b.shape}"
    # inf positions must match; finite values must be equal
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return False, f"inf-mask differs ({(fa != fb).sum()} cells)"
    if not np.array_equal(a[fa], b[fb]):
        d = np.abs(a[fa] - b[fb])
        return False, f"finite values differ (max abs {d.max()})"
    return True, "ok"


def run(map_path):
    print(f"=== {map_path} ===")
    res = {}
    for native in (False, True):
        la = LatticeAudit(map_path=map_path, use_native=native)
        t0 = time.time()
        pc = la.precompute(use_cache=False)
        dt = time.time() - t0
        p0, p1, info = la.build_route(pc)
        res[native] = (pc, p0, p1, info, dt)
        print(f"  native={native}: enum={pc['enum_time']:.1f}s total={dt:.1f}s "
              f"edges={len(pc['edges'])} route=({len(p0)},{len(p1)}) "
              f"d_comb_spawn={info.get('spawn_d_combined')}")

    (pcL, p0L, p1L, iL, _), (pcN, p0N, p1N, iN, _) = res[False], res[True]
    ok = True
    for nm, k in (("d_comb", "d_comb"), ("d_ex", "d_ex")):
        eq, msg = d_equal(pcL[k], pcN[k])
        ok = ok and eq
        print(f"  {nm} identical={eq} ({msg})")
    r0 = [arc_key(a) for a in p0L] == [arc_key(a) for a in p0N]
    r1 = [arc_key(a) for a in p1L] == [arc_key(a) for a in p1N]
    iok = (iL == iN)
    ok = ok and r0 and r1 and iok
    print(f"  phase0 route identical={r0}  phase1 route identical={r1}")
    print(f"  route info identical={iok}")
    if not iok:
        print("   legacy info:", iL)
        print("   native info:", iN)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok = True
    for mp in sys.argv[1:]:
        ok = run(mp) and ok
    sys.exit(0 if ok else 1)
