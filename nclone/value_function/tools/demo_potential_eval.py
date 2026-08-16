"""Score a PBRS potential against REAL winning play, not against the lattice.

``calibrate_physics_cost`` grades a field by rank-correlating it with the
surface lattice's ``d*``. That oracle has a selection problem: it can only score
levels it can itself solve, which are the surface-y ones where the geometric
field was already fine. On ``di-required`` -- the level where the two cost
models disagree most -- the lattice refuses to build at all ("switch unreachable:
no surface contact node"), so the measurement is missing exactly where the claim
is strongest.

A verified demo is an oracle that works everywhere a demo exists. Re-simulate
the tape, and at every frame the TRUE cost-to-go is known exactly: the number of
frames the expert actually needed from there. That is the quantity a potential
is supposed to rank.

Four numbers per field, because "is it a better potential" is not one question:

``rho``       Spearman(cost, true frames remaining). Global ordering -- does the
              field know which states are further from the goal?
``progress``  Fraction of demo frames where the cost strictly DECREASES. PBRS
              pays out on the per-step delta, so a field can rank well globally
              and still hand the agent a negative signal on half of an expert's
              own moves.
``|F| std``   Standard deviation of the shaping term ``w*(gamma*Phi' - Phi)``
              along the demo, with Phi normalised the way the runtime does
              (``-clamp(cost/D0, 0, 1)``, ``D0`` = spawn cost). This is the
              variance the critic has to absorb; a field with spatial cliffs
              scores badly here even when ``rho`` looks fine.
``cover``     Fraction of demo frames where the cost is finite at all. A dead
              field scores vacuously well on the other three.

Usage::

    python -m nclone.value_function.tools.demo_potential_eval \\
        datasets/precision_pack_v12 --limit 100

    # A/B the arc edges rather than the cost model
    python -m nclone.value_function.tools.demo_potential_eval <dir> --compare arcs
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation with tie averaging (avoids a scipy dependency)."""
    if a.size < 3:
        return float("nan")

    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        ranks = np.empty(v.size, np.float64)
        ranks[order] = np.arange(v.size, dtype=np.float64)
        sv = v[order]
        i = 0
        while i < sv.size:
            j = i + 1
            while j < sv.size and sv[j] == sv[i]:
                j += 1
            if j - i > 1:
                ranks[order[i:j]] = ranks[order[i:j]].mean()
            i = j
        return ranks

    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _load_replay(path: str) -> Optional[Tuple[List[int], List[Tuple[int, int]]]]:
    """``(map_data, [(horizontal, jump), ...])`` from a compact V1 ``.replay``."""
    try:
        data = open(path, "rb").read()
        _ver, mlen, ilen = struct.unpack("<III", data[0:12])
        map_data = list(data[13 : 13 + mlen])
        raw = data[13 + mlen : 13 + mlen + ilen]
        if not map_data or not raw:
            return None
        inputs = [(1 if b & 2 else (-1 if b & 4 else 0), b & 1) for b in raw]
        return map_data, inputs
    except Exception:
        return None


def _trace(map_data: List[int], inputs: Sequence[Tuple[int, int]]):
    """Re-simulate a tape; position, phase and airborne flags through the win.

    Returns None unless the tape actually wins on the CURRENT simulator. The
    replay file's own success flag is not trusted — a stale or mis-encoded tape
    carries success=True while the re-simulation dies, which would silently
    admit a trajectory whose "true frames remaining" is a fiction.
    """
    from ...nplay_headless import NPlayHeadless

    nph = NPlayHeadless(render_mode=None, enable_animation=False,
                        enable_logging=False, enable_debug_overlay=False)
    try:
        nph.load_map_from_map_data(map_data)
        nph.reset()
        xs, sw, airborne, velocities, jump_remaining = [], [], [], [], []
        for hor, jump in inputs:
            x, y = nph.ninja_position()
            xs.append((float(x), float(y)))
            sw.append(bool(nph.exit_switch_activated()))
            ninja = nph.sim.ninja
            airborne.append(bool(getattr(ninja, "airborn", False)))
            velocities.append(tuple(float(v) for v in nph.ninja_velocity()))
            if (
                getattr(ninja, "state", 4) == 3
                and getattr(ninja, "jump_input_old", 0) != 0
            ):
                from ...constants.physics_constants import MAX_JUMP_DURATION
                jf = max(
                    0, MAX_JUMP_DURATION
                    - int(getattr(ninja, "jump_duration", 0))
                )
            else:
                jf = 0
            jump_remaining.append(jf)
            nph.tick(hor, jump)
            if nph.ninja_has_won():
                x, y = nph.ninja_position()
                xs.append((float(x), float(y)))
                sw.append(bool(nph.exit_switch_activated()))
                ninja = nph.sim.ninja
                airborne.append(bool(getattr(ninja, "airborn", False)))
                velocities.append(tuple(float(v) for v in nph.ninja_velocity()))
                jump_remaining.append(0)
                return (
                    np.asarray(xs),
                    np.asarray(sw),
                    np.asarray(airborne),
                    np.asarray(velocities),
                    np.asarray(jump_remaining),
                )
        return None  # never won: no ground-truth time-to-go
    finally:
        try:
            nph.exit()
        except Exception:
            pass


def _score(src, pos: np.ndarray, sw: np.ndarray, airborne: np.ndarray,
           velocities: np.ndarray, jump_remaining: np.ndarray,
           gamma: float, weight: float, projector=None
           ) -> Optional[Dict[str, float]]:
    """The four metrics for one field on one demo trajectory."""
    costs = []
    for i, ((x, y), phase) in enumerate(zip(pos, sw)):
        direct = src.cost_to_completion(float(x), float(y), bool(phase))
        if projector is None or not airborne[i]:
            costs.append(direct)
            continue
        best = float("inf")
        vx, vy = velocities[i]
        for u in (-1, 0, 1):
            cls_, lx, ly, _nx, _ny, tau, imp, threshold, _cw, *_ = (
                projector.project(
                    float(x), float(y), float(vx), float(vy), int(u),
                    int(jump_remaining[i]), 180,
                )
            )
            if cls_ in (3, 4) or (threshold > 0.0 and imp > threshold):
                continue
            landing = (
                src.cost_to_go(float(lx), float(ly), True)
                if phase
                else src.cost_to_completion(float(lx), float(ly), False)
            )
            if np.isfinite(landing):
                best = min(best, max(float(tau), 0.0) + float(landing))
        costs.append(best if np.isfinite(best) else direct)
    cost = np.asarray(costs, dtype=np.float64)
    finite = np.isfinite(cost)
    cover = float(finite.mean())
    if finite.sum() < 3:
        return None

    # True remaining frames: the expert needed exactly this many from here.
    true_left = np.arange(len(cost) - 1, -1, -1, dtype=np.float64)

    rho = _spearman(cost[finite], true_left[finite])

    # Per-step: does the field agree that an expert move made progress?
    step_ok = finite[:-1] & finite[1:]
    progress = (float((cost[1:][step_ok] < cost[:-1][step_ok]).mean())
                if step_ok.any() else float("nan"))
    improves = cost[1:] < cost[:-1]
    # Boundary transitions belong to the airborne metric: takeoff/landing is
    # exactly where a projected potential can differ from the static field.
    air_steps = step_ok & (airborne[:-1] | airborne[1:])
    ground_steps = step_ok & ~(airborne[:-1] | airborne[1:])
    progress_air = (
        float(improves[air_steps].mean()) if air_steps.any() else float("nan")
    )
    progress_ground = (
        float(improves[ground_steps].mean())
        if ground_steps.any() else float("nan")
    )

    # Runtime normalisation: Phi = -clamp(cost / D0, 0, 1), D0 = spawn cost.
    d0 = cost[0] if np.isfinite(cost[0]) and cost[0] > 0 else np.nanmax(cost[finite])
    if not np.isfinite(d0) or d0 <= 0:
        return None
    phi = -np.clip(cost / d0, 0.0, 1.0)
    f = weight * (gamma * phi[1:] - phi[:-1])
    f_std = float(np.std(f[step_ok])) if step_ok.any() else float("nan")

    return {
        "rho": rho,
        "progress": progress,
        "progress_air": progress_air,
        "progress_ground": progress_ground,
        "f_std": f_std,
        "cover": cover,
        "frames": float(len(cost)),
    }


def _variants(compare: str):
    """``[(label, kwargs, module_overrides), ...]`` for the requested A/B."""
    if compare == "arcs":
        return [("phys-noarc", {"physics_cost": True}, {"_ARC_EDGES": False}),
                ("phys+arc", {"physics_cost": True}, {"_ARC_EDGES": True})]
    if compare == "air-k":
        # _AIR_UNSUPPORTED_K scales horizontal cost by how far above the floor a
        # cell is. A demo spends its jumps exactly there, so if the penalty is
        # mis-set the potential fights the expert mid-arc. This is the one
        # constant in the model that physics does not hand us.
        return [(f"air_k={k}", {"physics_cost": True}, {"_AIR_UNSUPPORTED_K": k})
                for k in (0.0, 0.25, 0.5, 1.0, 2.0)] + [
            ("geometric", {"physics_cost": False}, {})]
    if compare == "global":
        return [
            ("seedless", {
                "physics_cost": True, "mine_hard": True, "one_way": False,
            }, {}),
            ("global", {"_source": "global"}, {}),
        ]
    if compare == "doors":
        # ONE switch: locked-door phase legs, with one-ways ON in both arms.
        # Run this on levels that HAVE doors; elsewhere the arms are identical
        # by construction and the comparison is vacuous.
        return [
            ("global-nodoors", {"_source": "global", "_door_aware": False}, {}),
            ("global-doors", {"_source": "global", "_door_aware": True}, {}),
        ]
    if compare == "oneway":
        # ONE switch: directional one-way edges, with doors ON in both arms.
        return [
            ("global-nooneway", {"_source": "global", "_one_way": False}, {}),
            ("global-oneway", {"_source": "global", "_one_way": True}, {}),
        ]
    if compare == "projection":
        return [
            ("global-static", {
                "_source": "global", "_projection": False,
            }, {}),
            ("global-projected", {
                "_source": "global", "_projection": True,
            }, {}),
        ]
    return [("geometric", {"physics_cost": False}, {}),
            ("physics", {"physics_cost": True}, {})]


def _projector(map_data):
    """Native live-state projector with the map's initially deadly mines."""
    try:
        import nclone_native

        if not hasattr(nclone_native, "TrajectoryProjector"):
            return None
        proj = nclone_native.TrajectoryProjector.load(list(map_data))
        sim = nclone_native.Simulator.load(list(map_data))
        xs, ys = [], []
        for e in sim.get_state().entities:
            if (
                e.entity_type in (1, 21)
                and int(getattr(e, "state", 1)) == 0
            ):
                xs.append(float(e.x))
                ys.append(float(e.y))
        proj.set_deadly_mines(xs, ys)
        return proj
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo_dirs", nargs="+", help="directories of .replay files")
    ap.add_argument("--limit", type=int, default=60,
                    help="max demos to score (default 60)")
    ap.add_argument("--gamma", type=float, default=0.999,
                    help="pbrs_gamma used for the shaping-variance term")
    ap.add_argument("--weight", type=float, default=1.0, help="pbrs_weight")
    ap.add_argument("--compare",
                    choices=("model", "arcs", "air-k", "global", "projection",
                             "doors", "oneway"),
                    default="model",
                    help="'model' = geometric vs physics; 'arcs' = physics A/B")
    ap.add_argument("--per-demo", action="store_true", help="print every demo")
    args = ap.parse_args(argv)

    import nclone.value_function.free_space_distance as fsd

    files: List[str] = []
    for d in args.demo_dirs:
        if os.path.isfile(d):
            files.append(d)
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".replay"):
                files.append(os.path.join(d, name))
    # Even stride, not a head slice: these directories are sorted by generator
    # family, so the first N files are N seeds of ONE level archetype and would
    # measure nothing about the corpus.
    if len(files) > args.limit:
        idx = np.linspace(0, len(files) - 1, args.limit).astype(int)
        files = [files[i] for i in sorted(set(idx.tolist()))]
    if not files:
        print("no .replay files found")
        return 1

    variants = _variants(args.compare)
    rows: Dict[str, List[Dict[str, float]]] = {lbl: [] for lbl, _, _ in variants}
    n_won = n_skipped = 0

    for i, path in enumerate(files):
        loaded = _load_replay(path)
        if loaded is None:
            n_skipped += 1
            continue
        map_data, inputs = loaded
        traced = _trace(map_data, inputs)
        if traced is None:
            n_skipped += 1
            continue
        pos, sw, airborne, velocities, jump_remaining = traced
        live_projector = _projector(map_data)
        n_won += 1

        per_demo = {}
        for label, kwargs, overrides in variants:
            saved = {k: getattr(fsd, k) for k in overrides}
            for k, v in overrides.items():
                setattr(fsd, k, v)
            try:
                fsd._FS_MEM_CACHE.clear()
                build_kwargs = dict(kwargs)
                source = build_kwargs.pop("_source", "freespace")
                use_projection = bool(build_kwargs.pop("_projection", True))
                if source == "global":
                    from ..global_potential import GlobalPotentialSource
                    # ``_door_aware`` / ``_one_way`` let a variant toggle ONE of
                    # the global source's switches while holding the others
                    # fixed. Without them the only available comparison is
                    # seedless-vs-global, which moves doors, one-ways AND
                    # mine_split_topology together and cannot attribute a
                    # difference to any of them.
                    src = GlobalPotentialSource(
                        list(map_data), lattice_fallback=False,
                        door_aware=bool(build_kwargs.pop("_door_aware", True)),
                        one_way=bool(build_kwargs.pop("_one_way", True)),
                    )
                else:
                    src = fsd.FreeSpaceDistanceSource(
                        list(map_data), **build_kwargs
                    )
                if not src.available:
                    continue
                s = _score(
                    src, pos, sw, airborne, velocities, jump_remaining,
                    args.gamma, args.weight,
                    projector=(
                        live_projector
                        if source == "global" and use_projection else None
                    ),
                )
            except Exception:
                s = None
            finally:
                for k, v in saved.items():
                    setattr(fsd, k, v)
            if s is not None:
                s["demo"] = path
                rows[label].append(s)
                per_demo[label] = s
        if args.per_demo and per_demo:
            tag = os.path.basename(path)[:28]
            cells = "  ".join(
                f"{lbl}: rho={per_demo[lbl]['rho']:+.3f} "
                f"prog={per_demo[lbl]['progress']:.2f}"
                for lbl, _, _ in variants if lbl in per_demo)
            print(f"  {tag:30s} {int(len(pos)):5d}f  {cells}")
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(files)} scanned, {n_won} usable",
                  file=sys.stderr)

    print(f"\ndemos: {len(files)} scanned, {n_won} re-simulated to a WIN, "
          f"{n_skipped} skipped (unparseable or did not win)\n")
    hdr = (f"{'field':14s} {'n':>5s} {'rho':>8s} {'progress':>9s} "
           f"{'air':>8s} {'ground':>8s} {'|F| std':>9s} {'cover':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for label, _, _ in variants:
        r = rows[label]
        if not r:
            print(f"{label:14s} {'0':>5s}   (no usable demos)")
            continue
        print(f"{label:14s} {len(r):5d} "
              f"{np.nanmean([x['rho'] for x in r]):8.3f} "
              f"{np.nanmean([x['progress'] for x in r]):9.3f} "
              f"{np.nanmean([x['progress_air'] for x in r]):8.3f} "
              f"{np.nanmean([x['progress_ground'] for x in r]):8.3f} "
              f"{np.nanmean([x['f_std'] for x in r]):9.5f} "
              f"{np.nanmean([x['cover'] for x in r]):7.3f}")

    # Paired deltas: same demos, so the comparison is per-trajectory, not
    # across two differently-filtered populations.
    if len(variants) == 2:
        a, b = variants[0][0], variants[1][0]
        by_a = {x["demo"]: x for x in rows[a]}
        by_b = {x["demo"]: x for x in rows[b]}
        paired = sorted(set(by_a) & set(by_b))
        n = len(paired)
        if n:
            # higher_better=False for |F| std: that one is variance the critic
            # eats, so a bigger number is a worse potential, not a better one.
            for key, name, higher_better in (("rho", "rho", True),
                                             ("progress", "progress", True),
                                             ("f_std", "|F| std", False)):
                da = np.array([by_a[p][key] for p in paired])
                db = np.array([by_b[p][key] for p in paired])
                d = db - da
                gain = d if higher_better else -d
                wins = int((gain > 1e-9).sum())
                loss = int((gain < -1e-9).sum())
                # Paired bootstrap confidence interval over levels. The fixed
                # seed makes the acceptance report reproducible.
                rng = np.random.default_rng(0)
                idx = rng.integers(0, n, size=(2000, n))
                means = np.nanmean(d[idx], axis=1)
                lo, hi = np.nanpercentile(means, [2.5, 97.5])
                arrow = "higher=better" if higher_better else "LOWER=better"
                print(f"\npaired {name:9s} {b} - {a}: mean {np.nanmean(d):+.5f} "
                      f"95%CI[{lo:+.5f},{hi:+.5f}] ({arrow})  "
                      f"{b} better on {wins}/{n}, worse on {loss}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
