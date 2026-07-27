import glob, pickle, time, sys, collections
from scripts._far_objective_solver import solve_far_objective

# Pick representative levels per tier, trying to span presets.
def pick(tier, n):
    fs = sorted(glob.glob(f'datasets/mechanic_benchmark_v3/train/FAR_OBJECTIVE/{tier}/*.pkl'))
    by_preset = collections.defaultdict(list)
    for f in fs:
        d = pickle.load(open(f, 'rb'))
        by_preset[d.get('preset')].append((f, d))
    out = []
    presets = list(by_preset)
    i = 0
    while len(out) < n and any(by_preset[p] for p in presets):
        p = presets[i % len(presets)]
        if by_preset[p]:
            out.append(by_preset[p].pop(0))
        i += 1
    return out

BUDGET = {
    'T1': dict(n_states=200, a_max_iter=60000, b_max_iter=100000),
    'T2': dict(n_states=300, a_max_iter=120000, b_max_iter=150000),
    'T3': dict(n_states=300, a_max_iter=150000, b_max_iter=200000),
    'T4': dict(n_states=400, a_max_iter=200000, b_max_iter=300000),
}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
tiers = sys.argv[2].split(',') if len(sys.argv) > 2 else ['T1','T2','T3','T4']

results = collections.defaultdict(list)
for tier in tiers:
    for f, d in pick(tier, N):
        md = list(d['map_data'])
        t = time.time()
        r = solve_far_objective(md, seed=0, log=lambda m: None, **BUDGET[tier])
        dt = time.time() - t
        won = bool(r and r['won'])
        frames = r['frames'] if won else None
        results[tier].append((won, frames, dt, d.get('preset')))
        print(f"{tier} {d['level_id']} preset={d.get('preset'):18s} "
              f"won={won} frames={frames} time={dt:.1f}s", flush=True)

print("\n=== WIN RATE BY TIER ===")
for tier in tiers:
    rs = results[tier]
    nw = sum(1 for w, *_ in rs if w)
    fr = [f for w, f, *_ in rs if w]
    tm = [t for _, _, t, _ in rs]
    fr_s = f"len {min(fr)}-{max(fr)} med {sorted(fr)[len(fr)//2]}" if fr else "-"
    print(f"{tier}: {nw}/{len(rs)} won | demo {fr_s} | "
          f"avg solve {sum(tm)/len(tm):.1f}s")
