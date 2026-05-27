#!/usr/bin/env python3
"""Export planner-found winning routes as ``.replay`` demos for BC seeding.

The d*-guided forward search finds frame-exact winning routes; npp-rl's BC / DAPG
pipeline trains on ``CompactReplay`` ``.replay`` files in ``--replay-data-dir``
(read by ``bc_dataset`` via ``ReplayExecutor``).  This is the bridge — the
"Distill" in Plan→Distill→Search: search-found routes become demonstrations,
precisely for the execution-bound levels a field-shaped agent can't finish alone.

**Frame-skip alignment (critical).**  ``ReplayExecutor`` replays each input byte
with ``env.step`` at the agent ``frame_skip`` (=4): one input == one agent step ==
``frame_skip`` physics ticks (base-env: "1=demo, 4=agent").  So a demo's
``input_sequence`` is **one action per agent step**, NOT per physics tick.  We
therefore run the search with ``macro = frame_skip`` (each search action is held
exactly ``frame_skip`` ticks = one agent decision), and write one input per
action — faithful on replay AND already at the policy's decision cadence.

Every demo is **validated** (replayed at ``frame_skip`` on the canonical native
sim, which is bit-identical to the env) and only written if it WINS.

Usage::

    .venv/bin/python -m nclone.value_function.tools.export_demos \
        --map nclone/test_maps/mine-very-difficult --out-dir datasets/planner_demos
    .venv/bin/python -m nclone.value_function.tools.export_demos \
        --maps-dir nclone/maps/official/SL --out-dir datasets/planner_demos --processes 6
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import nclone_native as Nnative

from nclone.replay import CompactReplay
from nclone.replay.gameplay_recorder import map_action_to_input
from nclone.replay.input_decoder import InputDecoder


# ---------------------------------------------------------------------------
# Planning: per-agent-step action indices (0-5) that win, via d*-guided search.
# ---------------------------------------------------------------------------
def plan_action_sequence(map_path: str, *, frame_skip: int = 4,
                         w_risk: Optional[float] = None,
                         jump_tax: Optional[float] = None,
                         max_expansions: int = 150000, max_frames: int = 2600,
                         h_weight: float = 2.0
                         ) -> Tuple[Optional[List[int]], Optional[int]]:
    """Return ``(actions, frames)`` — per-step action indices (0-5) for a win —
    or ``(None, None)``.  ``macro == frame_skip`` so each action is one agent
    step; the d* heuristic guides the forward search through the safe corridor."""
    from .dstar_guided_spike import plan_dstar_guided
    res = plan_dstar_guided(
        map_path, macro=frame_skip, h_weight=h_weight,
        max_expansions=max_expansions, max_frames=max_frames, w_risk=w_risk,
        jump_tax=jump_tax)
    if res.get("solved"):
        return [int(a) for a in res["actions"]], int(res.get("frames") or 0)
    return None, None


# ---------------------------------------------------------------------------
# Action sequence <-> CompactReplay
# ---------------------------------------------------------------------------
def build_replay(map_path: str, actions: List[int], episode_id: str,
                 level_id: Optional[str] = None, success: bool = True) -> CompactReplay:
    """One input byte per agent-step action (0-5 -> 0-7 N++ input encoding)."""
    with open(map_path, "rb") as f:
        map_bytes = f.read()
    now = datetime.now()
    return CompactReplay(
        episode_id=episode_id,
        map_data=map_bytes,
        input_sequence=[map_action_to_input(int(a)) for a in actions],
        level_id=level_id,
        start_time=now,
        end_time=now,
        success=success,
    )


def validate_replay(replay: CompactReplay, frame_skip: int = 4) -> Tuple[bool, int]:
    """Replay at ``frame_skip`` (each input held ``frame_skip`` ticks, exactly as
    ``ReplayExecutor`` does) on the canonical native sim; return ``(won, tick)``.
    This is the gate — we never write a demo that doesn't win on ingestion."""
    dec = InputDecoder()
    hs, js = dec.decode_inputs(list(replay.input_sequence))
    sim = Nnative.Simulator.load([int(b) for b in replay.map_data])
    sim.fast_reset()
    t = 0
    for h, j in zip(hs, js):
        for _ in range(frame_skip):
            sim.tick(int(h), int(j))
            t += 1
            if sim.has_won():
                return True, t
            if sim.has_died():
                return False, t
    return False, t


def write_replay(replay: CompactReplay, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_"
                   for c in replay.episode_id)
    path = os.path.join(out_dir, f"{safe}.replay")
    with open(path, "wb") as f:
        f.write(replay.to_binary())
    return path


def export_one(map_path: str, out_dir: str, *, frame_skip: int = 4,
               w_risk: Optional[float] = None, **plan_kw) -> dict:
    """Plan -> encode -> validate@frame_skip -> write one ``.replay``."""
    name = os.path.basename(map_path.rstrip("/"))
    try:
        actions, frames = plan_action_sequence(
            map_path, frame_skip=frame_skip, w_risk=w_risk, **plan_kw)
        if actions is None:
            return {"name": name, "status": "no_win"}
        replay = build_replay(map_path, actions, episode_id=name, level_id=name)
        won, tick = validate_replay(replay, frame_skip=frame_skip)
        if not won:
            return {"name": name, "status": "validate_failed", "win_tick": tick}
        path = write_replay(replay, out_dir)
        return {"name": name, "status": "ok", "frames": frames, "win_tick": tick,
                "n_steps": len(actions), "out": path}
    except Exception as e:  # noqa: BLE001
        return {"name": name, "status": "error", "reason": f"{type(e).__name__}: {e}"}


def _worker(args):
    map_path, out_dir, frame_skip, w_risk, plan_kw = args
    return export_one(map_path, out_dir, frame_skip=frame_skip, w_risk=w_risk, **plan_kw)


# ---------------------------------------------------------------------------
def _gather_maps(args) -> List[str]:
    paths: List[str] = []
    if args.map:
        paths.append(args.map)
    if args.maps_dir:
        paths.extend(p for p in sorted(glob.glob(os.path.join(args.maps_dir, "*")))
                     if os.path.isfile(p))
    if args.list:
        with open(args.list) as f:
            paths.extend(line.strip() for line in f if line.strip())
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map")
    p.add_argument("--maps-dir")
    p.add_argument("--list", help="text file of map paths, one per line")
    p.add_argument("--out-dir", required=True,
                   help="write .replay demos here; point npp-rl --replay-data-dir at it")
    p.add_argument("--frame-skip", type=int, default=4,
                   help="agent frame_skip; demos are one action per frame_skip ticks")
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--w-risk", type=float, default=None)
    p.add_argument("--jump-tax", type=float, default=None,
                   help="additive per-jump frame cost (discourages wall-climb "
                        "spam; None => planner default DEFAULT_JUMP_TAX)")
    p.add_argument("--max-expansions", type=int, default=150000)
    p.add_argument("--max-frames", type=int, default=2600)
    p.add_argument("--h-weight", type=float, default=2.0)
    args = p.parse_args()

    maps = _gather_maps(args)
    if not maps:
        p.error("no maps (use --map, --maps-dir, or --list)")
    plan_kw = dict(max_expansions=args.max_expansions, max_frames=args.max_frames,
                   h_weight=args.h_weight, jump_tax=args.jump_tax)
    print(f"exporting demos for {len(maps)} maps -> {args.out_dir} "
          f"(frame_skip={args.frame_skip}, processes={args.processes})")

    results = []
    if args.processes <= 1:
        for i, m in enumerate(maps):
            r = export_one(m, args.out_dir, frame_skip=args.frame_skip,
                           w_risk=args.w_risk, **plan_kw)
            results.append(r)
            _progress(i + 1, len(maps), r)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.processes) as ex:
            futs = [ex.submit(_worker, (m, args.out_dir, args.frame_skip,
                                        args.w_risk, plan_kw)) for m in maps]
            for i, f in enumerate(as_completed(futs)):
                r = f.result()
                results.append(r)
                _progress(i + 1, len(maps), r)

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n=== demo export ===\n"
          f"  maps={len(maps)}  demos_written={len(ok)}  out={args.out_dir}\n"
          f"  no_win={sum(1 for r in results if r['status']=='no_win')}  "
          f"validate_failed={sum(1 for r in results if r['status']=='validate_failed')}  "
          f"errors={sum(1 for r in results if r['status']=='error')}")
    print(f"  -> train with: python scripts/train.py ... --replay-data-dir {args.out_dir}")
    for r in (r for r in results if r["status"] == "error"):
        print(f"    ERROR {r['name']}: {r['reason']}")
    return 0 if ok else 2


def _progress(done, total, r):
    tag = {"ok": "OK", "no_win": "··", "validate_failed": "VF",
           "error": "ER"}.get(r["status"], "?")
    extra = (f"steps={r['n_steps']} win_tick={r['win_tick']}"
             if r["status"] == "ok" else r.get("reason", ""))
    print(f"  [{done}/{total}] {tag} {r['name']}  {extra}")


if __name__ == "__main__":
    sys.exit(main())
