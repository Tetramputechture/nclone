"""Render completion videos for the official levels the planner+executor wins.

Reads one or more benchmark JSON result files, collects the won maps, and renders
an overlaid mp4 per level via render_lattice_video (which re-plans, validates the
win on the full Python render sim, and writes the video).

Usage:
    .venv/bin/python -m nclone.value_function.tools.render_winners \
        OUT_DIR result1.json [result2.json ...]
"""
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from nclone.value_function.tools.lattice_video import render_lattice_video


def _safe_name(map_path):
    parts = map_path.rstrip("/").split("/")
    base = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else parts[-1]
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in base)


def main(out_dir, json_paths):
    seen, winners = set(), []
    for jp in json_paths:
        for r in json.load(open(jp)):
            if r.get("won") and r["map"] not in seen:
                seen.add(r["map"])
                winners.append(r["map"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(winners)} winners -> {out_dir}")
    ok = 0
    for i, mp in enumerate(winners):
        out = os.path.join(out_dir, _safe_name(mp) + ".mp4")
        try:
            res = render_lattice_video(map_path=mp, output=out, use_cache=True,
                                       overlay=True)
            wrote = bool(res.get("output"))
            ok += wrote
            print(f"[{i+1}/{len(winners)}] {'OK  ' if wrote else 'skip'} "
                  f"{os.path.basename(out)}", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(winners)}] ERR  {os.path.basename(out)}: "
                  f"{type(e).__name__}: {e}", flush=True)
    print(f"\nrendered {ok}/{len(winners)} videos -> {out_dir}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmark_videos"
    json_paths = sys.argv[2:] or ["/tmp/bench_intro_v3.json", "/tmp/bench_sample.json"]
    main(out_dir, json_paths)
