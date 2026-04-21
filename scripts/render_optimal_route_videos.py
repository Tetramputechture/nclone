"""Render MP4 videos of optimal routes for the trivial-map planner.

Generates videos for a handful of maps in each supported topology:

* ``wall_jump_shaft`` — :class:`MapWallJump` seeds, routed by
  :class:`WallJumpShaftPlanner`.
* ``flat_run_corridor`` — :class:`MapHorizontalCorridor` seeds, routed by
  :class:`FlatRunCorridorPlanner`.

Each video is the frame-by-frame replay of the planner's ``action_sequence``
in :class:`nclone.nplay_headless.NPlayHeadless` with animation enabled, so
what you see is exactly what the real simulator does, rendered at 60 FPS.

Usage
-----
::

    python -m scripts.render_optimal_route_videos \\
        --output-dir optimal_route_videos \\
        --per-category 6 --seed-selection spread

``--seed-selection spread`` (default) scans ``[0, --search-limit)``, collects
every seed that classifies and verifies, then picks ``--per-category`` seeds
spread across that list for visual variety. Use ``first`` for the earliest
qualifying seeds only.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


logger = logging.getLogger("render_optimal_route_videos")


def _ensure_repo_on_path() -> None:
    """Make ``nclone`` importable when this script is run via ``python scripts/...``.

    The repo root is one level above ``scripts/``.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

# Deferred imports so the path fix above takes effect first.
from nclone.map_generation.map_horizontal_corridor import MapHorizontalCorridor  # noqa: E402
from nclone.map_generation.map_wall_jump import MapWallJump  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402
from nclone.planning.optimal_route import (  # noqa: E402
    OptimalRoute,
    TOPOLOGY_FLAT_RUN_CORRIDOR,
    TOPOLOGY_WALL_JUMP_SHAFT,
    classify,
    extract_level_geometry,
    plan_optimal_route,
)


def _collect_all_qualifying_seeds(
    generator_cls,
    expected_topology: str,
    search_limit: int,
) -> List[int]:
    """Every seed in ``range(search_limit)`` that classifies and verifies."""
    qualifying: List[int] = []
    for seed in range(search_limit):
        gen = generator_cls()
        gen.generate(seed=seed)
        md = gen.map_data()
        geom = extract_level_geometry(md)
        if geom is None or classify(geom) != expected_topology:
            continue
        route = plan_optimal_route(md, verify=True)
        if route is None or not route.verified_in_sim:
            continue
        qualifying.append(seed)
    return qualifying


def _spread_pick(seeds: List[int], needed: int) -> List[int]:
    """Pick up to ``needed`` seeds spread across sorted ``seeds``."""
    if not seeds or needed <= 0:
        return []
    if len(seeds) <= needed:
        return seeds[:]
    if needed == 1:
        return [seeds[len(seeds) // 2]]
    out: List[int] = []
    for i in range(needed):
        idx = round(i * (len(seeds) - 1) / (needed - 1))
        out.append(seeds[idx])
    return out


def _find_seeds(
    generator_cls,
    expected_topology: str,
    needed: int,
    search_limit: int = 300,
    explicit: Optional[List[int]] = None,
    selection: str = "spread",
) -> List[int]:
    """Return ``needed`` seeds that classify and yield a verified optimal route.

    If ``explicit`` is provided, those seeds are returned in order (no
    topology / verify check — use for known-good lists).

    Otherwise ``_collect_all_qualifying_seeds`` scans ``[0, search_limit)``,
    then:

    * ``selection == "first"`` — first ``needed`` seeds;
    * ``selection == "spread"`` — evenly spaced across the full qualifying
      list for visual variety.
    """
    if explicit is not None:
        return list(explicit)
    qualifying = _collect_all_qualifying_seeds(
        generator_cls, expected_topology, search_limit
    )
    if len(qualifying) < needed:
        logger.warning(
            "Only found %d/%d qualifying seeds for %s in range(%d)",
            len(qualifying),
            needed,
            expected_topology,
            search_limit,
        )
    if selection == "spread":
        return _spread_pick(qualifying, needed)
    return qualifying[:needed]


def _render_route_to_mp4(
    map_data: list,
    route: OptimalRoute,
    output_path: Path,
    fps: int = 60,
    tail_frames: int = 45,
) -> None:
    """Replay ``route.action_sequence`` in a headless sim and dump ``output_path``.

    ``tail_frames`` idle frames are appended after the final action so the
    ninja's "celebrating" animation is visible at the end of the video.
    """
    import imageio.v2 as imageio  # lazy import so --help works without imageio

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    nph = NPlayHeadless(
        render_mode="rgb_array",
        enable_animation=True,
        enable_logging=False,
        enable_debug_overlay=False,
        enable_rendering=True,
        debug=False,
    )
    nph.load_map_from_map_data(list(map_data))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    try:
        initial_frame = nph.render()
        if initial_frame is not None:
            writer.append_data(np.asarray(initial_frame, dtype=np.uint8))

        for hor, jump in route.action_sequence:
            nph.tick(int(hor), 1 if jump else 0)
            frame = nph.render()
            if frame is not None:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
            if nph.ninja_has_won():
                break

        for _ in range(tail_frames):
            nph.tick(0, 0)
            frame = nph.render()
            if frame is not None:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
            if nph.ninja_has_won():
                break
    finally:
        writer.close()


def _render_category(
    category: str,
    seeds: List[int],
    output_dir: Path,
    fps: int,
) -> List[Path]:
    """Render one MP4 per seed in ``seeds`` for ``category``."""
    videos: List[Path] = []
    if category == "wall_jump":
        generator_cls = MapWallJump
    elif category == "flat_run":
        generator_cls = MapHorizontalCorridor
    else:
        raise ValueError(f"Unknown category: {category}")

    for seed in seeds:
        gen = generator_cls()
        gen.generate(seed=seed)
        md = gen.map_data()
        route = plan_optimal_route(md, verify=True)
        if route is None:
            logger.warning("[%s seed=%d] no route returned, skipping", category, seed)
            continue
        if route.strategy == "no_solution":
            logger.warning(
                "[%s seed=%d] planner returned no_solution, skipping", category, seed
            )
            continue
        out_path = output_dir / f"{category}_seed{seed:03d}_{route.strategy}.mp4"
        logger.info(
            "[%s seed=%d] strategy=%s optimal_frames=%d verified=%s verified_frames=%s -> %s",
            category,
            seed,
            route.strategy,
            route.optimal_frames,
            route.verified_in_sim,
            route.verified_frames,
            out_path.name,
        )
        _render_route_to_mp4(md, route, out_path, fps=fps)
        videos.append(out_path)
    return videos


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Render MP4 videos of optimal-route-planner solutions.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optimal_route_videos"),
        help="Directory to write MP4 files to (created if missing).",
    )
    p.add_argument(
        "--per-category",
        type=int,
        default=6,
        help="How many videos to render per category (default: 6).",
    )
    p.add_argument(
        "--seed-selection",
        choices=("first", "spread"),
        default="spread",
        help=(
            "Auto seed pick: 'first' = earliest qualifying seeds in "
            "[0, --search-limit); 'spread' = spread across all qualifying "
            "seeds for variety (default)."
        ),
    )
    p.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Output framerate in FPS (default: 60, matches sim tick rate).",
    )
    p.add_argument(
        "--wj-seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit wall-jump seeds (overrides auto-selection).",
    )
    p.add_argument(
        "--fr-seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit flat-run seeds (overrides auto-selection).",
    )
    p.add_argument(
        "--search-limit",
        type=int,
        default=300,
        help="Seed range to search when auto-selecting (default: 300).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging."
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    wj_seeds = _find_seeds(
        MapWallJump,
        TOPOLOGY_WALL_JUMP_SHAFT,
        needed=args.per_category,
        search_limit=args.search_limit,
        explicit=args.wj_seeds,
        selection=args.seed_selection,
    )
    fr_seeds = _find_seeds(
        MapHorizontalCorridor,
        TOPOLOGY_FLAT_RUN_CORRIDOR,
        needed=args.per_category,
        search_limit=args.search_limit,
        explicit=args.fr_seeds,
        selection=args.seed_selection,
    )

    logger.info("Selected wall-jump seeds: %s", wj_seeds)
    logger.info("Selected flat-run seeds:  %s", fr_seeds)

    videos = []
    videos += _render_category("wall_jump", wj_seeds, args.output_dir, fps=args.fps)
    videos += _render_category("flat_run", fr_seeds, args.output_dir, fps=args.fps)

    if not videos:
        logger.error("No videos rendered.")
        return 1

    logger.info("Rendered %d video(s):", len(videos))
    for v in videos:
        logger.info("  %s", v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
