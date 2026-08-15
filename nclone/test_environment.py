"""N++ environment test harness.

Thin entry point: parse arguments, build the environment, and hand off to the
interactive developer tool (:mod:`nclone.dev_tool`). The dev tool shows a
clickable panel to the right of the game window with every debug overlay and
map-source control, so keyboard shortcuts no longer have to be memorised (the
shortcuts still work and are shown on each button).

Map sources (pick one):
  (default)            the env's default / custom map
  --map FILE           a single map file (hot-reloads on disk changes)
  --test-generators    browse EVERY curriculum generator + live solvability
  --test-buckets       browse BUCKET_CATALOG cells (bucket x tier), generated LIVE
  --test-official-maps cycle the bundled official SI maps
  --benchmark-dataset  sample a mechanic-primary benchmark dataset
  --benchmark-montage  render the WHOLE dataset to per-bucket contact-sheet PNGs

Examples
--------
python -m nclone.test_environment
python -m nclone.test_environment --test-generators
# Validate one bucket's difficulty ladder: [ / ] change bucket, - / = change tier,
# F re-filters live, R re-rolls the seed. The panel names the hazard entities in
# the level, which is what tells a correct DRONE level from a valid-looking one.
python -m nclone.test_environment --test-buckets --bucket-filter DRONE
python -m nclone.test_environment --headless --export-frame /tmp/level.png
python -m nclone.test_environment --benchmark-dataset --test-dataset-path datasets/mechanic_benchmark_v2
# Visualize the ENTIRE curriculum at a glance (one contact sheet per bucket):
python -m nclone.test_environment --benchmark-montage --benchmark-split both \
    --test-dataset-path datasets/mechanic_benchmark_v2 --montage-output /tmp/curriculum
python -m nclone.test_environment --headless --profile-frames 2000
"""

import argparse
import cProfile
import os
import pstats
import sys

import numpy as np
import pygame
from PIL import Image

from nclone.dev_tool import layout, run_dev_tool
from nclone.gym_environment.config import (
    EnvironmentConfig,
    GraphConfig,
    ReachabilityConfig,
    RenderConfig,
)
from nclone.gym_environment.environment_factory import create_visual_testing_env


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test the N++ environment with a dev panel.")

    # Runtime / profiling.
    p.add_argument("--headless", action="store_true", help="Run without a GUI window.")
    p.add_argument("--sdl-driver", type=str, default="auto",
                   help="SDL video driver ('auto', 'x11', 'wayland', ...). 'auto' "
                        "prefers x11 under WSLg, where the wayland backend does "
                        "not present the mouse cursor.")
    p.add_argument("--log-frametimes", action="store_true", help="Print per-frame times.")
    p.add_argument("--profile-frames", type=int, default=None,
                   help="Run N frames then exit (for cProfile timing).")
    p.add_argument("--profile-memory", action="store_true",
                   help="Enable tracemalloc memory profiling.")
    p.add_argument("--memory-snapshot-interval", type=int, default=100,
                   help="Frames between memory snapshots (default: 100).")
    p.add_argument("--export-frame", type=str, default=None,
                   help="Render the first frame to this image file and exit.")
    p.add_argument("--no-dev-panel", dest="dev_panel", action="store_false",
                   help="Disable the right-hand dev panel (window stays game-sized).")
    p.add_argument("--render3d", action="store_true",
                   help="Show a live 3D view of the same sim in a middle strip.")
    p.add_argument("--render3d-width", type=int, default=560,
                   help="Width in px of the 3D view strip (with --render3d).")
    p.add_argument("--render3d-fpv", action="store_true",
                   help="Use the agent-perspective (first-person) camera in the 3D view.")

    # Map source.
    p.add_argument("--map", type=str, default=None,
                   help="Custom map file to load (hot-reloads on disk changes).")
    p.add_argument("--no-hot-reload", action="store_true",
                   help="Disable hot reloading of the --map file.")
    p.add_argument("--hot-reload-poll-frames", type=int, default=15,
                   help="Frames between --map change checks (default: 15).")
    p.add_argument("--test-generators", action="store_true",
                   help="Browse every curriculum generator + live solvability.")
    p.add_argument("--generator-seed-start", type=int, default=10000,
                   help="Starting seed for generator testing (default: 10000).")
    p.add_argument("--generator-filter", type=str, default="",
                   help="With --test-generators, restrict browsing to generators "
                        "whose 'type:preset' contains this (case-insensitive). "
                        "e.g. --generator-filter mine_maze, or mine_maze:heavy.")
    p.add_argument("--generator-certify", action="store_true",
                   help="Run the d* ROUTE CERTIFICATE while browsing overlay "
                        "generators (mine_overlay / hazard_overlay). OFF by "
                        "default: it forks a planner per level with a 12s cap and "
                        "up to 12 base retries, which freezes the window for as "
                        "long as it takes. With it off you see the same LAYOUT, "
                        "solvability is still checked, but no validated route is "
                        "produced. Turn it on to browse exactly what a dataset "
                        "build would emit.")
    p.add_argument("--test-buckets", action="store_true",
                   help="Browse BUCKET_CATALOG cells (bucket x tier), generating "
                        "each level LIVE. Unlike --benchmark-dataset this needs no "
                        "packed dataset, so a bucket can be eyeballed the moment "
                        "its catalog entry exists.")
    p.add_argument("--bucket-filter", type=str, default="",
                   help="With --test-buckets, restrict to buckets whose name "
                        "contains this (case-insensitive), e.g. --bucket-filter DRONE. "
                        "Editable live in the panel via the Filter field (F).")
    p.add_argument("--bucket-tier", type=int, default=None, choices=(1, 2, 3, 4),
                   help="With --test-buckets, restrict to a single tier.")
    p.add_argument("--test-official-maps", action="store_true",
                   help="Cycle the bundled official SI maps.")
    p.add_argument("--benchmark-dataset", action="store_true",
                   help="Sample a mechanic-primary benchmark dataset (manifest.json).")
    p.add_argument("--benchmark-split", choices=("train", "test", "both"), default="train",
                   help="Split(s) for --benchmark-dataset (default: train).")
    p.add_argument("--test-dataset-path", type=str, default="datasets/test",
                   help="Benchmark dataset root (with --benchmark-dataset; defaults to "
                        "datasets/mechanic_benchmark when left at datasets/test).")
    p.add_argument("--start-level", type=int, default=0,
                   help="Starting level index for --benchmark-dataset.")
    p.add_argument("--auto-advance", action="store_true",
                   help="Benchmark: advance on any episode end (not just success).")
    p.add_argument("--no-benchmark-hud", action="store_true",
                   help="Benchmark: skip the on-screen HUD text.")
    p.add_argument("--no-linger-on-death", dest="linger_on_death",
                   action="store_false",
                   help="Reset immediately when the ninja dies. By default a "
                        "death keeps running so the ragdoll plays out, and R "
                        "resets. Ignored under --headless / --auto-advance / "
                        "while recording.")
    p.add_argument("--benchmark-montage", action="store_true",
                   help="Render EVERY level in the dataset to per-bucket contact-sheet "
                        "PNGs (one image per mechanic bucket) and exit. Visualizes the "
                        "whole curriculum at a glance.")
    p.add_argument("--montage-output", type=str, default="nclone/viz/curriculum_montage",
                   help="Output directory for --benchmark-montage PNGs.")
    p.add_argument("--montage-thumb", type=int, default=240,
                   help="Per-level thumbnail width in px for --benchmark-montage.")
    p.add_argument("--montage-cols", type=int, default=10,
                   help="Thumbnails per row in each --benchmark-montage sheet.")

    # Input modes.
    p.add_argument("--discrete-actions", action="store_true",
                   help="One action per keypress (no repeat on hold).")
    p.add_argument("--step-on-input", action="store_true",
                   help="Advance the sim only when an input is pressed (frame-perfect).")

    # Recording.
    p.add_argument("--record", action="store_true",
                   help="Record gameplay replays for behavioral cloning.")
    p.add_argument("--recording-output", type=str, default="datasets/human_replays",
                   help="Output directory for recorded replays.")

    # Demo checkpoint overlay.
    p.add_argument("--demo-replays", type=str, default=None,
                   help="Directory of demo replays for the checkpoint-heatmap overlay.")

    # TAS editor.
    p.add_argument("--tas", action="store_true",
                   help="Launch the frame-by-frame TAS editor instead of the dev tool.")
    p.add_argument("--tas-output", type=str, default="datasets/tas_movies",
                   help="Directory for TAS movies (.replay).")
    p.add_argument("--tas-load", type=str, default=None,
                   help="Optional .replay to preload into the TAS editor.")
    return p


def resolve_args(args) -> None:
    """Validate mutually exclusive modes and derive a couple of fields."""
    sources = [args.test_generators, getattr(args, "test_buckets", False),
               args.test_official_maps, args.benchmark_dataset, bool(args.map)]
    if sum(bool(s) for s in sources) > 1:
        sys.exit("Error: choose only one of --map / --test-generators / "
                 "--test-buckets / --test-official-maps / --benchmark-dataset.")

    # Benchmark dataset root convenience default.
    args.benchmark_root = args.test_dataset_path
    if (args.benchmark_dataset or args.benchmark_montage) and args.test_dataset_path == "datasets/test":
        args.benchmark_root = "datasets/mechanic_benchmark_v2"

    if args.step_on_input and args.headless:
        print("Warning: --step-on-input requires a display; ignoring in headless mode.")
        args.step_on_input = False
    if args.step_on_input:
        args.discrete_actions = True

    if args.tas and args.headless:
        sys.exit("Error: --tas requires a visible display (not --headless).")


def make_env(args, force_human=False):
    """Build the visual-testing env with the right render mode + panel width.

    ``force_human`` builds a renderer-backed human-mode env even under
    ``--headless`` (used by --export-frame, with a dummy SDL video driver so no
    window opens); the policy env otherwise has no renderer in headless mode.
    """
    if force_human:
        render_mode = "human"
        panel_width = 0
    else:
        render_mode = "grayscale_array" if args.headless else "human"
        panel_active = (not args.headless) and args.dev_panel and not args.tas
        panel_width = layout.PANEL_WIDTH if panel_active else 0
        # Reserve a middle strip for the live 3D view (drawn by the dev tool
        # between the game and the dev panel).
        if getattr(args, "render3d", False) and not args.headless and not args.tas:
            panel_width += max(0, int(args.render3d_width))

    render_config = RenderConfig(
        render_mode=render_mode,
        enable_animation=not args.headless,
        enable_debug_overlay=True,
        panel_width=panel_width,
    )
    config = EnvironmentConfig(
        custom_map_path=args.map,
        enable_logging=False,
        render=render_config,
        graph=GraphConfig(debug=False),
        reachability=ReachabilityConfig(debug=False),
        test_dataset_path=args.test_dataset_path,
        frame_skip=1,
        enable_profiling=args.profile_frames is not None,
        # The dev tool always loads its own first map (a generator, official map,
        # benchmark level, file, or the default source's safe load), so skip the
        # construction-time curriculum map selection -- it is both wasted work and
        # currently flaky (the default weights can pick a category the generator
        # registry does not expose).
        defer_initial_map_load=True,
    )
    return create_visual_testing_env(config=config)


def export_frame(env, path: str) -> None:
    """Render the first frame to an image and exit (for AI / regression checks).

    ``env`` is a human-mode env (see ``make_env(force_human=True)``); render()
    returns the pygame surface, which we copy out as RGB.
    """
    try:
        env.reset()
    except Exception as exc:
        # Curriculum default map can be unselectable; fall back to an official map.
        print(f"Default map load failed ({exc}); using official map for export.")
        from nclone.dev_tool.loaders import OfficialMapSource

        OfficialMapSource().load_current(env)
    env.step(0)
    surface = env.render()
    if not isinstance(surface, pygame.Surface):
        sys.exit("Frame export failed: renderer produced no surface.")
    rgb = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).astype(np.uint8)
    Image.fromarray(rgb).save(path)
    print(f"Frame exported to {path} ({rgb.shape[1]}x{rgb.shape[0]})")


def export_montage(args) -> None:
    """Render every level in the benchmark dataset into per-bucket contact sheets.

    Visualizes the ENTIRE curriculum at a glance: one PNG per mechanic bucket,
    thumbnails ordered by tier then level, captioned with tier + level id. Runs
    windowless (dummy SDL) so it works headless. Reuses the human-mode render
    path (``make_env(force_human=True)``) for clean per-level frames.
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from collections import defaultdict
    from PIL import ImageDraw
    from nclone.evaluation.mechanic_benchmark_loader import MechanicBenchmarkLoader
    from nclone.dev_tool.loaders import load_map_data_into_env

    splits = ("train", "test") if args.benchmark_split == "both" else (args.benchmark_split,)
    loader = MechanicBenchmarkLoader(args.benchmark_root, splits=splits)
    n = len(loader)
    if n == 0:
        sys.exit(f"No levels found under {args.benchmark_root!r}")
    print(f"Montage: rendering {n} levels from {args.benchmark_root} (splits={splits}) ...")

    env = make_env(args, force_human=True)
    groups: dict = defaultdict(list)  # bucket -> [(tier, level_id, thumb_img)]
    thumb_w = max(64, args.montage_thumb)
    for i in range(n):
        payload = loader.load_payload(i)
        bucket = loader.bucket_name_for_index(i)
        lid = payload.get("level_id", i)
        load_map_data_into_env(env, payload["map_data"], map_name=f"bm_{lid}")
        env.step(0)
        surface = env.render()
        if not isinstance(surface, pygame.Surface):
            sys.exit("Montage failed: renderer produced no surface.")
        rgb = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).astype(np.uint8)
        img = Image.fromarray(rgb)
        w, h = img.size
        thumb = img.resize((thumb_w, max(1, int(h * thumb_w / w))))
        groups[bucket].append((payload.get("tier"), lid, thumb))
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  rendered {i + 1}/{n}")
    env.close()

    os.makedirs(args.montage_output, exist_ok=True)
    cols = max(1, args.montage_cols)
    pad, label_h, header_h = 6, 16, 26
    bg, fg, cap = (18, 18, 26), (235, 235, 245), (170, 200, 170)
    written = []
    for bucket in sorted(groups):
        items = sorted(groups[bucket], key=lambda x: ((x[0] or 0), (x[1] or 0)))
        tw, thh = items[0][2].size
        cellw, cellh = tw + pad, thh + label_h + pad
        rows = (len(items) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cellw + pad, header_h + rows * cellh + pad), bg)
        draw = ImageDraw.Draw(sheet)
        tiers = sorted({t for t, _, _ in items if t is not None})
        draw.text((pad, 7), f"{bucket}   {len(items)} levels   tiers {tiers}", fill=fg)
        for idx, (tier, lid, thumb) in enumerate(items):
            r, c = divmod(idx, cols)
            x, y = pad + c * cellw, header_h + pad + r * cellh
            sheet.paste(thumb, (x, y))
            draw.text((x + 1, y + thh + 1), f"T{tier} #{lid}", fill=cap)
        out = os.path.join(args.montage_output, f"montage_{bucket}.png")
        sheet.save(out)
        written.append((bucket, len(items), out))
        print(f"  wrote {out}  ({len(items)} levels)")
    print(f"\nMontage complete: {sum(c for _, c, _ in written)} levels across "
          f"{len(written)} buckets -> {args.montage_output}")


def run_tas(env, args) -> None:
    from nclone.tas_tool import run_tas_editor
    from nclone.tas_tool.movie import TASMovie

    preload = None
    if args.tas_load:
        preload = TASMovie.load(args.tas_load).actions
        print(f"Preloaded {len(preload)} frames from {args.tas_load}")
    level_id = getattr(getattr(env, "map_loader", None), "current_map_name", None)
    run_tas_editor(env, initial_actions=preload, output_dir=args.tas_output, level_id=level_id)


def _select_sdl_driver(requested: str) -> None:
    """Choose the SDL video driver before pygame initialises the display.

    Under WSLg, SDL 2.28 picks the `wayland` backend, whose client-side cursor
    surface WSLg's compositor commonly fails to present -- the mouse pointer
    simply does not appear over the window. The dev panel is mouse-driven, so
    that makes the tool unusable. WSLg also exports a working X server, and the
    `x11` backend draws the cursor correctly there.

    Only applied when every condition holds: we are on WSL, the user has not
    already pinned SDL_VIDEODRIVER, a DISPLAY is available, AND the x11 backend
    actually initialises. That last check is not paranoia -- pygame wheels are
    routinely built without x11 (this machine's SDL 2.28.4 has only wayland,
    offscreen and dummy), and pinning a missing driver makes `pygame.display.init`
    raise, taking the whole tool down. When x11 is unavailable the dev tool falls
    back to drawing a software cursor instead; see `dev_tool.app`.

    On any other platform this is a no-op, so a native Linux/Wayland desktop
    keeps the backend it chose.
    """
    if requested and requested != "auto":
        os.environ["SDL_VIDEODRIVER"] = requested
        return
    if os.environ.get("SDL_VIDEODRIVER"):
        return  # explicitly pinned by the caller; respect it
    try:
        is_wsl = "microsoft" in open("/proc/version").read().lower()
    except OSError:
        is_wsl = False
    if not (is_wsl and os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY")):
        return
    os.environ["SDL_VIDEODRIVER"] = "x11"
    try:
        pygame.display.init()
        pygame.display.quit()
    except pygame.error:
        # No x11 in this SDL build. Leave the driver unset so SDL picks its
        # default, and let the software cursor cover the WSLg gap.
        del os.environ["SDL_VIDEODRIVER"]


def main() -> None:
    # Parse BEFORE pygame.init(): the video driver has to be chosen before the
    # display subsystem comes up, and pygame.init() brings it up.
    args = build_parser().parse_args()
    resolve_args(args)
    if not args.headless:
        _select_sdl_driver(getattr(args, "sdl_driver", "auto"))

    pygame.init()
    pygame.display.set_caption("N++ Environment Test")
    if not args.headless:
        pygame.mouse.set_visible(True)
        print(f"SDL video driver: {pygame.display.get_driver()}")

    print(f"Headless: {args.headless}")

    if args.benchmark_montage:
        export_montage(args)
        pygame.quit()
        return

    if args.export_frame:
        # Need a renderer-backed surface; build a windowless human-mode env.
        if args.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        env = make_env(args, force_human=True)
        export_frame(env, args.export_frame)
        pygame.quit()
        env.close()
        return

    env = make_env(args)

    if args.tas:
        env.reset()
        run_tas(env, args)
        pygame.quit()
        env.close()
        return

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run_dev_tool(env, args)
    finally:
        profiler.disable()
        if (
            args.profile_frames is not None
            and getattr(env, "profile_enabled", False)
            and hasattr(env, "_print_profiling_report")
        ):
            env._print_profiling_report()
        pygame.quit()
        env.close()

    if args.profile_frames is not None:
        with open("profiling_stats.txt", "w") as f:
            pstats.Stats(profiler, stream=f).sort_stats("cumulative").print_stats()
        print("Profiling stats saved to profiling_stats.txt")


if __name__ == "__main__":
    main()
