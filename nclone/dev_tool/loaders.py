"""Map sources for the dev tool + shared reset / hot-reload helpers.

A :class:`MapSource` is a uniform interface over the different ways the dev tool
puts a level into the env (a generator, the official-maps folder, a mechanic
benchmark dataset, a single file with hot reload, or the env's default map). The
app drives one source via ``advance``/``retreat``/``regenerate`` and shows
``status_lines`` in the panel.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from ..evaluation.mechanic_benchmark_loader import MechanicBenchmarkLoader
from ..map_augmentation.transforms import mirror_map_data
from .generators import BucketBrowser, GeneratorBrowser

# Suffix appended to (and stripped from) the level name of a mirrored map, so
# name-keyed caches never serve the un-mirrored level's data for it.
MIRROR_SUFFIX = "_mirror"


# ---------------------------------------------------------------------------
# env reset helpers (ported from the old test_environment.py)
# ---------------------------------------------------------------------------


def manual_reset(env, map_name: Optional[str] = None) -> None:
    """Reset the env after externally loading a new map.

    Delegates to the env's own ``reset(options={"skip_map_load": True})`` so all
    caches are cleared and rebuilt against the just-loaded map data.
    """
    options = {"skip_map_load": True}
    if map_name is not None:
        options["map_name"] = map_name
    env.reset(options=options)


def refresh_observation(env):
    """Recompute and process the env observation after a manual map swap."""
    initial_obs = env._get_observation()
    return env._process_observation(initial_obs)


def load_map_data_into_env(env, map_data, map_name: Optional[str] = None):
    """Load raw ``map_data`` into the env, reset, and refresh the observation."""
    env.nplay_headless.load_map_from_map_data(list(map_data))
    manual_reset(env, map_name=map_name)
    return refresh_observation(env)


def guarded_load(env, browser, get_map_data, what: str, tag: str) -> bool:
    """Generate + load one level, containing and reporting any failure.

    Returns True when a level actually reached the env.

    WHY THIS EXISTS. Browsing generators is a search for the ones that are
    BROKEN, so the tool has to outlive them. Two independent things can raise
    and both used to close the window:

    * the generator itself -- `hazard_overlay` and `mine_overlay` raise
      `RuntimeError` when they cannot certify a level inside their retry budget,
      which is a normal outcome for a seed rather than a defect;
    * the LOAD -- `load_map_data_into_env` resets the env, which rebuilds the
      graph, reachability and observation caches against a level nothing has
      ever seen. This is the more dangerous half and the easier one to forget.

    On failure the previously loaded level stays in the env, so the tool remains
    usable, and the reason is printed with a full traceback (to stdout, so the
    message and the traceback cannot be separated by a redirect).
    """
    import sys
    import traceback

    try:
        browser.generate()
    except Exception as exc:  # noqa: BLE001 - containment is the point
        print(f"\n[{tag}] {what}: GENERATE failed: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        print(f"[{tag}] previous level kept; press next to continue.\n")
        return False

    map_data = get_map_data()
    if map_data is None:
        return False
    try:
        load_map_data_into_env(env, map_data)
    except Exception as exc:  # noqa: BLE001
        # Record it on the browser too, so the panel shows it next to the combo
        # that produced it rather than only in the terminal.
        browser.status = getattr(browser, "ERROR", "ERROR")
        browser.last_error = f"load: {type(exc).__name__}: {exc}"
        print(f"\n[{tag}] {what}: LOAD failed: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        print(f"[{tag}] the generated map is bad, not the tool; continuing.\n")
        return False
    return True


def load_map_file_into_env(env, path: str):
    """Load a map file into the env, reset, and refresh the observation."""
    env.nplay_headless.load_map(path)
    return load_after_file(env, path)


def load_after_file(env, path: str):
    map_name = Path(path).stem
    manual_reset(env, map_name=map_name)
    return refresh_observation(env)


def mirror_current_map(env) -> Optional[List[int]]:
    """Reload the level currently in ``env``, horizontally mirrored.

    Works off the raw ``map_data`` the sim already holds, so it applies to every
    map source (generator, official file, benchmark payload, ...) uniformly. The
    transform is involutive, so calling this twice restores the original level
    byte-for-byte -- which is exactly how the panel's mirror toggle works.

    Returns:
        The mirrored ``map_data`` that was loaded, or None if the env had no map.
    """
    map_data = getattr(env.nplay_headless, "current_map_data", None)
    if map_data is None:
        return None
    name = getattr(getattr(env, "map_loader", None), "current_map_name", None)
    if name is not None:
        name = str(name)
        name = (
            name[: -len(MIRROR_SUFFIX)]
            if name.endswith(MIRROR_SUFFIX)
            else name + MIRROR_SUFFIX
        )
    mirrored = mirror_map_data(list(map_data))
    load_map_data_into_env(env, mirrored, map_name=name)
    return mirrored


def map_file_signature(path: str):
    """Lightweight (mtime_ns, size) change signature, or None if unreadable."""
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# map sources
# ---------------------------------------------------------------------------


class MapSource:
    """Base class: a navigable source of levels for the env.

    Subclasses implement :meth:`_load_current`; :meth:`load_current` is the
    template method every navigation path goes through, so the horizontal-mirror
    augmentation is re-applied to each newly loaded level while it is enabled.
    """

    title = "map"
    mirrored = False  # horizontal mirror re-applied after every load
    _mirrored_data: Optional[List[int]] = None  # last map_data we mirrored in

    def load_current(self, env) -> None:
        self._load_current(env)
        self._apply_mirror(env)

    def _load_current(self, env) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _apply_mirror(self, env) -> None:
        """Re-mirror a freshly loaded level (no-op while mirroring is off).

        Idempotent: a plain ``env.reset()`` keeps the already-loaded map data, so
        mirroring it again would silently flip the level back. Comparing against
        the data we last loaded catches that without relying on the level name
        (which some sources do not set).
        """
        if not self.mirrored:
            return
        current = getattr(env.nplay_headless, "current_map_data", None)
        if current is not None and self._mirrored_data == list(current):
            return  # the mirrored level is still loaded
        self._mirrored_data = mirror_current_map(env)

    def toggle_mirror(self, env) -> bool:
        """Flip the level left-to-right in place (panel 'Mirror' / key O).

        Mirrors the loaded level directly rather than reloading the source, so
        the exact same level is kept (a curriculum / generator reload could roll
        a different one) and toggling back is byte-exact.
        """
        mirrored = mirror_current_map(env)
        if mirrored is None:
            return False
        self.mirrored = not self.mirrored
        self._mirrored_data = mirrored if self.mirrored else None
        return True

    def advance(self, env) -> None:
        """Move to the next selection and load it (panel 'Next' / key G)."""

    def retreat(self, env) -> None:
        """Move to the previous selection and load it (panel 'Prev')."""

    def regenerate(self, env) -> None:
        """Produce a fresh variant of the current selection (panel 'Regen')."""
        self.load_current(env)

    def status_lines(self) -> List[str]:
        return [self.title]

    def on_episode_end(self, env, won: bool, auto_advance: bool) -> bool:
        """Handle end-of-episode. Return True if a new map was loaded here."""
        return False


class DefaultMapSource(MapSource):
    """The env's own default / curriculum map, with a safe fallback.

    The env's curriculum map loader can currently pick a category the generator
    registry does not expose (a pre-existing mismatch), which raises on reset. To
    keep the dev tool usable we fall back to the first official map when that
    happens, so it self-heals once the curriculum is fixed.
    """

    title = "default map"

    def __init__(self):
        self._fallback: Optional[OfficialMapSource] = None

    def _safe_reset(self, env) -> None:
        try:
            env.reset()
            self._fallback = None
        except Exception as exc:
            print(f"Default curriculum map load failed ({exc}); using official map.")
            if self._fallback is None:
                self._fallback = OfficialMapSource()
            self._fallback.load_current(env)

    def _load_current(self, env) -> None:
        self._safe_reset(env)

    def on_episode_end(self, env, won: bool, auto_advance: bool) -> bool:
        self.load_current(env)
        return True

    def status_lines(self) -> List[str]:
        if self._fallback is not None:
            return ["source: default (official fallback)", self._fallback.files[self._fallback.index]]
        return ["source: default map", "R resets"]


class FileMapSource(MapSource):
    """A single ``--map`` file, with optional hot reload on disk changes."""

    title = "file map"

    def __init__(self, path: str, hot_reload: bool = True, poll_frames: int = 15):
        self.path = path
        self.hot_reload = hot_reload
        self.poll_frames = max(1, int(poll_frames))
        self._sig = map_file_signature(path) if hot_reload else None

    def _load_current(self, env) -> None:
        env.reset()

    def poll(self, env, frame_counter: int) -> bool:
        """Reload + reset if the watched file changed. Returns True on reload."""
        if not self.hot_reload or frame_counter % self.poll_frames != 0:
            return False
        sig = map_file_signature(self.path)
        if sig is None or sig == self._sig:
            return False
        try:
            load_map_file_into_env(env, self.path)
            self._apply_mirror(env)
            self._sig = sig
            print(f"[HOT-RELOAD] Map reloaded from: {self.path}")
            return True
        except Exception as exc:  # partial write; retry next poll
            print(f"[HOT-RELOAD] Failed (will retry): {exc}")
            return False

    def status_lines(self) -> List[str]:
        return [f"source: {Path(self.path).name}", "hot-reload on" if self.hot_reload else "hot-reload off"]


class GeneratorSource(MapSource):
    """Browse every curriculum generator and show solvability (interactive)."""

    title = "generators"

    def __init__(self, browser: GeneratorBrowser):
        self.browser = browser

    def _load_current(self, env) -> None:
        # The GENERATE and the LOAD are both guarded, and the load is the more
        # dangerous half: `load_map_data_into_env` resets the env, which rebuilds
        # the graph, reachability and observation caches against a level nothing
        # has ever seen before. Guarding only the generator would leave the
        # window closing on exactly the maps most worth looking at.
        guarded_load(
            env, self.browser,
            lambda: self.browser.current_map.map_data()
            if self.browser.current_map is not None else None,
            f"{self.browser.gen_type}:{self.browser.preset}",
            "generators",
        )

    def advance(self, env) -> None:
        self.browser.next()
        self.load_current(env)

    def retreat(self, env) -> None:
        self.browser.prev()
        self.load_current(env)

    def regenerate(self, env) -> None:
        self.browser.bump_seed()
        self.load_current(env)

    def on_episode_end(self, env, won: bool, auto_advance: bool) -> bool:
        # Always roll a fresh map so the tester keeps cycling.
        self.regenerate(env)
        return True

    def status_lines(self) -> List[str]:
        return ["source: generators"] + self.browser.info_lines()


class BucketSource(MapSource):
    """Browse ``BUCKET_CATALOG`` cells, generating each level on the spot.

    Distinct from :class:`BenchmarkSource`, which browses a dataset that has
    already been built to disk. This generates live, so a bucket can be
    validated the moment its catalog entry exists -- which for the hazard
    buckets is the difference between a few seconds and a multi-hour dataset
    build (~13 s per level, and the planner certificate dominates).
    """

    title = "buckets"

    def __init__(self, browser: BucketBrowser):
        self.browser = browser

    def _load_current(self, env) -> None:
        guarded_load(
            env, self.browser,
            lambda: self.browser.current_map.map_data()
            if self.browser.current_map is not None else None,
            f"{self.browser.bucket}/T{self.browser.tier} "
            f"({self.browser.entry[0]}:{self.browser.entry[1]})",
            "buckets",
        )

    def advance(self, env) -> None:
        self.browser.next()
        self.load_current(env)

    def retreat(self, env) -> None:
        self.browser.prev()
        self.load_current(env)

    def regenerate(self, env) -> None:
        self.browser.bump_seed()
        self.load_current(env)

    def next_bucket(self, env) -> None:
        self.browser.next_bucket()
        self.load_current(env)

    def prev_bucket(self, env) -> None:
        self.browser.prev_bucket()
        self.load_current(env)

    def next_tier(self, env) -> None:
        self.browser.next_tier()
        self.load_current(env)

    def prev_tier(self, env) -> None:
        self.browser.prev_tier()
        self.load_current(env)

    def on_episode_end(self, env, won: bool, auto_advance: bool) -> bool:
        self.regenerate(env)
        return True

    def status_lines(self) -> List[str]:
        return ["source: buckets"] + self.browser.info_lines()


class OfficialMapSource(MapSource):
    """Cycle the bundled official SI maps."""

    title = "official maps"

    def __init__(self, maps_dir: Optional[str] = None):
        if maps_dir is None:
            maps_dir = os.path.join(os.path.dirname(__file__), "..", "maps", "official", "SI")
        self.maps_dir = os.path.abspath(maps_dir)
        if not os.path.isdir(self.maps_dir):
            raise FileNotFoundError(f"Official maps directory not found: {self.maps_dir}")
        self.files = sorted(
            f for f in os.listdir(self.maps_dir)
            if os.path.isfile(os.path.join(self.maps_dir, f))
        )
        if not self.files:
            raise ValueError(f"No map files found in {self.maps_dir}")
        self.index = 0

    def _path(self) -> str:
        return os.path.join(self.maps_dir, self.files[self.index])

    def _load_current(self, env) -> None:
        load_map_file_into_env(env, self._path())

    def advance(self, env) -> None:
        self.index = (self.index + 1) % len(self.files)
        self.load_current(env)

    def retreat(self, env) -> None:
        self.index = (self.index - 1) % len(self.files)
        self.load_current(env)

    def on_episode_end(self, env, won: bool, auto_advance: bool) -> bool:
        self.load_current(env)  # reload same map
        return True

    def status_lines(self) -> List[str]:
        return [
            "source: official maps",
            f"{self.index + 1}/{len(self.files)}",
            self.files[self.index],
        ]


class BenchmarkSource(MapSource):
    """Sample levels from a mechanic-primary benchmark dataset."""

    title = "benchmark"

    def __init__(self, root: str, splits=("train",), start_index: int = 0):
        self.loader = MechanicBenchmarkLoader(root, splits=splits)
        if len(self.loader) == 0:
            raise ValueError(f"No levels found under {root!r}")
        self.index = max(0, min(start_index, len(self.loader) - 1))
        self.payload = None

    def _map_name(self) -> str:
        lid = self.payload.get("level_id", self.index) if self.payload else self.index
        return f"benchmark_{lid}"

    def _load_current(self, env) -> None:
        self.payload = self.loader.load_payload(self.index)
        load_map_data_into_env(env, self.payload["map_data"], map_name=self._map_name())

    def advance(self, env) -> None:
        self.index = (self.index + 1) % len(self.loader)
        self.load_current(env)

    def retreat(self, env) -> None:
        self.index = (self.index - 1) % len(self.loader)
        self.load_current(env)

    def next_bucket(self, env) -> None:
        self.index = self.loader.next_category_index(self.index)
        self.load_current(env)

    def prev_bucket(self, env) -> None:
        self.index = self.loader.prev_category_index(self.index)
        self.load_current(env)

    def on_episode_end(self, env, won: bool, auto_advance: bool) -> bool:
        if won or auto_advance:
            self.advance(env)
            return True
        return False

    def hud_lines(self, auto_advance: bool) -> List[str]:
        if self.payload is None:
            return []
        lines = MechanicBenchmarkLoader.overlay_lines(self.payload, self.index, len(self.loader))
        lines.append("N next  P prev  [ ] bucket  R reset")
        return lines

    def status_lines(self) -> List[str]:
        if self.payload is None:
            return ["source: benchmark"]
        return [
            "source: benchmark",
            f"{self.index + 1}/{len(self.loader)}",
            f"{self.payload.get('bucket')} T{self.payload.get('tier')}",
            f"{self.payload.get('generator')}:{self.payload.get('preset')}",
        ]
