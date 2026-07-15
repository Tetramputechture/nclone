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
from .generators import GeneratorBrowser


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


def load_map_file_into_env(env, path: str):
    """Load a map file into the env, reset, and refresh the observation."""
    env.nplay_headless.load_map(path)
    return load_after_file(env, path)


def load_after_file(env, path: str):
    map_name = Path(path).stem
    manual_reset(env, map_name=map_name)
    return refresh_observation(env)


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
    """Base class: a navigable source of levels for the env."""

    title = "map"

    def load_current(self, env) -> None:  # pragma: no cover - interface
        raise NotImplementedError

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

    def load_current(self, env) -> None:
        self._safe_reset(env)

    def regenerate(self, env) -> None:
        self._safe_reset(env)

    def on_episode_end(self, env, won: bool, auto_advance: bool) -> bool:
        self._safe_reset(env)
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

    def load_current(self, env) -> None:
        env.reset()

    def regenerate(self, env) -> None:
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

    def load_current(self, env) -> None:
        self.browser.generate()
        if self.browser.current_map is not None:
            load_map_data_into_env(env, self.browser.current_map.map_data())

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

    def load_current(self, env) -> None:
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

    def load_current(self, env) -> None:
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
