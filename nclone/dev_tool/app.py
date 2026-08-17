"""DevToolApp: the interactive test-environment main loop.

Owns the pygame event loop, ninja input handling (continuous / discrete /
step-on-input), per-frame overlay application, map-source navigation, optional
replay recording, and the dev-panel draw + single flip. ``test_environment.py``
is now just argument parsing + env construction + a call to :func:`run_dev_tool`.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import pygame

from . import controls, layout
from .controls import ACTION, FIELD, HEADER, STATUS, TOGGLE, Row
from .generators import BucketBrowser, GeneratorBrowser
from .loaders import (
    BenchmarkSource,
    BucketSource,
    DefaultMapSource,
    FileMapSource,
    GeneratorSource,
    MapSource,
    OfficialMapSource,
)
from .ninja_state import ninja_state_lines
from .overlays import OverlayManager
from .panel import DevPanel

HOLD_REPEAT_DELAY_MS = 200  # ms before held keys repeat (enables single-frame taps)


class DevToolApp:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.headless = bool(getattr(args, "headless", False))
        self.use_panel = (not self.headless) and getattr(args, "dev_panel", True)

        self.recorder = self._build_recorder()
        self.demo_system = self._build_demo_system()
        self.overlays = OverlayManager(args=args, demo_system=self.demo_system)
        self.map_source = self._build_map_source()

        self.panel = DevPanel() if self.use_panel else None
        self.focused_field: Optional[str] = None
        self.field_buffer: str = ""
        #: field_name -> (commit callback, accepts-non-digits). Registered by
        #: :meth:`_focus_field` when a field row is activated, so adding a field
        #: does not mean editing the keydown handler.
        self._field_commit: dict = {}
        #: Last contained control failure, shown in the panel. See
        #: :meth:`_safe_activate`.
        self.last_action_error: Optional[str] = None
        #: True while a death is being watched instead of reset. See
        #: :meth:`_should_linger`.
        self._lingering = False
        #: Show the ninja buffer / contact read-out (see :mod:`.ninja_state`).
        #: On by default: it is the only view of the frame-scale state that
        #: decides whether a jump does anything, and it costs a dozen rows.
        self.show_ninja_state = True

        # Optional live 3D view of the same sim, blitted into a reserved strip.
        self.panel3d = None
        if getattr(args, "render3d", False) and not self.headless:
            try:
                from ..render3d.devpanel import Render3DPanel

                surf = pygame.display.get_surface()
                h = surf.get_height() if surf else 600
                self.panel3d = Render3DPanel(
                    int(args.render3d_width), h,
                    fpv=getattr(args, "render3d_fpv", False))
            except Exception as exc:  # noqa: BLE001 - keep the dev tool usable
                print(f"3D panel unavailable: {exc}")

        # step-on-input state
        self.input_sequence: List[int] = []
        self._hold_start_ms: Optional[float] = None
        self._hold_action = 0
        # discrete-action state
        self._keys_just_pressed: set = set()
        self._keys_held: set = set()

        self.observation = None
        self.frame_counter = 0
        self.running = True

        from .memory_profiler import MemoryProfiler

        self.memory_profiler = None
        if getattr(args, "profile_memory", False):
            self.memory_profiler = MemoryProfiler(
                snapshot_interval=getattr(args, "memory_snapshot_interval", 100)
            )
            self.memory_profiler.start()

    # -- construction helpers ---------------------------------------------

    def _build_recorder(self):
        if not getattr(self.args, "record", False):
            return None
        from ..replay.gameplay_recorder import GameplayRecorder

        return GameplayRecorder(output_dir=self.args.recording_output)

    def _build_demo_system(self):
        if not getattr(self.args, "demo_replays", None):
            return None
        try:
            from ..replay.demo_checkpoint_seeder import DemoCheckpointSeeder

            return {
                "seeder": DemoCheckpointSeeder(
                    replay_dir=self.args.demo_replays,
                    max_demos_per_level=10,
                    min_cumulative_reward=0.0,
                ),
                "checkpoint_cache": {},
                "cached_level_name": None,
            }
        except Exception as exc:
            print(f"Demo checkpoint system unavailable: {exc}")
            return None

    def _build_map_source(self) -> MapSource:
        args = self.args
        if getattr(args, "test_generators", False):
            browser = GeneratorBrowser(
                start_seed=getattr(args, "generator_seed_start", 10000),
                name_filter=getattr(args, "generator_filter", "") or "",
                certify_routes=getattr(args, "generator_certify", False),
            )
            src = GeneratorSource(browser)
            src.load_current(self.env)
            return src
        if getattr(args, "test_buckets", False):
            browser = BucketBrowser(
                start_seed=getattr(args, "generator_seed_start", 10000),
                bucket_filter=getattr(args, "bucket_filter", "") or "",
                tier=getattr(args, "bucket_tier", None),
                certify_routes=getattr(args, "generator_certify", False),
            )
            src = BucketSource(browser)
            src.load_current(self.env)
            return src
        if getattr(args, "test_official_maps", False):
            src = OfficialMapSource()
            src.load_current(self.env)
            return src
        if getattr(args, "benchmark_dataset", False):
            splits = ("train", "test") if args.benchmark_split == "both" else (args.benchmark_split,)
            src = BenchmarkSource(args.benchmark_root, splits=splits, start_index=args.start_level)
            src.load_current(self.env)
            return src
        if getattr(args, "map", None):
            src = FileMapSource(
                args.map,
                hot_reload=not getattr(args, "no_hot_reload", False),
                poll_frames=getattr(args, "hot_reload_poll_frames", 15),
            )
            src.load_current(self.env)
            return src
        src = DefaultMapSource()
        src.load_current(self.env)
        return src

    # -- actions wired to controls ----------------------------------------

    def reset_current(self):
        self._lingering = False
        if self.recorder is not None and self.recorder.is_recording:
            won = self.env.nplay_headless.ninja_has_won()
            self.recorder.stop_recording(success=won, save=won)
        self.map_source.regenerate(self.env)

    def source_next(self):
        self.map_source.advance(self.env)

    def source_prev(self):
        self.map_source.retreat(self.env)

    def toggle_mirror(self):
        """Horizontally mirror the loaded level (tests ``mirror_map_data``).

        The mirror is involutive, so pressing this again restores the original
        level exactly; it also stays applied to levels loaded afterwards until
        toggled off.
        """
        if not self.map_source.toggle_mirror(self.env):
            print("Mirror: no map data loaded yet")

    def scan_generators(self):
        if isinstance(self.map_source, GeneratorSource):
            self.map_source.browser.scan_all()

    def scan_buckets(self):
        if isinstance(self.map_source, BucketSource):
            self.map_source.browser.scan_all()

    def _safe_activate(self, callback, label: str = "") -> None:
        """Run a panel control's callback, containing and REPORTING any failure.

        Every control used to dispatch straight into its callback, so a single
        raising generator closed the window -- with the traceback going to a
        terminal the user may not have been watching, which made it look like a
        crash with no cause.

        Generators raise legitimately and often: `map_hazard_overlay` and
        `map_mine_overlay` both raise `RuntimeError` when they cannot certify a
        level within their retry budget, and that is a normal outcome for a
        seed, not a bug. Browsing exists to FIND those, so the tool has to
        survive them: the env keeps the previously loaded level, the reason is
        printed once with a traceback, and the panel shows it until the next
        successful action.
        """
        if callback is None:
            return
        try:
            callback()
            self.last_action_error = None
        except Exception as exc:  # noqa: BLE001 - containment is the point
            import sys
            import traceback

            where = label or getattr(callback, "__name__", "control")
            self.last_action_error = f"{where}: {type(exc).__name__}: {exc}"
            # Traceback to STDOUT, with the message, deliberately: print_exc
            # defaults to stderr, which interleaves unpredictably with the rest
            # of the tool's output and is lost entirely by a `> log` redirect --
            # so the report would arrive split in half or not at all.
            print(f"\n[dev-tool] '{where}' failed and was contained:")
            traceback.print_exc(file=sys.stdout)
            print("[dev-tool] the previous level is still loaded; carry on.\n")

    def _nav(self, method: str):
        """Call a navigation method on the current source's browser and reload."""
        browser = getattr(self.map_source, "browser", None)
        fn = getattr(browser, method, None)
        if fn is None:
            return
        fn()
        self.map_source.load_current(self.env)

    def dump_ascii(self):
        browser = getattr(self.map_source, "browser", None)
        if browser is not None and hasattr(browser, "to_ascii"):
            print("\n" + browser.to_ascii() + "\n")

    # -- editable panel fields --------------------------------------------

    def _focus_field(self, name: str, initial: str, commit, numeric: bool = True):
        self.focused_field = name
        self.field_buffer = str(initial)
        self._field_commit[name] = (commit, numeric)

    def commit_field(self):
        """Apply the focused field's buffer through its registered callback."""
        name = self.focused_field
        entry = self._field_commit.get(name) if name else None
        if entry is not None and self.field_buffer:
            commit, _numeric = entry
            try:
                commit(self.field_buffer)
            except (ValueError, RuntimeError) as exc:
                # A typo'd seed or a filter that matches nothing must not take
                # the tool down; the browser keeps its previous state.
                print(f"[panel] {name}: {exc}")
        self.focused_field = None
        self.field_buffer = ""

    def _focused_is_numeric(self) -> bool:
        entry = self._field_commit.get(self.focused_field or "")
        return True if entry is None else entry[1]

    def focus_seed(self):
        browser = getattr(self.map_source, "browser", None)
        if browser is None:
            return
        self._focus_field(
            "seed", browser.seed,
            lambda text: (browser.set_seed(int(text)),
                          self.map_source.load_current(self.env)),
        )

    def focus_filter(self):
        """Type a generator or bucket filter without restarting the tool.

        The CLI flags can only be set before the window opens, so validating a
        second generator used to mean quitting and relaunching.
        """
        browser = getattr(self.map_source, "browser", None)
        if browser is None or not hasattr(browser, "set_filter"):
            return
        current = getattr(browser, "bucket_filter", None)
        if current is None:
            current = getattr(browser, "name_filter", "")

        def _commit(text: str) -> None:
            if not browser.set_filter(text):
                raise RuntimeError(f"no match for {text!r} — filter unchanged")
            self.map_source.load_current(self.env)

        self._focus_field("filter", current, _commit, numeric=False)

    def start_recording(self):
        if self.recorder is None or self.recorder.is_recording:
            return
        current = list(self.env.nplay_headless.current_map_data)
        from .loaders import manual_reset, refresh_observation

        manual_reset(self.env)
        self.env.nplay_headless.reset()
        self.env.nplay_headless.load_map_from_map_data(current)
        self.observation = refresh_observation(self.env)
        map_data = bytes(max(0, min(255, int(b))) for b in current)
        map_name = getattr(getattr(self.env, "map_loader", None), "current_map_name", "unknown")
        self.recorder.start_recording(map_data, map_name, None)
        self.input_sequence.clear()

    def stop_recording(self):
        if self.recorder is not None and self.recorder.is_recording:
            self.recorder.stop_recording(success=False, save=False)
            print("Recording stopped (not saved)")

    # -- row model ---------------------------------------------------------

    def toggle_ninja_state(self):
        self.show_ninja_state = not self.show_ninja_state

    def _ninja_state_lines(self) -> List[str]:
        """Buffer / contact read-out, or why it is unavailable.

        Contained: this is a debug read-out on the frame path, so a backend that
        does not expose some field must not take the whole tool down.
        """
        try:
            ninja = self.env.nplay_headless.sim.ninja
        except Exception as exc:  # noqa: BLE001 - read-out must not be fatal
            return [f"unavailable: {type(exc).__name__}"]
        try:
            return ninja_state_lines(ninja)
        except Exception as exc:  # noqa: BLE001
            return [f"read failed: {type(exc).__name__}"]

    def build_rows(self) -> List[Row]:
        rows: List[Row] = [Row(HEADER, "Map source")]
        rows.append(Row(ACTION, "Next", "G", pygame.K_g, on_activate=self.source_next))
        rows.append(Row(ACTION, "Prev", "", None, on_activate=self.source_prev))
        rows.append(Row(ACTION, "Reset / regen", "R", pygame.K_r, on_activate=self.reset_current))
        rows.append(Row(TOGGLE, "Mirror (H-flip)", "O", pygame.K_o,
                        get_state=lambda: self.map_source.mirrored,
                        on_activate=self.toggle_mirror))

        if isinstance(self.map_source, GeneratorSource):
            # Type is the coarse axis: ~155 presets across ~39 generators, so
            # stepping preset-by-preset to reach another generator is hopeless.
            rows.append(Row(ACTION, "Prev generator", "[", pygame.K_LEFTBRACKET,
                            on_activate=lambda: self._nav("prev_type")))
            rows.append(Row(ACTION, "Next generator", "]", pygame.K_RIGHTBRACKET,
                            on_activate=lambda: self._nav("next_type")))
            rows.append(Row(FIELD, "Filter", "F", pygame.K_f,
                            get_value=lambda: self.map_source.browser.name_filter or "(all)",
                            on_activate=self.focus_filter, field_name="filter",
                            numeric=False))
            rows.append(Row(FIELD, "Seed", get_value=lambda: str(self.map_source.browser.seed),
                            on_activate=self.focus_seed, field_name="seed"))
            rows.append(Row(ACTION, "Scan all generators", "", None, on_activate=self.scan_generators))
            rows.append(Row(ACTION, "Dump ASCII", "V", pygame.K_v, on_activate=self.dump_ascii))
        if isinstance(self.map_source, BucketSource):
            rows.append(Row(ACTION, "Prev bucket", "[", pygame.K_LEFTBRACKET,
                            on_activate=lambda: self.map_source.prev_bucket(self.env)))
            rows.append(Row(ACTION, "Next bucket", "]", pygame.K_RIGHTBRACKET,
                            on_activate=lambda: self.map_source.next_bucket(self.env)))
            # Tier moves WITHIN the current bucket. Comparing T1 against T4 of
            # one bucket is the check that matters: a T4 that reads easier than
            # T1 is exactly what the tier presets exist to prevent.
            rows.append(Row(ACTION, "Prev tier", "-", pygame.K_MINUS,
                            on_activate=lambda: self.map_source.prev_tier(self.env)))
            rows.append(Row(ACTION, "Next tier", "=", pygame.K_EQUALS,
                            on_activate=lambda: self.map_source.next_tier(self.env)))
            rows.append(Row(FIELD, "Filter", "F", pygame.K_f,
                            get_value=lambda: self.map_source.browser.bucket_filter or "(all)",
                            on_activate=self.focus_filter, field_name="filter",
                            numeric=False))
            rows.append(Row(FIELD, "Seed", get_value=lambda: str(self.map_source.browser.seed),
                            on_activate=self.focus_seed, field_name="seed"))
            rows.append(Row(ACTION, "Scan all buckets", "", None,
                            on_activate=self.scan_buckets))
            rows.append(Row(ACTION, "Dump ASCII", "V", pygame.K_v, on_activate=self.dump_ascii))
        if isinstance(self.map_source, BenchmarkSource):
            rows.append(Row(ACTION, "Prev bucket", "[", pygame.K_LEFTBRACKET,
                            on_activate=lambda: self.map_source.prev_bucket(self.env)))
            rows.append(Row(ACTION, "Next bucket", "]", pygame.K_RIGHTBRACKET,
                            on_activate=lambda: self.map_source.next_bucket(self.env)))

        for line in self.map_source.status_lines():
            rows.append(Row(STATUS, line))
        if self.last_action_error:
            # Shown until the next successful action. Without this the only
            # trace of a contained failure is a terminal the user may not be
            # watching, which reads as "the button did nothing".
            rows.append(Row(STATUS, f"! {self.last_action_error}"[:44]))

        rows.append(Row(HEADER, "Overlays"))
        for name, label, key, key_label in self.overlays.toggle_specs():
            rows.append(Row(
                TOGGLE, label, key_label, key,
                get_state=(lambda n=name: self.overlays.is_on(n)),
                on_activate=(lambda n=name: self.overlays.toggle(n)),
            ))

        rows.append(Row(HEADER, "Ninja state"))
        # "S": J is the jump-cost-to-go overlay, and S is the only letter left.
        rows.append(Row(TOGGLE, "Buffers / contacts", "S", pygame.K_s,
                        get_state=lambda: self.show_ninja_state,
                        on_activate=self.toggle_ninja_state))
        if self.show_ninja_state:
            for line in self._ninja_state_lines():
                rows.append(Row(STATUS, line))

        if self.recorder is not None:
            rows.append(Row(HEADER, "Recording"))
            rows.append(Row(ACTION, "Start (resets to spawn)", "B", pygame.K_b,
                            on_activate=self.start_recording))
            rows.append(Row(ACTION, "Stop (discard)", "N", pygame.K_n,
                            on_activate=self.stop_recording))
            state = "recording" if self.recorder.is_recording else "idle"
            rows.append(Row(STATUS, f"state: {state}"))
        return rows

    # -- event handling ----------------------------------------------------

    def _handle_events(self, rows: List[Row]):
        self._keys_just_pressed.clear()
        if self.headless:
            for event in pygame.event.get(pygame.QUIT):
                if event.type == pygame.QUIT:
                    self.running = False
            return
        if self.panel is not None:
            self.panel.set_hover(pygame.mouse.get_pos())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.panel:
                row = self.panel.control_at(event.pos)
                if row is not None and row.on_activate is not None:
                    if row.kind != FIELD:
                        self.focused_field = None
                    self._safe_activate(row.on_activate, row.label)
            elif event.type == pygame.MOUSEWHEEL and self.panel:
                # Wheel-up (+y) reveals content above, i.e. a smaller offset.
                if self.panel.contains(pygame.mouse.get_pos()):
                    self.panel.scroll_by(-event.y * layout.SCROLL_STEP)
            elif event.type == pygame.KEYUP:
                self._keys_held.discard(event.key)
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event, rows)

    def _handle_keydown(self, event, rows: List[Row]):
        # Field editing takes priority over hotkeys / movement registration.
        if self.focused_field is not None:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.commit_field()
            elif event.key == pygame.K_ESCAPE:
                self.focused_field = None
                self.field_buffer = ""
            elif event.key == pygame.K_BACKSPACE:
                self.field_buffer = self.field_buffer[:-1]
            elif event.unicode and (
                event.unicode.isdigit()
                or (not self._focused_is_numeric() and event.unicode.isprintable())
            ):
                self.field_buffer += event.unicode
            return

        # Panel scrolling (keyboard alternative to the mouse wheel).
        if self.panel is not None and event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
            step = layout.SCROLL_STEP * 3
            self.panel.scroll_by(-step if event.key == pygame.K_PAGEUP else step)
            return

        # Panel hotkeys (toggles / actions).
        row = controls.row_for_key(rows, event.key)
        if row is not None:
            self._safe_activate(row.on_activate, row.label)
            return

        # Discrete-action arrow registration.
        if getattr(self.args, "discrete_actions", False) and event.key in (
            pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP,
        ):
            if event.key not in self._keys_held:
                self._keys_just_pressed.add(event.key)
                self._keys_held.add(event.key)

    # -- ninja input -------------------------------------------------------

    def _compute_action(self) -> int:
        if self.headless or self.focused_field is not None:
            return 0
        if getattr(self.args, "step_on_input", False):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_z]:
                return 4
            if keys[pygame.K_x]:
                return 5
            if keys[pygame.K_PERIOD]:
                return 0
            if keys[pygame.K_UP]:
                return 4 if keys[pygame.K_LEFT] else 5 if keys[pygame.K_RIGHT] else 3
            return 1 if keys[pygame.K_LEFT] else 2 if keys[pygame.K_RIGHT] else 0
        if getattr(self.args, "discrete_actions", False):
            jp = self._keys_just_pressed
            if pygame.K_UP in jp:
                return 4 if pygame.K_LEFT in jp else 5 if pygame.K_RIGHT in jp else 3
            return 1 if pygame.K_LEFT in jp else 2 if pygame.K_RIGHT in jp else 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return 4 if keys[pygame.K_LEFT] else 5 if keys[pygame.K_RIGHT] else 3
        return 1 if keys[pygame.K_LEFT] else 2 if keys[pygame.K_RIGHT] else 0

    def _step(self, action: int):
        """Advance the sim. Returns (terminated, truncated, did_step)."""
        if getattr(self.args, "step_on_input", False) and not self.headless:
            explicit_noop = pygame.key.get_pressed()[pygame.K_PERIOD]
            if action == 0 and not explicit_noop:
                self._hold_start_ms = None
                self._hold_action = 0
                return False, False, False
            now_ms = time.perf_counter() * 1000
            if self._hold_start_ms is None or self._hold_action != action:
                self._hold_start_ms = now_ms
                self._hold_action = action
            elif (now_ms - self._hold_start_ms) < HOLD_REPEAT_DELAY_MS:
                return False, False, False
            _obs, _r, term, trunc, _info = self.env.step(action)
            self.observation = _obs
            self.input_sequence.append(action)
            return term, trunc, True
        _obs, _r, term, trunc, _info = self.env.step(action)
        self.observation = _obs
        return term, trunc, True

    # -- main loop ---------------------------------------------------------

    def run(self):
        clock = pygame.time.Clock() if not self.headless else None
        while self.running:
            if isinstance(self.map_source, FileMapSource):
                self.map_source.poll(self.env, self.frame_counter)

            rows = self.build_rows()
            self._handle_events(rows)

            # step-on-input overlay
            if getattr(self.args, "step_on_input", False) and not self.headless:
                self.env.set_input_sequence_overlay(self.input_sequence, enabled=True)

            self.overlays.apply(self.env)

            action = self._compute_action()
            terminated, truncated, did_step = self._step(action)

            if did_step and self.recorder is not None and self.recorder.is_recording:
                self.recorder.record_action(action)

            if (
                did_step and self.memory_profiler is not None
                and self.frame_counter % self.memory_profiler.snapshot_interval == 0
            ):
                self.memory_profiler.take_snapshot(self.frame_counter, obs=self.observation)

            if did_step and (terminated or truncated):
                if self._should_linger(truncated):
                    # Keep stepping: `env.step` still ticks the sim after it
                    # reports terminated, and `nsim.tick` drives the ragdoll for
                    # as long as the ninja is in state 6. So the corpse plays
                    # out on screen instead of vanishing on the death frame.
                    if not self._lingering:
                        self._lingering = True
                        print("Ninja died -- watching the ragdoll. R to reset.")
                else:
                    self._on_episode_end()

            self._render(clock, rows)

            self.frame_counter += 1
            if (
                getattr(self.args, "profile_frames", None) is not None
                and self.frame_counter >= self.args.profile_frames
            ):
                self.running = False

        self._finalize()

    def _should_linger(self, truncated: bool) -> bool:
        """Whether to let a death play out on screen rather than reset.

        The ragdoll is the only thing worth watching after a death and it is
        gone the instant the episode resets, so in the interactive tool a death
        holds until the user presses R. Anything running unattended -- headless,
        auto-advance, an active recording -- keeps the old behaviour, as does a
        truncation, which has no corpse to show.
        """
        if truncated or self.headless:
            return False
        if not getattr(self.args, "linger_on_death", True):
            return False
        if getattr(self.args, "auto_advance", False):
            return False
        if self.recorder is not None and self.recorder.is_recording:
            return False
        return self.env.nplay_headless.ninja_has_died()

    def _on_episode_end(self):
        self._lingering = False
        if getattr(self.args, "step_on_input", False):
            self.input_sequence.clear()
        won = self.env.nplay_headless.ninja_has_won()
        if self.recorder is not None and self.recorder.is_recording:
            self.recorder.stop_recording(success=won, save=won)
        handled = self.map_source.on_episode_end(
            self.env, won, getattr(self.args, "auto_advance", False)
        )
        if not handled:
            self.observation, _info = self.env.reset()

    def _render(self, clock, rows):
        if self.headless:
            return
        # Benchmark HUD lines (drawn by the renderer's debug overlay).
        if isinstance(self.map_source, BenchmarkSource) and not getattr(self.args, "no_benchmark_hud", False):
            self.env._benchmark_hud_lines = self.map_source.hud_lines(
                getattr(self.args, "auto_advance", False)
            )
        else:
            self.env._benchmark_hud_lines = None

        self.env.render()
        screen = pygame.display.get_surface()
        if self.panel3d is not None and screen is not None:
            try:
                pw = self.env.nplay_headless.sim_renderer.panel_width
                self.panel3d.resize(self.panel3d.w, screen.get_height())
                surf = self.panel3d.render(self.env.nplay_headless.sim)
                screen.blit(surf, (screen.get_width() - pw, 0))
            except Exception as exc:  # noqa: BLE001 - keep the dev tool usable
                print(f"3D panel render error: {exc}")
                self.panel3d.release()
                self.panel3d = None
        if self.panel is not None:
            self.panel.draw(screen, rows, self.focused_field, self.field_buffer)
            _draw_software_cursor(screen)
            pygame.display.flip()
        elif self.panel3d is not None:
            _draw_software_cursor(screen)
            pygame.display.flip()
        if clock is not None:
            clock.tick(60)

    def _finalize(self):
        if self.memory_profiler is not None:
            self.memory_profiler.finalize()
            if len(self.memory_profiler.snapshots) >= 2:
                self.memory_profiler.compare_snapshots(0, -1)
        if self.recorder is not None:
            self.recorder.print_statistics()


def _software_cursor_wanted() -> bool:
    """True when SDL will not present a hardware cursor and we must draw one.

    SDL's wayland backend under WSLg routinely fails to show the pointer: the
    client has to attach its own cursor surface and WSLg's compositor commonly
    drops it. The dev panel is mouse-driven -- toggles, sliders, text fields --
    so an invisible pointer makes the tool unusable.

    The clean fix is the x11 backend, which draws the cursor correctly, but
    pygame wheels are frequently built without it (SDL 2.28.4 here ships only
    wayland/offscreen/dummy), so `test_environment._select_sdl_driver` cannot
    always get there. This is the fallback for that case, and it is deliberately
    narrow: only when the wayland backend is actually in use, so a working
    hardware cursor is never doubled up with a drawn one.

    ``NCLONE_SOFTWARE_CURSOR=1`` / ``=0`` forces it on or off.
    """
    forced = os.environ.get("NCLONE_SOFTWARE_CURSOR")
    if forced is not None:
        return forced == "1"
    try:
        return pygame.display.get_driver() == "wayland"
    except pygame.error:
        return False


_SW_CURSOR_CACHE = {}


def _draw_software_cursor(screen) -> None:
    """Blit an arrow at the mouse position, if a software cursor is needed."""
    if screen is None or not _software_cursor_wanted():
        return
    if not pygame.mouse.get_focused():
        return  # pointer is outside the window; drawing one would be a lie
    cursor = _SW_CURSOR_CACHE.get("arrow")
    if cursor is None:
        # A classic arrow: white fill, black outline so it reads against both
        # the dark level art and the light panel.
        points = [(0, 0), (0, 16), (4, 12), (7, 18), (10, 17), (7, 11), (12, 11)]
        cursor = pygame.Surface((14, 20), pygame.SRCALPHA)
        pygame.draw.polygon(cursor, (0, 0, 0, 255), points)
        pygame.draw.polygon(cursor, (255, 255, 255, 255), points, 0)
        pygame.draw.polygon(cursor, (0, 0, 0, 255), points, 1)
        _SW_CURSOR_CACHE["arrow"] = cursor
    screen.blit(cursor, pygame.mouse.get_pos())


def run_dev_tool(env, args):
    """Construct and run the dev tool against an already-created env."""
    DevToolApp(env, args).run()
