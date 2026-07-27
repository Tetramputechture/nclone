"""Dev-tool panel tests: the mirror control and the scrollable row list.

Both are exercised without a window: the map-source mirror logic runs against a
tiny fake env (only the handful of attributes the loaders touch), and the panel
draws onto an offscreen ``pygame.Surface``.
"""

import pygame
import pytest

from nclone.dev_tool import layout
from nclone.dev_tool.controls import ACTION, HEADER, STATUS, TOGGLE, Row
from nclone.dev_tool.loaders import MIRROR_SUFFIX, MapSource
from nclone.dev_tool.panel import DevPanel
from nclone.map_augmentation.transforms import mirror_map_data


class _FakeHeadless:
    def __init__(self):
        self.current_map_data = None

    def load_map_from_map_data(self, map_data):
        self.current_map_data = list(map_data)


class _FakeMapLoader:
    def __init__(self):
        self.current_map_name = None


class _FakeEnv:
    """The minimal env surface ``loaders`` uses to swap a level in."""

    def __init__(self):
        self.nplay_headless = _FakeHeadless()
        self.map_loader = _FakeMapLoader()

    def reset(self, options=None):
        if options and options.get("map_name") is not None:
            self.map_loader.current_map_name = options["map_name"]
        return None, {}

    def _get_observation(self):
        return {}

    def _process_observation(self, obs):
        return obs


class _FakeSource(MapSource):
    """A map source whose level list is supplied by the test."""

    def __init__(self, levels):
        self.levels = levels
        self.index = 0

    def _load_current(self, env):
        env.nplay_headless.load_map_from_map_data(self.levels[self.index])
        env.map_loader.current_map_name = f"level{self.index}"

    def advance(self, env):
        self.index = (self.index + 1) % len(self.levels)
        self.load_current(env)


def _level(seed: int):
    """A map_data-shaped list that is not left-right symmetric."""
    data = [0] * 1300
    data[184 + 3] = 6          # a 45-degree slope near the left of row 0
    data[184 + 10] = 1         # a solid tile
    data[1231], data[1232] = 40, 100   # ninja spawn
    data[1233] = 1                     # facing right
    data[1235:1240] = [3, 24 + seed, 60, 0, 0]  # one entity record
    return data


@pytest.fixture
def env_and_source():
    env = _FakeEnv()
    src = _FakeSource([_level(0), _level(7)])
    src.load_current(env)
    return env, src


def test_toggle_mirror_flips_level_and_name(env_and_source):
    env, src = env_and_source
    original = list(env.nplay_headless.current_map_data)

    assert src.toggle_mirror(env) is True
    assert src.mirrored is True
    assert env.nplay_headless.current_map_data == mirror_map_data(original)
    assert env.map_loader.current_map_name.endswith(MIRROR_SUFFIX)


def test_toggle_mirror_is_involutive(env_and_source):
    env, src = env_and_source
    original = list(env.nplay_headless.current_map_data)

    src.toggle_mirror(env)
    src.toggle_mirror(env)

    assert src.mirrored is False
    assert env.nplay_headless.current_map_data == original
    assert env.map_loader.current_map_name == "level0"


def test_toggle_mirror_without_a_map_is_a_no_op():
    env = _FakeEnv()
    src = _FakeSource([_level(0)])

    assert src.toggle_mirror(env) is False
    assert src.mirrored is False


def test_mirror_is_reapplied_to_newly_loaded_levels(env_and_source):
    env, src = env_and_source
    src.toggle_mirror(env)

    src.advance(env)

    assert env.nplay_headless.current_map_data == mirror_map_data(_level(7))


def test_reload_does_not_unmirror_an_already_mirrored_level(env_and_source):
    """A plain reset keeps the loaded map data; re-mirroring it would flip back."""
    env, src = env_and_source
    src.toggle_mirror(env)
    mirrored = list(env.nplay_headless.current_map_data)

    for _ in range(3):
        src._apply_mirror(env)  # what every load path calls

    assert env.nplay_headless.current_map_data == mirrored


# -- panel layout / scrolling -------------------------------------------------


def _rows(n_toggles: int):
    rows = [Row(HEADER, "Map source"), Row(ACTION, "Next", "G")]
    rows.append(Row(TOGGLE, "Mirror (H-flip)", "O", get_state=lambda: False))
    rows.append(Row(STATUS, "source: test"))
    rows.append(Row(HEADER, "Overlays"))
    rows += [Row(TOGGLE, f"toggle {i}", "T", get_state=lambda: False) for i in range(n_toggles)]
    rows.append(Row(ACTION, "Last control", "Z"))
    return rows


@pytest.fixture
def panel_and_screen():
    pygame.init()
    screen = pygame.Surface((1056 + layout.PANEL_WIDTH, 600))
    yield DevPanel(), screen
    pygame.quit()


def test_short_row_list_does_not_scroll(panel_and_screen):
    panel, screen = panel_and_screen
    rows = _rows(4)
    assert layout.content_height(rows) <= 600

    panel.draw(screen, rows, None, "")

    assert panel._max_scroll == 0
    panel.scroll_by(layout.SCROLL_STEP)
    assert panel.scroll == 0


def test_every_control_is_reachable_by_scrolling(panel_and_screen):
    panel, screen = panel_and_screen
    rows = _rows(40)
    panel.draw(screen, rows, None, "")
    assert panel._max_scroll > 0
    assert all(row.label != "Last control" for _rect, row in panel._hit)

    panel.scroll_by(10 * layout.SCROLL_STEP)  # clamps at the bottom
    panel.draw(screen, rows, None, "")

    assert panel.scroll == panel._max_scroll
    assert panel._hit[-1][1].label == "Last control"


def test_scrolled_out_rows_are_not_clickable(panel_and_screen):
    panel, screen = panel_and_screen
    rows = _rows(40)
    panel.draw(screen, rows, None, "")
    first_rect, first_row = panel._hit[0]
    assert panel.control_at(first_rect.center) is first_row

    panel.scroll_by(10 * layout.SCROLL_STEP)
    panel.draw(screen, rows, None, "")

    assert all(row is not first_row for _rect, row in panel._hit)
    assert all(rect.bottom > 0 and rect.top < 600 for rect, _row in panel._hit)


def test_scroll_is_reclamped_when_the_row_list_shrinks(panel_and_screen):
    panel, screen = panel_and_screen
    panel.draw(screen, _rows(40), None, "")
    panel.scroll_by(10 * layout.SCROLL_STEP)
    assert panel.scroll > 0

    panel.draw(screen, _rows(4), None, "")

    assert panel.scroll == 0


def test_contains_only_matches_the_panel_strip(panel_and_screen):
    panel, screen = panel_and_screen
    panel.draw(screen, _rows(4), None, "")
    w = screen.get_width()

    assert panel.contains((w - 10, 300))
    assert not panel.contains((10, 300))
