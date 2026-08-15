"""P3: the catch-floor mine line must span the floor, not just the chamber.

``MapJumpPlatforms``' own docstring states the design contract:

    "The catch-floor is ALWAYS mined: the ninja spawns on an elevated platform and
     only ever touches the floor by falling, so a missed hop dies instantly (a
     clean terminal) instead of wandering to a timeout."

``place_floor_mines`` mined ``chamber_x1+1 .. chamber_x2+1`` on a 42-tile map, so
a ninja that leaves the last platform toward the chamber edge lands just past the
end of the line. Measured on PLATFORMING/T2 level 131837852402414503, the line is
exactly as dense as advertised (no internal gap wider than 12.5 px) but a ninja at
(950, 470) sits 17.2 px from the nearest mine against a 14 px kill threshold and
survived 5,000 frames immobile. The contract is violated on 100% of the corpus,
and the cost is the difference between a -3.5 death and a multi-thousand-frame
stall.

``FULL_WIDTH_FLOOR_MINES`` closes it. It is OPT-IN because enabling it changes the
levels, invalidating the benchmark dataset, the planner demos, the curricula and
every cross-run det-eval comparison -- so the default path must stay byte-identical.
"""

from __future__ import annotations

import dataclasses

import pytest

from nclone.constants import MAP_TILE_WIDTH
from nclone.map_generation.generator_configs import GENERATOR_PRESETS
from nclone.map_generation.generator_factory import GeneratorFactory

TILE = 24.0
ENT_START, ENT_STRIDE = 1235, 5
MINE_TYPES = frozenset({1, 21})
# NINJA_RADIUS 10 + toggle-mine radius 4.
KILL_PX = 14.0
PRESETS = ("platforms_t1", "platforms_t2", "platforms_t3")


def _gen(preset: str, seed: int, *, full_width: bool):
    cfg = GENERATOR_PRESETS["jump_platforms"][preset]
    if full_width:
        cfg = dataclasses.replace(cfg, full_width_floor_mines=True)
    g = GeneratorFactory.create_generator("jump_platforms", cfg, seed)
    g.generate(seed=seed)
    return [int(v) & 0xFF for v in g.map_data()]


def _floor_mines(md):
    """(x_px, y_px) of the mines on the lowest (catch-floor) row."""
    mines = []
    i = ENT_START
    while i + 4 < len(md):
        if md[i] in MINE_TYPES:
            mines.append((md[i + 1] * 6.0, md[i + 2] * 6.0))
        i += ENT_STRIDE
    if not mines:
        return []
    floor_y = max(m[1] for m in mines)
    return [m for m in mines if m[1] == floor_y]


def _safe_columns(md) -> int:
    """Tile columns where a ninja standing on the catch floor is NOT in a kill disk."""
    line = _floor_mines(md)
    if not line:
        return MAP_TILE_WIDTH
    floor_y = line[0][1]
    stand_y = floor_y - 10.0
    safe = 0
    for tx in range(MAP_TILE_WIDTH):
        px = tx * TILE + TILE / 2.0
        nearest = min(((px - mx) ** 2 + (stand_y - my) ** 2) ** 0.5
                      for mx, my in line)
        if nearest > KILL_PX:
            safe += 1
    return safe


@pytest.mark.parametrize("preset", PRESETS)
def test_default_still_leaves_the_measured_survivable_margin(preset):
    """The DEFAULT path must be unchanged -- this is the regression guard.

    These fractions are the ones measured across the 850-level corpus
    (T1 0.405, T2 0.357, T3 0.191); if the default starts mining full width, the
    benchmark dataset has silently changed.
    """
    safe = [_safe_columns(_gen(preset, s, full_width=False)) for s in range(2000, 2020)]
    assert all(v > 0 for v in safe), "default unexpectedly mines the whole floor"


@pytest.mark.parametrize("preset", PRESETS)
def test_full_width_leaves_no_survivable_catch_floor_column(preset):
    """The acceptance criterion: safe columns -> 0 on every level."""
    for s in range(2000, 2020):
        assert _safe_columns(_gen(preset, s, full_width=True)) == 0


@pytest.mark.parametrize("preset", PRESETS)
def test_full_width_line_spans_the_map(preset):
    md = _gen(preset, 2000, full_width=True)
    line = _floor_mines(md)
    assert line
    xs = [m[0] for m in line]
    assert min(xs) <= TILE, "line does not reach the left edge"
    assert max(xs) >= (MAP_TILE_WIDTH - 1) * TILE, "line does not reach the right edge"


@pytest.mark.parametrize("preset", PRESETS)
def test_line_density_is_preserved(preset):
    """Widening must not thin the line -- the internal gaps are what make it lethal."""
    line = sorted(m[0] for m in _floor_mines(_gen(preset, 2000, full_width=True)))
    gaps = [b - a for a, b in zip(line, line[1:])]
    assert max(gaps) <= 12.5, f"gap of {max(gaps)} px opened inside the line"


def test_mine_free_escape_hatch_still_wins_over_full_width():
    """MINE_SPACING >= 50 is the documented 'no floor mines' escape hatch, and the
    P8 OOD suite depends on it for the mine-free control. Full-width must not
    override it into a mined floor."""
    cfg = dataclasses.replace(
        GENERATOR_PRESETS["jump_platforms"]["platforms_t2"],
        mine_spacing=50.0, full_width_floor_mines=True)
    g = GeneratorFactory.create_generator("jump_platforms", cfg, 2000)
    g.generate(seed=2000)
    assert _floor_mines([int(v) & 0xFF for v in g.map_data()]) == []
