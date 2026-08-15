"""Selecting a generator or a BUCKET in the dev tool, and generating from it.

The dev tool could already browse the `(generator_type, preset)` space, which is
the wrong axis for validating a curriculum: what training samples is a
`(bucket, tier)` CELL, and a cell is a weighted SET of presets. Answering "what
does DRONE T3 actually look like" through the generator browser meant knowing
which presets that cell happens to draw and stepping to each by hand -- past
~155 combos in alphabetical order.

`BenchmarkSource` browses by bucket but only over a dataset already built to
disk, which is the wrong tool the moment a bucket is NEW: the hazard buckets
cost ~13 s per level to generate, so the loop for "did my new bucket come out
right" was a multi-hour dataset build. `BucketBrowser` generates the cell on the
spot instead.
"""

from __future__ import annotations

import pytest

from nclone.dev_tool.generators import BucketBrowser, GeneratorBrowser
from nclone.map_generation.generator_configs import BUCKET_CATALOG


# ---------------------------------------------------------------------------
# generator browser: the coarse axis and live re-filtering
# ---------------------------------------------------------------------------


def test_generator_browser_can_jump_by_TYPE_not_just_by_preset():
    """Stepping one preset at a time is fine inside a generator and useless
    between them -- `single_chamber` alone has a dozen presets."""
    b = GeneratorBrowser(check_solvable=False)
    types = b.types()
    assert len(types) > 5

    first = b.gen_type
    b.next_type()
    assert b.gen_type != first
    b.prev_type()
    assert b.gen_type == first


def test_generator_type_jump_lands_on_that_types_first_preset():
    b = GeneratorBrowser(check_solvable=False)
    b.next_type()
    landed = b.gen_type
    assert b.combos[b.index] == next(c for c in b.combos if c[0] == landed)


def test_generator_filter_is_editable_live():
    """The CLI flag can only be set before the window opens, so validating a
    second generator used to mean restarting the tool."""
    b = GeneratorBrowser(check_solvable=False)
    assert b.set_filter("hazard_overlay")
    assert {g for g, _p in b.combos} == {"hazard_overlay"}
    assert b.index == 0


def test_a_filter_that_matches_nothing_leaves_the_cursor_alone():
    """A typo must not empty the browser and strand the tool with no map."""
    b = GeneratorBrowser(check_solvable=False)
    assert b.set_filter("hazard_overlay")
    before = list(b.combos)
    assert b.set_filter("no_such_generator_xyz") is False
    assert b.combos == before


# ---------------------------------------------------------------------------
# bucket browser
# ---------------------------------------------------------------------------


def test_it_enumerates_every_populated_cell_and_skips_the_empty_placeholders():
    """SWITCH_PUZZLE / LAUNCH_PAD are declared but empty until their phase
    lands; browsing them would raise on every step."""
    b = BucketBrowser(check_solvable=False)
    buckets = {name for name, _tier in b.cells}
    assert "DRONE" in buckets and "FLAT_RUN" in buckets
    assert "SWITCH_PUZZLE" not in buckets
    assert "LAUNCH_PAD" not in buckets
    for name, tier in b.cells:
        assert BUCKET_CATALOG[name][tier], f"{name}/T{tier} is empty"


def test_filters_by_bucket_and_by_tier():
    b = BucketBrowser(check_solvable=False, bucket_filter="DRONE")
    assert {n for n, _t in b.cells} == {"DRONE"}
    assert sorted(t for _n, t in b.cells) == [1, 2, 3, 4]

    one = BucketBrowser(check_solvable=False, bucket_filter="DRONE", tier=3)
    assert one.cells == [("DRONE", 3)]


def test_an_unmatched_filter_raises_rather_than_opening_on_nothing():
    with pytest.raises(RuntimeError, match="No bucket cells match"):
        BucketBrowser(check_solvable=False, bucket_filter="NO_SUCH_BUCKET")


def test_tier_navigation_stays_inside_the_current_bucket():
    """Comparing T1 against T4 of ONE bucket is the check that matters: a T4
    that reads easier than its T1 is what the tier presets exist to prevent."""
    b = BucketBrowser(check_solvable=False)
    while b.bucket != "DRONE":
        b.next_bucket()
    seen = []
    for _ in range(4):
        seen.append((b.bucket, b.tier))
        b.next_tier()
    assert seen == [("DRONE", 1), ("DRONE", 2), ("DRONE", 3), ("DRONE", 4)]
    assert b.tier == 1, "tier navigation must wrap within the bucket"


def test_bucket_navigation_moves_whole_buckets_not_cells():
    b = BucketBrowser(check_solvable=False)
    start = b.bucket
    b.next_bucket()
    assert b.bucket != start
    assert b.tier == min(BUCKET_CATALOG[b.bucket]), "should land on the first tier"
    b.prev_bucket()
    assert b.bucket == start


def test_entry_navigation_walks_a_cells_weighted_set_then_rolls_over():
    """A cell is a weighted SET of presets. Sampling picks by weight; for
    eyeballing you want each entry deliberately rather than by luck."""
    b = BucketBrowser(check_solvable=False, bucket_filter="FLAT_RUN", tier=1)
    n = len(b.entries)
    assert n > 1, "FLAT_RUN/T1 is expected to hold several entries"
    seen = set()
    for _ in range(n):
        seen.add(b.entry[:2])
        b.next()
    assert len(seen) == n
    assert b.entry_index == 0, "should have rolled back to the first entry"


def test_bucket_filter_is_editable_live_and_rejects_a_miss():
    b = BucketBrowser(check_solvable=False)
    assert b.set_filter("ROCKET")
    assert {n for n, _t in b.cells} == {"ROCKET"}
    before = list(b.cells)
    assert b.set_filter("nope_xyz") is False
    assert b.cells == before, "a filter miss must not empty the browser"


# ---------------------------------------------------------------------------
# generation and the entity readout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bucket,expect", [
    ("ROCKET", "rocket"),
    ("FLOOR_GUARD", "guard"),
])
def test_generating_a_hazard_cell_names_its_entities(bucket, expect):
    """The entity readout is the line you actually read when validating a hazard
    bucket. A DRONE level with no drone, or one carrying a stray turret, is a
    valid level and looks completely normal on screen -- the panel has to say so
    or the whole exercise is watching pretty pictures.
    """
    b = BucketBrowser(start_seed=99, check_solvable=False, bucket_filter=bucket,
                      tier=1)
    b.generate()
    assert b.current_map is not None
    summary = b.entity_summary()
    assert expect in summary, summary
    assert any(bucket in line for line in b.info_lines())


def test_a_non_hazard_cell_reports_no_hazard_entities():
    b = BucketBrowser(start_seed=99, check_solvable=False,
                      bucket_filter="FLAT_RUN", tier=1)
    b.generate()
    assert b.entity_summary() == "entities: none"


def test_info_lines_survive_a_generator_that_has_not_run_yet():
    """The panel draws before the first generate() on a source that failed to
    load, and a crash there takes the whole tool down."""
    b = BucketBrowser(check_solvable=False, bucket_filter="ROCKET")
    assert b.entity_summary() == "entities: -"
    assert b.info_lines()
    assert b.to_ascii().startswith("(no map")


def test_seed_controls_are_present_on_both_browsers():
    for b in (GeneratorBrowser(check_solvable=False),
              BucketBrowser(check_solvable=False)):
        b.set_seed(4242)
        assert b.seed == 4242
        b.bump_seed()
        assert b.seed == 4243


# ---------------------------------------------------------------------------
# the MapSource wrapper
# ---------------------------------------------------------------------------


class _StubEnv:
    """Captures the map data the source pushes, without a simulator."""

    def __init__(self):
        self.loaded = []


def test_bucket_source_navigation_loads_a_map_each_time(monkeypatch):
    from nclone.dev_tool import loaders

    env = _StubEnv()
    monkeypatch.setattr(
        loaders, "load_map_data_into_env",
        lambda e, md, **kw: e.loaded.append(list(md)),
    )
    src = loaders.BucketSource(
        BucketBrowser(start_seed=7, check_solvable=False,
                      bucket_filter="ROCKET")
    )
    src.load_current(env)
    src.next_tier(env)
    src.advance(env)
    src.regenerate(env)
    assert len(env.loaded) == 4
    assert all(len(md) > 1000 for md in env.loaded)


def test_a_raising_generator_does_not_close_the_window(monkeypatch):
    """The reported symptom: pressing 'next generator' sometimes errored and the
    window closed.

    `GeneratorBrowser.generate()` re-raises after recording the failure, nothing
    between it and the pygame event loop caught anything, so one raising preset
    killed the tool. And generators raise LEGITIMATELY -- `hazard_overlay` and
    `mine_overlay` both raise `RuntimeError` when they cannot certify a level
    inside their retry budget, which is a normal outcome for a seed and one that
    browsing exists to discover.
    """
    from nclone.dev_tool import loaders

    env = _StubEnv()
    monkeypatch.setattr(
        loaders, "load_map_data_into_env",
        lambda e, md, **kw: e.loaded.append(list(md)),
    )
    browser = GeneratorBrowser(check_solvable=False, name_filter="single_chamber")
    src = loaders.GeneratorSource(browser)
    src.load_current(env)
    assert len(env.loaded) == 1

    def _boom():
        browser.status = browser.ERROR
        browser.last_error = "RuntimeError: no certified level"
        raise RuntimeError("hazard_overlay: no certified level after 12 attempts")

    monkeypatch.setattr(browser, "generate", _boom)
    src.advance(env)  # must not raise
    assert len(env.loaded) == 1, "the previous level must stay loaded"
    assert any("no certified level" in line for line in src.status_lines())

    # ...and the tool recovers on the next working combo.
    monkeypatch.undo()
    monkeypatch.setattr(
        loaders, "load_map_data_into_env",
        lambda e, md, **kw: e.loaded.append(list(md)),
    )
    src.advance(env)
    assert len(env.loaded) == 2


@pytest.mark.parametrize("source_kind", ["generator", "bucket"])
def test_a_failing_LOAD_does_not_close_the_window_either(monkeypatch, capsys,
                                                        source_kind):
    """The load is the more dangerous half and the easier one to forget.

    `load_map_data_into_env` resets the env, which rebuilds the graph,
    reachability and observation caches against a level nothing has ever seen.
    An earlier version of this fix guarded only `generate()`, which would have
    left the window closing on exactly the maps most worth looking at.
    """
    from nclone.dev_tool import loaders

    env = _StubEnv()

    def _explode(e, md, **kw):
        raise ValueError("graph build failed on a level nobody has seen")

    monkeypatch.setattr(loaders, "load_map_data_into_env", _explode)

    if source_kind == "generator":
        browser = GeneratorBrowser(check_solvable=False,
                                   name_filter="single_chamber")
        src = loaders.GeneratorSource(browser)
    else:
        browser = BucketBrowser(check_solvable=False, bucket_filter="FLAT_RUN")
        src = loaders.BucketSource(browser)

    src.load_current(env)  # must not raise
    out = capsys.readouterr().out
    assert "LOAD failed" in out
    assert "Traceback" in out, "the traceback must reach stdout with the message"
    assert browser.status == browser.ERROR
    assert "load:" in (browser.last_error or "")
    assert any("load:" in line for line in src.status_lines()), (
        "the panel must name the failure next to the combo that caused it"
    )


def test_interactive_browsing_skips_the_route_certificate_by_default():
    """The reported crash was on 'Next generator' from `drop_landing` -- which
    lands on `hazard_overlay`, the very next type alphabetically.

    The overlay generators certify each level by planning a route in a FORKED
    child with a 12 s cap and up to 12 base retries. Inside an event loop that is
    an unbounded freeze, and the fork inherits the parent's SDL/X11 connection.
    Browsing does not need the certificate: the layout is the same and
    `is_map_solvable` still runs.
    """
    from nclone.map_generation.generator_factory import GeneratorFactory

    b = GeneratorBrowser(check_solvable=False, name_filter="hazard_overlay")
    assert b.certify_routes is False
    b.generate()
    assert b.current_map is not None
    assert b.current_map.REQUIRE_ROUTE_CERTIFICATE is False
    assert any("certificate: OFF" in line for line in b.info_lines()), (
        "an uncertified level looks identical to a certified one; the panel has "
        "to say which it is"
    )

    # And a plain generator is untouched by the budget -- it has no such knob.
    plain = GeneratorFactory.create_from_preset("single_chamber", "flat_run_t2", 1)
    from nclone.dev_tool.generators import _apply_interactive_budget

    _apply_interactive_budget(plain, certify_routes=False)  # must not raise


def test_opting_back_in_restores_the_certificate():
    b = GeneratorBrowser(check_solvable=False, name_filter="hazard_overlay",
                         certify_routes=True)
    b.generate()
    assert b.current_map.REQUIRE_ROUTE_CERTIFICATE is True
    assert not any("certificate: OFF" in line for line in b.info_lines())


def test_the_route_worker_forces_the_dummy_driver_rather_than_defaulting_it():
    """`setdefault` is a no-op when the parent already set SDL_VIDEODRIVER, which
    is exactly the case in the dev tool -- so the forked child kept the parent's
    display. The override has to be unconditional."""
    import ast
    import inspect
    import textwrap

    from nclone.map_generation import map_mine_overlay

    src = textwrap.dedent(inspect.getsource(map_mine_overlay._route_worker))
    fn = ast.parse(src).body[0]
    # Drop the docstring: it *explains* the setdefault bug, so a naive substring
    # search over the source matches the prose and not the code.
    body = fn.body[1:] if (
        isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
    ) else fn.body
    code = "\n".join(ast.unparse(node) for node in body)

    assert "setdefault" not in code, (
        "SDL_VIDEODRIVER must be FORCED: setdefault is a no-op when the parent "
        "already set it, which is exactly the dev tool's case"
    )
    assert "environ['SDL_VIDEODRIVER'] = 'dummy'" in code
    assert "display.quit()" in code


def test_row_for_key_returns_the_row_without_activating_it():
    """Separating lookup from activation is what lets the app own the call and
    contain a failure."""
    from nclone.dev_tool import controls

    fired = []
    row = controls.Row(controls.ACTION, "Boom", "X", ord("x"),
                       on_activate=lambda: fired.append(1))
    found = controls.row_for_key([row], ord("x"))
    assert found is row
    assert fired == [], "lookup must not activate"
    assert controls.row_for_key([row], ord("z")) is None


def test_safe_activate_contains_and_records_any_control_failure(capsys):
    """The general backstop: EVERY panel control dispatches through this, not
    just the generator ones, so no single callback can close the window."""
    from nclone.dev_tool.app import DevToolApp

    class _Bare:
        last_action_error = None

    app = _Bare()
    safe = DevToolApp._safe_activate.__get__(app)

    safe(lambda: (_ for _ in ()).throw(RuntimeError("kaboom")), "Next generator")
    assert app.last_action_error is not None
    assert "kaboom" in app.last_action_error
    assert "Next generator" in app.last_action_error
    out = capsys.readouterr().out
    assert "contained" in out and "Traceback" in out, (
        "the traceback must reach the terminal -- a silent containment reads as "
        "'the button did nothing'"
    )

    # A subsequent success clears it, so the panel does not show a stale error.
    safe(lambda: None, "Next generator")
    assert app.last_action_error is None


def test_a_raising_bucket_generator_does_not_kill_the_tool(monkeypatch):
    """The whole point of browsing is to FIND the broken cell. A generator that
    raises must leave the tool running with the failure shown in the panel."""
    from nclone.dev_tool import loaders

    env = _StubEnv()
    monkeypatch.setattr(
        loaders, "load_map_data_into_env",
        lambda e, md, **kw: e.loaded.append(list(md)),
    )
    browser = BucketBrowser(check_solvable=False, bucket_filter="ROCKET")

    def _boom():
        raise RuntimeError("no certified level")

    monkeypatch.setattr(browser, "generate", _boom)
    src = loaders.BucketSource(browser)
    src.load_current(env)  # must not raise
    assert env.loaded == []
    assert src.status_lines()
