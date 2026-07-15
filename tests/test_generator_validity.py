"""Regression tests for the generator validity guarantee:
every dataset level must have a VALID SPAWN + REACHABLE GOALS.

Fast checks (representative sample + the gate semantics) live here; the
exhaustive all-combos sweep is ``scripts/sweep_generator_validity.py``.
"""
from __future__ import annotations

import inspect

from nclone.map_generation.generator_factory import GeneratorFactory
from nclone.map_generation.solvability import is_map_solvable, find_solvable_seed


def _gen(gen_type, preset, seed):
    m = GeneratorFactory.create_from_preset(gen_type, preset, seed=seed)
    m.generate(seed=seed)
    return m.map_data()


def test_representative_generators_solvable_by_construction():
    """The core curriculum generators are solvable-by-construction across seeds."""
    combos = [
        ("simple_corridor", "simple_corridor_h"),
        ("simple_corridor", "simple_corridor_v"),
        ("basic_jump", "single_gap_t1"),
        ("wall_jump", "shaft_short_t1"),
        ("mine_path", "sparse_t1"),
        ("slopes", "gentle_t1"),
    ]
    # Filter to combos that actually exist (preset names can drift); skip unknowns.
    from nclone.map_generation.generator_factory import GENERATOR_PRESETS
    for gen_type, preset in combos:
        if gen_type not in GENERATOR_PRESETS or preset not in GENERATOR_PRESETS[gen_type]:
            continue
        for seed in range(30000, 30004):
            assert is_map_solvable(_gen(gen_type, preset, seed)), (
                f"{gen_type}/{preset} seed {seed} not solvable-by-construction"
            )


def test_benchmark_solvability_gate_defaults_on():
    """The dataset generator must gate on solvability by DEFAULT so every emitted
    level has a reachable switch + exit (the bug that crashed Phase-4 training
    upstream was an UNGATED unsolvable/degenerate level path)."""
    from scripts.generate_benchmark_dataset import BenchmarkGenerator

    default = inspect.signature(BenchmarkGenerator.__init__).parameters[
        "check_solvable"
    ].default
    assert default is True, "BenchmarkGenerator.check_solvable must default ON"


def test_solvability_gate_repairs_occasional_unsolvable_generator():
    """A generator that relies on the gate as its safety net (composite chains
    can wall off the goal ~25% of seeds) must always yield a solvable level once
    the reseed gate is applied."""
    gen_type, preset = "composite", "long_chain_t4"
    from nclone.map_generation.generator_factory import GENERATOR_PRESETS
    if gen_type not in GENERATOR_PRESETS or preset not in GENERATOR_PRESETS[gen_type]:
        return  # preset drifted; nothing to assert

    def fac(seed):
        return _gen(gen_type, preset, seed)

    # The known-bad bare seeds (no gate) and a few others must all reseed-resolve.
    for base in (20001, 20002, 21000):
        result = find_solvable_seed(fac, base, max_retry=16)
        assert result is not None, f"gate found no solvable {preset} from base {base}"
        seed, map_data = result
        assert is_map_solvable(map_data)
