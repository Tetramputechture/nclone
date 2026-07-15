"""Regression guard: the benchmark generator can NEVER emit an unsolvable level.

Unsolvable levels silently corrupt training -- an unreachable switch/exit makes
the free-space cost-to-go ``inf``, which kills the PBRS / goal-distance curriculum
signal for that level and floods logs with "position far outside graph bounds".
The v4 dataset shipped with ~1% such levels (genuine bad procgen -- mine-walled
chambers, disconnected spawn pockets) because the per-seed ``check_solvable`` gate
was only as current as the reachability model at generation time.

This test locks in the two-layer guarantee added alongside the fix:

  1. ``BenchmarkGenerator(check_solvable=True)`` reseeds out unsolvable seeds, AND
  2. ``run()`` performs a whole-dataset ``is_map_solvable`` re-audit before it
     returns 0 -- so a per-seed-gate bypass can never ship.

It generates a small real dataset and asserts EVERY emitted level is solvable
under the current jump-aware traversability graph. The dataset itself is
gitignored (too large / regenerable), so the invariant is tested at the
GENERATION boundary, which is where it must hold.
"""
import pickle
from pathlib import Path

import pytest

from nclone.map_generation.solvability import is_map_solvable


@pytest.fixture(scope="module")
def generated_dataset(tmp_path_factory):
    """Generate a small check_solvable dataset once for the module."""
    from scripts.generate_benchmark_dataset import (
        BenchmarkGenerator,
        DEFAULT_SEED_BASE_TRAIN,
        DEFAULT_SEED_BASE_TEST,
    )

    out = tmp_path_factory.mktemp("mbench_solvable")
    gen = BenchmarkGenerator(
        output_dir=out,
        # A spread of generators incl. the mine/chamber families that produced
        # the v4 bad levels, kept small for test speed.
        buckets=["FLAT_RUN", "BASIC_JUMP", "MINE_NAVIGATION", "CHAMBER_MAZE"],
        tiers=[1, 2],
        per_bucket_per_tier_train=4,
        per_bucket_per_tier_test=2,
        seed_base_train=DEFAULT_SEED_BASE_TRAIN,
        seed_base_test=DEFAULT_SEED_BASE_TEST,
        check_solvable=True,
    )
    rc = gen.run("both")
    return out, rc


def test_generation_returns_success(generated_dataset):
    """run() returns 0 -- which now includes the post-generation audit passing."""
    _out, rc = generated_dataset
    assert rc == 0, f"generation (incl. post-gen solvability audit) failed rc={rc}"


def test_every_emitted_level_is_solvable(generated_dataset):
    """Independently re-audit every pickle: not one may be unsolvable."""
    out, _rc = generated_dataset
    pkls = [p for split in ("train", "test") for p in (out / split).rglob("*.pkl")]
    assert pkls, "no levels were generated"

    unsolvable = []
    for pkl in pkls:
        with open(pkl, "rb") as f:
            md = list(pickle.load(f)["map_data"])
        if not is_map_solvable(md):
            unsolvable.append(f"{pkl.parts[-3]}/{pkl.parts[-2]}/{pkl.stem}")

    assert not unsolvable, (
        f"{len(unsolvable)}/{len(pkls)} generated levels are UNSOLVABLE: "
        f"{unsolvable[:10]}"
    )


def test_post_gen_audit_hook_detects_planted_unsolvable(generated_dataset, tmp_path):
    """The run() audit hook must FLAG an unsolvable level, not pass it.

    Copy one generated level, corrupt it into an unsolvable map (wall off the
    switch by filling the play area with solid tiles), and confirm the audit
    helper reports it -- proving the guarantee is a real gate, not a no-op.
    """
    from scripts.generate_benchmark_dataset import BenchmarkGenerator

    out, _rc = generated_dataset
    src = next((out / "train").rglob("*.pkl"))
    payload = pickle.load(open(src, "rb"))
    md = list(payload["map_data"])
    # Fill the entire 42x23 tile grid with full-solid tiles -> nothing reachable.
    for i in range(966):
        md[i] = 1
    payload["map_data"] = md

    planted = tmp_path / "train" / "FLAT_RUN" / "T1"
    planted.mkdir(parents=True)
    with open(planted / f"{payload['level_id']}.pkl", "wb") as f:
        pickle.dump(payload, f)

    gen = BenchmarkGenerator(
        output_dir=tmp_path,
        buckets=["FLAT_RUN"],
        tiers=[1],
        per_bucket_per_tier_train=1,
        per_bucket_per_tier_test=1,
        seed_base_train=1,
        seed_base_test=1,
    )
    bad = gen._audit_output_solvable()
    assert bad, "audit hook failed to flag a planted unsolvable level"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
