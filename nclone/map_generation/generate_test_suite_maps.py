"""DEPRECATED -- legacy CATEGORIES-based test-suite generator (now a shim).

This script used to generate a difficulty-category dataset from the legacy
``CATEGORIES`` taxonomy. That taxonomy is no longer the source of truth: the
mechanic-primary ``BUCKET_CATALOG`` in ``generator_configs.py`` is, and it now
covers all 23 ``map_*`` generators (see ``CURRICULUM_SCHEDULE``).

To avoid two diverging dataset pipelines, this module is now a thin shim that
delegates to ``scripts/generate_benchmark_dataset.py`` (the canonical procedural
curriculum-dataset generator). The on-disk layout it emits is the
``MechanicBenchmarkLoader`` format -- the same one ``TestSuiteLoader`` /
``EnvMapLoader`` already read.

Prefer calling the canonical generator directly::

    python -m scripts.generate_benchmark_dataset --output_dir ./datasets/mechanic_benchmark

This shim is retained only for backward compatibility (the ``TestSuiteGenerator``
symbol is imported by ``gym_environment/env_map_loader.py``) and for any caller
still invoking::

    python -m nclone.map_generation.generate_test_suite_maps --mode both
"""

import argparse
import sys
import warnings
from pathlib import Path


# Map the legacy --map_count knob onto per-bucket-per-tier counts. The benchmark
# generator sizes by (bucket x tier); divide the requested total across the
# in-scope bucket-tier cells, with a sane floor.
def _per_cell_counts(map_count: int) -> int:
    from nclone.map_generation.generator_configs import PHASE_1_BUCKETS, BUCKET_TIERS

    cells = max(1, len(PHASE_1_BUCKETS) * len(BUCKET_TIERS))
    return max(2, map_count // cells)


def _run_benchmark(output_dir: str, mode: str, map_count: int) -> int:
    """Delegate to the canonical mechanic-benchmark generator."""
    # Imported lazily so merely importing this module (e.g. for the
    # TestSuiteGenerator symbol) never requires the scripts package on sys.path.
    from scripts.generate_benchmark_dataset import (
        BenchmarkGenerator,
        DEFAULT_SEED_BASE_TRAIN,
        DEFAULT_SEED_BASE_TEST,
    )
    from nclone.map_generation.generator_configs import PHASE_1_BUCKETS, BUCKET_TIERS

    per_cell = _per_cell_counts(map_count)
    generator = BenchmarkGenerator(
        output_dir=Path(output_dir),
        buckets=list(PHASE_1_BUCKETS),
        tiers=list(BUCKET_TIERS),
        per_bucket_per_tier_train=per_cell,
        per_bucket_per_tier_test=max(1, per_cell // 2),
        seed_base_train=DEFAULT_SEED_BASE_TRAIN,
        seed_base_test=DEFAULT_SEED_BASE_TEST,
        # The legacy generators lack internal solvability checks; gate the new
        # superset dataset by default so degenerate maps are reseeded out.
        check_solvable=True,
    )
    return generator.run(mode)


class TestSuiteGenerator:
    """DEPRECATED thin wrapper retained for backward compatibility.

    Historically this class drove CATEGORIES-based generation. It now delegates
    to the mechanic-benchmark generator. Only the constructor and
    ``generate_all`` are kept, since those are the only members external callers
    used (``env_map_loader`` constructs it but never calls generation on it).
    """

    def __init__(self, base_output_dir: str = "./datasets", map_count: int = 2000):
        warnings.warn(
            "TestSuiteGenerator is deprecated; the mechanic BUCKET_CATALOG is now "
            "the single source of truth. Use scripts.generate_benchmark_dataset.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.base_output_dir = base_output_dir
        self.map_count = map_count

    def generate_all(self, mode: str = "both") -> None:
        _run_benchmark(self.base_output_dir, mode, self.map_count)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DEPRECATED shim -> scripts.generate_benchmark_dataset",
    )
    parser.add_argument(
        "--mode", choices=["train", "test", "both"], default="both",
        help="Which split(s) to emit (default: both)",
    )
    parser.add_argument(
        "--output_dir", default="./datasets/mechanic_benchmark",
        help="Base output directory (default: ./datasets/mechanic_benchmark)",
    )
    parser.add_argument(
        "--map_count", type=int, default=2000,
        help="Approximate total maps per split (mapped to per-bucket-tier counts)",
    )
    args = parser.parse_args()

    print(
        "[DEPRECATED] generate_test_suite_maps is a shim. Delegating to the "
        "canonical mechanic-benchmark generator (BUCKET_CATALOG).",
        file=sys.stderr,
    )
    return _run_benchmark(args.output_dir, args.mode, args.map_count)


if __name__ == "__main__":
    sys.exit(main())
