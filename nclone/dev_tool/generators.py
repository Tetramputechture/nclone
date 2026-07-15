"""Browse and solvability-check every curriculum map generator.

Walks the full ``(generator_type, preset)`` space exposed by
:data:`nclone.map_generation.generator_factory.GENERATOR_PRESETS` (all hardcoded
generators *and* the auto-registered ``template_*`` composable templates), not
just one difficulty category. Each generated level is classified with
:func:`nclone.map_generation.solvability.is_map_solvable` so the dev panel can
show live whether a level is *possible* (exit switch + door reachable from
spawn).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Importing the factory module ensures the compose templates are registered into
# GENERATOR_PRESETS (its import side-effect runs ``_register_compose_templates``).
from ..map_generation.generator_factory import GeneratorFactory, GENERATOR_PRESETS
from ..map_generation.solvability import is_map_solvable


def all_generator_combos() -> List[Tuple[str, str]]:
    """Return every ``(generator_type, preset)`` pair, sorted for stable order."""
    combos: List[Tuple[str, str]] = []
    for gen_type in sorted(GENERATOR_PRESETS):
        for preset in GENERATOR_PRESETS[gen_type]:
            combos.append((gen_type, preset))
    return combos


class GeneratorBrowser:
    """Stateful cursor over all generator/preset combos with solvability status."""

    # Solvability status values.
    UNKNOWN = "?"
    SOLVABLE = "SOLVABLE"
    UNSOLVABLE = "UNSOLVABLE"
    ERROR = "ERROR"

    def __init__(self, start_seed: int = 10000, check_solvable: bool = True):
        self.combos = all_generator_combos()
        if not self.combos:
            raise RuntimeError("No generators registered in GENERATOR_PRESETS")
        self.index = 0
        self.seed = int(start_seed)
        self.check_solvable = check_solvable
        # Cached result of the most recent generate().
        self.current_map = None
        self.status = self.UNKNOWN
        self.last_error: Optional[str] = None
        # Optional summary line set by scan_all().
        self.scan_summary: Optional[str] = None

    # -- navigation --------------------------------------------------------

    @property
    def gen_type(self) -> str:
        return self.combos[self.index][0]

    @property
    def preset(self) -> str:
        return self.combos[self.index][1]

    def next(self) -> None:
        self.index = (self.index + 1) % len(self.combos)

    def prev(self) -> None:
        self.index = (self.index - 1) % len(self.combos)

    def set_seed(self, seed: int) -> None:
        self.seed = int(seed)

    def bump_seed(self, delta: int = 1) -> None:
        self.seed += delta

    # -- generation --------------------------------------------------------

    def generate(self):
        """Generate the current combo at the current seed; classify solvability.

        Returns the generated ``Map`` (also stored on ``self.current_map``), or
        re-raises on a hard generation failure after recording the error.
        """
        gen_type, preset = self.combos[self.index]
        self.last_error = None
        try:
            map_obj = GeneratorFactory.create_from_preset(gen_type, preset, seed=self.seed)
            map_obj.generate(seed=self.seed)
        except Exception as exc:
            self.current_map = None
            self.status = self.ERROR
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        self.current_map = map_obj
        if self.check_solvable:
            try:
                self.status = (
                    self.SOLVABLE
                    if is_map_solvable(map_obj.map_data())
                    else self.UNSOLVABLE
                )
            except Exception as exc:  # solvability gate is best-effort
                self.status = self.ERROR
                self.last_error = f"solvability: {exc}"
        else:
            self.status = self.UNKNOWN
        return map_obj

    def to_ascii(self) -> str:
        if self.current_map is None:
            return "(no map generated yet)"
        return self.current_map.to_ascii(show_coords=False)

    # -- batch scan --------------------------------------------------------

    def scan_all(self, seed: Optional[int] = None) -> dict:
        """Generate + solvability-check every combo once; print a report.

        Does not change the current cursor/seed. Returns a dict of
        ``{(gen_type, preset): status}``. Intended to answer "are all the
        curriculum generators producing possible levels?".
        """
        scan_seed = self.seed if seed is None else int(seed)
        results: dict = {}
        n_solvable = n_unsolvable = n_error = 0
        print("\n" + "=" * 70)
        print(f"GENERATOR SOLVABILITY SCAN  (seed={scan_seed}, {len(self.combos)} combos)")
        print("=" * 70)
        for gen_type, preset in self.combos:
            try:
                map_obj = GeneratorFactory.create_from_preset(
                    gen_type, preset, seed=scan_seed
                )
                map_obj.generate(seed=scan_seed)
                status = (
                    self.SOLVABLE
                    if is_map_solvable(map_obj.map_data())
                    else self.UNSOLVABLE
                )
            except Exception as exc:
                status = f"{self.ERROR} ({type(exc).__name__})"
            results[(gen_type, preset)] = status
            if status == self.SOLVABLE:
                n_solvable += 1
            elif status == self.UNSOLVABLE:
                n_unsolvable += 1
                print(f"  UNSOLVABLE  {gen_type}:{preset}")
            else:
                n_error += 1
                print(f"  {status:<18}{gen_type}:{preset}")
        print("-" * 70)
        print(
            f"  solvable={n_solvable}  unsolvable={n_unsolvable}  error={n_error}"
            f"  total={len(self.combos)}"
        )
        print("=" * 70 + "\n")
        self.scan_summary = (
            f"scan@{scan_seed}: {n_solvable} ok / {n_unsolvable} bad / {n_error} err"
        )
        return results

    # -- display -----------------------------------------------------------

    def info_lines(self) -> List[str]:
        lines = [
            f"{self.gen_type}:{self.preset}",
            f"combo {self.index + 1}/{len(self.combos)}   seed {self.seed}",
            f"status: {self.status}",
        ]
        if self.last_error:
            lines.append(self.last_error[:38])
        if self.scan_summary:
            lines.append(self.scan_summary)
        return lines
