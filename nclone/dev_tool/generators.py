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


def _apply_interactive_budget(map_obj, certify_routes: bool) -> None:
    """Turn OFF the d* route certificate for interactive browsing.

    The overlay generators (`mine_overlay`, `hazard_overlay`) certify every level
    by planning a winning route through it, in a forked child with a wall-clock
    cap, retrying up to `MAX_BASE_RETRY` base levels. That is right for a dataset
    build and wrong inside an event loop twice over:

    * **It blocks for an unbounded time.** Typical is well under a second, but
      the cap is per-attempt: 12 base retries x a 12 s planner cap is a window
      that stops pumping events for minutes. A frozen window is what a desktop
      environment offers to kill for you.
    * **It forks a process that owns a display.** The child inherits the
      parent's SDL/X11 connection. `_route_worker` now forces the dummy driver
      and drops the inherited display, but not forking at all is strictly safer.

    Measured on `hazard_overlay:drone_hazard_t1`: 0.13 s off, 0.68 s on for an
    easy seed, and ~25 s observed when the planner hits its timeout.

    The cost is honest and small: an uncertified overlay level has no validated
    route, so `hazard_outcome` is always "safe" and it ships no demo. For LOOKING
    at a map that changes nothing -- and `is_map_solvable` still runs, so the
    panel's SOLVABLE/UNSOLVABLE readout is unaffected. Pass
    ``--generator-certify`` to browse exactly what a dataset build would emit.
    """
    if certify_routes:
        return
    if hasattr(map_obj, "REQUIRE_ROUTE_CERTIFICATE"):
        map_obj.REQUIRE_ROUTE_CERTIFICATE = False


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

    def __init__(self, start_seed: int = 10000, check_solvable: bool = True,
                 name_filter: str = "", certify_routes: bool = False):
        """``name_filter`` restricts the cursor to matching ``gen_type:preset``.

        Without it the cursor walks all ~155 combos in alphabetical order, so
        reaching one specific generator means pressing *next* dozens of times --
        which makes the browser useless for the common case of "show me THIS
        generator". Matching is case-insensitive substring on ``type:preset``,
        so ``mine_maze`` selects all its presets and ``mine_maze:heavy`` one.
        """
        self.combos = all_generator_combos()
        if name_filter:
            needle = name_filter.strip().lower()
            filtered = [(g, p) for (g, p) in self.combos
                        if needle in f"{g}:{p}".lower()]
            if not filtered:
                raise RuntimeError(
                    f"No generator matches {name_filter!r}. Available types: "
                    + ", ".join(sorted({g for g, _ in self.combos})))
            self.combos = filtered
        if not self.combos:
            raise RuntimeError("No generators registered in GENERATOR_PRESETS")
        self.name_filter = (name_filter or "").strip().lower()
        self.index = 0
        self.seed = int(start_seed)
        self.check_solvable = check_solvable
        #: Route certification is OFF for interactive browsing -- see
        #: `_apply_interactive_budget`.
        self.certify_routes = bool(certify_routes)
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

    def next_type(self) -> None:
        """Jump to the first preset of the NEXT generator type.

        Stepping one preset at a time is fine inside a generator and useless
        between them: `single_chamber` alone has a dozen presets, so reaching the
        next generator means pressing *next* until it happens. This is the
        coarse axis, and it is what makes the browser usable for "show me the
        drone bucket's generator" without restarting with a `--generator-filter`.
        """
        self._jump_type(+1)

    def prev_type(self) -> None:
        self._jump_type(-1)

    def _jump_type(self, step: int) -> None:
        types = self.types()
        if len(types) <= 1:
            return
        cur = types.index(self.gen_type)
        want = types[(cur + step) % len(types)]
        self.index = next(i for i, (g, _p) in enumerate(self.combos) if g == want)

    def types(self) -> List[str]:
        """Generator types present in the (possibly filtered) cursor, in order."""
        seen: List[str] = []
        for gen_type, _preset in self.combos:
            if gen_type not in seen:
                seen.append(gen_type)
        return seen

    def set_filter(self, name_filter: str) -> bool:
        """Re-filter the cursor in place. Returns False and keeps the old cursor
        if nothing matches -- a filter typo should not empty the browser.

        Live re-filtering is the point: the CLI flag can only be set before the
        window opens, so validating a second generator meant restarting.
        """
        combos = all_generator_combos()
        needle = (name_filter or "").strip().lower()
        if needle:
            combos = [(g, p) for (g, p) in combos if needle in f"{g}:{p}".lower()]
        if not combos:
            return False
        self.combos = combos
        self.name_filter = needle
        self.index = 0
        return True

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
            _apply_interactive_budget(map_obj, self.certify_routes)
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
        types = self.types()
        lines = [
            f"{self.gen_type}:{self.preset}",
            f"combo {self.index + 1}/{len(self.combos)}"
            f"   type {types.index(self.gen_type) + 1}/{len(types)}",
            f"seed {self.seed}   status: {self.status}",
        ]
        if self.name_filter:
            lines.append(f"filter: {self.name_filter[:30]}")
        if not self.certify_routes:
            # Say so explicitly: an uncertified overlay level LOOKS identical to
            # a certified one and ships no validated route.
            lines.append("route certificate: OFF (--generator-certify)")
        if self.last_error:
            lines.append(self.last_error[:38])
        if self.scan_summary:
            lines.append(self.scan_summary)
        return lines


class BucketBrowser:
    """Cursor over ``BUCKET_CATALOG`` entries, generating each level LIVE.

    The generator browser walks the ``(generator_type, preset)`` space, which is
    the wrong axis for validating a curriculum: what training actually samples is
    a ``(bucket, tier)`` cell, and a cell is a WEIGHTED SET of generator presets,
    not one preset. Eyeballing "what does DRONE T3 look like" through the
    generator browser means knowing which presets that cell happens to draw and
    stepping to each by hand.

    This also does not need a packed dataset, which is the difference that
    matters when a bucket has just been added: `BenchmarkSource` can only browse
    levels that have already been generated to disk (~13 s each for the hazard
    buckets), so the fastest possible loop for "did my new bucket come out right"
    was a multi-hour dataset build. Here the cell is generated on the spot.
    """

    UNKNOWN = "?"
    SOLVABLE = "SOLVABLE"
    UNSOLVABLE = "UNSOLVABLE"
    ERROR = "ERROR"

    def __init__(self, start_seed: int = 10000, check_solvable: bool = True,
                 bucket_filter: str = "", tier: Optional[int] = None,
                 include_empty: bool = False, certify_routes: bool = False):
        from ..map_generation.generator_configs import BUCKET_CATALOG

        self.catalog = BUCKET_CATALOG
        self.include_empty = bool(include_empty)
        self.bucket_filter = (bucket_filter or "").strip().upper()
        self.tier_filter = int(tier) if tier else None
        self.cells: List[Tuple[str, int]] = []
        self._rebuild_cells()
        if not self.cells:
            raise RuntimeError(
                f"No bucket cells match filter={bucket_filter!r} tier={tier!r}. "
                "Available: " + ", ".join(self.all_bucket_names()))
        self.index = 0
        #: Which entry WITHIN the current cell to draw. A cell is a weighted set
        #: and the weights are what training samples by, but for eyeballing you
        #: want to see each entry deliberately rather than by luck.
        self.entry_index = 0
        self.seed = int(start_seed)
        self.check_solvable = check_solvable
        self.certify_routes = bool(certify_routes)
        self.current_map = None
        self.status = self.UNKNOWN
        self.last_error: Optional[str] = None
        self.scan_summary: Optional[str] = None

    # -- structure ---------------------------------------------------------

    def all_bucket_names(self) -> List[str]:
        return [b for b, tiers in self.catalog.items() if tiers or self.include_empty]

    def _rebuild_cells(self) -> None:
        cells: List[Tuple[str, int]] = []
        for bucket, tiers in self.catalog.items():
            if self.bucket_filter and self.bucket_filter not in bucket.upper():
                continue
            if not tiers and not self.include_empty:
                # SWITCH_PUZZLE / LAUNCH_PAD are declared but empty until their
                # phase lands; browsing them would just raise on every step.
                continue
            for tier in sorted(tiers):
                if self.tier_filter and tier != self.tier_filter:
                    continue
                cells.append((bucket, tier))
        self.cells = cells

    def set_filter(self, bucket_filter: str) -> bool:
        """Re-filter in place; False (and no change) when nothing matches."""
        prev_filter, prev_cells = self.bucket_filter, self.cells
        self.bucket_filter = (bucket_filter or "").strip().upper()
        self._rebuild_cells()
        if not self.cells:
            self.bucket_filter, self.cells = prev_filter, prev_cells
            return False
        self.index = 0
        self.entry_index = 0
        return True

    @property
    def bucket(self) -> str:
        return self.cells[self.index][0]

    @property
    def tier(self) -> int:
        return self.cells[self.index][1]

    @property
    def entries(self) -> List[Tuple[str, str, int]]:
        return list(self.catalog[self.bucket][self.tier])

    @property
    def entry(self) -> Tuple[str, str, int]:
        entries = self.entries
        return entries[self.entry_index % len(entries)]

    # -- navigation --------------------------------------------------------

    def next(self) -> None:
        """Next ENTRY, rolling into the next cell at the end of this one."""
        if self.entry_index + 1 < len(self.entries):
            self.entry_index += 1
        else:
            self.next_cell()

    def prev(self) -> None:
        if self.entry_index > 0:
            self.entry_index -= 1
        else:
            self.prev_cell()
            self.entry_index = len(self.entries) - 1

    def next_cell(self) -> None:
        self.index = (self.index + 1) % len(self.cells)
        self.entry_index = 0

    def prev_cell(self) -> None:
        self.index = (self.index - 1) % len(self.cells)
        self.entry_index = 0

    def next_bucket(self) -> None:
        self._jump_bucket(+1)

    def prev_bucket(self) -> None:
        self._jump_bucket(-1)

    def _jump_bucket(self, step: int) -> None:
        names = [b for b, _t in self.cells]
        ordered: List[str] = []
        for b in names:
            if b not in ordered:
                ordered.append(b)
        if len(ordered) <= 1:
            return
        want = ordered[(ordered.index(self.bucket) + step) % len(ordered)]
        self.index = next(i for i, (b, _t) in enumerate(self.cells) if b == want)
        self.entry_index = 0

    def next_tier(self) -> None:
        self._jump_tier(+1)

    def prev_tier(self) -> None:
        self._jump_tier(-1)

    def _jump_tier(self, step: int) -> None:
        """Move within the CURRENT bucket only. Tier is the difficulty axis, and
        comparing T1 against T4 of the same bucket is the whole point of looking
        -- a T4 that reads easier than T1 is the failure the tier presets exist
        to avoid."""
        same = [i for i, (b, _t) in enumerate(self.cells) if b == self.bucket]
        if len(same) <= 1:
            return
        pos = same.index(self.index)
        self.index = same[(pos + step) % len(same)]
        self.entry_index = 0

    def set_seed(self, seed: int) -> None:
        self.seed = int(seed)

    def bump_seed(self, delta: int = 1) -> None:
        self.seed += delta

    # -- generation --------------------------------------------------------

    def generate(self):
        gen_type, preset, _weight = self.entry
        self.last_error = None
        try:
            map_obj = GeneratorFactory.create_from_preset(
                gen_type, preset, seed=self.seed)
            _apply_interactive_budget(map_obj, self.certify_routes)
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
            except Exception as exc:  # best-effort, same as GeneratorBrowser
                self.status = self.ERROR
                self.last_error = f"solvability: {exc}"
        else:
            self.status = self.UNKNOWN
        return map_obj

    def to_ascii(self) -> str:
        if self.current_map is None:
            return "(no map generated yet)"
        return self.current_map.to_ascii(show_coords=False)

    # -- display -----------------------------------------------------------

    def entity_summary(self) -> str:
        """Non-mine, non-goal entity types in the current map, with counts.

        This is the line you actually read when validating a hazard bucket: a
        DRONE level with no drone, or one carrying a stray turret, is a valid
        level and looks completely normal on screen.
        """
        if self.current_map is None:
            return "entities: -"
        md = list(self.current_map.map_data())
        counts: dict = {}
        i = 1230
        while i + 4 < len(md):
            t = md[i]
            if t in _HAZARD_ENTITY_NAMES:
                counts[t] = counts.get(t, 0) + 1
            i += 5
        if not counts:
            return "entities: none"
        return "entities: " + " ".join(
            f"{_HAZARD_ENTITY_NAMES[t]}x{n}" for t, n in sorted(counts.items()))

    def info_lines(self) -> List[str]:
        gen_type, preset, weight = self.entry
        entries = self.entries
        return [
            f"{self.bucket} T{self.tier}",
            f"{gen_type}:{preset} (w{weight})",
            f"entry {self.entry_index % len(entries) + 1}/{len(entries)}"
            f"   cell {self.index + 1}/{len(self.cells)}",
            f"seed {self.seed}   status: {self.status}",
            self.entity_summary()[:38],
        ] + ([] if self.certify_routes
             else ["route certificate: OFF (--generator-certify)"]) \
          + ([f"filter: {self.bucket_filter[:30]}"] if self.bucket_filter else []) \
          + ([self.last_error[:38]] if self.last_error else []) \
          + ([self.scan_summary] if self.scan_summary else [])

    # -- batch scan --------------------------------------------------------

    def scan_all(self, seed: Optional[int] = None) -> dict:
        """Generate every entry of every in-scope cell once and report.

        The per-cell view answers "is this bucket right"; this answers "is the
        catalog right", which is the question after adding buckets to it.
        """
        scan_seed = self.seed if seed is None else int(seed)
        results: dict = {}
        n_ok = n_bad = n_err = 0
        print("\n" + "=" * 70)
        print(f"BUCKET CATALOG SCAN  (seed={scan_seed}, {len(self.cells)} cells)")
        print("=" * 70)
        for bucket, tier in self.cells:
            for gen_type, preset, _w in self.catalog[bucket][tier]:
                try:
                    m = GeneratorFactory.create_from_preset(
                        gen_type, preset, seed=scan_seed)
                    m.generate(seed=scan_seed)
                    status = (self.SOLVABLE if is_map_solvable(m.map_data())
                              else self.UNSOLVABLE)
                except Exception as exc:
                    status = f"{self.ERROR} ({type(exc).__name__})"
                results[(bucket, tier, gen_type, preset)] = status
                if status == self.SOLVABLE:
                    n_ok += 1
                elif status == self.UNSOLVABLE:
                    n_bad += 1
                    print(f"  UNSOLVABLE  {bucket}/T{tier}  {gen_type}:{preset}")
                else:
                    n_err += 1
                    print(f"  {status:<22}{bucket}/T{tier}  {gen_type}:{preset}")
        print("-" * 70)
        print(f"  solvable={n_ok}  unsolvable={n_bad}  error={n_err}")
        print("=" * 70 + "\n")
        self.scan_summary = f"scan@{scan_seed}: {n_ok} ok / {n_bad} bad / {n_err} err"
        return results


#: Entity types worth naming in the panel's one-line summary -- the hazards and
#: solid bodies a bucket is validated ON. Mines/gold/goals are omitted because
#: every level has them and they would crowd out the signal.
_HAZARD_ENTITY_NAMES = {
    12: "chaingun", 13: "laserdrone", 14: "zap", 15: "chaser", 16: "guard",
    17: "bounce", 18: "rocket", 19: "turret", 20: "thwump", 22: "evilninja",
    23: "duallaser", 25: "deathball", 26: "mini", 27: "eyebat", 28: "shwump",
}
