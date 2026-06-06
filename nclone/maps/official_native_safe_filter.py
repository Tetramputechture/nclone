"""Filter official N++ maps down to a *native-safe* subset.

A map is "native-safe" when every entity it contains is one of:

    * mines           -- TOGGLE_MINE (1), TOGGLE_MINE_TOGGLED (21)
    * gold            -- GOLD (2)            (0+ allowed)
    * exactly one exit door   -- EXIT_DOOR (3)
    * exactly one exit switch -- EXIT_SWITCH (4)
    * the ninja spawn record  -- NINJA (0)   (always present once)

i.e. NO locked doors, regular/trap doors, launch pads, one-ways, drones,
bounce blocks, thwumps, boost/death balls, etc.  These are exactly the
entities the Rust ``nclone-native`` backend simulates faithfully (minus the
locked-door pair, which we intentionally exclude here), so the resulting
subset can be loaded and evaluated on ``--sim-backend native`` without the
native loader silently dropping an unsupported hazard (nclone-native
``loader.rs`` skips unknown entity types with no warning).

Use this to build a held-out generalization benchmark (e.g. from
``maps/official/S*``) that is safe to run on native.

Entity records live at byte offset 1230, 5 bytes each:
``[type, x, y, orientation, mode]`` (see MapLoader.load_map_entities).
The exit-door count is stored at byte 1156.

CLI::

    python -m nclone.maps.official_native_safe_filter \
        --root nclone/maps/official --dirs S S2 SL \
        --out-list /data/official_native_safe.txt --out-json /tmp/report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..constants.entity_types import EntityType

# Entity record layout in the raw map_data byte stream.
_ENTITY_START = 1230
_ENTITY_STRIDE = 5
_EXIT_DOOR_COUNT_OFFSET = 1156

# Mines may appear in either the active or toggled-down state.
_MINE_TYPES = {EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED}
# Everything allowed in a native-safe level (besides the ninja spawn record).
_ALLOWED_TYPES = _MINE_TYPES | {
    EntityType.NINJA,        # spawn record, always present once
    EntityType.GOLD,
    EntityType.EXIT_DOOR,
    EntityType.EXIT_SWITCH,
}


def entity_type_histogram(map_data: List[int]) -> Counter:
    """Return a Counter of entity_type -> count parsed from raw map_data."""
    counts: Counter = Counter()
    idx = _ENTITY_START
    n = len(map_data)
    while idx + 4 < n:
        counts[map_data[idx]] += 1
        idx += _ENTITY_STRIDE
    return counts


def classify_map_data(map_data: List[int]) -> Tuple[bool, str, Counter]:
    """Return ``(is_native_safe, reason, histogram)`` for one map's bytes.

    ``reason`` is ``"ok"`` when safe, else a short human-readable explanation
    of the first disqualifying condition.
    """
    if len(map_data) <= _ENTITY_START + 4:
        return False, "truncated/empty map_data", Counter()

    hist = entity_type_histogram(map_data)

    disallowed = sorted(t for t in hist if t not in _ALLOWED_TYPES)
    if disallowed:
        names = ", ".join(f"{t}({_type_name(t)})x{hist[t]}" for t in disallowed)
        return False, f"disallowed entities: {names}", hist

    n_door = hist.get(EntityType.EXIT_DOOR, 0)
    n_switch = hist.get(EntityType.EXIT_SWITCH, 0)
    if n_door != 1:
        return False, f"exit-door count != 1 (got {n_door})", hist
    if n_switch != 1:
        return False, f"exit-switch count != 1 (got {n_switch})", hist

    declared = map_data[_EXIT_DOOR_COUNT_OFFSET]
    if declared != 1:
        return False, f"header exit_door_count != 1 (got {declared})", hist

    return True, "ok", hist


_TYPE_NAMES = {
    getattr(EntityType, n): n
    for n in dir(EntityType)
    if not n.startswith("_") and isinstance(getattr(EntityType, n), int)
}


def _type_name(t: int) -> str:
    return _TYPE_NAMES.get(t, "UNKNOWN")


def classify_file(path: Path) -> Tuple[bool, str, Counter]:
    """Read a raw official map file and classify it."""
    with open(path, "rb") as fh:
        map_data = [int(b) for b in fh.read()]
    return classify_map_data(map_data)


def scan(
    root: Path, dirs: Optional[List[str]] = None
) -> Dict[str, dict]:
    """Scan official map directories, returning per-file results.

    Returns a dict keyed by relative path string with
    ``{"safe": bool, "reason": str, "hist": {type: count}}``.
    """
    root = Path(root)
    subdirs = [root / d for d in dirs] if dirs else [
        p for p in sorted(root.iterdir()) if p.is_dir()
    ]
    results: Dict[str, dict] = {}
    for sub in subdirs:
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            if not f.is_file():
                continue
            try:
                safe, reason, hist = classify_file(f)
            except Exception as exc:  # noqa: BLE001 - report, don't abort the scan
                safe, reason, hist = False, f"parse error: {exc}", Counter()
            results[str(f.relative_to(root))] = {
                "safe": safe,
                "reason": reason,
                "hist": dict(sorted(hist.items())),
            }
    return results


def _summarize(results: Dict[str, dict]) -> None:
    by_dir: Dict[str, List[bool]] = {}
    for rel, info in results.items():
        d = rel.split("/", 1)[0]
        by_dir.setdefault(d, []).append(info["safe"])
    total_safe = sum(1 for r in results.values() if r["safe"])
    print(f"{'dir':<8} {'safe':>6} / {'total':>6}")
    print("-" * 24)
    for d in sorted(by_dir):
        flags = by_dir[d]
        print(f"{d:<8} {sum(flags):>6} / {len(flags):>6}")
    print("-" * 24)
    print(f"{'ALL':<8} {total_safe:>6} / {len(results):>6}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "official",
        help="Root of official maps (default: nclone/maps/official).",
    )
    ap.add_argument(
        "--dirs",
        nargs="*",
        default=None,
        help="Subdirectories to scan (e.g. S S2 SL). Default: all.",
    )
    ap.add_argument(
        "--out-list",
        type=Path,
        default=None,
        help="Write absolute paths of native-safe maps, one per line.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write the full per-file classification report as JSON.",
    )
    args = ap.parse_args()

    results = scan(args.root, args.dirs)
    _summarize(results)

    if args.out_list is not None:
        safe_paths = [
            str((args.root / rel).resolve())
            for rel, info in sorted(results.items())
            if info["safe"]
        ]
        args.out_list.write_text("\n".join(safe_paths) + "\n")
        print(f"\nWrote {len(safe_paths)} native-safe paths -> {args.out_list}")

    if args.out_json is not None:
        args.out_json.write_text(json.dumps(results, indent=2))
        print(f"Wrote full report -> {args.out_json}")


if __name__ == "__main__":
    main()
