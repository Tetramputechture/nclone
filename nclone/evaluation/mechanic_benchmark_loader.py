"""Load mechanic-primary procedural benchmark datasets for local tooling.

Dataset layout matches ``scripts/generate_benchmark_dataset.py`` output
(see ``docs/BENCHMARK_DATASET.md``): ``manifest.json``, per-split
``<split>_metadata.json``, and ``<split>/<BUCKET>/T<tier>/<level_id>.pkl``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _resolve_root(root: str | Path) -> Path:
    p = Path(root).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Benchmark dataset root not found: {p}")
    manifest = p / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Not a mechanic benchmark dataset (missing manifest.json): {p}"
        )
    return p


class MechanicBenchmarkLoader:
    """Ordered level access for mechanic-primary benchmark pickles."""

    def __init__(
        self,
        dataset_root: str | Path,
        splits: Sequence[str] = ("train",),
    ) -> None:
        self.root = _resolve_root(dataset_root)
        self.splits = tuple(splits)
        self._entries: List[Tuple[str, Path, int]] = []
        self._cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._build_index()

    def _build_index(self) -> None:
        for split in self.splits:
            meta_path = self.root / split / f"{split}_metadata.json"
            if not meta_path.is_file():
                continue
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            level_ids: List[int] = meta.get("level_ids") or []
            for lid in level_ids:
                pkl = self._find_pickle(split, int(lid))
                if pkl is None:
                    raise FileNotFoundError(
                        f"Missing pickle for level_id={lid} split={split} under {self.root}"
                    )
                self._entries.append((split, pkl, int(lid)))

    def _find_pickle(self, split: str, level_id: int) -> Optional[Path]:
        split_dir = self.root / split
        matches = list(split_dir.rglob(f"{level_id}.pkl"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return min(matches, key=lambda p: len(str(p)))
        return None

    def __len__(self) -> int:
        return len(self._entries)

    def get_level_ids(self) -> List[int]:
        return [e[2] for e in self._entries]

    def get_level_path(self, index: int) -> Path:
        return self._entries[index][1]

    def load_payload(self, index: int) -> Dict[str, Any]:
        """Return the raw pickle dict for the level at ``index``."""
        split, path, lid = self._entries[index]
        key = (split, lid)
        if key in self._cache:
            return self._cache[key]
        with path.open("rb") as f:
            payload = pickle.load(f)
        self._cache[key] = payload
        return payload

    def get_map_data(self, index: int) -> List[Any]:
        return list(self.load_payload(index)["map_data"])

    @staticmethod
    def overlay_lines(
        payload: Dict[str, Any], index: int, n_levels: int
    ) -> List[str]:
        """Human-readable lines for on-screen HUD (mechanic metadata)."""
        meta = payload.get("metadata") or {}
        bucket = payload.get("bucket", "?")
        tier = payload.get("tier", "?")
        gen = payload.get("generator", "?")
        preset = payload.get("preset", "?")
        lid = payload.get("level_id", "?")
        seed = payload.get("seed", "?")
        split = meta.get("split", "?")
        desc = meta.get("description", "")
        lines = [
            f"[benchmark] {index + 1}/{n_levels}  split={split}",
            f"{bucket}  T{tier}  id={lid}",
            f"{gen} / {preset}",
            f"seed={seed}",
        ]
        if desc:
            short = desc if len(desc) <= 72 else desc[:69] + "..."
            lines.append(short)
        return lines
