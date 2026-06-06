"""Save / load map data to disk in the formats the repo actually uses.

Formats (inferred from the file extension, or forced via ``fmt=``):

* ``binary`` (``.map`` / ``.bin`` / ``.dat`` / no extension) -- the raw N++ byte
  array, exactly what ``NPlayHeadless.load_map`` reads. A map saved this way is
  directly loadable by the simulator and the RL env. ``map_data()`` already emits
  ints in the 0-255 byte range, so this is loss-less and faithful.
* ``json`` (``.json``) -- ``{"map_data": [...], "metadata": {...}}``; human-readable
  and easy to diff / inspect.
* ``pickle`` (``.pkl`` / ``.pickle``) -- a dict shaped like a
  ``generate_benchmark_dataset.py`` record (``map_data`` + ``secondary`` +
  ``metadata``), so a hand-built / composed map can be dropped into the dataset /
  loader path.

``load_map_data`` reverses any of them back to a ``map_data`` list.
"""

from __future__ import annotations

import json
import pickle
from typing import Dict, List, Optional

_BINARY_EXTS = {"map", "bin", "dat", ""}
_JSON_EXTS = {"json"}
_PICKLE_EXTS = {"pkl", "pickle"}


def _infer_fmt(path: str) -> str:
    ext = path.lower().rsplit(".", 1)[-1] if "." in path.rsplit("/", 1)[-1] else ""
    if ext in _JSON_EXTS:
        return "json"
    if ext in _PICKLE_EXTS:
        return "pickle"
    return "binary"


def save_map(map_data: List[int], path: str, fmt: Optional[str] = None,
             metadata: Optional[Dict] = None, secondary: Optional[List[str]] = None,
             **extra) -> str:
    """Write ``map_data`` to ``path``. Returns ``path``.

    ``metadata`` / ``secondary`` / ``extra`` are stored for the json/pickle formats
    (binary carries tiles+entities only). For binary, a value outside 0-255 raises
    (it would not be a valid byte map).
    """
    fmt = fmt or _infer_fmt(path)
    md = [int(v) for v in map_data]
    if fmt == "binary":
        bad = [v for v in md if not 0 <= v < 256]
        if bad:
            raise ValueError(
                f"cannot write binary .map: {len(bad)} value(s) out of byte range, "
                f"e.g. {bad[:3]} (use fmt='json' to keep them)"
            )
        with open(path, "wb") as f:
            f.write(bytes(md))
    elif fmt == "json":
        with open(path, "w") as f:
            json.dump({"map_data": md, "metadata": metadata or {},
                       "secondary": list(secondary or []), **extra}, f)
    elif fmt == "pickle":
        record = {"map_data": md, "secondary": list(secondary or []),
                  "metadata": metadata or {}}
        record.update(extra)
        with open(path, "wb") as f:
            pickle.dump(record, f)
    else:
        raise ValueError(f"unknown fmt {fmt!r} (binary|json|pickle)")
    return path


def load_map_data(path: str, fmt: Optional[str] = None) -> List[int]:
    """Read a saved map back into a ``map_data`` list (any supported format)."""
    fmt = fmt or _infer_fmt(path)
    if fmt == "binary":
        with open(path, "rb") as f:
            return [int(b) for b in f.read()]
    if fmt == "json":
        with open(path) as f:
            obj = json.load(f)
        return obj["map_data"] if isinstance(obj, dict) else list(obj)
    if fmt == "pickle":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj["map_data"] if isinstance(obj, dict) else list(obj)
    raise ValueError(f"unknown fmt {fmt!r} (binary|json|pickle)")


def load_map(path: str, fmt: Optional[str] = None):
    """Load a saved map into a fresh :class:`~nclone.map_generation.map.Map`."""
    from ..map import Map

    return Map.from_map_data(load_map_data(path, fmt))
