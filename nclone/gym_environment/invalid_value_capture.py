"""Diagnostic capture for invalid-value events (OOB graph distance -> inf,
non-finite observation features, etc.).

When the reachability graph / grid produces a non-finite value for a legitimately
placed ninja, that almost always points at one of two bugs worth inspecting
offline: (a) a BAD PROCGEN level (unreachable goal / buried entity), or (b) a
subtle GRAPH/GRID-building logic error (the reachable region is computed wrong).
Either way we want the evidence, not a silently-swallowed inf.

This module saves, per unique (reason, level) the first time it is hit:
  * ``<reason>_<maphash>_<n>.json`` -- the FULL ``map_data`` + parsed entity
    positions + the trigger context (ninja/switch/exit positions, graph bounds,
    the offending distances, step). map_data alone lets you re-render, re-run
    ``is_map_solvable``, and rebuild the graph offline -- the primary artifact.
  * ``<reason>_<maphash>_<n>.png`` -- a ground-truth render, BEST EFFORT (a
    second GL renderer inside a training worker can fail; the JSON is the
    guaranteed artifact and ``scripts/render_invalid_dumps.py`` re-renders any
    dump cleanly offline).

Gating / safety:
  * Enabled when ``NCLONE_INVALID_DUMP_DIR`` is set, else falls back to
    ``$NCLONE_GRAPH_CACHE_DIR/invalid_dumps`` when that is set, else DISABLED.
  * Deduped by (reason, map hash) and capped (``NCLONE_INVALID_DUMP_CAP``,
    default 60) so a recurring event can never flood disk or slow training.
  * Never raises -- a diagnostic must not be able to break a training run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_seen = set()  # {(reason, map_hash)} -- one dump per unique level+reason
_count = 0


def _dump_dir() -> Optional[str]:
    d = os.environ.get("NCLONE_INVALID_DUMP_DIR", "").strip()
    if d and d != "0":
        return d
    cache = os.environ.get("NCLONE_GRAPH_CACHE_DIR", "").strip()
    if cache:
        return os.path.join(cache, "invalid_dumps")
    return None


def _cap() -> int:
    try:
        return int(os.environ.get("NCLONE_INVALID_DUMP_CAP", "60"))
    except ValueError:
        return 60


def capture_invalid_value(
    reason: str,
    map_data: Optional[List[int]],
    context: Dict[str, Any],
    render: Optional[bool] = None,
) -> None:
    """Save a debug dump for an invalid-value event. Deduped, capped, never raises.

    Args:
        reason: short slug for the event (e.g. ``"oob_graph_distance"``).
        map_data: the current level's raw map_data (the primary artifact).
        context: JSON-serializable trigger context (positions, bounds, values).
        render: attempt an in-process ground-truth PNG. Default OFF (None ->
            ``NCLONE_INVALID_DUMP_RENDER=1``) because a GL renderer inside a
            headless training worker is unreliable; the JSON carries map_data so
            ``scripts/render_invalid_dumps.py`` re-renders cleanly offline.
    """
    if render is None:
        render = os.environ.get("NCLONE_INVALID_DUMP_RENDER", "0").strip() == "1"
    global _count
    try:
        d = _dump_dir()
        if not d or map_data is None:
            return
        md = [int(b) for b in map_data]
        h = hashlib.md5(bytes(b & 0xFF for b in md)).hexdigest()[:10]
        key = (reason, h)
        if key in _seen or _count >= _cap():
            return
        _seen.add(key)
        _count += 1

        os.makedirs(d, exist_ok=True)
        try:
            from ..map_generation.map_visualizer import parse_entities_from_map_data
            entities = parse_entities_from_map_data(md)
        except Exception:
            entities = None

        base = os.path.join(d, f"{reason}_{h}_{_count:03d}")
        payload = {
            "reason": reason,
            "context": _jsonable(context),
            "entities": _jsonable(entities),
            "map_hash": h,
            "map_data": md,
            "timestamp": time.time(),
        }
        with open(base + ".json", "w") as f:
            json.dump(payload, f)

        if render:
            try:
                from ..map_generation.compose.visualize import render_png
                render_png(md, base + ".png")
            except Exception as exc:
                logger.debug("invalid-value PNG render failed (%s); JSON saved", exc)

        logger.warning(
            "Captured invalid-value diagnostic [%s] map=%s -> %s.json", reason, h, base
        )
    except Exception as exc:  # diagnostics must never break training
        logger.debug("invalid-value capture failed: %s", exc)


def _jsonable(obj):
    """Best-effort coercion of numpy / tuples to JSON-friendly values."""
    try:
        import numpy as np
    except Exception:
        np = None
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (int, float, str, bool)):
        return obj
    return str(obj)
