"""Diagnostic capture for PRECISION-bucket mine deaths.

The PRECISION bucket is an execution wall (v15 @4M: p(solve)=0.06, 0.81 of
episodes end in a mine death). PRECISION levels require precise platforming
jumps across gaps that end in a mine if the ninja falls short or long; none of
them need trick primitives. To fix the execution we first need to know which
failure mode dominates:

  * UNDERSHOOT  -- jumped too weakly / too late, fell into the gap short of the
    far platform (descending, death_x well short of the landing edge).
  * OVERSHOOT / PREMATURE -- jumped too early or too hard and sailed past the
    safe landing into a mine beyond it.
  * TIMING -- right trajectory shape but the wrong jump-hold duration (released
    too early -> short hop, or held into a bad apex).

This module saves, per PRECISION mine death (capped + per-level capped so a
single sticky level can't flood disk), the FULL ``map_data`` plus the last-N
frames of ninja telemetry leading into the death and a small derived-triage
summary. ``map_data`` lets ``scripts/render_precision_dumps.py`` re-render the
level and overlay the trajectory offline against the true platform/mine
geometry, which is what actually separates undershoot from overshoot.

Gating / safety (mirrors ``invalid_value_capture``):
  * Enabled when ``NCLONE_PRECISION_DUMP_DIR`` is set, else falls back to
    ``$NCLONE_GRAPH_CACHE_DIR/precision_dumps`` when that is set, else DISABLED.
  * Total cap ``NCLONE_PRECISION_DUMP_CAP`` (default 200) and per-level cap
    ``NCLONE_PRECISION_DUMP_PER_LEVEL`` (default 8) so a recurring death can
    neither flood disk nor bias the sample toward one sticky level.
  * Never raises -- a diagnostic must not be able to break a training run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_count = 0
_per_level: Dict[str, int] = {}  # map_hash -> dumps written for that level


def _dump_dir() -> Optional[str]:
    d = os.environ.get("NCLONE_PRECISION_DUMP_DIR", "").strip()
    if d and d != "0":
        return d
    cache = os.environ.get("NCLONE_GRAPH_CACHE_DIR", "").strip()
    if cache:
        return os.path.join(cache, "precision_dumps")
    return None


def is_enabled() -> bool:
    """True when captures would be written -- cheap gate for the hot path."""
    return _dump_dir() is not None


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _triage(trajectory: Sequence[Sequence[float]], death: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a coarse triage summary from the trajectory + death state.

    Each trajectory frame is ``(x, y, vx, vy, airborn, state, jump_input,
    jump_duration)``. The offline renderer does the geometry-aware classification;
    this only extracts the airborne segment's shape so a quick scan of the JSON
    already hints undershoot vs overshoot vs timing.
    """
    out: Dict[str, Any] = {}
    try:
        # Last ground->air transition before death = the fatal takeoff.
        takeoff_idx = None
        for i in range(len(trajectory) - 1, 0, -1):
            airborn = bool(trajectory[i][4])
            prev_airborn = bool(trajectory[i - 1][4])
            if airborn and not prev_airborn:
                takeoff_idx = i
                break
        death_x = float(death.get("x", trajectory[-1][0] if trajectory else 0.0))
        death_vy = float(death.get("vy", trajectory[-1][3] if trajectory else 0.0))
        out["descending_at_death"] = death_vy > 0.0
        out["death_vy"] = death_vy
        out["death_vx"] = float(death.get("vx", 0.0))
        if takeoff_idx is not None:
            t = trajectory[takeoff_idx]
            out["takeoff_x"] = float(t[0])
            out["takeoff_y"] = float(t[1])
            out["takeoff_vx"] = float(t[2])
            out["takeoff_vy"] = float(t[3])
            out["takeoff_frame_offset"] = int(len(trajectory) - takeoff_idx)
            out["horizontal_travel"] = death_x - float(t[0])
            # jump held = consecutive airborne frames from takeoff with jump_input.
            held = 0
            for k in range(takeoff_idx, len(trajectory)):
                if bool(trajectory[k][4]) and float(trajectory[k][6]) != 0.0:
                    held += 1
                else:
                    break
            out["jump_hold_frames"] = held
            arc = trajectory[takeoff_idx:]
            if arc:
                out["arc_peak_y"] = min(float(f[1]) for f in arc)  # min y = apex
                out["arc_frames"] = len(arc)
        else:
            out["takeoff_x"] = None
            out["note"] = "no ground->air transition in buffer (fell without jumping)"
    except Exception as exc:  # triage is best-effort
        out["triage_error"] = str(exc)
    return out


def capture_precision_failure(
    map_data: Optional[List[int]],
    context: Dict[str, Any],
    trajectory: Sequence[Sequence[float]],
    render: Optional[bool] = None,
) -> None:
    """Save a debug dump for a PRECISION mine death. Capped, never raises.

    Args:
        map_data: the current level's raw map_data (primary artifact; lets the
            offline renderer overlay the trajectory on true geometry).
        context: JSON-serializable trigger context (bucket, level id, death
            position/velocity/cause, switch/exit positions, step).
        trajectory: last-N ninja telemetry frames, oldest-first, each
            ``(x, y, vx, vy, airborn, state, jump_input, jump_duration)``.
        render: attempt an in-process ground-truth PNG. Default OFF (None ->
            ``NCLONE_PRECISION_DUMP_RENDER=1``); the JSON carries map_data so
            ``scripts/render_precision_dumps.py`` re-renders cleanly offline.
    """
    if render is None:
        render = os.environ.get("NCLONE_PRECISION_DUMP_RENDER", "0").strip() == "1"
    global _count
    try:
        d = _dump_dir()
        if not d or map_data is None:
            return
        cap = _int_env("NCLONE_PRECISION_DUMP_CAP", 200)
        per_level_cap = _int_env("NCLONE_PRECISION_DUMP_PER_LEVEL", 8)
        if _count >= cap:
            return
        md = [int(b) for b in map_data]
        h = hashlib.md5(bytes(b & 0xFF for b in md)).hexdigest()[:10]
        if _per_level.get(h, 0) >= per_level_cap:
            return
        _per_level[h] = _per_level.get(h, 0) + 1
        _count += 1

        os.makedirs(d, exist_ok=True)
        traj = [[_num(v) for v in frame] for frame in trajectory]
        base = os.path.join(d, f"precision_mine_{h}_{_count:04d}")
        payload = {
            "reason": "precision_mine_death",
            "context": _jsonable(context),
            "triage": _triage(trajectory, context),
            "trajectory": traj,
            "trajectory_fields": [
                "x", "y", "vx", "vy", "airborn", "state",
                "jump_input", "jump_duration",
            ],
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
                logger.debug("precision-dump PNG render failed (%s); JSON saved", exc)

        logger.info(
            "Captured PRECISION mine-death dump map=%s (%d/%d, level %d/%d) -> %s.json",
            h, _count, cap, _per_level[h], per_level_cap, base,
        )
    except Exception as exc:  # diagnostics must never break training
        logger.debug("precision-failure capture failed: %s", exc)


def _num(v):
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except Exception:
        return v


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
