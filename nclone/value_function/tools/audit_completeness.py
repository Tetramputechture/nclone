#!/usr/bin/env python3
"""Completeness audit for the surface-graph edge enumeration.

The precompute pipeline sweeps a fixed set of action primitives (run durations,
jump-hold durations, horizontal inputs, post-release inputs) when generating
arc templates from every surface node.  If an expert demonstration requires a
timing outside this grid - e.g. a very tight hold-duration, an off-axis wall
jump, or a chained launch-pad kick - the graph is missing the corresponding
edge and the demo's (src -> dst) transition will not be representable.

This tool compares a precomputed ``.npz`` graph against a directory of expert
``.replay`` demos and reports the fraction of observed surface transitions
that are *absent* from the graph.  The output guides the operator to either
(a) broaden the built-in primitive tables in ``edge_enumeration.py`` or
(b) pass additional templates through the ``--extra-primitives`` JSON flag
(see ``precompute_cli``) so the caching pass can emit them.

Usage::

    python -m nclone.value_function.tools.audit_completeness \
        --npz value_functions/simple-walk_surface_pbrs.npz \
        --demo-dir demos/simple-walk/ \
        --map-path nclone/test_maps/simple-walk.map

Output is a human-readable summary + a JSON sidecar with the per-demo
missing-transition list.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow running from either project root
_NCLONE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_NCLONE_ROOT))

from nclone.value_function.surface_extraction import (  # noqa: E402
    SURFACE_FLOOR,
    SURFACE_WALL,
    extract_surfaces,
)
from nclone.value_function.surface_graph import (  # noqa: E402
    SurfaceGraph,
    build_graph,
    find_nearest_point,
    node_id,
    num_velocity_bins,
    velocity_to_bin,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extra-primitives JSON schema (documented here so precompute_cli can reference
# a single canonical description).  A primitive template is a dict of the form
# ``{"kind": "floor_jump", "h_during": 1, "hold": 18, "h_after": 0}`` where
# ``kind`` is one of {"run", "floor_jump", "wall_jump", "wall_slide_fall",
# "launch_pad"} and the remaining fields are integer action parameters.
# ---------------------------------------------------------------------------
_ALLOWED_KINDS = (
    "run",
    "floor_jump",
    "wall_jump",
    "wall_slide_fall",
    "launch_pad",
)


def load_extra_primitives(path: Optional[str]) -> List[Dict[str, Any]]:
    """Parse and validate an extra-primitives JSON file.

    Returns an empty list when ``path`` is ``None``.  Raises ``ValueError`` on
    schema violations so the caller can fail-fast before precomputation.
    """
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("extra-primitives JSON must be a list of dicts")
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {i} is not a dict")
        kind = entry.get("kind")
        if kind not in _ALLOWED_KINDS:
            raise ValueError(
                f"entry {i} has unknown kind {kind!r}; "
                f"expected one of {_ALLOWED_KINDS}"
            )
    return data


# ---------------------------------------------------------------------------
# Demo -> transition extraction
# ---------------------------------------------------------------------------
@dataclass
class DemoTransition:
    """A single surface -> surface transition observed in a demo."""

    src_pt: int
    src_vel_bin: int
    dst_pt: int
    dst_vel_bin: int
    frames: int
    # Raw state for diagnostics
    src_x: float = 0.0
    src_y: float = 0.0
    dst_x: float = 0.0
    dst_y: float = 0.0


def _nearest_node_from_state(
    graph: SurfaceGraph,
    x: float,
    y: float,
    vx: float,
    vy: float,
    on_wall: bool,
) -> Optional[Tuple[int, int]]:
    """Return ``(point_idx, velocity_bin)`` for the nearest graph node."""
    stype = SURFACE_WALL if on_wall else SURFACE_FLOOR
    pt_idx = find_nearest_point(graph, x, y, stype)
    if pt_idx is None:
        return None
    vel = vy if on_wall else vx
    vb = velocity_to_bin(vel, stype)
    return pt_idx, vb


def _extract_transitions_from_demo(
    demo_states: Sequence[Tuple[float, float, float, float, int]],
    graph: SurfaceGraph,
) -> List[DemoTransition]:
    """Walk a demo's per-frame state list and emit surface -> surface transitions.

    ``demo_states`` is a sequence of ``(x, y, vx, vy, state_code)``.  A
    transition is opened the first time the ninja is on a surface and closed
    the next time it is on a (potentially different) surface.  Purely
    airborne frames in between are ignored by this audit - the point is
    exactly to check whether the graph contains an edge between the two
    surface endpoints.
    """
    transitions: List[DemoTransition] = []
    # Ninja state codes: 0..2 grounded, 5 wall-sliding.
    _GROUND = {0, 1, 2}
    _WALL = {5}

    open_src: Optional[Tuple[int, int, float, float, int]] = None
    for idx, (x, y, vx, vy, state) in enumerate(demo_states):
        on_ground = state in _GROUND
        on_wall = state in _WALL
        if not (on_ground or on_wall):
            continue
        node = _nearest_node_from_state(graph, x, y, vx, vy, on_wall)
        if node is None:
            continue
        if open_src is None:
            open_src = (node[0], node[1], x, y, idx)
            continue
        src_pt, src_vb, src_x, src_y, src_idx = open_src
        if node[0] == src_pt and node[1] == src_vb:
            # Still sitting on the same discrete node - keep the older start.
            open_src = (src_pt, src_vb, src_x, src_y, src_idx)
            continue
        transitions.append(
            DemoTransition(
                src_pt=src_pt,
                src_vel_bin=src_vb,
                dst_pt=node[0],
                dst_vel_bin=node[1],
                frames=idx - src_idx,
                src_x=src_x,
                src_y=src_y,
                dst_x=x,
                dst_y=y,
            )
        )
        open_src = (node[0], node[1], x, y, idx)
    return transitions


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------
@dataclass
class AuditResult:
    total_transitions: int = 0
    covered_transitions: int = 0
    uncovered: List[DemoTransition] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "total": self.total_transitions,
            "covered": self.covered_transitions,
            "coverage_pct": (
                100.0 * self.covered_transitions / self.total_transitions
                if self.total_transitions
                else 0.0
            ),
            "n_uncovered_unique": len({
                (t.src_pt, t.src_vel_bin, t.dst_pt, t.dst_vel_bin)
                for t in self.uncovered
            }),
        }


def _build_edge_set(graph: SurfaceGraph) -> set:
    """Return the ``(src_node_id, dst_node_id)`` set from the graph."""
    if graph.edges_src.size == 0:
        return set()
    return {
        (int(s), int(d))
        for s, d in zip(graph.edges_src.tolist(), graph.edges_dst.tolist())
    }


def audit_coverage(
    graph: SurfaceGraph,
    transitions: Sequence[DemoTransition],
) -> AuditResult:
    """Check which observed transitions are covered by the graph's edges."""
    edges = _build_edge_set(graph)
    result = AuditResult()
    for t in transitions:
        result.total_transitions += 1
        src_id = node_id(t.src_pt, t.src_vel_bin, graph.max_vel_bins)
        dst_id = node_id(t.dst_pt, t.dst_vel_bin, graph.max_vel_bins)
        if (src_id, dst_id) in edges:
            result.covered_transitions += 1
        else:
            result.uncovered.append(t)
    return result


# ---------------------------------------------------------------------------
# Demo loading
# ---------------------------------------------------------------------------
def _iter_demo_state_files(demo_dir: Path) -> List[Path]:
    return sorted(demo_dir.glob("*.npz")) + sorted(demo_dir.glob("*.replay"))


def _load_demo_states_npz(path: Path) -> List[Tuple[float, float, float, float, int]]:
    """Load a simple ``.npz`` demo with per-frame ``(x,y,vx,vy,state)`` arrays."""
    data = np.load(path)
    keys = ("x", "y", "vx", "vy", "state")
    if not all(k in data for k in keys):
        raise ValueError(
            f"demo {path} is missing one of the required keys {keys}"
        )
    return list(
        zip(
            data["x"].astype(float).tolist(),
            data["y"].astype(float).tolist(),
            data["vx"].astype(float).tolist(),
            data["vy"].astype(float).tolist(),
            data["state"].astype(int).tolist(),
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit edge-enumeration completeness against expert demos",
    )
    parser.add_argument(
        "--npz",
        required=True,
        help="Path to precomputed ``*_surface_pbrs.npz`` file",
    )
    parser.add_argument(
        "--demo-dir",
        required=True,
        help="Directory containing per-frame demo ``.npz`` files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for the machine-readable report",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    npz_path = Path(args.npz)
    demo_dir = Path(args.demo_dir)
    if not npz_path.exists():
        logger.error("npz not found: %s", npz_path)
        return 1
    if not demo_dir.exists():
        logger.error("demo dir not found: %s", demo_dir)
        return 1

    data = np.load(npz_path, allow_pickle=True)
    # Reconstruct the surface graph from the saved data
    from nclone.value_function.surface_extraction import SurfacePoint
    surface_points = [
        SurfacePoint(
            x=float(x),
            y=float(y),
            normal_x=float(nx),
            normal_y=float(ny),
            surface_type=int(st),
            segment_id=int(sid),
            is_edge=bool(ie),
        )
        for x, y, nx, ny, st, sid, ie in zip(
            data["surface_x"], data["surface_y"],
            data["surface_nx"], data["surface_ny"],
            data["surface_type"], data["surface_segment"],
            data["surface_is_edge"],
        )
    ]
    graph = build_graph(surface_points)
    graph.edges_src = data["edges_src"].astype(np.int32)
    graph.edges_dst = data["edges_dst"].astype(np.int32)
    graph.edges_weight = data["edges_weight"].astype(np.float32)

    # Load demos
    demo_files = _iter_demo_state_files(demo_dir)
    if not demo_files:
        logger.error("no demo files found in %s", demo_dir)
        return 1

    all_transitions: List[DemoTransition] = []
    for f in demo_files:
        try:
            states = _load_demo_states_npz(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping %s: %s", f.name, exc)
            continue
        t = _extract_transitions_from_demo(states, graph)
        logger.info("%s: %d transitions", f.name, len(t))
        all_transitions.extend(t)

    result = audit_coverage(graph, all_transitions)
    summary = result.summary()

    print("Completeness audit results")
    print(f"  Total transitions:        {summary['total']}")
    print(f"  Covered transitions:      {summary['covered']}")
    print(f"  Coverage %:               {summary['coverage_pct']:.2f}%")
    print(f"  Unique uncovered pairs:   {summary['n_uncovered_unique']}")

    if args.output:
        payload = {
            "npz": str(npz_path),
            "demo_dir": str(demo_dir),
            "summary": summary,
            "uncovered": [
                {
                    "src_pt": t.src_pt,
                    "src_vel_bin": t.src_vel_bin,
                    "dst_pt": t.dst_pt,
                    "dst_vel_bin": t.dst_vel_bin,
                    "frames": t.frames,
                    "src_xy": [t.src_x, t.src_y],
                    "dst_xy": [t.dst_x, t.dst_y],
                }
                for t in result.uncovered
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  Wrote JSON report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
