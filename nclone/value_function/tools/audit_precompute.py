#!/usr/bin/env python3
"""Per-stage memory & time audit of the surface-graph PBRS precompute pipeline.

Reproduces the same stages that ``precompute_surface_pbrs`` executes, but wraps
each one with ``tracemalloc`` snapshots and ``psutil`` RSS captures so we can
attribute peak memory and wall time to individual stages across many levels.

The audit is read-only by design: it does not write any ``.npz`` artifact.  It
emits two output files:

1. ``audit_results.json`` — machine-readable per-level, per-stage metrics.
2. ``docs/PRECOMPUTE_AUDIT.md`` — a human-readable table derived from (1).

Usage
-----
    # Audit every registered benchmark level (slow but thorough)
    python -m nclone.value_function.tools.audit_precompute --all

    # Audit a subset
    python -m nclone.value_function.tools.audit_precompute \\
        --levels simple-walk only-jump minefield

    # Single-process for deterministic memory profiling
    python -m nclone.value_function.tools.audit_precompute --all --num-workers 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

# Allow running from either project root
_NCLONE_ROOT = Path(__file__).resolve().parents[3]
_NPPRL_ROOT = _NCLONE_ROOT.parent / "npp-rl"
sys.path.insert(0, str(_NCLONE_ROOT))
if _NPPRL_ROOT.exists():
    sys.path.insert(0, str(_NPPRL_ROOT))

from nclone.value_function.arc_cache import ArcCache
from nclone.value_function.edge_enumeration import enumerate_edges, populate_graph_edges
from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
from nclone.value_function.surface_extraction import (
    compute_mine_boundary_nodes,
    extract_surfaces,
)
from nclone.value_function.surface_graph import (
    _build_reverse_adjacency,
    build_graph,
    compute_cost_to_go_with_tree,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-stage measurement
# ---------------------------------------------------------------------------
@dataclass
class StageMetric:
    """Wall time + memory deltas captured around a single pipeline stage."""

    name: str
    seconds: float = 0.0
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    rss_peak_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LevelAudit:
    """Complete per-level audit record."""

    name: str
    map_path: str
    status: str = "ok"
    error: Optional[str] = None
    total_seconds: float = 0.0
    process_peak_rss_mb: float = 0.0
    stages: List[StageMetric] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class _RSSPoller:
    """Background thread that samples RSS at a fixed interval.

    Lightweight alternative to ``tracemalloc`` — adds no instrumentation
    overhead to Python bytecode, so it does not distort the timings of heavy
    pure-Python stages such as ``ArcCache.build``.
    """

    def __init__(self, interval_s: float = 0.1):
        self._proc = psutil.Process()
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.peak_rss_mb: float = 0.0

    def _sample_once(self) -> float:
        rss = self._proc.memory_info().rss / (1024 * 1024)
        with self._lock:
            if rss > self.peak_rss_mb:
                self.peak_rss_mb = rss
        return rss

    def start(self) -> None:
        self._stop.clear()
        self.peak_rss_mb = 0.0
        self._sample_once()

        def _loop() -> None:
            while not self._stop.is_set():
                self._sample_once()
                self._stop.wait(self._interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def reset(self) -> None:
        with self._lock:
            self.peak_rss_mb = 0.0
        self._sample_once()

    def current(self) -> float:
        return self._sample_once()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


class _StageTimer:
    """Context manager that captures stage wall time + peak RSS."""

    def __init__(self, audit: LevelAudit, name: str, poller: _RSSPoller):
        self.audit = audit
        self.metric = StageMetric(name=name)
        self._poller = poller

    def __enter__(self) -> StageMetric:
        self.metric.rss_before_mb = self._poller.current()
        self._poller.reset()
        self._t0 = time.perf_counter()
        return self.metric

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.metric.seconds = time.perf_counter() - self._t0
        self.metric.rss_after_mb = self._poller.current()
        self.metric.rss_peak_mb = max(
            self._poller.peak_rss_mb,
            self.metric.rss_before_mb,
            self.metric.rss_after_mb,
        )
        self.audit.stages.append(self.metric)


def _mb(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Single-level audit driver
# ---------------------------------------------------------------------------
def audit_level(
    name: str,
    map_path: str,
    num_workers: int = 1,
    verbose: bool = False,
    arc_waypoint_interval: int = 1,
    max_arc_frames: Optional[int] = None,
    collect_waypoints: bool = True,
) -> LevelAudit:
    """Run the full surface-graph PBRS pipeline with per-stage instrumentation.

    Mirrors ``precompute_surface_pbrs`` exactly but discards outputs; we only
    keep the metrics.

    The ``arc_waypoint_interval``, ``max_arc_frames`` and ``collect_waypoints``
    arguments forward the same memory knobs exposed by ``precompute_cli``, so
    the audit can quantify their impact on peak RSS and arc-cache coverage.
    """
    audit = LevelAudit(name=name, map_path=map_path)
    audit.summary["knobs"] = {
        "num_workers": num_workers,
        "arc_waypoint_interval": arc_waypoint_interval,
        "max_arc_frames": max_arc_frames,
        "collect_waypoints": collect_waypoints,
    }
    poller = _RSSPoller(interval_s=0.1)
    poller.start()
    t_start = time.perf_counter()

    try:
        # ------------------------------------------------------------------
        # Stage 1: Load map + entities
        # ------------------------------------------------------------------
        with _StageTimer(audit, "load_map", poller) as m:
            with open(map_path, "rb") as f:
                raw = f.read()
            map_data = [int(b) for b in raw]
            physics = PhysicsStepWrapper(map_path=map_path)
            switch_pos = physics.get_goal_position("switch")
            exit_pos = physics.get_goal_position("exit")

            mine_entries = []
            for e in physics.entity_dic.get(21, []):
                mine_entries.append((e.xpos, e.ypos, 21, int(e.state)))
            for e in physics.entity_dic.get(1, []):
                mine_entries.append((e.xpos, e.ypos, 1, int(e.state)))

            if mine_entries:
                mine_positions_list = [(float(m2[0]), float(m2[1])) for m2 in mine_entries]
                mine_positions = np.array(
                    [[m2[0], m2[1]] for m2 in mine_entries], dtype=np.float32
                )
            else:
                mine_positions_list = []
                mine_positions = np.empty((0, 2), dtype=np.float32)

            n_tiles = len(physics.tile_dic)
            m.extra.update(
                map_bytes=len(raw),
                n_tiles=n_tiles,
                n_mines=len(mine_entries),
                has_switch=switch_pos is not None,
                has_exit=exit_pos is not None,
            )

        # ------------------------------------------------------------------
        # Stage 2: Extract surfaces + mine boundary nodes
        # ------------------------------------------------------------------
        with _StageTimer(audit, "extract_surfaces", poller) as m:
            surface_points = extract_surfaces(
                physics.tile_dic, mine_positions=mine_positions_list
            )
            mine_xs = mine_positions[:, 0] if len(mine_positions) else np.empty(0, dtype=np.float32)
            mine_ys = mine_positions[:, 1] if len(mine_positions) else np.empty(0, dtype=np.float32)
            mine_init_states = np.array(
                [e[3] for e in mine_entries], dtype=np.uint8
            ) if mine_entries else np.empty(0, dtype=np.uint8)
            always_deadly = [
                (float(mine_xs[i]), float(mine_ys[i]))
                for i in range(len(mine_xs))
                if mine_init_states[i] == 0
            ]
            mine_boundary_nodes = compute_mine_boundary_nodes(
                surface_points, always_deadly
            )
            n_floor = sum(1 for p in surface_points if p.surface_type == 0)
            n_wall = sum(1 for p in surface_points if p.surface_type == 1)
            n_edge = sum(1 for p in surface_points if p.is_edge)
            m.extra.update(
                n_surface_points=len(surface_points),
                n_floor=n_floor,
                n_wall=n_wall,
                n_edge=n_edge,
                n_mine_boundary_pairs=int(np.sum(mine_boundary_nodes >= 0)),
            )

        # ------------------------------------------------------------------
        # Stage 3: Build graph
        # ------------------------------------------------------------------
        with _StageTimer(audit, "build_graph", poller) as m:
            graph = build_graph(surface_points)
            m.extra.update(
                n_graph_nodes=int(graph.n_nodes),
                max_vel_bins=int(graph.max_vel_bins),
                spatial_cells=len(graph.spatial_index),
            )

        # ------------------------------------------------------------------
        # Stage 4: Enumerate edges (heaviest stage)
        # ------------------------------------------------------------------
        with _StageTimer(audit, "enumerate_edges", poller) as m:
            enumeration_stats: dict = {}
            enum_kwargs: Dict[str, Any] = dict(
                graph=graph,
                map_data=map_data,
                num_workers=num_workers,
                collect_waypoints=collect_waypoints,
                verbose=verbose,
                mine_positions=mine_positions,
                stats_out=enumeration_stats,
                waypoint_interval=arc_waypoint_interval,
            )
            if max_arc_frames is not None:
                enum_kwargs["max_arc_frames"] = max_arc_frames
            edges, waypoints = enumerate_edges(**enum_kwargs)
            populate_graph_edges(graph, edges)

            m.extra.update(
                n_edges=len(edges),
                n_waypoints=waypoints.n_entries,
                waypoints_per_node=(
                    waypoints.n_entries / max(1, graph.n_nodes)
                ),
                n_edges_at_clip_bin=int(
                    enumeration_stats.get("n_clipped_landings", 0)
                ),
                n_total_landings=int(
                    enumeration_stats.get("n_total_landings", 0)
                ),
                pct_edges_at_clip_bin=round(
                    enumeration_stats.get("pct_clipped", 0.0), 2,
                ),
            )

        # ------------------------------------------------------------------
        # Stage 5: Dijkstra x2
        # ------------------------------------------------------------------
        # Share the reversed adjacency list across both goals so we don't
        # pay the O(|E|) build cost twice.
        rev_adj = _build_reverse_adjacency(graph)

        with _StageTimer(audit, "dijkstra_switch", poller) as m:
            d_star_switch, predecessor_switch, best_edge_switch = (
                compute_cost_to_go_with_tree(
                    graph, switch_pos[0], switch_pos[1], "switch",
                    rev_adj=rev_adj,
                )
            )
            finite = np.isfinite(d_star_switch)
            n_reachable = int(np.sum(finite))
            d_raw = float(np.max(d_star_switch[finite])) if n_reachable else 0.0
            d_p99 = (
                float(np.percentile(d_star_switch[finite], 99.0))
                if n_reachable
                else 0.0
            )
            m.extra.update(
                n_reachable=n_reachable,
                reachable_pct=round(100.0 * n_reachable / max(1, graph.n_nodes), 2),
                d_max_raw=round(d_raw, 2),
                d_max_p99=round(d_p99, 2),
                inflation_ratio=round(d_raw / max(d_p99, 1.0), 2),
                n_tree_predecessors=int(np.sum(predecessor_switch >= 0)),
            )

        with _StageTimer(audit, "dijkstra_exit", poller) as m:
            d_star_exit, predecessor_exit, best_edge_exit = (
                compute_cost_to_go_with_tree(
                    graph, exit_pos[0], exit_pos[1], "exit",
                    rev_adj=rev_adj,
                )
            )
            finite = np.isfinite(d_star_exit)
            n_reachable = int(np.sum(finite))
            d_raw = float(np.max(d_star_exit[finite])) if n_reachable else 0.0
            d_p99 = (
                float(np.percentile(d_star_exit[finite], 99.0))
                if n_reachable
                else 0.0
            )
            m.extra.update(
                n_reachable=n_reachable,
                reachable_pct=round(100.0 * n_reachable / max(1, graph.n_nodes), 2),
                d_max_raw=round(d_raw, 2),
                d_max_p99=round(d_p99, 2),
                inflation_ratio=round(d_raw / max(d_p99, 1.0), 2),
                n_tree_predecessors=int(np.sum(predecessor_exit >= 0)),
            )

        # ------------------------------------------------------------------
        # Stage 6: Build arc caches x2
        # ------------------------------------------------------------------
        with _StageTimer(audit, "arc_cache_build", poller) as m:
            arc_cache_switch = ArcCache()
            arc_cache_switch.build(waypoints, d_star_switch)
            arc_cache_exit = ArcCache()
            arc_cache_exit.build(waypoints, d_star_exit)
            m.extra.update(
                switch_cells=arc_cache_switch.n_entries,
                exit_cells=arc_cache_exit.n_entries,
                total_cells=arc_cache_switch.n_entries + arc_cache_exit.n_entries,
            )

        # ------------------------------------------------------------------
        # Stage 7: Serialize sizes (without actually writing)
        # ------------------------------------------------------------------
        with _StageTimer(audit, "serialize_estimate", poller) as m:
            sw_keys, sw_vals = arc_cache_switch.to_arrays()
            ex_keys, ex_vals = arc_cache_exit.to_arrays()
            approx_bytes = sum(
                a.nbytes
                for a in (
                    graph.edges_src,
                    graph.edges_dst,
                    graph.edges_weight,
                    d_star_switch,
                    d_star_exit,
                    sw_keys,
                    sw_vals,
                    ex_keys,
                    ex_vals,
                )
                if isinstance(a, np.ndarray)
            )
            m.extra.update(raw_array_mb=round(_mb(approx_bytes), 2))

        audit.total_seconds = time.perf_counter() - t_start

        # Process peak RSS across the whole run (max over all stage peaks)
        audit.process_peak_rss_mb = round(
            max((s.rss_peak_mb for s in audit.stages), default=0.0), 2
        )
        audit.summary = dict(
            n_surface_points=len(surface_points),
            n_graph_nodes=int(graph.n_nodes),
            n_edges=int(len(graph.edges_dst)),
            n_waypoints=len(waypoints),
            arc_cells_switch=arc_cache_switch.n_entries,
            arc_cells_exit=arc_cache_exit.n_entries,
            reachable_pct_switch=round(
                100.0 * int(np.sum(np.isfinite(d_star_switch)))
                / max(1, graph.n_nodes),
                2,
            ),
            reachable_pct_exit=round(
                100.0 * int(np.sum(np.isfinite(d_star_exit)))
                / max(1, graph.n_nodes),
                2,
            ),
        )

    except Exception as exc:
        audit.status = "error"
        audit.error = f"{type(exc).__name__}: {exc}"
        logger.exception("Audit failed for %s", name)
    finally:
        poller.stop()

    return audit


# ---------------------------------------------------------------------------
# Multi-level orchestration
# ---------------------------------------------------------------------------
def _collect_benchmark_levels() -> List[Dict[str, str]]:
    """Return registered benchmark levels from npp-rl mechanic_buckets.

    Falls back to the nclone ``test_maps`` directory if npp-rl is unavailable.
    """
    try:
        from npp_rl.benchmarks.mechanic_buckets import BUCKET_LEVELS
        seen = set()
        result = []
        for levels in BUCKET_LEVELS.values():
            for entry in levels:
                if entry["name"] in seen:
                    continue
                seen.add(entry["name"])
                abs_path = _NCLONE_ROOT / entry["path"]
                result.append(dict(name=entry["name"], path=str(abs_path)))
        return result
    except Exception:
        logger.warning("npp-rl benchmarks not importable; scanning test_maps/")
        test_maps = _NCLONE_ROOT / "nclone" / "test_maps"
        return [
            dict(name=p.name, path=str(p))
            for p in sorted(test_maps.glob("*"))
            if p.is_file() and p.suffix == ""
        ]


def _write_markdown_summary(
    audits: List[LevelAudit], output_path: Path, label: str = "baseline"
) -> None:
    """Render the per-level audit results as a compact markdown table."""
    lines = [
        f"# Precompute Audit — {label}",
        "",
        f"Generated by `nclone.value_function.tools.audit_precompute` on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "## Per-level summary",
        "",
        "| Level | Status | Pts | Nodes | Edges | Waypts | SwCells | ExCells | Reach(Sw/Ex %) | Peak RSS (MB) | Wall (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for a in audits:
        if a.status != "ok":
            lines.append(
                f"| {a.name} | ERROR | — | — | — | — | — | — | — | "
                f"{a.process_peak_rss_mb:.1f} | {a.total_seconds:.1f} |"
            )
            continue
        s = a.summary
        lines.append(
            f"| {a.name} | ok | {s.get('n_surface_points', 0)} | "
            f"{s.get('n_graph_nodes', 0)} | {s.get('n_edges', 0)} | "
            f"{s.get('n_waypoints', 0)} | {s.get('arc_cells_switch', 0)} | "
            f"{s.get('arc_cells_exit', 0)} | "
            f"{s.get('reachable_pct_switch', 0):.1f}/"
            f"{s.get('reachable_pct_exit', 0):.1f} | "
            f"{a.process_peak_rss_mb:.1f} | {a.total_seconds:.1f} |"
        )

    # Per-stage breakdown
    lines += ["", "## Per-stage wall time (seconds)", ""]
    stage_names = sorted({st.name for a in audits for st in a.stages})
    header = "| Level |" + "|".join(stage_names) + "|"
    lines.append(header)
    lines.append("|---|" + "|".join("---:" for _ in stage_names) + "|")
    for a in audits:
        if a.status != "ok":
            continue
        by_name = {s.name: s for s in a.stages}
        row = [a.name]
        for n in stage_names:
            row.append(f"{by_name[n].seconds:.2f}" if n in by_name else "—")
        lines.append("| " + " | ".join(row) + " |")

    # Per-stage peak RSS (sampled at 10 Hz)
    lines += ["", "## Per-stage peak RSS (MB, polled)", ""]
    lines.append(header)
    lines.append("|---|" + "|".join("---:" for _ in stage_names) + "|")
    for a in audits:
        if a.status != "ok":
            continue
        by_name = {s.name: s for s in a.stages}
        row = [a.name]
        for n in stage_names:
            row.append(
                f"{by_name[n].rss_peak_mb:.1f}" if n in by_name else "—"
            )
        lines.append("| " + " | ".join(row) + " |")

    # Errors
    errors = [a for a in audits if a.status != "ok"]
    if errors:
        lines += ["", "## Errors", ""]
        for a in errors:
            lines.append(f"- **{a.name}**: `{a.error}`")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-stage memory & time audit of the precompute pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Audit every registered benchmark level",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        metavar="LEVEL",
        help="Audit a subset of level names",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Workers for edge enumeration (default: 1 for deterministic RSS)",
    )
    parser.add_argument(
        "--output-json",
        default=str(_NCLONE_ROOT / "docs" / "audit_results.json"),
        help="Path for machine-readable JSON output",
    )
    parser.add_argument(
        "--output-md",
        default=str(_NCLONE_ROOT / "docs" / "PRECOMPUTE_AUDIT.md"),
        help="Path for human-readable markdown summary",
    )
    parser.add_argument(
        "--label",
        default="baseline",
        help="Label included in markdown output (e.g. 'baseline', 'post-B3')",
    )
    parser.add_argument(
        "--arc-waypoint-interval",
        type=int,
        default=1,
        help=(
            "Sample arc waypoints every N simulated frames (default: 1). "
            "Higher values reduce peak memory of the enumerate_edges stage "
            "at the cost of sparser arc-cache coverage."
        ),
    )
    parser.add_argument(
        "--max-arc-frames",
        type=int,
        default=None,
        help=(
            "Override the max arc duration (frames) used during waypoint "
            "collection; lower values reduce worker waypoint memory. "
            "Leave unset to use the pipeline default."
        ),
    )
    parser.add_argument(
        "--no-waypoints",
        action="store_true",
        help=(
            "Disable waypoint collection entirely (no arc cache); useful for "
            "profiling graph-only / Dijkstra-only cost on very tight budgets."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging from the pipeline itself",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    all_levels = _collect_benchmark_levels()
    if args.levels:
        wanted = set(args.levels)
        picked = [lv for lv in all_levels if lv["name"] in wanted]
        missing = wanted - {lv["name"] for lv in picked}
        if missing:
            print(f"WARN: unknown levels skipped: {sorted(missing)}")
    elif args.all:
        picked = all_levels
    else:
        parser.error("Pass --all or --levels ...")
        return 2  # unreachable

    if not picked:
        print("No levels selected.")
        return 1

    print(f"Auditing {len(picked)} levels with num_workers={args.num_workers}")
    audits: List[LevelAudit] = []
    t_total = time.perf_counter()

    for i, lv in enumerate(picked, 1):
        path = Path(lv["path"])
        if not path.exists():
            print(f"[{i}/{len(picked)}] {lv['name']}  SKIP (missing: {path})")
            audits.append(
                LevelAudit(
                    name=lv["name"],
                    map_path=str(path),
                    status="missing",
                    error=f"Map file not found: {path}",
                )
            )
            continue

        print(f"[{i}/{len(picked)}] {lv['name']}  auditing...")
        audit = audit_level(
            name=lv["name"],
            map_path=str(path),
            num_workers=args.num_workers,
            verbose=args.verbose,
            arc_waypoint_interval=args.arc_waypoint_interval,
            max_arc_frames=args.max_arc_frames,
            collect_waypoints=not args.no_waypoints,
        )
        audits.append(audit)
        stat = audit.status.upper()
        print(
            f"    {stat}  {audit.total_seconds:.1f}s  "
            f"peak {audit.process_peak_rss_mb:.0f} MB  "
            f"nodes={audit.summary.get('n_graph_nodes', 0)}  "
            f"edges={audit.summary.get('n_edges', 0)}  "
            f"wps={audit.summary.get('n_waypoints', 0)}"
        )

    total_wall = time.perf_counter() - t_total

    # JSON output
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": args.label,
        "num_workers": args.num_workers,
        "total_wall_seconds": total_wall,
        "levels": [asdict(a) for a in audits],
    }
    out_json.write_text(json.dumps(payload, indent=2))

    # Markdown summary
    _write_markdown_summary(audits, Path(args.output_md), label=args.label)

    print(f"\nTotal: {total_wall:.1f}s")
    print(f"JSON:     {out_json}")
    print(f"Markdown: {args.output_md}")

    errors = sum(1 for a in audits if a.status != "ok")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
