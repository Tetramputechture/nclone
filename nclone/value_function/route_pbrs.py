"""Route-tracking PBRS adapter.

Replaces SurfaceGraphPBRS with a route-graph + arc-targets approach:

1. On episode reset, extract the optimal route (list of ArcTargets) by tracing
   the Dijkstra shortest-path tree from the agent's start position.
2. Each physics frame, increment a per-arc frame counter.  The PBRS potential
   is -remaining_frames / total_route_cost, which decreases monotonically as
   the agent executes arcs correctly.
3. When the agent lands near the next arc's launch point, advance the arc index.
4. When the agent deviates significantly (lands on the wrong surface), replan
   from the current position via a fast tree traceback.
5. On switch collection (phase 0 -> 1), perform a full replan using the exit
   goal Dijkstra tree, handling the airborne-at-switch edge case.

Fallback: when no route can be found (unreachable start, disconnected component),
get_potential() returns the raw d_star / d_max scalar — identical to the legacy
SurfaceGraphPBRS behaviour.

This class exposes the same interface as SurfaceGraphPBRS:
  - get_potential(state_dict) -> float
  - get_gradient(state_dict) -> (float, float)  [backward compat stub]

Plus new methods:
  - update(state_dict)          — call once per physics frame
  - reset(x, y, vx, vy, cs)    — call on episode reset
  - get_arc_target_features(state_dict) -> np.ndarray  [10 dims]
  - get_route_metrics() -> dict  [episode-level TensorBoard metrics]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from ..constants.physics_constants import (
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
    NINJA_RADIUS,
    TOGGLE_MINE_RADIUS_TOGGLED,
)
from .surface_graph import SurfaceGraph, find_nearest_point, node_id
from .surface_extraction import SURFACE_FLOOR
from .route_extraction import (
    ArcTarget,
    extract_route,
    extract_and_simplify_route,
    find_nearest_start_node,
    _predict_ballistic_landing,
)

# Phase bonus added when switch is collected (matches SurfaceGraphPBRS constant)
C_PHASE = 0.5

# Pixel threshold for "agent is near a point" checks
_ARC_ADVANCE_THRESHOLD = 6.0    # advance arc if within 6px of next launch
_REPLAN_THRESHOLD = 36.0        # replan if > 36px from expected launch point

# Tile pixel size for feature normalization (24px per tile)
_TILE_PX = 24.0

# At the goal, d* and first-arc cumulative_cost are 0 — still need a positive
# denominator for potential / progress (avoid ZeroDivisionError).
_MIN_ROUTE_NORMALIZER = 1.0

# Tapered mine-kill mask: same parameters used by ``SurfaceGraphPBRS`` so the
# two potential functions match behaviour near deadly mines.
_MINE_KILL_RADIUS: float = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED   # 14 px
_MINE_KILL_TAPER_PX: float = 8.0
_MINE_KILL_RADIUS_SQ: float = _MINE_KILL_RADIUS * _MINE_KILL_RADIUS
_MINE_KILL_TAPER_OUTER_SQ: float = (
    (_MINE_KILL_RADIUS + _MINE_KILL_TAPER_PX) ** 2
)


def _mine_taper_weight(
    x: float,
    y: float,
    deadly_mines: List[Tuple[float, float]],
) -> float:
    """Return the linear taper blend factor ``t in [0, 1]`` (see PBRS doc)."""
    if not deadly_mines:
        return 0.0
    best_d2 = float("inf")
    for mx, my in deadly_mines:
        dx = x - mx
        dy = y - my
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
    if best_d2 >= _MINE_KILL_TAPER_OUTER_SQ:
        return 0.0
    if best_d2 <= _MINE_KILL_RADIUS_SQ:
        return 1.0
    d = math.sqrt(best_d2)
    taper = (_MINE_KILL_RADIUS + _MINE_KILL_TAPER_PX - d) / _MINE_KILL_TAPER_PX
    return max(0.0, min(1.0, float(taper)))


@dataclass
class RouteState:
    """Mutable tracking state for the current episode's route execution."""
    arc_idx: int = 0
    frames_into_current_arc: int = 0
    replan_count: int = 0
    on_route_frames: int = 0
    total_frames: int = 0
    phase_transitions: int = 0


class RoutePBRS:
    """Route-tracking PBRS with frame-counter-based potential.

    Uses the Dijkstra shortest-path tree to plan an explicit route at the start
    of each episode, then tracks progress along that route using a monotonic
    frame counter rather than distance-based estimation.

    Potential range: [-1, 0] for phase 0 (switch not collected),
                     [-1 + C_PHASE, C_PHASE] = [-0.5, 0.5] for phase 1.

    This matches the SurfaceGraphPBRS normalization convention.
    """

    def __init__(
        self,
        graph: SurfaceGraph,
        d_star_switch: np.ndarray,
        d_star_exit: np.ndarray,
        predecessor_switch: np.ndarray,
        predecessor_exit: np.ndarray,
        best_edge_switch: np.ndarray,
        best_edge_exit: np.ndarray,
        d_max_switch: float,
        d_max_exit: float,
        deadly_mines: Optional[List[Tuple[float, float]]] = None,
    ):
        self._graph = graph
        self._d_star_switch = d_star_switch
        self._d_star_exit = d_star_exit
        self._pred_switch = predecessor_switch
        self._pred_exit = predecessor_exit
        self._best_edge_switch = best_edge_switch
        self._best_edge_exit = best_edge_exit
        self._d_max_switch = max(d_max_switch, 1.0)
        self._d_max_exit = max(d_max_exit, 1.0)
        # Currently deadly mine positions (list is updated by the caller when
        # a mine is toggled on/off).  Used by the tapered kill mask below.
        self._deadly_mines: List[Tuple[float, float]] = list(
            deadly_mines or []
        )

        self._route: List[ArcTarget] = []
        self._state = RouteState()
        self._phase: int = 0
        self._total_route_cost: float = self._d_max_switch

    def set_deadly_mines(
        self, deadly_mines: List[Tuple[float, float]]
    ) -> None:
        """Refresh the cached list of currently-deadly mine positions."""
        self._deadly_mines = list(deadly_mines)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        start_x: float,
        start_y: float,
        start_vx: float,
        start_vy: float,
        contact_state: Optional[int],
    ) -> None:
        """Extract initial route from the agent's spawn position.

        Always starts in phase 0 (pre-switch collection).
        """
        self._phase = 0
        self._state = RouteState()
        self._route = self._extract_route_for_phase(
            start_x, start_y, start_vx, start_vy, contact_state, phase=0
        )
        self._total_route_cost = (
            max(self._route[0].cumulative_cost, _MIN_ROUTE_NORMALIZER)
            if self._route
            else self._d_max_switch
        )

    def update(self, state_dict: Dict[str, Any]) -> None:
        """Advance route state one physics frame.

        Call this once per physics frame (before calling get_potential).
        Handles:
          - Phase transition on switch collection.
          - Arc advancement when agent reaches the next launch point.
          - Replanning when agent deviates from expected surface.
        """
        self._state.total_frames += 1

        # Phase transition: switch was just collected
        if state_dict.get("switch_activated", False) and self._phase == 0:
            self._handle_phase_transition(state_dict)
            return

        if not self._route:
            return

        self._state.frames_into_current_arc += 1

        on_surface = state_dict.get("on_surface", False)
        if not on_surface:
            return

        x = float(state_dict.get("x", state_dict.get("player_x", 0.0)))
        y = float(state_dict.get("y", state_dict.get("player_y", 0.0)))

        # Arc advancement: landed near the next arc's launch point
        next_idx = self._state.arc_idx + 1
        if next_idx < len(self._route):
            next_arc = self._route[next_idx]
            if _dist(x, y, next_arc.launch_x, next_arc.launch_y) < _ARC_ADVANCE_THRESHOLD:
                self._state.arc_idx = next_idx
                self._state.frames_into_current_arc = 0
                self._state.on_route_frames += 1
                return

        # Replanning: on surface but far from expected launch point
        cur_arc = self._route[self._state.arc_idx]
        if _dist(x, y, cur_arc.launch_x, cur_arc.launch_y) > _REPLAN_THRESHOLD:
            vx = float(state_dict.get("vx", state_dict.get("player_xspeed", 0.0)))
            vy = float(state_dict.get("vy", state_dict.get("player_yspeed", 0.0)))
            cs = state_dict.get("contact_state", None)
            self._replan(x, y, vx, vy, cs)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get_potential(self, state_dict: Dict[str, Any]) -> float:
        """Compute PBRS potential for the current state.

        Frame-counter-based for on-route states; raw d_star fallback when
        no route is available (unreachable start or disconnected component).

        Returns value in [-1, C_PHASE].
        """
        if not self._route:
            return self._fallback_potential(state_dict)

        arc = self._route[self._state.arc_idx]
        frames_remaining = max(
            0.0,
            arc.arc_cost_frames - self._state.frames_into_current_arc,
        )
        total_remaining = frames_remaining + arc.cumulative_cost
        denom = max(self._total_route_cost, _MIN_ROUTE_NORMALIZER)
        phi = -total_remaining / denom
        # Apply tapered mine kill mask so PBRS smoothly repels from deadly
        # mines rather than exposing a hard step at the kill radius.
        x = float(state_dict.get("x", state_dict.get("player_x", 0.0)))
        y = float(state_dict.get("y", state_dict.get("player_y", 0.0)))
        taper = _mine_taper_weight(x, y, self._deadly_mines)
        if taper >= 1.0:
            phi = -1.0
        elif taper > 0.0:
            phi = taper * -1.0 + (1.0 - taper) * max(phi, -1.0)
        if self._phase == 1:
            phi += C_PHASE
        return float(max(-1.0, min(C_PHASE, phi)))

    def get_arc_target_features(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Compute the 10-dim arc target observation features.

        Feature layout (REACHABILITY_FEATURES_DIM = 10):
          0: dx_launch  - relative x to launch point (~4 tiles range, clipped ±1)
          1: dy_launch  - relative y to launch point (~4 tiles range, clipped ±1)
          2: vx_error   - (arc.launch_vx - vx) / MAX_HOR_SPEED — "build speed" signal
          3: dx_land    - relative x to landing point (~8 tiles range, clipped ±1)
          4: dy_land    - relative y to landing point (~8 tiles range, clipped ±1)
          5: jump_hold  - arc.jump_hold_frames / MAX_JUMP_DURATION  [0, 1]
          6: req_speed  - abs(arc.launch_vx) / MAX_HOR_SPEED  [0, 1]
          7: progress   - route completion fraction [0, 1]
          8: phase      - 1.0 if switch activated, else 0.0
          9: spare      - 0.0 (reserved for future use)

        Returns zeros when route is empty (fallback / unreachable).
        """
        if not self._route:
            return np.zeros(10, dtype=np.float32)

        arc = self._route[self._state.arc_idx]
        x = float(state_dict.get("x", state_dict.get("player_x", 0.0)))
        y = float(state_dict.get("y", state_dict.get("player_y", 0.0)))
        vx = float(state_dict.get("vx", state_dict.get("player_xspeed", 0.0)))

        dx_launch = (arc.launch_x - x) / (_TILE_PX * 4.0)
        dy_launch = (arc.launch_y - y) / (_TILE_PX * 4.0)
        vx_error = (arc.launch_vx - vx) / MAX_HOR_SPEED
        dx_land = (arc.landing_x - x) / (_TILE_PX * 8.0)
        dy_land = (arc.landing_y - y) / (_TILE_PX * 8.0)
        jump_hold_norm = arc.jump_hold_frames / max(MAX_JUMP_DURATION, 1)
        req_speed = abs(arc.launch_vx) / MAX_HOR_SPEED
        denom = max(self._total_route_cost, _MIN_ROUTE_NORMALIZER)
        progress = 1.0 - arc.cumulative_cost / denom
        phase = 1.0 if self._phase == 1 else 0.0

        return np.array(
            [
                float(np.clip(dx_launch, -1.0, 1.0)),
                float(np.clip(dy_launch, -1.0, 1.0)),
                float(np.clip(vx_error, -1.0, 1.0)),
                float(np.clip(dx_land, -1.0, 1.0)),
                float(np.clip(dy_land, -1.0, 1.0)),
                float(np.clip(jump_hold_norm, 0.0, 1.0)),
                float(np.clip(req_speed, 0.0, 1.0)),
                float(np.clip(progress, 0.0, 1.0)),
                phase,
                0.0,  # spare
            ],
            dtype=np.float32,
        )

    def get_gradient(self, state_dict: Dict[str, Any]) -> Tuple[float, float]:
        """Backward-compat stub returning direction toward current arc's launch point.

        Used by ReachabilityMixin as a fallback when RoutePBRS is active but
        arc target features are not yet wired (or for legacy code paths).
        """
        if not self._route:
            return (0.0, 0.0)
        arc = self._route[self._state.arc_idx]
        x = float(state_dict.get("x", state_dict.get("player_x", 0.0)))
        y = float(state_dict.get("y", state_dict.get("player_y", 0.0)))
        dx = arc.launch_x - x
        dy = arc.launch_y - y
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 1e-6:
            return (0.0, 0.0)
        return (dx / mag, dy / mag)

    def get_route_metrics(self) -> Dict[str, float]:
        """Episode-level metrics for TensorBoard logging."""
        return {
            "arc_idx": float(self._state.arc_idx),
            "num_arcs": float(len(self._route)),
            "replans": float(self._state.replan_count),
            "on_route_fraction": (
                self._state.on_route_frames
                / max(self._state.total_frames, 1)
            ),
            "route_empty": float(len(self._route) == 0),
            "phase_transitions": float(self._state.phase_transitions),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_phase_transition(self, state_dict: Dict[str, Any]) -> None:
        """Replan for exit goal on switch collection.

        Handles the airborne-at-switch edge case by predicting the landing
        point before routing.
        """
        self._phase = 1
        self._state.phase_transitions += 1

        x = float(state_dict.get("x", state_dict.get("player_x", 0.0)))
        y = float(state_dict.get("y", state_dict.get("player_y", 0.0)))
        vx = float(state_dict.get("vx", state_dict.get("player_xspeed", 0.0)))
        vy = float(state_dict.get("vy", state_dict.get("player_yspeed", 0.0)))
        cs = state_dict.get("contact_state", None)

        on_surface = state_dict.get("on_surface", False)
        if not on_surface:
            # Airborne: predict landing first, then find nearest node there
            landing = _predict_ballistic_landing(self._graph, x, y, vx, vy)
            if landing is not None:
                lx, ly, lvx = landing
                start = find_nearest_start_node(
                    self._graph, lx, ly, lvx, 0.0, None, self._d_star_exit
                )
            else:
                start = find_nearest_start_node(
                    self._graph, x, y, vx, vy, cs, self._d_star_exit
                )
        else:
            start = find_nearest_start_node(
                self._graph, x, y, vx, vy, cs, self._d_star_exit
            )

        if start is not None:
            self._route = extract_and_simplify_route(
                self._graph, self._d_star_exit,
                self._pred_exit, self._best_edge_exit, start,
            )
        else:
            self._route = []

        self._state.arc_idx = 0
        self._state.frames_into_current_arc = 0
        self._state.replan_count += 1
        self._total_route_cost = (
            max(self._route[0].cumulative_cost, _MIN_ROUTE_NORMALIZER)
            if self._route
            else self._d_max_exit
        )

    def _replan(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        contact_state: Optional[int],
    ) -> None:
        """Replan from agent's current surface position."""
        d_star = self._d_star_exit if self._phase == 1 else self._d_star_switch
        pred = self._pred_exit if self._phase == 1 else self._pred_switch
        best_edge = self._best_edge_exit if self._phase == 1 else self._best_edge_switch
        d_max = self._d_max_exit if self._phase == 1 else self._d_max_switch

        start = find_nearest_start_node(
            self._graph, x, y, vx, vy, contact_state, d_star
        )
        if start is not None:
            self._route = extract_and_simplify_route(
                self._graph, d_star, pred, best_edge, start
            )
        else:
            self._route = []

        self._state.arc_idx = 0
        self._state.frames_into_current_arc = 0
        self._state.replan_count += 1
        self._total_route_cost = (
            max(self._route[0].cumulative_cost, _MIN_ROUTE_NORMALIZER)
            if self._route
            else d_max
        )

    def _fallback_potential(self, state_dict: Dict[str, Any]) -> float:
        """Raw d_star / d_max potential when no route is available.

        Identical semantics to SurfaceGraphPBRS for disconnected/unreachable
        starts.  This preserves the "always has a potential value" guarantee.
        """
        d_star = self._d_star_exit if self._phase == 1 else self._d_star_switch
        d_max = self._d_max_exit if self._phase == 1 else self._d_max_switch

        x = float(state_dict.get("x", state_dict.get("player_x", 0.0)))
        y = float(state_dict.get("y", state_dict.get("player_y", 0.0)))
        vx = float(state_dict.get("vx", state_dict.get("player_xspeed", 0.0)))
        vy = float(state_dict.get("vy", state_dict.get("player_yspeed", 0.0)))
        cs = state_dict.get("contact_state", None)

        start = find_nearest_start_node(self._graph, x, y, vx, vy, cs, d_star)
        if start is not None and np.isfinite(d_star[start]):
            phi = -float(d_star[start]) / d_max
        else:
            phi = -1.0
        taper = _mine_taper_weight(x, y, self._deadly_mines)
        if taper >= 1.0:
            phi = -1.0
        elif taper > 0.0:
            phi = taper * -1.0 + (1.0 - taper) * max(phi, -1.0)
        if self._phase == 1:
            phi += C_PHASE
        return float(max(-1.0, min(C_PHASE, phi)))

    def _extract_route_for_phase(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        contact_state: Optional[int],
        phase: int,
    ) -> List[ArcTarget]:
        """Extract route for the given phase from position (x, y, vx, vy)."""
        d_star = self._d_star_exit if phase == 1 else self._d_star_switch
        pred = self._pred_exit if phase == 1 else self._pred_switch
        best_edge = self._best_edge_exit if phase == 1 else self._best_edge_switch

        start = find_nearest_start_node(
            self._graph, x, y, vx, vy, contact_state, d_star
        )
        if start is None:
            return []
        return extract_and_simplify_route(self._graph, d_star, pred, best_edge, start)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_route_pbrs(
    level_id: str,
    value_function_dir: str,
) -> Optional["RoutePBRS"]:
    """Load a RoutePBRS from a precomputed .npz file.

    Returns ``None`` if the .npz file does not exist or does not contain the
    route-tree arrays (predecessor_switch, predecessor_exit, etc.) — in which
    case the caller should fall back to ``create_two_phase_pbrs``.

    Args:
        level_id: Level identifier (filename without extension).
        value_function_dir: Directory containing ``{level_id}_surface_pbrs.npz``.

    Returns:
        RoutePBRS instance, or None if not available.
    """
    import os
    from .surface_extraction import SurfacePoint
    from .surface_graph import build_graph

    path = os.path.join(value_function_dir, f"{level_id}_surface_pbrs.npz")
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)

    # Check for route-tree arrays (absent in legacy .npz files)
    required = ("predecessor_switch", "predecessor_exit",
                 "best_edge_idx_switch", "best_edge_idx_exit")
    if not all(k in data for k in required):
        return None

    # Reconstruct surface points
    n = len(data["surface_points_x"])
    surface_points = [
        SurfacePoint(
            x=float(data["surface_points_x"][i]),
            y=float(data["surface_points_y"][i]),
            normal_x=float(data["surface_normals_x"][i]),
            normal_y=float(data["surface_normals_y"][i]),
            surface_type=int(data["surface_types"][i]),
            segment_id=int(data["surface_segment_ids"][i]),
            is_edge=bool(data["surface_is_edge"][i]),
        )
        for i in range(n)
    ]

    graph = build_graph(surface_points)

    # Load edges (required for route extraction)
    graph.edges_src = data["edges_src"].astype(np.int32)
    graph.edges_dst = data["edges_dst"].astype(np.int32)
    graph.edges_weight = data["edges_weight"].astype(np.float32)

    if "edges_jump_hold" in data:
        graph.edges_jump_hold = data["edges_jump_hold"].astype(np.int8)
    if "edges_h_input" in data:
        graph.edges_h_input = data["edges_h_input"].astype(np.int8)
    if "edges_h_after" in data:
        graph.edges_h_after = data["edges_h_after"].astype(np.int8)

    # Load d* arrays
    d_star_switch = data["d_star_switch"].astype(np.float32)
    d_star_exit = data["d_star_exit"].astype(np.float32)

    # Load normalization constants
    meta = data["metadata"].item() if "metadata" in data else {}
    d_max_switch = float(meta.get("d_max_switch", _robust_d_max(d_star_switch)))
    d_max_exit = float(meta.get("d_max_exit", _robust_d_max(d_star_exit)))

    # Load route-tree arrays
    predecessor_switch = data["predecessor_switch"].astype(np.int32)
    predecessor_exit = data["predecessor_exit"].astype(np.int32)
    best_edge_idx_switch = data["best_edge_idx_switch"].astype(np.int32)
    best_edge_idx_exit = data["best_edge_idx_exit"].astype(np.int32)

    return RoutePBRS(
        graph=graph,
        d_star_switch=d_star_switch,
        d_star_exit=d_star_exit,
        predecessor_switch=predecessor_switch,
        predecessor_exit=predecessor_exit,
        best_edge_switch=best_edge_idx_switch,
        best_edge_exit=best_edge_idx_exit,
        d_max_switch=d_max_switch,
        d_max_exit=d_max_exit,
    )


def _robust_d_max(arr: np.ndarray, percentile: float = 99.0) -> float:
    """99th-percentile of finite d* values (mirrors precompute_cli._robust_d_max)."""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return 1.0
    return float(np.percentile(finite, percentile))


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)
