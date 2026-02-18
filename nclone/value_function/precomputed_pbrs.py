"""Runtime PBRS using precomputed surface-graph cost-to-go.

Loads a precomputed surface graph (surface points, Dijkstra d* arrays,
arc cache) and provides O(1) potential lookups for PBRS at every
physics frame during FiGAR-PPO training.

Potential formulation (from spec Section 9):
    Phase 0 (switch not collected):
        Phi(s) = -d*_switch[node] / d_max_switch
    Phase 1 (switch collected):
        Phi(s) = -d*_exit[node] / d_max_exit + C_PHASE

    Terminal: Phi = 0  (enforced by caller, not this module)

Surface states use nearest-node lookup with velocity interpolation.
Airborne states use the arc cache (O(1)) or a pessimistic surface-below
fallback when no simulator is available for forward simulation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..constants.physics_constants import GRAVITY_FALL
from .surface_extraction import SurfacePoint, SURFACE_FLOOR, SURFACE_WALL
from .surface_graph import (
    build_graph,
    find_nearest_point,
    lookup_cost_interpolated,
    node_id,
)
from .arc_cache import ArcCache

logger = logging.getLogger(__name__)

# Phase bonus added when switch is collected (spec: 0.5)
C_PHASE = 0.5

# Maximum distance (pixels) to accept a surface node as "on surface"
_SURFACE_MATCH_RADIUS = 24.0

# Contact state constants
CONTACT_FLOOR = 0
CONTACT_WALL = 1
CONTACT_AIRBORNE = 2


class SurfaceGraphPBRS:
    """PBRS potential from precomputed surface-graph cost-to-go.

    Handles both phases (switch and exit) from a single precomputed file.

    Attributes:
        graph: The velocity-augmented surface graph.
        d_star_switch: Cost-to-go array to switch.
        d_star_exit: Cost-to-go array to exit.
        d_max_switch: Max finite cost to switch (for normalization).
        d_max_exit: Max finite cost to exit.
        arc_cache_switch: Arc cache for switch phase.
        arc_cache_exit: Arc cache for exit phase.
    """

    def __init__(self, data_path: str) -> None:
        """Load precomputed surface graph data from disk.

        Args:
            data_path: Path to ``{level}_surface_pbrs.npz`` file.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Surface graph PBRS data not found: {data_path}")

        data = np.load(data_path, allow_pickle=True)

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

        # Build graph structure
        self.graph = build_graph(surface_points)

        # Load edges (stored for reference, not needed for lookup)
        if "edges_src" in data:
            self.graph.edges_src = data["edges_src"]
            self.graph.edges_dst = data["edges_dst"]
            self.graph.edges_weight = data["edges_weight"]

        # Load d* arrays
        self.d_star_switch = data["d_star_switch"].astype(np.float32)
        self.d_star_exit = data["d_star_exit"].astype(np.float32)

        # Load normalization constants
        meta = data["metadata"].item() if "metadata" in data else {}
        self.d_max_switch = float(meta.get(
            "d_max_switch", _safe_max(self.d_star_switch)))
        self.d_max_exit = float(meta.get(
            "d_max_exit", _safe_max(self.d_star_exit)))

        # Load arc caches
        if "arc_cache_keys_switch" in data and len(data["arc_cache_keys_switch"]) > 0:
            self.arc_cache_switch = ArcCache.from_arrays(
                data["arc_cache_keys_switch"], data["arc_cache_values_switch"])
        else:
            self.arc_cache_switch = ArcCache()

        if "arc_cache_keys_exit" in data and len(data["arc_cache_keys_exit"]) > 0:
            self.arc_cache_exit = ArcCache.from_arrays(
                data["arc_cache_keys_exit"], data["arc_cache_values_exit"])
        else:
            self.arc_cache_exit = ArcCache()

        # Stats
        self._n_points = n
        self._n_nodes = self.graph.n_nodes

        logger.info(f"Loaded surface graph PBRS from {data_path}")
        logger.info(f"  Surface points: {n:,}")
        logger.info(f"  Graph nodes: {self._n_nodes:,}")
        logger.info(f"  Edges: {len(self.graph.edges_src):,}")
        logger.info(f"  d_max_switch: {self.d_max_switch:.1f} frames")
        logger.info(f"  d_max_exit: {self.d_max_exit:.1f} frames")
        logger.info(f"  Arc cache switch: {self.arc_cache_switch.n_entries:,} cells")
        logger.info(f"  Arc cache exit: {self.arc_cache_exit.n_entries:,} cells")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_potential(self, state: Dict[str, Any]) -> float:
        """Get PBRS potential for a game state.

        Compatible with the existing ``PrecomputedValuePBRS`` /
        ``TwoPhaseValuePBRS`` interface used by the reward calculator.

        Args:
            state: Dictionary with keys:
                - player_x, player_y: Position in pixels.
                - player_xspeed, player_yspeed: Velocity in px/frame.
                - jump_duration_remaining: Normalised [0,1] or frame count.
                - switch_activated: bool (phase indicator).
                - contact_state (optional): CONTACT_FLOOR / WALL / AIRBORNE.

        Returns:
            Potential value.  Range is approximately [-1, C_PHASE].
        """
        x = state["player_x"]
        y = state["player_y"]
        vx = state.get("player_xspeed", 0.0) or 0.0
        vy = state.get("player_yspeed", 0.0) or 0.0
        switch_activated = state.get("switch_activated", False)

        # Select phase-appropriate d* and normalization
        if switch_activated:
            d_star = self.d_star_exit
            d_max = self.d_max_exit
            phi_phase = C_PHASE
            arc_cache = self.arc_cache_exit
        else:
            d_star = self.d_star_switch
            d_max = self.d_max_switch
            phi_phase = 0.0
            arc_cache = self.arc_cache_switch

        if d_max <= 0:
            return phi_phase

        # Determine contact state
        contact = state.get("contact_state", None)

        if contact is None:
            # Heuristic: try surface lookup, fall back to airborne
            cost = self._try_surface_lookup(d_star, x, y, vx, vy)
            if cost is not None:
                phi_state = -cost / d_max
                return max(phi_state, -1.0) + phi_phase
            # Treat as airborne
            contact = CONTACT_AIRBORNE

        if contact == CONTACT_FLOOR:
            cost = lookup_cost_interpolated(
                self.graph, d_star, x, y, vx, SURFACE_FLOOR)
            if not np.isfinite(cost):
                cost = self._pessimistic_fallback(d_star, x, y)
            phi_state = -cost / d_max
            return max(phi_state, -1.0) + phi_phase

        if contact == CONTACT_WALL:
            cost = lookup_cost_interpolated(
                self.graph, d_star, x, y, vy, SURFACE_WALL)
            if not np.isfinite(cost):
                cost = self._pessimistic_fallback(d_star, x, y)
            phi_state = -cost / d_max
            return max(phi_state, -1.0) + phi_phase

        # AIRBORNE
        # 1. Try arc cache
        cached = arc_cache.lookup_neighborhood(x, y, vx, vy)
        if cached is not None:
            phi_state = -cached / d_max
            return max(phi_state, -1.0) + phi_phase

        # 2. Pessimistic fallback (nearest surface below)
        cost = self._pessimistic_fallback(d_star, x, y)
        phi_state = -cost / d_max
        return max(phi_state, -1.0) + phi_phase

    def get_gradient(
        self, state: Dict[str, Any], epsilon: float = 6.0,
    ) -> Tuple[float, float]:
        """Estimate potential gradient via finite differences."""
        phi0 = self.get_potential(state)

        sp_x = dict(state)
        sp_x["player_x"] = state["player_x"] + epsilon
        phi_x = self.get_potential(sp_x)

        sp_y = dict(state)
        sp_y["player_y"] = state["player_y"] + epsilon
        phi_y = self.get_potential(sp_y)

        return (
            (phi_x - phi0) / epsilon,
            (phi_y - phi0) / epsilon,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics for logging."""
        return {
            "n_surface_points": self._n_points,
            "n_graph_nodes": self._n_nodes,
            "n_edges": len(self.graph.edges_src),
            "d_max_switch": self.d_max_switch,
            "d_max_exit": self.d_max_exit,
            "arc_cache_switch_entries": self.arc_cache_switch.n_entries,
            "arc_cache_exit_entries": self.arc_cache_exit.n_entries,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _try_surface_lookup(
        self, d_star: np.ndarray, x: float, y: float,
        vx: float, vy: float,
    ) -> Optional[float]:
        """Try to find a surface node near (x, y) and return its cost.

        Returns None if no surface point is within _SURFACE_MATCH_RADIUS.
        """
        pt_idx = find_nearest_point(self.graph, x, y)
        if pt_idx is None:
            return None
        sp = self.graph.surface_points[pt_idx]
        d2 = (sp.x - x) ** 2 + (sp.y - y) ** 2
        if d2 > _SURFACE_MATCH_RADIUS ** 2:
            return None

        vel = vy if sp.surface_type == SURFACE_WALL else vx
        cost = lookup_cost_interpolated(
            self.graph, d_star, x, y, vel, sp.surface_type)
        if not np.isfinite(cost):
            return None
        return cost

    def _pessimistic_fallback(
        self, d_star: np.ndarray, x: float, y: float,
    ) -> float:
        """Pessimistic cost estimate for airborne states without cache hit.

        Uses the nearest reachable surface point's cost plus an estimated
        fall time based on vertical distance and gravity.
        """
        pt_idx = find_nearest_point(self.graph, x, y)
        if pt_idx is None:
            return float(self.d_max_switch)  # worst case

        sp = self.graph.surface_points[pt_idx]

        # Estimate fall distance (vertical)
        dy = max(0.0, sp.y - y)  # positive = below, in screen coords
        if GRAVITY_FALL > 0:
            fall_frames = int(2.0 * dy / GRAVITY_FALL) if dy > 0 else 0
        else:
            fall_frames = 0

        # Look up surface cost (use zero velocity bin as conservative est.)
        vel_bin = 0  # zero velocity
        nid = node_id(pt_idx, vel_bin, self.graph.max_vel_bins)
        if nid < len(d_star):
            surface_cost = float(d_star[nid])
        else:
            surface_cost = float(np.max(d_star[np.isfinite(d_star)]))

        if not np.isfinite(surface_cost):
            return float(self.d_max_switch)

        return fall_frames + surface_cost


# Backward-compatible alias
PrecomputedValuePBRS = SurfaceGraphPBRS


def _safe_max(arr: np.ndarray) -> float:
    """Max of finite values in an array, or 1.0 if none."""
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if len(finite) > 0 else 1.0
