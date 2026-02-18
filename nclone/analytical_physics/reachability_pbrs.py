"""Runtime PBRS using analytical reachability graph cost-to-go.

Provides O(1) potential lookups compatible with the existing reward
calculator interface.  Uses the velocity-augmented reachability graph
with analytically computed jump edges and backward Dijkstra d* arrays.

Potential formulation:
    Phase 0 (switch not collected):
        Phi(s) = -d*_switch[node] / d_max_switch
    Phase 1 (switch collected):
        Phi(s) = -d*_exit[node] / d_max_exit + C_PHASE

    C_PHASE = 0.5 ensures monotonic increase across phase transition.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from nclone.constants.physics_constants import GRAVITY_FALL
from .reachability_graph import (
    ReachabilityGraph,
    find_nearest_point,
    lookup_cost_interpolated,
    node_id,
)

logger = logging.getLogger(__name__)

# Phase bonus when switch is collected (same as existing PBRS)
C_PHASE = 0.5

# Maximum distance (px) to accept a surface node as "on surface"
_SURFACE_MATCH_RADIUS = 24.0

# Contact state constants (matching existing precomputed_pbrs.py)
CONTACT_FLOOR = 0
CONTACT_WALL = 1
CONTACT_AIRBORNE = 2


class ReachabilityPBRS:
    """PBRS potential from analytically-built reachability graph.

    Compatible with the existing SurfaceGraphPBRS / PrecomputedValuePBRS
    interface used by the reward calculator.

    Attributes:
        graph: The velocity-augmented reachability graph.
        d_star_switch: Cost-to-go array to switch (float32).
        d_star_exit: Cost-to-go array to exit (float32).
        d_max_switch: Max finite switch cost (for normalization).
        d_max_exit: Max finite exit cost.
        mine_state_signature: Hash of mine states for invalidation.
    """

    def __init__(
        self,
        graph: ReachabilityGraph,
        d_star_switch: np.ndarray,
        d_star_exit: np.ndarray,
        d_max_switch: float,
        d_max_exit: float,
    ) -> None:
        self.graph = graph
        self.d_star_switch = d_star_switch
        self.d_star_exit = d_star_exit
        self.d_max_switch = d_max_switch
        self.d_max_exit = d_max_exit
        self.mine_state_signature: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_potential(self, state: Dict[str, Any]) -> float:
        """Get PBRS potential for a game state.

        Args:
            state: Dictionary with keys:
                - player_x, player_y: Position in pixels.
                - player_xspeed, player_yspeed: Velocity in px/frame.
                - switch_activated: bool (phase indicator).
                - contact_state (optional): CONTACT_FLOOR / WALL / AIRBORNE.

        Returns:
            Potential value.  Range ~[-1, C_PHASE].
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
        else:
            d_star = self.d_star_switch
            d_max = self.d_max_switch
            phi_phase = 0.0

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
            contact = CONTACT_AIRBORNE

        if contact == CONTACT_FLOOR:
            cost = lookup_cost_interpolated(
                self.graph,
                d_star,
                x,
                y,
                vx,
                is_floor=True,
            )
            if not np.isfinite(cost):
                cost = self._pessimistic_fallback(d_star, x, y)
            phi_state = -cost / d_max
            return max(phi_state, -1.0) + phi_phase

        if contact == CONTACT_WALL:
            cost = lookup_cost_interpolated(
                self.graph,
                d_star,
                x,
                y,
                vy,
                is_floor=False,
            )
            if not np.isfinite(cost):
                cost = self._pessimistic_fallback(d_star, x, y)
            phi_state = -cost / d_max
            return max(phi_state, -1.0) + phi_phase

        # AIRBORNE: pessimistic fallback (nearest surface below)
        cost = self._pessimistic_fallback(d_star, x, y)
        phi_state = -cost / d_max
        return max(phi_state, -1.0) + phi_phase

    def get_gradient(
        self,
        state: Dict[str, Any],
        epsilon: float = 6.0,
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
            "n_surface_points": self.graph.n_points,
            "n_graph_nodes": self.graph.n_nodes,
            "n_edges": len(self.graph.edges_src),
            "d_max_switch": self.d_max_switch,
            "d_max_exit": self.d_max_exit,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _try_surface_lookup(
        self,
        d_star: np.ndarray,
        x: float,
        y: float,
        vx: float,
        vy: float,
    ) -> Optional[float]:
        """Try surface node lookup near (x, y).

        Returns cost or None if no surface point within match radius.
        """
        pt_idx = find_nearest_point(self.graph, x, y)
        if pt_idx is None:
            return None

        px, py = self.graph.point_positions[pt_idx]
        d2 = (px - x) ** 2 + (py - y) ** 2
        if d2 > _SURFACE_MATCH_RADIUS**2:
            return None

        is_floor = self.graph.point_is_floor[pt_idx]
        vel = vx if is_floor else vy
        cost = lookup_cost_interpolated(
            self.graph,
            d_star,
            x,
            y,
            vel,
            is_floor=is_floor,
        )
        if not np.isfinite(cost):
            return None
        return cost

    def _pessimistic_fallback(
        self,
        d_star: np.ndarray,
        x: float,
        y: float,
    ) -> float:
        """Pessimistic cost estimate for airborne states.

        Uses nearest surface point cost + estimated fall frames.
        """
        pt_idx = find_nearest_point(self.graph, x, y)
        if pt_idx is None:
            return float(self.d_max_switch)

        px, py = self.graph.point_positions[pt_idx]

        # Estimate fall frames (vertical distance / gravity)
        dy = max(0.0, py - y)  # positive = below in screen coords
        if GRAVITY_FALL > 0:
            fall_frames = int(2.0 * dy / GRAVITY_FALL) if dy > 0 else 0
        else:
            fall_frames = 0

        # Look up surface cost (zero velocity bin as conservative est.)
        nid = node_id(pt_idx, 0)
        if nid < len(d_star):
            surface_cost = float(d_star[nid])
        else:
            finite = d_star[np.isfinite(d_star)]
            surface_cost = (
                float(np.max(finite)) if len(finite) > 0 else self.d_max_switch
            )

        if not np.isfinite(surface_cost):
            return float(self.d_max_switch)

        return fall_frames + surface_cost
