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
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..constants.physics_constants import (
    GRAVITY_FALL,
    NINJA_RADIUS,
    TOGGLE_MINE_RADIUS_TOGGLED,
)
from .surface_extraction import (
    SurfacePoint,
    SURFACE_FLOOR,
    SURFACE_WALL,
    MINE_SURFACE_EXCLUSION_RADIUS,
)
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

# Kill radius for toggle mines: ninja center must stay outside this distance
_MINE_KILL_RADIUS = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED  # 14.0 px
_MINE_KILL_RADIUS_SQ = _MINE_KILL_RADIUS * _MINE_KILL_RADIUS

# Exclusion zone radius: matches the surface-extraction virtual gap radius.
# The annular band [_MINE_KILL_RADIUS, _MINE_EXCLUSION_RADIUS) has no surface
# nodes (virtual gap), so nearest-node lookup can snap to the wrong side.
_MINE_EXCLUSION_RADIUS_SQ = MINE_SURFACE_EXCLUSION_RADIUS * MINE_SURFACE_EXCLUSION_RADIUS

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

        # Load toggle mine positions for runtime hazard masking.
        # Mines are NOT blocked at precompute time because their deadly state is
        # determined solely by ninja contact, not by switch activation.
        # Backward compatible: old .npz files without mine arrays get no masking.
        if "mine_positions_x" in data and len(data["mine_positions_x"]) > 0:
            mine_xs = data["mine_positions_x"].astype(np.float32)
            mine_ys = data["mine_positions_y"].astype(np.float32)
            mine_init = data["mine_initial_states"].astype(np.uint8)

            # Type-21 mines start toggled (deadly, initial_state=0) and remain
            # deadly unless a ninja touches them.  For the basics level every
            # mine is type-21 and is always a hazard.
            always_deadly_mask = mine_init == 0
            self._always_deadly_mines: List[Tuple[float, float]] = [
                (float(mine_xs[i]), float(mine_ys[i]))
                for i in range(len(mine_xs))
                if always_deadly_mask[i]
            ]
            # Type-1 mines start safe; they only become deadly after ninja contact.
            # These are stored separately for potential future dynamic-state support.
            self._conditionally_deadly_mines: List[Tuple[float, float]] = [
                (float(mine_xs[i]), float(mine_ys[i]))
                for i in range(len(mine_xs))
                if not always_deadly_mask[i]
            ]
        else:
            self._always_deadly_mines = []
            self._conditionally_deadly_mines = []

        # Load per-mine boundary node pairs for exclusion-zone-aware lookup.
        # Shape (num_always_deadly_mines, 2): matches _always_deadly_mines order.
        # Backward compatible: old .npz files without this array get no pinning.
        if (
            "mine_boundary_nodes" in data
            and len(self._always_deadly_mines) > 0
            and len(data["mine_boundary_nodes"]) == len(self._always_deadly_mines)
        ):
            self._mine_boundary_nodes: Optional[np.ndarray] = (
                data["mine_boundary_nodes"].astype(np.int32)
            )
        else:
            self._mine_boundary_nodes = None

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
        logger.info(
            f"  Mines (always deadly): {len(self._always_deadly_mines):,}, "
            f"(conditionally deadly): {len(self._conditionally_deadly_mines):,}"
        )
        if self._mine_boundary_nodes is not None:
            n_pinned = int(np.sum(self._mine_boundary_nodes >= 0))
            logger.info(
                f"  Mine boundary nodes: {self._mine_boundary_nodes.shape[0]} mines, "
                f"{n_pinned} boundary node slots populated (exclusion-zone pinning active)"
            )
        else:
            logger.info("  Mine boundary nodes: not available (exclusion-zone pinning disabled)")

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

        # Runtime mine mask: if the agent is inside the kill radius of a
        # currently-deadly mine, return the worst possible potential for this
        # phase.  This prevents PBRS from rewarding the agent for occupying a
        # position that would instantly kill it.
        if self._always_deadly_mines and self._is_near_deadly_mine(x, y):
            return -1.0 + phi_phase

        # Exclusion-zone pinning: if the agent is inside the virtual gap
        # (between kill radius 14 px and exclusion radius 18 px), the normal
        # nearest-node search reaches past the gap and may snap to the post-mine
        # boundary node, producing a spurious positive reward for running toward
        # the mine.  Pin the lookup to the entry-side boundary node instead.
        eff_x, eff_y = self._apply_exclusion_zone_pin(x, y)

        # Determine contact state
        contact = state.get("contact_state", None)

        if contact is None:
            # Heuristic: try surface lookup, fall back to airborne
            cost = self._try_surface_lookup(d_star, eff_x, eff_y, vx, vy)
            if cost is not None:
                phi_state = -cost / d_max
                return max(phi_state, -1.0) + phi_phase
            # Treat as airborne
            contact = CONTACT_AIRBORNE

        if contact == CONTACT_FLOOR:
            cost = lookup_cost_interpolated(
                self.graph, d_star, eff_x, eff_y, vx, SURFACE_FLOOR)
            if not np.isfinite(cost):
                cost = self._pessimistic_fallback(d_star, x, y)
            phi_state = -cost / d_max
            return max(phi_state, -1.0) + phi_phase

        if contact == CONTACT_WALL:
            cost = lookup_cost_interpolated(
                self.graph, d_star, eff_x, eff_y, vy, SURFACE_WALL)
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
    def _is_near_deadly_mine(self, x: float, y: float) -> bool:
        """Return True if (x, y) is within kill range of any always-deadly mine.

        Uses the pre-built ``_always_deadly_mines`` list (type-21 mines that
        start toggled and remain deadly unless explicitly toggled off by ninja
        contact).  The kill range is NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED
        = 14.0 px, matching the in-game death condition.
        """
        for mx, my in self._always_deadly_mines:
            dx = x - mx
            dy = y - my
            if dx * dx + dy * dy <= _MINE_KILL_RADIUS_SQ:
                return True
        return False

    def _in_mine_exclusion_zone(self, x: float, y: float) -> Optional[int]:
        """Return the mine index if (x, y) is in the exclusion zone but outside the kill zone.

        The exclusion zone is the annular band where no surface nodes exist
        (virtual gap) but the agent has not yet died: distances in the range
        ``[_MINE_KILL_RADIUS, MINE_SURFACE_EXCLUSION_RADIUS)`` from the mine
        center.  Only checks always-deadly mines because those are the mines
        with precomputed boundary nodes.

        Returns:
            Mine index into ``_always_deadly_mines`` / ``_mine_boundary_nodes``,
            or ``None`` if not in any exclusion zone.
        """
        for i, (mx, my) in enumerate(self._always_deadly_mines):
            dx = x - mx
            dy = y - my
            d2 = dx * dx + dy * dy
            if _MINE_KILL_RADIUS_SQ < d2 < _MINE_EXCLUSION_RADIUS_SQ:
                return i
        return None

    def _get_entry_boundary_pos(
        self, mine_idx: int, agent_x: float, agent_y: float,
    ) -> Optional[Tuple[float, float]]:
        """Return the position of the boundary node on the agent's entry side.

        Picks whichever of the two precomputed boundary nodes for ``mine_idx``
        is closest to the agent.  The entry-side node is always closer since
        the agent has not yet crossed to the far side.

        Args:
            mine_idx: Index into ``_always_deadly_mines`` / ``_mine_boundary_nodes``.
            agent_x, agent_y: Current agent position.

        Returns:
            ``(node_x, node_y)`` of the entry-side boundary node, or ``None``
            if both node slots are -1 (no boundary nodes found at precompute).
        """
        assert self._mine_boundary_nodes is not None
        left_idx = int(self._mine_boundary_nodes[mine_idx, 0])
        right_idx = int(self._mine_boundary_nodes[mine_idx, 1])

        best_idx: Optional[int] = None
        best_d2 = float("inf")
        for idx in (left_idx, right_idx):
            if idx < 0:
                continue
            sp = self.graph.surface_points[idx]
            d2 = (sp.x - agent_x) ** 2 + (sp.y - agent_y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_idx = idx

        if best_idx is None:
            return None
        sp = self.graph.surface_points[best_idx]
        return (sp.x, sp.y)

    def _apply_exclusion_zone_pin(
        self, x: float, y: float,
    ) -> Tuple[float, float]:
        """Return a pinned position if the agent is in a mine exclusion zone.

        If the agent is inside the virtual gap around a mine (annular band
        between kill radius and surface exclusion radius), return the position
        of the entry-side boundary node so that ``lookup_cost_interpolated``
        finds the correct pre-mine boundary node instead of snapping to the
        post-mine node on the far side.

        Returns the original ``(x, y)`` unchanged if no pinning applies.
        """
        if self._mine_boundary_nodes is None or not self._always_deadly_mines:
            return x, y

        mine_idx = self._in_mine_exclusion_zone(x, y)
        if mine_idx is None:
            return x, y

        pinned = self._get_entry_boundary_pos(mine_idx, x, y)
        if pinned is None:
            return x, y
        return pinned

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

        Prefers surface points directly below the agent (same horizontal
        band) because that is where a falling agent will actually land.
        Falls back to nearest Euclidean point if nothing is below.
        """
        # Prefer a floor point below the agent within a horizontal band of
        # ±60 px, since that is the actual landing target for a falling ninja.
        best_pt_idx = self._find_surface_below(x, y)
        if best_pt_idx is None:
            best_pt_idx = find_nearest_point(self.graph, x, y)
        if best_pt_idx is None:
            return float(self.d_max_switch)

        sp = self.graph.surface_points[best_pt_idx]

        # If the fallback landing point is itself inside a mine kill zone, treat
        # the position as unreachable (worst cost) rather than returning a
        # misleadingly low cost from that node.
        if self._always_deadly_mines and self._is_near_deadly_mine(sp.x, sp.y):
            return float(self.d_max_switch)

        # Estimate fall frames using correct free-fall kinematics: t = sqrt(2h/g)
        dy = max(0.0, sp.y - y)  # positive = below in screen coords (y grows down)
        fall_frames = int(math.sqrt(2.0 * dy / GRAVITY_FALL)) if dy > 0 and GRAVITY_FALL > 0 else 0

        # Look up surface cost (use zero velocity bin as conservative estimate)
        vel_bin = 0
        nid = node_id(best_pt_idx, vel_bin, self.graph.max_vel_bins)
        if nid < len(d_star):
            surface_cost = float(d_star[nid])
        else:
            surface_cost = float(np.max(d_star[np.isfinite(d_star)]))

        if not np.isfinite(surface_cost):
            return float(self.d_max_switch)

        return fall_frames + surface_cost

    def _find_surface_below(
        self, x: float, y: float, horizontal_band_px: float = 60.0,
    ) -> Optional[int]:
        """Find the nearest floor surface point directly below (x, y).

        Searches surface points with |point.x - x| <= horizontal_band_px and
        point.y > y (below in screen coordinates).  Returns the index of the
        highest such floor point (smallest dy), or None if none exists.
        """
        best_idx: Optional[int] = None
        best_dy = float("inf")
        for i, sp in enumerate(self.graph.surface_points):
            if sp.surface_type != SURFACE_FLOOR:
                continue
            if abs(sp.x - x) > horizontal_band_px:
                continue
            dy = sp.y - y
            if 0 < dy < best_dy:
                best_dy = dy
                best_idx = i
        return best_idx


# Backward-compatible alias
PrecomputedValuePBRS = SurfaceGraphPBRS


def _safe_max(arr: np.ndarray) -> float:
    """Max of finite values in an array, or 1.0 if none."""
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if len(finite) > 0 else 1.0
