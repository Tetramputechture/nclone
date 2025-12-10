"""Adaptive waypoint discovery system for robust reward shaping.

Automatically discovers waypoints from successful agent trajectories and rewards
distance reduction rather than strict path adherence. This enables:
1. Learning inflection points (direction changes) from successful runs
2. Rewarding novel explorations that reduce distance to goal
3. Decaying waypoints that don't lead to continued success
4. Clustering similar waypoints to avoid redundancy

Key difference from fixed waypoints:
- Fixed waypoints penalize deviations from a specific path
- Adaptive waypoints reward any trajectory that achieves distance reduction
- Discovered from agent's own successful behavior (data-driven)

This addresses the counter-intuitive path problem where agent must:
- Go right up ramp (toward mines)
- Then jump left to switch (away from apparent goal direction)
Without penalizing novel solutions that achieve the same goal differently.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class AdaptiveWaypointSystem:
    """Discovers and tracks waypoints from successful agent trajectories.

    Waypoints are learned from successful runs by identifying:
    1. Inflection points: Direction changes > 45 degrees
    2. Rapid progress points: Positions where distance decreased significantly
    3. Pre-obstacle maneuvers: Positions before detouring around hazards

    Waypoint rewards:
    - Bonus for reaching discovered waypoint (crossing threshold)
    - No penalty for missing waypoints (allows novel paths)
    - Waypoints decay if not reinforced by continued success
    """

    def __init__(
        self,
        waypoint_radius: float = 50.0,
        max_waypoints_per_level: int = 10,
        min_waypoint_value: float = 0.1,
        waypoint_decay_rate: float = 0.95,
        cluster_radius: float = 30.0,
    ):
        """Initialize adaptive waypoint system.

        Args:
            waypoint_radius: Radius for considering waypoint "reached" (pixels)
            max_waypoints_per_level: Maximum waypoints to track per level
            min_waypoint_value: Minimum value before waypoint is removed
            waypoint_decay_rate: Decay multiplier per non-reinforced episode
            cluster_radius: Distance threshold for merging similar waypoints
        """
        self.waypoint_radius = waypoint_radius
        self.max_waypoints_per_level = max_waypoints_per_level
        self.min_waypoint_value = min_waypoint_value
        self.waypoint_decay_rate = waypoint_decay_rate
        self.cluster_radius = cluster_radius

        # Waypoints per level: level_id -> List[(pos, value, type, discovery_count)]
        # pos: (x, y) position in pixels
        # value: importance score (decays if not reinforced)
        # type: "inflection" or "progress" (for diagnostics)
        # discovery_count: number of times this waypoint was reached in successful runs
        self.waypoints: Dict[str, List[Tuple[Tuple[float, float], float, str, int]]] = (
            defaultdict(list)
        )

        # Track current level for level-specific waypoints
        self.current_level_id: Optional[str] = None

        # Track waypoints reached this episode (for logging)
        self.waypoints_reached_this_episode: List[Tuple[float, float]] = []

        # Statistics
        self.total_waypoints_discovered = 0
        self.total_waypoint_bonuses = 0.0
        self.waypoints_decayed = 0

        # Success rate tracking for curriculum-aware decay
        self.recent_success_rate = 0.0  # Updated externally by reward calculator

        logger.info(
            f"Adaptive waypoint system initialized: "
            f"radius={waypoint_radius}px, max_per_level={max_waypoints_per_level}, "
            f"cluster_radius={cluster_radius}px"
        )

    def extract_waypoints_from_trajectory(
        self,
        positions: List[Tuple[float, float]],
        distances: List[float],
        success: bool,
        level_id: str = "default",
    ) -> int:
        """Extract potential waypoints from a trajectory.

        Identifies inflection points and rapid progress positions,
        then clusters and adds them to the waypoint archive.

        Args:
            positions: List of (x, y) positions along trajectory
            distances: List of distances to goal at each position
            success: Whether trajectory completed successfully
            level_id: Level identifier for level-specific waypoints

        Returns:
            Number of new waypoints added
        """
        if not success or len(positions) < 5:
            return 0  # Only learn from successful runs with enough data

        new_waypoints = []

        # Calculate total path distance for distance weighting
        # Use initial distance (at spawn) as the total path distance
        total_path_distance = distances[0] if distances else 1.0
        if total_path_distance < 1.0:
            total_path_distance = 1.0  # Avoid division by zero

        # === INFLECTION POINT DETECTION ===
        # Find positions where agent changed direction significantly
        # These often indicate strategic decision points (e.g., top of ramp)
        inflection_indices = []
        for i in range(2, len(positions) - 2):
            # Velocity before (over 2 steps for stability)
            v1_x = positions[i][0] - positions[i - 2][0]
            v1_y = positions[i][1] - positions[i - 2][1]

            # Velocity after (over 2 steps for stability)
            v2_x = positions[i + 2][0] - positions[i][0]
            v2_y = positions[i + 2][1] - positions[i][1]

            mag1 = (v1_x * v1_x + v1_y * v1_y) ** 0.5
            mag2 = (v2_x * v2_x + v2_y * v2_y) ** 0.5

            if mag1 > 5.0 and mag2 > 5.0:  # Significant movement (> 5px)
                # Dot product to detect direction change
                dot = v1_x * v2_x + v1_y * v2_y
                cos_angle = dot / (mag1 * mag2)

                # Threshold: cos(45°) ≈ 0.7, so < 0.7 means > 45° turn
                if cos_angle < 0.7:
                    # Distance-weighted value: earlier waypoints (farther from goal) more valuable
                    distance_to_goal = distances[i] if i < len(distances) else 0.0
                    distance_factor = (
                        distance_to_goal / total_path_distance
                    )  # 0.0 at goal, 1.0 at spawn
                    base_value = 1.0
                    weighted_value = base_value * (1.0 + distance_factor)

                    new_waypoints.append((positions[i], weighted_value, "inflection"))
                    inflection_indices.append(i)
                    logger.debug(
                        f"Inflection point detected at ({positions[i][0]:.0f}, {positions[i][1]:.0f}), "
                        f"angle_cos={cos_angle:.2f}, distance_factor={distance_factor:.2f}, "
                        f"weighted_value={weighted_value:.2f}"
                    )

        # === MOMENTUM PEAK DETECTION ===
        # For each inflection point, look back 10-20 steps to find momentum-building position
        # This captures preparation maneuvers (e.g., running left to build speed before jumping right)
        for inflection_idx in inflection_indices:
            # Look back window: 10-20 steps before inflection
            lookback_start = max(2, inflection_idx - 20)
            lookback_end = max(
                2, inflection_idx - 5
            )  # Don't look too close to inflection

            if lookback_start >= lookback_end:
                continue

            # Find position with maximum velocity magnitude in lookback window
            max_velocity_mag = 0.0
            max_velocity_idx = None

            for j in range(lookback_start, lookback_end):
                # Calculate velocity over 2-step window for stability
                if j >= 2:
                    vel_x = positions[j][0] - positions[j - 2][0]
                    vel_y = positions[j][1] - positions[j - 2][1]
                    vel_mag = (vel_x * vel_x + vel_y * vel_y) ** 0.5

                    if vel_mag > max_velocity_mag:
                        max_velocity_mag = vel_mag
                        max_velocity_idx = j

            # Add momentum peak waypoint if significant velocity found (> 10px over 2 steps)
            if max_velocity_idx is not None and max_velocity_mag > 10.0:
                # Check if too close to existing waypoints (avoid duplicates)
                is_duplicate = False
                for existing_wp_pos, _, _ in new_waypoints:
                    dx = positions[max_velocity_idx][0] - existing_wp_pos[0]
                    dy = positions[max_velocity_idx][1] - existing_wp_pos[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < 30.0:  # Within clustering radius
                        is_duplicate = True
                        break

                if not is_duplicate:
                    # Distance-weighted value for momentum waypoints
                    distance_to_goal = (
                        distances[max_velocity_idx]
                        if max_velocity_idx < len(distances)
                        else 0.0
                    )
                    distance_factor = distance_to_goal / total_path_distance
                    base_value = 1.2
                    weighted_value = base_value * (1.0 + distance_factor)

                    new_waypoints.append(
                        (positions[max_velocity_idx], weighted_value, "momentum")
                    )
                    logger.debug(
                        f"Momentum peak detected at ({positions[max_velocity_idx][0]:.0f}, {positions[max_velocity_idx][1]:.0f}), "
                        f"velocity_mag={max_velocity_mag:.1f}px, {inflection_idx - max_velocity_idx} steps before inflection, "
                        f"distance_factor={distance_factor:.2f}, weighted_value={weighted_value:.2f}"
                    )

        # === RAPID PROGRESS DETECTION ===
        # Find positions where distance decreased rapidly
        # Window size: 5 steps (with frame_skip=4, this is ~20 frames)
        window_size = 5
        progress_threshold = 20.0  # 20px progress in window

        for i in range(window_size, len(distances)):
            window_improvement = distances[i - window_size] - distances[i]
            if window_improvement > progress_threshold:
                # Significant progress in this window - apply distance weighting
                distance_to_goal = distances[i] if i < len(distances) else 0.0
                distance_factor = distance_to_goal / total_path_distance
                base_value = 0.5
                weighted_value = base_value * (1.0 + distance_factor)

                new_waypoints.append((positions[i], weighted_value, "progress"))
                logger.debug(
                    f"Rapid progress point detected at ({positions[i][0]:.0f}, {positions[i][1]:.0f}), "
                    f"improvement={window_improvement:.1f}px, distance_factor={distance_factor:.2f}, "
                    f"weighted_value={weighted_value:.2f}"
                )

        # === WAYPOINT CLUSTERING AND ADDITION ===
        # Cluster new waypoints and add to archive
        waypoints_added = 0
        for wp_pos, wp_value, wp_type in new_waypoints:
            if self._add_or_reinforce_waypoint(level_id, wp_pos, wp_value, wp_type):
                waypoints_added += 1

        if waypoints_added > 0:
            logger.info(
                f"Extracted {waypoints_added} waypoints from successful trajectory "
                f"(level: {level_id}, {len(positions)} positions, {len(new_waypoints)} candidates)"
            )

        return waypoints_added

    def _add_or_reinforce_waypoint(
        self,
        level_id: str,
        position: Tuple[float, float],
        value: float,
        waypoint_type: str,
    ) -> bool:
        """Add new waypoint or reinforce existing nearby waypoint.

        Checks for existing waypoints within cluster_radius and either:
        - Reinforces existing waypoint (increases value, increments count)
        - Adds new waypoint if no nearby waypoint exists

        Args:
            level_id: Level identifier
            position: Waypoint position (x, y)
            value: Initial waypoint value
            waypoint_type: "inflection" or "progress"

        Returns:
            True if new waypoint added, False if existing waypoint reinforced
        """
        level_waypoints = self.waypoints[level_id]

        # Check for nearby existing waypoint (clustering)
        for i, (wp_pos, wp_value, wp_type, wp_count) in enumerate(level_waypoints):
            dx = position[0] - wp_pos[0]
            dy = position[1] - wp_pos[1]
            distance = (dx * dx + dy * dy) ** 0.5

            if distance < self.cluster_radius:
                # Found nearby waypoint - reinforce it
                # Increase value (capped at 2.0) and increment discovery count
                new_value = min(2.0, wp_value + value * 0.5)
                new_count = wp_count + 1
                level_waypoints[i] = (wp_pos, new_value, wp_type, new_count)
                logger.debug(
                    f"Reinforced waypoint at ({wp_pos[0]:.0f}, {wp_pos[1]:.0f}): "
                    f"value {wp_value:.2f} → {new_value:.2f}, count={new_count}"
                )
                return False  # Existing waypoint reinforced, not new

        # No nearby waypoint - add new one if we haven't hit limit
        if len(level_waypoints) >= self.max_waypoints_per_level:
            # Remove lowest-value waypoint to make room
            level_waypoints.sort(key=lambda x: x[1])  # Sort by value
            removed = level_waypoints.pop(0)
            logger.debug(
                f"Removed low-value waypoint at ({removed[0][0]:.0f}, {removed[0][1]:.0f}), "
                f"value={removed[1]:.2f}"
            )
            self.waypoints_decayed += 1

        # Add new waypoint
        level_waypoints.append((position, value, waypoint_type, 1))
        self.total_waypoints_discovered += 1
        logger.info(
            f"Added new {waypoint_type} waypoint at ({position[0]:.0f}, {position[1]:.0f}), "
            f"value={value:.2f} (level: {level_id})"
        )
        return True

    def get_waypoint_bonus(
        self,
        current_pos: Tuple[float, float],
        previous_pos: Optional[Tuple[float, float]] = None,
        level_id: str = "default",
    ) -> float:
        """Calculate bonus for reaching discovered waypoints.

        Returns bonus if agent just crossed a waypoint threshold (from outside
        radius to inside radius). This rewards approaching waypoints but doesn't
        penalize missing them, allowing novel paths.

        Args:
            current_pos: Current agent position (x, y)
            previous_pos: Previous agent position (x, y)
            level_id: Level identifier

        Returns:
            Waypoint bonus reward (0.0 if no waypoints reached)
        """
        if previous_pos is None or level_id not in self.waypoints:
            return 0.0

        level_waypoints = self.waypoints[level_id]
        if not level_waypoints:
            return 0.0

        total_bonus = 0.0
        waypoints_reached = []
        gradient_bonus_applied = 0.0

        for wp_pos, wp_value, wp_type, wp_count in level_waypoints:
            # Calculate distance to waypoint from previous and current positions
            prev_dx = previous_pos[0] - wp_pos[0]
            prev_dy = previous_pos[1] - wp_pos[1]
            prev_dist = (prev_dx * prev_dx + prev_dy * prev_dy) ** 0.5

            curr_dx = current_pos[0] - wp_pos[0]
            curr_dy = current_pos[1] - wp_pos[1]
            curr_dist = (curr_dx * curr_dx + curr_dy * curr_dy) ** 0.5

            # Check if just entered waypoint radius (threshold crossing)
            if prev_dist > self.waypoint_radius and curr_dist <= self.waypoint_radius:
                # Bonus scales with waypoint value and counts reinforcement
                # Base bonus: 20.0, scaled by value (0.1 to 2.0)
                # So bonuses range from 2.0 to 40.0
                # 10x increase to make waypoints visible to learning algorithm
                # relative to PBRS (~0.75/step) and terminal rewards (50-75)
                bonus = 20.0 * wp_value
                total_bonus += bonus
                waypoints_reached.append((wp_pos, wp_type, bonus))

                logger.debug(
                    f"Waypoint reached: ({wp_pos[0]:.0f}, {wp_pos[1]:.0f}), "
                    f"type={wp_type}, value={wp_value:.2f}, bonus={bonus:.2f}, "
                    f"discovery_count={wp_count}"
                )

            # Gradient bonus: small reward for getting closer to waypoint
            # Provides denser signal to guide agent toward waypoints
            # Only applies within 100px radius to avoid affecting entire level
            elif curr_dist <= 100.0:
                distance_improvement = prev_dist - curr_dist
                if distance_improvement > 0:
                    # Gradient bonus scaled by:
                    # - Waypoint value (0.1 to 2.0)
                    # - Distance improvement normalized to 50px
                    # - Base factor 0.1 to keep gradient bonus small relative to threshold bonus
                    # Result: 0.002 to 0.4 per step for typical improvements (1-10px)
                    gradient_bonus = 0.1 * wp_value * (distance_improvement / 50.0)
                    total_bonus += gradient_bonus
                    gradient_bonus_applied += gradient_bonus

        if waypoints_reached:
            self.waypoints_reached_this_episode.extend(
                [wp[0] for wp in waypoints_reached]
            )
            self.total_waypoint_bonuses += total_bonus
            logger.info(
                f"Waypoint bonus: {total_bonus:.2f} from {len(waypoints_reached)} waypoints"
            )

        if gradient_bonus_applied > 0:
            logger.debug(
                f"Waypoint gradient bonus: {gradient_bonus_applied:.4f} for approaching waypoints"
            )

        return total_bonus

    def decay_waypoints(self, level_id: str = "default") -> None:
        """Decay waypoint values for non-reinforced episode.

        Called at episode end when trajectory was NOT successful.
        This gradually removes waypoints that don't lead to success.

        Uses curriculum-aware decay rate that adapts to success rate:
        - Low success (<10%): Aggressive decay (0.90) to quickly forget bad waypoints
        - Medium success (10-50%): Standard decay (0.95) for gradual forgetting
        - High success (>50%): Conservative decay (0.98) to preserve good waypoints

        Args:
            level_id: Level identifier
        """
        if level_id not in self.waypoints:
            return

        level_waypoints = self.waypoints[level_id]
        decayed_waypoints = []

        # Curriculum-aware decay rate based on recent success rate
        if self.recent_success_rate < 0.1:
            adaptive_decay_rate = 0.90  # Aggressive forgetting when struggling
        elif self.recent_success_rate < 0.5:
            adaptive_decay_rate = 0.95  # Standard decay for moderate performance
        else:
            adaptive_decay_rate = 0.98  # Conservative decay when succeeding

        logger.debug(
            f"Applying curriculum-aware decay: rate={adaptive_decay_rate:.2f} "
            f"(success_rate={self.recent_success_rate:.1%})"
        )

        for wp_pos, wp_value, wp_type, wp_count in level_waypoints:
            # Apply adaptive decay rate
            new_value = wp_value * adaptive_decay_rate

            # Only keep if above minimum threshold
            if new_value >= self.min_waypoint_value:
                decayed_waypoints.append((wp_pos, new_value, wp_type, wp_count))
            else:
                self.waypoints_decayed += 1
                logger.debug(
                    f"Decayed waypoint removed: ({wp_pos[0]:.0f}, {wp_pos[1]:.0f}), "
                    f"value {wp_value:.2f} → {new_value:.2f} < threshold"
                )

        # Update waypoints list
        self.waypoints[level_id] = decayed_waypoints

    def reset_episode(self) -> None:
        """Reset episode-level tracking."""
        self.waypoints_reached_this_episode = []

    def set_level(self, level_id: str) -> None:
        """Set current level for level-specific waypoint tracking.

        Args:
            level_id: Level identifier
        """
        self.current_level_id = level_id

    def update_success_rate(self, success_rate: float) -> None:
        """Update recent success rate for curriculum-aware decay.

        Should be called periodically (e.g., every evaluation) by the training loop.

        Args:
            success_rate: Recent success rate (0.0 to 1.0)
        """
        self.recent_success_rate = max(0.0, min(1.0, success_rate))
        logger.debug(
            f"Updated waypoint system success rate: {self.recent_success_rate:.1%}"
        )

    def get_waypoints_for_level(
        self, level_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """Get all waypoints for a level (for visualization/debugging).

        Args:
            level_id: Level identifier

        Returns:
            List of waypoint dictionaries with keys:
            - position: (x, y)
            - value: importance score
            - type: "inflection" or "progress"
            - discovery_count: number of successful traversals
        """
        if level_id not in self.waypoints:
            # DIAGNOSTIC: Show what keys exist vs what we're looking for
            existing_keys = list(self.waypoints.keys())
            print(f"[WAYPOINT_LOOKUP_FAIL] Level '{level_id}' not found in waypoints")
            print(f"[WAYPOINT_LOOKUP_FAIL] Available keys ({len(existing_keys)}): {existing_keys[:10]}")  # Show first 10
            return []

        waypoints = []
        for wp_pos, wp_value, wp_type, wp_count in self.waypoints[level_id]:
            waypoints.append(
                {
                    "position": wp_pos,
                    "value": wp_value,
                    "type": wp_type,
                    "discovery_count": wp_count,
                    "radius": self.waypoint_radius,
                }
            )
        return waypoints

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about waypoint system.

        Returns:
            Dictionary with waypoint statistics
        """
        total_active_waypoints = sum(len(wps) for wps in self.waypoints.values())
        avg_value = 0.0
        if total_active_waypoints > 0:
            all_values = [
                wp[1] for level_wps in self.waypoints.values() for wp in level_wps
            ]
            avg_value = sum(all_values) / len(all_values)

        return {
            "total_waypoints_discovered": self.total_waypoints_discovered,
            "active_waypoints": total_active_waypoints,
            "levels_with_waypoints": len(self.waypoints),
            "waypoints_decayed": self.waypoints_decayed,
            "total_bonuses_awarded": self.total_waypoint_bonuses,
            "average_waypoint_value": avg_value,
        }
