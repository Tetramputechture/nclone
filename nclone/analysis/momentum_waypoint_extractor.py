"""Extract momentum-building waypoints from expert demonstrations.

This module analyzes expert trajectories to identify points where the optimal
strategy involves temporarily moving away from the goal to build momentum for
momentum-dependent maneuvers (e.g., long jumps, sloped jumps over hazards).

These waypoints inform PBRS potential functions, ensuring the agent receives
positive rewards for necessary momentum-building behavior rather than being
penalized for "moving away from goal."

Key Insight:
Expert demonstrations already solve momentum problems correctly. By extracting
their "detour patterns," we can make PBRS momentum-aware without expensive
physics simulation.

Usage:
    extractor = MomentumWaypointExtractor()
    waypoints = extractor.extract_from_episode(demo_episode, level_data)
    # Use waypoints in PBRS potential calculation
"""

import logging
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Thresholds for momentum waypoint detection
MIN_SPEED_FOR_MOMENTUM = 1.5  # pixels/frame - must be moving significantly
SPEED_INCREASE_THRESHOLD = 0.8  # Speed must increase by this amount
EUCLIDEAN_DISTANCE_INCREASE_THRESHOLD = 5.0  # Must move 5px away from goal
LOOKAHEAD_WINDOW = 20  # Check next 20 frames for jump action
MIN_WAYPOINT_SEPARATION = 50.0  # Waypoints must be 50px apart (avoid duplicates)


@dataclass
class MomentumWaypoint:
    """Represents a point where momentum-building occurs in expert trajectory.

    Attributes:
        position: (x, y) pixel position where momentum-building starts
        velocity: (vx, vy) velocity at this waypoint
        speed: Scalar speed magnitude
        approach_direction: Normalized (dx, dy) direction of approach
        frame_index: Frame index in the demonstration
        leads_to_jump: Whether this waypoint precedes a jump action
        distance_to_goal: Euclidean distance to goal at this waypoint
    """

    position: Tuple[float, float]
    velocity: Tuple[float, float]
    speed: float
    approach_direction: Tuple[float, float]
    frame_index: int
    leads_to_jump: bool
    distance_to_goal: float


class MomentumWaypointExtractor:
    """Extract momentum-building waypoints from expert demonstrations.

    Analyzes trajectory to find points where:
    1. Agent moves away from goal (Euclidean distance increases)
    2. Agent builds velocity (speed increases significantly)
    3. Followed by momentum-dependent action (jump, slope navigation)

    These waypoints represent "necessary detours" that should be rewarded
    by PBRS rather than penalized.
    """

    def __init__(
        self,
        min_speed: float = MIN_SPEED_FOR_MOMENTUM,
        speed_increase_threshold: float = SPEED_INCREASE_THRESHOLD,
        distance_increase_threshold: float = EUCLIDEAN_DISTANCE_INCREASE_THRESHOLD,
        lookahead_window: int = LOOKAHEAD_WINDOW,
        min_separation: float = MIN_WAYPOINT_SEPARATION,
    ):
        """Initialize waypoint extractor with detection thresholds.

        Args:
            min_speed: Minimum speed to consider as "momentum"
            speed_increase_threshold: Required speed increase for momentum-building
            distance_increase_threshold: Required Euclidean distance increase from goal
            lookahead_window: Number of frames to look ahead for jump detection
            min_separation: Minimum distance between waypoints (avoids duplicates)
        """
        self.min_speed = min_speed
        self.speed_increase_threshold = speed_increase_threshold
        self.distance_increase_threshold = distance_increase_threshold
        self.lookahead_window = lookahead_window
        self.min_separation = min_separation

    def extract_from_episode(
        self,
        positions: List[Tuple[float, float]],
        velocities: List[Tuple[float, float]],
        actions: List[int],
        goal_position: Tuple[float, float],
        switch_position: Optional[Tuple[float, float]] = None,
        switch_activation_frame: Optional[int] = None,
    ) -> List[MomentumWaypoint]:
        """Extract momentum waypoints from a single demonstration episode.

        Detects segments where expert trajectory:
        - Moves away from current goal (switch or exit)
        - Builds velocity significantly
        - Precedes jump or momentum-dependent action

        Args:
            positions: List of (x, y) positions per frame
            velocities: List of (vx, vy) velocities per frame
            actions: List of action indices per frame
            goal_position: Exit door position (x, y)
            switch_position: Optional switch position (x, y)
            switch_activation_frame: Optional frame when switch activated

        Returns:
            List of MomentumWaypoint objects representing momentum-building locations
        """
        if len(positions) < self.lookahead_window + 10:
            # Episode too short for meaningful momentum analysis
            return []

        waypoints = []

        # Analyze trajectory in windows
        for i in range(10, len(positions) - self.lookahead_window):
            # Determine current goal based on switch activation
            if switch_activation_frame is not None and i >= switch_activation_frame:
                current_goal = goal_position  # Exit door
            elif switch_position is not None:
                current_goal = switch_position  # Switch
            else:
                current_goal = goal_position  # Exit door (fallback)

            # Calculate Euclidean distances to goal
            dist_now = self._euclidean_distance(positions[i], current_goal)
            dist_prev = self._euclidean_distance(positions[i - 1], current_goal)
            dist_earlier = self._euclidean_distance(
                positions[max(0, i - 5)], current_goal
            )

            # Check if moving away from goal (Euclidean distance increasing)
            if dist_now <= dist_prev:
                continue  # Not moving away - skip

            distance_increase = dist_now - dist_earlier
            if distance_increase < self.distance_increase_threshold:
                continue  # Not moving away significantly enough

            # Calculate speed at current and previous frames
            speed_now = self._velocity_magnitude(velocities[i])
            speed_prev = self._velocity_magnitude(velocities[max(0, i - 5)])

            # Check if speed is building
            if speed_now <= speed_prev + self.speed_increase_threshold:
                continue  # Speed not building significantly

            if speed_now < self.min_speed:
                continue  # Speed too low to be meaningful momentum

            # Check if followed by jump action in lookahead window
            leads_to_jump = self._has_jump_in_window(
                actions[i : i + self.lookahead_window]
            )

            # Create waypoint candidate
            approach_direction = self._normalize_vector(velocities[i])
            waypoint = MomentumWaypoint(
                position=positions[i],
                velocity=velocities[i],
                speed=speed_now,
                approach_direction=approach_direction,
                frame_index=i,
                leads_to_jump=leads_to_jump,
                distance_to_goal=dist_now,
            )

            # Check if waypoint is sufficiently separated from existing waypoints
            if self._is_waypoint_unique(waypoint, waypoints):
                waypoints.append(waypoint)
                logger.debug(
                    f"Momentum waypoint detected at frame {i}: "
                    f"pos={positions[i]}, speed={speed_now:.2f}, "
                    f"moving_away={distance_increase:.1f}px, "
                    f"leads_to_jump={leads_to_jump}"
                )

        logger.info(
            f"Extracted {len(waypoints)} momentum waypoints from {len(positions)}-frame episode"
        )
        return waypoints

    def extract_from_demonstration_buffer(
        self,
        demo_buffer: Any,
        level_filter: Optional[str] = None,
    ) -> Dict[str, List[MomentumWaypoint]]:
        """Extract momentum waypoints from all demonstrations in buffer.

        Groups waypoints by level_id for level-specific caching.

        Args:
            demo_buffer: DemonstrationBuffer instance with loaded episodes
            level_filter: Optional level_id to extract waypoints for (None = all levels)

        Returns:
            Dictionary mapping level_id -> List[MomentumWaypoint]
        """
        waypoints_by_level = {}

        if not hasattr(demo_buffer, "episodes"):
            logger.warning("Demo buffer has no episodes attribute")
            return waypoints_by_level

        for episode in demo_buffer.episodes:
            # Get level_id from episode metadata
            level_id = getattr(episode, "level_id", None)
            if level_id is None:
                logger.warning("Episode missing level_id, skipping waypoint extraction")
                continue

            # Apply level filter if specified
            if level_filter is not None and level_id != level_filter:
                continue

            # Extract positions, velocities, actions from transitions
            positions = []
            velocities = []
            actions = []

            for transition in episode.transitions:
                obs = transition.observation
                positions.append((obs["player_x"], obs["player_y"]))
                velocities.append((obs["player_xspeed"], obs["player_yspeed"]))
                actions.append(transition.action)

            # Get goal positions from last observation
            if episode.transitions:
                last_obs = episode.transitions[-1].observation
                goal_position = (last_obs["exit_door_x"], last_obs["exit_door_y"])
                switch_position = (last_obs["switch_x"], last_obs["switch_y"])

                # Detect switch activation frame (if any)
                switch_activation_frame = None
                for idx, trans in enumerate(episode.transitions):
                    if trans.observation.get("switch_activated", False):
                        switch_activation_frame = idx
                        break
            else:
                continue  # No transitions, skip

            # Extract waypoints for this episode
            episode_waypoints = self.extract_from_episode(
                positions=positions,
                velocities=velocities,
                actions=actions,
                goal_position=goal_position,
                switch_position=switch_position,
                switch_activation_frame=switch_activation_frame,
            )

            # Group by level_id
            if level_id not in waypoints_by_level:
                waypoints_by_level[level_id] = []
            waypoints_by_level[level_id].extend(episode_waypoints)

        # Deduplicate waypoints per level (cluster nearby waypoints)
        for level_id in waypoints_by_level:
            waypoints_by_level[level_id] = self._deduplicate_waypoints(
                waypoints_by_level[level_id]
            )
            logger.info(
                f"Level {level_id}: {len(waypoints_by_level[level_id])} unique momentum waypoints"
            )

        return waypoints_by_level

    def _euclidean_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two positions."""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return math.sqrt(dx * dx + dy * dy)

    def _velocity_magnitude(self, velocity: Tuple[float, float]) -> float:
        """Calculate velocity magnitude (speed)."""
        vx, vy = velocity
        return math.sqrt(vx * vx + vy * vy)

    def _normalize_vector(self, vector: Tuple[float, float]) -> Tuple[float, float]:
        """Normalize vector to unit length."""
        magnitude = self._velocity_magnitude(vector)
        if magnitude < 0.001:
            return (0.0, 0.0)
        return (vector[0] / magnitude, vector[1] / magnitude)

    def _has_jump_in_window(self, action_window: List[int]) -> bool:
        """Check if jump action (3, 4, 5) appears in action window."""
        # Jump actions in N++: 3=Jump, 4=Jump+Left, 5=Jump+Right
        return any(action in [3, 4, 5] for action in action_window)

    def _is_waypoint_unique(
        self, waypoint: MomentumWaypoint, existing_waypoints: List[MomentumWaypoint]
    ) -> bool:
        """Check if waypoint is sufficiently separated from existing waypoints."""
        for existing in existing_waypoints:
            distance = self._euclidean_distance(waypoint.position, existing.position)
            if distance < self.min_separation:
                return False  # Too close to existing waypoint
        return True

    def _deduplicate_waypoints(
        self, waypoints: List[MomentumWaypoint]
    ) -> List[MomentumWaypoint]:
        """Cluster nearby waypoints and keep representative ones.

        Uses simple greedy clustering: keep first waypoint in each cluster.

        Args:
            waypoints: List of waypoints to deduplicate

        Returns:
            Deduplicated list of waypoints
        """
        if len(waypoints) <= 1:
            return waypoints

        # Sort by frame index for temporal consistency
        sorted_waypoints = sorted(waypoints, key=lambda w: w.frame_index)

        deduplicated = []
        for waypoint in sorted_waypoints:
            # Check if sufficiently separated from already selected waypoints
            if self._is_waypoint_unique(waypoint, deduplicated):
                deduplicated.append(waypoint)

        logger.debug(
            f"Deduplicated {len(waypoints)} waypoints -> {len(deduplicated)} unique waypoints"
        )
        return deduplicated

    def save_waypoints_to_cache(
        self,
        waypoints_by_level: Dict[str, List[MomentumWaypoint]],
        cache_dir: str = "momentum_waypoints_cache",
    ) -> None:
        """Save extracted waypoints to cache files for fast loading.

        Args:
            waypoints_by_level: Dictionary mapping level_id -> waypoints
            cache_dir: Directory to save cache files
        """
        import os
        import pickle

        os.makedirs(cache_dir, exist_ok=True)

        for level_id, waypoints in waypoints_by_level.items():
            # Sanitize level_id for filename
            safe_level_id = level_id.replace("/", "_").replace("\\", "_")
            cache_file = os.path.join(cache_dir, f"{safe_level_id}.pkl")

            with open(cache_file, "wb") as f:
                pickle.dump(waypoints, f)

            logger.info(
                f"Saved {len(waypoints)} waypoints for level {level_id} to {cache_file}"
            )

    @staticmethod
    def load_waypoints_from_cache(
        level_id: str, cache_dir: str = "momentum_waypoints_cache"
    ) -> Optional[List[MomentumWaypoint]]:
        """Load waypoints from cache file for specific level.

        Args:
            level_id: Level identifier
            cache_dir: Directory containing cache files

        Returns:
            List of waypoints if cache exists, None otherwise
        """
        import os
        import pickle

        # Sanitize level_id for filename
        safe_level_id = level_id.replace("/", "_").replace("\\", "_")
        cache_file = os.path.join(cache_dir, f"{safe_level_id}.pkl")

        if not os.path.exists(cache_file):
            return None

        try:
            with open(cache_file, "rb") as f:
                waypoints = pickle.load(f)
            logger.debug(f"Loaded {len(waypoints)} waypoints for level {level_id}")
            return waypoints
        except Exception as e:
            logger.warning(f"Failed to load waypoints cache for {level_id}: {e}")
            return None

