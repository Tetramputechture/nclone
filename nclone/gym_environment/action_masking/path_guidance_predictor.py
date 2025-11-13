"""Distance-based action masking using shortest path guidance.

This module provides the PathGuidancePredictor which masks actions that
significantly increase the distance to the goal without valid physics justification,
improving sample efficiency by reducing wasted exploration.
"""

from typing import Tuple, Optional, List, Dict, Any
from ..reward_calculation.pbrs_potentials import _flood_fill_reachable_nodes


class PathGuidancePredictor:
    """Distance-based action masking using shortest path guidance.

    Key Optimization: Recalculate path only when ninja crosses 12x12px node boundaries,
    not on a time interval. This leverages continuous ninja movement vs discrete graph.

    The predictor masks actions that significantly increase the distance to the goal
    (>10% longer than current distance) unless there's a valid physics justification
    (near walls, braking, near waypoints, or vertical jumps). This is very conservative
    to ensure masking never prevents physics exploitation.
    """

    NODE_SIZE = 12  # pixels per node (N++ constant)

    def __init__(self, path_calculator, strict_mode: bool = False, sim=None):
        """Initialize path guidance predictor.

        Args:
            path_calculator: CachedPathDistanceCalculator instance
            strict_mode: If True, mask more aggressively (experimental, not recommended)
            sim: Optional simulator reference for tile map access (for wall detection)
        """
        self.path_calculator = path_calculator
        self.strict_mode = strict_mode
        self.sim = sim  # Reference to simulator for tile checking

        # Spatial-based caching
        self.cached_path: Optional[List[Tuple[int, int]]] = None
        self.cached_goal_pos: Optional[Tuple[int, int]] = None
        self.last_ninja_node: Optional[Tuple[int, int]] = None  # Grid cell (12x12)

        # Cache adjacency and graph_data for use in action masking
        self.cached_adjacency: Optional[Dict] = None
        self.cached_graph_data: Optional[Dict] = None

        # Progress tracking for backtracking detection
        self.last_waypoint_index: Optional[int] = None
        self.frames_without_progress: int = 0
        self.PROGRESS_TIMEOUT = 180  # 3 seconds at 60fps

        # Monotonic path direction caching (for backtracking prevention)
        self.cached_path_direction: Optional[Tuple[float, float]] = None
        self.cached_monotonicity: Optional[float] = None

        # Reachable nodes caching (for performance optimization)
        self._cached_reachable_nodes: Optional[set] = None
        self._cached_reachable_from_node: Optional[Tuple[int, int]] = None

    def _pos_to_node(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert continuous position to discrete node grid cell.

        Args:
            pos: (x, y) position in pixels

        Returns:
            (node_x, node_y) in 12x12 grid coordinates
        """
        return (int(pos[0] // self.NODE_SIZE), int(pos[1] // self.NODE_SIZE))

    def _get_closest_waypoint_index(
        self, ninja_pos: Tuple[float, float]
    ) -> Optional[int]:
        """Find index of closest waypoint in cached path.

        This tracks forward progress along the path to detect stagnation.

        Args:
            ninja_pos: Current ninja position (x, y) in pixels

        Returns:
            Index of closest waypoint, or None if no cached path
        """
        if not self.cached_path:
            return None

        ninja_node_pos = (ninja_pos[0] - 24, ninja_pos[1] - 24)
        min_dist_sq = float("inf")
        closest_idx = 0

        for idx, waypoint in enumerate(self.cached_path):
            dx = waypoint[0] - ninja_node_pos[0]
            dy = waypoint[1] - ninja_node_pos[1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_idx = idx

        return closest_idx

    def _analyze_path_direction(
        self, ninja_pos: Tuple[float, float]
    ) -> Optional[Tuple[float, float, float]]:
        """Analyze cached path to determine dominant direction and monotonicity.

        Calculates the average direction vector from the path start (first waypoint)
        to the goal, weighted by inverse distance (closer waypoints matter more).
        Then computes monotonicity as the percentage of segments aligned with this
        average direction.

        Args:
            ninja_pos: Current ninja position (x, y) in pixels (unused, kept for interface compatibility)

        Returns:
            Tuple of (direction_x, direction_y, monotonicity_score) where:
            - direction_x: normalized X component of average direction (-1 to 1)
            - direction_y: normalized Y component of average direction (-1 to 1)
            - monotonicity_score: 0-1, where 1.0 = perfectly monotonic path
            Returns None if path is too short or analysis fails
        """
        if not self.cached_path or len(self.cached_path) < 2:
            return None

        # Use the first waypoint in the path as the starting point for direction analysis
        # This matches where the path actually starts (from a graph node near the ninja)
        # rather than from the ninja's exact floating-point position
        start_waypoint = self.cached_path[0]

        # Calculate weighted average direction from start waypoint to remaining waypoints
        total_weight = 0.0
        weighted_dx = 0.0
        weighted_dy = 0.0

        for waypoint in self.cached_path[1:]:  # Skip first waypoint (start)
            # Vector from start to waypoint
            dx = waypoint[0] - start_waypoint[0]
            dy = waypoint[1] - start_waypoint[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist < 1.0:  # Skip if too close (avoid division by zero)
                continue

            # Weight by inverse distance (closer waypoints more important)
            weight = 1.0 / dist
            weighted_dx += (dx / dist) * weight  # Normalize direction, then weight
            weighted_dy += (dy / dist) * weight
            total_weight += weight

        if total_weight < 0.001:  # No valid waypoints
            return None

        # Normalize weighted average direction
        avg_dx = weighted_dx / total_weight
        avg_dy = weighted_dy / total_weight
        avg_magnitude = (avg_dx * avg_dx + avg_dy * avg_dy) ** 0.5

        if avg_magnitude < 0.001:  # Degenerate case
            return None

        # Normalize to unit vector
        avg_dx /= avg_magnitude
        avg_dy /= avg_magnitude

        # Calculate monotonicity: how aligned are path segments with average direction?
        # Use consecutive waypoint segments for more accurate assessment
        alignment_scores = []
        for i in range(len(self.cached_path) - 1):
            p1 = self.cached_path[i]
            p2 = self.cached_path[i + 1]

            # Segment direction
            seg_dx = p2[0] - p1[0]
            seg_dy = p2[1] - p1[1]
            seg_length = (seg_dx * seg_dx + seg_dy * seg_dy) ** 0.5

            if seg_length < 1.0:  # Skip very short segments
                continue

            # Normalize segment
            seg_dx /= seg_length
            seg_dy /= seg_length

            # Dot product: alignment with average direction (-1 to 1)
            dot_product = seg_dx * avg_dx + seg_dy * avg_dy
            alignment_scores.append(dot_product)

        if not alignment_scores:
            return None

        # Monotonicity: percentage of path segments aligned with average direction
        # This is more robust than raw averaging because a few backward segments
        # (from navigating obstacles) won't drastically reduce the score
        # Count segments with positive alignment (moving in the right general direction)
        aligned_segments = sum(1 for score in alignment_scores if score > 0)
        monotonicity = aligned_segments / len(alignment_scores)

        # Result is already in [0, 1] range:
        # - 1.0 = all segments aligned with average direction
        # - 0.5 = half aligned, half opposed
        # - 0.0 = all segments opposed to average direction

        return (avg_dx, avg_dy, monotonicity)

    def update_path(
        self,
        ninja_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        adjacency: Dict,
        graph_data: Optional[Dict] = None,
        level_data=None,
    ) -> bool:
        """Update path cache when ninja crosses node boundaries.

        This method implements the spatial-based update strategy:
        - Only recalculate when ninja moves to a new 12x12px grid cell
        - Also recalculate when goal position changes
        - Much more efficient than time-based updates

        Args:
            ninja_pos: Current ninja position (x, y) in pixels
            goal_pos: Goal position (x, y) in pixels
            adjacency: Graph adjacency structure
            graph_data: Optional graph data with spatial_hash for fast lookup
            level_data: Optional level data for caching

        Returns:
            True if path was recalculated, False if cached path is still valid
        """
        current_ninja_node = self._pos_to_node(ninja_pos)

        # Recalculate only when necessary
        needs_recompute = (
            self.cached_path is None  # First time
            or self.cached_goal_pos != goal_pos  # Goal changed
            or self.last_ninja_node != current_ninja_node  # Node boundary crossed
        )

        if needs_recompute:
            # Use extended path calculator to get both path and distance
            self.cached_path, _ = self.path_calculator.get_path_and_distance(
                ninja_pos,
                goal_pos,
                adjacency,
                level_data=level_data,
                graph_data=graph_data,
            )
            self.cached_goal_pos = goal_pos
            self.last_ninja_node = current_ninja_node

            # Cache adjacency and graph_data for use in action masking
            self.cached_adjacency = adjacency
            self.cached_graph_data = graph_data

            # Reset progress tracking when path recalculates
            self.last_waypoint_index = None
            self.frames_without_progress = 0

            # Cache path direction analysis for monotonic backtracking detection
            direction_result = self._analyze_path_direction(ninja_pos)
            if direction_result is not None:
                dir_x, dir_y, monotonicity = direction_result
                self.cached_path_direction = (dir_x, dir_y)
                self.cached_monotonicity = monotonicity
            else:
                self.cached_path_direction = None
                self.cached_monotonicity = None

            return True

        return False

    def _is_action_opposing_monotonic_path(
        self,
        action_idx: int,
        ninja_pos: Tuple[float, float],
        ninja_vel: Tuple[float, float],
        wall_normal: float,
    ) -> bool:
        """Check if action opposes a monotonic path direction.

        This method implements precise backtracking prevention for monotonic paths
        (like the scenario in the image where the goal is directly left). It only
        masks when the path has a clear dominant direction AND the action opposes it
        AND there's no wall interaction justification.

        Masking criteria:
        1. Path has dominant direction (monotonicity > 0.7)
        2. Action moves opposite to that direction (alignment < -0.5)
        3. No wall interaction (not touching wall with wall_normal != 0)

        Note: No velocity-based exceptions - monotonic masking is strict to prevent
        any backtracking on paths with clear direction.

        Args:
            action_idx: Action to check (0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP, 4=JUMP+LEFT, 5=JUMP+RIGHT)
            ninja_pos: Current position (x, y) in pixels
            ninja_vel: Current velocity (vx, vy) in pixels/frame (unused, kept for interface compatibility)
            wall_normal: Wall normal vector (>0: wall on left, <0: wall on right, 0: no wall)

        Returns:
            True if should mask (action opposes monotonic path), False otherwise
        """
        # Note: ninja_vel parameter kept for interface compatibility but not used
        # Monotonic masking is strict - no velocity-based exceptions

        # Early exit: No cached direction analysis
        if self.cached_path_direction is None or self.cached_monotonicity is None:
            return False

        # Early exit: Path is not sufficiently monotonic (threshold: 70% alignment)
        MONOTONICITY_THRESHOLD = 0.7
        if self.cached_monotonicity < MONOTONICITY_THRESHOLD:
            return False

        # Extract cached direction
        path_dir_x, path_dir_y = self.cached_path_direction
        print(
            f"[MONOTONIC] Path direction: ({path_dir_x:.3f}, {path_dir_y:.3f}), monotonicity: {self.cached_monotonicity:.3f}"
        )

        # Map action to horizontal movement direction
        # Actions 0 (NOOP) and 3 (JUMP) have no horizontal component
        action_dir_x = 0.0
        if action_idx in [1, 4]:  # LEFT or JUMP+LEFT
            action_dir_x = -1.0
        elif action_idx in [2, 5]:  # RIGHT or JUMP+RIGHT
            action_dir_x = 1.0
        else:
            return False  # No horizontal movement, can't oppose horizontal path

        # Calculate alignment: dot product of action direction with path direction
        # Focus on horizontal component (main concern for monotonic paths)
        alignment = action_dir_x * path_dir_x
        # Threshold: Consider opposing if alignment < -0.5 (moving significantly opposite)
        OPPOSING_THRESHOLD = -0.5
        print(f"OPPOSING_THRESHOLD: {OPPOSING_THRESHOLD}")
        if alignment >= OPPOSING_THRESHOLD:
            print(f"Alignment >= OPPOSING_THRESHOLD: {alignment >= OPPOSING_THRESHOLD}")
            return False  # Not opposing enough to mask

        # Action is opposing - check for physics justifications

        # Justification 1: Wall in direction of movement (for wall jump setup)
        # Only allow opposite movement if there's a wall on the side we're moving toward
        # This is more strict than the general near-wall check used in distance-based masking
        if wall_normal != 0.0:
            print(f"Wall in direction of movement: {wall_normal}")
            # Currently touching a wall - allow movement for wall slide/jump control
            return False

        # Check for wall specifically in the direction we're trying to move
        # LEFT actions (1, 4): allow if wall on LEFT (wall_normal > 0 means wall on left)
        # RIGHT actions (2, 5): allow if wall on RIGHT (wall_normal < 0 means wall on right)
        # Note: We already checked wall_normal != 0 above, so wall_normal is 0 here
        # We don't use the general _is_near_wall() here because it's too permissive
        # (detects platform edges below, not just walls we'd interact with)

        # No other justifications for monotonic path masking
        # The whole point of this check is to be strict about preventing backtracking
        # on paths with clear direction. We don't allow velocity-based exceptions here
        # because that would defeat the purpose - if the path is monotonic, backtracking
        # is always counterproductive regardless of current velocity.
        return True

    def is_action_counterproductive(
        self,
        action_idx: int,
        ninja_pos: Tuple[float, float],
        ninja_vel: Tuple[float, float],
        ninja_state: int,
        ninja_wall_normal: float = 0.0,
        adjacency: Optional[Dict] = None,
        graph_data: Optional[Dict] = None,
    ) -> bool:
        """Check if action significantly increases path distance without physics justification.

        Conservative masking policy based on DISTANCE INCREASE:
        1. Calculate distance to goal before and after action
        2. Only mask if distance increase is significant (>36px = 3 tiles)
        3. Allow action if it has valid physics justification (wall interaction, braking, etc.)
        4. Never mask during special states or when stuck

        This approach is much more lenient than segment-based masking, allowing the agent
        to learn advanced physics exploitation (momentum building, wall jumps, etc.)

        Args:
            action_idx: Action to check (0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP, 4=JUMP+LEFT, 5=JUMP+RIGHT)
            ninja_pos: Current position (x, y) in pixels
            ninja_vel: Current velocity (vx, vy) in pixels/frame
            ninja_state: Ninja state (0=Immobile, 1=Running, 2=Ground sliding, 3=Jumping,
                        4=Falling, 5=Wall sliding, 6-9=Special states)
            ninja_wall_normal: Wall normal vector (>0: wall on left, <0: wall on right, 0: no wall)
            adjacency: Graph adjacency structure (REQUIRED for distance calculations)
            graph_data: Optional graph data with spatial_hash for fast lookup

        Returns:
            True if action should be masked (counterproductive), False otherwise

        Raises:
            RuntimeError: If adjacency is None (required for path distance calculation)
        """
        # Early exits for safety

        # No path available or path too short
        if self.cached_path is None or len(self.cached_path) < 2:
            return False

        # Don't mask during special states where control is limited
        # State 5: Wall sliding - needs all actions for wall jump control
        # States 6-9: Dead, awaiting death, celebrating, disabled
        if ninja_state in [5, 6, 7, 8, 9]:
            return False

        # Track progress to detect stagnation
        current_waypoint_idx = self._get_closest_waypoint_index(ninja_pos)

        if current_waypoint_idx is not None:
            if self.last_waypoint_index is not None:
                # Check if making forward progress
                if current_waypoint_idx > self.last_waypoint_index:
                    # Moving forward - reset timeout
                    self.frames_without_progress = 0
                elif current_waypoint_idx == self.last_waypoint_index:
                    # Stuck on same waypoint
                    self.frames_without_progress += 1
                else:
                    # Moving backward - increment but allow (might be optimal)
                    self.frames_without_progress += 1

            self.last_waypoint_index = current_waypoint_idx

        # If stuck for too long, disable masking to allow exploration
        if self.frames_without_progress > self.PROGRESS_TIMEOUT:
            return False

        # Monotonic path check (fast, before expensive distance calculations)
        # This provides precise backtracking prevention for paths with clear direction
        monotonic_result = self._is_action_opposing_monotonic_path(
            action_idx, ninja_pos, ninja_vel, ninja_wall_normal
        )
        print(f"[ACTION {action_idx}] Monotonic check result: {monotonic_result}")
        if monotonic_result:
            print(f"[ACTION {action_idx}] MASKED by monotonic check")
            return True  # Mask due to backtracking on monotonic path
        print(
            f"[ACTION {action_idx}] Monotonic check passed, continuing to distance-based check"
        )

        # Use cached adjacency/graph_data if not provided as parameters
        if adjacency is None:
            adjacency = self.cached_adjacency
        if graph_data is None:
            graph_data = self.cached_graph_data

        # Validate required parameters
        if adjacency is None:
            raise RuntimeError(
                "is_action_counterproductive requires adjacency graph but it is None. "
                "Ensure update_path() is called before action masking."
            )

        # Distance-based masking: Calculate current and estimated distances
        goal_pos = self.cached_path[-1]  # Final waypoint is the goal

        try:
            # Calculate current distance to goal
            current_distance = self._calculate_distance_to_goal(
                ninja_pos, goal_pos, adjacency, graph_data
            )

            # Estimate position after action (simple 1-frame physics approximation)
            estimated_pos = self._estimate_position_after_action(
                action_idx, ninja_pos, ninja_vel, ninja_state
            )

            # Calculate estimated distance after action
            estimated_distance = self._calculate_distance_to_goal(
                estimated_pos, goal_pos, adjacency, graph_data
            )

        except Exception:
            # If distance calculation fails, don't mask (be conservative)
            return False

        # Calculate distance increase
        distance_increase = estimated_distance - current_distance

        # Very conservative threshold: only mask if significant increase
        # 36px = 3 tiles, allows for trajectory arcs and momentum building
        DISTANCE_THRESHOLD = 36.0
        if distance_increase < DISTANCE_THRESHOLD:
            return False  # Not significant enough to mask

        # Check for valid physics justifications
        has_justification = self._has_physics_justification(
            action_idx, ninja_pos, ninja_vel, ninja_state, ninja_wall_normal
        )
        print(
            f"[ACTION {action_idx}] Distance increase: {distance_increase:.2f}px, has_justification: {has_justification}"
        )
        if has_justification:
            print(f"[ACTION {action_idx}] NOT masked (has physics justification)")
            return False  # Has valid reason, don't mask

        # Only now mask - action increases distance significantly with no justification
        print(f"[ACTION {action_idx}] MASKED by distance-based check")
        return True

    def reset(self):
        """Reset the predictor state (call on level change)."""
        self.cached_path = None
        self.cached_goal_pos = None
        self.last_ninja_node = None
        self.cached_adjacency = None
        self.cached_graph_data = None
        self.last_waypoint_index = None
        self.frames_without_progress = 0
        self.cached_path_direction = None
        self.cached_monotonicity = None
        self._cached_reachable_nodes = None
        self._cached_reachable_from_node = None

    def _calculate_distance_to_goal(
        self,
        pos: Tuple[float, float],
        goal: Tuple[float, float],
        adjacency: Dict,
        graph_data: Optional[Dict] = None,
    ) -> float:
        """Calculate path distance from position to goal using cached path calculator.

        Args:
            pos: Current position (x, y) in pixels
            goal: Goal position (x, y) in pixels
            adjacency: Graph adjacency structure (REQUIRED)
            graph_data: Optional graph data with spatial_hash

        Returns:
            Shortest path distance in pixels, or float('inf') if unreachable

        Raises:
            RuntimeError: If path_calculator or adjacency is None
        """
        if self.path_calculator is None:
            raise RuntimeError(
                "PathGuidancePredictor requires path_calculator but it is None. "
                "This should never happen - check initialization."
            )

        if adjacency is None:
            raise RuntimeError(
                "PathGuidancePredictor requires adjacency graph but it is None. "
                "This indicates graph building failed or is disabled."
            )

        # Use path calculator to get actual shortest path distance
        distance = self.path_calculator.get_distance(
            pos,
            goal,
            adjacency,
            cache_key=None,  # Don't cache these intermediate distance checks
            graph_data=graph_data,
        )

        return distance

    def _estimate_position_after_action(
        self,
        action_idx: int,
        pos: Tuple[float, float],
        vel: Tuple[float, float],
        state: int,
    ) -> Tuple[float, float]:
        """Estimate position after 1 frame of action using simple physics approximation.

        This is a conservative approximation that only considers immediate velocity changes
        from input acceleration, not full trajectory simulation.

        Args:
            action_idx: Action index (0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP, 4=JUMP+LEFT, 5=JUMP+RIGHT)
            pos: Current position (x, y) in pixels
            vel: Current velocity (vx, vy) in pixels/frame
            state: Ninja state (0-9)

        Returns:
            Estimated position after 1 frame
        """
        # Import physics constants
        from nclone.constants import GROUND_ACCEL, AIR_ACCEL, MAX_HOR_SPEED

        # Determine horizontal input from action
        hor_input = 0
        if action_idx in [1, 4]:  # LEFT or JUMP+LEFT
            hor_input = -1
        elif action_idx in [2, 5]:  # RIGHT or JUMP+RIGHT
            hor_input = 1

        # Estimate velocity change from horizontal input (simplified)
        vx, vy = vel
        on_ground = state in [0, 1, 2]  # Immobile, Running, Ground sliding
        accel = GROUND_ACCEL if on_ground else AIR_ACCEL

        # Apply horizontal acceleration
        vx_new = vx + accel * hor_input
        # Clamp to max speed
        if abs(vx_new) > MAX_HOR_SPEED:
            vx_new = MAX_HOR_SPEED if vx_new > 0 else -MAX_HOR_SPEED

        # Conservative: don't try to predict jump trajectory changes
        # Just use current vertical velocity
        vy_new = vy

        # Estimate new position (1 frame forward)
        pos_x_new = pos[0] + vx_new
        pos_y_new = pos[1] + vy_new

        return (pos_x_new, pos_y_new)

    def _has_physics_justification(
        self,
        action_idx: int,
        pos: Tuple[float, float],
        vel: Tuple[float, float],
        state: int,
        wall_normal: float,
    ) -> bool:
        """Check if action has valid physics justification despite increasing path distance.

        Returns True if ANY of these conditions apply:
        - Near wall (touching OR within 24px) - allows wall interaction/jump setup
        - High opposite velocity - allows braking/control
        - Near path waypoint transition - allows trajectory adjustment
        - Goal has vertical component and action includes jump

        Args:
            action_idx: Action index (0-5)
            pos: Current position (x, y)
            vel: Current velocity (vx, vy)
            state: Ninja state
            wall_normal: Wall normal (>0 = wall on left, <0 = wall on right, 0 = no wall)

        Returns:
            True if action has physics justification, False otherwise
        """
        # Justification 1: Near REACHABLE wall (allow all actions for wall interaction/setup)
        # Check if touching wall (wall_normal != 0) OR near reachable wall (within threshold)
        # Use adjacency-aware check to avoid allowing actions toward unreachable walls
        is_near_reachable = self._is_near_reachable_wall(pos, threshold=24.0)
        if wall_normal != 0.0 or is_near_reachable:
            print(
                f"[HAS_PHYSICS_JUSTIFICATION] Near wall: touching={wall_normal != 0.0}, reachable={is_near_reachable}"
            )
            return True

        # Justification 2: High opposite velocity (braking/control)
        vx, vy = vel
        VELOCITY_THRESHOLD = 2.0  # pixels/frame

        # Check if action opposes high velocity (braking)
        if action_idx == 1 and vx > VELOCITY_THRESHOLD:  # LEFT when moving fast RIGHT
            return True
        if action_idx == 2 and vx < -VELOCITY_THRESHOLD:  # RIGHT when moving fast LEFT
            return True

        # Justification 3: Near path waypoint (allow adjustments near targets)
        if self.cached_path and len(self.cached_path) >= 2:
            # Check distance to next waypoint
            next_waypoint = self.cached_path[min(1, len(self.cached_path) - 1)]
            dx = next_waypoint[0] - pos[0]
            dy = next_waypoint[1] - pos[1]
            dist_to_waypoint = (dx * dx + dy * dy) ** 0.5

            # Within 2 tiles (48px) of waypoint, allow adjustments
            if dist_to_waypoint < 48.0:
                return True

        # Justification 4: Goal has vertical component and action includes jump
        if action_idx in [3, 4, 5]:  # JUMP actions
            if self.cached_path and len(self.cached_path) >= 1:
                goal = self.cached_path[-1]
                # Check if goal is significantly above current position
                dy_to_goal = (
                    pos[1] - goal[1]
                )  # Positive if goal is above (y increases downward)
                if dy_to_goal > 24.0:  # Goal is more than 1 tile above
                    return True

        # No justification found
        return False

    def _is_near_wall(self, pos: Tuple[float, float], threshold: float = 24.0) -> bool:
        """Check if ninja is near a vertical wall segment within threshold distance.

        Uses cached tile segments from sim to efficiently check proximity to walls.
        Walls are defined as completely vertical segments (dx=0).

        NOTE: This method does NOT respect the adjacency graph - it detects walls
        through solid obstacles. Use _is_near_reachable_wall() for adjacency-aware checks.

        Args:
            pos: Ninja position (x, y) in pixels
            threshold: Distance threshold in pixels (default 24px = 1 tile)

        Returns:
            True if within threshold distance of a wall, False otherwise
        """
        if not self.sim:
            return False

        from nclone.physics import gather_segments_from_region
        from nclone.constants import NINJA_RADIUS

        # Gather segments in expanded radius (ninja radius + threshold)
        x, y = pos
        search_radius = NINJA_RADIUS + threshold
        segments = gather_segments_from_region(
            self.sim,
            x - search_radius,
            y - search_radius,
            x + search_radius,
            y + search_radius,
        )

        # Check each segment
        for segment in segments:
            # Get segment endpoints
            result = segment.get_closest_point(x, y)
            closest_x, closest_y = result[1], result[2]

            # Calculate distance to segment
            dx = x - closest_x
            dy = y - closest_y
            dist = (dx * dx + dy * dy) ** 0.5

            # Check if segment is vertical (wall) and within threshold
            # Vertical segments have endpoints with same x-coordinate
            if hasattr(segment, "x1") and hasattr(segment, "x2"):
                is_vertical = abs(segment.x2 - segment.x1) < 0.1
                if is_vertical and dist <= threshold:
                    return True

        return False

    def _is_near_reachable_wall(
        self, pos: Tuple[float, float], threshold: float = 24.0
    ) -> bool:
        """Check if ninja is near a wall that is reachable via the adjacency graph.

        This method respects the adjacency graph, only returning True if there's
        a wall within threshold distance that the ninja can actually reach
        (not blocked by solid obstacles).

        Uses flood-fill to find ALL reachable nodes from the ninja's position, then checks
        if any reachable nodes within the threshold distance have walls nearby.

        Performance: Caches reachable nodes per ninja node position to avoid repeated flood-fills.

        Args:
            pos: Ninja position (x, y) in pixels
            threshold: Distance threshold in pixels (default 24px = 1 tile)

        Returns:
            True if within threshold distance of a REACHABLE wall, False otherwise
        """
        # If no adjacency graph available, fall back to conservative check
        if self.cached_adjacency is None:
            return False

        # Get current ninja node for caching
        from nclone.graph.reachability.pathfinding_utils import (
            find_closest_node_to_position,
            extract_spatial_lookups_from_graph_data,
        )

        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            self.cached_graph_data
        )

        ninja_node = find_closest_node_to_position(
            pos,
            self.cached_adjacency,
            threshold=10.0,  # Ninja radius
            spatial_hash=spatial_hash,
            subcell_lookup=subcell_lookup,
        )

        if ninja_node is None:
            return False

        # Check cache - reuse flood-fill results if ninja is still on same node
        if (
            self._cached_reachable_nodes is not None
            and self._cached_reachable_from_node == ninja_node
        ):
            reachable_nodes = self._cached_reachable_nodes
        else:
            # Cache miss - compute flood-fill and cache results
            reachable_nodes = _flood_fill_reachable_nodes(
                pos, self.cached_adjacency, self.cached_graph_data
            )
            self._cached_reachable_nodes = reachable_nodes
            self._cached_reachable_from_node = ninja_node

        if not reachable_nodes:
            return False

        # Check each reachable node to see if it's within threshold and has a wall
        for node in reachable_nodes:
            # Convert node position to world space (nodes are in tile data space)
            node_world_pos = (node[0] + 24, node[1] + 24)

            # Calculate distance from ninja to this node
            dx = pos[0] - node_world_pos[0]
            dy = pos[1] - node_world_pos[1]
            node_distance = (dx * dx + dy * dy) ** 0.5

            # Only check nodes within threshold distance
            if node_distance > threshold:
                continue

            # Calculate remaining distance budget for wall check
            # If node is close to threshold, we have less budget for wall distance
            remaining_distance = threshold - node_distance
            wall_check_threshold = max(1.0, min(12.0, remaining_distance))

            # Check if this node is near a wall within the remaining budget
            if self._is_near_wall(node_world_pos, threshold=wall_check_threshold):
                return True

        return False

    def get_reachable_wall_segments_for_debug(
        self, ninja_pos: Tuple[float, float], threshold: float = 24.0
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get list of reachable wall segments for debug visualization.

        Returns wall segments (as line segments) that are reachable via the adjacency
        graph within the threshold distance. Used for debugging wall detection logic.

        Uses flood-fill to find ALL reachable nodes from the ninja's position, then filters
        for nodes within threshold distance and gathers wall segments near those nodes.

        Performance: Reuses cached reachable nodes if available from previous calls.

        Args:
            ninja_pos: Current ninja position (x, y) in pixels
            threshold: Distance threshold in pixels (default 24px = 1 tile)

        Returns:
            List of wall segments as ((x1, y1), (x2, y2)) tuples in world space
        """
        wall_segments = []

        # If no adjacency graph available, return empty
        if self.cached_adjacency is None or not self.sim:
            return wall_segments

        # Import required utilities
        from nclone.physics import gather_segments_from_region
        from nclone.constants import NINJA_RADIUS
        from nclone.graph.reachability.pathfinding_utils import (
            find_closest_node_to_position,
            extract_spatial_lookups_from_graph_data,
        )

        # Get current ninja node for cache lookup
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            self.cached_graph_data
        )

        ninja_node = find_closest_node_to_position(
            ninja_pos,
            self.cached_adjacency,
            threshold=10.0,  # Ninja radius
            spatial_hash=spatial_hash,
            subcell_lookup=subcell_lookup,
        )

        if ninja_node is None:
            return wall_segments

        # Check cache - reuse flood-fill results if ninja is still on same node
        if (
            self._cached_reachable_nodes is not None
            and self._cached_reachable_from_node == ninja_node
        ):
            reachable_nodes = self._cached_reachable_nodes
        else:
            # Cache miss - compute flood-fill and cache results
            reachable_nodes = _flood_fill_reachable_nodes(
                ninja_pos, self.cached_adjacency, self.cached_graph_data
            )
            self._cached_reachable_nodes = reachable_nodes
            self._cached_reachable_from_node = ninja_node

        if not reachable_nodes:
            return wall_segments

        # Filter for nodes within threshold distance and gather wall segments
        for node in reachable_nodes:
            # Convert node position to world space (nodes are in tile data space)
            node_world_x = node[0] + 24
            node_world_y = node[1] + 24

            # Calculate distance from ninja to this node
            dx = ninja_pos[0] - node_world_x
            dy = ninja_pos[1] - node_world_y
            node_distance = (dx * dx + dy * dy) ** 0.5

            # Only check nodes within threshold distance
            if node_distance > threshold:
                continue

            # Gather segments near this node
            search_radius = NINJA_RADIUS + 12.0  # Small radius for node check
            segments = gather_segments_from_region(
                self.sim,
                node_world_x - search_radius,
                node_world_y - search_radius,
                node_world_x + search_radius,
                node_world_y + search_radius,
            )

            # Filter for vertical walls that are actually within threshold from ninja
            for segment in segments:
                if hasattr(segment, "x1") and hasattr(segment, "x2"):
                    is_vertical = abs(segment.x2 - segment.x1) < 0.1
                    if is_vertical:
                        # Calculate distance from ninja to the wall segment
                        # For vertical walls, find closest point on segment to ninja
                        wall_x = segment.x1  # Vertical wall has constant x
                        # Clamp ninja y to segment bounds
                        wall_y = max(
                            min(segment.y1, segment.y2),
                            min(ninja_pos[1], max(segment.y1, segment.y2)),
                        )

                        # Distance from ninja to closest point on wall
                        wall_dx = ninja_pos[0] - wall_x
                        wall_dy = ninja_pos[1] - wall_y
                        wall_distance = (wall_dx * wall_dx + wall_dy * wall_dy) ** 0.5

                        # Only add wall if it's actually within threshold from ninja
                        if wall_distance <= threshold:
                            seg_tuple = (
                                (segment.x1, segment.y1),
                                (segment.x2, segment.y2),
                            )
                            if seg_tuple not in wall_segments:
                                wall_segments.append(seg_tuple)

        return wall_segments

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the predictor state for debugging.

        Returns:
            Dictionary with current state information
        """
        stats = {
            "has_cached_path": self.cached_path is not None,
            "path_length": len(self.cached_path) if self.cached_path else 0,
            "last_ninja_node": self.last_ninja_node,
            "strict_mode": self.strict_mode,
            "frames_without_progress": self.frames_without_progress,
            "cached_path_direction": self.cached_path_direction,
            "cached_monotonicity": self.cached_monotonicity,
        }

        return stats
