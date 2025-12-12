"""Path-based waypoint extraction from optimal A* paths.

Extracts dense, strategic waypoints from optimal paths to provide immediate
guidance for complex navigation. Unlike trajectory-learning systems, path waypoints
are available from the first episode and provide deterministic guidance.

Multi-tier extraction system:
1. Critical waypoints: Physics transitions, sharp turns (high value: 1.2-1.8)
2. Strategic waypoints: Medium turns, segment midpoints (medium value: 0.8-1.2)
3. Progress waypoints: Regular spacing for dense coverage (base value: 0.4-0.8)

Expected density: 15-30 waypoints per path depending on complexity.
"""

import logging
import math
from typing import List, Tuple, Dict, Any, NamedTuple
from collections import OrderedDict

from ...graph.reachability.pathfinding_utils import NODE_WORLD_COORD_OFFSET

logger = logging.getLogger(__name__)


class PathWaypoint(NamedTuple):
    """Waypoint extracted from optimal path.
    
    Attributes:
        position: World coordinates (x, y) in pixels
        waypoint_type: Type of waypoint for prioritization
        value: Importance score (0.4-1.8) with progress gradient
        phase: "pre_switch" or "post_switch" for phase-aware filtering
        node_index: Index in original path for ordering
        physics_state: "grounded", "aerial", or "walled"
        curvature: Turn angle in degrees (0-180)
        exit_direction: Direction vector (dx, dy) to next waypoint on path (normalized)
    """
    position: Tuple[float, float]
    waypoint_type: str
    value: float
    phase: str
    node_index: int
    physics_state: str
    curvature: float
    exit_direction: Optional[Tuple[float, float]] = None


class PathWaypointExtractor:
    """Extracts dense waypoints from optimal A* paths.
    
    Provides immediate, deterministic guidance for complex navigation
    by identifying strategic points along the optimal path.
    """
    
    def __init__(
        self,
        progress_spacing: float = 100.0,
        min_turn_angle: float = 45.0,
        cluster_radius: float = 40.0,
        segment_min_length: float = 150.0,
        max_cache_size: int = 100,
    ):
        """Initialize path waypoint extractor.
        
        Args:
            progress_spacing: Distance between progress checkpoint waypoints (pixels)
            min_turn_angle: Minimum angle for turn detection (degrees)
            cluster_radius: Radius for merging nearby waypoints (pixels)
            segment_min_length: Minimum length for segment midpoint insertion (pixels)
            max_cache_size: Maximum number of levels to cache
        """
        self.progress_spacing = progress_spacing
        self.min_turn_angle = min_turn_angle
        self.cluster_radius = cluster_radius
        self.segment_min_length = segment_min_length
        
        # LRU cache: level_id -> List[PathWaypoint]
        self._waypoint_cache: OrderedDict[str, List[PathWaypoint]] = OrderedDict()
        self._max_cache_size = max_cache_size
        
        # Statistics
        self.total_waypoints_extracted = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(
            f"PathWaypointExtractor initialized: "
            f"progress_spacing={progress_spacing}px, "
            f"min_angle={min_turn_angle}°, "
            f"cluster_radius={cluster_radius}px"
        )
    
    def extract_waypoints_from_paths(
        self,
        spawn_to_switch_path: List[Tuple[int, int]],
        switch_to_exit_path: List[Tuple[int, int]],
        physics_cache: Dict[Tuple[int, int], Dict[str, bool]],
        level_id: str,
        use_cache: bool = True,
    ) -> List[PathWaypoint]:
        """Extract waypoints from both path segments (spawn→switch, switch→exit).
        
        Args:
            spawn_to_switch_path: Path nodes from spawn to switch
            switch_to_exit_path: Path nodes from switch to exit
            physics_cache: Pre-computed physics properties per node
            level_id: Level identifier for caching
            use_cache: If True, use cached waypoints if available
            
        Returns:
            List of PathWaypoint objects for both phases
        """
        # Check cache first
        if use_cache and level_id in self._waypoint_cache:
            self._waypoint_cache.move_to_end(level_id)  # LRU update
            self.cache_hits += 1
            cached_waypoints = self._waypoint_cache[level_id]
            logger.debug(
                f"Using cached waypoints for level {level_id}: "
                f"{len(cached_waypoints)} waypoints"
            )
            return list(cached_waypoints)  # Return copy
        
        self.cache_misses += 1
        
        # Extract waypoints from each path segment
        pre_switch_waypoints = self._extract_waypoints_from_path(
            path_nodes=spawn_to_switch_path,
            physics_cache=physics_cache,
            phase="pre_switch",
        )
        
        post_switch_waypoints = self._extract_waypoints_from_path(
            path_nodes=switch_to_exit_path,
            physics_cache=physics_cache,
            phase="post_switch",
        )
        
        # Combine both phases
        all_waypoints = pre_switch_waypoints + post_switch_waypoints
        
        # Cache the result
        self._waypoint_cache[level_id] = all_waypoints
        self._waypoint_cache.move_to_end(level_id)
        
        # Evict oldest entry if cache exceeds max size
        if len(self._waypoint_cache) > self._max_cache_size:
            self._waypoint_cache.popitem(last=False)
        
        self.total_waypoints_extracted += len(all_waypoints)
        
        logger.info(
            f"Extracted {len(all_waypoints)} path waypoints for level {level_id}: "
            f"{len(pre_switch_waypoints)} pre-switch, {len(post_switch_waypoints)} post-switch"
        )
        
        return all_waypoints
    
    def _extract_waypoints_from_path(
        self,
        path_nodes: List[Tuple[int, int]],
        physics_cache: Dict[Tuple[int, int], Dict[str, bool]],
        phase: str = "pre_switch",
    ) -> List[PathWaypoint]:
        """Extract waypoints from a single path segment using three-pass algorithm.
        
        Args:
            path_nodes: List of (x, y) node positions in path order
            physics_cache: Pre-computed physics properties per node
            phase: "pre_switch" or "post_switch"
            
        Returns:
            List of PathWaypoint objects (deduplicated and clustered)
        """
        if len(path_nodes) < 2:
            return []
        
        waypoints = []
        
        # Pass 1: Critical waypoints (physics transitions, sharp turns)
        waypoints.extend(
            self._detect_physics_transitions(path_nodes, physics_cache, phase)
        )
        waypoints.extend(
            self._detect_sharp_turns(path_nodes, phase, min_angle=90.0)
        )
        waypoints.extend(
            self._detect_direction_reversals(path_nodes, phase)
        )
        
        # Pass 2: Strategic waypoints (medium turns, segment midpoints)
        waypoints.extend(
            self._detect_medium_turns(path_nodes, phase, min_angle=self.min_turn_angle, max_angle=90.0)
        )
        waypoints.extend(
            self._add_segment_midpoints(path_nodes, waypoints, physics_cache, phase)
        )
        
        # Pass 3: Progress waypoints (regular spacing)
        waypoints.extend(
            self._add_progress_checkpoints(path_nodes, waypoints, phase)
        )
        
        # Apply progress gradient to values (waypoints near goal worth more)
        waypoints = self._apply_progress_gradient(waypoints, len(path_nodes))
        
        # Cluster and deduplicate
        waypoints = self._cluster_waypoints(waypoints)
        
        # Compute exit directions for sequential guidance
        waypoints = self._compute_exit_directions(waypoints, path_nodes)
        
        logger.debug(
            f"Extracted {len(waypoints)} waypoints for {phase} path "
            f"(length: {len(path_nodes)} nodes)"
        )
        
        return waypoints
    
    def _detect_physics_transitions(
        self,
        path_nodes: List[Tuple[int, int]],
        physics_cache: Dict[Tuple[int, int], Dict[str, bool]],
        phase: str,
    ) -> List[PathWaypoint]:
        """Detect physics state transitions (grounded↔aerial, wall proximity).
        
        These are the most critical waypoints as they indicate where the agent
        must change their movement strategy (jump, land, navigate corners).
        
        Args:
            path_nodes: Path node positions
            physics_cache: Physics properties per node
            phase: Waypoint phase
            
        Returns:
            List of physics transition waypoints
        """
        waypoints = []
        
        for i in range(len(path_nodes) - 1):
            curr_node = path_nodes[i]
            next_node = path_nodes[i + 1]
            
            curr_physics = physics_cache.get(curr_node, {})
            next_physics = physics_cache.get(next_node, {})
            
            # Detect grounded ↔ aerial transitions
            curr_grounded = curr_physics.get("grounded", False)
            next_grounded = next_physics.get("grounded", False)
            
            if curr_grounded != next_grounded:
                transition_type = "takeoff" if curr_grounded else "landing"
                waypoints.append(PathWaypoint(
                    position=(float(curr_node[0]) + NODE_WORLD_COORD_OFFSET, float(curr_node[1]) + NODE_WORLD_COORD_OFFSET),
                    waypoint_type=f"physics_transition_{transition_type}",
                    value=1.5,  # High value, will be scaled by progress gradient
                    phase=phase,
                    node_index=i,
                    physics_state="grounded" if curr_grounded else "aerial",
                    curvature=0.0,
                ))
                
                logger.debug(
                    f"Physics transition ({transition_type}) at node {i}: {curr_node}"
                )
        
        return waypoints
    
    def _detect_sharp_turns(
        self,
        path_nodes: List[Tuple[int, int]],
        phase: str,
        min_angle: float = 90.0,
    ) -> List[PathWaypoint]:
        """Detect sharp turns in path (>90° curvature).
        
        Sharp turns indicate critical decision points where the agent must
        change direction significantly.
        
        Args:
            path_nodes: Path node positions
            phase: Waypoint phase
            min_angle: Minimum turn angle to detect (degrees)
            
        Returns:
            List of sharp turn waypoints
        """
        waypoints = []
        
        for i in range(1, len(path_nodes) - 1):
            prev_node = path_nodes[i - 1]
            curr_node = path_nodes[i]
            next_node = path_nodes[i + 1]
            
            angle = self._calculate_turn_angle(prev_node, curr_node, next_node)
            
            if angle >= min_angle:
                waypoints.append(PathWaypoint(
                    position=(float(curr_node[0]) + NODE_WORLD_COORD_OFFSET, float(curr_node[1]) + NODE_WORLD_COORD_OFFSET),
                    waypoint_type="sharp_turn",
                    value=1.8 * (angle / 180.0),  # Scale by sharpness
                    phase=phase,
                    node_index=i,
                    physics_state="unknown",  # Will be filled if needed
                    curvature=angle,
                ))
                
                logger.debug(
                    f"Sharp turn ({angle:.1f}°) at node {i}: {curr_node}"
                )
        
        return waypoints
    
    def _detect_medium_turns(
        self,
        path_nodes: List[Tuple[int, int]],
        phase: str,
        min_angle: float = 45.0,
        max_angle: float = 90.0,
    ) -> List[PathWaypoint]:
        """Detect medium turns in path (45-90° curvature).
        
        Args:
            path_nodes: Path node positions
            phase: Waypoint phase
            min_angle: Minimum turn angle (degrees)
            max_angle: Maximum turn angle (degrees)
            
        Returns:
            List of medium turn waypoints
        """
        waypoints = []
        
        for i in range(1, len(path_nodes) - 1):
            prev_node = path_nodes[i - 1]
            curr_node = path_nodes[i]
            next_node = path_nodes[i + 1]
            
            angle = self._calculate_turn_angle(prev_node, curr_node, next_node)
            
            if min_angle <= angle < max_angle:
                waypoints.append(PathWaypoint(
                    position=(float(curr_node[0]) + NODE_WORLD_COORD_OFFSET, float(curr_node[1]) + NODE_WORLD_COORD_OFFSET),
                    waypoint_type="medium_turn",
                    value=1.2 * (angle / 90.0),  # Scale by turn sharpness
                    phase=phase,
                    node_index=i,
                    physics_state="unknown",
                    curvature=angle,
                ))
                
                logger.debug(
                    f"Medium turn ({angle:.1f}°) at node {i}: {curr_node}"
                )
        
        return waypoints
    
    def _detect_direction_reversals(
        self,
        path_nodes: List[Tuple[int, int]],
        phase: str,
    ) -> List[PathWaypoint]:
        """Detect direction reversals in path (going LEFT after RIGHT, etc.).
        
        Direction reversals are critical for levels requiring backtracking
        or counter-intuitive navigation.
        
        Args:
            path_nodes: Path node positions
            phase: Waypoint phase
            
        Returns:
            List of direction reversal waypoints
        """
        waypoints = []
        
        if len(path_nodes) < 5:
            return []  # Need enough history to detect reversals
        
        # Track dominant direction over sliding window
        window_size = 3
        
        for i in range(window_size, len(path_nodes) - window_size):
            # Calculate direction in previous window
            prev_start = path_nodes[i - window_size]
            prev_end = path_nodes[i]
            prev_dx = prev_end[0] - prev_start[0]
            prev_dy = prev_end[1] - prev_start[1]
            
            # Calculate direction in next window
            next_start = path_nodes[i]
            next_end = path_nodes[i + window_size]
            next_dx = next_end[0] - next_start[0]
            next_dy = next_end[1] - next_start[1]
            
            # Detect horizontal reversal (going LEFT after RIGHT, or vice versa)
            if abs(prev_dx) > abs(prev_dy) * 2 and abs(next_dx) > abs(next_dy) * 2:
                # Both windows are dominated by horizontal movement
                if prev_dx * next_dx < 0:  # Opposite signs = reversal
                    waypoints.append(PathWaypoint(
                        position=(float(path_nodes[i][0]) + NODE_WORLD_COORD_OFFSET, float(path_nodes[i][1]) + NODE_WORLD_COORD_OFFSET),
                        waypoint_type="direction_reversal_horizontal",
                        value=1.8,  # High value for reversals
                        phase=phase,
                        node_index=i,
                        physics_state="unknown",
                        curvature=180.0,  # Maximum curvature for reversal
                    ))
                    
                    logger.debug(
                        f"Horizontal direction reversal at node {i}: {path_nodes[i]}"
                    )
            
            # Detect vertical reversal (going UP after DOWN, or vice versa)
            if abs(prev_dy) > abs(prev_dx) * 2 and abs(next_dy) > abs(next_dx) * 2:
                # Both windows are dominated by vertical movement
                if prev_dy * next_dy < 0:  # Opposite signs = reversal
                    waypoints.append(PathWaypoint(
                        position=(float(path_nodes[i][0]) + NODE_WORLD_COORD_OFFSET, float(path_nodes[i][1]) + NODE_WORLD_COORD_OFFSET),
                        waypoint_type="direction_reversal_vertical",
                        value=1.8,  # High value for reversals
                        phase=phase,
                        node_index=i,
                        physics_state="unknown",
                        curvature=180.0,
                    ))
                    
                    logger.debug(
                        f"Vertical direction reversal at node {i}: {path_nodes[i]}"
                    )
        
        return waypoints
    
    def _add_segment_midpoints(
        self,
        path_nodes: List[Tuple[int, int]],
        existing_waypoints: List[PathWaypoint],
        physics_cache: Dict[Tuple[int, int], Dict[str, bool]],
        phase: str,
    ) -> List[PathWaypoint]:
        """Add midpoint waypoints on long straight sections.
        
        Splits path into segments between existing waypoints and adds
        midpoints for segments longer than threshold.
        
        Args:
            path_nodes: Path node positions
            existing_waypoints: Already detected waypoints
            physics_cache: Physics properties per node
            phase: Waypoint phase
            
        Returns:
            List of segment midpoint waypoints
        """
        waypoints = []
        
        # Build set of indices that already have waypoints
        waypoint_indices = {wp.node_index for wp in existing_waypoints}
        
        # Find segments between waypoints
        last_waypoint_idx = 0
        
        for i in range(len(path_nodes)):
            if i in waypoint_indices or i == len(path_nodes) - 1:
                # Found waypoint or end - check if segment is long enough
                segment_start = last_waypoint_idx
                segment_end = i
                segment_nodes = path_nodes[segment_start:segment_end + 1]
                
                if len(segment_nodes) >= 3:  # Need at least 3 nodes for midpoint
                    segment_length = self._calculate_path_length(segment_nodes)
                    
                    if segment_length > self.segment_min_length:
                        # Add midpoint waypoint
                        midpoint_idx = segment_start + len(segment_nodes) // 2
                        midpoint_node = path_nodes[midpoint_idx]
                        
                        # Get physics state for this node
                        physics = physics_cache.get(midpoint_node, {})
                        physics_state = self._get_physics_state_string(physics)
                        
                        waypoints.append(PathWaypoint(
                            position=(float(midpoint_node[0]) + NODE_WORLD_COORD_OFFSET, float(midpoint_node[1]) + NODE_WORLD_COORD_OFFSET),
                            waypoint_type="segment_midpoint",
                            value=1.0,  # Medium value
                            phase=phase,
                            node_index=midpoint_idx,
                            physics_state=physics_state,
                            curvature=0.0,
                        ))
                        
                        logger.debug(
                            f"Segment midpoint at node {midpoint_idx}: {midpoint_node} "
                            f"(segment length: {segment_length:.1f}px)"
                        )
                
                last_waypoint_idx = i
        
        return waypoints
    
    def _add_progress_checkpoints(
        self,
        path_nodes: List[Tuple[int, int]],
        existing_waypoints: List[PathWaypoint],
        phase: str,
    ) -> List[PathWaypoint]:
        """Add regular progress checkpoints along path for dense coverage.
        
        Ensures no gaps longer than progress_spacing without a waypoint.
        
        Args:
            path_nodes: Path node positions
            existing_waypoints: Already detected waypoints
            phase: Waypoint phase
            
        Returns:
            List of progress checkpoint waypoints
        """
        waypoints = []
        
        # Build set of positions that already have waypoints (with tolerance)
        existing_positions = {wp.position for wp in existing_waypoints}
        
        # Track cumulative distance along path
        cumulative_distance = 0.0
        last_checkpoint_distance = 0.0
        
        for i in range(len(path_nodes) - 1):
            curr_node = path_nodes[i]
            next_node = path_nodes[i + 1]
            
            # Calculate edge length
            dx = next_node[0] - curr_node[0]
            dy = next_node[1] - curr_node[1]
            edge_length = math.sqrt(dx * dx + dy * dy)
            
            cumulative_distance += edge_length
            
            # Check if we need a checkpoint
            if cumulative_distance - last_checkpoint_distance >= self.progress_spacing:
                curr_pos = (float(curr_node[0]) + NODE_WORLD_COORD_OFFSET, float(curr_node[1]) + NODE_WORLD_COORD_OFFSET)
                
                # Only add if no existing waypoint nearby
                if not self._has_waypoint_within_radius(
                    existing_positions, curr_pos, self.cluster_radius
                ):
                    waypoints.append(PathWaypoint(
                        position=curr_pos,
                        waypoint_type="progress_checkpoint",
                        value=0.6,  # Base value for regular progress
                        phase=phase,
                        node_index=i,
                        physics_state="unknown",
                        curvature=0.0,
                    ))
                    
                    # Update existing positions and last checkpoint
                    existing_positions.add(curr_pos)
                    last_checkpoint_distance = cumulative_distance
                    
                    logger.debug(
                        f"Progress checkpoint at node {i}: {curr_node} "
                        f"(distance: {cumulative_distance:.1f}px)"
                    )
        
        return waypoints
    
    def _apply_progress_gradient(
        self,
        waypoints: List[PathWaypoint],
        total_path_length: int,
    ) -> List[PathWaypoint]:
        """Apply progress gradient to waypoint values.
        
        Waypoints closer to the goal are worth more (up to +30%) to provide
        stronger guidance in the final critical section.
        
        Args:
            waypoints: Waypoints to apply gradient to
            total_path_length: Total number of nodes in path
            
        Returns:
            Waypoints with adjusted values
        """
        if total_path_length <= 1:
            return waypoints
        
        adjusted = []
        
        for wp in waypoints:
            # Calculate progress ratio (0.0 at start, 1.0 at end)
            progress_ratio = wp.node_index / max(1, total_path_length - 1)
            
            # Progress multiplier: 1.0 at start, 1.3 at goal
            progress_multiplier = 1.0 + 0.3 * progress_ratio
            
            # Apply multiplier to value
            adjusted_value = wp.value * progress_multiplier
            
            # Create new waypoint with adjusted value
            adjusted.append(PathWaypoint(
                position=wp.position,
                waypoint_type=wp.waypoint_type,
                value=adjusted_value,
                phase=wp.phase,
                node_index=wp.node_index,
                physics_state=wp.physics_state,
                curvature=wp.curvature,
            ))
        
        return adjusted
    
    def _cluster_waypoints(
        self,
        waypoints: List[PathWaypoint],
    ) -> List[PathWaypoint]:
        """Cluster waypoints within radius and keep highest value.
        
        Removes redundancy by merging nearby waypoints, keeping the
        most important one in each cluster.
        
        Args:
            waypoints: Waypoints to cluster
            
        Returns:
            Deduplicated waypoints
        """
        if not waypoints:
            return []
        
        # Sort by value (descending) to prioritize high-value waypoints
        sorted_waypoints = sorted(waypoints, key=lambda w: -w.value)
        
        clustered = []
        
        for wp in sorted_waypoints:
            # Check if any existing clustered waypoint is within radius
            is_redundant = False
            for existing_wp in clustered:
                dx = wp.position[0] - existing_wp.position[0]
                dy = wp.position[1] - existing_wp.position[1]
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < self.cluster_radius:
                    is_redundant = True
                    break
            
            if not is_redundant:
                clustered.append(wp)
        
        # Sort by node index to maintain path order
        clustered.sort(key=lambda w: w.node_index)
        
        logger.debug(
            f"Clustered {len(waypoints)} waypoints -> {len(clustered)} final waypoints "
            f"(removed {len(waypoints) - len(clustered)} redundant)"
        )
        
        return clustered
    
    def _calculate_turn_angle(
        self,
        prev_node: Tuple[int, int],
        curr_node: Tuple[int, int],
        next_node: Tuple[int, int],
    ) -> float:
        """Calculate turn angle at current node using 3-point curvature.
        
        Args:
            prev_node: Previous node position
            curr_node: Current node position
            next_node: Next node position
            
        Returns:
            Turn angle in degrees (0-180)
        """
        # Vector from prev to curr
        v1_x = curr_node[0] - prev_node[0]
        v1_y = curr_node[1] - prev_node[1]
        v1_len = math.sqrt(v1_x * v1_x + v1_y * v1_y)
        
        # Vector from curr to next
        v2_x = next_node[0] - curr_node[0]
        v2_y = next_node[1] - curr_node[1]
        v2_len = math.sqrt(v2_x * v2_x + v2_y * v2_y)
        
        if v1_len < 0.001 or v2_len < 0.001:
            return 0.0  # Degenerate vectors
        
        # Normalize vectors
        v1_x /= v1_len
        v1_y /= v1_len
        v2_x /= v2_len
        v2_y /= v2_len
        
        # Dot product for angle calculation
        dot_product = v1_x * v2_x + v1_y * v2_y
        
        # Clamp to [-1, 1] to handle floating point errors
        dot_product = max(-1.0, min(1.0, dot_product))
        
        # Calculate angle in degrees
        angle_radians = math.acos(dot_product)
        angle_degrees = math.degrees(angle_radians)
        
        return angle_degrees
    
    def _calculate_path_length(
        self,
        path_nodes: List[Tuple[int, int]],
    ) -> float:
        """Calculate total Euclidean length of path segment.
        
        Args:
            path_nodes: Path node positions
            
        Returns:
            Total path length in pixels
        """
        if len(path_nodes) < 2:
            return 0.0
        
        total_length = 0.0
        
        for i in range(len(path_nodes) - 1):
            curr = path_nodes[i]
            next_node = path_nodes[i + 1]
            
            dx = next_node[0] - curr[0]
            dy = next_node[1] - curr[1]
            edge_length = math.sqrt(dx * dx + dy * dy)
            
            total_length += edge_length
        
        return total_length
    
    def _has_waypoint_within_radius(
        self,
        positions: set,
        query_pos: Tuple[float, float],
        radius: float,
    ) -> bool:
        """Check if any existing waypoint is within radius of query position.
        
        Args:
            positions: Set of existing waypoint positions
            query_pos: Query position
            radius: Search radius
            
        Returns:
            True if waypoint exists within radius
        """
        for pos in positions:
            dx = query_pos[0] - pos[0]
            dy = query_pos[1] - pos[1]
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < radius:
                return True
        
        return False
    
    def _get_physics_state_string(
        self,
        physics: Dict[str, bool],
    ) -> str:
        """Convert physics cache dict to readable state string.
        
        Args:
            physics: Physics properties dict
            
        Returns:
            "grounded", "aerial", "walled", or "unknown"
        """
        grounded = physics.get("grounded", False)
        walled = physics.get("walled", False)
        
        if walled:
            return "walled"
        elif grounded:
            return "grounded"
        elif not grounded:
            return "aerial"
        else:
            return "unknown"
    
    def _compute_exit_directions(
        self,
        waypoints: List[PathWaypoint],
        path_nodes: List[Tuple[int, int]],
    ) -> List[PathWaypoint]:
        """Compute exit direction for each waypoint pointing toward next waypoint.
        
        For sequential guidance, each waypoint needs to know which direction the
        agent should continue traveling after collecting it. This is computed by
        finding the direction to the next waypoint in the sequence.
        
        Args:
            waypoints: List of waypoints (sorted by node_index)
            path_nodes: Original path nodes for fallback direction
            
        Returns:
            List of waypoints with exit_direction populated
        """
        if not waypoints:
            return waypoints
        
        updated_waypoints = []
        
        for i, wp in enumerate(waypoints):
            exit_dir = None
            
            # Find direction to next waypoint in sequence
            if i < len(waypoints) - 1:
                next_wp = waypoints[i + 1]
                dx = next_wp.position[0] - wp.position[0]
                dy = next_wp.position[1] - wp.position[1]
                dist = math.sqrt(dx * dx + dy * dy)
                
                if dist > 1.0:
                    # Normalize direction vector
                    exit_dir = (dx / dist, dy / dist)
            else:
                # Last waypoint - use direction along path toward goal
                # Find nodes near this waypoint in the path
                wp_node_idx = wp.node_index
                if wp_node_idx < len(path_nodes) - 1:
                    # Direction from this node to next node in path
                    curr_node = path_nodes[wp_node_idx]
                    next_node = path_nodes[wp_node_idx + 1]
                    
                    # Convert to world coordinates
                    curr_x = curr_node[0] + NODE_WORLD_COORD_OFFSET
                    curr_y = curr_node[1] + NODE_WORLD_COORD_OFFSET
                    next_x = next_node[0] + NODE_WORLD_COORD_OFFSET
                    next_y = next_node[1] + NODE_WORLD_COORD_OFFSET
                    
                    dx = next_x - curr_x
                    dy = next_y - curr_y
                    dist = math.sqrt(dx * dx + dy * dy)
                    
                    if dist > 1.0:
                        exit_dir = (dx / dist, dy / dist)
            
            # Create new waypoint with exit_direction
            updated_waypoints.append(PathWaypoint(
                position=wp.position,
                waypoint_type=wp.waypoint_type,
                value=wp.value,
                phase=wp.phase,
                node_index=wp.node_index,
                physics_state=wp.physics_state,
                curvature=wp.curvature,
                exit_direction=exit_dir,
            ))
        
        logger.debug(
            f"Computed exit directions for {len(updated_waypoints)} waypoints "
            f"({sum(1 for wp in updated_waypoints if wp.exit_direction is not None)} have valid directions)"
        )
        
        return updated_waypoints
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction and caching statistics.
        
        Returns:
            Statistics dictionary
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "total_waypoints_extracted": self.total_waypoints_extracted,
            "cached_levels": len(self._waypoint_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
        }

