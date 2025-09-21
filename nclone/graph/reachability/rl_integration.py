"""
RL integration API for enhanced reachability system.

This module provides RL-specific methods and optimizations for:
- Hierarchical Reinforcement Learning (HRL)
- Curiosity-driven exploration
- Level completion planning
- Performance-optimized state representations
"""

from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass
import numpy as np
import math

from .frontier_detector import Frontier
from ...constants.physics_constants import TILE_PIXEL_SIZE


@dataclass
class RLState:
    """RL-optimized state representation."""
    
    # Core reachability data
    reachable_positions: Set[Tuple[int, int]]
    switch_states: Dict[int, bool]
    
    # RL-specific data
    subgoals: List
    frontiers: List[Frontier]
    curiosity_map: np.ndarray  # 2D array of curiosity values
    accessibility_map: np.ndarray  # 2D array of accessibility scores
    
    # Performance metrics
    analysis_time: float
    cache_hit: bool


class RLIntegrationAPI:
    """
    RL integration API for enhanced reachability system.
    
    Provides optimized methods for:
    - Hierarchical RL subgoal planning
    - Curiosity-driven exploration rewards
    - Level completion path planning
    - Performance-optimized state queries
    """
    
    def __init__(self, reachability_analyzer, debug: bool = False):
        """
        Initialize RL integration API.
        
        Args:
            reachability_analyzer: Enhanced reachability analyzer instance
            debug: Enable debug output
        """
        self.analyzer = reachability_analyzer
        self.debug = debug
        
    def get_rl_state(
        self,
        level_data,
        ninja_position: Tuple[float, float],
        initial_switch_states: Optional[Dict[int, bool]] = None
    ) -> RLState:
        """
        Get RL-optimized state representation.
        
        Args:
            level_data: Level data
            ninja_position: Current ninja position
            initial_switch_states: Initial switch states
            
        Returns:
            RL-optimized state representation
        """
        import time
        start_time = time.time()
        
        # Get full reachability analysis
        reachability_state = self.analyzer.analyze_reachability(
            level_data, ninja_position, initial_switch_states
        )
        
        analysis_time = time.time() - start_time
        
        # Extract enhanced data
        subgoals = getattr(reachability_state, 'enhanced_subgoals', [])
        frontiers = getattr(reachability_state, 'frontiers', [])
        
        # Generate curiosity and accessibility maps
        curiosity_map = self._generate_curiosity_map(level_data, frontiers)
        accessibility_map = self._generate_accessibility_map(
            level_data, reachability_state.reachable_positions
        )
        
        # Check if result was cached
        cache_hit = hasattr(reachability_state, '_cache_hit') and reachability_state._cache_hit
        
        return RLState(
            reachable_positions=reachability_state.reachable_positions,
            switch_states=reachability_state.switch_states,
            subgoals=subgoals,
            frontiers=frontiers,
            curiosity_map=curiosity_map,
            accessibility_map=accessibility_map,
            analysis_time=analysis_time,
            cache_hit=cache_hit
        )
    
    def get_hierarchical_subgoals(
        self,
        rl_state: RLState,
        max_subgoals: int = 5,
        prioritize_critical_path: bool = True
    ) -> List:
        """
        Get prioritized subgoals for hierarchical RL.
        
        Args:
            rl_state: Current RL state
            max_subgoals: Maximum number of subgoals to return
            prioritize_critical_path: Whether to prioritize critical path subgoals
            
        Returns:
            List of prioritized subgoals
        """
        # Runtime import to avoid circular dependency
        subgoals = rl_state.subgoals.copy()
        
        if prioritize_critical_path:
            # Boost priority of critical path subgoals
            critical_types = {'exit_switch', 'exit_door'}
            for subgoal in subgoals:
                if subgoal.goal_type in critical_types:
                    subgoal.priority -= 10  # Lower number = higher priority
        
        # Sort by priority and return top subgoals
        subgoals.sort(key=lambda s: s.priority)
        return subgoals[:max_subgoals]
    
    def calculate_curiosity_reward(
        self,
        rl_state: RLState,
        position: Tuple[float, float],
        radius: float = 3.0
    ) -> float:
        """
        Calculate curiosity reward for a position.
        
        Args:
            rl_state: Current RL state
            position: Position to calculate reward for (pixel coordinates)
            radius: Search radius for nearby frontiers
            
        Returns:
            Curiosity reward value (0.0 to 1.0)
        """
        # Convert position to sub-grid coordinates
        sub_row = int(position[1] // TILE_PIXEL_SIZE)
        sub_col = int(position[0] // TILE_PIXEL_SIZE)
        
        # Get curiosity value from map
        if (0 <= sub_row < rl_state.curiosity_map.shape[0] and
            0 <= sub_col < rl_state.curiosity_map.shape[1]):
            base_curiosity = rl_state.curiosity_map[sub_row, sub_col]
        else:
            base_curiosity = 0.0
        
        # Add bonus for nearby high-value frontiers
        frontier_bonus = 0.0
        for frontier in rl_state.frontiers:
            f_row, f_col = frontier.position
            distance = math.sqrt((sub_row - f_row) ** 2 + (sub_col - f_col) ** 2)
            
            if distance <= radius:
                distance_factor = 1.0 - (distance / radius)
                frontier_bonus += frontier.exploration_value * distance_factor * 0.3
        
        return min(base_curiosity + frontier_bonus, 1.0)
    
    def get_exploration_targets(
        self,
        rl_state: RLState,
        max_targets: int = 3,
        min_exploration_value: float = 0.3
    ) -> List[Tuple[int, int]]:
        """
        Get high-value exploration targets.
        
        Args:
            rl_state: Current RL state
            max_targets: Maximum number of targets to return
            min_exploration_value: Minimum exploration value threshold
            
        Returns:
            List of exploration target positions (sub-grid coordinates)
        """
        high_value_frontiers = [
            f for f in rl_state.frontiers 
            if f.exploration_value >= min_exploration_value
        ]
        
        # Sort by exploration value (descending)
        high_value_frontiers.sort(key=lambda f: f.exploration_value, reverse=True)
        
        return [f.position for f in high_value_frontiers[:max_targets]]
    
    def plan_level_completion_path(
        self,
        rl_state: RLState,
        current_position: Tuple[float, float]
    ) -> List:
        """
        Plan optimal path for level completion.
        
        Args:
            rl_state: Current RL state
            current_position: Current ninja position
            
        Returns:
            Ordered list of subgoals for level completion
        """
        # Get critical path subgoals
        critical_subgoals = [
            s for s in rl_state.subgoals 
            if s.goal_type in {'exit_switch', 'exit_door'}
        ]
        
        # Get supporting subgoals (doors, switches)
        supporting_subgoals = [
            s for s in rl_state.subgoals
            if s.goal_type in {'door_switch', 'locked_door'}
        ]
        
        # Build dependency-aware path
        completion_path = []
        
        # Add supporting subgoals first (in dependency order)
        for subgoal in supporting_subgoals:
            if not subgoal.dependencies:  # No dependencies
                completion_path.append(subgoal)
        
        # Add dependent supporting subgoals
        for subgoal in supporting_subgoals:
            if subgoal.dependencies and subgoal not in completion_path:
                completion_path.append(subgoal)
        
        # Add critical path subgoals
        for subgoal in critical_subgoals:
            completion_path.append(subgoal)
        
        return completion_path
    
    def get_reachability_features(
        self,
        rl_state: RLState,
        position: Tuple[float, float],
        feature_radius: int = 5
    ) -> Dict[str, float]:
        """
        Extract RL features from reachability analysis.
        
        Args:
            rl_state: Current RL state
            position: Position to extract features for
            feature_radius: Radius for local feature extraction
            
        Returns:
            Dictionary of RL features
        """
        # Convert to sub-grid coordinates
        sub_row = int(position[1] // TILE_PIXEL_SIZE)
        sub_col = int(position[0] // TILE_PIXEL_SIZE)
        
        features = {}
        
        # Basic reachability features
        features['is_reachable'] = float((sub_row, sub_col) in rl_state.reachable_positions)
        features['reachable_area_size'] = float(len(rl_state.reachable_positions))
        
        # Curiosity and exploration features
        features['curiosity_value'] = self.calculate_curiosity_reward(rl_state, position)
        features['num_nearby_frontiers'] = float(self._count_nearby_frontiers(
            rl_state.frontiers, (sub_row, sub_col), feature_radius
        ))
        
        # Subgoal features
        features['distance_to_nearest_subgoal'] = self._distance_to_nearest_subgoal(
            rl_state.subgoals, (sub_row, sub_col)
        )
        features['num_critical_subgoals'] = float(len([
            s for s in rl_state.subgoals 
            if s.goal_type in {'exit_switch', 'exit_door'}
        ]))
        
        # Accessibility features
        if (0 <= sub_row < rl_state.accessibility_map.shape[0] and
            0 <= sub_col < rl_state.accessibility_map.shape[1]):
            features['accessibility_score'] = float(rl_state.accessibility_map[sub_row, sub_col])
        else:
            features['accessibility_score'] = 0.0
        
        # Local density features
        features['local_reachable_density'] = self._calculate_local_density(
            rl_state.reachable_positions, (sub_row, sub_col), feature_radius
        )
        
        return features
    
    def get_performance_metrics(self, rl_state: RLState) -> Dict[str, Any]:
        """
        Get performance metrics for the RL integration.
        
        Args:
            rl_state: Current RL state
            
        Returns:
            Dictionary of performance metrics
        """
        return {
            'analysis_time_ms': rl_state.analysis_time * 1000,
            'cache_hit': rl_state.cache_hit,
            'reachable_positions': len(rl_state.reachable_positions),
            'num_subgoals': len(rl_state.subgoals),
            'num_frontiers': len(rl_state.frontiers),
            'curiosity_map_size': rl_state.curiosity_map.size,
            'memory_usage_estimate': self._estimate_memory_usage(rl_state)
        }
    
    def _generate_curiosity_map(self, level_data, frontiers: List[Frontier]) -> np.ndarray:
        """Generate 2D curiosity value map."""
        height, width = level_data.tiles.shape
        curiosity_map = np.zeros((height, width), dtype=np.float32)
        
        for frontier in frontiers:
            row, col = frontier.position
            if 0 <= row < height and 0 <= col < width:
                # Set curiosity value at frontier position
                curiosity_map[row, col] = frontier.exploration_value
                
                # Spread curiosity to nearby positions
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        adj_row, adj_col = row + dr, col + dc
                        if (0 <= adj_row < height and 0 <= adj_col < width):
                            distance = max(abs(dr), abs(dc))
                            if distance > 0:
                                decay_factor = 1.0 / (distance + 1)
                                spread_value = frontier.exploration_value * decay_factor * 0.3
                                curiosity_map[adj_row, adj_col] = max(
                                    curiosity_map[adj_row, adj_col], spread_value
                                )
        
        return curiosity_map
    
    def _generate_accessibility_map(
        self, 
        level_data, 
        reachable_positions: Set[Tuple[int, int]]
    ) -> np.ndarray:
        """Generate 2D accessibility score map."""
        height, width = level_data.tiles.shape
        accessibility_map = np.zeros((height, width), dtype=np.float32)
        
        if not reachable_positions:
            return accessibility_map
        
        # Calculate centroid of reachable area
        centroid_row = sum(pos[0] for pos in reachable_positions) / len(reachable_positions)
        centroid_col = sum(pos[1] for pos in reachable_positions) / len(reachable_positions)
        
        # Set accessibility scores
        max_distance = math.sqrt(height ** 2 + width ** 2)
        
        for row, col in reachable_positions:
            # Bounds checking
            if 0 <= row < height and 0 <= col < width:
                distance = math.sqrt((row - centroid_row) ** 2 + (col - centroid_col) ** 2)
                accessibility = max(0.0, 1.0 - (distance / max_distance))
                accessibility_map[row, col] = accessibility
        
        return accessibility_map
    
    def _count_nearby_frontiers(
        self, 
        frontiers: List[Frontier], 
        position: Tuple[int, int], 
        radius: int
    ) -> int:
        """Count frontiers within radius of position."""
        row, col = position
        count = 0
        
        for frontier in frontiers:
            f_row, f_col = frontier.position
            distance = max(abs(f_row - row), abs(f_col - col))
            if distance <= radius:
                count += 1
        
        return count
    
    def _distance_to_nearest_subgoal(
        self, 
        subgoals: List, 
        position: Tuple[int, int]
    ) -> float:
        """Calculate distance to nearest subgoal."""
        if not subgoals:
            return float('inf')
        
        row, col = position
        min_distance = float('inf')
        
        for subgoal in subgoals:
            s_row, s_col = subgoal.position
            distance = math.sqrt((row - s_row) ** 2 + (col - s_col) ** 2)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _calculate_local_density(
        self, 
        reachable_positions: Set[Tuple[int, int]], 
        position: Tuple[int, int], 
        radius: int
    ) -> float:
        """Calculate local density of reachable positions."""
        row, col = position
        local_count = 0
        total_positions = (2 * radius + 1) ** 2
        
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if (row + dr, col + dc) in reachable_positions:
                    local_count += 1
        
        return local_count / total_positions
    
    def _estimate_memory_usage(self, rl_state: RLState) -> int:
        """Estimate memory usage of RL state in bytes."""
        # Rough estimation
        base_size = 1000  # Base overhead
        
        # Reachable positions (assuming 8 bytes per tuple)
        reachable_size = len(rl_state.reachable_positions) * 16
        
        # Switch states (assuming 12 bytes per entry)
        switch_size = len(rl_state.switch_states) * 12
        
        # Subgoals (assuming 200 bytes per subgoal)
        subgoal_size = len(rl_state.subgoals) * 200
        
        # Frontiers (assuming 150 bytes per frontier)
        frontier_size = len(rl_state.frontiers) * 150
        
        # Maps (4 bytes per float32)
        map_size = rl_state.curiosity_map.size * 4 + rl_state.accessibility_map.size * 4
        
        return base_size + reachable_size + switch_size + subgoal_size + frontier_size + map_size