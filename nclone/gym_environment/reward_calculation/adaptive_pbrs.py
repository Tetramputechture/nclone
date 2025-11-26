"""Adaptive multi-path PBRS calculator with online route learning.

This module extends the standard PBRS system to handle multiple candidate paths
with uncertainty estimates, dynamically adjusting path preferences based on
observed training outcomes for better generalization to unseen levels.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

from .pbrs_potentials import PBRSCalculator

logger = logging.getLogger(__name__)


@dataclass 
class PathPotential:
    """Represents potential calculation for a specific candidate path."""
    path_id: str
    waypoints: List[Tuple[int, int]]
    potential_value: float
    confidence: float  # 0.0 to 1.0
    path_type: str  # 'direct', 'mine_avoidance', etc.
    estimated_cost: float
    risk_level: float  # 0.0 (safe) to 1.0 (risky)


@dataclass
class MultiPathPotentialResult:
    """Result of multi-path potential calculation."""
    weighted_potential: float  # Final weighted potential
    individual_potentials: List[PathPotential]  # Potentials for each candidate path
    dominant_path_id: str  # ID of path with highest weighted contribution
    uncertainty: float  # Overall uncertainty (0.0 to 1.0)


class AdaptiveMultiPathPBRS(PBRSCalculator):
    """Enhanced PBRS calculator with multiple candidate path awareness.
    
    This calculator extends the standard PBRS approach by:
    1. Accepting multiple candidate paths with confidence scores
    2. Computing weighted PBRS based on path confidence and quality
    3. Learning path preferences online based on training outcomes
    4. Providing exploration bonuses for uncertain but promising paths
    """
    
    def __init__(self, 
                 path_calculator: Optional[Any] = None,
                 learning_rate: float = 0.05,
                 uncertainty_bonus_weight: float = 0.1,
                 min_confidence_threshold: float = 0.1,
                 path_memory_size: int = 500):
        """Initialize adaptive multi-path PBRS calculator.
        
        Args:
            path_calculator: CachedPathDistanceCalculator instance
            learning_rate: Rate for updating path preferences (0.0 to 1.0)
            uncertainty_bonus_weight: Weight for uncertainty-based exploration bonus
            min_confidence_threshold: Minimum confidence to consider a path
            path_memory_size: Maximum number of path outcomes to remember
        """
        super().__init__(path_calculator)
        
        self.learning_rate = learning_rate
        self.uncertainty_bonus_weight = uncertainty_bonus_weight
        self.min_confidence_threshold = min_confidence_threshold
        self.path_memory_size = path_memory_size
        
        # Online learning of path preferences
        self.path_quality_memory = {}  # path_hash -> quality_score
        self.path_usage_counts = defaultdict(int)
        self.path_success_rates = defaultdict(float)
        
        # Recent performance tracking for adaptive adjustment
        self.recent_outcomes = deque(maxlen=100)
        
        # Statistics
        self.paths_evaluated = 0
        self.preference_updates = 0
        
        logger.info("Initialized AdaptiveMultiPathPBRS calculator")
    
    def calculate_multipath_potential(self, 
                                    state: Dict[str, Any],
                                    candidate_paths: List[Dict[str, Any]],
                                    path_confidences: List[float],
                                    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
                                    level_data: Any,
                                    graph_data: Optional[Dict[str, Any]] = None,
                                    scale_factor: float = 1.0) -> MultiPathPotentialResult:
        """Calculate PBRS potential using multiple candidate paths.
        
        Args:
            state: Game state dictionary
            candidate_paths: List of candidate path dictionaries
            path_confidences: Confidence scores for each candidate path
            adjacency: Graph adjacency structure
            level_data: Level data object
            graph_data: Optional graph data for optimization
            scale_factor: Curriculum normalization adjustment
            
        Returns:
            MultiPathPotentialResult with weighted potential and metadata
        """
        if not candidate_paths or not path_confidences:
            # Fallback to standard PBRS if no candidate paths provided
            standard_potential = self.calculate_combined_potential(
                state, adjacency, level_data, graph_data, scale_factor=scale_factor
            )
            return MultiPathPotentialResult(
                weighted_potential=standard_potential,
                individual_potentials=[],
                dominant_path_id="standard",
                uncertainty=0.0
            )
        
        if len(candidate_paths) != len(path_confidences):
            logger.warning("Mismatch between candidate paths and confidences, using standard PBRS")
            standard_potential = self.calculate_combined_potential(
                state, adjacency, level_data, graph_data, scale_factor=scale_factor
            )
            return MultiPathPotentialResult(
                weighted_potential=standard_potential,
                individual_potentials=[],
                dominant_path_id="standard", 
                uncertainty=0.0
            )
        
        # Calculate potential for each candidate path
        individual_potentials = []
        total_weighted_potential = 0.0
        total_weight = 0.0
        uncertainties = []
        
        for i, (path_info, base_confidence) in enumerate(zip(candidate_paths, path_confidences)):
            if base_confidence < self.min_confidence_threshold:
                continue
                
            # Extract path information
            waypoints = path_info.get('waypoints', [])
            path_type = path_info.get('path_type', 'unknown')
            estimated_cost = path_info.get('estimated_cost', 1.0)
            risk_level = path_info.get('risk_level', 0.5)
            
            if not waypoints:
                continue
            
            # Create path identifier for learning
            path_id = self._create_path_id(waypoints, path_type)
            
            # Calculate PBRS potential along this specific path
            path_potential = self._calculate_path_specific_potential(
                state, waypoints, adjacency, level_data, graph_data, scale_factor
            )
            
            # Adjust confidence based on online learning
            adjusted_confidence = self._adjust_confidence_with_learning(
                path_id, base_confidence, path_type, estimated_cost, risk_level
            )
            
            # Calculate uncertainty (higher for less explored paths)
            path_uncertainty = self._calculate_path_uncertainty(path_id, adjusted_confidence)
            uncertainties.append(path_uncertainty)
            
            # Create PathPotential object
            path_potential_obj = PathPotential(
                path_id=path_id,
                waypoints=waypoints,
                potential_value=path_potential,
                confidence=adjusted_confidence,
                path_type=path_type,
                estimated_cost=estimated_cost,
                risk_level=risk_level
            )
            individual_potentials.append(path_potential_obj)
            
            # Add to weighted sum
            weight = adjusted_confidence
            total_weighted_potential += path_potential * weight
            total_weight += weight
            
            self.paths_evaluated += 1
        
        # Calculate final weighted potential
        if total_weight > 0:
            weighted_potential = total_weighted_potential / total_weight
        else:
            # Fallback to standard PBRS
            weighted_potential = self.calculate_combined_potential(
                state, adjacency, level_data, graph_data, scale_factor=scale_factor
            )
        
        # Add exploration bonus for high uncertainty situations
        overall_uncertainty = np.mean(uncertainties) if uncertainties else 0.0
        exploration_bonus = self.uncertainty_bonus_weight * overall_uncertainty
        weighted_potential += exploration_bonus
        
        # Find dominant path (highest weighted contribution)
        dominant_path_id = "none"
        if individual_potentials:
            dominant_path = max(
                individual_potentials, 
                key=lambda p: p.potential_value * p.confidence
            )
            dominant_path_id = dominant_path.path_id
        
        return MultiPathPotentialResult(
            weighted_potential=max(0.0, min(1.0, weighted_potential)),
            individual_potentials=individual_potentials,
            dominant_path_id=dominant_path_id,
            uncertainty=overall_uncertainty
        )
    
    def update_path_rewards(self, level_outcomes: Dict[str, Any]) -> None:
        """Update path preferences based on observed level outcomes.
        
        Args:
            level_outcomes: Dictionary containing:
                - 'attempted_paths': List of path information that were tried
                - 'path_outcomes': List of outcome results for each path
                - 'level_success': Whether the level was completed
                - 'completion_time': Time taken if successful
        """
        attempted_paths = level_outcomes.get('attempted_paths', [])
        path_outcomes = level_outcomes.get('path_outcomes', [])
        level_success = level_outcomes.get('level_success', False)
        completion_time = level_outcomes.get('completion_time', None)
        
        if len(attempted_paths) != len(path_outcomes):
            logger.warning("Mismatch between attempted paths and outcomes")
            return
        
        # Update path quality scores based on outcomes
        for path_info, outcome in zip(attempted_paths, path_outcomes):
            waypoints = path_info.get('waypoints', [])
            path_type = path_info.get('path_type', 'unknown')
            
            if not waypoints:
                continue
                
            path_id = self._create_path_id(waypoints, path_type)
            
            # Calculate quality score based on outcome
            quality_score = self._calculate_outcome_quality_score(
                outcome, level_success, completion_time
            )
            
            # Update path memory with exponential moving average
            current_quality = self.path_quality_memory.get(path_id, 0.5)
            new_quality = ((1 - self.learning_rate) * current_quality + 
                          self.learning_rate * quality_score)
            
            self.path_quality_memory[path_id] = new_quality
            self.path_usage_counts[path_id] += 1
            
            # Update success rate
            current_success_rate = self.path_success_rates[path_id]
            path_success = outcome.get('success', False)
            new_success_rate = ((1 - self.learning_rate) * current_success_rate + 
                               self.learning_rate * (1.0 if path_success else 0.0))
            self.path_success_rates[path_id] = new_success_rate
            
            # Track recent outcomes
            self.recent_outcomes.append(quality_score)
            
            self.preference_updates += 1
        
        # Prune old path memories to limit memory usage
        self._prune_path_memory()
        
        logger.debug(f"Updated preferences for {len(attempted_paths)} paths")
    
    def _calculate_path_specific_potential(self, 
                                         state: Dict[str, Any],
                                         waypoints: List[Tuple[int, int]], 
                                         adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
                                         level_data: Any,
                                         graph_data: Optional[Dict[str, Any]],
                                         scale_factor: float) -> float:
        """Calculate PBRS potential for a specific path route."""
        
        # For now, use waypoint-based distance estimation
        # In future versions, could integrate more sophisticated path cost calculation
        
        player_pos = (int(state["player_x"]), int(state["player_y"]))
        
        # Find closest waypoint as proxy for path progress
        if not waypoints:
            return 0.0
        
        min_distance_to_waypoint = float('inf')
        for waypoint in waypoints:
            dx = player_pos[0] - waypoint[0]
            dy = player_pos[1] - waypoint[1]
            distance = np.sqrt(dx * dx + dy * dy)
            min_distance_to_waypoint = min(min_distance_to_waypoint, distance)
        
        # Get surface area for normalization (same as standard PBRS)
        surface_area = state.get("_pbrs_surface_area")
        if not surface_area:
            # Compute if not available
            surface_area = self._compute_reachable_surface_area(adjacency, level_data, graph_data)
            state["_pbrs_surface_area"] = surface_area
        
        # Normalize distance similar to standard PBRS
        area_scale = np.sqrt(surface_area) * 12  # SUB_NODE_SIZE = 12
        area_scale = area_scale * scale_factor
        
        from ..constants import LEVEL_DIAGONAL
        max_scale = LEVEL_DIAGONAL * 0.5
        area_scale = min(area_scale, max_scale)
        
        normalized_distance = min(1.0, min_distance_to_waypoint / area_scale)
        potential = 1.0 - normalized_distance
        
        return max(0.0, min(1.0, potential))
    
    def _adjust_confidence_with_learning(self, 
                                       path_id: str, 
                                       base_confidence: float,
                                       path_type: str,
                                       estimated_cost: float,
                                       risk_level: float) -> float:
        """Adjust path confidence based on learned preferences."""
        
        # Start with base confidence
        adjusted_confidence = base_confidence
        
        # Apply learned quality adjustment
        if path_id in self.path_quality_memory:
            learned_quality = self.path_quality_memory[path_id]
            # Blend learned quality with base confidence
            adjusted_confidence = 0.7 * adjusted_confidence + 0.3 * learned_quality
        
        # Apply success rate adjustment
        if path_id in self.path_success_rates:
            success_rate = self.path_success_rates[path_id] 
            # Boost confidence for high success rate paths
            success_bonus = (success_rate - 0.5) * 0.2  # ±0.1 adjustment
            adjusted_confidence += success_bonus
        
        # Apply cost/risk penalties
        cost_penalty = min(0.1, estimated_cost / 10.0)  # Penalize high cost paths
        risk_penalty = risk_level * 0.1  # Penalize risky paths
        adjusted_confidence -= (cost_penalty + risk_penalty)
        
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _calculate_path_uncertainty(self, path_id: str, confidence: float) -> float:
        """Calculate uncertainty for a path (higher for less explored paths)."""
        
        usage_count = self.path_usage_counts.get(path_id, 0)
        
        # Higher uncertainty for less explored paths
        exploration_uncertainty = 1.0 / (1.0 + usage_count / 10.0)
        
        # Uncertainty inversely related to confidence
        confidence_uncertainty = 1.0 - confidence
        
        # Combined uncertainty
        total_uncertainty = 0.6 * exploration_uncertainty + 0.4 * confidence_uncertainty
        
        return max(0.0, min(1.0, total_uncertainty))
    
    def _calculate_outcome_quality_score(self, 
                                       outcome: Dict[str, Any],
                                       level_success: bool,
                                       completion_time: Optional[float]) -> float:
        """Calculate quality score for a path outcome (0.0 to 1.0)."""
        
        # Base success component
        if not outcome.get('success', False) or not level_success:
            return 0.0
        
        # Start with base success score
        quality = 0.6
        
        # Time bonus (faster completion is better)
        if completion_time is not None and completion_time > 0:
            # Assume 60 seconds is slow, normalize accordingly
            time_factor = max(0.0, min(1.0, (60.0 - completion_time) / 60.0))
            quality += 0.2 * time_factor
        
        # Distance efficiency (from outcome if available)
        distance_traveled = outcome.get('distance_traveled', 0.0)
        if distance_traveled > 0:
            # Normalize distance - less is better
            distance_factor = 1.0 / (1.0 + distance_traveled / 1000.0)
            quality += 0.1 * distance_factor
        
        # Safety factor (fewer deaths/mines triggered)
        deaths = outcome.get('deaths_occurred', 0)
        mines_triggered = outcome.get('mines_triggered', 0)
        safety_factor = 1.0 / (1.0 + deaths + mines_triggered * 0.5)
        quality += 0.1 * safety_factor
        
        return max(0.0, min(1.0, quality))
    
    def _create_path_id(self, waypoints: List[Tuple[int, int]], path_type: str) -> str:
        """Create unique identifier for a path."""
        
        if len(waypoints) <= 3:
            waypoint_str = '_'.join(f"{x}_{y}" for x, y in waypoints)
        else:
            # Use key points for longer paths
            key_points = [waypoints[0], waypoints[len(waypoints)//2], waypoints[-1]]
            waypoint_str = '_'.join(f"{x}_{y}" for x, y in key_points)
        
        path_signature = f"{path_type}_{waypoint_str}"
        return str(hash(path_signature) % 1000000)  # Limit size
    
    def _prune_path_memory(self) -> None:
        """Remove old/poor-performing path memories to limit memory usage."""
        
        if len(self.path_quality_memory) <= self.path_memory_size:
            return
        
        # Sort paths by quality and usage, keep top performers
        path_scores = [
            (path_id, quality * (1 + np.log1p(self.path_usage_counts[path_id])))
            for path_id, quality in self.path_quality_memory.items()
        ]
        path_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top 80% of memory limit
        keep_count = int(self.path_memory_size * 0.8)
        paths_to_keep = set(path_id for path_id, _ in path_scores[:keep_count])
        
        # Prune memory dictionaries
        self.path_quality_memory = {
            path_id: quality for path_id, quality in self.path_quality_memory.items()
            if path_id in paths_to_keep
        }
        
        # Prune usage counts and success rates
        for path_id in list(self.path_usage_counts.keys()):
            if path_id not in paths_to_keep:
                del self.path_usage_counts[path_id]
        
        for path_id in list(self.path_success_rates.keys()):
            if path_id not in paths_to_keep:
                del self.path_success_rates[path_id]
        
        logger.debug(f"Pruned path memory, keeping {len(paths_to_keep)} paths")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptive PBRS statistics."""
        
        recent_performance = np.mean(self.recent_outcomes) if self.recent_outcomes else 0.0
        
        return {
            'paths_evaluated': self.paths_evaluated,
            'preference_updates': self.preference_updates,
            'paths_in_memory': len(self.path_quality_memory),
            'recent_performance': recent_performance,
            'avg_path_quality': np.mean(list(self.path_quality_memory.values())) 
                               if self.path_quality_memory else 0.0,
            'avg_success_rate': np.mean(list(self.path_success_rates.values()))
                               if self.path_success_rates else 0.0
        }
    
    def reset_for_new_level(self) -> None:
        """Reset state for a new level (but keep learned preferences)."""
        # Reset episode-specific state but keep learned path preferences
        self.reset()  # Call parent reset
        
        # Don't reset path_quality_memory, path_usage_counts, path_success_rates
        # These should persist across levels for generalization
        
    def save_adaptive_state(self, filepath: str) -> None:
        """Save adaptive learning state to file."""
        import pickle
        
        state = {
            'path_quality_memory': dict(self.path_quality_memory),
            'path_usage_counts': dict(self.path_usage_counts),
            'path_success_rates': dict(self.path_success_rates),
            'recent_outcomes': list(self.recent_outcomes),
            'paths_evaluated': self.paths_evaluated,
            'preference_updates': self.preference_updates
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved adaptive PBRS state to {filepath}")
    
    def load_adaptive_state(self, filepath: str) -> None:
        """Load adaptive learning state from file."""
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.path_quality_memory = state.get('path_quality_memory', {})
            self.path_usage_counts = defaultdict(int, state.get('path_usage_counts', {}))
            self.path_success_rates = defaultdict(float, state.get('path_success_rates', {}))
            self.recent_outcomes = deque(state.get('recent_outcomes', []), maxlen=100)
            self.paths_evaluated = state.get('paths_evaluated', 0)
            self.preference_updates = state.get('preference_updates', 0)
            
            logger.info(f"Loaded adaptive PBRS state from {filepath}")
            
        except FileNotFoundError:
            logger.warning(f"Adaptive PBRS state file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading adaptive PBRS state: {e}")


def create_adaptive_multipath_pbrs(config: Dict[str, Any]) -> AdaptiveMultiPathPBRS:
    """Factory function to create AdaptiveMultiPathPBRS from config.
    
    Args:
        config: Configuration dictionary with PBRS parameters
        
    Returns:
        Configured AdaptiveMultiPathPBRS instance
    """
    from ...graph.reachability.path_distance_calculator import CachedPathDistanceCalculator
    
    # Create path calculator
    path_calculator = CachedPathDistanceCalculator(
        max_cache_size=config.get('path_cache_size', 200),
        use_astar=config.get('use_astar', True)
    )
    
    return AdaptiveMultiPathPBRS(
        path_calculator=path_calculator,
        learning_rate=config.get('learning_rate', 0.05),
        uncertainty_bonus_weight=config.get('uncertainty_bonus_weight', 0.1),
        min_confidence_threshold=config.get('min_confidence_threshold', 0.1),
        path_memory_size=config.get('path_memory_size', 500)
    )
