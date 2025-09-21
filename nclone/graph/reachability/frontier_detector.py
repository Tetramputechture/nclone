"""
Frontier detection for curiosity-driven exploration in RL.

This module provides comprehensive frontier detection to identify boundaries
between reachable and unreachable areas, classify exploration value, and
support curiosity-driven RL algorithms.
"""

from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum
import math



class FrontierType(Enum):
    """Types of exploration frontiers."""
    
    EXPLORATION = "exploration"  # Unknown areas to explore
    BLOCKED = "blocked"  # Areas blocked by walls/obstacles
    HAZARD = "hazard"  # Areas blocked by hazards
    ENTITY = "entity"  # Areas blocked by entities
    PHYSICS = "physics"  # Areas requiring advanced physics (wall jumps, etc.)


@dataclass
class Frontier:
    """Represents an exploration frontier."""
    
    position: Tuple[int, int]  # Sub-grid coordinates
    frontier_type: FrontierType
    exploration_value: float  # 0.0 to 1.0, higher = more valuable
    accessibility_score: float  # How easy it is to reach this frontier
    potential_area: int  # Estimated size of area beyond frontier
    metadata: Optional[Dict] = None


class FrontierDetector:
    """
    Detects and classifies exploration frontiers for curiosity-driven RL.
    
    Features:
    - Boundary detection between reachable/unreachable areas
    - Frontier classification by type and exploration value
    - Accessibility scoring for frontier prioritization
    - Potential area estimation for exploration planning
    - Integration with entity and hazard systems
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize frontier detector.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        self.frontiers: List[Frontier] = []
        self.frontier_map: Dict[Tuple[int, int], Frontier] = {}
        
    def detect_frontiers(
        self,
        level_data,
        reachability_state,
        hazard_extension=None,
        position_validator=None
    ) -> List[Frontier]:
        """
        Detect all exploration frontiers in the level.
        
        Args:
            level_data: Level data containing tiles and entities
            reachability_state: Current reachability state
            hazard_extension: Optional hazard extension for hazard information
            position_validator: Optional position validator
            
        Returns:
            List of detected frontiers
        """
        self.frontiers.clear()
        self.frontier_map.clear()
        
        reachable_positions = reachability_state.reachable_positions
        height, width = level_data.tiles.shape
        
        # Find all frontier positions
        frontier_candidates = self._find_frontier_candidates(
            reachable_positions, level_data.tiles, height, width
        )
        
        # Classify and evaluate each frontier
        for frontier_pos in frontier_candidates:
            frontier = self._classify_frontier(
                frontier_pos, level_data, reachable_positions,
                hazard_extension, position_validator
            )
            
            if frontier:
                self.frontiers.append(frontier)
                self.frontier_map[frontier_pos] = frontier
        
        # Calculate exploration values and accessibility scores
        self._calculate_exploration_values(level_data, reachable_positions)
        self._calculate_accessibility_scores(reachable_positions)
        
        # Sort by exploration value (descending)
        self.frontiers.sort(key=lambda f: f.exploration_value, reverse=True)
        
        if self.debug:
            print(f"DEBUG: Detected {len(self.frontiers)} frontiers")
            for i, frontier in enumerate(self.frontiers[:5]):  # Show top 5
                print(f"  {i+1}. {frontier.frontier_type.value} at {frontier.position} "
                      f"(value: {frontier.exploration_value:.3f})")
        
        return self.frontiers.copy()
    
    def _find_frontier_candidates(
        self,
        reachable_positions: Set[Tuple[int, int]],
        tiles,
        height: int,
        width: int
    ) -> Set[Tuple[int, int]]:
        """Find positions that are potential frontiers."""
        frontier_candidates = set()
        
        for row, col in reachable_positions:
            # Check all adjacent positions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                adj_row, adj_col = row + dr, col + dc
                
                # Check if adjacent position is within bounds
                if 0 <= adj_row < height and 0 <= adj_col < width:
                    # If adjacent position is not reachable, current position is a frontier
                    if (adj_row, adj_col) not in reachable_positions:
                        frontier_candidates.add((row, col))
                        break
        
        return frontier_candidates
    
    def _classify_frontier(
        self,
        frontier_pos: Tuple[int, int],
        level_data,
        reachable_positions: Set[Tuple[int, int]],
        hazard_extension=None,
        position_validator=None
    ) -> Optional[Frontier]:
        """Classify a frontier position by type."""
        row, col = frontier_pos
        height, width = level_data.tiles.shape
        
        # Analyze what's blocking progress from this frontier
        blocking_reasons = self._analyze_blocking_reasons(
            frontier_pos, level_data, reachable_positions,
            hazard_extension, position_validator
        )
        
        # Determine frontier type based on blocking reasons
        frontier_type = self._determine_frontier_type(blocking_reasons)
        
        # Estimate potential area beyond frontier
        potential_area = self._estimate_potential_area(
            frontier_pos, level_data.tiles, reachable_positions
        )
        
        return Frontier(
            position=frontier_pos,
            frontier_type=frontier_type,
            exploration_value=0.0,  # Will be calculated later
            accessibility_score=0.0,  # Will be calculated later
            potential_area=potential_area,
            metadata={'blocking_reasons': blocking_reasons}
        )
    
    def _analyze_blocking_reasons(
        self,
        frontier_pos: Tuple[int, int],
        level_data,
        reachable_positions: Set[Tuple[int, int]],
        hazard_extension=None,
        position_validator=None
    ) -> Dict[str, int]:
        """Analyze what's blocking progress from a frontier position."""
        row, col = frontier_pos
        height, width = level_data.tiles.shape
        
        blocking_reasons = {
            'walls': 0,
            'hazards': 0,
            'entities': 0,
            'unknown': 0,
            'physics': 0
        }
        
        # Check all adjacent unreachable positions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_row, adj_col = row + dr, col + dc
            
            if (0 <= adj_row < height and 0 <= adj_col < width and
                (adj_row, adj_col) not in reachable_positions):
                
                # Check if blocked by wall
                if level_data.tiles[adj_row, adj_col] != 0:
                    blocking_reasons['walls'] += 1
                else:
                    # Check if blocked by hazard
                    if hazard_extension and position_validator:
                        pixel_x, pixel_y = position_validator.convert_sub_grid_to_pixel(
                            adj_row, adj_col
                        )
                        if not hazard_extension.is_position_safe_for_reachability((pixel_x, pixel_y)):
                            blocking_reasons['hazards'] += 1
                        else:
                            blocking_reasons['unknown'] += 1
                    else:
                        blocking_reasons['unknown'] += 1
        
        return blocking_reasons
    
    def _determine_frontier_type(self, blocking_reasons: Dict[str, int]) -> FrontierType:
        """Determine frontier type based on blocking reasons."""
        total_blocks = sum(blocking_reasons.values())
        
        if total_blocks == 0:
            return FrontierType.EXPLORATION
        
        # Find dominant blocking reason
        dominant_reason = max(blocking_reasons.items(), key=lambda x: x[1])
        
        type_mapping = {
            'walls': FrontierType.BLOCKED,
            'hazards': FrontierType.HAZARD,
            'entities': FrontierType.ENTITY,
            'physics': FrontierType.PHYSICS,
            'unknown': FrontierType.EXPLORATION
        }
        
        return type_mapping.get(dominant_reason[0], FrontierType.EXPLORATION)
    
    def _estimate_potential_area(
        self,
        frontier_pos: Tuple[int, int],
        tiles,
        reachable_positions: Set[Tuple[int, int]]
    ) -> int:
        """Estimate the size of unexplored area beyond a frontier."""
        row, col = frontier_pos
        height, width = tiles.shape
        
        # Use flood fill to estimate connected unreachable area
        visited = set()
        queue = []
        
        # Start from adjacent unreachable positions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_row, adj_col = row + dr, col + dc
            
            if (0 <= adj_row < height and 0 <= adj_col < width and
                (adj_row, adj_col) not in reachable_positions and
                tiles[adj_row, adj_col] == 0):  # Empty but unreachable
                queue.append((adj_row, adj_col))
        
        # Flood fill to count connected area
        area_size = 0
        max_area = 50  # Limit search to avoid performance issues
        
        while queue and area_size < max_area:
            current_row, current_col = queue.pop(0)
            
            if (current_row, current_col) in visited:
                continue
            
            visited.add((current_row, current_col))
            area_size += 1
            
            # Add adjacent empty unreachable positions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_row, next_col = current_row + dr, current_col + dc
                
                if (0 <= next_row < height and 0 <= next_col < width and
                    (next_row, next_col) not in visited and
                    (next_row, next_col) not in reachable_positions and
                    tiles[next_row, next_col] == 0):
                    queue.append((next_row, next_col))
        
        return area_size
    
    def _calculate_exploration_values(self, level_data, reachable_positions: Set[Tuple[int, int]]):
        """Calculate exploration values for all frontiers."""
        for frontier in self.frontiers:
            value = 0.0
            
            # Base value from potential area
            value += min(frontier.potential_area / 20.0, 0.5)
            
            # Bonus for exploration frontiers
            if frontier.frontier_type == FrontierType.EXPLORATION:
                value += 0.3
            
            # Penalty for blocked frontiers
            elif frontier.frontier_type == FrontierType.BLOCKED:
                value += 0.1
            
            # Moderate value for hazard frontiers (might be navigable)
            elif frontier.frontier_type == FrontierType.HAZARD:
                value += 0.2
            
            # Bonus for physics frontiers (advanced techniques)
            elif frontier.frontier_type == FrontierType.PHYSICS:
                value += 0.25
            
            # Distance from center bonus (encourage exploration of edges)
            center_row = len(level_data.tiles) // 2
            center_col = len(level_data.tiles[0]) // 2
            distance_from_center = math.sqrt(
                (frontier.position[0] - center_row) ** 2 +
                (frontier.position[1] - center_col) ** 2
            )
            max_distance = math.sqrt(center_row ** 2 + center_col ** 2)
            value += (distance_from_center / max_distance) * 0.2
            
            frontier.exploration_value = min(value, 1.0)
    
    def _calculate_accessibility_scores(self, reachable_positions: Set[Tuple[int, int]]):
        """Calculate accessibility scores for all frontiers."""
        # Find the centroid of reachable positions
        if not reachable_positions:
            return
        
        centroid_row = sum(pos[0] for pos in reachable_positions) / len(reachable_positions)
        centroid_col = sum(pos[1] for pos in reachable_positions) / len(reachable_positions)
        
        for frontier in self.frontiers:
            # Distance from centroid (closer = more accessible)
            distance = math.sqrt(
                (frontier.position[0] - centroid_row) ** 2 +
                (frontier.position[1] - centroid_col) ** 2
            )
            
            # Normalize to 0-1 range (closer = higher score)
            max_distance = 50  # Reasonable maximum distance
            accessibility = max(0.0, 1.0 - (distance / max_distance))
            
            frontier.accessibility_score = accessibility
    
    def get_frontiers_by_type(self, frontier_type: FrontierType) -> List[Frontier]:
        """Get all frontiers of a specific type."""
        return [f for f in self.frontiers if f.frontier_type == frontier_type]
    
    def get_high_value_frontiers(self, threshold: float = 0.5) -> List[Frontier]:
        """Get frontiers with exploration value above threshold."""
        return [f for f in self.frontiers if f.exploration_value >= threshold]
    
    def get_accessible_frontiers(self, threshold: float = 0.3) -> List[Frontier]:
        """Get frontiers with accessibility score above threshold."""
        return [f for f in self.frontiers if f.accessibility_score >= threshold]
    
    def get_frontier_at_position(self, position: Tuple[int, int]) -> Optional[Frontier]:
        """Get frontier at a specific position."""
        return self.frontier_map.get(position)
    
    def calculate_curiosity_bonus(self, position: Tuple[int, int], radius: int = 3) -> float:
        """
        Calculate curiosity bonus for a position based on nearby frontiers.
        
        Args:
            position: Position to calculate bonus for
            radius: Search radius for nearby frontiers
            
        Returns:
            Curiosity bonus value (0.0 to 1.0)
        """
        row, col = position
        bonus = 0.0
        
        for frontier in self.frontiers:
            f_row, f_col = frontier.position
            distance = max(abs(f_row - row), abs(f_col - col))
            
            if distance <= radius:
                # Closer frontiers contribute more to curiosity
                distance_factor = 1.0 - (distance / radius)
                frontier_contribution = frontier.exploration_value * distance_factor
                bonus += frontier_contribution
        
        return min(bonus, 1.0)
    
    def get_exploration_statistics(self) -> Dict[str, any]:
        """Get statistics about detected frontiers."""
        if not self.frontiers:
            return {'total_frontiers': 0}
        
        type_counts = {}
        for frontier_type in FrontierType:
            type_counts[frontier_type.value] = len(self.get_frontiers_by_type(frontier_type))
        
        return {
            'total_frontiers': len(self.frontiers),
            'type_distribution': type_counts,
            'avg_exploration_value': sum(f.exploration_value for f in self.frontiers) / len(self.frontiers),
            'avg_accessibility_score': sum(f.accessibility_score for f in self.frontiers) / len(self.frontiers),
            'total_potential_area': sum(f.potential_area for f in self.frontiers),
            'high_value_frontiers': len(self.get_high_value_frontiers()),
            'accessible_frontiers': len(self.get_accessible_frontiers())
        }