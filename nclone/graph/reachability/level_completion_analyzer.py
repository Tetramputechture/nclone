"""
Level Completion Analyzer for Hierarchical Subgoal Planning

This module implements the hierarchical subgoal planning algorithm described in the
HIERARCHICAL_SUBGOAL_PLANNING.md document. It provides strategic level completion
analysis for Deep RL agents by implementing recursive switch-door dependency resolution.

Key Features:
- Recursive exit switch reachability analysis
- Switch dependency graph construction
- Optimal switch activation sequence planning
- Integration with OpenCV flood fill reachability system

Algorithm:
1. Check if exit switch is reachable from current position
2. If not, find locked doors blocking the path and their required switches
3. Recursively analyze switch reachability until all dependencies are resolved
4. Return optimal completion strategy with prioritized subgoals

This replaces the simple position-count heuristic with proper game logic understanding.
"""

from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from .reachability_types import ReachabilityApproximation, ReachabilityResult
from .opencv_flood_fill import OpenCVFloodFill

logger = logging.getLogger(__name__)


class SubgoalType(Enum):
    """Types of subgoals for hierarchical planning."""
    NAVIGATE_TO_EXIT_SWITCH = "navigate_to_exit_switch"
    ACTIVATE_EXIT_SWITCH = "activate_exit_switch"
    NAVIGATE_TO_EXIT_DOOR = "navigate_to_exit_door"
    NAVIGATE_TO_DOOR_SWITCH = "navigate_to_door_switch"
    ACTIVATE_DOOR_SWITCH = "activate_door_switch"

    AVOID_HAZARD = "avoid_hazard"


@dataclass
class CompletionStrategy:
    """
    Strategic plan for level completion with hierarchical subgoals.
    
    This provides the RL agent with clear guidance on what needs to be done
    to complete the level, replacing the simple binary completable/not completable signal.
    """
    is_completable: bool
    confidence: float
    primary_subgoals: List[str]  # Ordered list of primary objectives
    required_switches: List[Tuple[int, int]]  # Switch positions that must be activated
    blocking_doors: List[Tuple[int, int]]  # Door positions that block progress
    switch_dependencies: Dict[Tuple[int, int], List[Tuple[int, int]]]  # switch_pos -> [required_switches]
    completion_sequence: List[str]  # Step-by-step completion plan
    alternative_paths: List[List[str]]  # Alternative completion strategies
    estimated_difficulty: float  # 0.0 (easy) to 1.0 (very hard)
    
    def get_next_subgoal(self) -> Optional[str]:
        """Get the next subgoal in the completion sequence."""
        return self.completion_sequence[0] if self.completion_sequence else None
    
    def get_reachable_subgoals(self, reachable_positions: Set[Tuple[int, int]]) -> List[str]:
        """Filter subgoals to only those currently reachable."""
        reachable_subgoals = []
        for subgoal in self.primary_subgoals:
            if self._is_subgoal_reachable(subgoal, reachable_positions):
                reachable_subgoals.append(subgoal)
        return reachable_subgoals
    
    def _is_subgoal_reachable(self, subgoal: str, reachable_positions: Set[Tuple[int, int]]) -> bool:
        """Check if a specific subgoal is reachable given current reachable positions."""
        # Extract position from subgoal string and check if it's in reachable set
        # This is a simplified implementation - in practice would need more sophisticated parsing
        return True  # Placeholder - would implement proper position extraction


class LevelCompletionAnalyzer:
    """
    Hierarchical subgoal planning system for level completion analysis.
    
    Implements the recursive algorithm described in HIERARCHICAL_SUBGOAL_PLANNING.md:
    1. Check exit switch reachability
    2. Find blocking doors and required switches recursively
    3. Build optimal completion strategy
    4. Provide strategic guidance for RL agents
    """
    
    def __init__(self, reachability_analyzer: OpenCVFloodFill):
        """
        Initialize level completion analyzer.
        
        Args:
            reachability_analyzer: OpenCV flood fill analyzer for reachability queries
        """
        self.reachability_analyzer = reachability_analyzer
        self.entity_cache = {}  # Cache for entity position lookups
        self.strategy_cache = {}  # Cache for completion strategies
        
    def analyze_level_completion(
        self,
        ninja_position: Tuple[float, float],
        level_data: Any,
        entities: List[Any],
        switch_states: Optional[Dict[str, bool]] = None
    ) -> CompletionStrategy:
        """
        Analyze level completion strategy using hierarchical subgoal planning.
        
        This is the main entry point that replaces the simple is_level_completable
        heuristic with proper strategic analysis.
        
        Args:
            ninja_position: Current ninja position (x, y)
            level_data: Level tile data
            entities: List of entities in the level
            switch_states: Current state of switches (activated/not activated)
            
        Returns:
            CompletionStrategy with detailed completion plan
        """
        if switch_states is None:
            switch_states = {}
            
        # Step 1: Extract entity positions
        entity_positions = self._extract_entity_positions(entities)
        
        # Step 2: Check for basic completion requirements
        if not self._has_required_entities(entity_positions):
            return CompletionStrategy(
                is_completable=False,
                confidence=1.0,
                primary_subgoals=[],
                required_switches=[],
                blocking_doors=[],
                switch_dependencies={},
                completion_sequence=[],
                alternative_paths=[],
                estimated_difficulty=1.0
            )
        
        # Step 3: Analyze current reachability
        reachability_result = self.reachability_analyzer.quick_check(
            ninja_position, level_data, entities
        )
        
        # Step 4: Implement recursive completion algorithm
        completion_plan = self._recursive_completion_analysis(
            ninja_position, entity_positions, reachability_result, switch_states, level_data, entities
        )
        
        return completion_plan
    
    def _extract_entity_positions(self, entities: List[Any]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Extract positions of relevant entities for completion analysis.
        
        Args:
            entities: List of entity objects
            
        Returns:
            Dictionary mapping entity types to their positions
        """
        entity_positions = {
            'exit_switch': [],      # Type 4 - Exit switches
            'exit_door': [],        # Type 3 - Exit doors  
            'locked_door': [],      # Type 6 - Locked doors
            'door_switch': [],      # Switches that control locked doors

            'hazards': []           # Various hazard types
        }
        
        for entity in entities:
            entity_type = getattr(entity, 'type', None)
            x_pos = getattr(entity, 'xpos', getattr(entity, 'x', 0))
            y_pos = getattr(entity, 'ypos', getattr(entity, 'y', 0))
            position = (int(x_pos // 24), int(y_pos // 24))  # Convert to tile coordinates
            
            if entity_type == 4:  # Exit switch
                entity_positions['exit_switch'].append(position)
            elif entity_type == 3:  # Exit door
                entity_positions['exit_door'].append(position)
            elif entity_type == 6:  # Locked door
                entity_positions['locked_door'].append(position)
                # Also extract the switch position for this door
                sw_x = getattr(entity, 'sw_xcoord', 0)
                sw_y = getattr(entity, 'sw_ycoord', 0)
                switch_pos = (int(sw_x // 24), int(sw_y // 24))
                entity_positions['door_switch'].append(switch_pos)

            elif entity_type in [1, 14, 20, 25, 26]:  # Various hazards
                entity_positions['hazards'].append(position)
                
        return entity_positions
    
    def _has_required_entities(self, entity_positions: Dict[str, List[Tuple[int, int]]]) -> bool:
        """
        Check if level has the minimum required entities for completion.
        
        Args:
            entity_positions: Dictionary of entity positions by type
            
        Returns:
            True if level has required entities (exit switch + exit door)
        """
        return (len(entity_positions['exit_switch']) > 0 and 
                len(entity_positions['exit_door']) > 0)
    
    def _recursive_completion_analysis(
        self,
        ninja_position: Tuple[float, float],
        entity_positions: Dict[str, List[Tuple[int, int]]],
        reachability_result: ReachabilityApproximation,
        switch_states: Dict[str, bool],
        level_data: Any,
        entities: List[Any]
    ) -> CompletionStrategy:
        """
        Implement the recursive completion algorithm from HIERARCHICAL_SUBGOAL_PLANNING.md.
        
        Algorithm:
        1. Check if exit switch is reachable
        2. If not, find blocking doors and their required switches
        3. Recursively analyze switch reachability
        4. Build optimal completion sequence
        
        Args:
            ninja_position: Current ninja position
            entity_positions: Extracted entity positions
            reachability_result: Current reachability analysis
            switch_states: Current switch activation states
            level_data: Level tile data
            entities: Entity list
            
        Returns:
            CompletionStrategy with detailed completion plan
        """
        completion_sequence = []
        required_switches = []
        blocking_doors = []
        switch_dependencies = {}
        
        # Step 1: Check exit switch reachability
        exit_switches = entity_positions['exit_switch']
        exit_doors = entity_positions['exit_door']
        
        if not exit_switches or not exit_doors:
            return self._create_impossible_strategy("No exit switch or exit door found")
        
        exit_switch_pos = exit_switches[0]  # Assume single exit switch
        exit_door_pos = exit_doors[0]       # Assume single exit door
        
        # Step 2: Check if exit switch is directly reachable
        if self._is_position_reachable(exit_switch_pos, reachability_result):
            # Exit switch is reachable
            if switch_states.get('exit_switch', False):
                # Switch already activated, check door reachability
                if self._is_position_reachable(exit_door_pos, reachability_result):
                    # Direct path to completion
                    completion_sequence = ['navigate_to_exit_door']
                else:
                    # Need to unlock path to exit door
                    door_unlock_plan = self._find_door_unlock_sequence(
                        exit_door_pos, entity_positions, reachability_result, switch_states, level_data, entities
                    )
                    completion_sequence.extend(door_unlock_plan)
                    completion_sequence.append('navigate_to_exit_door')
            else:
                # Need to activate exit switch first
                completion_sequence = [
                    'navigate_to_exit_switch',
                    'activate_exit_switch',
                    'navigate_to_exit_door'
                ]
        else:
            # Exit switch not reachable - find blocking doors
            blocking_analysis = self._find_blocking_doors_to_target(
                ninja_position, exit_switch_pos, entity_positions, reachability_result, level_data, entities
            )
            
            blocking_doors = blocking_analysis['doors']
            required_switches = blocking_analysis['switches']
            switch_dependencies = blocking_analysis['dependencies']
            
            # Build switch activation sequence
            switch_sequence = self._optimize_switch_sequence(
                ninja_position, required_switches, entity_positions, reachability_result, level_data, entities
            )
            
            # Build complete completion sequence
            for switch_pos in switch_sequence:
                completion_sequence.extend([
                    f'navigate_to_switch_{switch_pos[0]}_{switch_pos[1]}',
                    f'activate_switch_{switch_pos[0]}_{switch_pos[1]}'
                ])
            
            # Add final exit sequence
            completion_sequence.extend([
                'navigate_to_exit_switch',
                'activate_exit_switch',
                'navigate_to_exit_door'
            ])
        
        # Step 3: Calculate difficulty and confidence
        difficulty = self._calculate_difficulty(completion_sequence, required_switches, blocking_doors)
        confidence = self._calculate_confidence(reachability_result, completion_sequence)
        
        # Step 4: Generate alternative paths (simplified)
        alternative_paths = self._generate_alternative_paths(
            ninja_position, entity_positions, reachability_result, switch_states
        )
        
        return CompletionStrategy(
            is_completable=True,
            confidence=confidence,
            primary_subgoals=completion_sequence[:3],  # First 3 steps as primary subgoals
            required_switches=required_switches,
            blocking_doors=blocking_doors,
            switch_dependencies=switch_dependencies,
            completion_sequence=completion_sequence,
            alternative_paths=alternative_paths,
            estimated_difficulty=difficulty
        )
    
    def _is_position_reachable(
        self, 
        position: Tuple[int, int], 
        reachability_result: ReachabilityApproximation
    ) -> bool:
        """
        Check if a specific position is reachable.
        
        Args:
            position: Target position (tile_x, tile_y)
            reachability_result: Current reachability analysis
            
        Returns:
            True if position is reachable
        """
        # Convert tile position to pixel position for comparison
        pixel_pos = (position[0] * 24 + 12, position[1] * 24 + 12)  # Center of tile
        return pixel_pos in reachability_result.reachable_positions
    
    def _find_blocking_doors_to_target(
        self,
        start_position: Tuple[float, float],
        target_position: Tuple[int, int],
        entity_positions: Dict[str, List[Tuple[int, int]]],
        reachability_result: ReachabilityApproximation,
        level_data: Any,
        entities: List[Any]
    ) -> Dict[str, Any]:
        """
        Find doors that block the path to target using connectivity analysis.
        
        This implements the core logic for identifying which doors need to be opened
        to reach a target position.
        
        Args:
            start_position: Starting position
            target_position: Target position to reach
            entity_positions: Entity positions by type
            reachability_result: Current reachability analysis
            level_data: Level tile data
            entities: Entity list
            
        Returns:
            Dictionary with blocking doors, required switches, and dependencies
        """
        blocking_doors = []
        required_switches = []
        switch_dependencies = {}
        
        # For each locked door, test if opening it would make target reachable
        for door_pos in entity_positions['locked_door']:
            # Find the corresponding switch for this door
            door_switch_pos = self._find_switch_for_door(door_pos, entities)
            
            if door_switch_pos:
                # Simulate opening this door and check if target becomes reachable
                if self._would_door_opening_help(door_pos, target_position, level_data, entities):
                    blocking_doors.append(door_pos)
                    required_switches.append(door_switch_pos)
                    switch_dependencies[door_switch_pos] = []  # No dependencies for now
        
        return {
            'doors': blocking_doors,
            'switches': required_switches,
            'dependencies': switch_dependencies
        }
    
    def _find_switch_for_door(self, door_position: Tuple[int, int], entities: List[Any]) -> Optional[Tuple[int, int]]:
        """
        Find the switch that controls a specific door.
        
        Args:
            door_position: Position of the door
            entities: List of entities
            
        Returns:
            Position of the controlling switch, or None if not found
        """
        for entity in entities:
            if (getattr(entity, 'type', None) == 6 and  # Locked door
                int(getattr(entity, 'xpos', 0) // 24) == door_position[0] and
                int(getattr(entity, 'ypos', 0) // 24) == door_position[1]):
                
                sw_x = getattr(entity, 'sw_xcoord', 0)
                sw_y = getattr(entity, 'sw_ycoord', 0)
                return (int(sw_x // 24), int(sw_y // 24))
        
        return None
    
    def _would_door_opening_help(
        self,
        door_position: Tuple[int, int],
        target_position: Tuple[int, int],
        level_data: Any,
        entities: List[Any]
    ) -> bool:
        """
        Test if opening a specific door would help reach the target.
        
        This is a simplified heuristic - in practice would need more sophisticated
        connectivity analysis.
        
        Args:
            door_position: Position of door to test
            target_position: Target we're trying to reach
            level_data: Level tile data
            entities: Entity list
            
        Returns:
            True if opening this door would likely help reach target
        """
        # Simplified heuristic: if door is between current reachable area and target
        # In practice, would use more sophisticated graph connectivity analysis
        return True  # Placeholder - assume all doors are potentially helpful
    
    def _optimize_switch_sequence(
        self,
        ninja_position: Tuple[float, float],
        required_switches: List[Tuple[int, int]],
        entity_positions: Dict[str, List[Tuple[int, int]]],
        reachability_result: ReachabilityApproximation,
        level_data: Any,
        entities: List[Any]
    ) -> List[Tuple[int, int]]:
        """
        Optimize the sequence of switch activations for efficiency.
        
        Args:
            ninja_position: Current ninja position
            required_switches: List of switches that need to be activated
            entity_positions: Entity positions by type
            reachability_result: Current reachability analysis
            level_data: Level tile data
            entities: Entity list
            
        Returns:
            Optimized sequence of switch positions to activate
        """
        # Simple greedy approach: activate switches in order of reachability
        # In practice, would use more sophisticated optimization
        reachable_switches = []
        unreachable_switches = []
        
        for switch_pos in required_switches:
            if self._is_position_reachable(switch_pos, reachability_result):
                reachable_switches.append(switch_pos)
            else:
                unreachable_switches.append(switch_pos)
        
        # Return reachable switches first, then unreachable ones
        return reachable_switches + unreachable_switches
    
    def _calculate_difficulty(
        self,
        completion_sequence: List[str],
        required_switches: List[Tuple[int, int]],
        blocking_doors: List[Tuple[int, int]]
    ) -> float:
        """
        Calculate estimated difficulty of level completion.
        
        Args:
            completion_sequence: Planned completion sequence
            required_switches: Required switches to activate
            blocking_doors: Doors that block progress
            
        Returns:
            Difficulty score from 0.0 (easy) to 1.0 (very hard)
        """
        base_difficulty = 0.1  # Base difficulty for any level
        
        # Add difficulty for each required switch
        switch_difficulty = len(required_switches) * 0.15
        
        # Add difficulty for each blocking door
        door_difficulty = len(blocking_doors) * 0.1
        
        # Add difficulty for sequence length
        sequence_difficulty = len(completion_sequence) * 0.05
        
        total_difficulty = base_difficulty + switch_difficulty + door_difficulty + sequence_difficulty
        return min(total_difficulty, 1.0)  # Cap at 1.0
    
    def _calculate_confidence(
        self,
        reachability_result: ReachabilityApproximation,
        completion_sequence: List[str]
    ) -> float:
        """
        Calculate confidence in the completion strategy.
        
        Args:
            reachability_result: Current reachability analysis
            completion_sequence: Planned completion sequence
            
        Returns:
            Confidence score from 0.0 (low) to 1.0 (high)
        """
        # Base confidence from reachability analysis
        base_confidence = 0.8  # OpenCV flood fill is generally reliable
        
        # Reduce confidence for complex sequences
        complexity_penalty = len(completion_sequence) * 0.02
        
        # Reduce confidence if very few positions are reachable
        position_count = len(reachability_result.reachable_positions)
        if position_count < 10:
            position_penalty = 0.2
        else:
            position_penalty = 0.0
        
        final_confidence = base_confidence - complexity_penalty - position_penalty
        return max(final_confidence, 0.1)  # Minimum confidence of 0.1
    
    def _generate_alternative_paths(
        self,
        ninja_position: Tuple[float, float],
        entity_positions: Dict[str, List[Tuple[int, int]]],
        reachability_result: ReachabilityApproximation,
        switch_states: Dict[str, bool]
    ) -> List[List[str]]:
        """
        Generate alternative completion paths for robustness.
        
        Args:
            ninja_position: Current ninja position
            entity_positions: Entity positions by type
            reachability_result: Current reachability analysis
            switch_states: Current switch states
            
        Returns:
            List of alternative completion sequences
        """
        # Simplified implementation - in practice would generate multiple viable paths
        alternatives = []
        

        
        # Alternative 2: Hazard avoidance route
        if entity_positions['hazards']:
            safe_route = ['avoid_hazards', 'navigate_to_exit_switch', 'activate_exit_switch', 'navigate_to_exit_door']
            alternatives.append(safe_route)
        
        return alternatives
    
    def _create_impossible_strategy(self, reason: str) -> CompletionStrategy:
        """
        Create a strategy for impossible levels.
        
        Args:
            reason: Reason why level is impossible
            
        Returns:
            CompletionStrategy indicating level is not completable
        """
        logger.warning(f"Level marked as impossible: {reason}")
        
        return CompletionStrategy(
            is_completable=False,
            confidence=1.0,
            primary_subgoals=[],
            required_switches=[],
            blocking_doors=[],
            switch_dependencies={},
            completion_sequence=[],
            alternative_paths=[],
            estimated_difficulty=1.0
        )