"""
Unit tests for the simplified completion strategy system.

This module tests the SimplifiedCompletionStrategy and updated SubgoalPlanner
to ensure they provide clear, RL-friendly objectives for Phase 1 training.
"""

import unittest
import numpy as np
from typing import List, Dict, Any
import sys
import os

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nclone.graph.simple_objective_system import (
    SimplifiedCompletionStrategy, SimpleObjective, ObjectiveType
)
from nclone.graph.subgoal_planner import SubgoalPlanner
from nclone.graph.subgoal_types import SubgoalPlan


class MockEntity:
    """Mock entity for testing."""
    
    def __init__(self, entity_type: str, x: float, y: float, entity_id: str = None):
        self.entity_type = entity_type
        self.x = x
        self.y = y
        self.id = entity_id or f"{entity_type}_{x}_{y}"


class TestSimplifiedCompletionStrategy(unittest.TestCase):
    """Test the SimplifiedCompletionStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = SimplifiedCompletionStrategy(debug=False)
        self.level_data = np.zeros((42, 23), dtype=int)  # Simple open level
        self.ninja_pos = (100.0, 100.0)
        
    def test_exit_switch_priority(self):
        """Test that exit switch gets highest priority when reachable."""
        entities = [
            MockEntity('exit_switch', 200.0, 200.0),
            MockEntity('door_switch', 150.0, 150.0),
            MockEntity('exit_door', 300.0, 300.0)
        ]
        
        objective = self.strategy.get_next_objective(
            self.ninja_pos, self.level_data, entities, {}
        )
        
        self.assertIsNotNone(objective)
        self.assertEqual(objective.objective_type, ObjectiveType.REACH_EXIT_SWITCH)
        self.assertEqual(objective.position, (200.0, 200.0))
        self.assertEqual(objective.priority, 1.0)
        
    def test_door_switch_when_exit_switch_unreachable(self):
        """Test that door switch is selected when exit switch is unreachable."""
        # Create a level with walls blocking the exit switch
        level_with_walls = np.zeros((42, 23), dtype=int)
        # Add walls around exit switch area
        for i in range(10, 15):
            for j in range(10, 15):
                level_with_walls[i, j] = 1  # Solid wall
        
        entities = [
            MockEntity('exit_switch', 312.0, 312.0),  # Behind walls (13*24, 13*24)
            MockEntity('door_switch', 150.0, 150.0),  # Accessible
        ]
        
        objective = self.strategy.get_next_objective(
            self.ninja_pos, level_with_walls, entities, {}
        )
        
        # Should select door switch since exit switch is blocked
        self.assertIsNotNone(objective)
        self.assertEqual(objective.objective_type, ObjectiveType.REACH_DOOR_SWITCH)
        self.assertEqual(objective.position, (150.0, 150.0))
        
    def test_exit_door_when_no_switches(self):
        """Test that exit door is selected when no switches are present."""
        entities = [
            MockEntity('exit_door', 200.0, 200.0)
        ]
        
        objective = self.strategy.get_next_objective(
            self.ninja_pos, self.level_data, entities, {}
        )
        
        self.assertIsNotNone(objective)
        self.assertEqual(objective.objective_type, ObjectiveType.REACH_EXIT_DOOR)
        self.assertEqual(objective.position, (200.0, 200.0))
        
    def test_any_switch_fallback(self):
        """Test fallback to any reachable switch."""
        entities = [
            MockEntity('some_switch', 150.0, 150.0),
            MockEntity('another_switch', 250.0, 250.0)
        ]
        
        objective = self.strategy.get_next_objective(
            self.ninja_pos, self.level_data, entities, {}
        )
        
        self.assertIsNotNone(objective)
        self.assertEqual(objective.objective_type, ObjectiveType.REACH_SWITCH)
        # Should select the nearest switch
        self.assertEqual(objective.position, (150.0, 150.0))
        
    def test_no_objective_when_nothing_reachable(self):
        """Test that None is returned when nothing is reachable."""
        # Create level with walls blocking access to entities
        # Leave ninja area clear but block entity areas
        level_blocked = np.zeros((42, 23), dtype=int)
        
        # Block the area around the entities (tiles 8-10 for 200.0 position)
        for i in range(7, 12):
            for j in range(7, 12):
                if 0 <= i < 42 and 0 <= j < 23:
                    level_blocked[i, j] = 1  # Wall
        
        # Block area around 300.0 position (tiles 12-14)
        for i in range(11, 16):
            for j in range(11, 16):
                if 0 <= i < 42 and 0 <= j < 23:
                    level_blocked[i, j] = 1  # Wall
        
        entities = [
            MockEntity('exit_switch', 200.0, 200.0),  # At tile (8,8) - blocked
            MockEntity('exit_door', 300.0, 300.0)     # At tile (12,12) - blocked
        ]
        
        objective = self.strategy.get_next_objective(
            self.ninja_pos, level_blocked, entities, {}
        )
        
        # Should return None since entities are unreachable
        self.assertIsNone(objective)
        
    def test_switch_state_handling(self):
        """Test that activated switches are ignored."""
        entities = [
            MockEntity('door_switch', 150.0, 150.0, 'switch_1'),
            MockEntity('door_switch', 250.0, 250.0, 'switch_2'),
        ]
        
        switch_states = {'switch_1': True}  # First switch already activated
        
        objective = self.strategy.get_next_objective(
            self.ninja_pos, self.level_data, entities, switch_states
        )
        
        self.assertIsNotNone(objective)
        # Should select the second switch since first is already activated
        self.assertEqual(objective.position, (250.0, 250.0))
        
    def test_objective_reached_detection(self):
        """Test objective reached detection."""
        entities = [MockEntity('exit_switch', 120.0, 120.0)]
        
        objective = self.strategy.get_next_objective(
            self.ninja_pos, self.level_data, entities, {}
        )
        
        self.assertIsNotNone(objective)
        
        # Test when ninja is close to objective
        close_pos = (115.0, 115.0)  # Within 24 pixel threshold
        self.assertTrue(self.strategy.is_objective_reached(close_pos))
        
        # Test when ninja is far from objective
        far_pos = (200.0, 200.0)  # Beyond 24 pixel threshold
        self.assertFalse(self.strategy.is_objective_reached(far_pos))
        
    def test_rl_feature_encoding(self):
        """Test RL feature encoding for TASK_003 compatibility."""
        entities = [MockEntity('exit_switch', 200.0, 200.0)]
        
        objective = self.strategy.get_next_objective(
            self.ninja_pos, self.level_data, entities, {}
        )
        
        features = self.strategy.get_objective_for_rl_features(self.ninja_pos)
        
        self.assertEqual(features['has_objective'], 1.0)
        self.assertGreater(features['objective_distance'], 0.0)
        self.assertLessEqual(features['objective_distance'], 1.0)
        self.assertEqual(features['objective_type_exit_switch'], 1.0)
        self.assertEqual(features['objective_type_door_switch'], 0.0)
        self.assertEqual(features['objective_type_exit_door'], 0.0)
        self.assertEqual(features['objective_type_switch'], 0.0)
        self.assertEqual(features['objective_priority'], 1.0)
        
    def test_no_objective_rl_features(self):
        """Test RL feature encoding when no objective exists."""
        # Empty entities list
        objective = self.strategy.get_next_objective(
            self.ninja_pos, self.level_data, [], {}
        )
        
        self.assertIsNone(objective)
        
        features = self.strategy.get_objective_for_rl_features(self.ninja_pos)
        
        self.assertEqual(features['has_objective'], 0.0)
        self.assertEqual(features['objective_distance'], 1.0)  # Max distance
        self.assertEqual(features['objective_priority'], 0.0)


class TestSubgoalPlannerIntegration(unittest.TestCase):
    """Test the updated SubgoalPlanner with simplified strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.planner = SubgoalPlanner(debug=False)
        self.level_data = np.zeros((42, 23), dtype=int)
        self.ninja_pos = (100.0, 100.0)
        
    def test_hierarchical_completion_plan_backward_compatibility(self):
        """Test that the hierarchical completion plan API still works."""
        entities = [
            MockEntity('exit_switch', 200.0, 200.0),
            MockEntity('exit_door', 300.0, 300.0)
        ]
        
        plan = self.planner.create_hierarchical_completion_plan(
            self.ninja_pos, self.level_data, entities, {}
        )
        
        self.assertIsNotNone(plan)
        self.assertIsInstance(plan, SubgoalPlan)
        self.assertEqual(len(plan.subgoals), 1)  # Single objective in simplified approach
        self.assertEqual(len(plan.execution_order), 1)
        self.assertGreater(plan.total_estimated_cost, 0.0)
        
        # Check that subgoal has correct properties
        subgoal = plan.subgoals[0]
        self.assertEqual(subgoal.goal_type, 'exit_switch')
        
    def test_current_objective_access(self):
        """Test direct access to current objective."""
        entities = [MockEntity('exit_switch', 200.0, 200.0)]
        
        # Create plan to set current objective
        plan = self.planner.create_hierarchical_completion_plan(
            self.ninja_pos, self.level_data, entities, {}
        )
        
        self.assertIsNotNone(plan)
        
        # Test direct objective access
        current_objective = self.planner.get_current_objective()
        self.assertIsNotNone(current_objective)
        self.assertEqual(current_objective.objective_type, ObjectiveType.REACH_EXIT_SWITCH)
        
    def test_rl_integration_methods(self):
        """Test RL integration methods."""
        entities = [MockEntity('exit_switch', 200.0, 200.0)]
        
        # Create plan
        plan = self.planner.create_hierarchical_completion_plan(
            self.ninja_pos, self.level_data, entities, {}
        )
        
        self.assertIsNotNone(plan)
        
        # Test RL feature encoding
        features = self.planner.get_objective_for_rl_features(self.ninja_pos)
        self.assertIsInstance(features, dict)
        self.assertIn('has_objective', features)
        self.assertEqual(features['has_objective'], 1.0)
        
        # Test objective reached detection
        close_pos = (195.0, 195.0)  # Close to switch
        far_pos = (50.0, 50.0)     # Far from switch
        
        self.assertTrue(self.planner.is_objective_reached(close_pos, threshold=30.0))
        self.assertFalse(self.planner.is_objective_reached(far_pos, threshold=30.0))
        
        # Test objective clearing
        self.planner.clear_objective()
        cleared_objective = self.planner.get_current_objective()
        self.assertIsNone(cleared_objective)
        
    def test_switch_state_updates(self):
        """Test that objectives update when switch states change."""
        entities = [
            MockEntity('exit_switch', 300.0, 300.0, 'exit_switch'),
            MockEntity('door_switch', 150.0, 150.0, 'door_switch')
        ]
        
        # Initially, should target exit switch
        plan1 = self.planner.create_hierarchical_completion_plan(
            self.ninja_pos, self.level_data, entities, {}
        )
        
        self.assertIsNotNone(plan1)
        self.assertEqual(plan1.subgoals[0].goal_type, 'exit_switch')
        
        # After activating door switch, should still target exit switch
        switch_states = {'door_switch': True}
        plan2 = self.planner.create_hierarchical_completion_plan(
            self.ninja_pos, self.level_data, entities, switch_states
        )
        
        self.assertIsNotNone(plan2)
        self.assertEqual(plan2.subgoals[0].goal_type, 'exit_switch')
        
    def test_no_plan_when_no_objectives(self):
        """Test that None is returned when no objectives are available."""
        # Empty entities list
        plan = self.planner.create_hierarchical_completion_plan(
            self.ninja_pos, self.level_data, [], {}
        )
        
        self.assertIsNone(plan)


class TestSimplificationBenefits(unittest.TestCase):
    """Test that the simplified approach provides expected benefits."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.planner = SubgoalPlanner(debug=False)
        self.level_data = np.zeros((42, 23), dtype=int)
        
    def test_single_clear_objective(self):
        """Test that only one clear objective is provided at a time."""
        entities = [
            MockEntity('exit_switch', 200.0, 200.0),
            MockEntity('door_switch', 150.0, 150.0),
            MockEntity('exit_door', 300.0, 300.0),
            MockEntity('another_switch', 250.0, 250.0)
        ]
        
        plan = self.planner.create_hierarchical_completion_plan(
            (100.0, 100.0), self.level_data, entities, {}
        )
        
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.subgoals), 1)  # Only one objective
        self.assertEqual(plan.subgoals[0].goal_type, 'exit_switch')  # Highest priority
        
    def test_reactive_behavior(self):
        """Test that objectives change reactively based on switch states."""
        entities = [
            MockEntity('exit_switch', 300.0, 300.0, 'exit_switch'),
            MockEntity('door_switch', 150.0, 150.0, 'door_switch')
        ]
        
        # Block exit switch with walls
        level_with_walls = np.zeros((42, 23), dtype=int)
        for i in range(12, 15):
            for j in range(12, 15):
                level_with_walls[i, j] = 1
        
        # Should target door switch when exit switch is blocked
        plan1 = self.planner.create_hierarchical_completion_plan(
            (100.0, 100.0), level_with_walls, entities, {}
        )
        
        self.assertIsNotNone(plan1)
        self.assertEqual(plan1.subgoals[0].goal_type, 'locked_door_switch')
        
        # After activating door switch, should target exit switch if now reachable
        switch_states = {'door_switch': True}
        plan2 = self.planner.create_hierarchical_completion_plan(
            (100.0, 100.0), self.level_data, entities, switch_states  # Open level now
        )
        
        self.assertIsNotNone(plan2)
        self.assertEqual(plan2.subgoals[0].goal_type, 'exit_switch')
        
    def test_performance_characteristics(self):
        """Test that the simplified approach is fast."""
        import time
        
        entities = [
            MockEntity('exit_switch', 200.0, 200.0),
            MockEntity('door_switch', 150.0, 150.0),
            MockEntity('exit_door', 300.0, 300.0)
        ]
        
        # Time multiple plan creations
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            plan = self.planner.create_hierarchical_completion_plan(
                (100.0, 100.0), self.level_data, entities, {}
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            times.append(elapsed_ms)
            
            self.assertIsNotNone(plan)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Should be very fast (much faster than complex hierarchical planning)
        self.assertLess(avg_time, 50.0, f"Average time too slow: {avg_time:.2f}ms")
        self.assertLess(max_time, 100.0, f"Max time too slow: {max_time:.2f}ms")
        
        print(f"Simplified planning performance: {avg_time:.2f}ms avg, {max_time:.2f}ms max")


if __name__ == '__main__':
    unittest.main(verbosity=2)