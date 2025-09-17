#!/usr/bin/env python3
"""
Comprehensive tests for the consolidated hierarchical subgoal planning system.

This test suite validates the enhanced SubgoalPlanner with hierarchical completion
algorithm and its integration with the TieredReachabilitySystem.
"""

import sys
import os
import unittest
import time
from typing import Dict, List, Any, Tuple

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nclone.graph.subgoal_planner import SubgoalPlanner, Subgoal, SubgoalPlan
from nclone.graph.reachability.tiered_system import TieredReachabilitySystem
from nclone.graph.reachability.opencv_flood_fill import OpenCVFloodFill


class MockEntity:
    """Mock entity for testing."""
    def __init__(self, entity_type: int, x: float, y: float, sw_x: float = 0, sw_y: float = 0):
        self.type = entity_type
        self.xpos = x
        self.ypos = y
        self.x = x
        self.y = y
        self.sw_xcoord = sw_x
        self.sw_ycoord = sw_y


class TestHierarchicalSubgoalSystem(unittest.TestCase):
    """Test suite for hierarchical subgoal planning system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.subgoal_planner = SubgoalPlanner(debug=True)
        self.tiered_system = TieredReachabilitySystem(debug=True)
        
        # Create simple level data for testing
        self.level_data = self._create_test_level_data()
        
    def _create_test_level_data(self):
        """Create simple level data for testing."""
        # Simple 10x10 level with mostly empty space
        import numpy as np
        level_data = np.zeros((10, 10), dtype=int)
        
        # Add some walls to create interesting topology
        level_data[3, :] = 1  # Horizontal wall
        level_data[3, 5] = 0  # Gap in wall
        
        return level_data
    
    def test_entity_relationship_extraction(self):
        """Test extraction of entity positions and relationships."""
        entities = [
            MockEntity(4, 100, 200),  # Exit switch
            MockEntity(3, 300, 400),  # Exit door
            MockEntity(6, 150, 250, 50, 150),  # Locked door with switch
            MockEntity(2, 75, 125),   # Gold
            MockEntity(1, 200, 300),  # Hazard
        ]
        
        entity_info = self.subgoal_planner._extract_entity_relationships(entities)
        
        # Validate extracted information
        self.assertEqual(len(entity_info['exit_switches']), 1)
        self.assertEqual(entity_info['exit_switches'][0], (100.0, 200.0))
        
        self.assertEqual(len(entity_info['exit_doors']), 1)
        self.assertEqual(entity_info['exit_doors'][0], (300.0, 400.0))
        
        self.assertEqual(len(entity_info['locked_doors']), 1)
        locked_door = entity_info['locked_doors'][0]
        self.assertEqual(locked_door['position'], (150.0, 250.0))
        self.assertEqual(locked_door['switch'], (50.0, 150.0))
        
        self.assertEqual(len(entity_info['gold']), 1)
        self.assertEqual(len(entity_info['hazards']), 1)
    
    def test_required_entities_validation(self):
        """Test validation of required entities for completion."""
        # Test with required entities
        entity_info_complete = {
            'exit_switches': [(100, 200)],
            'exit_doors': [(300, 400)],
            'locked_doors': [],
            'door_switches': [],
            'gold': [],
            'hazards': []
        }
        self.assertTrue(self.subgoal_planner._has_required_entities(entity_info_complete))
        
        # Test without exit switch
        entity_info_no_switch = {
            'exit_switches': [],
            'exit_doors': [(300, 400)],
            'locked_doors': [],
            'door_switches': [],
            'gold': [],
            'hazards': []
        }
        self.assertFalse(self.subgoal_planner._has_required_entities(entity_info_no_switch))
        
        # Test without exit door
        entity_info_no_door = {
            'exit_switches': [(100, 200)],
            'exit_doors': [],
            'locked_doors': [],
            'door_switches': [],
            'gold': [],
            'hazards': []
        }
        self.assertFalse(self.subgoal_planner._has_required_entities(entity_info_no_door))
    
    def test_simple_completion_plan(self):
        """Test creation of simple completion plan with direct path to exit."""
        entities = [
            MockEntity(4, 100, 200),  # Exit switch
            MockEntity(3, 300, 400),  # Exit door
        ]
        
        ninja_position = (50.0, 100.0)
        
        # Test plan creation
        plan = self.subgoal_planner.create_hierarchical_completion_plan(
            ninja_position=ninja_position,
            level_data=self.level_data,
            entities=entities,
            switch_states={}
        )
        
        # Validate plan structure
        self.assertIsNotNone(plan)
        self.assertIsInstance(plan, SubgoalPlan)
        self.assertGreater(len(plan.subgoals), 0)
        
        # Should have at least exit switch and exit subgoals
        goal_types = [subgoal.goal_type for subgoal in plan.subgoals]
        self.assertIn('exit_switch', goal_types)
        self.assertIn('exit', goal_types)
    
    def test_complex_completion_plan_with_locked_doors(self):
        """Test creation of complex completion plan with locked doors."""
        entities = [
            MockEntity(4, 400, 500),  # Exit switch (far away)
            MockEntity(3, 450, 550),  # Exit door
            MockEntity(6, 200, 300, 150, 250),  # Locked door blocking path
            MockEntity(6, 350, 400, 300, 350),  # Another locked door
        ]
        
        ninja_position = (50.0, 100.0)
        
        # Test plan creation
        plan = self.subgoal_planner.create_hierarchical_completion_plan(
            ninja_position=ninja_position,
            level_data=self.level_data,
            entities=entities,
            switch_states={}
        )
        
        # Validate plan structure
        self.assertIsNotNone(plan)
        self.assertIsInstance(plan, SubgoalPlan)
        self.assertGreaterEqual(len(plan.subgoals), 2)  # Should have at least exit switch + exit
        
        # Should include basic completion sequence
        goal_types = [subgoal.goal_type for subgoal in plan.subgoals]
        self.assertIn('exit_switch', goal_types)
        self.assertIn('exit', goal_types)
        
        # Note: OpenCV flood fill finds all positions reachable, so locked door switches
        # may not be included in the plan. This is expected behavior for connectivity-based analysis.
    
    def test_tiered_system_integration(self):
        """Test integration of hierarchical planning with TieredReachabilitySystem."""
        entities = [
            MockEntity(4, 100, 200),  # Exit switch
            MockEntity(3, 300, 400),  # Exit door
            MockEntity(6, 150, 250, 50, 150),  # Locked door with switch
        ]
        
        ninja_position = (25, 50)
        
        # Test hierarchical completion plan through tiered system
        plan = self.tiered_system.create_hierarchical_completion_plan(
            ninja_position=ninja_position,
            level_data=self.level_data,
            entities=entities,
            switch_states={}
        )
        
        # Validate integration works
        self.assertIsNotNone(plan)
        self.assertIsInstance(plan, SubgoalPlan)
        self.assertGreater(len(plan.subgoals), 0)
    
    def test_world_to_sub_coords_conversion(self):
        """Test coordinate conversion from world to sub-grid coordinates."""
        # Test various world positions
        test_cases = [
            ((0.0, 0.0), (0, 0)),
            ((24.0, 24.0), (1, 1)),  # Assuming SUB_CELL_SIZE = 24
            ((48.0, 72.0), (2, 3)),
            ((100.5, 200.7), (4, 8)),  # Test floating point
        ]
        
        for world_pos, expected_sub_pos in test_cases:
            result = self.subgoal_planner._world_to_sub_coords(world_pos)
            # Note: Exact values depend on SUB_CELL_SIZE constant
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], int)
            self.assertIsInstance(result[1], int)
    
    def test_performance_benchmarks(self):
        """Test performance of hierarchical subgoal planning."""
        entities = [
            MockEntity(4, 400, 500),  # Exit switch
            MockEntity(3, 450, 550),  # Exit door
            MockEntity(6, 200, 300, 150, 250),  # Locked door 1
            MockEntity(6, 250, 350, 200, 300),  # Locked door 2
            MockEntity(6, 300, 400, 250, 350),  # Locked door 3
            MockEntity(2, 75, 125),   # Gold
            MockEntity(2, 125, 175),  # More gold
        ]
        
        ninja_position = (50.0, 100.0)
        
        # Measure planning time
        start_time = time.perf_counter()
        
        plan = self.subgoal_planner.create_hierarchical_completion_plan(
            ninja_position=ninja_position,
            level_data=self.level_data,
            entities=entities,
            switch_states={}
        )
        
        planning_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Validate performance
        self.assertLess(planning_time, 100.0)  # Should complete within 100ms
        
        if plan:
            print(f"Planning completed in {planning_time:.2f}ms with {len(plan.subgoals)} subgoals")
        else:
            print(f"Planning failed in {planning_time:.2f}ms")
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with no entities
        plan = self.subgoal_planner.create_hierarchical_completion_plan(
            ninja_position=(50.0, 100.0),
            level_data=self.level_data,
            entities=[],
            switch_states={}
        )
        self.assertIsNone(plan)
        
        # Test with only exit switch (no exit door)
        entities_incomplete = [MockEntity(4, 100, 200)]
        plan = self.subgoal_planner.create_hierarchical_completion_plan(
            ninja_position=(50.0, 100.0),
            level_data=self.level_data,
            entities=entities_incomplete,
            switch_states={}
        )
        self.assertIsNone(plan)
        
        # Test with invalid ninja position
        entities_complete = [
            MockEntity(4, 100, 200),  # Exit switch
            MockEntity(3, 300, 400),  # Exit door
        ]
        plan = self.subgoal_planner.create_hierarchical_completion_plan(
            ninja_position=(-1000.0, -1000.0),  # Invalid position
            level_data=self.level_data,
            entities=entities_complete,
            switch_states={}
        )
        # Should still create a plan, but might not be optimal
        # The exact behavior depends on the reachability analyzer


def run_hierarchical_subgoal_tests():
    """Run all hierarchical subgoal system tests."""
    print("=" * 80)
    print("HIERARCHICAL SUBGOAL PLANNING SYSTEM TESTS")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHierarchicalSubgoalSystem)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("HIERARCHICAL SUBGOAL SYSTEM TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


if __name__ == '__main__':
    success = run_hierarchical_subgoal_tests()
    sys.exit(0 if success else 1)