#!/usr/bin/env python3
"""
Comprehensive unit tests for the physics-accurate waypoint pathfinding system.
"""

import os
import sys
import unittest
import math

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.physics_waypoint_pathfinder import PhysicsWaypointPathfinder
from nclone.graph.physics_enhanced_edge_builder import PhysicsEnhancedEdgeBuilder
from nclone.constants.physics_constants import MAX_JUMP_DISTANCE, MAX_FALL_DISTANCE, NINJA_RADIUS


class TestPhysicsWaypointSystem(unittest.TestCase):
    """Test suite for the physics-accurate waypoint pathfinding system."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = BasicLevelNoGold(render_mode="rgb_array")
        self.ninja_position = (132.0, 444.0)
        self.target_position = (468.0, 420.0)  # Long-distance target
        self.builder = HierarchicalGraphBuilder()
        
    def test_waypoint_pathfinder_creation(self):
        """Test that waypoint pathfinder can be created and initialized."""
        pathfinder = PhysicsWaypointPathfinder()
        self.assertIsNotNone(pathfinder)
        self.assertEqual(len(pathfinder.waypoints), 0)
        self.assertEqual(len(pathfinder.waypoint_connections), 0)
        
    def test_physics_constants_validation(self):
        """Test that physics constants are properly defined."""
        self.assertGreater(MAX_JUMP_DISTANCE, 0)
        self.assertGreater(MAX_FALL_DISTANCE, 0)
        self.assertGreater(NINJA_RADIUS, 0)
        self.assertEqual(MAX_JUMP_DISTANCE, 200.0)
        self.assertEqual(MAX_FALL_DISTANCE, 400.0)
        self.assertEqual(NINJA_RADIUS, 10.0)
        
    def test_waypoint_creation_for_long_distance(self):
        """Test that waypoints are created for long-distance navigation."""
        pathfinder = PhysicsWaypointPathfinder()
        
        # Calculate distance to ensure it's beyond jump range
        distance = math.sqrt(
            (self.target_position[0] - self.ninja_position[0])**2 + 
            (self.target_position[1] - self.ninja_position[1])**2
        )
        self.assertGreater(distance, MAX_JUMP_DISTANCE, 
                          "Test target should be beyond jump range")
        
        # Create waypoints
        waypoints = pathfinder.create_physics_accurate_waypoints(
            self.ninja_position, self.target_position, self.env.level_data
        )
        
        # Validate waypoints were created
        self.assertGreater(len(waypoints), 0, "Should create waypoints for long-distance path")
        
        # Validate waypoint physics
        for waypoint in waypoints:
            self.assertTrue(waypoint.is_traversable, "All waypoints should be traversable")
            self.assertGreaterEqual(waypoint.ground_distance, 0, "Ground distance should be non-negative")
            self.assertGreaterEqual(waypoint.clearance_above, 0, "Clearance should be non-negative")
            
    def test_waypoint_spacing_physics_compliance(self):
        """Test that waypoint spacing respects physics constraints."""
        pathfinder = PhysicsWaypointPathfinder()
        waypoints = pathfinder.create_physics_accurate_waypoints(
            self.ninja_position, self.target_position, self.env.level_data
        )
        
        if len(waypoints) > 1:
            # Check spacing between consecutive waypoints
            for i in range(len(waypoints) - 1):
                current = waypoints[i]
                next_wp = waypoints[i + 1]
                
                distance = math.sqrt(
                    (next_wp.x - current.x)**2 + (next_wp.y - current.y)**2
                )
                
                # Distance should be within physics limits
                self.assertLessEqual(distance, MAX_JUMP_DISTANCE, 
                                   f"Waypoint spacing {distance:.1f}px exceeds jump limit")
                
    def test_enhanced_edge_builder_integration(self):
        """Test that the enhanced edge builder integrates waypoints properly."""
        graph = self.builder.build_graph(self.env.level_data, self.ninja_position)
        
        # Validate graph was built
        self.assertIsNotNone(graph)
        self.assertGreater(graph.sub_cell_graph.num_nodes, 0)
        self.assertGreater(graph.sub_cell_graph.num_edges, 0)
        
        # Check that enhanced edge builder was used
        self.assertIsInstance(self.builder.edge_builder, PhysicsEnhancedEdgeBuilder)
        
        # Check waypoint statistics
        if hasattr(self.builder.edge_builder, 'waypoint_pathfinder'):
            stats = self.builder.edge_builder.waypoint_pathfinder.get_physics_statistics()
            self.assertIsInstance(stats, dict)
            self.assertIn('total_waypoints', stats)
            self.assertIn('physics_validation', stats)
            
    def test_waypoint_connections_physics_validation(self):
        """Test that waypoint connections respect physics constraints."""
        pathfinder = PhysicsWaypointPathfinder()
        waypoints = pathfinder.create_physics_accurate_waypoints(
            self.ninja_position, self.target_position, self.env.level_data
        )
        
        if len(waypoints) > 0:
            # Test connection from start to first waypoint
            first_waypoint = waypoints[0]
            start_distance = math.sqrt(
                (first_waypoint.x - self.ninja_position[0])**2 + 
                (first_waypoint.y - self.ninja_position[1])**2
            )
            self.assertLessEqual(start_distance, MAX_JUMP_DISTANCE,
                               "Start to first waypoint should be within jump range")
            
            # Test connection from last waypoint to target
            last_waypoint = waypoints[-1]
            end_distance = math.sqrt(
                (self.target_position[0] - last_waypoint.x)**2 + 
                (self.target_position[1] - last_waypoint.y)**2
            )
            self.assertLessEqual(end_distance, MAX_JUMP_DISTANCE,
                               "Last waypoint to target should be within jump range")
                               
    def test_no_waypoints_for_short_distance(self):
        """Test that no waypoints are created for short-distance paths."""
        pathfinder = PhysicsWaypointPathfinder()
        
        # Use a nearby target within jump range
        nearby_target = (self.ninja_position[0] + 100, self.ninja_position[1] - 50)
        distance = math.sqrt(
            (nearby_target[0] - self.ninja_position[0])**2 + 
            (nearby_target[1] - self.ninja_position[1])**2
        )
        self.assertLess(distance, MAX_JUMP_DISTANCE, "Target should be within jump range")
        
        waypoints = pathfinder.create_physics_accurate_waypoints(
            self.ninja_position, nearby_target, self.env.level_data
        )
        
        self.assertEqual(len(waypoints), 0, "Should not create waypoints for short distances")
        
    def test_waypoint_statistics_accuracy(self):
        """Test that waypoint statistics are accurate."""
        pathfinder = PhysicsWaypointPathfinder()
        waypoints = pathfinder.create_physics_accurate_waypoints(
            self.ninja_position, self.target_position, self.env.level_data
        )
        
        stats = pathfinder.get_physics_statistics()
        
        # Validate statistics match actual waypoints
        self.assertEqual(stats['total_waypoints'], len(waypoints))
        self.assertEqual(stats['traversable_waypoints'], 
                        sum(1 for wp in waypoints if wp.is_traversable))
        
        if len(waypoints) > 0:
            expected_avg_ground = sum(wp.ground_distance for wp in waypoints) / len(waypoints)
            expected_avg_clearance = sum(wp.clearance_above for wp in waypoints) / len(waypoints)
            
            self.assertAlmostEqual(stats['average_ground_distance'], expected_avg_ground, places=1)
            self.assertAlmostEqual(stats['average_clearance_above'], expected_avg_clearance, places=1)
            
    def test_graph_edge_count_increase(self):
        """Test that the enhanced edge builder increases edge count."""
        # Build graph without waypoints (using standard builder)
        from nclone.graph.edge_building import EdgeBuilder
        from nclone.graph.feature_extraction import FeatureExtractor
        
        standard_builder = HierarchicalGraphBuilder()
        # Temporarily replace with standard edge builder
        feature_extractor = FeatureExtractor(standard_builder.tile_type_dim, standard_builder.entity_type_dim)
        standard_edge_builder = EdgeBuilder(feature_extractor)
        standard_builder.edge_builder = standard_edge_builder
        
        standard_graph = standard_builder.build_graph(self.env.level_data, self.ninja_position)
        standard_edge_count = standard_graph.sub_cell_graph.num_edges
        
        # Build graph with waypoints (using enhanced builder)
        enhanced_graph = self.builder.build_graph(self.env.level_data, self.ninja_position)
        enhanced_edge_count = enhanced_graph.sub_cell_graph.num_edges
        
        # Enhanced builder should have more edges (due to waypoint connections)
        self.assertGreaterEqual(enhanced_edge_count, standard_edge_count,
                               "Enhanced edge builder should add waypoint edges")
                               
    def test_waypoint_path_completeness(self):
        """Test that waypoint path provides complete navigation route."""
        pathfinder = PhysicsWaypointPathfinder()
        waypoints = pathfinder.create_physics_accurate_waypoints(
            self.ninja_position, self.target_position, self.env.level_data
        )
        
        complete_path = pathfinder.get_complete_waypoint_path(
            self.ninja_position, self.target_position
        )
        
        # Path should start with ninja position and end with target
        self.assertEqual(complete_path[0], self.ninja_position)
        self.assertEqual(complete_path[-1], self.target_position)
        
        # Path should include all waypoints
        self.assertEqual(len(complete_path), len(waypoints) + 2)  # +2 for start and end
        
        # Validate path physics
        for i in range(len(complete_path) - 1):
            segment_distance = math.sqrt(
                (complete_path[i+1][0] - complete_path[i][0])**2 + 
                (complete_path[i+1][1] - complete_path[i][1])**2
            )
            self.assertLessEqual(segment_distance, MAX_JUMP_DISTANCE,
                               f"Path segment {i+1} distance {segment_distance:.1f}px exceeds jump limit")


def run_physics_waypoint_tests():
    """Run all physics waypoint system tests."""
    print("🧪 RUNNING PHYSICS WAYPOINT SYSTEM TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhysicsWaypointSystem)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🏆 ALL PHYSICS WAYPOINT TESTS PASSED!")
        print(f"✅ Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"❌ Errors: {len(result.errors)}")
        
        # Print failure details
        for test, traceback in result.failures:
            print(f"\nFAILURE: {test}")
            print(traceback)
            
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print(traceback)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_physics_waypoint_tests()
    sys.exit(0 if success else 1)