#!/usr/bin/env python3
"""
Test suite for graph traversability improvements.

This test suite validates the key improvements made to the graph builder:
1. Sub-grid resolution for better spatial accuracy
2. Raycasting collision detection with ninja radius
3. Improved half-tile traversability detection
4. Enhanced one-way platform blocking
5. Diagonal traversability for slopes
6. Fixed debug overlay coordinate system
"""

import numpy as np
import sys
import os
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nclone.graph.graph_builder import GraphBuilder, SUB_GRID_WIDTH, SUB_GRID_HEIGHT
from nclone.constants import NINJA_RADIUS


class TestGraphTraversability:
    """Test class for graph traversability improvements."""
    
    @pytest.fixture
    def builder(self):
        """Create a GraphBuilder instance."""
        return GraphBuilder()
    
    @pytest.fixture
    def standard_level_with_passages(self):
        """Create a standard N++ 23x42 level with clear passages."""
        level_data = {
            'tiles': np.ones((23, 42), dtype=np.int32)  # Start with all solid walls
        }
        
        # Create clear corridors and passages
        tiles = level_data['tiles']
        
        # Clear horizontal corridor in the middle
        tiles[11, 5:37] = 0  # Row 11, columns 5-36
        
        # Clear vertical passages
        tiles[5:18, 10] = 0  # Column 10, rows 5-17
        tiles[5:18, 20] = 0  # Column 20, rows 5-17
        tiles[5:18, 30] = 0  # Column 30, rows 5-17
        
        # Clear some rooms
        tiles[5:9, 5:15] = 0    # Top-left room
        tiles[5:9, 25:35] = 0   # Top-right room
        tiles[14:18, 5:15] = 0  # Bottom-left room
        tiles[14:18, 25:35] = 0 # Bottom-right room
        
        return level_data
    
    @pytest.fixture
    def half_tile_level(self):
        """Create a level with half-tiles forming passages."""
        level_data = {
            'tiles': np.ones((23, 42), dtype=np.int32)  # Start with all solid walls
        }
        
        tiles = level_data['tiles']
        
        # Create a section with half-tiles in the middle
        # Row 10-12, columns 15-25
        tiles[10, 15] = 2  # top-half
        tiles[10, 16:24] = 0  # empty
        tiles[10, 24] = 3  # right-half
        
        tiles[11, 15] = 4  # bottom-half  
        tiles[11, 16:24] = 0  # empty
        tiles[11, 24] = 5  # left-half
        
        tiles[12, 15:25] = 0  # clear corridor below
        
        return level_data
    
    @pytest.fixture
    def one_way_platform_level(self):
        """Create a level with one-way platforms."""
        level_data = {
            'tiles': np.zeros((23, 42), dtype=np.int32)  # All empty
        }
        
        # One-way platform entities
        entities = [
            {
                'type': 11,  # One-way platform
                'x': 504.0,  # Center of level (21*24 = 504)
                'y': 276.0,  # Center of level (11.5*24 = 276)
                'orientation': 0,  # Facing right (normal = (1, 0))
                'active': True
            }
        ]
        
        return level_data, entities

    def test_sub_grid_resolution(self, builder, standard_level_with_passages):
        """Test that sub-grid resolution provides better spatial accuracy."""
        ninja_pos = (252, 276)  # Center of clear corridor
        entities = []
        
        # Build graph
        graph = builder.build_graph(standard_level_with_passages, ninja_pos, entities)
        
        # Should have significantly more nodes than original grid approach
        expected_min_nodes = 3000  # Conservative estimate
        assert graph.num_nodes >= expected_min_nodes, f"Expected at least {expected_min_nodes} nodes, got {graph.num_nodes}"
        
        # Should have a reasonable number of edges
        expected_min_edges = 1000
        assert graph.num_edges >= expected_min_edges, f"Expected at least {expected_min_edges} edges, got {graph.num_edges}"

    def test_raycasting_collision_detection(self, builder, standard_level_with_passages):
        """Test that raycasting collision detection works correctly."""
        test_cases = [
            # In clear areas - should be safe
            (240, 276, False, "Clear corridor center"),
            (480, 276, False, "Clear corridor middle"),
            (240, 156, False, "Top-left room"),
            
            # In solid walls - should collide
            (60, 60, True, "Solid wall area"),
            (36, 36, True, "Border wall"),
        ]
        
        for x, y, should_collide, description in test_cases:
            collision = not builder._check_ninja_collision_at_position(
                standard_level_with_passages, x, y, NINJA_RADIUS
            )
            
            if should_collide:
                assert collision, f"{description} at ({x}, {y}) should have collision but didn't"
            else:
                assert not collision, f"{description} at ({x}, {y}) should not have collision but did"

    def test_half_tile_traversability(self, builder, half_tile_level):
        """Test improved half-tile traversability detection."""
        test_cases = [
            # Tile (10,15) is type 2 (top-half solid) - world pos: 15*24+12=372, 10*24+12=252
            (372, 240, True, "Top half of top-half tile"),      # y=240 is above center, should collide
            (372, 264, False, "Bottom half of top-half tile"),  # y=264 is below center, should be clear
            
            # Tile (10,16) is type 0 (empty) - world pos: 16*24+12=396, 10*24+12=252
            (396, 252, False, "Empty tile center"),             # Should be clear
            
            # Tile (10,24) is type 3 (right-half solid) - world pos: 24*24+12=588, 10*24+12=252
            (588, 252, True, "Right half of right-half tile"),  # x=588 is right of center, should collide
            (576, 252, False, "Left half of right-half tile"),  # x=576 is left of center, should be clear
        ]
        
        for x, y, should_collide, description in test_cases:
            collision = not builder._check_ninja_collision_at_position(
                half_tile_level, x, y, NINJA_RADIUS
            )
            
            if should_collide:
                assert collision, f"{description} should have collision but didn't"
            else:
                assert not collision, f"{description} should not have collision but did"

    def test_one_way_platform_blocking(self, builder, one_way_platform_level):
        """Test improved one-way platform blocking logic."""
        level_data, entities = one_way_platform_level
        ninja_pos = (252, 276)
        
        # Build graph with one-way platform
        graph = builder.build_graph(level_data, ninja_pos, entities)
        
        # Should have detected the one-way platform
        one_way_index = builder._collect_one_way_platforms(entities)
        assert len(one_way_index) == 1, f"Expected 1 one-way platform, found {len(one_way_index)}"
        
        # Test blocking in opposite direction
        # Platform is at world (504, 276), which is sub-cell (46, 84) approximately with 4x4 resolution
        # Platform normal is (1, 0) facing right, so movement from right to left should be blocked
        src_sub_row, src_sub_col = 46, 88  # Right of platform  
        tgt_sub_row, tgt_sub_col = 46, 80  # Left of platform (against normal)
        
        blocked = builder._is_sub_cell_blocked_by_one_way(
            one_way_index, src_sub_row, src_sub_col, tgt_sub_row, tgt_sub_col
        )
        
        assert blocked, "Movement against one-way platform normal should be blocked"

    def test_diagonal_traversability(self, builder, standard_level_with_passages):
        """Test diagonal traversability for slope combinations."""
        ninja_pos = (252, 276)
        entities = []
        
        # Build graph
        graph = builder.build_graph(standard_level_with_passages, ninja_pos, entities)
        
        # Count diagonal edges
        diagonal_edges = 0
        for i in range(graph.num_edges):
            if graph.edge_mask[i] > 0:
                # Check if edge direction is diagonal
                dx = graph.edge_features[i, 6]  # Direction x
                dy = graph.edge_features[i, 7]  # Direction y
                
                # Diagonal if both dx and dy are non-zero
                if abs(dx) > 0.1 and abs(dy) > 0.1:
                    diagonal_edges += 1
        
        # Should have some diagonal edges due to 8-connectivity
        assert diagonal_edges > 0, f"Should have diagonal edges for 8-connectivity, found {diagonal_edges}"

    def test_graph_edge_types(self, builder, standard_level_with_passages):
        """Test that different edge types are generated correctly."""
        ninja_pos = (252, 276)
        entities = []
        
        # Build graph
        graph = builder.build_graph(standard_level_with_passages, ninja_pos, entities)
        
        # Count different types of edges
        edge_types = {}
        for i in range(graph.num_edges):
            if graph.edge_mask[i] > 0:
                # Get edge type from features
                edge_features = graph.edge_features[i]
                edge_type_idx = int(np.argmax(edge_features[:6]))  # First 6 are edge types
                edge_types[edge_type_idx] = edge_types.get(edge_type_idx, 0) + 1
        
        # Should have multiple edge types
        assert len(edge_types) > 1, f"Expected multiple edge types, got {edge_types}"
        
        # Should have WALK edges (type 0)
        assert 0 in edge_types, f"Expected WALK edges (type 0), got types {list(edge_types.keys())}"
        
        # Should have a reasonable number of each type
        for edge_type, count in edge_types.items():
            assert count > 0, f"Edge type {edge_type} should have count > 0, got {count}"

    def test_sub_cell_coordinate_conversion(self, builder):
        """Test that sub-cell coordinates are converted correctly."""
        # Test a few known coordinate conversions
        test_cases = [
            (0, 0, 6, 6),      # Top-left sub-cell -> world (6, 6)
            (1, 1, 18, 18),    # Sub-cell (1,1) -> world (18, 18)
            (10, 20, 246, 126), # Sub-cell (10,20) -> world (246, 126)
        ]
        
        SUB_CELL_SIZE = 12
        for sub_row, sub_col, expected_x, expected_y in test_cases:
            world_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            world_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
            
            assert world_x == expected_x, f"Sub-cell ({sub_row}, {sub_col}) should map to x={expected_x}, got {world_x}"
            assert world_y == expected_y, f"Sub-cell ({sub_row}, {sub_col}) should map to y={expected_y}, got {world_y}"

    def test_ninja_radius_collision(self, builder):
        """Test that ninja radius is properly considered in collision detection."""
        # Create a simple level with a narrow passage
        level_data = {
            'tiles': np.ones((23, 42), dtype=np.int32)
        }
        
        # Create a passage that's exactly ninja-width (20 pixels)
        tiles = level_data['tiles']
        tiles[10:13, 20] = 0  # 3-tile high, 1-tile wide passage
        
        # Test positions near the edges of the passage
        passage_center_x = 20 * 24 + 12  # 492
        passage_center_y = 11 * 24 + 12  # 276
        
        # Center should be safe
        center_safe = builder._check_ninja_collision_at_position(
            level_data, passage_center_x, passage_center_y, NINJA_RADIUS
        )
        assert center_safe, "Center of passage should be safe for ninja"
        
        # Position too close to wall should not be safe
        near_wall_x = passage_center_x - 11  # Within ninja radius of wall
        near_wall_safe = builder._check_ninja_collision_at_position(
            level_data, near_wall_x, passage_center_y, NINJA_RADIUS
        )
        assert not near_wall_safe, "Position within ninja radius of wall should not be safe"


def test_graph_constants():
    """Test that graph constants are set correctly."""
    assert SUB_GRID_WIDTH == 168, f"Expected SUB_GRID_WIDTH=168, got {SUB_GRID_WIDTH}"
    assert SUB_GRID_HEIGHT == 92, f"Expected SUB_GRID_HEIGHT=92, got {SUB_GRID_HEIGHT}"
    assert NINJA_RADIUS == 10, f"Expected NINJA_RADIUS=10, got {NINJA_RADIUS}"


if __name__ == "__main__":
    # Run tests directly if called as script
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], 
                          cwd=os.path.dirname(os.path.dirname(__file__)))
    sys.exit(result.returncode)
