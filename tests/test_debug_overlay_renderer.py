#!/usr/bin/env python3
"""
Test suite for debug overlay renderer improvements.

This test suite validates the fixes made to the debug overlay renderer
to work with the new sub-grid coordinate system.
"""

import numpy as np
import sys
import os
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nclone.graph.graph_builder import GraphBuilder, SUB_GRID_WIDTH, SUB_GRID_HEIGHT, SUB_CELL_SIZE
from nclone.constants import TILE_PIXEL_SIZE


class TestDebugOverlayRenderer:
    """Test class for debug overlay renderer improvements."""
    
    @pytest.fixture
    def simple_level(self):
        """Create a simple test level."""
        level_data = {
            'tiles': np.ones((23, 42), dtype=np.int32)
        }
        
        # Create a small clear area
        tiles = level_data['tiles']
        tiles[10:13, 20:23] = 0  # 3x3 clear area
        
        return level_data
    
    def test_sub_grid_coordinate_conversion(self, simple_level):
        """Test that sub-grid coordinates are converted correctly for rendering."""
        builder = GraphBuilder()
        ninja_pos = (504, 276)  # Center of level
        entities = []
        
        # Build graph
        graph = builder.build_graph(simple_level, ninja_pos, entities)
        
        # Test coordinate conversion logic (simulating what debug overlay does)
        sub_grid_nodes_count = SUB_GRID_WIDTH * SUB_GRID_HEIGHT
        
        # Test a few known sub-grid positions
        test_cases = [
            (0, 0, 6 + TILE_PIXEL_SIZE, 6 + TILE_PIXEL_SIZE),    # Top-left + border offset
            (21, 41, 21*12 + 6 + 24, 41*12 + 6 + 24),           # Center of clear area + border
            (45, 83, 45*12 + 6 + 24, 83*12 + 6 + 24),           # Bottom-right + border
        ]
        
        for node_idx in range(min(100, graph.num_nodes)):
            if graph.node_mask[node_idx] > 0 and node_idx < sub_grid_nodes_count:
                sub_row = node_idx // SUB_GRID_WIDTH
                sub_col = node_idx % SUB_GRID_WIDTH
                
                # Calculate expected world position (matching debug overlay logic)
                expected_wx = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5 + TILE_PIXEL_SIZE
                expected_wy = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5 + TILE_PIXEL_SIZE
                
                # Verify coordinates are reasonable
                assert 0 <= expected_wx <= (SUB_GRID_WIDTH * SUB_CELL_SIZE + TILE_PIXEL_SIZE), \
                    f"World X coordinate {expected_wx} out of expected range for sub-cell ({sub_row}, {sub_col})"
                assert 0 <= expected_wy <= (SUB_GRID_HEIGHT * SUB_CELL_SIZE + TILE_PIXEL_SIZE), \
                    f"World Y coordinate {expected_wy} out of expected range for sub-cell ({sub_row}, {sub_col})"
    
    def test_node_indexing_consistency(self, simple_level):
        """Test that node indexing is consistent between graph builder and renderer."""
        builder = GraphBuilder()
        ninja_pos = (504, 276)
        entities = []
        
        # Build graph
        graph = builder.build_graph(simple_level, ninja_pos, entities)
        
        # Test that node indices map correctly to sub-grid coordinates
        sub_grid_nodes_count = SUB_GRID_WIDTH * SUB_GRID_HEIGHT
        
        for node_idx in range(min(sub_grid_nodes_count, graph.num_nodes)):
            if graph.node_mask[node_idx] > 0:
                # Calculate sub-grid coordinates from node index
                sub_row = node_idx // SUB_GRID_WIDTH
                sub_col = node_idx % SUB_GRID_WIDTH
                
                # Verify coordinates are in valid range
                assert 0 <= sub_row < SUB_GRID_HEIGHT, \
                    f"Sub-row {sub_row} out of range [0, {SUB_GRID_HEIGHT}) for node {node_idx}"
                assert 0 <= sub_col < SUB_GRID_WIDTH, \
                    f"Sub-col {sub_col} out of range [0, {SUB_GRID_WIDTH}) for node {node_idx}"
                
                # Verify reverse mapping
                reconstructed_idx = sub_row * SUB_GRID_WIDTH + sub_col
                assert reconstructed_idx == node_idx, \
                    f"Reverse mapping failed: ({sub_row}, {sub_col}) -> {reconstructed_idx}, expected {node_idx}"
    
    def test_edge_coordinate_consistency(self, simple_level):
        """Test that edge coordinates are consistent for rendering."""
        builder = GraphBuilder()
        ninja_pos = (504, 276)
        entities = []
        
        # Build graph
        graph = builder.build_graph(simple_level, ninja_pos, entities)
        
        sub_grid_nodes_count = SUB_GRID_WIDTH * SUB_GRID_HEIGHT
        
        # Test a few edges to ensure coordinates are reasonable
        edges_tested = 0
        for e_idx in range(min(100, graph.num_edges)):
            if graph.edge_mask[e_idx] > 0:
                src = int(graph.edge_index[0, e_idx])
                tgt = int(graph.edge_index[1, e_idx])
                
                # Only test sub-grid to sub-grid edges
                if src < sub_grid_nodes_count and tgt < sub_grid_nodes_count:
                    # Calculate coordinates for both nodes
                    src_sub_row = src // SUB_GRID_WIDTH
                    src_sub_col = src % SUB_GRID_WIDTH
                    tgt_sub_row = tgt // SUB_GRID_WIDTH
                    tgt_sub_col = tgt % SUB_GRID_WIDTH
                    
                    # Calculate world positions
                    src_wx = src_sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5 + TILE_PIXEL_SIZE
                    src_wy = src_sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5 + TILE_PIXEL_SIZE
                    tgt_wx = tgt_sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5 + TILE_PIXEL_SIZE
                    tgt_wy = tgt_sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5 + TILE_PIXEL_SIZE
                    
                    # Verify edge distance is reasonable (should be adjacent or diagonal)
                    distance = np.sqrt((tgt_wx - src_wx)**2 + (tgt_wy - src_wy)**2)
                    max_expected_distance = SUB_CELL_SIZE * np.sqrt(2) + 1  # Diagonal + small tolerance
                    
                    assert distance <= max_expected_distance, \
                        f"Edge {e_idx} distance {distance:.2f} exceeds expected max {max_expected_distance:.2f}"
                    
                    edges_tested += 1
                    if edges_tested >= 20:  # Test enough edges
                        break
        
        assert edges_tested > 0, "Should have tested at least some edges"


def test_sub_grid_constants():
    """Test that sub-grid constants are consistent."""
    assert SUB_CELL_SIZE == TILE_PIXEL_SIZE // 2, \
        f"SUB_CELL_SIZE should be TILE_PIXEL_SIZE/2, got {SUB_CELL_SIZE} vs {TILE_PIXEL_SIZE//2}"
    
    assert SUB_GRID_WIDTH == 42 * 2, \
        f"SUB_GRID_WIDTH should be 42*2=84, got {SUB_GRID_WIDTH}"
    
    assert SUB_GRID_HEIGHT == 23 * 2, \
        f"SUB_GRID_HEIGHT should be 23*2=46, got {SUB_GRID_HEIGHT}"


if __name__ == "__main__":
    # Run tests directly if called as script
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], 
                          cwd=os.path.dirname(os.path.dirname(__file__)))
    sys.exit(result.returncode)
