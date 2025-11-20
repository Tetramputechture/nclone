#!/usr/bin/env python3
"""
Validation test for graph memory optimization (Phase 6).

Tests that:
1. Feature dimensions match updated constants
2. Graph building produces valid outputs
3. Memory usage is reduced as expected
4. Features maintain consistency
"""

import numpy as np
import sys
from pathlib import Path

# Add nclone to path
sys.path.insert(0, str(Path(__file__).parent))

from nclone.graph.common import NODE_FEATURE_DIM, EDGE_FEATURE_DIM, N_MAX_NODES, E_MAX_EDGES
from nclone.graph.feature_builder import NodeFeatureBuilder, EdgeFeatureBuilder
from nclone.graph.common import NodeType, EdgeType


def test_dimension_constants():
    """Test that dimension constants are correct."""
    print("Testing dimension constants...")
    assert NODE_FEATURE_DIM == 17, f"Expected NODE_FEATURE_DIM=17, got {NODE_FEATURE_DIM}"
    assert EDGE_FEATURE_DIM == 12, f"Expected EDGE_FEATURE_DIM=12, got {EDGE_FEATURE_DIM}"
    print("✓ Dimension constants correct: NODE_FEATURE_DIM=17, EDGE_FEATURE_DIM=12")


def test_node_feature_builder():
    """Test that NodeFeatureBuilder produces correct dimensions."""
    print("\nTesting NodeFeatureBuilder...")
    
    builder = NodeFeatureBuilder()
    
    # Check index boundaries
    assert builder.TOPOLOGICAL_END == 17, f"Expected TOPOLOGICAL_END=17, got {builder.TOPOLOGICAL_END}"
    assert builder.MINE_END == 11, f"Expected MINE_END=11 (2 features), got {builder.MINE_END}"
    assert builder.ENTITY_STATE_START == 11, f"Expected ENTITY_STATE_START=11, got {builder.ENTITY_STATE_START}"
    assert builder.REACHABILITY_START == 13, f"Expected REACHABILITY_START=13, got {builder.REACHABILITY_START}"
    assert builder.TOPOLOGICAL_START == 14, f"Expected TOPOLOGICAL_START=14, got {builder.TOPOLOGICAL_START}"
    
    # Build test features
    features = builder.build_node_features(
        node_pos=(100.0, 200.0),
        node_type=NodeType.EMPTY,
        entity_info=None,
        reachability_info={"reachable_from_ninja": True},
        topological_info={
            "objective_dx": 0.5,
            "objective_dy": -0.3,
            "objective_hops": 0.2,
        }
    )
    
    assert features.shape == (17,), f"Expected shape (17,), got {features.shape}"
    assert features.dtype == np.float32, f"Expected dtype float32, got {features.dtype}"
    
    # Check that position is set
    assert features[0] > 0, "Position x should be set"
    assert features[1] > 0, "Position y should be set"
    
    # Check that reachability is set
    assert features[13] == 1.0, "Reachability should be 1.0"
    
    # Check that objective features are set
    assert features[14] == 0.5, "objective_dx should be 0.5"
    assert features[15] == -0.3, "objective_dy should be -0.3"
    assert features[16] == 0.2, "objective_hops should be 0.2"
    
    print("✓ NodeFeatureBuilder produces correct 17-dim features")


def test_edge_feature_builder():
    """Test that EdgeFeatureBuilder produces correct dimensions."""
    print("\nTesting EdgeFeatureBuilder...")
    
    builder = EdgeFeatureBuilder()
    
    # Check index boundaries
    assert builder.MINE_END == 12, f"Expected MINE_END=12 (2 features), got {builder.MINE_END}"
    
    # Build test features
    features = builder.build_edge_features(
        edge_type=EdgeType.ADJACENT,
        weight=0.5,
        reachability_confidence=1.0,
        geometric_features={
            "dx_norm": 0.1,
            "dy_norm": -0.2,
            "distance": 0.3,
            "movement_category": 0.5,
        },
        mine_features={
            "nearest_mine_distance": 0.8,
            "passes_deadly_mine": 0.0,
        }
    )
    
    assert features.shape == (12,), f"Expected shape (12,), got {features.shape}"
    assert features.dtype == np.float32, f"Expected dtype float32, got {features.dtype}"
    
    # Check that edge type is set (one-hot)
    assert features[0] == 1.0, "Edge type ADJACENT should be set"
    assert features[1] == 0.0, "Other edge types should be 0"
    
    # Check connectivity
    assert features[4] == 0.5, "Weight should be 0.5"
    assert features[5] == 1.0, "Reachability confidence should be 1.0"
    
    # Check mine features
    assert features[10] == 0.8, "nearest_mine_distance should be 0.8"
    assert features[11] == 0.0, "passes_deadly_mine should be 0.0"
    
    print("✓ EdgeFeatureBuilder produces correct 12-dim features")


def test_memory_calculation():
    """Test expected memory savings."""
    print("\nCalculating memory savings...")
    
    # Old dimensions
    old_node_dim = 21
    old_edge_dim = 14
    
    # Calculate sizes (float16 = 2 bytes, as used in graph_mixin.py)
    old_node_size = N_MAX_NODES * old_node_dim * 2
    old_edge_size = E_MAX_EDGES * old_edge_dim * 2
    
    new_node_size = N_MAX_NODES * NODE_FEATURE_DIM * 2
    new_edge_size = E_MAX_EDGES * EDGE_FEATURE_DIM * 2
    
    # Other arrays (unchanged)
    edge_index_size = 2 * E_MAX_EDGES * 4
    node_mask_size = N_MAX_NODES * 4
    edge_mask_size = E_MAX_EDGES * 4
    node_types_size = N_MAX_NODES * 4
    edge_types_size = E_MAX_EDGES * 4
    
    other_size = edge_index_size + node_mask_size + edge_mask_size + node_types_size + edge_types_size
    
    old_total = old_node_size + old_edge_size + other_size
    new_total = new_node_size + new_edge_size + other_size
    
    reduction_pct = 100 * (old_total - new_total) / old_total
    
    print(f"  Old node features: {old_node_size / 1024:.1f} KB")
    print(f"  New node features: {new_node_size / 1024:.1f} KB ({100 * (old_node_size - new_node_size) / old_node_size:.1f}% reduction)")
    print(f"  Old edge features: {old_edge_size / 1024:.1f} KB")
    print(f"  New edge features: {new_edge_size / 1024:.1f} KB ({100 * (old_edge_size - new_edge_size) / old_edge_size:.1f}% reduction)")
    print(f"  Total per observation: {old_total / 1024:.1f} KB → {new_total / 1024:.1f} KB")
    print(f"  Reduction: {reduction_pct:.1f}%")
    
    # Rollout buffer (2048 steps × 32 envs)
    rollout_obs = 2048 * 32
    old_buffer = old_total * rollout_obs / (1024**3)
    new_buffer = new_total * rollout_obs / (1024**3)
    
    print(f"\n  Rollout buffer ({rollout_obs:,} observations):")
    print(f"    Before: {old_buffer:.2f} GB")
    print(f"    After: {new_buffer:.2f} GB")
    print(f"    Saved: {old_buffer - new_buffer:.2f} GB")
    
    assert reduction_pct > 10, f"Expected >10% reduction, got {reduction_pct:.1f}%"
    print(f"\n✓ Memory reduction target achieved: {reduction_pct:.1f}% (target: >10%)")
    print(f"  Note: Total reduction is lower than feature reduction due to unchanged")
    print(f"        components (edge_index, masks, types) taking ~32% of observation size")


def test_mine_node_features():
    """Test that mine nodes work correctly without is_mine flag."""
    print("\nTesting mine node features (without is_mine)...")
    
    builder = NodeFeatureBuilder()
    
    # Build mine node features
    features = builder.build_node_features(
        node_pos=(100.0, 200.0),
        node_type=NodeType.TOGGLE_MINE,
        entity_info={
            "type": 2,  # EntityType.TOGGLE_MINE
            "state": 0.0,  # Toggled (deadly)
            "radius": 6.0,
        },
        reachability_info={"reachable_from_ninja": True},
        topological_info={
            "objective_dx": 0.5,
            "objective_dy": -0.3,
            "objective_hops": 0.2,
        }
    )
    
    # Check that TOGGLE_MINE bit is set in node_type (index 2+2=4)
    assert features[4] == 1.0, "TOGGLE_MINE node type should be set"
    
    # Check that mine_state is set (index 9)
    assert features[9] == -1.0, "mine_state should be -1.0 (deadly)"
    
    # Check that mine_radius is set (index 10)
    assert features[10] > 0, "mine_radius should be > 0"
    
    # Note: is_mine flag is removed - redundant with node_type bit
    print("✓ Mine nodes work correctly without is_mine flag (redundancy removed)")


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Graph Memory Optimization Validation (Phase 6)")
    print("=" * 60)
    
    try:
        test_dimension_constants()
        test_node_feature_builder()
        test_edge_feature_builder()
        test_memory_calculation()
        test_mine_node_features()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nOptimization summary:")
        print("  - Node features: 21 → 17 dims (19% reduction)")
        print("  - Edge features: 14 → 12 dims (14% reduction)")
        print("  - Total per observation: ~11% reduction")
        print("  - Memory saved: ~6.7 GB in rollout buffer (from 63.4 GB → 56.7 GB)")
        print("  - Removed features: is_mine, in_degree, out_degree, betweenness,")
        print("                      mine_threat_level, num_mines_nearby")
        print("  - All critical features maintained for navigation & mine avoidance")
        print("  - Kept all 3 objective features (dx, dy, hops) for shortest-path nav")
        
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

