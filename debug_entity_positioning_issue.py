#!/usr/bin/env python3
"""
Debug script to analyze entity positioning issues in graph visualization.

The image shows entity nodes (black circles) floating incorrectly rather than
being properly positioned relative to level geometry.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import EdgeType

def analyze_entity_positioning():
    """Analyze entity positioning in the doortest map."""
    print("=" * 80)
    print("🔍 ANALYZING ENTITY POSITIONING ISSUE")
    print("=" * 80)
    
    # Initialize environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    
    # Reset environment to ensure entities are loaded
    print("🔄 Resetting environment to load entities...")
    env.reset()
    
    print(f"📍 Map: doortest (default)")
    print(f"🗺️  Map size: {env.level_data.shape[1]}x{env.level_data.shape[0]} tiles")
    print(f"🎮 Total entities: {len(env.entities)}")
    
    # Analyze entity data structure
    print("\n" + "="*60)
    print("🎯 ENTITY DATA ANALYSIS")
    print("="*60)
    
    for i, entity in enumerate(env.entities):
        print(f"\nEntity {i+1}:")
        print(f"  Type: {entity.get('type', 'Unknown')}")
        print(f"  Raw data: {entity}")
        
        # Check for position fields
        pos_fields = ['x', 'y', 'position', 'pos', 'location']
        found_pos = False
        for field in pos_fields:
            if field in entity:
                print(f"  Position field '{field}': {entity[field]}")
                found_pos = True
        
        if not found_pos:
            print("  ⚠️  No position field found!")
    
    # Build graph and analyze entity nodes
    print("\n" + "="*60)
    print("🏗️  GRAPH BUILDING AND ENTITY NODE ANALYSIS")
    print("="*60)
    
    builder = HierarchicalGraphBuilder()
    
    # Get ninja position from environment
    ninja_pos = (env.nplay_headless.sim.ninja.xpos, env.nplay_headless.sim.ninja.ypos)
    print(f"🥷 Ninja position: {ninja_pos}")
    
    # Check what entities are in level_data
    print(f"📋 Level data entities: {getattr(env.level_data, 'entities', 'No entities attribute')}")
    if hasattr(env.level_data, 'entities'):
        print(f"📋 Level data entities count: {len(env.level_data.entities)}")
        for i, entity in enumerate(env.level_data.entities):
            print(f"  Entity {i+1}: {entity}")
    
    # Debug entity extraction process
    print(f"\n🔍 DEBUGGING ENTITY EXTRACTION PROCESS")
    print("="*60)
    
    # Check each entity type extraction method
    try:
        exit_switches = env.nplay_headless.sim.entity_dic.get(4, []) if hasattr(env.nplay_headless.sim, "entity_dic") else []
        print(f"Exit switches found: {len(exit_switches)}")
        for i, es in enumerate(exit_switches):
            print(f"  Exit switch {i+1}: pos=({es.xpos}, {es.ypos}), active={es.active}")
    except Exception as e:
        print(f"Exit switches error: {e}")
    
    try:
        regular_doors = env.nplay_headless.regular_doors()
        print(f"Regular doors found: {len(regular_doors)}")
        for i, rd in enumerate(regular_doors):
            print(f"  Regular door {i+1}: pos=({rd.xpos}, {rd.ypos})")
    except Exception as e:
        print(f"Regular doors error: {e}")
    
    try:
        locked_doors = env.nplay_headless.locked_doors()
        print(f"Locked doors found: {len(locked_doors)}")
        for i, ld in enumerate(locked_doors):
            print(f"  Locked door {i+1}: pos=({ld.xpos}, {ld.ypos})")
    except Exception as e:
        print(f"Locked doors error: {e}")
    
    try:
        trap_doors = env.nplay_headless.trap_doors()
        print(f"Trap doors found: {len(trap_doors)}")
        for i, td in enumerate(trap_doors):
            print(f"  Trap door {i+1}: pos=({td.xpos}, {td.ypos})")
    except Exception as e:
        print(f"Trap doors error: {e}")
    
    try:
        one_ways = env.nplay_headless.sim.entity_dic.get(11, []) if hasattr(env.nplay_headless.sim, "entity_dic") else []
        print(f"One-way platforms found: {len(one_ways)}")
        for i, ow in enumerate(one_ways):
            print(f"  One-way {i+1}: pos=({ow.xpos}, {ow.ypos}), orientation={ow.orientation}")
    except Exception as e:
        print(f"One-way platforms error: {e}")
    
    hierarchical_graph = builder.build_graph(env.level_data, ninja_pos)
    
    # Use the sub-cell graph (finest resolution) for analysis
    graph = hierarchical_graph.sub_cell_graph
    
    print(f"📊 Sub-cell Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Find entity nodes in the graph
    entity_node_indices = []
    for i in range(graph.num_nodes):
        if graph.node_mask[i] == 1:  # Valid node
            if graph.node_types[i] == 1:  # NodeType.ENTITY
                entity_node_indices.append(i)
    
    print(f"🎯 Found {len(entity_node_indices)} entity nodes in graph")
    
    # Analyze entity node positions
    print("\n" + "="*40)
    print("📍 ENTITY NODE POSITIONS")
    print("="*40)
    
    for i, node_idx in enumerate(entity_node_indices):
        print(f"\nEntity Node {i+1} (index {node_idx}):")
        print(f"  Node type: {graph.node_types[node_idx]}")
        print(f"  Features shape: {graph.node_features[node_idx].shape}")
        print(f"  Features (first 10): {graph.node_features[node_idx][:10]}")
        
        # Note: Position information might be encoded in the features
        # We'll need to investigate the feature encoding to extract positions
    
    # Analyze coordinate system conversion
    print("\n" + "="*60)
    print("🔄 COORDINATE SYSTEM ANALYSIS")
    print("="*60)
    
    print(f"Map dimensions (tiles): {env.level_data.shape[1]} x {env.level_data.shape[0]}")
    print(f"Map dimensions (pixels): {env.level_data.shape[1] * 24} x {env.level_data.shape[0] * 24}")
    
    # Check if entities have reasonable positions for the map size
    print("\nEntity position validation:")
    for i, entity in enumerate(env.entities):
        entity_type = entity.get('type', 'Unknown')
        
        # Try to find position in entity data
        x, y = None, None
        if 'x' in entity and 'y' in entity:
            x, y = entity['x'], entity['y']
        elif 'position' in entity:
            pos = entity['position']
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                x, y = pos[0], pos[1]
        
        if x is not None and y is not None:
            print(f"  Entity {i+1} (type {entity_type}): ({x}, {y})")
            
            # Check if coordinates make sense
            map_width_px = env.level_data.shape[1] * 24
            map_height_px = env.level_data.shape[0] * 24
            
            if x < 0 or x >= map_width_px or y < 0 or y >= map_height_px:
                print(f"    ⚠️  OUT OF BOUNDS! Map: {map_width_px}x{map_height_px}")
            else:
                print(f"    ✅ Within bounds")
        else:
            print(f"  Entity {i+1} (type {entity_type}): No position found")
    
    return env, graph, entity_node_indices

if __name__ == "__main__":
    env, graph, entity_node_indices = analyze_entity_positioning()