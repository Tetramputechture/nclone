#!/usr/bin/env python3

"""
Debug functional edge creation in the graph system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import EdgeType
from nclone.constants.entity_types import EntityType

def debug_functional_edges():
    """Debug functional edge creation."""
    print("=" * 80)
    print("🔍 DEBUGGING FUNCTIONAL EDGE CREATION")
    print("=" * 80)
    
    # Initialize environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    env.reset()
    
    print(f"📍 Map: doortest")
    print(f"🎮 Total entities: {len(env.entities)}")
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    ninja_pos = (env.nplay_headless.sim.ninja.xpos, env.nplay_headless.sim.ninja.ypos)
    hierarchical_graph = builder.build_graph(env.level_data, ninja_pos)
    graph = hierarchical_graph.sub_cell_graph
    
    print(f"📊 Sub-cell Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Find entity nodes
    entity_nodes = []
    for i in range(graph.num_nodes):
        if graph.node_types[i] == 1:  # NodeType.ENTITY
            entity_nodes.append(i)
    
    print(f"🎯 Found {len(entity_nodes)} entity nodes")
    
    # Analyze entity node details
    print("\n" + "="*60)
    print("🏷️  ENTITY NODE ANALYSIS")
    print("="*60)
    
    for i, node_idx in enumerate(entity_nodes):
        features = graph.node_features[node_idx]
        x, y = features[0], features[1]
        print(f"Entity Node {i+1} (index {node_idx}): pos=({x}, {y})")
        
        # Try to match this to original entities
        for j, entity in enumerate(env.level_data.entities):
            entity_x = entity.get('x', 0)
            entity_y = entity.get('y', 0)
            door_x = entity.get('door_x', entity_x)
            door_y = entity.get('door_y', entity_y)
            
            if abs(x - entity_x) < 1 and abs(y - entity_y) < 1:
                print(f"  → Matches entity {j+1} (switch): type={entity.get('type')}, id={entity.get('entity_id')}")
                print(f"    is_door_part: {entity.get('is_door_part', 'not set')}")
            elif abs(x - door_x) < 1 and abs(y - door_y) < 1:
                print(f"  → Matches entity {j+1} (door): type={entity.get('type')}, id={entity.get('entity_id')}")
                print(f"    is_door_part: should be True")
    
    # Analyze functional edges
    print("\n" + "="*60)
    print("🔗 FUNCTIONAL EDGE ANALYSIS")
    print("="*60)
    
    functional_edges = []
    for i in range(graph.num_edges):
        if graph.edge_types[i] == EdgeType.FUNCTIONAL:
            src_idx = graph.edge_index[0, i]
            dst_idx = graph.edge_index[1, i]
            functional_edges.append((src_idx, dst_idx, i))
    
    print(f"Found {len(functional_edges)} functional edges")
    
    for i, (src_idx, dst_idx, edge_idx) in enumerate(functional_edges):
        src_pos = graph.node_features[src_idx][:2]
        dst_pos = graph.node_features[dst_idx][:2]
        print(f"Functional Edge {i+1}: {src_idx} → {dst_idx}")
        print(f"  Source: ({src_pos[0]}, {src_pos[1]})")
        print(f"  Target: ({dst_pos[0]}, {dst_pos[1]})")
        print(f"  Edge features: {graph.edge_features[edge_idx][:10]}")
    
    # Check why functional edges might be missing
    print("\n" + "="*60)
    print("🔍 FUNCTIONAL EDGE MATCHING ANALYSIS")
    print("="*60)
    
    # Look for locked door entities specifically
    locked_door_entities = [e for e in env.level_data.entities if e.get('type') == EntityType.LOCKED_DOOR]
    print(f"Found {len(locked_door_entities)} locked door entities:")
    
    for i, entity in enumerate(locked_door_entities):
        print(f"Locked Door {i+1}:")
        print(f"  Type: {entity.get('type')}")
        print(f"  Switch pos: ({entity.get('x')}, {entity.get('y')})")
        print(f"  Door pos: ({entity.get('door_x')}, {entity.get('door_y')})")
        print(f"  Entity ID: {entity.get('entity_id')}")
        print(f"  is_door_part: {entity.get('is_door_part', 'not set')}")
        
        # Find corresponding entity nodes
        switch_nodes = []
        door_nodes = []
        
        for node_idx in entity_nodes:
            node_pos = graph.node_features[node_idx][:2]
            switch_x, switch_y = entity.get('x', 0), entity.get('y', 0)
            door_x, door_y = entity.get('door_x', 0), entity.get('door_y', 0)
            
            if abs(node_pos[0] - switch_x) < 1 and abs(node_pos[1] - switch_y) < 1:
                switch_nodes.append(node_idx)
            elif abs(node_pos[0] - door_x) < 1 and abs(node_pos[1] - door_y) < 1:
                door_nodes.append(node_idx)
        
        print(f"  Switch nodes found: {switch_nodes}")
        print(f"  Door nodes found: {door_nodes}")
        
        # Check if functional edge should exist
        if switch_nodes and door_nodes:
            for switch_node in switch_nodes:
                for door_node in door_nodes:
                    # Check if edge exists
                    edge_exists = False
                    for edge_i in range(graph.num_edges):
                        if (graph.edge_index[0, edge_i] == switch_node and 
                            graph.edge_index[1, edge_i] == door_node and
                            graph.edge_types[edge_i] == EdgeType.FUNCTIONAL):
                            edge_exists = True
                            break
                    
                    print(f"  Functional edge {switch_node} → {door_node}: {'✅ EXISTS' if edge_exists else '❌ MISSING'}")

if __name__ == "__main__":
    debug_functional_edges()