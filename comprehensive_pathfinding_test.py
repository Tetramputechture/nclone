#!/usr/bin/env python3
"""
Comprehensive pathfinding test with browser visualization.
"""

import sys
import os
import math
import json

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder

def comprehensive_pathfinding_test():
    """Comprehensive test with detailed analysis and browser visualization."""
    print("=" * 80)
    print("🎯 COMPREHENSIVE PATHFINDING TEST")
    print("=" * 80)
    
    # Load environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    env.reset()
    
    ninja_pos = env.nplay_headless.ninja_position()
    print(f"✅ Ninja position: {ninja_pos}")
    
    # Find the leftmost switch
    switches = []
    for entity in env.entities:
        if isinstance(entity, dict):
            entity_type = entity.get('entity_type', entity.get('type', 'unknown'))
            x = entity.get('x', 0)
            y = entity.get('y', 0)
        else:
            entity_type = getattr(entity, 'entity_type', getattr(entity, 'type', 'unknown'))
            x = getattr(entity, 'x', 0)
            y = getattr(entity, 'y', 0)
            
        if entity_type == 4:  # Switch
            switches.append((x, y))
    
    if switches:
        leftmost_switch = min(switches, key=lambda s: s[0])
        print(f"🎯 Target switch: {leftmost_switch}")
    else:
        print("❌ No switches found!")
        return
    
    # Build graph
    print(f"\n🔧 Building graph...")
    try:
        builder = HierarchicalGraphBuilder()
        graph_data = builder.build_graph(env.level_data, ninja_pos)
        
        # Get the finest resolution graph
        graph = graph_data.sub_cell_graph
        print(f"✅ Graph built successfully!")
        print(f"   Nodes: {len(graph.nodes)}")
        print(f"   Edges: {len(graph.edges)}")
        
    except Exception as e:
        print(f"❌ Graph building failed: {e}")
        return
    
    # Analyze edge types
    edge_types = {}
    for edge in graph.edges:
        edge_type = edge.type.name
        if edge_type not in edge_types:
            edge_types[edge_type] = 0
        edge_types[edge_type] += 1
    
    print(f"\n📊 EDGE TYPE DISTRIBUTION:")
    total_edges = sum(edge_types.values())
    for edge_type, count in sorted(edge_types.items()):
        percentage = (count / total_edges) * 100
        print(f"   {edge_type}: {count} edges ({percentage:.1f}%)")
    
    # Calculate movement diversity
    movement_types = [t for t in edge_types.keys() if t in ['WALK', 'JUMP', 'FALL']]
    movement_diversity = len(movement_types) / 3.0
    print(f"\n🎲 Movement diversity: {movement_diversity:.3f} ({len(movement_types)}/3 types)")
    
    if movement_diversity < 0.67:  # Less than 2/3 types
        print("   ⚠️  Low movement diversity - need more JUMP/FALL edges")
    else:
        print("   ✅ Good movement diversity")
    
    # Find ninja and target nodes
    ninja_node = None
    target_node = None
    
    min_ninja_dist = float('inf')
    min_target_dist = float('inf')
    
    for node in graph.nodes:
        # Check ninja distance
        ninja_dist = math.sqrt((node.x - ninja_pos[0])**2 + (node.y - ninja_pos[1])**2)
        if ninja_dist < min_ninja_dist:
            min_ninja_dist = ninja_dist
            ninja_node = node
        
        # Check target distance
        target_dist = math.sqrt((node.x - leftmost_switch[0])**2 + (node.y - leftmost_switch[1])**2)
        if target_dist < min_target_dist:
            min_target_dist = target_dist
            target_node = node
    
    if not ninja_node or not target_node:
        print("❌ Could not find ninja or target nodes!")
        return
    
    print(f"\n🗺️  NODE ANALYSIS:")
    print(f"   Ninja node: {ninja_node.id} at ({ninja_node.x:.1f}, {ninja_node.y:.1f}) - {min_ninja_dist:.1f}px from ninja")
    print(f"   Target node: {target_node.id} at ({target_node.x:.1f}, {target_node.y:.1f}) - {min_target_dist:.1f}px from switch")
    
    # Analyze connectivity around ninja
    ninja_edges = []
    for edge in graph.edges:
        if edge.source == ninja_node.id:
            ninja_edges.append(edge)
    
    print(f"\n🔗 NINJA CONNECTIVITY:")
    print(f"   Outgoing edges: {len(ninja_edges)}")
    
    ninja_edge_types = {}
    for edge in ninja_edges:
        edge_type = edge.type.name
        if edge_type not in ninja_edge_types:
            ninja_edge_types[edge_type] = 0
        ninja_edge_types[edge_type] += 1
    
    for edge_type, count in sorted(ninja_edge_types.items()):
        print(f"   {edge_type}: {count} edges")
    
    # Create visualization data
    print(f"\n🎨 CREATING VISUALIZATION DATA...")
    
    # Create level visualization data
    level_data = {
        'width': env.level_data.width,
        'height': env.level_data.height,
        'tiles': []
    }
    
    for y in range(env.level_data.height):
        row = []
        for x in range(env.level_data.width):
            tile_value = env.level_data.get_tile(y, x)
            row.append(tile_value)
        level_data['tiles'].append(row)
    
    # Create graph visualization data
    graph_data_viz = {
        'nodes': [],
        'edges': []
    }
    
    for node in graph.nodes:
        graph_data_viz['nodes'].append({
            'id': node.id,
            'x': node.x,
            'y': node.y,
            'type': node.type.name if hasattr(node.type, 'name') else str(node.type)
        })
    
    for edge in graph.edges:
        graph_data_viz['edges'].append({
            'source': edge.source,
            'target': edge.target,
            'type': edge.type.name
        })
    
    # Create entities data
    entities_data = []
    for entity in env.entities:
        if isinstance(entity, dict):
            entities_data.append({
                'type': entity.get('entity_type', entity.get('type', 'unknown')),
                'x': entity.get('x', 0),
                'y': entity.get('y', 0)
            })
        else:
            entities_data.append({
                'type': getattr(entity, 'entity_type', getattr(entity, 'type', 'unknown')),
                'x': getattr(entity, 'x', 0),
                'y': getattr(entity, 'y', 0)
            })
    
    # Save visualization data
    viz_data = {
        'ninja_position': ninja_pos,
        'target_switch': leftmost_switch,
        'level': level_data,
        'graph': graph_data_viz,
        'entities': entities_data,
        'ninja_node_id': ninja_node.id,
        'target_node_id': target_node.id,
        'edge_type_stats': edge_types,
        'movement_diversity': movement_diversity
    }
    
    with open('/workspace/nclone/pathfinding_visualization.json', 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    print(f"✅ Visualization data saved to pathfinding_visualization.json")
    
    # Create HTML visualization
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>N++ Pathfinding Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .stats {{ background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .canvas-container {{ border: 2px solid #333; margin: 20px 0; }}
        canvas {{ display: block; }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 15px; margin: 10px 0; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .color-box {{ width: 20px; height: 20px; border: 1px solid #000; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>N++ Pathfinding Visualization</h1>
        
        <div class="stats">
            <h3>Statistics</h3>
            <p><strong>Ninja Position:</strong> ({ninja_pos[0]:.1f}, {ninja_pos[1]:.1f})</p>
            <p><strong>Target Switch:</strong> ({leftmost_switch[0]:.1f}, {leftmost_switch[1]:.1f})</p>
            <p><strong>Graph Nodes:</strong> {len(graph.nodes)}</p>
            <p><strong>Graph Edges:</strong> {len(graph.edges)}</p>
            <p><strong>Movement Diversity:</strong> {movement_diversity:.3f} ({len(movement_types)}/3 types)</p>
        </div>
        
        <div class="stats">
            <h3>Edge Type Distribution</h3>
            {''.join([f'<p><strong>{edge_type}:</strong> {count} edges ({(count/total_edges)*100:.1f}%)</p>' for edge_type, count in sorted(edge_types.items())])}
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="color-box" style="background: #333;"></div>
                <span>Solid Tiles</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: #fff;"></div>
                <span>Empty Tiles</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: #00ff00;"></div>
                <span>WALK Edges</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: #ff8800;"></div>
                <span>JUMP Edges</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: #0088ff;"></div>
                <span>FALL Edges</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: #ff0000;"></div>
                <span>Ninja</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: #ffff00;"></div>
                <span>Target Switch</span>
            </div>
        </div>
        
        <div class="canvas-container">
            <canvas id="visualization" width="1008" height="552"></canvas>
        </div>
        
        <script>
            const canvas = document.getElementById('visualization');
            const ctx = canvas.getContext('2d');
            
            // Load and render visualization
            fetch('pathfinding_visualization.json')
                .then(response => response.json())
                .then(data => {{
                    renderVisualization(data);
                }});
            
            function renderVisualization(data) {{
                // Clear canvas
                ctx.fillStyle = '#888';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Render tiles
                const tileSize = 24;
                for (let y = 0; y < data.level.height; y++) {{
                    for (let x = 0; x < data.level.width; x++) {{
                        const tileValue = data.level.tiles[y][x];
                        ctx.fillStyle = tileValue === 0 ? '#fff' : '#333';
                        ctx.fillRect(x * tileSize, y * tileSize, tileSize, tileSize);
                    }}
                }}
                
                // Render edges
                data.graph.edges.forEach(edge => {{
                    const sourceNode = data.graph.nodes.find(n => n.id === edge.source);
                    const targetNode = data.graph.nodes.find(n => n.id === edge.target);
                    
                    if (sourceNode && targetNode) {{
                        ctx.beginPath();
                        ctx.moveTo(sourceNode.x, sourceNode.y);
                        ctx.lineTo(targetNode.x, targetNode.y);
                        
                        switch(edge.type) {{
                            case 'WALK': ctx.strokeStyle = '#00ff00'; break;
                            case 'JUMP': ctx.strokeStyle = '#ff8800'; break;
                            case 'FALL': ctx.strokeStyle = '#0088ff'; break;
                            case 'FUNCTIONAL': ctx.strokeStyle = '#ffff00'; break;
                            default: ctx.strokeStyle = '#666'; break;
                        }}
                        
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }}
                }});
                
                // Render entities
                data.entities.forEach(entity => {{
                    ctx.fillStyle = entity.type === 4 ? '#ffff00' : '#888';
                    ctx.fillRect(entity.x - 6, entity.y - 6, 12, 12);
                }});
                
                // Render ninja
                ctx.fillStyle = '#ff0000';
                ctx.fillRect(data.ninja_position[0] - 8, data.ninja_position[1] - 8, 16, 16);
                
                // Render target
                ctx.strokeStyle = '#ff0000';
                ctx.lineWidth = 3;
                ctx.strokeRect(data.target_switch[0] - 10, data.target_switch[1] - 10, 20, 20);
            }}
        </script>
    </div>
</body>
</html>
"""
    
    with open('/workspace/nclone/pathfinding_visualization.html', 'w') as f:
        f.write(html_content)
    
    print(f"✅ HTML visualization saved to pathfinding_visualization.html")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"   Graph building: ✅ SUCCESS")
    print(f"   Movement diversity: {'✅ GOOD' if movement_diversity >= 0.67 else '⚠️  NEEDS IMPROVEMENT'}")
    print(f"   Ninja connectivity: {len(ninja_edges)} outgoing edges")
    print(f"   Visualization: ✅ READY")
    
    print(f"\n🌐 To view the visualization:")
    print(f"   Open: http://localhost:42934/pathfinding_visualization.html")
    print(f"   Or serve the files from /workspace/nclone/")

if __name__ == "__main__":
    comprehensive_pathfinding_test()