#!/usr/bin/env python3
"""
Analyze the actual map layout to understand connectivity between areas.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold


def analyze_map_layout():
    """Analyze the map layout to understand connectivity."""
    print("=" * 80)
    print("ANALYZING MAP LAYOUT FOR CONNECTIVITY")
    print("=" * 80)
    
    # Create environment
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    # Reset to load the map
    env.reset()
    
    # Get level data
    level_data = env.level_data
    ninja_pos = env.nplay_headless.ninja_position()
    
    print(f"Map: {level_data.width}x{level_data.height} tiles")
    print(f"Ninja: {ninja_pos}")
    
    # Print the map layout
    print(f"\nMap layout (0=empty, 1=solid, 2+=other):")
    print("   ", end="")
    for x in range(level_data.width):
        print(f"{x%10}", end="")
    print()
    
    for y in range(level_data.height):
        print(f"{y:2d} ", end="")
        for x in range(level_data.width):
            tile_value = level_data.get_tile(y, x)
            if tile_value == 0:
                char = "."
            elif tile_value == 1:
                char = "#"
            else:
                char = str(tile_value)
            print(char, end="")
        print()
    
    # Find empty tile clusters
    print(f"\n" + "=" * 60)
    print("FINDING EMPTY TILE CLUSTERS")
    print("=" * 60)
    
    visited = set()
    clusters = []
    
    for y in range(level_data.height):
        for x in range(level_data.width):
            if (x, y) in visited or level_data.get_tile(y, x) != 0:
                continue
            
            # Found new empty tile cluster
            cluster = []
            stack = [(x, y)]
            
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited:
                    continue
                
                visited.add((cx, cy))
                cluster.append((cx, cy))
                
                # Check 4-connected neighbors
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if (0 <= nx < level_data.width and 0 <= ny < level_data.height and
                        (nx, ny) not in visited and level_data.get_tile(ny, nx) == 0):
                        stack.append((nx, ny))
            
            clusters.append(cluster)
    
    # Sort clusters by size
    clusters.sort(key=len, reverse=True)
    
    print(f"Found {len(clusters)} empty tile clusters:")
    
    for i, cluster in enumerate(clusters, 1):
        # Calculate bounding box
        min_x = min(x for x, y in cluster)
        max_x = max(x for x, y in cluster)
        min_y = min(y for x, y in cluster)
        max_y = max(y for x, y in cluster)
        
        # Calculate center
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_pixel_x = center_x * 24 + 12
        center_pixel_y = center_y * 24 + 12
        
        # Check if ninja is in this cluster
        ninja_tile_x = int(ninja_pos[0] // 24)
        ninja_tile_y = int(ninja_pos[1] // 24)
        contains_ninja = (ninja_tile_x, ninja_tile_y) in cluster
        ninja_status = "🥷 NINJA" if contains_ninja else ""
        
        print(f"  Cluster {i:2d}: {len(cluster):2d} tiles - Area ({min_x:2d},{min_y:2d}) to ({max_x:2d},{max_y:2d}) - Center ({center_pixel_x:5.1f},{center_pixel_y:5.1f}) {ninja_status}")
    
    # Check potential connections between clusters
    print(f"\n" + "=" * 60)
    print("ANALYZING POTENTIAL CONNECTIONS BETWEEN CLUSTERS")
    print("=" * 60)
    
    if len(clusters) >= 2:
        # Check distances between cluster centers
        print(f"Distances between cluster centers:")
        
        for i in range(len(clusters)):
            for j in range(i + 1, min(i + 5, len(clusters))):  # Check first few pairs
                cluster1 = clusters[i]
                cluster2 = clusters[j]
                
                # Calculate centers
                center1_x = sum(x for x, y in cluster1) / len(cluster1) * 24 + 12
                center1_y = sum(y for x, y in cluster1) / len(cluster1) * 24 + 12
                center2_x = sum(x for x, y in cluster2) / len(cluster2) * 24 + 12
                center2_y = sum(y for x, y in cluster2) / len(cluster2) * 24 + 12
                
                distance = ((center2_x - center1_x)**2 + (center2_y - center1_y)**2)**0.5
                
                print(f"  Cluster {i+1} to Cluster {j+1}: {distance:.1f} pixels")
                
                # Check if there's a potential corridor
                # Sample points along the line between centers
                num_samples = int(distance / 12)  # Sample every 12 pixels
                corridor_blocked = False
                blocking_tiles = []
                
                for k in range(1, num_samples):
                    t = k / num_samples
                    sample_x = center1_x + t * (center2_x - center1_x)
                    sample_y = center1_y + t * (center2_y - center1_y)
                    
                    # Check tile at this position
                    tile_x = int(sample_x // 24)
                    tile_y = int(sample_y // 24)
                    
                    if (0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height):
                        tile_value = level_data.get_tile(tile_y, tile_x)
                        if tile_value == 1:  # Solid tile
                            corridor_blocked = True
                            blocking_tiles.append((tile_x, tile_y))
                
                if corridor_blocked:
                    print(f"    ❌ Blocked by {len(set(blocking_tiles))} solid tiles")
                else:
                    print(f"    ✅ Potential corridor exists!")
    
    # Check ninja's cluster connectivity
    print(f"\n" + "=" * 60)
    print("NINJA CLUSTER ANALYSIS")
    print("=" * 60)
    
    ninja_tile_x = int(ninja_pos[0] // 24)
    ninja_tile_y = int(ninja_pos[1] // 24)
    ninja_tile_value = level_data.get_tile(ninja_tile_y, ninja_tile_x)
    
    print(f"Ninja tile: ({ninja_tile_x}, {ninja_tile_y}) = {ninja_tile_value} ({'empty' if ninja_tile_value == 0 else 'solid' if ninja_tile_value == 1 else 'other'})")
    
    # Find which cluster contains ninja (if any)
    ninja_cluster_id = None
    for i, cluster in enumerate(clusters):
        if (ninja_tile_x, ninja_tile_y) in cluster:
            ninja_cluster_id = i
            break
    
    if ninja_cluster_id is not None:
        ninja_cluster = clusters[ninja_cluster_id]
        print(f"Ninja is in cluster {ninja_cluster_id + 1} with {len(ninja_cluster)} tiles")
        
        # Show tiles in ninja's cluster
        print(f"Ninja's cluster tiles:")
        for x, y in sorted(ninja_cluster):
            pixel_x = x * 24 + 12
            pixel_y = y * 24 + 12
            print(f"  Tile ({x:2d}, {y:2d}) -> Pixel ({pixel_x:3d}, {pixel_y:3d})")
    else:
        print(f"❌ Ninja is not in any empty tile cluster (ninja is in solid/other tile)")
    
    # Conclusion
    print(f"\n" + "=" * 60)
    print("CONNECTIVITY CONCLUSION")
    print("=" * 60)
    
    if len(clusters) == 1:
        print("✅ SINGLE CONNECTED AREA: All empty tiles are connected")
        print("   Long-distance pathfinding should work perfectly")
    elif len(clusters) <= 3:
        print(f"⚠️  MODERATE FRAGMENTATION: {len(clusters)} separate empty areas")
        print("   Some long-distance paths impossible due to solid barriers")
        print("   This may be intentional level design")
    else:
        print(f"❌ HIGH FRAGMENTATION: {len(clusters)} separate empty areas")
        print("   Most long-distance paths impossible")
        print("   May indicate level design issues or collision detection problems")
    
    if ninja_cluster_id is not None:
        ninja_cluster_size = len(clusters[ninja_cluster_id])
        total_empty_tiles = sum(len(cluster) for cluster in clusters)
        ninja_reachability = ninja_cluster_size / total_empty_tiles * 100
        
        print(f"\nNinja reachability: {ninja_cluster_size}/{total_empty_tiles} empty tiles ({ninja_reachability:.1f}%)")
        
        if ninja_reachability >= 80:
            print("✅ EXCELLENT: Ninja can reach most empty areas")
        elif ninja_reachability >= 50:
            print("✅ GOOD: Ninja can reach majority of empty areas")
        elif ninja_reachability >= 20:
            print("⚠️  LIMITED: Ninja can reach some empty areas")
        else:
            print("❌ POOR: Ninja can reach very few empty areas")
    
    return {
        'total_clusters': len(clusters),
        'ninja_cluster_id': ninja_cluster_id,
        'ninja_cluster_size': len(clusters[ninja_cluster_id]) if ninja_cluster_id is not None else 0,
        'total_empty_tiles': sum(len(cluster) for cluster in clusters),
        'largest_cluster_size': len(clusters[0]) if clusters else 0
    }


if __name__ == '__main__':
    results = analyze_map_layout()
    print(f"\nMap analysis completed. Results: {results}")