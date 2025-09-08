#!/usr/bin/env python3
"""
Create a physics-accurate pathfinding system that follows N++ game mechanics.

This system uses a simpler tile-based approach instead of the complex hierarchical graph.
"""

import sys
import os
sys.path.insert(0, '/workspace/nclone')

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.constants.physics_constants import TILE_PIXEL_SIZE
from nclone.graph.common import EdgeType

@dataclass
class PhysicsNode:
    """A node in the physics-accurate pathfinding graph."""
    x: float
    y: float
    tile_x: int
    tile_y: int
    is_walkable: bool
    is_platform: bool  # Has solid ground below
    
@dataclass
class PhysicsEdge:
    """An edge representing a physically possible movement."""
    from_node: int
    to_node: int
    movement_type: EdgeType
    cost: float
    is_valid: bool

class PhysicsAccuratePathfinder:
    """A pathfinder that follows N++ physics rules."""
    
    def __init__(self, level_data, entities):
        self.level_data = level_data
        self.entities = entities
        self.nodes: List[PhysicsNode] = []
        self.edges: List[PhysicsEdge] = []
        self.adjacency: Dict[int, List[int]] = {}
        
    def build_physics_graph(self) -> None:
        """Build a graph that follows N++ physics rules."""
        print("🔧 Building physics-accurate graph...")
        
        # Create nodes for walkable tiles
        self._create_tile_nodes()
        
        # Create edges for physically possible movements
        self._create_walk_edges()
        self._create_jump_edges()
        self._create_fall_edges()
        
        print(f"✅ Physics graph: {len(self.nodes)} nodes, {len(self.edges)} edges")
        
    def _create_tile_nodes(self) -> None:
        """Create nodes for each walkable tile in the level."""
        for tile_y in range(self.level_data.height):
            for tile_x in range(self.level_data.width):
                # Calculate tile center position
                center_x = tile_x * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                center_y = tile_y * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                
                # Check if this tile is walkable (empty space)
                tile_value = self.level_data.get_tile(tile_y, tile_x)
                is_walkable = (tile_value == 0)
                
                if is_walkable:
                    # Check if this is a platform (has solid ground below or is at bottom)
                    is_platform = self._is_platform_tile(tile_x, tile_y)
                    
                    # Only create nodes for platforms or special cases
                    if is_platform:
                        node = PhysicsNode(
                            x=center_x,
                            y=center_y,
                            tile_x=tile_x,
                            tile_y=tile_y,
                            is_walkable=True,
                            is_platform=is_platform
                        )
                        self.nodes.append(node)
                    
        # Also add entity positions as special nodes
        print(f"🔍 Checking {len(self.entities)} entities...")
        for entity in self.entities:
            print(f"   Entity: {entity}")
            if entity.get('type') in [0, 4]:  # 0=ninja, 4=switch
                entity_x = entity['x']
                entity_y = entity['y']
                tile_x = int(entity_x // TILE_PIXEL_SIZE)
                tile_y = int(entity_y // TILE_PIXEL_SIZE)
                
                # Check if we already have a node at this tile
                existing_node = None
                for node in self.nodes:
                    if node.tile_x == tile_x and node.tile_y == tile_y:
                        existing_node = node
                        break
                        
                if existing_node is None:
                    # Create new node for entity
                    node = PhysicsNode(
                        x=entity_x,
                        y=entity_y,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        is_walkable=True,
                        is_platform=True  # Assume entities are on platforms
                    )
                    self.nodes.append(node)
                    print(f"📍 Added entity node: {entity['type']} at ({entity_x}, {entity_y})")
                else:
                    # Update existing node position to match entity
                    existing_node.x = entity_x
                    existing_node.y = entity_y
                    print(f"📍 Updated node for {entity['type']} at ({entity_x}, {entity_y})")
                    
        print(f"📍 Created {len(self.nodes)} walkable tile nodes")
        
    def _is_platform_tile(self, tile_x: int, tile_y: int) -> bool:
        """Check if a tile has solid ground below it (is a platform)."""
        # Check tile below
        below_y = tile_y + 1
        if below_y >= self.level_data.height:
            return True  # Bottom of level is considered platform
            
        below_tile = self.level_data.get_tile(below_y, tile_x)
        return below_tile == 1  # Solid tile below
        
    def _create_walk_edges(self) -> None:
        """Create walking edges between adjacent platform tiles."""
        walk_edges = 0
        
        for i, node1 in enumerate(self.nodes):
            if not node1.is_platform:
                continue
                
            for j, node2 in enumerate(self.nodes):
                if i == j or not node2.is_platform:
                    continue
                    
                # Check if nodes are horizontally adjacent
                dx = abs(node1.tile_x - node2.tile_x)
                dy = abs(node1.tile_y - node2.tile_y)
                
                if dx == 1 and dy == 0:  # Horizontally adjacent
                    # Check if path is clear (no walls between)
                    if self._is_walk_path_clear(node1, node2):
                        cost = TILE_PIXEL_SIZE  # 24 pixels
                        edge = PhysicsEdge(i, j, EdgeType.WALK, cost, True)
                        self.edges.append(edge)
                        walk_edges += 1
                        
        print(f"🚶 Created {walk_edges} walk edges")
        
    def _is_walk_path_clear(self, node1: PhysicsNode, node2: PhysicsNode) -> bool:
        """Check if the walking path between two nodes is clear."""
        # For adjacent tiles, just check that both are platforms
        return node1.is_platform and node2.is_platform
        
    def _create_jump_edges(self) -> None:
        """Create jumping edges for upward movement."""
        jump_edges = 0
        max_jump_height = 3  # tiles
        max_jump_distance = 4  # tiles
        
        for i, node1 in enumerate(self.nodes):
            if not node1.is_platform:
                continue
                
            for j, node2 in enumerate(self.nodes):
                if i == j:
                    continue
                    
                dx = node2.tile_x - node1.tile_x
                dy = node2.tile_y - node1.tile_y
                
                # Jump conditions: target is above and within jump range
                if dy < 0 and abs(dy) <= max_jump_height and abs(dx) <= max_jump_distance:
                    if self._is_jump_path_clear(node1, node2):
                        # Calculate jump cost based on distance and height
                        distance = math.sqrt(dx*dx + dy*dy) * TILE_PIXEL_SIZE
                        height_penalty = abs(dy) * 10  # Penalty for height
                        cost = distance + height_penalty
                        
                        edge = PhysicsEdge(i, j, EdgeType.JUMP, cost, True)
                        self.edges.append(edge)
                        jump_edges += 1
                        
                # Also allow jumps to same level (horizontal jumps) for switches and special targets
                elif dy == 0 and abs(dx) <= 2:  # Short horizontal jumps
                    if self._is_jump_path_clear(node1, node2):
                        distance = abs(dx) * TILE_PIXEL_SIZE
                        cost = distance + 5  # Small penalty for horizontal jump
                        
                        edge = PhysicsEdge(i, j, EdgeType.JUMP, cost, True)
                        self.edges.append(edge)
                        jump_edges += 1
                        
        print(f"🦘 Created {jump_edges} jump edges")
        
    def _is_jump_path_clear(self, node1: PhysicsNode, node2: PhysicsNode) -> bool:
        """Check if the jumping path between two nodes is clear."""
        # Simple check: ensure target is a valid landing spot
        return node2.is_walkable
        
    def _create_fall_edges(self) -> None:
        """Create falling edges for downward movement."""
        fall_edges = 0
        max_fall_distance = 6  # tiles
        
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i == j:
                    continue
                    
                dx = node2.tile_x - node1.tile_x
                dy = node2.tile_y - node1.tile_y
                
                # Fall conditions: target is below and within fall range
                if dy > 0 and dy <= max_fall_distance and abs(dx) <= 2:
                    if self._is_fall_path_clear(node1, node2):
                        # Calculate fall cost (falling is relatively cheap)
                        distance = math.sqrt(dx*dx + dy*dy) * TILE_PIXEL_SIZE
                        cost = distance * 0.8  # Falling is easier than walking
                        
                        edge = PhysicsEdge(i, j, EdgeType.FALL, cost, True)
                        self.edges.append(edge)
                        fall_edges += 1
                        
                # Also allow short vertical drops (1 tile) with small horizontal movement
                elif dy == 1 and abs(dx) <= 1:
                    if self._is_fall_path_clear(node1, node2):
                        distance = math.sqrt(dx*dx + dy*dy) * TILE_PIXEL_SIZE
                        cost = distance * 0.9  # Short falls are easy
                        
                        edge = PhysicsEdge(i, j, EdgeType.FALL, cost, True)
                        self.edges.append(edge)
                        fall_edges += 1
                        
        print(f"🪂 Created {fall_edges} fall edges")
        
    def _is_fall_path_clear(self, node1: PhysicsNode, node2: PhysicsNode) -> bool:
        """Check if the falling path between two nodes is clear."""
        # Ensure target is a platform to land on
        return node2.is_platform
        
    def find_path(self, start_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> Optional[List[Tuple[int, EdgeType]]]:
        """Find a physics-accurate path from start to target."""
        # Find closest nodes to start and target positions
        start_node = self._find_closest_node(start_pos)
        target_node = self._find_closest_node(target_pos)
        
        if start_node is None or target_node is None:
            return None
            
        print(f"🎯 Pathfinding: Node {start_node} → Node {target_node}")
        
        # Build adjacency list
        self._build_adjacency_list()
        
        # Use Dijkstra's algorithm
        return self._dijkstra(start_node, target_node)
        
    def _find_closest_node(self, pos: Tuple[float, float]) -> Optional[int]:
        """Find the closest node to a given position."""
        min_distance = float('inf')
        closest_node = None
        
        for i, node in enumerate(self.nodes):
            distance = math.sqrt((node.x - pos[0])**2 + (node.y - pos[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_node = i
                
        return closest_node
        
    def _build_adjacency_list(self) -> None:
        """Build adjacency list from edges."""
        self.adjacency = {i: [] for i in range(len(self.nodes))}
        
        for edge in self.edges:
            if edge.is_valid:
                self.adjacency[edge.from_node].append(edge.to_node)
                
        # Debug: Check connectivity for ninja and target nodes
        print(f"🔍 Adjacency debugging:")
        ninja_node = None
        target_node = None
        
        # Find ninja and target nodes (use dynamic positions)
        ninja_pos = (108, 372)  # From entity data
        target_pos = (588, 324)  # From entity data
        
        for i, node in enumerate(self.nodes):
            if abs(node.x - ninja_pos[0]) < 5 and abs(node.y - ninja_pos[1]) < 5:
                ninja_node = i
            if abs(node.x - target_pos[0]) < 5 and abs(node.y - target_pos[1]) < 5:
                target_node = i
                
        if ninja_node is not None:
            print(f"   Ninja node {ninja_node} has {len(self.adjacency[ninja_node])} connections: {self.adjacency[ninja_node][:5]}...")
        if target_node is not None:
            print(f"   Target node {target_node} has {len(self.adjacency[target_node])} connections: {self.adjacency[target_node][:5]}...")
            
        # Check if there are any connections at all
        total_connections = sum(len(neighbors) for neighbors in self.adjacency.values())
        print(f"   Total connections in graph: {total_connections}")
                
    def _dijkstra(self, start: int, target: int) -> Optional[List[Tuple[int, EdgeType]]]:
        """Find shortest path using Dijkstra's algorithm."""
        import heapq
        
        # Initialize distances and previous nodes
        distances = {i: float('inf') for i in range(len(self.nodes))}
        previous = {i: None for i in range(len(self.nodes))}
        edge_types = {i: None for i in range(len(self.nodes))}
        
        distances[start] = 0
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == target:
                break
                
            # Check all neighbors
            for neighbor in self.adjacency.get(current, []):
                if neighbor in visited:
                    continue
                    
                # Find the edge to this neighbor
                edge_cost = None
                edge_type = None
                for edge in self.edges:
                    if edge.from_node == current and edge.to_node == neighbor:
                        edge_cost = edge.cost
                        edge_type = edge.movement_type
                        break
                        
                if edge_cost is None:
                    continue
                    
                new_dist = current_dist + edge_cost
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    edge_types[neighbor] = edge_type
                    heapq.heappush(pq, (new_dist, neighbor))
                    
        # Reconstruct path
        if distances[target] == float('inf'):
            return None
            
        path = []
        current = target
        while previous[current] is not None:
            path.append((current, edge_types[current]))
            current = previous[current]
        path.append((start, None))  # Start node has no incoming edge
        path.reverse()
        
        return path

def test_physics_pathfinding():
    """Test the physics-accurate pathfinding system."""
    print("=" * 80)
    print("🧪 TESTING PHYSICS-ACCURATE PATHFINDING")
    print("=" * 80)
    
    # Load environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    level_data = env.level_data
    entities = env.entities
    
    # Get ninja position
    ninja_position = (132, 444)  # Known ninja position
    
    # Find target (exit switch from entities)
    target_position = None
    for entity in entities:
        if entity.get('type') == 4:  # Switch
            target_position = (entity['x'], entity['y'])
            break
    
    if target_position is None:
        target_position = (564, 276)  # Fallback known switch position
    
    print(f"📍 Ninja: {ninja_position}")
    print(f"🎯 Target: {target_position}")
    
    # Create pathfinder
    pathfinder = PhysicsAccuratePathfinder(level_data, entities)
    pathfinder.build_physics_graph()
    
    # Debug: Check what tiles are around the target
    target_tile_x = int(target_position[0] // TILE_PIXEL_SIZE)
    target_tile_y = int(target_position[1] // TILE_PIXEL_SIZE)
    print(f"🔍 Target tile: ({target_tile_x}, {target_tile_y})")
    
    # Check surrounding tiles
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            check_x = target_tile_x + dx
            check_y = target_tile_y + dy
            if 0 <= check_x < level_data.width and 0 <= check_y < level_data.height:
                tile_value = level_data.get_tile(check_y, check_x)
                print(f"   Tile ({check_x}, {check_y}): {tile_value}")
                
    # Also check if there are any nodes near the target
    print(f"🔍 Nodes near target:")
    for i, node in enumerate(pathfinder.nodes):
        distance = math.sqrt((node.x - target_position[0])**2 + (node.y - target_position[1])**2)
        if distance < 100:  # Within 100 pixels
            print(f"   Node {i}: ({node.x:.0f}, {node.y:.0f}) tile({node.tile_x}, {node.tile_y}) platform={node.is_platform} dist={distance:.1f}px")
    
    # Find path
    path = pathfinder.find_path(ninja_position, target_position)
    
    if path is None:
        print("❌ No path found!")
        return
        
    print(f"\n🛤️  PHYSICS-ACCURATE PATH:")
    print(f"   Path length: {len(path)} nodes")
    
    total_cost = 0
    movement_types = []
    
    for i, (node_idx, movement_type) in enumerate(path):
        node = pathfinder.nodes[node_idx]
        print(f"   Step {i+1}: Node {node_idx} at ({node.x:.0f}, {node.y:.0f}) tile({node.tile_x}, {node.tile_y})")
        
        if movement_type is not None:
            print(f"            Movement: {movement_type.name}")
            movement_types.append(movement_type.name)
            
            # Find edge cost
            for edge in pathfinder.edges:
                if edge.to_node == node_idx and edge.movement_type == movement_type:
                    total_cost += edge.cost
                    print(f"            Cost: {edge.cost:.1f}px")
                    break
                    
    print(f"\n📊 PATH SUMMARY:")
    print(f"   Total cost: {total_cost:.1f}px")
    print(f"   Movement types: {movement_types}")
    
    # Analyze movement sequence
    print(f"\n🔍 MOVEMENT ANALYSIS:")
    walk_count = movement_types.count('WALK')
    jump_count = movement_types.count('JUMP')
    fall_count = movement_types.count('FALL')
    
    print(f"   WALK movements: {walk_count}")
    print(f"   JUMP movements: {jump_count}")
    print(f"   FALL movements: {fall_count}")
    
    # Check if this follows expected N++ physics
    expected_sequence = ["WALK", "JUMP", "WALK", "JUMP", "JUMP"]
    if movement_types == expected_sequence:
        print("✅ Movement sequence matches expected N++ physics!")
    else:
        print(f"⚠️  Movement sequence differs from expected: {expected_sequence}")

if __name__ == "__main__":
    test_physics_pathfinding()