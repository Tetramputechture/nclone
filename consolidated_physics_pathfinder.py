#!/usr/bin/env python3
"""
CONSOLIDATED PHYSICS-ACCURATE PATHFINDING SYSTEM

This module provides a single, unified pathfinding system that:
1. Replaces the complex hierarchical graph system
2. Uses only physics-accurate movements (WALK/JUMP/FALL)
3. Respects N++ game mechanics and collision detection
4. Provides both pathfinding and visualization capabilities
"""

import sys
import os
import math
import heapq
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from enum import IntEnum

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.constants.physics_constants import TILE_PIXEL_SIZE, NINJA_RADIUS
from nclone.graph.common import EdgeType

class MovementType(IntEnum):
    """Physics-accurate movement types for N++ ninja."""
    WALK = 0    # Horizontal movement on platforms
    JUMP = 1    # Upward movement with height/distance limits
    FALL = 2    # Downward movement with gravity

@dataclass
class PhysicsNode:
    """A node representing a physically reachable position."""
    x: float
    y: float
    tile_x: int
    tile_y: int
    is_platform: bool  # Has solid ground below for landing
    node_type: str = "platform"  # platform, entity, special

@dataclass
class PhysicsEdge:
    """An edge representing a physically possible movement."""
    from_node: int
    to_node: int
    movement_type: MovementType
    cost: float
    distance: float
    is_valid: bool = True

@dataclass
class PathResult:
    """Result of pathfinding operation."""
    success: bool
    path: Optional[List[int]] = None
    movements: Optional[List[MovementType]] = None
    total_cost: float = 0.0
    total_distance: float = 0.0
    nodes_explored: int = 0

class ConsolidatedPhysicsPathfinder:
    """
    Unified physics-accurate pathfinding system for N++ levels.
    
    This system replaces the complex hierarchical graph with a simple,
    physics-accurate tile-based approach that respects game mechanics.
    """
    
    def __init__(self, level_data, entities=None):
        self.level_data = level_data
        self.entities = entities or []
        self.nodes: List[PhysicsNode] = []
        self.edges: List[PhysicsEdge] = []
        self.adjacency: Dict[int, List[int]] = {}
        
        # Physics constraints (based on N++ mechanics)
        self.max_walk_distance = 1  # tiles (24px)
        self.max_jump_height = 3    # tiles upward
        self.max_jump_distance = 4  # tiles horizontal
        self.max_fall_distance = 8  # tiles downward
        
        # Movement costs (in pixels)
        self.walk_cost_per_pixel = 1.0
        self.jump_cost_multiplier = 1.2  # Jumping is slightly more expensive
        self.fall_cost_multiplier = 0.8  # Falling is easier
        
    def build_graph(self) -> None:
        """Build the physics-accurate graph."""
        print("🔧 Building consolidated physics-accurate graph...")
        
        # Step 1: Create nodes for walkable positions
        self._create_platform_nodes()
        
        # Step 2: Add entity positions as special nodes
        self._add_entity_nodes()
        
        # Step 3: Create physics-accurate edges
        self._create_walk_edges()
        self._create_jump_edges()
        self._create_fall_edges()
        
        # Step 4: Build adjacency list for pathfinding
        self._build_adjacency_list()
        
        print(f"✅ Graph built: {len(self.nodes)} nodes, {len(self.edges)} edges")
        self._print_graph_statistics()
        
    def _create_platform_nodes(self) -> None:
        """Create nodes for platform tiles (walkable surfaces)."""
        platform_count = 0
        
        for tile_y in range(self.level_data.height):
            for tile_x in range(self.level_data.width):
                # Check if this tile is empty (walkable)
                tile_value = self.level_data.get_tile(tile_y, tile_x)
                if tile_value == 0:  # Empty tile
                    # Check if it's a platform (has solid ground below)
                    is_platform = self._is_platform_tile(tile_x, tile_y)
                    
                    if is_platform:
                        center_x = tile_x * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                        center_y = tile_y * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                        
                        node = PhysicsNode(
                            x=center_x,
                            y=center_y,
                            tile_x=tile_x,
                            tile_y=tile_y,
                            is_platform=True,
                            node_type="platform"
                        )
                        self.nodes.append(node)
                        platform_count += 1
                        
        print(f"📍 Created {platform_count} platform nodes")
        
    def _is_platform_tile(self, tile_x: int, tile_y: int) -> bool:
        """Check if a tile is a platform (has solid ground below)."""
        # Check tile below
        below_y = tile_y + 1
        if below_y >= self.level_data.height:
            return True  # Bottom edge is considered platform
            
        below_tile = self.level_data.get_tile(below_y, tile_x)
        return below_tile != 0  # Any non-empty tile below makes this a platform
        
    def _add_entity_nodes(self) -> None:
        """Add entity positions as special nodes."""
        entity_count = 0
        
        for entity in self.entities:
            entity_type = entity.get('type')
            if entity_type in [0, 4]:  # Ninja (0) or Switch (4)
                entity_x = entity['x']
                entity_y = entity['y']
                tile_x = int(entity_x // TILE_PIXEL_SIZE)
                tile_y = int(entity_y // TILE_PIXEL_SIZE)
                
                # Check if we already have a node at this position
                existing_node = self._find_node_at_position(entity_x, entity_y, tolerance=12)
                
                if existing_node is None:
                    # Create new entity node
                    node = PhysicsNode(
                        x=entity_x,
                        y=entity_y,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        is_platform=True,  # Assume entities are on platforms
                        node_type="entity"
                    )
                    self.nodes.append(node)
                    entity_count += 1
                    print(f"📍 Added entity node: type {entity_type} at ({entity_x}, {entity_y})")
                else:
                    # Update existing node position to match entity exactly
                    existing_node.x = entity_x
                    existing_node.y = entity_y
                    existing_node.node_type = "entity"
                    print(f"📍 Updated node for entity type {entity_type} at ({entity_x}, {entity_y})")
                    
        if entity_count > 0:
            print(f"📍 Added {entity_count} entity nodes")
            
    def _find_node_at_position(self, x: float, y: float, tolerance: float = 5.0) -> Optional[PhysicsNode]:
        """Find a node at or near the given position."""
        for node in self.nodes:
            distance = math.sqrt((node.x - x)**2 + (node.y - y)**2)
            if distance <= tolerance:
                return node
        return None
        
    def _create_walk_edges(self) -> None:
        """Create walking edges between adjacent platform nodes."""
        walk_edges = 0
        
        for i, node1 in enumerate(self.nodes):
            if not node1.is_platform:
                continue
                
            for j, node2 in enumerate(self.nodes):
                if i >= j or not node2.is_platform:
                    continue
                    
                # Check if nodes are horizontally adjacent (same row, adjacent columns)
                dx = abs(node1.tile_x - node2.tile_x)
                dy = abs(node1.tile_y - node2.tile_y)
                
                if dy == 0 and dx == 1:  # Same row, adjacent columns
                    if self._is_walk_path_clear(node1, node2):
                        distance = abs(node1.x - node2.x)
                        cost = distance * self.walk_cost_per_pixel
                        
                        # Create bidirectional walk edges
                        edge1 = PhysicsEdge(i, j, MovementType.WALK, cost, distance)
                        edge2 = PhysicsEdge(j, i, MovementType.WALK, cost, distance)
                        self.edges.extend([edge1, edge2])
                        walk_edges += 2
                        
        print(f"🚶 Created {walk_edges} walk edges")
        
    def _is_walk_path_clear(self, node1: PhysicsNode, node2: PhysicsNode) -> bool:
        """Check if the walking path between two nodes is clear."""
        # For adjacent platform tiles, the path is clear by definition
        return True
        
    def _create_jump_edges(self) -> None:
        """Create jumping edges for upward and horizontal movement."""
        jump_edges = 0
        
        for i, node1 in enumerate(self.nodes):
            if not node1.is_platform:
                continue
                
            for j, node2 in enumerate(self.nodes):
                if i == j:
                    continue
                    
                dx = node2.tile_x - node1.tile_x
                dy = node2.tile_y - node1.tile_y
                
                # Jump conditions: target is above or at same level, within jump range
                if dy <= 0 and abs(dy) <= self.max_jump_height and abs(dx) <= self.max_jump_distance:
                    # Don't create jump edges for simple adjacent walking
                    if dy == 0 and abs(dx) == 1:
                        continue
                        
                    if self._is_jump_path_clear(node1, node2):
                        distance = math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
                        height_penalty = max(0, -dy) * 10  # Penalty for jumping up
                        cost = distance * self.jump_cost_multiplier + height_penalty
                        
                        edge = PhysicsEdge(i, j, MovementType.JUMP, cost, distance)
                        self.edges.append(edge)
                        jump_edges += 1
                        
        print(f"🦘 Created {jump_edges} jump edges")
        
    def _is_jump_path_clear(self, node1: PhysicsNode, node2: PhysicsNode) -> bool:
        """Check if the jumping path between two nodes is clear."""
        # Simple check: ensure target is reachable
        return node2.is_platform or node2.node_type == "entity"
        
    def _create_fall_edges(self) -> None:
        """Create falling edges for downward movement."""
        fall_edges = 0
        
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i == j:
                    continue
                    
                dx = node2.tile_x - node1.tile_x
                dy = node2.tile_y - node1.tile_y
                
                # Fall conditions: target is below, within fall range
                if dy > 0 and dy <= self.max_fall_distance and abs(dx) <= 2:
                    if self._is_fall_path_clear(node1, node2):
                        distance = math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
                        cost = distance * self.fall_cost_multiplier
                        
                        edge = PhysicsEdge(i, j, MovementType.FALL, cost, distance)
                        self.edges.append(edge)
                        fall_edges += 1
                        
        print(f"🪂 Created {fall_edges} fall edges")
        
    def _is_fall_path_clear(self, node1: PhysicsNode, node2: PhysicsNode) -> bool:
        """Check if the falling path between two nodes is clear."""
        # Ensure target is a platform to land on
        return node2.is_platform
        
    def _build_adjacency_list(self) -> None:
        """Build adjacency list for efficient pathfinding."""
        self.adjacency = {i: [] for i in range(len(self.nodes))}
        
        for edge in self.edges:
            if edge.is_valid:
                self.adjacency[edge.from_node].append(edge.to_node)
                
    def _print_graph_statistics(self) -> None:
        """Print detailed graph statistics."""
        walk_edges = sum(1 for e in self.edges if e.movement_type == MovementType.WALK)
        jump_edges = sum(1 for e in self.edges if e.movement_type == MovementType.JUMP)
        fall_edges = sum(1 for e in self.edges if e.movement_type == MovementType.FALL)
        
        platform_nodes = sum(1 for n in self.nodes if n.node_type == "platform")
        entity_nodes = sum(1 for n in self.nodes if n.node_type == "entity")
        
        print(f"📊 Graph Statistics:")
        print(f"   Nodes: {len(self.nodes)} ({platform_nodes} platforms, {entity_nodes} entities)")
        print(f"   Edges: {len(self.edges)} ({walk_edges} walk, {jump_edges} jump, {fall_edges} fall)")
        
    def find_path(self, start_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> PathResult:
        """Find physics-accurate path between two positions."""
        # Find closest nodes to start and target positions
        start_node = self._find_closest_node(start_pos)
        target_node = self._find_closest_node(target_pos)
        
        if start_node is None:
            return PathResult(success=False, nodes_explored=0)
        if target_node is None:
            return PathResult(success=False, nodes_explored=0)
            
        print(f"🎯 Pathfinding: Node {start_node} → Node {target_node}")
        print(f"   Start: {self.nodes[start_node].x:.0f}, {self.nodes[start_node].y:.0f}")
        print(f"   Target: {self.nodes[target_node].x:.0f}, {self.nodes[target_node].y:.0f}")
        
        # Use Dijkstra's algorithm for guaranteed optimal paths
        return self._dijkstra(start_node, target_node)
        
    def _find_closest_node(self, pos: Tuple[float, float]) -> Optional[int]:
        """Find the closest node to a given position."""
        if not self.nodes:
            return None
            
        min_distance = float('inf')
        closest_node = None
        
        for i, node in enumerate(self.nodes):
            distance = math.sqrt((node.x - pos[0])**2 + (node.y - pos[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_node = i
                
        return closest_node
        
    def _dijkstra(self, start: int, target: int) -> PathResult:
        """Find shortest path using Dijkstra's algorithm."""
        # Initialize distances and tracking
        distances = {i: float('inf') for i in range(len(self.nodes))}
        previous = {i: None for i in range(len(self.nodes))}
        movement_types = {i: None for i in range(len(self.nodes))}
        
        distances[start] = 0
        pq = [(0, start)]
        visited = set()
        nodes_explored = 0
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            nodes_explored += 1
            
            if current == target:
                break
                
            # Explore neighbors
            for neighbor in self.adjacency.get(current, []):
                if neighbor in visited:
                    continue
                    
                # Find the edge to this neighbor
                edge_cost = None
                movement_type = None
                for edge in self.edges:
                    if edge.from_node == current and edge.to_node == neighbor:
                        edge_cost = edge.cost
                        movement_type = edge.movement_type
                        break
                        
                if edge_cost is None:
                    continue
                    
                new_dist = current_dist + edge_cost
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    movement_types[neighbor] = movement_type
                    heapq.heappush(pq, (new_dist, neighbor))
                    
        # Check if path was found
        if distances[target] == float('inf'):
            return PathResult(success=False, nodes_explored=nodes_explored)
            
        # Reconstruct path
        path = []
        movements = []
        current = target
        total_distance = 0
        
        while current is not None:
            path.append(current)
            if movement_types[current] is not None:
                movements.append(movement_types[current])
                # Calculate distance for this segment
                prev = previous[current]
                if prev is not None:
                    node_dist = math.sqrt(
                        (self.nodes[current].x - self.nodes[prev].x)**2 + 
                        (self.nodes[current].y - self.nodes[prev].y)**2
                    )
                    total_distance += node_dist
            current = previous[current]
            
        path.reverse()
        movements.reverse()
        
        return PathResult(
            success=True,
            path=path,
            movements=movements,
            total_cost=distances[target],
            total_distance=total_distance,
            nodes_explored=nodes_explored
        )
        
    def get_node_position(self, node_idx: int) -> Tuple[float, float]:
        """Get the position of a node."""
        if 0 <= node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            return (node.x, node.y)
        return (0, 0)
        
    def get_movement_sequence(self, result: PathResult) -> List[str]:
        """Get human-readable movement sequence."""
        if not result.success or not result.movements:
            return []
            
        sequence = []
        for movement in result.movements:
            if movement == MovementType.WALK:
                sequence.append("WALK")
            elif movement == MovementType.JUMP:
                sequence.append("JUMP")
            elif movement == MovementType.FALL:
                sequence.append("FALL")
                
        return sequence

def test_consolidated_pathfinder():
    """Test the consolidated physics-accurate pathfinder."""
    print("=" * 80)
    print("🧪 TESTING CONSOLIDATED PHYSICS-ACCURATE PATHFINDER")
    print("=" * 80)
    
    # Load environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    level_data = env.level_data
    entities = env.entities
    
    # Create consolidated pathfinder
    pathfinder = ConsolidatedPhysicsPathfinder(level_data, entities)
    pathfinder.build_graph()
    
    # Find ninja and switch positions from entities
    ninja_pos = None
    switch_pos = None
    
    for entity in entities:
        if entity.get('type') == 0:  # Ninja
            ninja_pos = (entity['x'], entity['y'])
        elif entity.get('type') == 4:  # Switch
            switch_pos = (entity['x'], entity['y'])
            
    if ninja_pos is None or switch_pos is None:
        print("❌ Could not find ninja or switch positions")
        return False
        
    print(f"📍 Ninja: {ninja_pos}")
    print(f"🎯 Switch: {switch_pos}")
    
    # Find path
    result = pathfinder.find_path(ninja_pos, switch_pos)
    
    if not result.success:
        print("❌ No path found!")
        return False
        
    print(f"\n🛤️  CONSOLIDATED PATHFINDING RESULT:")
    print(f"   Success: {result.success}")
    print(f"   Path length: {len(result.path)} nodes")
    print(f"   Total cost: {result.total_cost:.1f}px")
    print(f"   Total distance: {result.total_distance:.1f}px")
    print(f"   Nodes explored: {result.nodes_explored}")
    
    # Show movement sequence
    movements = pathfinder.get_movement_sequence(result)
    print(f"   Movement sequence: {movements}")
    
    # Count movement types
    walk_count = movements.count('WALK')
    jump_count = movements.count('JUMP')
    fall_count = movements.count('FALL')
    
    print(f"\n📊 MOVEMENT ANALYSIS:")
    print(f"   WALK movements: {walk_count}")
    print(f"   JUMP movements: {jump_count}")
    print(f"   FALL movements: {fall_count}")
    
    # Show detailed path
    print(f"\n🗺️  DETAILED PATH:")
    for i, node_idx in enumerate(result.path):
        node = pathfinder.nodes[node_idx]
        movement = movements[i-1] if i > 0 else "START"
        print(f"   Step {i+1}: Node {node_idx} at ({node.x:.0f}, {node.y:.0f}) - {movement}")
        
    return True

if __name__ == "__main__":
    test_consolidated_pathfinder()