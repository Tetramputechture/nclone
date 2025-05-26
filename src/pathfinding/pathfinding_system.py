import numpy as np
from typing import List, Tuple, Dict, Optional

from .surface_parser import SurfaceParser, Surface
from .navigation_graph import NavigationGraphBuilder, JumpCalculator, JumpTrajectory
from .astar_pathfinder import PlatformerAStar, MultiObjectivePathfinder
from .dynamic_pathfinding import DynamicPathfinder, EnemyPredictor
from .path_executor import PathOptimizer, MovementController
from .pathfinding_visualizer import PathfindingVisualizer
from .utils import CollisionChecker, Enemy # For stubs and type hinting

class N2PlusPathfindingSystem:
    """Complete pathfinding system for N++ simulation"""
    
    def __init__(self, tile_map: np.ndarray, physics_params: Optional[dict] = None):
        if physics_params is None:
            physics_params = {} # Default empty dict

        # Initialize components
        self.tile_map = tile_map
        self.collision_checker = CollisionChecker(tile_map)
        
        self.surface_parser = SurfaceParser(tile_map)
        self.surfaces: List[Surface] = self.surface_parser.parse_surfaces()
        
        self.graph_builder = NavigationGraphBuilder(self.surfaces, self.collision_checker, tile_map.shape)
        # Build the graph without jump edges first
        self.nav_graph = self.graph_builder.build_graph() 
        
        self.jump_calculator = JumpCalculator(self.collision_checker)
        # After initial graph is built, add jump edges
        self._add_jump_edges_to_graph() 
        
        self.pathfinder = PlatformerAStar(self.nav_graph, self.jump_calculator)
        self.multi_objective_pathfinder = MultiObjectivePathfinder(self.pathfinder)
        
        # Dynamic components (initialized when enemies are known)
        self.enemies: List[Enemy] = []
        self.enemy_predictor: Optional[EnemyPredictor] = None
        self.dynamic_pathfinder: Optional[DynamicPathfinder] = None
        
        self.path_optimizer = PathOptimizer(self.collision_checker)
        self.movement_controller = MovementController(physics_params)
        
        # Visualization (optional)
        self.visualizer = PathfindingVisualizer()
        print("N2PlusPathfindingSystem initialized.")

    def _add_jump_edges_to_graph(self):
        """Iterate through graph nodes and use JumpCalculator to add jump/fall edges."""
        print("Attempting to add jump edges to the navigation graph...")
        if not self.nav_graph or not self.jump_calculator:
            print("Nav graph or jump calculator not initialized. Skipping jump edge creation.")
            return

        nodes_to_consider = list(self.nav_graph.nodes(data=True))
        num_potential_jumps_added = 0

        # This can be computationally expensive (N^2 node pairs)
        # Optimization: only consider jumps between nodes that are reasonably close
        # and where a jump is plausible (e.g., from a floor/wall to another floor/wall).
        for i in range(len(nodes_to_consider)):
            node1_id, data1 = nodes_to_consider[i]
            nav_node1 = data1.get('nav_node_object')
            if not nav_node1 or not nav_node1.surface: continue

            for j in range(len(nodes_to_consider)):
                if i == j: continue
                node2_id, data2 = nodes_to_consider[j]
                nav_node2 = data2.get('nav_node_object')
                if not nav_node2 or not nav_node2.surface: continue
                
                # Heuristic: Don't try to jump if too far (e.g. > 15 tiles = 360 pixels)
                dist_sq = (nav_node1.position[0] - nav_node2.position[0])**2 + \
                          (nav_node1.position[1] - nav_node2.position[1])**2
                if dist_sq > (360*360): 
                    continue

                # Try to calculate a jump from node1 to node2
                # Surface type of the START node is important for jump initiation
                trajectory: Optional[JumpTrajectory] = self.jump_calculator.calculate_jump(
                    nav_node1.position, 
                    nav_node2.position, 
                    nav_node1.surface.type
                )
                
                if trajectory:
                    # Add edge to graph with trajectory data
                    self.nav_graph.add_edge(
                        node1_id, node2_id, 
                        weight=trajectory.total_frames, # Cost could be time or energy
                        move_type=trajectory.jump_type, 
                        trajectory_data=trajectory,
                        frames=trajectory.total_frames # For dynamic pathfinder
                    )
                    num_potential_jumps_added +=1
        print(f"Added {num_potential_jumps_added} potential jump/fall edges to the graph.")

    def _find_nearest_node(self, world_pos: Tuple[float, float]) -> Optional[int]:
        """Find the nearest navigation graph node ID to a given world position."""
        if not self.nav_graph or not self.nav_graph.nodes:
            print("Warning: Nav graph is empty or not initialized in _find_nearest_node.")
            return None

        # Simple search: iterate all nodes. For performance, use spatial index (e.g. k-d tree or grid)
        # The NavigationGraphBuilder has nodes_by_position, but that's for builder internal use.
        # Here, we access the graph directly.
        min_dist_sq = float('inf')
        nearest_node_id = None
        
        for node_id, data in self.nav_graph.nodes(data=True):
            node_world_pos = data['position']
            dist_sq = (world_pos[0] - node_world_pos[0])**2 + (world_pos[1] - node_world_pos[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_node_id = node_id
        
        if nearest_node_id is not None:
            # Optional: Check if this node is actually reachable/valid from world_pos
            # (e.g. line of sight, on same surface if close enough)
            pass

        # print(f"Nearest node to {world_pos} is {nearest_node_id} (dist_sq={min_dist_sq:.2f})")
        return nearest_node_id

    def _get_edge_types(self, node_path_ids: List[int]) -> List[str]:
        """Get the movement types for each segment in a node path."""
        edge_types: List[str] = []
        if not node_path_ids or len(node_path_ids) < 2 or not self.nav_graph:
            return []
        
        for i in range(len(node_path_ids) - 1):
            u, v = node_path_ids[i], node_path_ids[i+1]
            if self.nav_graph.has_edge(u,v):
                edge_data = self.nav_graph.get_edge_data(u,v)
                edge_types.append(edge_data.get('move_type', 'walk')) # Default to walk
            else:
                print(f"Warning: No edge between {u} and {v} in _get_edge_types. Defaulting to 'walk'.")
                edge_types.append('walk') # Should not happen for a valid path
        return edge_types

    def find_path_to_exit(self, start_pos: Tuple[float, float],
                         switch_pos: Optional[Tuple[float, float]], # Switch can be optional
                         exit_pos: Tuple[float, float],
                         current_time: int = 0, # For dynamic pathfinding
                         enemies_list: Optional[List[Enemy]] = None, # For dynamic pathfinding
                         collect_gold_positions: Optional[List[Tuple[float, float]]] = None
                         ) -> Optional[List[dict]]: # Returns list of movement commands
        """Find complete path from start to exit, potentially via switch, avoiding dynamic obstacles."""
        
        start_node_id = self._find_nearest_node(start_pos)
        switch_node_id = self._find_nearest_node(switch_pos) if switch_pos else None
        exit_node_id = self._find_nearest_node(exit_pos)
        
        gold_node_ids: Optional[List[int]] = None
        if collect_gold_positions:
            gold_node_ids = []
            for gold_pos in collect_gold_positions:
                gn_id = self._find_nearest_node(gold_pos)
                if gn_id is not None: gold_node_ids.append(gn_id)
            if not gold_node_ids: gold_node_ids = None # Ensure it's None if no valid gold nodes found

        if start_node_id is None or exit_node_id is None:
            print("Error: Could not find nearest nodes for start or exit position.")
            return None
        if switch_pos and switch_node_id is None:
            print("Error: Switch position provided, but could not find nearest switch node.")
            return None

        node_path_ids: Optional[List[int]] = None
        temporal_path: Optional[List[Tuple[int, int]]] = None

        if enemies_list and len(enemies_list) > 0:
            print("Dynamic pathfinding mode with enemies.")
            self.update_with_enemies(enemies_list)
            if self.dynamic_pathfinder:
                # Dynamic pathfinder needs to handle multi-objective (switch, exit) itself, or be called sequentially.
                # For simplicity, let's assume dynamic pathfinder for now only does start -> goal.
                # A full multi-objective dynamic path is more complex.
                # Path to switch (if any), then path to exit.
                current_dyn_start_node = start_node_id
                current_dyn_start_time = current_time
                full_temporal_path_segments = []

                if switch_node_id is not None:
                    path_to_switch_dyn = self.dynamic_pathfinder.find_safe_path(current_dyn_start_node, switch_node_id, current_dyn_start_time)
                    if not path_to_switch_dyn:
                        print(f"Dynamic path to switch {switch_node_id} failed.")
                        return None
                    full_temporal_path_segments.extend(path_to_switch_dyn)
                    current_dyn_start_node = path_to_switch_dyn[-1][0] # Spatial node of last step
                    current_dyn_start_time = path_to_switch_dyn[-1][1] # Time of last step
                
                path_to_exit_dyn = self.dynamic_pathfinder.find_safe_path(current_dyn_start_node, exit_node_id, current_dyn_start_time)
                if not path_to_exit_dyn:
                    print(f"Dynamic path to exit {exit_node_id} failed.")
                    return None
                # If there was a switch path, avoid duplicating the common node
                full_temporal_path_segments.extend(path_to_exit_dyn[1:] if switch_node_id is not None and len(path_to_exit_dyn)>0 else path_to_exit_dyn)
                
                temporal_path = full_temporal_path_segments
                node_path_ids = [step[0] for step in temporal_path] # Extract spatial nodes for smoothing/commands
            else:
                print("Error: Enemies present but dynamic pathfinder not initialized.")
                return None
        else:
            print("Static pathfinding mode.")
            node_path_ids = self.multi_objective_pathfinder.find_complete_level_path(
                start_node_id, switch_node_id, exit_node_id, gold_node_ids
            )

        if not node_path_ids or len(node_path_ids) < 1:
            print("Pathfinding failed to find a node path.")
            return None
        
        # Smooth the path (list of node IDs to list of world coordinates)
        world_path_waypoints = self.path_optimizer.smooth_path(node_path_ids, self.nav_graph)
        if not world_path_waypoints or len(world_path_waypoints) < 1:
            # If smoothing returns empty (e.g. if node_path_ids was just one node), use original node positions
            if len(node_path_ids) == 1:
                world_path_waypoints = [self.nav_graph.nodes[node_path_ids[0]]['position']]
            else:
                print("Path smoothing failed or resulted in an empty path.")
                # Fallback: use raw node positions if smoothing fails badly
                world_path_waypoints = [self.nav_graph.nodes[nid]['position'] for nid in node_path_ids if nid in self.nav_graph.nodes]
                if not world_path_waypoints:
                     return None
        
        # Get edge types for the original node path (before smoothing)
        # This helps MovementController decide how to traverse segments.
        edge_movement_types = self._get_edge_types(node_path_ids)
        
        # Generate movement commands
        # Initial velocity is (0,0) for simplicity here. A real agent would have current velocity.
        commands = self.movement_controller.generate_commands(
            start_pos, (0,0), world_path_waypoints, edge_movement_types
        )
        
        if not commands:
            print("Movement command generation failed.")
            return None
            
        # print(f"Successfully generated {len(commands)} commands.")
        return commands

    def find_simple_path(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float]) -> Optional[List[dict]]:
        """Simplified pathfinding from start_pos to end_pos, no objectives, no enemies."""
        start_node = self._find_nearest_node(start_pos)
        end_node = self._find_nearest_node(end_pos)

        if start_node is None or end_node is None:
            print("Could not find start or end node for simple path.")
            return None

        node_path = self.pathfinder.find_path(start_node, end_node, avoid_hazards=True)
        if not node_path:
            print(f"Simple A* path from {start_node} to {end_node} failed.")
            return None
        
        world_path = self.path_optimizer.smooth_path(node_path, self.nav_graph)
        if not world_path:
            world_path = [self.nav_graph.nodes[nid]['position'] for nid in node_path if nid in self.nav_graph.nodes]
            if not world_path: return None

        edge_types = self._get_edge_types(node_path)
        commands = self.movement_controller.generate_commands(start_pos, (0,0), world_path, edge_types)
        return commands

    def update_with_enemies(self, enemies_list: List[Enemy]):
        """Update pathfinding system with current enemy information for dynamic avoidance."""
        self.enemies = enemies_list
        if not enemies_list:
            self.enemy_predictor = None
            self.dynamic_pathfinder = None
            print("Enemies list cleared, reverting to static pathfinding if applicable.")
            return

        self.enemy_predictor = EnemyPredictor(self.enemies)
        if self.nav_graph and self.enemy_predictor:
            self.dynamic_pathfinder = DynamicPathfinder(self.nav_graph, self.enemy_predictor)
            print(f"DynamicPathfinder updated with {len(enemies_list)} enemies.")
        else:
            print("Warning: Could not initialize DynamicPathfinder (nav_graph or enemy_predictor missing).")
    
    def visualize_current_state(self, screen, 
                                current_path_nodes: Optional[List[int]] = None,
                                current_path_world: Optional[List[Tuple[float,float]]] = None,
                                current_enemies_info: Optional[List[Tuple[Tuple[float,float], float]]] = None):
        """Render the navigation mesh, current path, enemies, etc., for debugging."""
        if not self.visualizer or not hasattr(self.visualizer, 'PYGAME_AVAILABLE') or not self.visualizer.PYGAME_AVAILABLE:
            return
        if screen is None: return

        self.visualizer.clear_screen(screen)
        self.visualizer.draw_surfaces(screen, self.surfaces)
        self.visualizer.draw_nav_graph(screen, self.nav_graph)

        if current_path_nodes:
            self.visualizer.draw_path(screen, current_path_nodes, self.nav_graph, color=self.visualizer.colors['path'])
        
        if current_path_world:
            # Could draw this as a series of lines if it's different from node path (e.g. smoothed)
            if len(current_path_world) >= 2:
                for i in range(len(current_path_world) - 1):
                    pygame.draw.line(screen, pygame.Color('deeppink'), 
                                   current_path_world[i], current_path_world[i+1], 2)
        
        if current_enemies_info:
            self.visualizer.draw_enemies(screen, current_enemies_info)
        elif self.enemy_predictor: # Draw predicted enemies at t=0 if no specific info given
            initial_enemy_pos = self.enemy_predictor.get_enemy_positions_at_time(0)
            self.visualizer.draw_enemies(screen, initial_enemy_pos)

        self.visualizer.update_display()
