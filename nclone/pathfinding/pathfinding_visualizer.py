import networkx as nx
from typing import List, Tuple, Dict, Optional

# Conditional import of pygame for visualization
PYGAME_AVAILABLE = False
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    print("Warning: Pygame module not found. PathfindingVisualizer will not be functional.")

from .surface_parser import Surface # For drawing surfaces

class PathfindingVisualizer:
    """Visualization tools for debugging pathfinding components."""
    
    def __init__(self):
        if not PYGAME_AVAILABLE:
            self.colors = {} # No colors needed if pygame is not there
            return

        self.colors = {
            'surface_floor': pygame.Color('blue'),
            'surface_wall': pygame.Color('brown'),
            'surface_slope': pygame.Color('green'),
            'surface_ceiling': pygame.Color('cyan'),
            'node': pygame.Color('red'),
            'edge': pygame.Color('lightgreen'),
            'path': pygame.Color('yellow'),
            'jump_trajectory': pygame.Color('orange'),
            'los_line': pygame.Color('magenta'),
            'enemy': pygame.Color('purple')
        }
    
    def draw_surfaces(self, screen, surfaces: List[Surface]):
        """Render parsed surfaces onto the pygame screen."""
        if not PYGAME_AVAILABLE or not screen:
            # print("Pygame not available or screen not provided to draw_surfaces")
            return
        
        for surface in surfaces:
            if not surface.tiles: continue
            
            color = self.colors.get(f'surface_{surface.type.name.lower()}', pygame.Color('gray'))
            
            # Draw each tile of the surface
            # Assuming tiles are 24x24 as per Surface class comments
            tile_size = 24 
            for tile_x, tile_y in surface.tiles:
                # Convert grid coords to world/screen coords
                rect = pygame.Rect(tile_x * tile_size, tile_y * tile_size, tile_size, tile_size)
                pygame.draw.rect(screen, color, rect, 1) # Draw outline of tile
            
            # Optionally, draw surface normal or start/end points
            if surface.start_pos and surface.end_pos:
                pygame.draw.line(screen, color, surface.start_pos, surface.end_pos, 3)
                if surface.normal:
                    mid_x = (surface.start_pos[0] + surface.end_pos[0]) / 2
                    mid_y = (surface.start_pos[1] + surface.end_pos[1]) / 2
                    norm_end_x = mid_x + surface.normal[0] * 20 # Scale normal for visibility
                    norm_end_y = mid_y + surface.normal[1] * 20
                    pygame.draw.line(screen, pygame.Color('pink'), (mid_x, mid_y), (norm_end_x, norm_end_y), 1)

    def draw_nav_graph(self, screen, graph: nx.DiGraph, draw_nodes=True, draw_edges=True):
        """Render the navigation graph (nodes and edges) onto the pygame screen."""
        if not PYGAME_AVAILABLE or not screen or not graph:
            return

        if draw_edges:
            for u_id, v_id, edge_data in graph.edges(data=True):
                if u_id in graph.nodes and v_id in graph.nodes:
                    start_pos = graph.nodes[u_id]['position']
                    end_pos = graph.nodes[v_id]['position']
                    edge_color = self.colors['edge']
                    move_type = edge_data.get('move_type')
                    if move_type == 'jump':
                        edge_color = self.colors['jump_trajectory']
                    elif move_type == 'wall_jump':
                        edge_color = pygame.Color('darkorange')
                    
                    # Convert to int for pygame drawing functions
                    start_pos_int = (int(start_pos[0]), int(start_pos[1]))
                    end_pos_int = (int(end_pos[0]), int(end_pos[1]))
                    pygame.draw.line(screen, edge_color, start_pos_int, end_pos_int, 1)
        
        if draw_nodes:
            for node_id, node_data in graph.nodes(data=True):
                pos = node_data['position']
                node_color = self.colors['node']
                node_type = node_data.get('node_type')
                if node_type == 'edge_start' or node_type == 'edge_end':
                    node_color = pygame.Color('crimson')
                
                pygame.draw.circle(screen, node_color, (int(pos[0]), int(pos[1])), 3) # Smaller radius

    def draw_path(self, screen, path_node_ids: List[int], graph: nx.DiGraph, color=None):
        """Draw a specific path (list of node IDs) on the screen."""
        if not PYGAME_AVAILABLE or not screen or not path_node_ids or not graph:
            return
        
        path_color = color if color else self.colors['path']
        if len(path_node_ids) < 2: return

        for i in range(len(path_node_ids) - 1):
            node1_id = path_node_ids[i]
            node2_id = path_node_ids[i+1]
            if node1_id in graph.nodes and node2_id in graph.nodes:
                pos1 = graph.nodes[node1_id]['position']
                pos2 = graph.nodes[node2_id]['position']
                pygame.draw.line(screen, path_color, 
                               (int(pos1[0]), int(pos1[1])),
                               (int(pos2[0]), int(pos2[1])), 3) # Thicker line for path

    def draw_jump_trajectory(self, screen, trajectory_frames: List[Tuple[Tuple[float,float], Tuple[float,float]]]):
        """Draw a calculated jump trajectory (list of positions)."""
        if not PYGAME_AVAILABLE or not screen or not trajectory_frames or len(trajectory_frames) < 2:
            return
        
        for i in range(len(trajectory_frames) - 1):
            pos1 = trajectory_frames[i][0] # Position is the first element of the tuple
            pos2 = trajectory_frames[i+1][0]
            pygame.draw.line(screen, self.colors['jump_trajectory'], 
                               (int(pos1[0]), int(pos1[1])),
                               (int(pos2[0]), int(pos2[1])), 2)

    def draw_los_lines(self, screen, los_path: List[Tuple[float,float]]):
        """Draw lines for a line-of-sight smoothed path."""
        if not PYGAME_AVAILABLE or not screen or not los_path or len(los_path) < 2:
            return
        for i in range(len(los_path) - 1):
            pygame.draw.line(screen, self.colors['los_line'], los_path[i], los_path[i+1], 2)

    def draw_enemies(self, screen, enemy_positions: List[Tuple[Tuple[float,float], float]]):
        """Draw enemies at their current positions with their radii."""
        if not PYGAME_AVAILABLE or not screen or not enemy_positions:
            return
        for pos, radius in enemy_positions:
            pygame.draw.circle(screen, self.colors['enemy'], (int(pos[0]), int(pos[1])), int(radius), 2) # Outline
            pygame.draw.circle(screen, self.colors['enemy'], (int(pos[0]), int(pos[1])), int(radius/2)) # Filled smaller circle

    def update_display(self):
        """Call pygame.display.flip() or pygame.display.update()."""
        if PYGAME_AVAILABLE:
            pygame.display.flip()

    def clear_screen(self, screen, color=(0,0,0)):
        """Fill the screen with a color (typically black)."""
        if PYGAME_AVAILABLE and screen:
            screen.fill(pygame.Color(color) if isinstance(color, tuple) else color)
