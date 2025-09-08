#!/usr/bin/env python3
"""
Standalone pathfinding visualization tool for N++ level analysis.

This tool provides an interactive interface for visualizing pathfinding
to different entity types, with comprehensive debugging information.

Usage:
    python standalone_pathfinding_viz.py --map doortest
    python standalone_pathfinding_viz.py --map custom_level.txt --entity-type locked_door_switch
"""

import pygame
import sys
import argparse
import os
from typing import Optional, List

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from nclone_environments.basic_level_no_gold import BasicLevelNoGold
from graph.hierarchical_builder import HierarchicalGraphBuilder
from graph.pathfinding_visualizer import PathfindingVisualizer, PathfindingVisualizationMode
from graph.visualization import VisualizationConfig
from graph.pathfinding import PathfindingAlgorithm
from constants.entity_types import EntityType


class StandalonePathfindingVisualizer:
    """Standalone pathfinding visualization application."""
    
    def __init__(self, map_path: str, target_entity_type: Optional[EntityType] = None):
        """Initialize the standalone visualizer."""
        self.map_path = map_path
        self.target_entity_type = target_entity_type or EntityType.LOCKED_DOOR_SWITCH
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Screen setup
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"Pathfinding Visualizer - {os.path.basename(map_path)}")
        
        # Initialize components
        self.env = None
        self.graph_data = None
        self.ninja_position = None
        self.pathfinding_viz = None
        self.current_algorithm = PathfindingAlgorithm.A_STAR
        
        # UI state
        self.show_help = True
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Available entity types for cycling
        self.available_entity_types = []
        self.current_entity_index = 0
        
        self.clock = pygame.time.Clock()
        
    def initialize_environment(self) -> bool:
        """Initialize the environment and graph data."""
        try:
            print(f"Loading map: {self.map_path}")
            
            # Initialize environment
            self.env = BasicLevelNoGold(custom_map_path=self.map_path)
            self.env.reset()
            
            # Get ninja position
            self.ninja_position = (
                float(self.env.nplay_headless.ninja_x),
                float(self.env.nplay_headless.ninja_y)
            )
            print(f"Ninja position: {self.ninja_position}")
            
            # Build graph
            print("Building hierarchical graph...")
            graph_builder = HierarchicalGraphBuilder(debug=False)
            level_data = getattr(self.env, "level_data", None)
            
            if not level_data:
                print("❌ No level data available")
                return False
                
            hierarchical_data = graph_builder.build_graph(level_data, self.ninja_position)
            self.graph_data = hierarchical_data.sub_cell_graph
            print(f"✅ Graph built: {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")
            
            # Initialize pathfinding visualizer
            config = VisualizationConfig()
            config.show_nodes = True
            config.show_edges = True
            config.node_size = 3.0
            config.edge_width = 1.0
            config.path_width = 4.0
            
            self.pathfinding_viz = PathfindingVisualizer(config)
            
            # Get available entity types
            self.available_entity_types = self.pathfinding_viz.get_available_entity_types(self.graph_data)
            print(f"Available entity types: {[et.name for et in self.available_entity_types]}")
            
            # Set initial target if available
            if self.target_entity_type in self.available_entity_types:
                self.current_entity_index = self.available_entity_types.index(self.target_entity_type)
            elif self.available_entity_types:
                self.current_entity_index = 0
                self.target_entity_type = self.available_entity_types[0]
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize environment: {e}")
            return False
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
                    
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                    
                elif event.key == pygame.K_SPACE:
                    # Cycle through entity types
                    if self.available_entity_types:
                        self.current_entity_index = (self.current_entity_index + 1) % len(self.available_entity_types)
                        self.target_entity_type = self.available_entity_types[self.current_entity_index]
                        print(f"Switched to target: {self.target_entity_type.name}")
                        
                elif event.key == pygame.K_a:
                    # Toggle algorithm
                    if self.current_algorithm == PathfindingAlgorithm.A_STAR:
                        self.current_algorithm = PathfindingAlgorithm.DIJKSTRA
                    else:
                        self.current_algorithm = PathfindingAlgorithm.A_STAR
                    print(f"Switched to algorithm: {self.current_algorithm.name}")
                    
                elif event.key == pygame.K_m:
                    # Show multiple paths
                    print("Showing paths to all available entity types...")
                    
                elif event.key == pygame.K_c:
                    # Clear paths
                    if self.pathfinding_viz:
                        self.pathfinding_viz.clear_paths()
                        
                elif event.key == pygame.K_i:
                    # Toggle path info
                    if self.pathfinding_viz:
                        self.pathfinding_viz.toggle_path_info()
                        
                elif event.key == pygame.K_l:
                    # Toggle entity labels
                    if self.pathfinding_viz:
                        self.pathfinding_viz.toggle_entity_labels()
                        
                elif event.key == pygame.K_d:
                    # Toggle distance info
                    if self.pathfinding_viz:
                        self.pathfinding_viz.toggle_distance_info()
        
        return True
    
    def render_frame(self):
        """Render a single frame."""
        self.screen.fill((40, 40, 40))
        
        if not self.graph_data or not self.pathfinding_viz:
            self._draw_loading_message()
            return
        
        try:
            # Visualize pathfinding to current target
            if self.target_entity_type:
                result = self.pathfinding_viz.visualize_path_to_entity(
                    self.screen,
                    self.graph_data,
                    self.ninja_position,
                    self.target_entity_type,
                    self.current_algorithm
                )
                
                if result and not result.success:
                    self._draw_error_overlay(result.error_message)
            
        except Exception as e:
            self._draw_error_overlay(f"Visualization error: {e}")
        
        # Draw UI overlays
        self._draw_status_bar()
        
        if self.show_help:
            self._draw_help_overlay()
    
    def _draw_loading_message(self):
        """Draw loading message."""
        text = self.font.render("Loading...", True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(text, text_rect)
    
    def _draw_error_overlay(self, message: str):
        """Draw error message overlay."""
        if not message:
            return
            
        # Error panel
        panel_width = min(600, self.screen_width - 40)
        panel_height = 100
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2
        
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((255, 0, 0, 100))
        pygame.draw.rect(panel_surface, (255, 0, 0, 255), (0, 0, panel_width, panel_height), 3)
        
        # Error text
        lines = message.split('\n')
        y_offset = 20
        
        for line in lines:
            text_surface = self.font.render(line, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(panel_width // 2, y_offset))
            panel_surface.blit(text_surface, text_rect)
            y_offset += 30
        
        self.screen.blit(panel_surface, (panel_x, panel_y))
    
    def _draw_status_bar(self):
        """Draw status bar with current settings."""
        # Status bar background
        status_height = 60
        status_surface = pygame.Surface((self.screen_width, status_height), pygame.SRCALPHA)
        status_surface.fill((0, 0, 0, 150))
        
        # Status text
        y_offset = 5
        
        # Current target
        if self.target_entity_type:
            target_text = f"Target: {self.target_entity_type.name} ({self.current_entity_index + 1}/{len(self.available_entity_types)})"
            text_surface = self.font.render(target_text, True, (255, 255, 255))
            status_surface.blit(text_surface, (10, y_offset))
        
        # Algorithm
        algo_text = f"Algorithm: {self.current_algorithm.name}"
        text_surface = self.small_font.render(algo_text, True, (200, 200, 200))
        status_surface.blit(text_surface, (10, y_offset + 25))
        
        # Controls hint
        if not self.show_help:
            hint_text = "Press H for help"
            text_surface = self.small_font.render(hint_text, True, (150, 150, 150))
            text_rect = text_surface.get_rect()
            text_rect.right = self.screen_width - 10
            text_rect.y = y_offset + 25
            status_surface.blit(text_surface, text_rect)
        
        self.screen.blit(status_surface, (0, 0))
    
    def _draw_help_overlay(self):
        """Draw help overlay with controls."""
        # Help panel
        panel_width = 400
        panel_height = 350
        panel_x = self.screen_width - panel_width - 20
        panel_y = 80
        
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 200))
        pygame.draw.rect(panel_surface, (255, 255, 255, 255), (0, 0, panel_width, panel_height), 2)
        
        # Help text
        help_lines = [
            "PATHFINDING VISUALIZER CONTROLS",
            "",
            "SPACE - Cycle through entity types",
            "A - Toggle algorithm (A*/Dijkstra)",
            "M - Show paths to all entities",
            "C - Clear current paths",
            "",
            "I - Toggle path information",
            "L - Toggle entity labels", 
            "D - Toggle distance markers",
            "",
            "H - Toggle this help",
            "ESC/Q - Quit",
            "",
            "LEGEND:",
            "Green circle - Ninja (start)",
            "Colored circles - Target entities",
            "Colored lines - Paths"
        ]
        
        y_offset = 10
        for line in help_lines:
            if line == "PATHFINDING VISUALIZER CONTROLS":
                text_surface = self.font.render(line, True, (255, 255, 0))
            elif line == "LEGEND:":
                text_surface = self.font.render(line, True, (255, 255, 0))
            elif line == "":
                y_offset += 10
                continue
            else:
                text_surface = self.small_font.render(line, True, (255, 255, 255))
            
            panel_surface.blit(text_surface, (10, y_offset))
            y_offset += 20
        
        self.screen.blit(panel_surface, (panel_x, panel_y))
    
    def run(self) -> bool:
        """Run the visualization application."""
        if not self.initialize_environment():
            return False
        
        print("\n🎮 Pathfinding Visualizer Started!")
        print("Press H for help, ESC to quit")
        
        running = True
        while running:
            running = self.handle_events()
            
            self.render_frame()
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Standalone pathfinding visualization tool")
    parser.add_argument("--map", required=True, help="Map file to load")
    parser.add_argument("--entity-type", 
                       choices=['locked_door_switch', 'trap_door_switch', 'exit_switch', 'exit'],
                       default='locked_door_switch',
                       help="Initial target entity type")
    
    args = parser.parse_args()
    
    # Convert entity type string to enum
    entity_type_map = {
        'locked_door_switch': EntityType.LOCKED_DOOR_SWITCH,
        'trap_door_switch': EntityType.TRAP_DOOR_SWITCH,
        'exit_switch': EntityType.EXIT_SWITCH,
        'exit': EntityType.EXIT
    }
    
    target_entity_type = entity_type_map.get(args.entity_type, EntityType.LOCKED_DOOR_SWITCH)
    
    # Create and run visualizer
    visualizer = StandalonePathfindingVisualizer(args.map, target_entity_type)
    success = visualizer.run()
    
    if success:
        print("✅ Pathfinding visualization completed successfully")
        return 0
    else:
        print("❌ Pathfinding visualization failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())