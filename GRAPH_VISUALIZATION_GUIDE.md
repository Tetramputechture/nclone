# Graph Visualization System Guide

This guide explains how to use the comprehensive graph visualization system for N++ level analysis, including realtime visualization of graph properties and shortest path trajectories.

## Overview

The graph visualization system provides two main visualization modes:

1. **Overlay Mode**: Visualizes graph data as an overlay on the simulator screen
2. **Standalone Mode**: Independent graph rendering for detailed analysis
3. **Side-by-Side Mode**: Both overlay and standalone views simultaneously

## Quick Start

### Basic Usage

```python
from nclone.graph.visualization import GraphVisualizer, VisualizationConfig
from nclone.graph.graph_construction import GraphBuilder

# Create graph from level data
builder = GraphBuilder()
graph_data = builder.build_graph(level_data, entities)

# Create visualizer with default settings
visualizer = GraphVisualizer()

# Generate standalone visualization
surface = visualizer.create_standalone_visualization(
    graph_data,
    width=1200,
    height=800,
    goal_position=(100, 200),
    start_position=(50, 300)
)

# Save or display the surface
pygame.image.save(surface, "graph_visualization.png")
```

### Interactive Visualization

```python
from nclone.graph.visualization import InteractiveGraphVisualizer

# Create interactive visualizer
interactive = InteractiveGraphVisualizer(width=1200, height=800)

# Run interactive session
interactive.run(graph_data)
```

## Visualization Features

### Graph Components

The system visualizes all major graph components:

- **Nodes**: Different colors for different node types (grid cells, entities, ninja)
- **Edges**: Color-coded by movement type (walk, jump, fall, wall slide, etc.)
- **Paths**: Highlighted shortest paths with trajectory visualization
- **Physics Properties**: Energy costs, success probabilities, timing data

### Node Types and Colors

- **Grid Cells**: Light blue/gray - navigable floor positions
- **Entities**: Red - interactive game objects (switches, doors, etc.)
- **Ninja**: Cyan - player start position

### Edge Types and Colors

- **Walk**: Green - ground-based movement
- **Jump**: Orange - jumping movement
- **Fall**: Blue - falling movement  
- **Wall Slide**: Purple - wall sliding movement
- **One Way**: Gray - directional movement restrictions
- **Functional**: Yellow - entity interaction edges

## Configuration Options

### VisualizationConfig

```python
from nclone.graph.visualization import VisualizationConfig

config = VisualizationConfig(
    # Display options
    show_nodes=True,
    show_edges=True,
    show_path=True,
    show_labels=True,
    
    # Edge filtering
    show_walk_edges=True,
    show_jump_edges=True,
    show_fall_edges=True,
    show_wall_slide_edges=True,
    show_one_way_edges=True,
    show_functional_edges=True,
    
    # Visual settings
    node_size=3.0,
    edge_width=1.0,
    path_width=3.0,
    alpha=0.8,
    
    # Colors (from constants)
    background_color=(20, 20, 30),
    grid_color=(60, 60, 70),
    text_color=(255, 255, 255)
)

visualizer = GraphVisualizer(config)
```

### Customizing Colors and Styles

Colors and visual constants are defined in `nclone/graph/constants.py`:

```python
from nclone.graph.constants import ColorScheme, VisualizationDefaults

# Access predefined colors
node_colors = ColorScheme.NODE_COLORS
edge_colors = ColorScheme.EDGE_COLORS
path_color = ColorScheme.PATH_COLOR

# Access size constants
node_size = VisualizationDefaults.NODE_SIZE
font_size = VisualizationDefaults.MEDIUM_FONT_SIZE
```

## Pathfinding Integration

### Visualizing Shortest Paths

```python
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm

# Create pathfinding engine
pathfinder = PathfindingEngine(level_data, entities)

# Find shortest path using A* (fast, for real-time visualization)
start_pos = (50, 300)
goal_pos = (100, 200)
result = pathfinder.find_shortest_path(
    graph_data,
    start_node_idx, 
    goal_node_idx, 
    algorithm=PathfindingAlgorithm.A_STAR
)

# For comprehensive analysis, use Dijkstra (slower but guaranteed optimal)
optimal_result = pathfinder.find_shortest_path(
    graph_data,
    start_node_idx, 
    goal_node_idx, 
    algorithm=PathfindingAlgorithm.DIJKSTRA
)

# Visualize with path highlighted
surface = visualizer.create_standalone_visualization(
    graph_data,
    goal_position=goal_pos,
    start_position=start_pos
)
```

### Algorithm Selection for Pathfinding

The system provides both A* and Dijkstra algorithms for different use cases:

**A* Algorithm (Default)**:
- **Use for**: Real-time visualization, RL training, interactive features
- **Advantages**: 10-100x faster, lower memory usage
- **Best when**: Single goal, standard level geometry, performance critical

**Dijkstra Algorithm**:
- **Use for**: Level analysis, complex levels, validation
- **Advantages**: Guaranteed optimal paths, handles complex graph structures
- **Best when**: Multiple goals, switches/teleporters, accuracy critical

```python
# Real-time pathfinding for interactive visualization
fast_path = pathfinder.find_shortest_path(
    graph_data, start, goal, 
    algorithm=PathfindingAlgorithm.A_STAR
)

# Comprehensive analysis for level design validation
optimal_path = pathfinder.find_shortest_path(
    graph_data, start, goal, 
    algorithm=PathfindingAlgorithm.DIJKSTRA
)
```

### Physics-Informed Pathfinding

The system uses actual N++ physics constants for accurate pathfinding:

- **Jump trajectories**: Calculated using real gravity and velocity values
- **Energy costs**: Based on movement type and distance
- **Success probabilities**: Factoring in difficulty and precision requirements
- **Timing constraints**: Considering acceleration and maximum speeds

## Overlay Mode

### Basic Overlay

```python
# Create overlay surface
overlay_surface = visualizer.create_overlay(
    graph_data,
    simulator_width=800,
    simulator_height=600,
    camera_offset=(0, 0),
    zoom_level=1.0
)

# Blend with simulator surface
simulator_surface.blit(overlay_surface, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)
```

### Real-time Updates

```python
class GameLoop:
    def __init__(self):
        self.visualizer = GraphVisualizer()
        self.pathfinder = PathfindingEngine(graph_data, level_data, entities)
    
    def update(self, ninja_pos, goal_pos):
        # Update pathfinding in real-time
        path_result = self.pathfinder.find_path(ninja_pos, goal_pos)
        
        # Create updated overlay
        overlay = self.visualizer.create_overlay(
            graph_data,
            simulator_width=800,
            simulator_height=600,
            current_path=path_result.path if path_result.found else None
        )
        
        return overlay
```

## Advanced Features

### Hierarchical Graph Visualization

```python
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder

# Build hierarchical graph
hierarchical_builder = HierarchicalGraphBuilder()
hierarchical_data = hierarchical_builder.build_hierarchical_graph(level_data, entities)

# Visualize with hierarchy levels
surface = visualizer.create_standalone_visualization(
    hierarchical_data,
    show_hierarchy_levels=True
)
```

### Debug Information

The system provides comprehensive debug overlays:

- **Node information**: Position, type, connections
- **Edge details**: Movement type, cost, physics parameters
- **Path analysis**: Total cost, estimated time, success probability
- **Performance metrics**: Graph size, pathfinding time

### Interactive Controls

When using `InteractiveGraphVisualizer`:

- **Mouse**: Click to set start/goal positions
- **Keyboard**:
  - `SPACE`: Toggle pathfinding
  - `R`: Reset positions
  - `1-6`: Toggle different edge types
  - `ESC`: Exit

## Performance Considerations

### Large Graphs

For levels with large graphs (>1000 nodes):

```python
config = VisualizationConfig(
    # Reduce visual complexity
    show_labels=False,
    node_size=2.0,
    edge_width=0.5,
    
    # Filter edge types
    show_fall_edges=False,  # Often numerous
    show_one_way_edges=False
)
```

### Real-time Updates

For smooth real-time visualization:

- Cache graph surfaces when possible
- Update only changed regions
- Use appropriate zoom levels
- Consider level-of-detail rendering

## Integration Examples

### With Pygame Simulator

```python
class NPPSimulator:
    def __init__(self):
        self.graph_visualizer = GraphVisualizer()
        self.show_graph_overlay = True
    
    def render(self):
        # Render game world
        self.screen.blit(self.world_surface, (0, 0))
        
        # Add graph overlay if enabled
        if self.show_graph_overlay:
            overlay = self.graph_visualizer.create_overlay(
                self.graph_data,
                self.screen.get_width(),
                self.screen.get_height(),
                self.camera.offset,
                self.camera.zoom
            )
            self.screen.blit(overlay, (0, 0))
```

### With RL Training

```python
class RLEnvironment:
    def __init__(self):
        self.graph_visualizer = GraphVisualizer()
    
    def render_debug(self, mode='human'):
        if mode == 'graph':
            return self.graph_visualizer.create_standalone_visualization(
                self.graph_data,
                goal_position=self.goal_pos,
                start_position=self.ninja_pos
            )
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure pygame is installed: `pip install pygame`
2. **Performance Issues**: Reduce graph complexity or use filtering
3. **Memory Usage**: Clear cached surfaces periodically
4. **Coordinate Misalignment**: Check camera offset and zoom calculations

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed pathfinding information will be logged
```

## API Reference

### Core Classes

- `GraphVisualizer`: Main visualization class
- `VisualizationConfig`: Configuration options
- `InteractiveGraphVisualizer`: Interactive visualization
- `PathfindingEngine`: Physics-informed pathfinding

### Key Methods

- `create_standalone_visualization()`: Generate independent graph view
- `create_overlay()`: Generate simulator overlay
- `find_path()`: Calculate shortest traversable path
- `run()`: Start interactive visualization session

For complete API documentation, see the docstrings in each module.