# N++ Graph Visualization System

This guide explains how to use the comprehensive graph visualization system for N++ levels, including pathfinding, trajectory analysis, and real-time visualization capabilities.

## Overview

The graph visualization system provides:

- **Standalone Graph Rendering**: Independent visualization of level graphs
- **Simulator Overlay**: Real-time path visualization on the game simulator
- **Side-by-Side View**: Combined simulator and graph visualization
- **Physics-Accurate Pathfinding**: Uses the same movement classification and trajectory calculation as the RL training system
- **Interactive Controls**: Real-time adjustment of visualization parameters

## Quick Start

### Basic Level Graph Visualization

```python
from nclone.graph.visualization_api import visualize_level_graph

# Simple level visualization
success = visualize_level_graph(
    level_data=your_level_data,
    entities=your_entities,
    output_path="level_graph.png",
    size=(1200, 800)
)
```

### Pathfinding with Visualization

```python
from nclone.graph.visualization_api import find_path_and_visualize

# Find path and create visualization
success, path_result = find_path_and_visualize(
    level_data=your_level_data,
    entities=your_entities,
    start_position=(100.0, 300.0),
    goal_position=(400.0, 200.0),
    output_path="path_visualization.png"
)

if success and path_result.success:
    print(f"Path found with {len(path_result.path)} nodes")
    print(f"Total cost: {path_result.total_cost:.2f}")
```

## Advanced Usage

### Using the Unified API

```python
from nclone.graph.visualization_api import GraphVisualizationAPI, VisualizationRequest, VisualizationMode

# Initialize API
api = GraphVisualizationAPI()

# Create detailed visualization request
request = VisualizationRequest(
    level_data=level_data,
    entities=entities,
    ninja_position=(50.0, 280.0),
    ninja_velocity=(5.0, 0.0),
    ninja_state=1,  # Running state
    goal_position=(400.0, 280.0),
    mode=VisualizationMode.SIDE_BY_SIDE,
    use_hierarchical=True,
    output_size=(1600, 800)
)

# Generate visualization
result = api.visualize_graph(request)

if result.success:
    # Save the visualization
    pygame.image.save(result.surface, "advanced_visualization.png")
    
    # Print statistics
    print("Graph Statistics:")
    for key, value in result.graph_stats.items():
        print(f"  {key}: {value}")
```

### Simulator Integration

```python
from nclone.graph.enhanced_debug_overlay import EnhancedDebugOverlay, OverlayMode

# Create enhanced overlay for simulator
overlay = EnhancedDebugOverlay(sim, screen, adjust, tile_x_offset, tile_y_offset)

# Set visualization mode
overlay.set_overlay_mode(OverlayMode.PATHFINDING)

# Set goal for pathfinding
overlay.set_goal_position((400.0, 200.0))

# In your render loop
overlay_surface = overlay.draw_overlay(debug_info)
screen.blit(overlay_surface, (0, 0))

# Handle keyboard controls
for event in pygame.event.get():
    if event.type == pygame.KEYDOWN:
        if overlay.handle_key_press(event.key):
            continue  # Overlay handled the key
```

### Interactive Visualization

```python
from nclone.graph.visualization_api import GraphVisualizationAPI

# Create interactive session
api = GraphVisualizationAPI()
interactive_viz = api.create_interactive_session(
    level_data=level_data,
    entities=entities,
    width=1400,
    height=900
)

# Run interactive visualization
interactive_viz.run(graph_data)
```

## Visualization Configuration

### Customizing Display Options

```python
from nclone.graph.visualization import VisualizationConfig

# Create custom configuration
config = VisualizationConfig(
    # Node display
    show_nodes=True,
    show_node_labels=True,
    node_size=4.0,
    
    # Edge display
    show_edges=True,
    show_edge_labels=False,
    edge_width=2.0,
    
    # Path visualization
    show_shortest_path=True,
    highlight_path_nodes=True,
    path_width=4.0,
    
    # Node type filtering
    show_grid_nodes=True,
    show_entity_nodes=True,
    show_ninja_node=True,
    
    # Edge type filtering
    show_walk_edges=True,
    show_jump_edges=True,
    show_fall_edges=True,
    show_wall_slide_edges=True,
    show_functional_edges=True,
    
    # Visual settings
    alpha=0.8,
    background_color=(20, 20, 30),
    text_color=(255, 255, 255)
)

# Use with visualizer
from nclone.graph.visualization import GraphVisualizer
visualizer = GraphVisualizer(config)
```

### Saving and Loading Configurations

```python
# Export configuration
api.export_visualization_config(config, "my_config.json")

# Import configuration
loaded_config = api.import_visualization_config("my_config.json")
```

## Physics-Accurate Pathfinding

The system uses the exact same movement classification and trajectory calculation logic as the RL training system for 100% accurate pathfinding.

### Movement Types Supported

- **WALK**: Horizontal ground movement
- **JUMP**: Upward trajectory movement with gravity
- **FALL**: Downward gravity movement
- **WALL_SLIDE**: Wall contact movement
- **WALL_JUMP**: Wall-assisted jump
- **LAUNCH_PAD**: Launch pad boost movement
- **BOUNCE_BLOCK**: Bounce block interaction
- **BOUNCE_CHAIN**: Chained bounce block sequence
- **BOUNCE_BOOST**: Repeated boost on extending bounce block

### Edge Cost Calculation

Edge costs are calculated using:

- **Energy Cost**: Physics-based energy requirements
- **Time Estimate**: Actual movement time using N++ physics
- **Difficulty**: Movement precision requirements
- **Success Probability**: Likelihood of successful execution
- **Trajectory Validation**: Collision detection and clearance checking

### Ninja State Integration

```python
# Ninja state affects pathfinding accuracy
ninja_state = {
    'movement_state': 1,  # 0=Immobile, 1=Running, 2=Ground Sliding, etc.
    'velocity': (5.0, 0.0),  # Current velocity
    'position': (100.0, 300.0),  # Current position
    'ground_contact': True,  # Whether ninja is on ground
    'wall_contact': False   # Whether ninja is touching wall
}
```

## Keyboard Controls (Interactive Mode)

- **G**: Cycle through overlay modes (Disabled → Basic → Pathfinding → Full Analysis)
- **H**: Toggle hierarchical graph view
- **P**: Set pathfinding goal to current mouse position
- **R**: Reset visualization (clear goal and path)
- **ESC**: Exit interactive mode

## Overlay Modes

### Basic Graph Mode
- Shows nodes and edges
- Minimal visual noise
- Good for understanding graph structure

### Pathfinding Mode
- Shows shortest path to goal
- Highlights path nodes and edges
- Displays pathfinding statistics

### Full Analysis Mode
- Shows all graph components
- Node and edge labels
- Detailed analysis information
- Performance metrics

## Performance Optimization

### Entity Caching
The system automatically caches static entities (launch pads, doors, etc.) for performance while tracking dynamic entity states.

### Graph Data Caching
Level graphs are cached and reused when the same level is visualized multiple times.

### Hierarchical Processing
Use hierarchical graphs for better performance on large levels:

```python
request.use_hierarchical = True
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure both nclone and npp-rl are in your Python path
2. **Physics Not Available**: The system falls back to simplified physics if npp-rl modules aren't available
3. **Performance Issues**: Use hierarchical processing for large levels
4. **Visualization Too Cluttered**: Adjust alpha values and disable unnecessary components

### Debug Information

```python
# Get performance statistics
stats = api.get_performance_stats()
print(f"Total visualizations: {stats['total_visualizations']}")
print(f"Average render time: {stats['average_render_time']:.2f}ms")
print(f"Cache hits: {stats['cache_hits']}")

# Clear cache if needed
api.clear_cache()
```

## Examples

### Example 1: Basic Level Analysis

```python
import numpy as np
from nclone.graph.visualization_api import visualize_level_graph

# Create simple test level
level_data = {
    'level_id': 'test_level',
    'tiles': np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]),
    'width': 5,
    'height': 5
}

entities = [
    {'type': 20, 'x': 100.0, 'y': 100.0, 'state': 0}  # Exit
]

# Visualize
success = visualize_level_graph(
    level_data, entities, "basic_level.png", size=(800, 600)
)
```

### Example 2: Complex Pathfinding

```python
from nclone.graph.visualization_api import GraphVisualizationAPI, VisualizationRequest

api = GraphVisualizationAPI()

# Complex level with multiple entity types
entities = [
    {'type': 10, 'x': 100.0, 'y': 300.0, 'state': 0, 'orientation': 0},  # Launch pad
    {'type': 15, 'x': 250.0, 'y': 200.0, 'state': 0},  # Bounce block
    {'type': 20, 'x': 400.0, 'y': 150.0, 'state': 0},  # Exit
]

request = VisualizationRequest(
    level_data=complex_level_data,
    entities=entities,
    ninja_position=(50.0, 320.0),
    ninja_velocity=(0.0, 0.0),
    ninja_state=0,
    goal_position=(400.0, 150.0),
    use_hierarchical=True
)

result = api.visualize_graph(request)

if result.success and result.path_result.success:
    print(f"Found path with {len(result.path_result.path)} nodes")
    print(f"Path uses {len(set(result.path_result.edge_types))} different movement types")
```

### Example 3: Real-time Simulator Integration

```python
# In your simulator's render loop
class YourSimulator:
    def __init__(self):
        self.overlay = EnhancedDebugOverlay(self, screen, 1.0, 0, 0)
        self.overlay.set_overlay_mode(OverlayMode.PATHFINDING)
    
    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if self.overlay.handle_key_press(event.key):
                return  # Overlay handled it
            # Handle other keys...
    
    def render(self):
        # Draw simulator
        self.draw_level()
        self.draw_entities()
        self.draw_ninja()
        
        # Draw overlay
        overlay_surface = self.overlay.draw_overlay()
        self.screen.blit(overlay_surface, (0, 0))
        
        pygame.display.flip()
```

## API Reference

See the individual module documentation for detailed API reference:

- `nclone.graph.visualization_api`: Main API interface
- `nclone.graph.visualization`: Core visualization components
- `nclone.graph.pathfinding`: Physics-accurate pathfinding
- `nclone.graph.enhanced_debug_overlay`: Simulator integration

## Contributing

When extending the visualization system:

1. Maintain 100% physics accuracy by using the same logic as npp-rl
2. Add comprehensive tests for new features
3. Update this documentation
4. Follow the existing code style and patterns