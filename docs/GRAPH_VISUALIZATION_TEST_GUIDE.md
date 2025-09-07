# Graph Visualization Test Guide

This document explains how to use the enhanced `test_environment.py` script for graph visualization and pathfinding testing.

## Overview

The `test_environment.py` script has been enhanced with comprehensive graph visualization capabilities, allowing you to:

- Visualize the level graph as an overlay on the simulator
- Display standalone graph windows for detailed analysis
- Use interactive graph exploration tools
- Test pathfinding algorithms (A* vs Dijkstra)
- Export graph visualizations to images
- Customize which edge types are displayed

## Basic Usage

### 1. Graph Overlay on Simulator

```bash
# Show graph overlay on the main simulator window
python nclone/test_environment.py --visualize-graph

# Include pathfinding with fast A* algorithm
python nclone/test_environment.py --visualize-graph --pathfind --algorithm astar
```

### 2. Standalone Graph Window

```bash
# Open a separate window showing just the graph
python nclone/test_environment.py --standalone-graph

# Combine with simulator for side-by-side view
python nclone/test_environment.py --visualize-graph --standalone-graph
```

### 3. Interactive Graph Exploration

```bash
# Launch interactive graph visualization tool
python nclone/test_environment.py --interactive-graph
```

### 4. Export Graph Visualizations

```bash
# Save graph to image file
python nclone/test_environment.py --headless --save-graph level_analysis.png

# Export with all edge types visible
python nclone/test_environment.py --headless --save-graph complete_graph.png --show-edges walk jump fall wall_slide one_way functional
```

## Algorithm Selection

### A* Algorithm (Default)
- **Use for**: Real-time visualization, RL training, interactive features
- **Advantages**: 10-100x faster, lower memory usage
- **Best when**: Single goal, standard level geometry, performance critical

```bash
python nclone/test_environment.py --pathfind --algorithm astar
```

### Dijkstra Algorithm
- **Use for**: Level analysis, complex levels, validation
- **Advantages**: Guaranteed optimal paths, handles complex graph structures
- **Best when**: Multiple goals, switches/teleporters, accuracy critical

```bash
python nclone/test_environment.py --pathfind --algorithm dijkstra
```

## Edge Type Customization

Control which types of movement connections are visualized:

```bash
# Show only basic movement
python nclone/test_environment.py --visualize-graph --show-edges walk jump

# Show all movement types
python nclone/test_environment.py --visualize-graph --show-edges walk jump fall wall_slide

# Include functional connections (switches, teleporters)
python nclone/test_environment.py --visualize-graph --show-edges walk jump functional
```

Available edge types:
- `walk` - Basic horizontal movement
- `jump` - Jump connections
- `fall` - Falling/gravity connections
- `wall_slide` - Wall sliding movement
- `one_way` - One-directional connections
- `functional` - Switch-door, teleporter connections

## Interactive Runtime Controls

While the simulator is running, use these keyboard shortcuts:

| Key | Action |
|-----|--------|
| `V` | Toggle graph overlay on/off |
| `P` | Trigger pathfinding demonstration |
| `S` | Save current graph visualization |
| `G` | Toggle built-in graph debug overlay |
| `E` | Toggle exploration debug overlay |
| `C` | Toggle grid debug overlay |
| `R` | Reset environment |

## Advanced Examples

### RL Training Visualization
```bash
# Fast pathfinding for real-time RL training
python nclone/test_environment.py --visualize-graph --pathfind --algorithm astar --show-edges walk jump
```

### Level Design Analysis
```bash
# Comprehensive analysis with optimal pathfinding
python nclone/test_environment.py --standalone-graph --pathfind --algorithm dijkstra --show-edges walk jump fall wall_slide functional
```

### Documentation Generation
```bash
# Export high-quality graph images for documentation
python nclone/test_environment.py --headless --save-graph documentation_graph.png --show-edges walk jump fall functional --profile-frames 1
```

### Custom Map Analysis
```bash
# Analyze a specific level file
python nclone/test_environment.py --map custom_level.txt --visualize-graph --pathfind --standalone-graph
```

## Troubleshooting

### Graph Visualization Not Available
If you see "Graph visualization not available", ensure all dependencies are installed:

```bash
pip install -e .  # Install nclone package
pip install opencv-python albumentations pycairo  # Additional dependencies
```

### Interactive Mode Issues
Interactive graph mode requires a display. Use `--headless` mode for server environments:

```bash
# Instead of --interactive-graph, use:
python nclone/test_environment.py --headless --save-graph analysis.png
```

### Performance Considerations
- Use A* algorithm for real-time applications
- Use Dijkstra for accuracy-critical analysis
- Limit edge types shown for better performance
- Use headless mode for batch processing

## Integration with Development Workflow

### Testing New Levels
```bash
# Quick validation of level connectivity
python nclone/test_environment.py --map new_level.txt --visualize-graph --pathfind

# Comprehensive analysis
python nclone/test_environment.py --map new_level.txt --standalone-graph --algorithm dijkstra --show-edges walk jump fall functional
```

### RL Algorithm Development
```bash
# Real-time visualization during training
python nclone/test_environment.py --visualize-graph --pathfind --algorithm astar

# Export training environment graphs
python nclone/test_environment.py --headless --save-graph training_env.png --show-edges walk jump
```

### Level Design Validation
```bash
# Verify all areas are reachable
python nclone/test_environment.py --standalone-graph --pathfind --algorithm dijkstra --show-edges walk jump fall wall_slide functional
```

This enhanced test environment provides a comprehensive toolkit for visualizing and analyzing the N++ level graphs, supporting both real-time interactive use and batch analysis workflows.