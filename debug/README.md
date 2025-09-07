# Debug Scripts

This directory contains debugging and analysis scripts used during the development and troubleshooting of the graph visualization system.

## Categories

### Analysis Scripts
- `analyze_doortest_issues.py` - Comprehensive analysis of the three main graph visualization issues
- `analyze_graph_fragmentation.py` - Analysis of graph connectivity and fragmentation issues
- `analyze_map_layout.py` - Map layout analysis to understand empty tile clusters and connectivity
- `final_validation.py` - Final comprehensive validation of all three resolved issues
- `quick_connectivity_analysis.py` - Quick analysis of ninja's graph connectivity

### Debug Scripts
- `debug_*.py` - Various debugging scripts for specific components:
  - Collision detection debugging
  - Entity structure analysis
  - Graph connectivity analysis
  - Node mapping and positioning
  - Pathfinding debugging
  - Traversability analysis

### Validation Scripts
- `validate_*.py` - Validation scripts for specific map configurations and scenarios

## Usage

These scripts are primarily for development and debugging purposes. They can be run individually to investigate specific aspects of the graph system:

```bash
cd /workspace/nclone
python debug/final_validation.py
python debug/analyze_graph_fragmentation.py
```

## Note

These scripts were created during the resolution of the three main graph visualization issues:
1. Missing functional edges between switches and doors
2. Invalid walkable edges in solid tiles
3. Ninja pathfinding from solid spawn tiles

All issues have been successfully resolved as of the latest commits.