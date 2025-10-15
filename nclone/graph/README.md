# Graph Optimization for Vision-Free Training

## Architecture

### Node Structure
- **Count**: 400-800 nodes (traversable tiles + critical entities)
- **Features**: 19 dimensions
- **Types**: 6 categories (EMPTY, WALL, ENTITY, HAZARD, SPAWN, EXIT)

### Entity Coverage
Only critical entities for navigation:
- Toggle mines (types 1, 21): state-dependent hazards
- Locked doors (type 6): gated progression
- Exit switch (type 4): primary objective
- Exit door (type 3): level completion

### Update Strategy
- **Static**: Graph topology built once per level
- **Dynamic**: Features updated only when entity states change
- **Triggers**: Mine toggles, door opens, switch activations

## Feature Breakdown (19 dimensions)

### Spatial Features (2 dims, indices 0-1)
- x_normalized: Position X / MAP_WIDTH
- y_normalized: Position Y / MAP_HEIGHT

### Node Type (6 dims, indices 2-7)
One-hot encoding:
- EMPTY (0): Traversable space
- WALL (1): Solid obstacle
- ENTITY (2): Interactive entity
- HAZARD (3): Dangerous area
- SPAWN (4): Player spawn
- EXIT (5): Level exit

### Tile Category (3 dims, indices 8-10)
One-hot encoding:
- Empty (0): tile_id == 0
- Solid (1): tile_id == 1 or tile_id >= 34
- Navigable (2): 2 <= tile_id <= 33 (slopes, half-tiles, curves)

### Entity Features (4 dims, indices 11-14)
- **type_encoding** (11): 0.0=none, 0.25=toggle_mine, 0.5=exit_switch, 0.75=locked_door, 1.0=exit_door
- **state** (12): Normalized entity state
  - Toggle mine: 0/1/2 → 0.0/0.5/1.0 (toggled/untoggled/toggling)
  - Locked door: closed=1.0, open=0.0
  - Exit door: accessible=0.0, inaccessible=1.0
- **active** (13): Binary flag (0/1)
- **radius** (14): Collision radius normalized by 2*NINJA_RADIUS

### Reachability (1 dim, index 15)
- **reachable_from_ninja**: Boolean from flood-fill (<1ms)

### Proximity Features (3 dims, indices 16-18)
- **dist_to_ninja**: Distance normalized by screen diagonal
- **dist_to_goal**: Distance to exit/switch normalized
- **is_ninja_node**: Binary flag for ninja's current tile

## Usage

### Basic Usage

```python
from nclone.graph.graph_builder import GraphBuilder
from nclone.graph.level_data import LevelData

# Initialize builder
builder = GraphBuilder(debug=False)

# Build graph for a level
graph_data = builder.build_graph(level_data)

# Access graph components
print(f"Nodes: {graph_data.fine_graph.num_nodes}")
print(f"Edges: {graph_data.fine_graph.num_edges}")
print(f"Node features shape: {graph_data.fine_graph.node_features.shape}")
```

### Training Configuration

```python
from npp_rl.training.architecture_configs import get_architecture_config

config = get_architecture_config("vision_free_simplified")
# - 19-dim node features
# - Simplified HGT (2 layers, 128 hidden)
# - Optimized for fast training
```

### Integration Notes

- **No physics pre-computation in graph**: Movement is continuous physics-based (gravity, friction, momentum) - too expensive to pre-compute at graph build time
- **Agent learns movement from experience**: Temporal state and rewards provide movement dynamics
- **Graph provides structural information only**: Connectivity, reachability, entity states
- **Reachability from fast flood-fill**: OpenCV-based flood-fill analysis (<1ms)

## Key Design Decisions

### 1. No Movement Pre-computation
Physics is continuous and too complex (gravity, friction, momentum, slopes, wall-jumps). The agent learns movement capabilities from:
- Temporal frame stacking (movement history)
- Game state vector (velocity, acceleration, surface contact)
- Reward signals (successful/failed movements)

### 2. Critical Entities Only
Focus on entities that affect level completion:
- Toggle mines: Dynamic hazards that change state
- Locked doors: Gated progression requiring switches
- Exit switch/door: Primary objectives

Excluded entities (agent learns from visual/state):
- Gold, drones, thwumps, boost pads, etc.

### 3. In-place Refactoring
No backward compatibility needed since we control the codebase. Clean migration from hierarchical to single-resolution.

### 4. Single Resolution
Tile-level (24px) is sufficient:
- Ninja radius: 10px fits in 24px tiles
- Natural discretization unit for N++ (TILE_PIXEL_SIZE)
- Agent handles sub-pixel precision through physics state

### 5. Hybrid Updates
- **Static topology**: Built once per level (nodes, edges, structure)
- **Dynamic features**: Updated only when entity states change
- **Selective updates**: Only affected nodes recomputed
- **Trigger conditions**: Mine toggles, door opens, switch activations

## Module Structure

```
nclone/graph/
├── common.py                  # Core data structures (optimized constants)
├── graph_builder.py           # Main builder (single resolution)
├── edge_building.py           # Static structure + edge creation
├── feature_builder.py         # 19-dim node/edge features
├── update_tracker.py          # Entity state change detection
├── level_data.py              # Level data structures
└── reachability/              # Fast flood-fill reachability
    └── reachability_system.py
```

## API Reference

### GraphBuilder

```python
class GraphBuilder:
    """Optimized single-resolution graph builder."""
    
    def __init__(self, debug: bool = False):
        """Initialize builder with optional debug output."""
        
    def build_graph(
        self, 
        level_data: LevelData, 
        entities: List = None, 
        ninja_pos: Tuple[int, int] = None
    ) -> HierarchicalGraphData:
        """
        Build or update graph.
        
        First call: Builds static structure
        Subsequent calls: Selective updates only
        """
```

### StaticGraphStructure

```python
@dataclass
class StaticGraphStructure:
    """Immutable graph topology."""
    node_positions: np.ndarray          # [num_nodes, 2]
    node_types: np.ndarray              # [num_nodes]
    tile_categories: np.ndarray         # [num_nodes]
    edge_index: np.ndarray              # [2, num_edges]
    edge_types: np.ndarray              # [num_edges]
    entity_node_indices: Dict[str, int] # entity_id -> node_index
    tile_to_node_index: Dict[Tuple, int]
    num_nodes: int
    num_edges: int
```

## Performance Characteristics

### Memory Usage
- **Graph representation**: ~45 KB per level (600 nodes × 19 dims × 4 bytes)
- **Static structure**: ~20 KB per level (cached)
- **Total per environment**: ~65 KB (vs. ~3.6 MB before)

### Computational Cost
- **Initial build**: 0.5-1ms per level
- **Selective update**: 0.01-0.1ms per entity change
- **Reachability analysis**: <1ms (OpenCV flood-fill)
- **Feature extraction**: ~0.1ms for 600 nodes

### Training Impact
- **Batch size**: Can increase 8x → 64 with same GPU memory
- **Samples/second**: 10-20x improvement expected
- **GPU memory savings**: 80x per graph in batch

## Testing

Run tests to verify optimization:

```bash
cd nclone
pytest tests/test_optimized_graph.py -v
```

Tests verify:
- Graph size within target range (400-800 nodes)
- Feature dimensions correct (19)
- Selective updates working
- Performance improvements achieved
- Critical entities only

## Troubleshooting

### Graph too large (>1000 nodes)
- Check level complexity (very open levels may have more traversable tiles)
- Verify entity filtering (should only include 4 critical types)

### Graph too small (<100 nodes)
- Check level has traversable tiles (not all walls)
- Verify tile type detection (0 and 2-33 are traversable)

### Updates not triggering
- Verify entity states are changing in simulator
- Check _get_entity_states() in graph_mixin.py
- Enable debug mode to see update logs

### Performance not improved
- Check that static structure is being cached
- Enable debug to measure build/update times

## References

- **Tile Definitions**: `nclone/tile_definitions.py` (38 tile types)
- **Entity Types**: `nclone/constants/entity_types.py`
- **Physics Constants**: `nclone/constants/physics_constants.py`
- **Sim Mechanics**: `nclone/docs/sim_mechanics_doc.md`

