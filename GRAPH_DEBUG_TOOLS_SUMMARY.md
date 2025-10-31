# Graph Debug Tools and Analysis Summary

## Created Debug Tools

### 1. `nclone/tools/debug_tile_connectivity.py`
**Purpose**: Analyze the precomputed tile-to-tile connectivity table

**Usage**:
```bash
python nclone/tools/debug_tile_connectivity.py
```

**Features**:
- Loads and analyzes the 5D connectivity table `[34, 34, 8, 4, 4]`
- Shows traversability percentages (currently 14.65%)
- Analyzes specific tile type pairs (empty→empty, empty→half, etc.)
- Identifies suspicious connections (e.g., empty→solid)
- Generates connectivity matrices for each direction

**Key Findings**:
- Empty→Empty allows ALL 16 sub-node combinations in all directions ✓
- Solid tiles (Type 1) show 0% traversability ✓
- Overall 14.6% traversability indicates strict filtering

### 2. `nclone/tools/detailed_graph_debug.py`
**Purpose**: Analyze actual graph building from environment level data

**Usage**:
```bash
python nclone/tools/detailed_graph_debug.py [MAP_ID]
```

**Features**:
- Creates full environment and builds graph with debug enabled
- Counts nodes by tile type
- Analyzes edges (within-tile vs cross-tile)
- Identifies problematic edges (e.g., diagonal through solid)
- Generates detailed edge report text file

**Key Findings** (Map 0):
- 102 empty tiles, 864 solid tiles
- 408 sub-nodes generated (102 × 4 = 408) ✓
- 2,992 total edges
- 1,768 cross-tile edges, all Type 0 → Type 0 ✓
- ✓ No problematic diagonal edges found
- ✓ No edges involving solid tiles found

### 3. `nclone/tools/visualize_tile_edges.py`
**Purpose**: ASCII visualization of tile-level edge connectivity

**Usage**:
```bash
python nclone/tools/visualize_tile_edges.py [MAP_ID]
```

**Features**:
- Shows ASCII map of which tiles have nodes (X = nodes, . = solid)
- Counts cross-tile edges by direction
- Identifies impossible connections at tile level
- Visual inspection of tile connectivity patterns

**Key Findings** (Map 0):
- 845 tiles with nodes
- 1,134 unique cross-tile edge pairs
- 2,652 total cross-tile edges
- Edge distribution: E/W (673 each), N/S (465 each), diagonals (~95 each)
- ✓ No impossible connections found at tile level

### 4. `test_inline_graph.py`
**Purpose**: Quick inline test for graph building verification

**Usage**:
```bash
python test_inline_graph.py
```

**Features**:
- Creates simple 5×5 test level
- Builds graph with debug enabled
- Shows immediate results
- Useful for quick testing of changes

## Analysis Summary

### What's Working ✓
1. **Tile-level connectivity**: No edges between solid tiles
2. **Empty tile connectivity**: All sub-nodes within empty tiles are correctly connected
3. **Diagonal blocking**: Some diagonal movements are being blocked correctly
4. **Overall statistics**: 14.6% traversability shows strict filtering is active

### Potential Issues ⚠
1. **Sub-node sampling**: The "all-points-clear" requirement might still allow some edge cases through
2. **Pixel-perfect accuracy**: Even with 11 samples per cardinal direction, there might be gaps
3. **Complex tile geometry**: Half-tiles, slopes, and circular segments have complex boundaries
4. **Diagnostic timing**: Debug counters show "0" initially because they're printed before adjacency building

## User's Reported Issue

The user states: "There are still issues with letting between-tile connectivity (from one tile to another) be too permissive. Our flood fill should not disrespect the connectivity rules."

### Observations from Images:
1. Nodes appear in areas that should be unreachable
2. The graph seems overly connected
3. Reachability extends beyond expected boundaries

### Hypotheses:
1. **Edge case in sampling**: The dense sampling (11 points) might still miss thin solid regions
2. **Sub-node alignment**: The sub-node positions (6, 18) might not align perfectly with tile boundaries
3. **Diagonal intermediate checks**: Even though we check both intermediate tiles, the specific sub-node positions might not be in solid areas
4. **Tile type geometry edge cases**: Some tile types might have incorrect geometry definitions

## Recommendations

### Immediate Actions:
1. **Add comprehensive edge logging**: Log EVERY edge decision (allowed and blocked) for a specific problematic tile pair
2. **Visual tile-by-tile inspection**: Create a tool that shows pixel-by-pixel what's considered traversable
3. **Sample point visualization**: Show exactly which sample points are being checked for each edge
4. **Threshold analysis**: Verify that "all points clear" is truly enforced

### Further Investigation:
1. Test with a minimal level that shows the problem
2. Add sub-pixel visualization of ninja clearance requirements
3. Compare with actual N++ physics to verify correctness
4. Create regression tests for specific known-bad connections

## Next Steps

User should:
1. Identify specific problematic tile pairs in the images
2. Note exact tile coordinates where impossible connections occur
3. Run the debug tools on those specific areas
4. Provide feedback on which tool output is most useful

This will allow us to pinpoint the exact cause and fix it holistically.

