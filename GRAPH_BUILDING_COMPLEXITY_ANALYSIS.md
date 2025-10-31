# Graph Building Complexity Analysis

## Current Implementation Analysis

### Overall Complexity: **O(N)** where N = level_width × level_height × 4 (sub-nodes)

This is acceptable since N is bounded by level dimensions (fixed grid size).

## Detailed Operation Complexity

### 1. `build_graph()` - **O(N) or O(1) cached**
- **O(1)** if level is cached (cache hit)
- **O(N)** for first build (cache miss):
  - Calls `_build_base_graph()` which is O(N)
  - Calls `_apply_entity_mask()` which is O(E) where E = number of edges ≤ 8N
  - Calls `_flood_fill_from_graph()` which is O(N) for BFS

**Status**: ✅ **Optimal** - Caching makes this O(1) after first call per level

### 2. `_generate_sub_nodes()` - **O(N)** 
```python
for tile_y in range(height):              # O(height)
    for tile_x in range(width):           # O(width)
        for sub_x, sub_y in [(0,0), (1,0), (0,1), (1,1)]:  # O(4) = O(1)
            # O(1) validity check via _check_subnode_validity_simple()
```
- **Total**: O(width × height × 4) = O(N)
- **Status**: ✅ **Optimal** - Must visit each tile at least once

### 3. `_build_reachable_adjacency()` - **O(N)** 
```python
for current_pos in all_sub_nodes:         # O(N) where N = number of sub-nodes
    for direction in 8_directions:        # O(8) = O(1)
        # O(1) lookup in all_sub_nodes dict
        # O(1) precomputed connectivity check
```
- **Total**: O(N × 8) = O(N)
- **Status**: ✅ **Optimal** - Must check each node's 8 neighbors

### 4. `_check_subnode_validity_simple()` - **O(1)**
- Simple arithmetic and comparisons
- No loops
- **Status**: ✅ **Optimal**

### 5. `_is_sub_node_traversable()` - **O(1)**
- Two O(1) validity checks
- One O(1) precomputed connectivity lookup: `connectivity_loader.is_traversable()`
- **Status**: ✅ **Optimal**

### 6. `_flood_fill_from_graph()` - **O(N + E)** = **O(N)**
```python
while queue:                              # O(N) - each node visited once
    for neighbor in adjacency[current]:   # O(deg) - at most 8 neighbors
        # O(1) operations
```
- **Total**: O(N + E) where E ≤ 8N, so O(N)
- **Status**: ✅ **Optimal** - Standard BFS complexity

### 7. `connectivity_loader.is_traversable()` - **O(1)**
```python
return bool(self._connectivity_table[tile_a, tile_b, dir_idx])
```
- Direct numpy array indexing
- **Status**: ✅ **Optimal**

## Potential Issues & Fixes

### ⚠️ Issue 1: `_find_closest_node()` - **O(M)** where M = number of sub-nodes

**Current Implementation:**
```python
def _find_closest_node(self, pos, sub_nodes):
    for node_pos in sub_nodes.keys():     # O(M) - linear search
        dist = ((nx - px) ** 2 + (ny - py) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_node = node_pos
```

**Problem**: Linear search through all sub-nodes

**Fix**: Use spatial indexing or snap to grid

**Proposed Solution:**
```python
def _find_closest_node_fast(self, pos: Tuple[int, int], sub_nodes: Dict) -> Optional[Tuple[int, int]]:
    """Find closest sub-node by snapping to 12px grid - O(1)."""
    px, py = pos
    
    # Snap to nearest sub-node position (12px grid)
    # Sub-nodes are at tile*24 + {6, 18} for each dimension
    snap_x = round((px - 6) / 12) * 12 + 6
    snap_y = round((py - 6) / 12) * 12 + 6
    
    # Check if snapped position exists
    if (snap_x, snap_y) in sub_nodes:
        return (snap_x, snap_y)
    
    # If not, check the 8 neighboring sub-nodes (still O(1))
    for dx in [-12, 0, 12]:
        for dy in [-12, 0, 12]:
            candidate = (snap_x + dx, snap_y + dy)
            if candidate in sub_nodes:
                return candidate
    
    # Fallback to linear search only if grid snap fails (rare)
    return self._find_closest_node_linear(pos, sub_nodes)
```

**Impact**: 
- Called once per `build_graph()` when ninja_pos provided
- Currently O(M) → Optimized to O(1) with fallback
- **Priority**: Medium (only called once, but could be 100s of sub-nodes)

## Summary

| Operation | Current Complexity | Status | Notes |
|-----------|-------------------|--------|-------|
| `build_graph()` | O(1) cached / O(N) first | ✅ Optimal | Caching is key |
| `_generate_sub_nodes()` | O(N) | ✅ Optimal | Must visit all tiles |
| `_build_reachable_adjacency()` | O(N) | ✅ Optimal | Must check all connections |
| `_check_subnode_validity_simple()` | O(1) | ✅ Optimal | Pure arithmetic |
| `_is_sub_node_traversable()` | O(1) | ✅ Optimal | Precomputed lookups |
| `_flood_fill_from_graph()` | O(N) | ✅ Optimal | Standard BFS |
| `_find_closest_node()` | O(M) | ⚠️ Can optimize | Grid snapping → O(1) |
| `connectivity_loader.is_traversable()` | O(1) | ✅ Optimal | Direct array access |

### Overall Assessment: ✅ **EXCELLENT**

The graph building is **O(N)** where N is bounded by level dimensions (typically 25×25 tiles = 625 tiles → 2500 sub-nodes max). This is:
- **Cacheable**: O(1) for repeated builds of the same level
- **Linear**: O(N) scaling with level size
- **Bounded**: N is constant per level (max ~2500-5000 sub-nodes)

### Recommended Optimization

Implement the O(1) grid-snapping version of `_find_closest_node()` to eliminate the one remaining O(M) operation.

**Expected Impact**: Minimal (saves ~0.1-1ms on graph builds), but improves worst-case guarantees.

