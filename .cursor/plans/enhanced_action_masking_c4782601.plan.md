---
name: Enhanced Action Masking
overview: Enhance the action masking system to leverage path context, physics/velocity state, hazard proximity, and vertical momentum to guide the agent along the optimal path while avoiding dangerous momentum buildup toward hazards.
todos:
  - id: add-constants
    content: Add new masking threshold constants to physics_constants.py
    status: pending
  - id: build-context
    content: Implement _build_enhanced_masking_context() in base_environment.py to gather SDF, corridor, and velocity data
    status: pending
  - id: momentum-masking
    content: "Add Layer 3: Momentum-aware masking in ninja.py get_valid_action_mask()"
    status: pending
  - id: hazard-masking
    content: "Add Layer 4: Hazard gradient masking using mine SDF escape gradients"
    status: pending
  - id: corridor-masking
    content: "Add Layer 5: Corridor deviation masking to keep agent within 48px of optimal path"
    status: pending
  - id: vertical-masking
    content: "Add Layer 6: Vertical efficiency masking to encourage sustained jumping toward goals above"
    status: pending
  - id: integration-test
    content: Test enhanced masking on diverse level types (horizontal, vertical, hazard-adjacent)
    status: pending
---

# Enhanced Action Masking System

## Current State

The existing action masking in [`ninja.py`](nclone/nclone/ninja.py) (lines 624-905) provides:

- **Physics-based masking**: Useless jumps, wall collision avoidance
- **Path-based masking**: Masks actions opposite to 3-4 hop path direction

Available infrastructure to leverage:

- `MineSignedDistanceField` in [`mine_proximity_cache.py`](nclone/nclone/graph/reachability/mine_proximity_cache.py) - precomputed SDF with escape gradients
- `multi_hop_direction_cache` in [`path_distance_cache.py`](nclone/nclone/graph/reachability/path_distance_cache.py) - 4-8 hop lookahead
- Ninja velocity state: `xspeed`, `yspeed` already tracked

## Architecture

```mermaid
flowchart TD
    subgraph PrecomputedPerLevel [Precomputed Per Level - O(1) Lookup]
        MineSDF[Mine SDF Grid<br/>Danger zones + escape gradients]
        PathCache[Path Distance Cache<br/>Multi-hop directions]
        PhysicsCache[Node Physics Cache<br/>Grounded/walled states]
    end

    subgraph RuntimeContext [Runtime Context - Per Frame]
        NinjaVel[Ninja Velocity<br/>xspeed, yspeed]
        NinjaPos[Ninja Position]
        PathDir[Path Direction<br/>from physics-optimal path]
    end

    subgraph MaskingLayers [Masking Layers - Applied in Order]
        L1[Layer 1: Physics Masking<br/>Useless jumps, wall collision]
        L2[Layer 2: Path Direction Masking<br/>Mask actions opposite to goal]
        L3[Layer 3: Momentum Masking<br/>Prevent bad momentum buildup]
        L4[Layer 4: Hazard Gradient Masking<br/>Mask actions toward danger]
        L5[Layer 5: Corridor Deviation Masking<br/>Keep within 48px of optimal path]
        L6[Layer 6: Vertical Efficiency Masking<br/>Encourage sustained jumping upward]
    end

    PrecomputedPerLevel --> MaskingLayers
    RuntimeContext --> MaskingLayers
    L1 --> L2 --> L3 --> L4 --> L5 --> L6
    L6 --> FinalMask[Final Action Mask]
```

## New Masking Layers

### Layer 3: Momentum-Aware Masking (Aggressive)

Mask actions that would build momentum away from the optimal path direction.

**Logic:**

- If `ninja.xspeed > 0` (moving right) and path direction `dx < 0` (goal left): mask RIGHT
- If `ninja.xspeed < 0` (moving left) and path direction `dx > 0` (goal right): mask LEFT  
- Apply when momentum magnitude > 0.5 px/frame (any meaningful velocity)

**Key insight**: Current system only masks based on path direction, not current velocity. This layer prevents the agent from *continuing* to build wrong-direction momentum.

### Layer 4: Hazard Gradient Masking (Path Priority)

Use mine SDF escape gradient to mask actions moving toward danger zones, but only when combined with path deviation.

**Logic:**

- Query mine SDF at ninja position for escape gradient `(grad_x, grad_y)`
- If in danger zone (SDF < 0):
  - Mask actions whose direction dot-product with escape gradient is negative (moving deeper into danger)
- If near danger (0 < SDF < danger_threshold):
  - Only mask if action also deviates from path (combined condition)

**Performance**: O(1) lookup from precomputed SDF grid

### Layer 5: Corridor Deviation Masking (~48px corridor)

Keep agent within 2-tile corridor around optimal path.

**Logic:**

- Track perpendicular distance from ninja position to optimal path line
- If distance > 48px, mask actions that increase deviation
- Compute using path segment direction as line, measure perpendicular distance

**Implementation approach:**

- Use `next_hop` from path cache to define path vector
- Compute signed distance from ninja to path line
- Mask actions pushing further from corridor

### Layer 6: Vertical Efficiency Masking (New)

Encourage sustained jumping when goal is above player.

**Logic:**

- When goal is above (`dy < 0` in path direction) AND ninja is grounded:
  - Mask NOOP (should be jumping)
  - Mask pure horizontal actions without jump (LEFT, RIGHT without jump)
  - Keep JUMP, JUMP+LEFT, JUMP+RIGHT valid
- When goal is above AND ninja is airborne with upward velocity (`yspeed < 0`):
  - Keep jump actions valid (sustain the jump)
  - Only mask actions if they would counter the upward progress AND move horizontally away from goal
- Threshold: Apply when vertical component `|dy| > 0.3` (significant upward requirement)

## Implementation Details

### New Data Structure for Masking Context

```python
# In ninja.py, new dataclass for enhanced masking context
@dataclass
class EnhancedMaskingContext:
    # Path context
    path_dx: float  # Normalized path direction X
    path_dy: float  # Normalized path direction Y
    distance_to_goal: float
    
    # Hazard context (from mine SDF)
    sdf_value: float  # Signed distance to nearest mine
    escape_grad_x: float  # Escape direction X
    escape_grad_y: float  # Escape direction Y
    
    # Corridor context
    perpendicular_distance: float  # Distance from optimal path line
    corridor_deviation_sign: float  # +1 if right of path, -1 if left
    
    # Computed at level load, passed to ninja
    in_danger_zone: bool
    near_danger_zone: bool
```

### Modified Flow in base_environment.py

```python
def _get_action_mask_with_path_update(self) -> np.ndarray:
    # Existing: Get masking mode and path direction
    masking_mode = self.reward_calculator.config.action_masking_mode
    path_direction_data = self._detect_straight_path_direction()
    
    # NEW: Build enhanced masking context
    enhanced_context = self._build_enhanced_masking_context(path_direction_data)
    
    # Pass both to simulator
    self.nplay_headless.sim._action_masking_mode = masking_mode
    self.nplay_headless.sim._path_direction_data = path_direction_data
    self.nplay_headless.sim._enhanced_masking_context = enhanced_context  # NEW
    
    return np.array(self.nplay_headless.get_action_mask(), dtype=np.int8)
```

### Constants to Add (physics_constants.py)

```python
# Enhanced action masking constants
MOMENTUM_MASKING_THRESHOLD = 0.5  # Min velocity to trigger momentum masking
HAZARD_PROXIMITY_THRESHOLD = 30.0  # SDF value below which near-danger masking applies
CORRIDOR_WIDTH = 48.0  # Half-width of safe corridor around optimal path (2 tiles)
VERTICAL_GOAL_THRESHOLD = 0.3  # |dy| threshold for vertical efficiency masking
```

## Files to Modify

1. **[`nclone/nclone/ninja.py`](nclone/nclone/ninja.py)** - Add new masking layers to `get_valid_action_mask()`
2. **[`nclone/nclone/gym_environment/base_environment.py`](nclone/nclone/gym_environment/base_environment.py)** - Add `_build_enhanced_masking_context()` method
3. **[`nclone/nclone/constants/physics_constants.py`](nclone/nclone/constants/physics_constants.py)** - Add new threshold constants

## Performance Considerations

- All SDF lookups: O(1) from precomputed grid
- Path direction: Already computed, just passed through
- Corridor distance: O(1) vector math per frame
- No pathfinding in masking loop - all precomputed

## Testing Strategy

- Verify mask always has at least one valid action (existing fallback)
- Test on levels with:
  - Horizontal corridors (should strongly favor one direction)
  - Vertical ascents (should encourage sustained jumping)
  - Mine-adjacent paths (should avoid actions toward mines)
  - Curved paths (should allow corridor deviation up to 48px)