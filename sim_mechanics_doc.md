## Overview

This is a simulation of the game N++. N++ is a 2D physics-based platformer where the player controls a ninja character through levels filled with obstacles, enemies, and objectives. The game features precise physics simulation with momentum-based movement, wall jumping, and various interactive elements.

## Level Structure

### Dimensions
- **Level Size**: 42×23 grid cells (1056×600 pixels)
- **Cell Size**: 24×24 pixels per grid cell
- **Coordinate System**: Origin at top-left, X increases rightward, Y increases downward
- **Visibility**: The entire level is always visible to the player

### Time Limit
- **Maximum Duration**: 20,000 frames per level
- **Frame Rate**: 60 FPS (standard game speed)
- **Real Time Limit**: ~5.5 minutes per level

## Player Character (Ninja)

### Physical Properties
- **Radius**: 10 pixels (circular collision shape)
- **Spawn Position**: Defined by map data at coordinates `(map_data[1231]*6, map_data[1232]*6)`

### Movement States
The ninja has 9 distinct movement states with specific transition conditions:
1. **Immobile** (0): Stationary on ground, no horizontal input
2. **Running** (1): Moving horizontally on ground with input
3. **Ground Sliding** (2): Sliding on ground with momentum, input opposite to movement
4. **Jumping** (3): Actively jumping (gravity reduced), up to 45 frames
5. **Falling** (4): In air, falling or moving without jump input
6. **Wall Sliding** (5): Sliding down a wall while holding toward it
7. **Dead** (6): Post-death ragdoll physics active
8. **Awaiting Death** (7): Single-frame transition to death state
9. **Celebrating** (8): Victory state, reduced drag applied
10. **Disabled** (9): Inactive state

### Physics Constants

#### Gravity
- **Fall Gravity**: 0.0667 pixels/frame² (when falling or not jumping)
- **Jump Gravity**: 0.0111 pixels/frame² (when actively jumping)

#### Acceleration
- **Ground Acceleration**: 0.0667 pixels/frame² (horizontal movement on ground)
- **Air Acceleration**: 0.0444 pixels/frame² (horizontal movement in air)

#### Speed Limits
- **Maximum Horizontal Speed**: 3.333 pixels/frame
- **Maximum Jump Duration**: 45 frames

#### Drag and Friction
- **Regular Drag**: 0.9933 (applied to both X and Y velocity each frame)
- **Slow Drag**: 0.8618 (applied in certain conditions)
- **Ground Friction**: 0.9459 (applied to horizontal velocity on ground)
- **Ground Friction (Slow)**: 0.8618 (applied when moving slowly on ground)
- **Wall Friction**: 0.9113 (applied to vertical velocity when wall sliding)

#### Death Conditions
- **Maximum Survivable Impact**: 6 pixels/frame (impact velocity that causes death)
- **Minimum Survivable Crushing**: 0.05 (crushing threshold)

### Movement Mechanics

#### State Transitions
**Ground States (0-2):**
- State 0→1: Horizontal input while stationary
- State 1→2: Input opposite to movement direction or no input while moving
- State 2→1: Input matching movement direction with sufficient projection
- State 2→0: Low projection (<0.1) on flat ground

**Air States (3-5):**
- Any→4: Becomes airborne without active jump
- State 3→4: Jump duration exceeds 45 frames or jump input released
- State 4→5: Touching wall while falling and holding toward wall
- State 5→4: Not touching wall or input away from wall

#### Jumping
- **Floor Jump**: From flat ground, applies velocity `(0, -2)`
- **Wall Jump (Regular)**: Applies velocity `(1 * wall_normal, -1.4)`
- **Wall Jump (Slide)**: From wall slide state, applies velocity `(2/3 * wall_normal, -1)`
- **Slope Jumping**: Complex mechanics based on slope angle and movement direction
  - Downhill with input: `(2/3 * slope_x, 2 * slope_y)`
  - Uphill perpendicular: `(2/3 * slope_x, 2 * slope_y)` with speed reset
  - Default uphill: `(0, -1.4)`

#### Wall Interaction
- **Wall Detection**: Ninja can interact with walls within radius + 0.1 pixels
- **Wall Normal Calculation**: Sum of normalized depenetration vectors from nearby segments
- **Wall Sliding**: Occurs when `yspeed > 0` and `input * wall_normal < 0`
- **Wall Jumping**: Can jump off walls with different velocities based on input direction

#### Slope Mechanics
- **Slope Detection**: Based on floor normal vectors from collision resolution
- **Uphill Movement**: Special friction formula: `speed_scalar - friction_force * floor_y²`
- **Downhill Movement**: Enhanced acceleration when moving with slope gradient

## Input System

### Controls
- **Horizontal Movement**: -1 (left), 0 (none), 1 (right)
- **Jump**: 0 (not pressed), 1 (pressed)
- **Input Processing**: Evaluated once per frame in `ninja.think()`

### Input Buffering (Critical for AI Timing)
- **Jump Buffer**: 5-frame window for jump inputs while airborne
- **Floor Buffer**: 5-frame window after touching ground to still jump
- **Wall Buffer**: 5-frame window after touching wall to wall jump
- **Launch Pad Buffer**: 4-frame window after touching launch pad

**Buffer Mechanics:**
- Buffers increment each frame when active (-1 to max)
- New jump input while airborne initiates jump buffer
- Touching wall/floor initiates respective buffers
- Buffers enable delayed action execution for precise timing

## Physics Simulation

### Integration Order (Per Frame)
1. **Apply Drag**: `velocity *= drag_coefficient`
2. **Apply Gravity**: `velocity_y += gravity`
3. **Update Position**: `position += velocity`
4. **Collision Resolution**: 4 physics substeps
   - Physical collisions with entities
   - Physical collisions with tiles (32 depenetration iterations)
5. **Post-Collision**: Logical entity interactions, state updates
6. **Think**: Input processing and state transitions

### Collision Detection
- **Method**: Continuous collision detection with swept circles
- **Precision**: 32 iterations for depenetration resolution
- **Interpolation**: `sweep_circle_vs_tiles()` prevents tunneling
- **Collision Priority**: Tiles processed before entity logical collisions

### Collision Response
- **Depenetration**: Move objects apart using shortest separation vector
- **Velocity Projection**: Project velocity onto collision surface normal
- **Impact Detection**: Check for lethal impact velocities against floor/ceiling
- **Wall Normal Accumulation**: Sum normalized vectors for wall interaction

## Tile System

### Tile Types (38 total types, 0-37)
- **Type 0**: Empty space
- **Type 1**: Full solid block
- **Types 2-5**: Half tiles (top, right, bottom, left)
- **Types 6-9**: 45-degree slopes
- **Types 10-13**: Quarter moon (concave curves)
- **Types 14-17**: Quarter pipes (convex curves)
- **Types 18-25**: Mild slopes (various angles)
- **Types 26-33**: Steep slopes (various angles)
- **Types 34-37**: Glitched tiles (special collision properties)

### Collision Geometry
Each tile can contain:
- **Linear Segments**: Straight collision edges with orientation
- **Circular Segments**: Curved collision surfaces (radius 24 pixels)
- **Quadtree Optimization**: Spatial partitioning for efficient collision queries

## Entities and Interactive Elements

### Core Entities

#### Exit Door (Type 3)
- **Radius**: 12 pixels
- **Function**: Level completion when touched by ninja
- **Activation**: Only interactable after exit switch is activated
- **AI Note**: Goal state - prioritize path to activated exit

#### Exit Switch (Type 4)
- **Radius**: 6 pixels
- **Function**: Activates the exit door when touched
- **Behavior**: Single-use, disappears after activation
- **AI Note**: Critical objective - must be reached before exit

#### Gold (Type 2)
- **Radius**: 6 pixels
- **Function**: Collectible items (optional in most modes)
- **Behavior**: Disappears when collected, increments `ninja.gold_collected`

### Hazards and Interactive Elements

#### Toggle Mines (Type 1/21)
- **States**: Untoggled (safe, 3.5px), Toggling (transitioning, 4.5px), Toggled (deadly, 4px)
- **State Transitions**:
  - Untoggled→Toggling: Ninja touches mine
  - Toggling→Toggled: Ninja stops touching mine  
  - Toggled→Untoggled: Ninja touches again
- **AI Strategy**: Avoid toggled state, use for strategic blocking

#### Launch Pads (Type 10)
- **Radius**: 6 pixels
- **Boost Strength**: 36/7 pixels/frame in specified direction
- **Function**: Propels ninja with velocity `(boost_x * 2/3, boost_y * 2/3)`
- **Orientations**: 8 possible directions (0-7)
- **AI Note**: Powerful movement tool, plan trajectory carefully

#### Bounce Blocks (Type 17)
- **Size**: 9×9 pixel square
- **Physics**: Spring-based with stiffness 0.0222, dampening 0.98
- **Interaction**: 80% force applied to block, 20% to ninja
- **AI Strategy**: Use for momentum preservation and creative routing

#### Thwumps (Type 20)
- **Size**: 9×9 pixel square
- **Movement**: Forward speed 20/7, backward speed 8/7 pixels/frame
- **Behavior**: Charges toward ninja when in line of sight, returns to origin
- **States**: Immobile (0), Forward (1), Backward (-1)
- **Activation Range**: 2 × (9 + 10) = 38 pixels
- **AI Strategy**: Use timing and positioning to avoid charging face

#### Drones (Types 14, 26)
- **Radius**: 7.5 pixels (regular), 4 pixels (mini)
- **Movement**: Grid-based patrolling with 4 modes:
  - 0: Follow wall clockwise, 1: Follow wall counter-clockwise
  - 2: Wander clockwise, 3: Wander counter-clockwise
- **Speed**: 8/7 pixels/frame (regular), 1.3 pixels/frame (mini)
- **AI Strategy**: Predict patrol patterns, time movements accordingly

#### Death Balls (Type 25)
- **Radius**: 5 pixels (ninja collision), 8 pixels (environment collision)
- **Physics**: Acceleration 0.04, max speed 0.85 pixels/frame
- **Behavior**: Seeks ninja, bounces off walls, interacts with other death balls
- **AI Strategy**: Use walls and momentum to redirect threats

### Movement Aids

#### One-Way Platforms (Type 11)
- **Size**: 12×12 pixel square
- **Function**: Allows passage from one direction only
- **Collision**: Depends on approach angle and velocity
- **AI Note**: Critical for routing, understand approach requirements

#### Doors (Types 5, 6, 8)
- **Regular Doors (5)**: Open when ninja within 10 pixels, auto-close after 5 frames
- **Locked Doors (6)**: Require switch activation, permanent opening
- **Trap Doors (8)**: Start open, close permanently when switch activated
- **AI Strategy**: Plan switch timing for locked/trap doors

## Game Rules

### Victory Conditions
- Touch the exit door after activating the exit switch
- All gold collection is optional (in basic mode)

### Death Conditions
1. **Impact Death**: Hitting surface at velocity > 6 pixels/frame
2. **Crushing Death**: Being compressed below 0.05 threshold
3. **Entity Contact**: Touching mines, drones, thwumps, or other hazards
4. **Impact Formula**: `impact_vel > MAX_SURVIVABLE_IMPACT - 4/3 * abs(normal_y)`

### Level Progression
- Levels are procedurally generated or loaded from map files
- Three main generation types: Simple Horizontal, Jump Required, Maze
- Evaluation mode uses only Jump Required and Maze types

## AI Decision Framework

### Critical State Information
- **Position**: `(ninja.xpos, ninja.ypos)` normalized to [0,1]
- **Velocity**: `(ninja.xspeed, ninja.yspeed)` within [-3.333, 3.333]
- **State**: Current movement state (0-9) affects available actions
- **Buffers**: Active input buffers enable delayed actions
- **Physics**: Applied gravity/drag/friction indicate current physics mode

### Key Decision Points
1. **Jump Timing**: Use buffers for precise jump execution
2. **Wall Interactions**: Leverage wall normals for efficient movement
3. **Entity Avoidance**: Predict hazard states and patrol patterns
4. **Route Planning**: Sequence switch→exit with optimal entity interactions
5. **Momentum Management**: Use launch pads and bounce blocks strategically

### Action Space
- **Horizontal Input**: {-1, 0, 1}
- **Jump Input**: {0, 1}
- **Combined Actions**: 6 total combinations
- **Timing Sensitivity**: Frame-perfect inputs often required

## Technical Implementation

### Optimization Features
- **Spatial Partitioning**: 24×24 pixel grid cells with quadtree structure
- **Entity Management**: Active entity filtering and collision caching
- **Cache Systems**: sqrt calculations and cell neighborhood lookups
- **Deterministic**: Fixed timestep ensures consistent physics across runs

### Performance Considerations
- **Collision Detection**: O(log n) spatial queries via quadtree
- **Entity Updates**: Only active entities processed each frame
- **Memory Management**: Periodic cache clearing prevents growth
- **Headless Mode**: Supports training without rendering overhead
