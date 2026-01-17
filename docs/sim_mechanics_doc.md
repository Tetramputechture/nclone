## Overview

This is a simulation of the game N++. N++ is a 2D physics-based platformer where the player controls a ninja character through levels filled with obstacles, enemies, and objectives. The game features precise physics simulation with momentum-based movement, wall jumping, and various interactive elements.

## Level Structure

### Dimensions
- **Level Size**: 42 * 23 grid cells (1056 * 600 pixels) - see `MAP_TILE_WIDTH`, `MAP_TILE_HEIGHT`, `FULL_MAP_WIDTH_PX`, `FULL_MAP_HEIGHT_PX` in physics_constants.py
- **Cell Size**: 24*24 pixels per grid cell - see `TILE_PIXEL_SIZE` in physics_constants.py
- **Coordinate System**: Origin at top-left, X increases rightward, Y increases downward
- **Visibility**: The entire level is always visible to the player

### Tile Data Coordinate System
- **1-Tile Padding**: During simulation, levels have a 1-tile border of solid (type 1) tiles around the playable area
- **Tile Data Offset**: Tile data arrays exclude this padding, so coordinates are offset by -1 tile (-24 pixels) from map_data coordinates
- **Coordinate Conversion**: When converting from map_data coordinates (e.g., ninja spawn at `map_data[1231]*6, map_data[1232]*6`) to tile data coordinates, subtract 24 pixels (1 tile) from both x and y
- **Reason**: This allows graph construction and pathfinding to work with the inner playable area without accounting for the padding

### Time Limit
- **Maximum Duration**: 20,000 frames per level
- **Frame Rate**: 60 FPS (standard game speed)
- **Real Time Limit**: ~5.5 minutes per level

## Player Character (Ninja)

### Physical Properties
- **Radius**: 10 pixels (circular collision shape) - see `NINJA_RADIUS` in physics_constants.py
- **Spawn Position**: Defined by map data at coordinates `(map_data[1231]*6, map_data[1232]*6)` in full map space
  - To convert to tile data coordinate space: subtract 24 pixels (1 tile) from both x and y coordinates
  - This accounts for the 1-tile solid padding around the level during simulation

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
- **Fall Gravity**: 0.06666666666666665 pixels/frame² (when falling or not jumping) - see `GRAVITY_FALL` in physics_constants.py
- **Jump Gravity**: 0.01111111111111111 pixels/frame² (when actively jumping) - see `GRAVITY_JUMP` in physics_constants.py

#### Acceleration
- **Ground Acceleration**: 0.06666666666666665 pixels/frame² (horizontal movement on ground) - see `GROUND_ACCEL` in physics_constants.py
- **Air Acceleration**: 0.04444444444444444 pixels/frame² (horizontal movement in air) - see `AIR_ACCEL` in physics_constants.py

#### Speed Limits
- **Maximum Horizontal Speed**: 3.333 pixels/frame - see `MAX_HOR_SPEED` in physics_constants.py
- **Maximum Jump Duration**: 45 frames - see `MAX_JUMP_DURATION` in physics_constants.py

#### Drag and Friction
- **Regular Drag**: 0.9933221725495059 (applied to both X and Y velocity each frame) - see `DRAG_REGULAR` in physics_constants.py
- **Slow Drag**: 0.8617738760127536 (applied in certain conditions) - see `DRAG_SLOW` in physics_constants.py
- **Ground Friction**: 0.9459290248857720 (applied to horizontal velocity on ground) - see `FRICTION_GROUND` in physics_constants.py
- **Ground Friction (Slow)**: 0.8617738760127536 (applied when moving slowly on ground) - see `FRICTION_GROUND_SLOW` in physics_constants.py
- **Wall Friction**: 0.9113380468927672 (applied to vertical velocity when wall sliding) - see `FRICTION_WALL` in physics_constants.py

#### Death Conditions
- **Maximum Survivable Impact**: 6 pixels/frame (impact velocity that causes death) - see `MAX_SURVIVABLE_IMPACT` in physics_constants.py
- **Minimum Survivable Crushing**: 0.05 (crushing threshold) - see `MIN_SURVIVABLE_CRUSHING` in physics_constants.py

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
- **Floor Jump**: From flat ground, applies velocity `(0, -2)` - see `JUMP_FLAT_GROUND_Y` in physics_constants.py
- **Wall Jump (Regular)**: Applies velocity `(1 * wall_normal, -1.4)` - see `JUMP_WALL_REGULAR_X`, `JUMP_WALL_REGULAR_Y` in physics_constants.py
- **Wall Jump (Slide)**: From wall slide state, applies velocity `(2/3 * wall_normal, -1)` - see `JUMP_WALL_SLIDE_X`, `JUMP_WALL_SLIDE_Y` in physics_constants.py
- **Slope Jumping**: Complex mechanics based on slope angle and movement direction
  - Downhill with input: `(2/3 * slope_x, 2 * slope_y)` - see `JUMP_SLOPE_DOWNHILL_X`, `JUMP_SLOPE_DOWNHILL_Y` in physics_constants.py
  - Uphill perpendicular: `(2/3 * slope_x, 2 * slope_y)` with speed reset - see `JUMP_SLOPE_UPHILL_PERP_X`, `JUMP_SLOPE_UPHILL_PERP_Y` in physics_constants.py
  - Default uphill: `(0, -1.4)` - see `JUMP_SLOPE_UPHILL_FORWARD_Y` in physics_constants.py

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
- **Types 34-37**: Glitched tiles (these have no collision and are not used in the game)

### Collision Geometry
Each tile can contain:
- **Linear Segments**: Straight collision edges with orientation
- **Circular Segments**: Curved collision surfaces (radius 24 pixels)
  
Collision queries use the 24*24 grid directly: segments are stored per cell in `segment_dic[(x,y)]`,
and physics gathers segments by iterating overlapped cells. 

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
- **Function**: Collectible items (not used in by our RL agent)
- **Behavior**: Disappears when collected, increments `ninja.gold_collected`

### Hazards and Interactive Elements

#### Toggle Mines (Type 1/21)
- **States**: Untoggled (safe, 3.5px), Toggling (transitioning, 4.5px), Toggled (deadly, 4px) - see `TOGGLE_MINE_RADII` in physics_constants.py
- **State Transitions**:
  - Untoggled→Toggling: Ninja touches mine
  - Toggling→Toggled: Ninja stops touching mine  
  - Toggled→Untoggled: Ninja touches again
- **AI Strategy**: Avoid toggled state, use for strategic blocking

#### Launch Pads (Type 10)
- **Radius**: 6 pixels
- **Boost Strength**: 36/7 pixels/frame in specified direction
- **Function**: Propels ninja with velocity `(boost_x * 2/3, boost_y * 2/3)` - see `JUMP_LAUNCH_PAD_BOOST_SCALAR`, `JUMP_LAUNCH_PAD_BOOST_FACTOR` in physics_constants.py
- **Orientations**: 8 possible directions (0-7)
- **AI Note**: Powerful movement tool, plan trajectory carefully

#### Bounce Blocks (Type 17)
- **Size**: 9*9 pixel square - see `BOUNCE_BLOCK_SIZE` in physics_constants.py
- **Physics**: Spring-based system - see bounce block constants in physics_constants.py
- **Interaction**: Force distribution system for ninja-block interaction
- **AI Strategy**: Use for momentum preservation and creative routing

#### Thwumps (Type 20)
- **Size**: 9*9 pixel square - see `THWUMP_SEMI_SIDE` in physics_constants.py
- **Facing**: Each thwump faces one of four sides (up, down, left, right)
- **Deadly Side**: Only the side the thwump is facing is deadly during a charge; other sides behave as solid walls or floors and can be safely touched or stood on
- **Movement**: Forward speed 20/7, backward speed 8/7 pixels/frame - see `THWUMP_FORWARD_SPEED`, `THWUMP_BACKWARD_SPEED` in physics_constants.py
- **Behavior**: Charges toward ninja when in line of sight, returns to origin
- **States**: Immobile (0), Forward (1), Backward (-1)
- **Activation Range**: 38 pixels - see `THWUMP_ACTIVATION_RANGE` in physics_constants.py
- **Special Interaction**: Horizontally-moving thwumps can be 'ridden' on top of, enabling advanced movement and routing strategies
- **AI Strategy**: Use timing and positioning to avoid the deadly face; consider using non-deadly sides for traversal or as moving platforms

#### Drones (Types 14, 26)
- **Radius**: 7.5 pixels (regular), 4 pixels (mini) - see `DRONE_RADIUS`, `MINI_DRONE_RADIUS` in physics_constants.py
- **Movement**: Grid-based patrolling with 4 modes:
  - 0: Follow wall clockwise, 1: Follow wall counter-clockwise
  - 2: Wander clockwise, 3: Wander counter-clockwise
- **Speed**: Various speeds based on drone type - see `DRONE_LAUNCH_SPEED` in physics_constants.py
- **AI Strategy**: Predict patrol patterns, time movements accordingly

#### Death Balls (Type 25)
- **Radius**: 5 pixels (ninja collision), 8 pixels (environment collision)
- **Physics**: Acceleration 0.04, max speed 0.85 pixels/frame
- **Behavior**: Seeks ninja, bounces off walls, interacts with other death balls
- **AI Strategy**: Use walls and momentum to redirect threats

### Movement Aids

#### One-Way Platforms (Type 11)
- **Size**: 12*12 pixel square - see `ONE_WAY_PLATFORM_SEMI_SIDE` in physics_constants.py
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

### Death Conditions
1. **Impact Death**: Hitting surface at velocity > 6 pixels/frame
2. **Crushing Death**: Being compressed below 0.05 threshold
3. **Entity Contact**: Touching mines, drones, thwumps, or other hazards
4. **Impact Formula**: `impact_vel > MAX_SURVIVABLE_IMPACT - 4/3 * abs(normal_y)`

### Action Space
- **Horizontal Input**: {-1, 0, 1}
- **Jump Input**: {0, 1}
- **Combined Actions**: 6 total combinations
- **Timing Sensitivity**: Frame-perfect inputs often required
