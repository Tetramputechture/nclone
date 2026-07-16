## Overview

This is a simulation of the game N++, a 2D physics-based platformer where the player controls a ninja character through levels filled with obstacles, enemies, and objectives. The game features precise physics simulation with momentum-based movement, wall jumping, and various interactive elements.

## Level Structure

### Dimensions
- Level size: 42 * 23 grid cells (1056 * 600 pixels). See `MAP_TILE_WIDTH`, `MAP_TILE_HEIGHT`, `FULL_MAP_WIDTH_PX`, `FULL_MAP_HEIGHT_PX` in physics_constants.py
- Cell size: 24*24 pixels per grid cell. See `TILE_PIXEL_SIZE` in physics_constants.py
- Coordinate system: origin at top-left, X increases rightward, Y increases downward
- Visibility: the entire level is always visible to the player

### Tile Data Coordinate System
- 1-tile padding: during simulation, levels have a 1-tile border of solid (type 1) tiles around the playable area
- Tile data offset: tile data arrays exclude this padding, so coordinates are offset by -1 tile (-24 pixels) from map_data coordinates
- Coordinate conversion: when converting from map_data coordinates (e.g. ninja spawn at `map_data[1231]*6, map_data[1232]*6`) to tile data coordinates, subtract 24 pixels (1 tile) from both x and y
- Reason: this lets graph construction and pathfinding work with the inner playable area without accounting for the padding

### Time Limit
- Maximum duration: 20,000 frames per level
- Frame rate: 60 FPS (standard game speed)
- Real time limit: about 5.5 minutes per level

## Player Character (Ninja)

### Physical Properties
- Radius: 10 pixels (circular collision shape). See `NINJA_RADIUS` in physics_constants.py
- Spawn position: defined by map data at coordinates `(map_data[1231]*6, map_data[1232]*6)` in full map space
  - To convert to tile data coordinate space: subtract 24 pixels (1 tile) from both x and y coordinates
  - This accounts for the 1-tile solid padding around the level during simulation

### Movement States
The ninja has 9 distinct movement states with specific transition conditions:
1. Immobile (0): stationary on ground, no horizontal input
2. Running (1): moving horizontally on ground with input
3. Ground Sliding (2): sliding on ground with momentum, input opposite to movement
4. Jumping (3): actively jumping (gravity reduced), up to 45 frames
5. Falling (4): in air, falling or moving without jump input
6. Wall Sliding (5): sliding down a wall while holding toward it
7. Dead (6): post-death ragdoll physics active
8. Awaiting Death (7): single-frame transition to death state
9. Celebrating (8): victory state, reduced drag applied
10. Disabled (9): inactive state

### Physics Constants

#### Gravity
- Fall gravity: 0.06666666666666665 pixels/frame² (when falling or not jumping). See `GRAVITY_FALL` in physics_constants.py
- Jump gravity: 0.01111111111111111 pixels/frame² (when actively jumping). See `GRAVITY_JUMP` in physics_constants.py

#### Acceleration
- Ground acceleration: 0.06666666666666665 pixels/frame² (horizontal movement on ground). See `GROUND_ACCEL` in physics_constants.py
- Air acceleration: 0.04444444444444444 pixels/frame² (horizontal movement in air). See `AIR_ACCEL` in physics_constants.py

#### Speed Limits
- Maximum horizontal speed: 3.333 pixels/frame. See `MAX_HOR_SPEED` in physics_constants.py
- Maximum jump duration: 45 frames. See `MAX_JUMP_DURATION` in physics_constants.py

#### Drag and Friction
- Regular drag: 0.9933221725495059 (applied to both X and Y velocity each frame). See `DRAG_REGULAR` in physics_constants.py
- Slow drag: 0.8617738760127536 (applied in certain conditions). See `DRAG_SLOW` in physics_constants.py
- Ground friction: 0.9459290248857720 (applied to horizontal velocity on ground). See `FRICTION_GROUND` in physics_constants.py
- Ground friction (slow): 0.8617738760127536 (applied when moving slowly on ground). See `FRICTION_GROUND_SLOW` in physics_constants.py
- Wall friction: 0.9113380468927672 (applied to vertical velocity when wall sliding). See `FRICTION_WALL` in physics_constants.py

#### Death Conditions
- Maximum survivable impact: 6 pixels/frame (impact velocity that causes death). See `MAX_SURVIVABLE_IMPACT` in physics_constants.py
- Minimum survivable crushing: 0.05 (crushing threshold). See `MIN_SURVIVABLE_CRUSHING` in physics_constants.py

### Movement Mechanics

#### State Transitions
Ground states (0-2):
- State 0→1: horizontal input while stationary
- State 1→2: input opposite to movement direction or no input while moving
- State 2→1: input matching movement direction with sufficient projection
- State 2→0: low projection (<0.1) on flat ground

Air states (3-5):
- Any→4: becomes airborne without active jump
- State 3→4: jump duration exceeds 45 frames or jump input released
- State 4→5: touching wall while falling and holding toward wall
- State 5→4: not touching wall or input away from wall

#### Jumping
- Floor jump: from flat ground, applies velocity `(0, -2)`. See `JUMP_FLAT_GROUND_Y` in physics_constants.py
- Wall jump (regular): applies velocity `(1 * wall_normal, -1.4)`. See `JUMP_WALL_REGULAR_X`, `JUMP_WALL_REGULAR_Y` in physics_constants.py
- Wall jump (slide): from wall slide state, applies velocity `(2/3 * wall_normal, -1)`. See `JUMP_WALL_SLIDE_X`, `JUMP_WALL_SLIDE_Y` in physics_constants.py
- Slope jumping: mechanics based on slope angle and movement direction
  - Downhill with input: `(2/3 * slope_x, 2 * slope_y)`. See `JUMP_SLOPE_DOWNHILL_X`, `JUMP_SLOPE_DOWNHILL_Y` in physics_constants.py
  - Uphill perpendicular: `(2/3 * slope_x, 2 * slope_y)` with speed reset. See `JUMP_SLOPE_UPHILL_PERP_X`, `JUMP_SLOPE_UPHILL_PERP_Y` in physics_constants.py
  - Default uphill: `(0, -1.4)`. See `JUMP_SLOPE_UPHILL_FORWARD_Y` in physics_constants.py

#### Wall Interaction
- Wall detection: ninja can interact with walls within radius + 0.1 pixels
- Wall normal calculation: sum of normalized depenetration vectors from nearby segments
- Wall sliding: occurs when `yspeed > 0` and `input * wall_normal < 0`
- Wall jumping: can jump off walls with different velocities based on input direction

#### Slope Mechanics
- Slope detection: based on floor normal vectors from collision resolution
- Uphill movement: special friction formula `speed_scalar - friction_force * floor_y²`
- Downhill movement: enhanced acceleration when moving with slope gradient

## Input System

### Controls
- Horizontal movement: -1 (left), 0 (none), 1 (right)
- Jump: 0 (not pressed), 1 (pressed)
- Input processing: evaluated once per frame in `ninja.think()`

### Input Buffering (Critical for AI Timing)
- Jump buffer: 5-frame window for jump inputs while airborne
- Floor buffer: 5-frame window after touching ground to still jump
- Wall buffer: 5-frame window after touching wall to wall jump
- Launch pad buffer: 4-frame window after touching launch pad

Buffer mechanics:
- Buffers increment each frame when active (-1 to max)
- New jump input while airborne initiates jump buffer
- Touching wall/floor initiates respective buffers

## Physics Simulation

### Integration Order (Per Frame)
1. Apply drag: `velocity *= drag_coefficient`
2. Apply gravity: `velocity_y += gravity`
3. Update position: `position += velocity`
4. Collision resolution: 4 physics substeps
   - Physical collisions with entities
   - Physical collisions with tiles (32 depenetration iterations)
5. Post-collision: logical entity interactions, state updates
6. Think: input processing and state transitions

### Collision Detection
- Method: continuous collision detection with swept circles
- Precision: 32 iterations for depenetration resolution
- Interpolation: `sweep_circle_vs_tiles()` prevents tunneling
- Collision priority: tiles processed before entity logical collisions

### Collision Response
- Depenetration: move objects apart using shortest separation vector
- Velocity projection: project velocity onto collision surface normal
- Impact detection: check for lethal impact velocities against floor/ceiling
- Wall normal accumulation: sum normalized vectors for wall interaction

## Tile System

### Tile Types (38 total types, 0-37)
- Type 0: empty space
- Type 1: full solid block
- Types 2-5: half tiles (top, right, bottom, left)
- Types 6-9: 45-degree slopes
- Types 10-13: quarter moon (concave curves)
- Types 14-17: quarter pipes (convex curves)
- Types 18-25: mild slopes (various angles)
- Types 26-33: steep slopes (various angles)
- Types 34-37: glitched tiles (no collision, not used in the game)

### Collision Geometry
Each tile can contain:
- Linear segments: straight collision edges with orientation
- Circular segments: curved collision surfaces (radius 24 pixels)

Collision queries use the 24*24 grid directly: segments are stored per cell in `segment_dic[(x,y)]`,
and physics gathers segments by iterating overlapped cells.

## Entities and Interactive Elements

### Core Entities

#### Exit Door (Type 3)
- Radius: 12 pixels
- Function: level completion when touched by ninja
- Activation: only interactable after exit switch is activated
- AI note: goal state, prioritize path to activated exit

#### Exit Switch (Type 4)
- Radius: 6 pixels
- Function: activates the exit door when touched
- Behavior: single-use, disappears after activation
- AI note: critical objective, must be reached before exit

#### Gold (Type 2)
- Radius: 6 pixels
- Function: collectible items (not used by our RL agent)
- Behavior: disappears when collected, increments `ninja.gold_collected`

### Hazards and Interactive Elements

#### Toggle Mines (Type 1/21)
- States: Untoggled (safe, 3.5px), Toggling (transitioning, 4.5px), Toggled (deadly, 4px). See `TOGGLE_MINE_RADII` in physics_constants.py
- State transitions:
  - Untoggled→Toggling: ninja touches mine
  - Toggling→Toggled: ninja stops touching mine
  - Toggled→Untoggled: ninja touches again
- AI strategy: avoid toggled state, use for strategic blocking

#### Launch Pads (Type 10)
- Radius: 6 pixels
- Boost strength: 36/7 pixels/frame in specified direction
- Function: propels ninja with velocity `(boost_x * 2/3, boost_y * 2/3)`. See `JUMP_LAUNCH_PAD_BOOST_SCALAR`, `JUMP_LAUNCH_PAD_BOOST_FACTOR` in physics_constants.py
- Orientations: 8 possible directions (0-7)
- AI note: powerful movement tool, plan trajectory carefully

#### Bounce Blocks (Type 17)
- Size: 9*9 pixel square. See `BOUNCE_BLOCK_SIZE` in physics_constants.py
- Physics: spring-based system. See bounce block constants in physics_constants.py
- Interaction: force distribution system for ninja-block interaction
- AI strategy: use for momentum preservation and creative routing

#### Thwumps (Type 20)
- Size: 9*9 pixel square. See `THWUMP_SEMI_SIDE` in physics_constants.py
- Facing: each thwump faces one of four sides (up, down, left, right)
- Deadly side: only the side the thwump is facing is deadly during a charge; other sides behave as solid walls or floors and can be safely touched or stood on
- Movement: forward speed 20/7, backward speed 8/7 pixels/frame. See `THWUMP_FORWARD_SPEED`, `THWUMP_BACKWARD_SPEED` in physics_constants.py
- Behavior: charges toward ninja when in line of sight, returns to origin
- States: Immobile (0), Forward (1), Backward (-1)
- Activation range: 38 pixels. See `THWUMP_ACTIVATION_RANGE` in physics_constants.py
- Special interaction: horizontally-moving thwumps can be 'ridden' on top of, enabling advanced movement and routing strategies
- AI strategy: use timing and positioning to avoid the deadly face; consider using non-deadly sides for traversal or as moving platforms

#### Drones (Types 14, 26)
- Radius: 7.5 pixels (regular), 4 pixels (mini). See `DRONE_RADIUS`, `MINI_DRONE_RADIUS` in physics_constants.py
- Movement: grid-based patrolling with 4 modes:
  - 0: follow wall clockwise, 1: follow wall counter-clockwise
  - 2: wander clockwise, 3: wander counter-clockwise
- Speed: various speeds based on drone type. See `DRONE_LAUNCH_SPEED` in physics_constants.py
- AI strategy: predict patrol patterns, time movements accordingly

#### Death Balls (Type 25)
- Radius: 5 pixels (ninja collision), 8 pixels (environment collision)
- Physics: acceleration 0.04, max speed 0.85 pixels/frame
- Behavior: seeks ninja, bounces off walls, interacts with other death balls
- AI strategy: use walls and momentum to redirect threats

### Movement Aids

#### One-Way Platforms (Type 11)
- Size: 12*12 pixel square. See `ONE_WAY_PLATFORM_SEMI_SIDE` in physics_constants.py
- Function: allows passage from one direction only
- Collision: depends on approach angle and velocity
- AI note: critical for routing, understand approach requirements

#### Doors (Types 5, 6, 8)
- Regular doors (5): open when ninja within 10 pixels, auto-close after 5 frames
- Locked doors (6): require switch activation, permanent opening
- Trap doors (8): start open, close permanently when switch activated
- AI strategy: plan switch timing for locked/trap doors

## Game Rules

### Victory Conditions
- Touch the exit door after activating the exit switch

### Death Conditions
1. Impact death: hitting surface at velocity > 6 pixels/frame
2. Crushing death: being compressed below 0.05 threshold
3. Entity contact: touching mines, drones, thwumps, or other hazards
4. Impact formula: `impact_vel > MAX_SURVIVABLE_IMPACT - 4/3 * abs(normal_y)`

### Action Space
- Horizontal input: {-1, 0, 1}
- Jump input: {0, 1}
- Combined actions: 6 total combinations
- Timing sensitivity: frame-perfect inputs often required
