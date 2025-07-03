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
The ninja has 6 distinct movement states:
1. **Immobile** (0): Stationary on ground
2. **Running** (1): Moving horizontally on ground
3. **Ground Sliding** (2): Sliding on ground with momentum
4. **Jumping** (3): Actively jumping (gravity reduced)
5. **Falling** (4): In air, falling
6. **Wall Sliding** (5): Sliding down a wall

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

#### Jumping
- **Floor Jump**: From flat ground, applies velocity `(0, -2)`
- **Wall Jump**: Applies velocity `(1 * wall_normal, -1.4)` or `(2/3 * wall_normal, -1)` for slide jumps
- **Slope Jumping**: Complex mechanics based on slope angle and movement direction
- **Jump Buffering**: 5-frame window for jump inputs
- **Coyote Time**: 5-frame window after leaving ground to still jump

#### Wall Interaction
- **Wall Detection**: Ninja can interact with walls within radius + 0.1 pixels
- **Wall Sliding**: Occurs when moving down a wall while holding toward it
- **Wall Jumping**: Can jump off walls with different velocities based on input direction

#### Slope Mechanics
- **Slope Detection**: Based on floor normal vectors
- **Uphill Movement**: Special friction formula applies
- **Downhill Movement**: Enhanced acceleration when moving with slope

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
- **Linear Segments**: Straight collision edges
- **Circular Segments**: Curved collision surfaces (radius 24 pixels)
- **Grid Edges**: Simplified collision for optimization

## Entities and Interactive Elements

### Core Entities

#### Exit Door (Type 3)
- **Radius**: 12 pixels
- **Function**: Level completion when touched by ninja
- **Activation**: Only interactable after exit switch is activated

#### Exit Switch (Type 4)
- **Radius**: 6 pixels
- **Function**: Activates the exit door when touched
- **Behavior**: Single-use, disappears after activation

#### Gold (Type 2)
- **Radius**: 6 pixels
- **Function**: Collectible items (score/completion tracking)
- **Behavior**: Disappears when collected

### Mines and Hazards

#### Toggle Mines (Type 1/21)
- **States**: Untoggled (safe), Toggling (transitioning), Toggled (deadly)
- **Radii**: Toggled: 4px, Untoggled: 3.5px, Toggling: 4.5px
- **Behavior**: Toggle between safe/deadly when ninja touches

### Doors

#### Regular Doors (Type 5)
- **Activation Radius**: 10 pixels
- **Behavior**: Opens when ninja approaches, closes after 5 frames without contact
- **Collision**: Creates 24-pixel linear segment when closed

#### Locked Doors (Type 6)
- **Activation Radius**: 5 pixels
- **Behavior**: Opens permanently when switch is activated
- **Function**: Requires key/switch to open

#### Trap Doors (Type 8)
- **Activation Radius**: 5 pixels
- **Behavior**: Starts open, closes permanently when switch is activated

### Movement Aids

#### Launch Pads (Type 10)
- **Radius**: 6 pixels
- **Boost Strength**: 36/7 pixels/frame
- **Function**: Propels ninja in specified direction
- **Orientations**: 8 possible directions

#### Bounce Blocks (Type 17)
- **Size**: 9×9 pixel square
- **Physics**: Spring-based movement with stiffness 0.0222, dampening 0.98
- **Interaction**: 80% force applied to block, 20% to ninja

### Enemies and Hazards

#### Thwumps (Type 20)
- **Size**: 9×9 pixel square
- **Movement**: Forward speed 20/7, backward speed 8/7 pixels/frame
- **Behavior**: Moves toward ninja when in line of sight, returns to origin
- **Collision**: Can crush ninja

#### Drones (Types 14, 26)
- **Radius**: 7.5 pixels
- **Movement**: Grid-based patrolling with 4 modes
- **Behavior**: Follow walls or wander, kill ninja on contact
- **Speed**: Variable based on type

#### Death Balls (Type 25)
- **Radius**: 5 pixels (ninja collision), 8 pixels (environment collision)
- **Physics**: Acceleration 0.04, max speed 0.85 pixels/frame
- **Behavior**: Seeks ninja, bounces off walls

### One-Way Platforms (Type 11)
- **Size**: 12×12 pixel square
- **Function**: Allows passage from one direction only
- **Orientations**: 8 possible directions

## Physics Simulation

### Collision Detection
- **Method**: Continuous collision detection with swept circles
- **Precision**: 32 iterations for depenetration resolution
- **Interpolation**: Prevents tunneling through thin walls

### Integration Order
1. **Apply Drag**: Velocity *= drag_coefficient
2. **Apply Gravity**: velocity_y += gravity
3. **Update Position**: position += velocity
4. **Collision Resolution**: Detect and resolve collisions
5. **State Updates**: Update ninja state and entity logic

### Collision Response
- **Depenetration**: Move objects apart to prevent overlap
- **Velocity Projection**: Project velocity onto collision surface
- **Impact Detection**: Check for lethal impact velocities

## Game Rules

### Victory Conditions
- Touch the exit door after activating the exit switch
- All gold collection is optional (in basic mode)

### Death Conditions
1. **Impact Death**: Hitting surface at velocity > 6 pixels/frame
2. **Crushing Death**: Being compressed below 0.05 threshold
3. **Entity Contact**: Touching mines, drones, or other hazards

### Level Progression
- Levels are procedurally generated or loaded from map files
- Three main generation types: Simple Horizontal, Jump Required, Maze
- Evaluation mode uses only Jump Required and Maze types

## Input System

### Controls
- **Horizontal Movement**: Left/Right or A/D keys
- **Jump**: Space or Up arrow
- **Reset**: R key (in test mode)

### Input Buffering
- **Jump Buffer**: 5-frame window for jump inputs
- **Floor Buffer**: 5-frame window after touching ground
- **Wall Buffer**: 5-frame window after touching wall
- **Launch Pad Buffer**: 4-frame window after touching launch pad

## Rendering and Observation

### Visual Representation
- **Player Frame**: 84×84 pixel view centered on ninja
- **Global View**: 176×100 pixel downsampled view of entire level
- **Grayscale**: All rendering in single-channel grayscale

### State Information
The game provides comprehensive state data including:
- Ninja position, velocity, and physics state
- Entity positions and states
- Time remaining
- Objective vectors (to switch and exit)
- Collision and interaction history

## Technical Implementation

### Grid System
- **Spatial Partitioning**: 24×24 pixel grid cells for optimization
- **Entity Management**: Entities tracked by grid position
- **Collision Optimization**: Quadtree structure for efficient collision queries

### Frame Rate
- **Simulation**: 60 FPS fixed timestep
- **Deterministic**: Consistent physics across runs
- **Headless Mode**: Supports training without rendering
