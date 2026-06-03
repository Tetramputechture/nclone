//! Physics constants for the N++ simulation.
//! This is the single source of truth -- Python imports these via PyO3.

// === ENTITY PHYSICS CONSTANTS ===
pub const NINJA_RADIUS: f64 = 10.0;
pub const NINJA_RADIUS_SQ: f64 = NINJA_RADIUS * NINJA_RADIUS;

// Entity collision radii
pub const EXIT_SWITCH_RADIUS: f64 = 6.0;
pub const EXIT_DOOR_RADIUS: f64 = 12.0;
pub const GOLD_RADIUS: f64 = 6.0;

// Locked door (type 6): 24px wall segment (radius 12) + switch trigger (radius 5).
pub const LOCKED_DOOR_RADIUS: f64 = 12.0;
pub const LOCKED_DOOR_SWITCH_RADIUS: f64 = 5.0;

// Toggle mine radii by state: 0=toggled(deadly), 1=untoggled(safe), 2=toggling
pub const TOGGLE_MINE_RADIUS_TOGGLED: f64 = 4.0;
pub const TOGGLE_MINE_RADIUS_UNTOGGLED: f64 = 3.5;
pub const TOGGLE_MINE_RADIUS_TOGGLING: f64 = 4.5;

pub fn toggle_mine_radius(state: u8) -> f64 {
    match state {
        0 => TOGGLE_MINE_RADIUS_TOGGLED,
        1 => TOGGLE_MINE_RADIUS_UNTOGGLED,
        2 => TOGGLE_MINE_RADIUS_TOGGLING,
        _ => TOGGLE_MINE_RADIUS_TOGGLED,
    }
}

// === MOVEMENT CONSTANTS ===
pub const MAX_HOR_SPEED: f64 = 3.333333333333333;
pub const MAX_VER_SPEED: f64 = 12.0;

// === GRAVITY AND PHYSICS ===
pub const GRAVITY_FALL: f64 = 0.06666666666666665;
pub const GRAVITY_JUMP: f64 = 0.01111111111111111;
pub const TERMINAL_VELOCITY: f64 = 12.0;

// === ACCELERATION ===
pub const GROUND_ACCEL: f64 = 0.06666666666666665;
pub const AIR_ACCEL: f64 = 0.04444444444444444;

// === DRAG ===
pub const DRAG_REGULAR: f64 = 0.9933221725495059; // 0.99^(2/3)
pub const DRAG_SLOW: f64 = 0.8617738760127536; // 0.80^(2/3)

// === FRICTION ===
pub const FRICTION_GROUND: f64 = 0.9459290248857720; // 0.92^(2/3)
pub const FRICTION_GROUND_SLOW: f64 = 0.8617738760127536; // 0.80^(2/3)
pub const FRICTION_WALL: f64 = 0.9113380468927672; // 0.87^(2/3)

// === JUMP MECHANICS ===
pub const MAX_JUMP_DURATION: i32 = 45;
pub const MAX_SURVIVABLE_IMPACT: f64 = 6.0;
pub const MIN_SURVIVABLE_CRUSHING: f64 = 0.05;

// Floor jump
pub const JUMP_FLOOR_X: f64 = 0.0;
pub const JUMP_FLOOR_Y: f64 = -2.0;

// Wall jump (regular)
pub const JUMP_WALL_REGULAR_X_MULTIPLIER: f64 = 1.0;
pub const JUMP_WALL_REGULAR_Y: f64 = -1.4;

// Wall jump (slide)
pub const JUMP_WALL_SLIDE_X_MULTIPLIER: f64 = 2.0 / 3.0;
pub const JUMP_WALL_SLIDE_Y: f64 = -1.0;

// Slope jump
pub const JUMP_SLOPE_DOWNHILL_X_MULTIPLIER: f64 = 2.0 / 3.0;
pub const JUMP_SLOPE_DOWNHILL_Y_MULTIPLIER: f64 = 2.0;
pub const JUMP_SLOPE_UPHILL_DEFAULT_Y: f64 = -1.4;
pub const JUMP_SLOPE_UPHILL_PERP_X_MULTIPLIER: f64 = 2.0 / 3.0;
pub const JUMP_SLOPE_UPHILL_PERP_Y_MULTIPLIER: f64 = 2.0;

// Launch pad
pub const JUMP_LAUNCH_PAD_BOOST_SCALAR: f64 = 1.7;
pub const JUMP_LAUNCH_PAD_BOOST_FACTOR: f64 = 2.0 / 3.0;

// === MAP CONSTANTS ===
pub const MAP_WIDTH: usize = 44;
pub const MAP_HEIGHT: usize = 25;
pub const TILE_SIZE: f64 = 24.0;
pub const TILE_SIZE_I: i32 = 24;

// Half-cell grid dimensions
pub const HALF_GRID_WIDTH: usize = 89; // MAP_WIDTH * 2 + 1
pub const HALF_GRID_HEIGHT: usize = 51; // MAP_HEIGHT * 2 + 1
