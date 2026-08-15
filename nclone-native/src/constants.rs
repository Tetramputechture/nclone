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

// One-way platform (type 11): directional 24px barrier (semi-side 12). Mirrors
// nclone/constants/physics_constants.py:ONE_WAY_PLATFORM_SEMI_SIDE.
pub const ONE_WAY_PLATFORM_SEMI_SIDE: f64 = 12.0;

// Launch pad (type 10): directional impulse. entity_launch_pad.py:RADIUS/BOOST.
pub const LAUNCH_PAD_RADIUS: f64 = 6.0;
pub const LAUNCH_PAD_BOOST: f64 = 36.0 / 7.0;

// Boost pad (type 24): adds 2 to the ninja's speed norm on initial contact.
pub const BOOST_PAD_RADIUS: f64 = 6.0;

// Drones (types 14, 26): grid-walking patrollers.
pub const DRONE_RADIUS: f64 = 7.5;
pub const DRONE_GRID_SIZE: f64 = 24.0;
pub const DRONE_ZAP_SPEED: f64 = 8.0 / 7.0;
pub const MINI_DRONE_RADIUS: f64 = 4.0;
pub const MINI_DRONE_GRID_SIZE: f64 = 12.0;
pub const MINI_DRONE_SPEED: f64 = 1.3;

// Bounce block (type 17): spring-mass square. NOTE the semi-side is 9, taken
// from entity_bounce_block.py:SEMI_SIDE -- NOT the 4.5 in
// physics_constants.py:BOUNCE_BLOCK_SEMI_SIDE, which the simulator never reads.
pub const BOUNCE_BLOCK_SEMI_SIDE: f64 = 9.0;
pub const BOUNCE_BLOCK_STIFFNESS: f64 = 0.02222222222222222;
pub const BOUNCE_BLOCK_DAMPENING: f64 = 0.98;
pub const BOUNCE_BLOCK_STRENGTH: f64 = 0.2;

// Thwump (type 20): charges on line of sight.
pub const THWUMP_SEMI_SIDE: f64 = 9.0;
pub const THWUMP_FORWARD_SPEED: f64 = 20.0 / 7.0;
pub const THWUMP_BACKWARD_SPEED: f64 = 8.0 / 7.0;

// Shove thwump (type 28): launches on contact; deadly inner core.
pub const SHOVE_THWUMP_SEMI_SIDE: f64 = 12.0;
pub const SHOVE_THWUMP_RADIUS: f64 = 8.0;

// Death ball (type 25): seeks the ninja.
pub const DEATH_BALL_RADIUS: f64 = 5.0; // vs ninja
pub const DEATH_BALL_RADIUS2: f64 = 8.0; // vs tiles and other balls
pub const DEATH_BALL_ACCELERATION: f64 = 0.04;
pub const DEATH_BALL_MAX_SPEED: f64 = 0.85;
pub const DEATH_BALL_DRAG_MAX_SPEED: f64 = 0.9;
pub const DEATH_BALL_DRAG_NO_TARGET: f64 = 0.95;

// ---------------------------------------------------------------------------
// Entities reverse-engineered from npp.dll. See docs/ENTITY_INTEGRATION_SPEC.md
// for the address of every function and the [V]/[D]/[I]/[U] confidence tag on
// every number. Values marked [I] below were chosen, not read.
// ---------------------------------------------------------------------------

// Floor guard (type 16), spec §8.1. Speed is the n/7 family (§6.4).
pub const FLOOR_GUARD_SPEED: f64 = 24.0 / 7.0;
/// `+0x28`: the step base AND the line-of-sight band offset, both 6.0.
pub const FLOOR_GUARD_OFFSET: f64 = 6.0;
/// `0x1056C360`, the cell size, reused as the detection band height.
pub const FLOOR_GUARD_BAND_HEIGHT: f64 = 24.0;
/// `0x10571CB8`, added to `step` on the frame the guard stops.
pub const FLOOR_GUARD_STOP_NUDGE: f64 = 0.01;
/// Kill radius. The decompiled logical collision passes `+0x28`; see the Python
/// class docstring for why the literal reading beats §8.0.1's summarised 12.0.
pub const FLOOR_GUARD_RADIUS: f64 = 6.0;
/// `0x10572340`, the cell-walk loop bound.
pub const FLOOR_GUARD_MAX_CELL_WALK: i32 = 100;
/// The guard walks the **half-cell** grid, not the 24px tile grid. `move` and
/// `think` divide by `*(double *)(param_1 + 0x3203F8)`, a field of the grid
/// object at `sim+0x3203E8`, and that object is built
/// `FUN_1050CDB0(sim + 0x3203E8, 44*2, 25*2, 24.0 * 0.5)` -- 88 x 50 at **12.0**
/// (`TILE_SEGMENT_TABLES.md` §7). The port used 24 and so probed one edge per
/// tile instead of one per half-tile.
pub const FLOOR_GUARD_CELL: f64 = 12.0;

// Eye bat (type 27), spec §8.2.
/// Radius against the ninja: `logical_collision` passes `DAT_10571CF8 = 5.0`.
/// The `12.0` this used to be is the push-vector magnitude, not a radius, and
/// the `12.0f` opening the const block is a frame count (the render pupil
/// extends over 12 frames). See ENTITY_RE_ROUND4_FINDINGS.md §10.5.
pub const EYE_BAT_RADIUS: f64 = 5.0;
/// `FUN_1040ADA0`'s argument -- distinct from the ninja-contact radius.
pub const EYE_BAT_TILE_RADIUS: f64 = 5.0;
/// `0x10571D98`, shared with the turret.
pub const EYE_BAT_ACCELERATION: f64 = 0.1;
/// `0x10572670`/`0x10572678`. A per-AXIS multiplicative decay, NOT a max speed --
/// the same number plays a different role for the death ball (§8.2).
pub const EYE_BAT_DRAG_NO_TARGET: f64 = 0.85;
/// `0x10572398`.
pub const EYE_BAT_MAX_SPEED: f64 = 3.0;
pub const EYE_BAT_MAX_CHASE_TIMER: i32 = 12;

// Chaser drone (type 15), spec §8.4. Chase speed is 2x patrol.
pub const DRONE_CHASER_CORRIDOR_HALF_WIDTH: f64 = 12.0;

// Shooter drones (types 12, 13), spec §8.3.
/// `0x10572390`.
pub const DRONE_CHAINGUN_SPEED: f64 = 4.0 / 7.0;
/// `0x10572478` -- the slowest drone in the game.
pub const DRONE_LASER_SPEED: f64 = 2.0 / 7.0;

// Rocket (18) / player-controlled rocket (29), spec §8.8/§8.9.
/// `+0x28`, the n/7 family again.
/// `+0x20` (`DAT_10572AD0`) -- the rocket's terminal speed.
pub const ROCKET_MAX_SPEED: f64 = 16.0 / 7.0;
/// `+0x28` (`DAT_10572AD8`) -- its initial per-frame speed increment.
pub const ROCKET_INITIAL_STEP: f64 = 2.0 / 45.0;
/// `+0x38` (`DAT_10572A10`) -- homing gain.
pub const ROCKET_TURN_RATE: f64 = 1.0 / 15.0;
/// `+0x48` (`DAT_105729E8`) -- frames of lead used when homing.
pub const ROCKET_LEAD_TIME: f64 = 1.5;
/// `+0x30`. Speed is MULTIPLICATIVE. The ctor stores `pow(1.1, 2/3)`, computed at construction from the same
/// `2/3` (`0x10572AC0`) the ninja ctor uses -- which is why a byte search for
/// the value finds nothing. A hardcoded 1.1 is the same class of bug round 3
/// closed for the ninja's drag and friction.
/// Written as the literal Python produces for `1.1 ** (2 / 3)` so the two sims
/// stay bit-identical; `powf` at runtime is not guaranteed to agree.
pub const ROCKET_SPEED_MULTIPLIER: f64 = 1.0656022367666107;
/// Radius against the NINJA. `EntityRocket::logical_collision` passes a literal
/// immediate `0` as `param_3` of `FUN_1050C650`, so the only radius in the test
/// is the ninja's own 10px. The `4.0` (`0x10571CF0`) that used to sit here is
/// the scale on the returned push vector; using it as a radius made the hit
/// circle 14px instead of 10 -- about +96% area.
pub const ROCKET_RADIUS: f64 = 0.0;
/// Safety bound on the odometer; every level is walled so the sweep normally
/// terminates a rocket first. The real bound is unread.
pub const ROCKET_MAX_TRAVEL: f64 = 4096.0;
/// `+0x40`, the launcher's telegraph in frames: `+0x88` counts up by one a frame
/// and the rocket leaves the tube when it reaches this. No longer inferred -- the
/// field sits inside the captured 176-byte stride and reads **15** on all 41
/// captures containing a launcher, and all 105 captured launches fire on the
/// frame the counter would reach exactly 15. It is also what stops the launcher
/// re-firing the instant a rocket bursts: without it, maps containing a rocket
/// fell from 59% demo wins to 31%.
pub const ROCKET_LAUNCH_DELAY: f64 = 15.0;
/// `0x10571D90`. **[I]** radians per frame per unit of horizontal input.
pub const PC_ROCKET_TURN_RATE: f64 = 0.05;

// Turret (type 19). Ctor binding VERIFIED against 0x10418F60
// (docs/ENTITY_RE_ROUND2_FINDINGS.md §1). Round 1 guessed every one of these and
// got them all wrong, which is why the turret killed 145 of 187 expert demos.
//
// All three arrays are FOUR long, not three: the band walk is
//     i = 0; while (i < 3 && thresh[i] > d2) { i++; band = i; }
// so `i` reaches 3 and indexes `+0x38 + i*8` / `+0x58 + i*8`.

/// `+0x20`/`+0x28`/`+0x30`: 96², 42², 24². Compared against the squared distance
/// from the AIM POINT to the predicted ninja position -- the aim ERROR, not range.
pub const TURRET_BANDS: [f64; 3] = [9216.0, 1764.0, 576.0];
/// `+0x38`..`+0x50`, index 0 = widest error.
pub const TURRET_AIM_CONVERGENCE: [f64; 4] = [0.02, 7.0 / 300.0, 1.0 / 30.0, 1.0 / 30.0];
/// `+0x58`..`+0x70`. **Band 0 is `0.0` and that is the safety valve**: while the
/// aim is more than 96px off the turret accumulates no charge and cannot fire.
/// The aim still eases in at 2%/frame, so it converges on a stationary target --
/// but a moving ninja holds the error above the gate and is never shot at.
pub const TURRET_CHARGE_RATES: [f64; 4] = [0.0, 0.5, 1.5, 3.5];
/// `+0xC8`. With the rates above: 180 frames at 42-96px, 60 at 24-42px, ~26
/// inside one cell, never beyond 96px.
pub const TURRET_CHARGE_THRESHOLD: f64 = 90.0;
/// `+0xD0`.
pub const TURRET_FIRE_DELAY: f64 = 15.0;
/// `+0xD8`. Frames spent in state 3 after the shot before tracking resumes. The
/// ctor loads `+0xD0` and `+0xD8` from one 16-byte block (`_DAT_10572CD0`), and
/// the round-9 capture of `10138417` has the turret fire on frame 164 and leave
/// state 3 on 179 -- so both halves are `15.0`.
pub const TURRET_RECOVER_DELAY: f64 = 15.0;
/// `+0x78`.
pub const TURRET_LEAD_TIME: f64 = 1.5;
/// `+0xF0`, written by the fire path. Decremented every think and clamped at 0,
/// but **never read by the simulation** -- state 0 has no gate on it, so it is a
/// muzzle-flash/audio counter.
pub const TURRET_MUZZLE_TIMER: i32 = 9;

// Dual laser (type 23), spec §8.6.
pub const DUAL_LASER_RADIUS: f64 = 5.9;
/// `0x10572508`, the per-frame angular step in spinner mode.
pub const DUAL_LASER_SPIN_STEP: f64 = 0.010471975;
/// `0x10572500`, the hit tolerance -- exactly HALF a step, by design, so the
/// accepted arc equals the arc swept (§8.6).
pub const DUAL_LASER_HIT_TOLERANCE: f64 = 0.0052359875;
/// `(double)(float)pi` and `(double)(float)2pi` as stored in `.rdata`: the game
/// computed these in `float` and widened, so they are NOT std::f64::consts
/// values. The wrap point is measurably not 6.283185307179586 (§6.3).
pub const PI_F32_WIDENED: f64 = 3.1415927410125732;
pub const TAU_F32_WIDENED: f64 = 6.283185005187988;
pub const DUAL_LASER_SURFACE_SPEED: f64 = 0.1;
/// A HARD loop bound, not a convergence test -- observable at corners (§8.6).
pub const DUAL_LASER_SURFACE_ITERATIONS: usize = 10;
/// `0.1 / (5.9 * pi)`.
/// `+0xA8`, an `atan2` the ctor computes from `+0xA0` and the 24px cell:
/// `+0xA0 = sqrt((0.1k*24)^2 + (18.1k*24 - 24)^2) * 1.000001`, `k = 1/sqrt(327.62)`
/// -- which finally assigns `18.1`, `327.62` and `1.000001`. Matches the value
/// this port had invented to seven digits, so the corner rate was never the bug.
/// `atan2(SURFACE_SPEED, 24 - RADIUS)` = `atan2(0.1, 18.1)`. Was the invented
/// literal `0.005524747952943415`, which is 5.8e-8 short; a laser pivoting a
/// corner takes ten sub-steps a frame, so that left the game's normal on frame
/// 1. Not a `const fn` -- `f64::atan2` is not const -- so it is written out.
pub const DUAL_LASER_CORNER_SPEED: f64 = 0.005524805665672641;
/// `0x10572510`. The beam angle is LERPED toward the surface normal at 20% a
/// frame, it does not snap to it.
pub const DUAL_LASER_BEAM_LERP: f64 = 0.2;
/// `0x105723A0` is a true f64 pi and bounds the wrap; `0x105723A8` is a
/// HAND-TYPED 7-digit 2pi and is the step. Verbatim -- a full-precision tau
/// drifts. A third pi in this one entity, distinct from the PI_F32 pair.
pub const DUAL_LASER_WRAP_BOUND: f64 = 3.141592653589793;
pub const DUAL_LASER_WRAP_STEP: f64 = 6.283185;
/// `0x1056C358`, the surface search half-extent.
pub const DUAL_LASER_SURFACE_GATHER: f64 = 12.0;
/// `0x10571CB8`, the dead-end / reverse test.
pub const DUAL_LASER_DEAD_END_TOLERANCE: f64 = 0.01;
/// `_PTR_105724F8` = FLT_EPSILON: did the contact point move (slide) or not
/// (pinned on a corner)?
pub const DUAL_LASER_ANCHOR_EPSILON: f64 = 1.1920928955078125e-07;

// Evil ninja (type 22), spec §8.10.
/// `0x10571D08` dormant, `0x10570620` active.
pub const EVIL_NINJA_RADIUS_DORMANT: f64 = 10.0;
pub const EVIL_NINJA_RADIUS_ACTIVE: f64 = 6.0;
pub const EVIL_NINJA_WAIT_FRAMES: i32 = 120;
pub const EVIL_NINJA_POST_FRAMES: i32 = 30;
pub const EVIL_NINJA_FADE_DIVISOR: f64 = 120.0;
/// 8 ninjas x 18000 records in the game; 18000 frames is 5 minutes at 60Hz. The
/// write index wraps at 17999.
pub const NINJA_TAPE_LENGTH: usize = 18000;

// Regular door (type 5) activation radius; trap door (type 8) switch radius.
pub const REGULAR_DOOR_RADIUS: f64 = 10.0;
pub const TRAP_DOOR_RADIUS: f64 = 5.0;

// Toggle mine radii by state: 0=toggled(deadly), 1=untoggled(safe), 2=toggling.
//
// Only the state-0 entry is the KILL radius, and that one is confirmed: the
// mine's `+0x28` is what `FUN_10417F30` hands to `FUN_1050C650`, and
// `tools/npptrace/minefields.py` reads it as 4.0 over 36 captures and 326,539
// mine-frames. It is 4.0 in *every* state there -- but propagating that to all
// three entries costs 44 demos, because 3.5 and 4.5 do not govern the kill.
// They govern the two contact tests in `think` (untoggled -> toggling and
// toggling -> toggled), which live in a different function and do not read
// `+0x28`. The state machine they drive agrees with the game on 100% of 207,895
// captured mine-frames, so leave them alone; `+0x28` is not the field behind
// them.
pub const TOGGLE_MINE_RADIUS_TOGGLED: f64 = 4.0;
pub const TOGGLE_MINE_RADIUS_UNTOGGLED: f64 = 3.5;
pub const TOGGLE_MINE_RADIUS_TOGGLING: f64 = 4.5;

/// Radius by state. `+0x28` reads a flat 4.0 in every captured frame of every
/// state, but that is not the collision radius: a flat 4.0 costs 13 demos.
///
/// **Note there is a second copy of this in `entities/toggle_mine.rs`
/// (`radius_for_state`), and that is the one the simulation calls.** Editing
/// this one alone changes nothing, which is exactly how a python-only change
/// once got measured as neutral on native.
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
/// `Ninja::jump` scales `(jx, jy)` to `(jx * 2/3, jy * 2)`.
/// `Ninja::jump` (`docs/decomp/R5_07_jump_apply_105110C0.c:24-26`) scales its two
/// raw arguments by three ninja fields, evaluated LEFT TO RIGHT:
///
/// ```text
/// x_impulse = (field(+0xE8) * jx) * field(+0x1C8)
/// y_impulse = ((field(+0xE8) + field(+0xF0)) * jy) * field(+0x1C8)
/// ```
///
/// Field values read live out of the running game (`docs/game_traces/`).
/// DO NOT collapse to `jx * 2/3` / `jy * 2` -- algebraically identical,
/// numerically 1 ULP apart, and that ULP decides demos.
pub const JUMP_SCALE_A: f64 = 1.0; // ninja + 0xE8
pub const JUMP_SCALE_B: f64 = 2.0; // ninja + 0xF0
pub const JUMP_SCALE_C: f64 = 2.0 / 3.0; // ninja + 0x1C8

/// Raw `Ninja::jump` arguments for the wall rows
/// (`NINJA_update_10515300.c:387-396`); `field(+0x118)` reads 0.0 in every
/// captured frame, so the raw Y is just `-sub`.
pub const JUMP_WALL_REGULAR_X_RAW_MULT: f64 = 1.5; // DAT_105729e8
pub const JUMP_WALL_REGULAR_Y_RAW: f64 = -0.7; // 0.0 - DAT_105866b8
pub const JUMP_WALL_SLIDE_X_RAW_MULT: f64 = 1.0; // DAT_1056c350
pub const JUMP_WALL_SLIDE_Y_RAW: f64 = -0.5; // 0.0 - DAT_1056c348

/// Effective magnitudes only. Never compute an impulse from these.
pub const JUMP_IMPULSE_X_SCALE: f64 = 2.0 / 3.0;
pub const JUMP_IMPULSE_Y_SCALE: f64 = 2.0;
/// Raw adverse-slope row argument (`DAT_10586710`), pre-scaling.
pub const JUMP_UPSLOPE_Y_RAW: f64 = -0.7;
/// `DAT_10586710 = -0.7`, the adverse-slope row, after scaling.
pub const JUMP_UPSLOPE_Y: f64 = -0.7 * JUMP_IMPULSE_Y_SCALE;

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

// --- animation constants that became physics inputs when the limb test landed ---
/// DAT_105866e0, added to the atan2 result in ull/ninja_10516280.c:35.
/// A STORED 64-bit constant, not FRAC_PI_2 -- they differ by ~4.9e-12.
/// Round 7 §3.3 asked for it verbatim; round 9 §2.3 confirmed it.
pub const TILT_QUARTER_TURN: f64 = 1.57079632679;
/// DAT_10571d98. The airborne tilt decay is  - a*0.1 (:71-72), NOT
///  *= 0.9; there is no 0.9 constant in that function. Round 9 §2.1.
pub const TILT_DECAY: f64 = 0.1;
