//! Core types used throughout the nclone native physics engine.

use pyo3::prelude::*;

/// 2D vector for positions and velocities.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub const fn new(x: f64, y: f64) -> Self {
        Vec2 { x, y }
    }

    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    pub fn length(self) -> f64 {
        self.length_sq().sqrt()
    }

    pub fn dot(self, other: Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn cross(self, other: Vec2) -> f64 {
        self.x * other.y - self.y * other.x
    }
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;
    fn add(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Mul<f64> for Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: f64) -> Vec2 {
        Vec2::new(self.x * rhs, self.y * rhs)
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Vec2) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

/// Ninja movement states matching the Python simulator exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NinjaState {
    Immobile = 0,
    Running = 1,
    GroundSliding = 2,
    Jumping = 3,
    Falling = 4,
    WallSliding = 5,
    Dead = 6,
    AwaitingDeath = 7,
    Celebrating = 8,
    Disabled = 9,
}

impl NinjaState {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => NinjaState::Immobile,
            1 => NinjaState::Running,
            2 => NinjaState::GroundSliding,
            3 => NinjaState::Jumping,
            4 => NinjaState::Falling,
            5 => NinjaState::WallSliding,
            6 => NinjaState::Dead,
            7 => NinjaState::AwaitingDeath,
            8 => NinjaState::Celebrating,
            9 => NinjaState::Disabled,
            _ => NinjaState::Immobile,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn is_ground(self) -> bool {
        matches!(
            self,
            NinjaState::Immobile | NinjaState::Running | NinjaState::GroundSliding
        )
    }

    pub fn is_air(self) -> bool {
        matches!(self, NinjaState::Jumping | NinjaState::Falling)
    }

    pub fn is_inactive(self) -> bool {
        matches!(
            self,
            NinjaState::Dead
                | NinjaState::AwaitingDeath
                | NinjaState::Celebrating
                | NinjaState::Disabled
        )
    }
}

/// Entity types supported by the native simulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EntityType {
    ToggleMine = 1,
    Gold = 2,
    ExitDoor = 3,
    ExitSwitch = 4,
    RegularDoor = 5,
    LockedDoor = 6,
    // Type 7 is a phantom in the Python model (the switch lives on the type-6
    // entity); recognized only so map parsing treats it as a no-op rather than
    // silently misreading subsequent bytes.
    LockedDoorSwitch = 7,
    TrapDoor = 8,
    TrapDoorSwitch = 9,
    LaunchPad = 10,
    OneWay = 11,
    DroneChaingun = 12,
    DroneLaser = 13,
    DroneZap = 14,
    DroneChaser = 15,
    FloorGuard = 16,
    BounceBlock = 17,
    Rocket = 18,
    Turret = 19,
    Thwump = 20,
    ToggleMineToggled = 21,
    EvilNinja = 22,
    DualLaser = 23,
    BoostPad = 24,
    DeathBall = 25,
    MiniDrone = 26,
    EyeBat = 27,
    ShoveThwump = 28,
    /// Spawned at runtime, never map-placed -- confirmed absent from 8000
    /// decoded attract maps. Recognized so the type table is complete.
    PlayerControlledRocket = 29,
}

impl EntityType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(EntityType::ToggleMine),
            2 => Some(EntityType::Gold),
            3 => Some(EntityType::ExitDoor),
            4 => Some(EntityType::ExitSwitch),
            5 => Some(EntityType::RegularDoor),
            6 => Some(EntityType::LockedDoor),
            7 => Some(EntityType::LockedDoorSwitch),
            8 => Some(EntityType::TrapDoor),
            9 => Some(EntityType::TrapDoorSwitch),
            10 => Some(EntityType::LaunchPad),
            11 => Some(EntityType::OneWay),
            12 => Some(EntityType::DroneChaingun),
            13 => Some(EntityType::DroneLaser),
            14 => Some(EntityType::DroneZap),
            15 => Some(EntityType::DroneChaser),
            16 => Some(EntityType::FloorGuard),
            17 => Some(EntityType::BounceBlock),
            18 => Some(EntityType::Rocket),
            19 => Some(EntityType::Turret),
            20 => Some(EntityType::Thwump),
            21 => Some(EntityType::ToggleMineToggled),
            22 => Some(EntityType::EvilNinja),
            23 => Some(EntityType::DualLaser),
            24 => Some(EntityType::BoostPad),
            25 => Some(EntityType::DeathBall),
            26 => Some(EntityType::MiniDrone),
            27 => Some(EntityType::EyeBat),
            28 => Some(EntityType::ShoveThwump),
            29 => Some(EntityType::PlayerControlledRocket),
            _ => None,
        }
    }
}

/// Complete ninja state snapshot exposed to Python.
/// Contains all fields that the Python observation/reward code reads.
#[pyclass]
#[derive(Debug, Clone)]
pub struct NinjaSnapshot {
    // Position/Velocity
    #[pyo3(get)]
    pub xpos: f64,
    #[pyo3(get)]
    pub ypos: f64,
    #[pyo3(get)]
    pub xspeed: f64,
    #[pyo3(get)]
    pub yspeed: f64,
    #[pyo3(get)]
    pub xpos_old: f64,
    #[pyo3(get)]
    pub ypos_old: f64,
    #[pyo3(get)]
    pub xspeed_old: f64,
    #[pyo3(get)]
    pub yspeed_old: f64,

    // State machine
    #[pyo3(get)]
    pub state: u8,
    #[pyo3(get)]
    pub airborn: bool,
    #[pyo3(get)]
    pub airborn_old: bool,
    #[pyo3(get)]
    pub walled: bool,
    #[pyo3(get)]
    pub wall_normal: f64,
    #[pyo3(get)]
    pub terminal_impact: bool,

    // Jump/buffer state
    #[pyo3(get)]
    pub jump_duration: i32,
    #[pyo3(get)]
    pub jump_input_old: i32,
    #[pyo3(get)]
    pub jump_buffer: i32,
    #[pyo3(get)]
    pub floor_buffer: i32,
    #[pyo3(get)]
    pub wall_buffer: i32,
    #[pyo3(get)]
    pub launch_pad_buffer: i32,
    #[pyo3(get)]
    pub last_jump_was_buffered: bool,

    // Collision normals
    #[pyo3(get)]
    pub floor_normalized_x: f64,
    #[pyo3(get)]
    pub floor_normalized_y: f64,
    #[pyo3(get)]
    pub ceiling_normalized_x: f64,
    #[pyo3(get)]
    pub ceiling_normalized_y: f64,
    #[pyo3(get)]
    pub floor_count: u32,
    #[pyo3(get)]
    pub wall_count: u32,
    #[pyo3(get)]
    pub ceiling_count: u32,

    // Physics parameters
    #[pyo3(get)]
    pub applied_gravity: f64,
    #[pyo3(get)]
    pub applied_drag: f64,
    #[pyo3(get)]
    pub applied_friction: f64,

    // Inputs
    #[pyo3(get)]
    pub hor_input: i32,
    #[pyo3(get)]
    pub jump_input: i32,

    // Temporal tracking
    #[pyo3(get)]
    pub frames_airborne: u32,
    #[pyo3(get)]
    pub state_change_frame: u32,
    #[pyo3(get)]
    pub previous_state: u8,
    #[pyo3(get)]
    pub consecutive_floor_frames: u32,
    #[pyo3(get)]
    pub consecutive_wall_frames: u32,

    // Game events
    #[pyo3(get)]
    pub has_won: bool,
    #[pyo3(get)]
    pub has_died: bool,
    #[pyo3(get)]
    pub death_cause: Option<String>,

    // Collectibles
    #[pyo3(get)]
    pub gold_collected: u32,
    #[pyo3(get)]
    pub doors_opened: u32,
}

/// Per-entity state snapshot exposed to Python.
#[pyclass]
#[derive(Debug, Clone)]
pub struct EntitySnapshot {
    #[pyo3(get)]
    pub entity_type: u8,
    #[pyo3(get)]
    pub x: f64,
    #[pyo3(get)]
    pub y: f64,
    #[pyo3(get)]
    pub active: bool,
    #[pyo3(get)]
    pub state: u8,
    #[pyo3(get)]
    pub radius: f64,
    // Orientation 0..7 (8-way). Meaningful for directional entities such as the
    // one-way platform (type 11), where it encodes the blocking normal; 0 for
    // entities without an orientation.
    #[pyo3(get)]
    pub orientation: u8,
    // Locked-door fields (zero/false for all other entity types). For a locked
    // door, (x, y) above is the SWITCH position and `active` means the switch is
    // uncollected; these expose the door barrier so the planner can model it.
    #[pyo3(get)]
    pub door_x: f64,
    #[pyo3(get)]
    pub door_y: f64,
    #[pyo3(get)]
    pub door_orientation: u8,
    #[pyo3(get)]
    pub door_closed: bool,
}

/// Top-level simulator state returned to Python each tick.
/// Includes tile_grid to avoid separate FFI call for get_tile_grid().
#[pyclass]
#[derive(Debug, Clone)]
pub struct SimulatorState {
    #[pyo3(get)]
    pub ninja: NinjaSnapshot,
    #[pyo3(get)]
    pub entities: Vec<EntitySnapshot>,
    #[pyo3(get)]
    pub frame: u32,
    #[pyo3(get)]
    pub action_mask: [bool; 6],
    #[pyo3(get)]
    pub tile_grid: Vec<u8>,
}

/// Grid cell coordinate (clamped to 0..43, 0..24).
pub type Cell = (usize, usize);

/// Clamp cell coordinates to valid map range.
pub fn clamp_cell(x: i32, y: i32) -> Cell {
    (x.clamp(0, 43) as usize, y.clamp(0, 24) as usize)
}

/// Clamp half-cell coordinates to valid range.
pub fn clamp_half_cell(x: i32, y: i32) -> (usize, usize) {
    (x.clamp(0, 87) as usize, y.clamp(0, 49) as usize)
}
