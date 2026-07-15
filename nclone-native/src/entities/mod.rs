pub mod toggle_mine;
pub mod gold;
pub mod exit_switch;
pub mod exit_door;
pub mod locked_door;
pub mod one_way_platform;

use crate::collision::spatial::cell_from_pos;
use crate::ninja::Ninja;
use crate::types::EntitySnapshot;

/// Represents any supported entity in the simulation.
#[derive(Debug, Clone)]
pub enum Entity {
    ToggleMine(toggle_mine::ToggleMine),
    Gold(gold::Gold),
    ExitSwitch(exit_switch::ExitSwitch),
    ExitDoor(exit_door::ExitDoor),
    LockedDoor(locked_door::LockedDoor),
    OneWay(one_way_platform::OneWayPlatform),
}

impl Entity {
    pub fn xpos(&self) -> f64 {
        match self {
            Entity::ToggleMine(e) => e.xpos,
            Entity::Gold(e) => e.xpos,
            Entity::ExitSwitch(e) => e.xpos,
            Entity::ExitDoor(e) => e.xpos,
            Entity::LockedDoor(e) => e.xpos,
            Entity::OneWay(e) => e.xpos,
        }
    }

    pub fn ypos(&self) -> f64 {
        match self {
            Entity::ToggleMine(e) => e.ypos,
            Entity::Gold(e) => e.ypos,
            Entity::ExitSwitch(e) => e.ypos,
            Entity::ExitDoor(e) => e.ypos,
            Entity::LockedDoor(e) => e.ypos,
            Entity::OneWay(e) => e.ypos,
        }
    }

    pub fn active(&self) -> bool {
        match self {
            Entity::ToggleMine(e) => e.active,
            Entity::Gold(e) => e.active,
            Entity::ExitSwitch(e) => e.active,
            Entity::ExitDoor(_e) => true, // Exit doors are always "active" in the entity sense
            Entity::LockedDoor(e) => e.active, // active while switch uncollected
            Entity::OneWay(_) => true,         // static, always present
        }
    }

    pub fn entity_type_id(&self) -> u8 {
        match self {
            Entity::ToggleMine(e) => e.entity_type,
            Entity::Gold(_) => 2,
            Entity::ExitSwitch(_) => 4,
            Entity::ExitDoor(_) => 3,
            Entity::LockedDoor(_) => 6,
            Entity::OneWay(_) => 11,
        }
    }

    pub fn cell(&self) -> (usize, usize) {
        cell_from_pos(self.xpos(), self.ypos())
    }

    pub fn is_thinkable(&self) -> bool {
        matches!(self, Entity::ToggleMine(_))
    }

    pub fn is_logical_collidable(&self) -> bool {
        match self {
            Entity::ToggleMine(_) => true,
            Entity::Gold(e) => e.active,
            Entity::ExitSwitch(e) => e.active,
            Entity::ExitDoor(e) => e.switch_hit,
            Entity::LockedDoor(e) => e.active, // switch collidable while uncollected
            Entity::OneWay(_) => true,
        }
    }

    /// Whether the entity depenetrates the ninja (physical collision). Only the
    /// one-way platform does so in the current scope.
    pub fn is_physical_collidable(&self) -> bool {
        matches!(self, Entity::OneWay(_))
    }

    /// Physical (depenetration) collision against the ninja, used by
    /// `Ninja::collide_vs_objects`. Returns `((normal_x, normal_y), (depen_len,
    /// 0.0))` or `None`. Read-only on the entity (all physical entities here are
    /// static).
    pub fn physical_collision(&self, ninja: &Ninja) -> Option<((f64, f64), (f64, f64))> {
        match self {
            Entity::OneWay(e) => e.physical_collision(ninja),
            _ => None,
        }
    }

    /// Perform logical collision with the ninja.
    /// Returns true if an interaction occurred.
    pub fn logical_collision(&mut self, ninja: &mut Ninja) -> bool {
        match self {
            Entity::ToggleMine(e) => e.logical_collision(ninja),
            Entity::Gold(e) => e.logical_collision(ninja),
            Entity::ExitSwitch(e) => e.logical_collision(ninja),
            Entity::ExitDoor(e) => e.logical_collision(ninja),
            Entity::LockedDoor(e) => e.logical_collision(ninja),
            // One-way wall-normal logic is handled inline in the simulator
            // (it contributes a wall normal, not a simple bool interaction).
            Entity::OneWay(_) => false,
        }
    }

    /// Think (per-frame entity AI). Only toggle mines have think behavior.
    pub fn think(&mut self, ninja: &Ninja) {
        match self {
            Entity::ToggleMine(e) => e.think(ninja),
            _ => {}
        }
    }

    /// Reset entity to initial state.
    pub fn reset_state(&mut self) {
        match self {
            Entity::ToggleMine(e) => e.reset_state(),
            Entity::Gold(e) => e.reset_state(),
            Entity::ExitSwitch(e) => e.reset_state(),
            Entity::ExitDoor(e) => e.reset_state(),
            Entity::LockedDoor(e) => e.reset_state(),
            Entity::OneWay(e) => e.reset_state(),
        }
    }

    /// Take a snapshot for Python.
    pub fn snapshot(&self) -> EntitySnapshot {
        match self {
            Entity::ToggleMine(e) => EntitySnapshot {
                entity_type: e.entity_type,
                x: e.xpos,
                y: e.ypos,
                active: e.active,
                state: e.state,
                radius: e.radius,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::Gold(e) => EntitySnapshot {
                entity_type: 2,
                x: e.xpos,
                y: e.ypos,
                active: e.active,
                state: 0,
                radius: crate::constants::GOLD_RADIUS,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::ExitSwitch(e) => EntitySnapshot {
                entity_type: 4,
                x: e.xpos,
                y: e.ypos,
                active: e.active,
                state: 0,
                radius: crate::constants::EXIT_SWITCH_RADIUS,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::ExitDoor(e) => EntitySnapshot {
                entity_type: 3,
                x: e.xpos,
                y: e.ypos,
                active: e.switch_hit,
                state: 0,
                radius: crate::constants::EXIT_DOOR_RADIUS,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::LockedDoor(e) => EntitySnapshot {
                entity_type: 6,
                // (x, y) is the SWITCH; door barrier is exposed via door_* fields.
                x: e.xpos,
                y: e.ypos,
                active: e.active,
                state: 0,
                radius: crate::constants::LOCKED_DOOR_SWITCH_RADIUS,
                orientation: 0,
                door_x: e.door_xpos,
                door_y: e.door_ypos,
                door_orientation: e.orientation,
                door_closed: e.closed,
            },
            Entity::OneWay(e) => EntitySnapshot {
                entity_type: 11,
                x: e.xpos,
                y: e.ypos,
                active: true,
                state: 0,
                radius: crate::constants::ONE_WAY_PLATFORM_SEMI_SIDE,
                orientation: e.orientation,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
        }
    }
}
