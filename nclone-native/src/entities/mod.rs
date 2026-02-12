pub mod toggle_mine;
pub mod exit_switch;
pub mod exit_door;

use crate::collision::spatial::cell_from_pos;
use crate::ninja::Ninja;
use crate::types::EntitySnapshot;

/// Represents any supported entity in the simulation.
#[derive(Debug, Clone)]
pub enum Entity {
    ToggleMine(toggle_mine::ToggleMine),
    ExitSwitch(exit_switch::ExitSwitch),
    ExitDoor(exit_door::ExitDoor),
}

impl Entity {
    pub fn xpos(&self) -> f64 {
        match self {
            Entity::ToggleMine(e) => e.xpos,
            Entity::ExitSwitch(e) => e.xpos,
            Entity::ExitDoor(e) => e.xpos,
        }
    }

    pub fn ypos(&self) -> f64 {
        match self {
            Entity::ToggleMine(e) => e.ypos,
            Entity::ExitSwitch(e) => e.ypos,
            Entity::ExitDoor(e) => e.ypos,
        }
    }

    pub fn active(&self) -> bool {
        match self {
            Entity::ToggleMine(e) => e.active,
            Entity::ExitSwitch(e) => e.active,
            Entity::ExitDoor(_e) => true, // Exit doors are always "active" in the entity sense
        }
    }

    pub fn entity_type_id(&self) -> u8 {
        match self {
            Entity::ToggleMine(e) => e.entity_type,
            Entity::ExitSwitch(_) => 4,
            Entity::ExitDoor(_) => 3,
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
            Entity::ExitSwitch(e) => e.active,
            Entity::ExitDoor(e) => e.switch_hit,
        }
    }

    /// Perform logical collision with the ninja.
    /// Returns true if an interaction occurred.
    pub fn logical_collision(&mut self, ninja: &mut Ninja) -> bool {
        match self {
            Entity::ToggleMine(e) => e.logical_collision(ninja),
            Entity::ExitSwitch(e) => e.logical_collision(ninja),
            Entity::ExitDoor(e) => e.logical_collision(ninja),
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
            Entity::ExitSwitch(e) => e.reset_state(),
            Entity::ExitDoor(e) => e.reset_state(),
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
            },
            Entity::ExitSwitch(e) => EntitySnapshot {
                entity_type: 4,
                x: e.xpos,
                y: e.ypos,
                active: e.active,
                state: 0,
                radius: crate::constants::EXIT_SWITCH_RADIUS,
            },
            Entity::ExitDoor(e) => EntitySnapshot {
                entity_type: 3,
                x: e.xpos,
                y: e.ypos,
                active: e.switch_hit,
                state: 0,
                radius: crate::constants::EXIT_DOOR_RADIUS,
            },
        }
    }
}
