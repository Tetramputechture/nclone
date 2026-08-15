pub mod bounce_block;
pub mod death_ball;
pub mod door_base;
pub mod drone;
pub mod shove_thwump;
pub mod switch_door;
pub mod thwump;
pub mod toggle_mine;
pub mod gold;
pub mod exit_switch;
pub mod exit_door;
pub mod launch_pad;
pub mod locked_door;
pub mod one_way_platform;
pub mod boost_pad;
// Reverse-engineered from npp.dll; see docs/ENTITY_INTEGRATION_SPEC.md.
pub mod dual_laser;
pub mod evil_ninja;
pub mod eye_bat;
pub mod floor_guard;
pub mod pc_rocket;
pub mod rocket;
pub mod turret;

use crate::collision::spatial::cell_from_pos;
use crate::constants::{MAP_HEIGHT, MAP_WIDTH};
use crate::grid_edges::GridEdges;
use crate::ninja::Ninja;
use crate::segments::Segment;
use crate::types::{Cell, EntitySnapshot};

/// Occupancy grid mapping a cell to the indices of the entities inside it.
pub type EntityGrid = [[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH];
/// Cell -> indices into the shared segment pool.
pub type SegmentGrid = [[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH];

/// Everything an entity may touch during its `move` or `think` phase.
///
/// The fields are disjoint members of `Simulator`, which is what lets the
/// simulator hand out `&mut entities[i]` and this context simultaneously.
///
/// Requirements are drawn from the Python entities:
///   * boost pad (24) writes ninja velocity during `move`;
///   * drones (14/26), thwumps (20) and shove thwumps (28) read `grid_edges` to
///     test the cell they are about to enter;
///   * the death ball (25) sweeps and depenetrates against tile geometry during
///     `think`, so it needs the segments;
///   * anything that changes position updates its cell in `entity_grid`.
///
/// The one interaction this cannot express is death-ball-vs-death-ball
/// repulsion, which mutates a *sibling* entity. That runs as a separate
/// simulator-side pass when the death ball lands.
pub struct EntityCtx<'a> {
    /// Index of the entity being stepped, needed to patch `entity_grid`.
    pub index: usize,
    pub entity_grid: &'a mut EntityGrid,
    pub grid_edges: &'a GridEdges,
    pub segments: &'a [Segment],
    pub segment_grid: &'a SegmentGrid,
    pub ninja: &'a mut Ninja,
}

impl EntityCtx<'_> {
    /// Move this entity between occupancy cells. Port of `Entity.grid_move`
    /// (nclone/entities.py:257).
    ///
    /// Call sites matter for parity: Python entities invoke `grid_move()` at
    /// specific points inside `move()`, and NOT always. The zap drone, for
    /// instance, skips it on the frame it reaches a cell centre and picks a new
    /// direction, so its recorded cell is briefly stale. Do not "fix" that by
    /// syncing the grid unconditionally after the move pass -- it changes which
    /// entities `gather_entities_from_neighbourhood` returns.
    pub fn grid_move(&mut self, old_cell: Cell, new_cell: Cell) {
        if old_cell == new_cell {
            return;
        }
        let slot = &mut self.entity_grid[old_cell.0][old_cell.1];
        if let Some(pos) = slot.iter().position(|&i| i == self.index) {
            slot.remove(pos);
        }
        self.entity_grid[new_cell.0][new_cell.1].push(self.index);
    }
}

/// Represents any supported entity in the simulation.
#[derive(Debug, Clone)]
pub enum Entity {
    ToggleMine(toggle_mine::ToggleMine),
    Gold(gold::Gold),
    ExitSwitch(exit_switch::ExitSwitch),
    ExitDoor(exit_door::ExitDoor),
    LockedDoor(locked_door::LockedDoor),
    OneWay(one_way_platform::OneWayPlatform),
    LaunchPad(launch_pad::LaunchPad),
    BoostPad(boost_pad::BoostPad),
    Drone(drone::Drone),
    BounceBlock(bounce_block::BounceBlock),
    Thwump(thwump::Thwump),
    ShoveThwump(shove_thwump::ShoveThwump),
    SwitchDoor(switch_door::SwitchDoor),
    DeathBall(death_ball::DeathBall),
    FloorGuard(floor_guard::FloorGuard),
    EyeBat(eye_bat::EyeBat),
    Rocket(rocket::Rocket),
    Turret(turret::Turret),
    DualLaser(dual_laser::DualLaser),
    EvilNinja(evil_ninja::EvilNinja),
    /// Type 29. Not map-placed -- `map/loader.rs` has no arm for it. Kept so the
    /// official type set is complete; see `pc_rocket.rs`.
    PcRocket(pc_rocket::PcRocket),
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
            Entity::LaunchPad(e) => e.xpos,
            Entity::BoostPad(e) => e.xpos,
            Entity::Drone(e) => e.xpos,
            Entity::BounceBlock(e) => e.xpos,
            Entity::Thwump(e) => e.xpos,
            Entity::ShoveThwump(e) => e.xpos,
            Entity::SwitchDoor(e) => e.xpos,
            Entity::DeathBall(e) => e.xpos,
            Entity::FloorGuard(e) => e.xpos,
            Entity::EyeBat(e) => e.xpos,
            Entity::Rocket(e) => e.xpos,
            Entity::Turret(e) => e.xpos,
            Entity::DualLaser(e) => e.xpos,
            Entity::EvilNinja(e) => e.xpos,
            Entity::PcRocket(e) => e.xpos,
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
            Entity::LaunchPad(e) => e.ypos,
            Entity::BoostPad(e) => e.ypos,
            Entity::Drone(e) => e.ypos,
            Entity::BounceBlock(e) => e.ypos,
            Entity::Thwump(e) => e.ypos,
            Entity::ShoveThwump(e) => e.ypos,
            Entity::SwitchDoor(e) => e.ypos,
            Entity::DeathBall(e) => e.ypos,
            Entity::FloorGuard(e) => e.ypos,
            Entity::EyeBat(e) => e.ypos,
            Entity::Rocket(e) => e.ypos,
            Entity::Turret(e) => e.ypos,
            Entity::DualLaser(e) => e.ypos,
            Entity::EvilNinja(e) => e.ypos,
            Entity::PcRocket(e) => e.ypos,
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
            // Pads never deactivate; the boost pad's one-shot latch is separate
            // from `active` (Python keeps `active` True and gates on
            // `is_touching_ninja`).
            Entity::LaunchPad(_) | Entity::BoostPad(_) => true,
            Entity::Drone(e) => e.active,
            // Square physics entities are permanent fixtures.
            Entity::BounceBlock(_) | Entity::Thwump(_) | Entity::ShoveThwump(_) => true,
            // Regular doors stay active forever; a trap door's switch
            // deactivates once collected.
            Entity::SwitchDoor(e) => e.active,
            Entity::DeathBall(_) => true,
            // Hazards with no deactivation: they are fixtures for the episode.
            Entity::FloorGuard(_)
            | Entity::EyeBat(_)
            | Entity::Rocket(_)
            | Entity::Turret(_)
            | Entity::DualLaser(_)
            | Entity::EvilNinja(_) => true,
            Entity::PcRocket(e) => e.active,
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
            Entity::LaunchPad(_) => 10,
            Entity::BoostPad(_) => 24,
            Entity::Drone(e) => e.kind.type_id(),
            Entity::BounceBlock(_) => 17,
            Entity::Thwump(_) => 20,
            Entity::ShoveThwump(_) => 28,
            Entity::SwitchDoor(e) => e.type_id(),
            Entity::DeathBall(_) => 25,
            Entity::FloorGuard(_) => 16,
            Entity::EyeBat(_) => 27,
            Entity::Rocket(_) => 18,
            Entity::Turret(_) => 19,
            Entity::DualLaser(_) => 23,
            Entity::EvilNinja(_) => 22,
            Entity::PcRocket(_) => 29,
        }
    }

    pub fn cell(&self) -> (usize, usize) {
        cell_from_pos(self.xpos(), self.ypos())
    }

    pub fn is_thinkable(&self) -> bool {
        matches!(
            self,
            Entity::ToggleMine(_)
                | Entity::Thwump(_)
                | Entity::ShoveThwump(_)
                | Entity::SwitchDoor(_)
                | Entity::DeathBall(_)
                // Spec §4: 22, 23, 27 and 29 think but never move -- they
                // integrate their own position inside `think`.
                | Entity::FloorGuard(_)
                | Entity::EyeBat(_)
                | Entity::Rocket(_)
                | Entity::Turret(_)
                | Entity::DualLaser(_)
                | Entity::EvilNinja(_)
                | Entity::PcRocket(_)
        ) || matches!(self, Entity::Drone(d) if d.is_thinkable())
    }

    /// Whether this entity participates in the per-frame `move` pass, which runs
    /// BEFORE the `think` pass and before ninja physics (nclone/nsim.py:398).
    ///
    /// Mirrors each Python class's `is_movable` flag. Must stay in sync with
    /// `move_entity` below -- an entity listed here with no `move_entity` arm is
    /// simply skipped, and one that moves without being listed never runs.
    pub fn is_movable(&self) -> bool {
        matches!(
            self,
            Entity::BoostPad(_)
                | Entity::Drone(_)
                | Entity::BounceBlock(_)
                | Entity::Thwump(_)
                // The only new type in the move pass. Types 18, 19 and 25 appear
                // in the game's move pass too, but their `move` is render-only
                // (spec §6.1) -- porting it would add frame-rate dependence.
                | Entity::FloorGuard(_)
        )
    }

    /// Per-frame movement. Mirrors `entity.move()` in Python. Only types listed
    /// in `is_movable` reach this.
    pub fn move_entity(&mut self, ctx: &mut EntityCtx) {
        match self {
            Entity::BoostPad(e) => e.move_entity(ctx),
            Entity::Drone(e) => e.move_entity(ctx),
            Entity::BounceBlock(e) => e.move_entity(ctx),
            Entity::Thwump(e) => e.move_entity(ctx),
            Entity::FloorGuard(e) => e.move_entity(ctx),
            _ => {}
        }
    }

    pub fn is_logical_collidable(&self) -> bool {
        match self {
            Entity::ToggleMine(_) => true,
            Entity::Gold(e) => e.active,
            Entity::ExitSwitch(e) => e.active,
            Entity::ExitDoor(e) => e.switch_hit,
            Entity::LockedDoor(e) => e.active, // switch collidable while uncollected
            Entity::OneWay(_) => true,
            // Launch pads are logical-only; boost pads act during `move` and are
            // NOT logical collidables in Python.
            Entity::LaunchPad(_) => true,
            Entity::BoostPad(_) => false,
            Entity::Drone(e) => e.active,
            Entity::BounceBlock(_) | Entity::Thwump(_) | Entity::ShoveThwump(_) => true,
            // The trigger is collidable while the switch is live.
            Entity::SwitchDoor(e) => e.active,
            Entity::DeathBall(_) => true,
            Entity::FloorGuard(_)
            | Entity::EyeBat(_)
            | Entity::Rocket(_)
            | Entity::EvilNinja(_)
            | Entity::PcRocket(_) => true,
            // Spec §6.6: types 12, 13, 19 and 23 have NO logical collision of
            // their own -- their bodies are harmless and all lethality comes from
            // beams resolved elsewhere. Do not add a kill-on-touch here.
            Entity::Turret(_) | Entity::DualLaser(_) => false,
        }
    }

    /// Whether the entity depenetrates the ninja (physical collision). Only the
    /// one-way platform does so in the current scope.
    pub fn is_physical_collidable(&self) -> bool {
        matches!(
            self,
            Entity::OneWay(_)
                | Entity::BounceBlock(_)
                | Entity::Thwump(_)
                | Entity::ShoveThwump(_)
        )
    }

    /// Physical (depenetration) collision against the ninja, used by
    /// `Simulator::collide_vs_objects`. Returns `((normal_x, normal_y),
    /// (depen_len, other_axis_pen))` or `None`.
    ///
    /// Takes `&mut self` because a bounce block does not merely report a
    /// depenetration -- it absorbs 80% of it into its own position and velocity
    /// and hands the remaining 20% back to the ninja
    /// (`entity_bounce_block.py:physical_collision`). One-way platforms are
    /// static and ignore the mutability.
    ///
    /// `ninja` stays immutable: every Python `physical_collision` only READS the
    /// ninja, and the caller applies the resulting push.
    pub fn physical_collision(&mut self, ninja: &Ninja) -> Option<((f64, f64), (f64, f64))> {
        match self {
            Entity::OneWay(e) => e.physical_collision(ninja),
            Entity::BounceBlock(e) => e.physical_collision(ninja),
            Entity::Thwump(e) => e.physical_collision(ninja),
            Entity::ShoveThwump(e) => e.physical_collision(ninja),
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
            // One-way wall-normal and launch-pad boost are handled inline in the
            // simulator: they contribute a wall normal / rewrite ninja state
            // rather than reporting a simple bool interaction.
            Entity::OneWay(_) | Entity::LaunchPad(_) | Entity::BoostPad(_) => false,
            // Drone kills, bounce-block / thwump / shwump wall normals and the
            // shwump's shove latch are all applied in the simulator's entity loop.
            Entity::Drone(_)
            | Entity::BounceBlock(_)
            | Entity::Thwump(_)
            | Entity::ShoveThwump(_)
            | Entity::SwitchDoor(_)
            | Entity::DeathBall(_) => false,
            Entity::FloorGuard(e) => e.logical_collision(ninja),
            Entity::EyeBat(e) => e.logical_collision(ninja),
            Entity::Rocket(e) => e.logical_collision(ninja),
            Entity::EvilNinja(e) => e.logical_collision(ninja),
            Entity::PcRocket(e) => e.logical_collision(ninja),
            // Bodies are harmless; the beam kills inside `think` (spec §6.6).
            Entity::Turret(_) | Entity::DualLaser(_) => false,
        }
    }

    /// Think (per-frame entity AI), running after the `move` pass and before
    /// ninja physics. Must stay in sync with `is_thinkable`.
    pub fn think(&mut self, ctx: &mut EntityCtx) {
        match self {
            Entity::ToggleMine(e) => e.think(ctx.ninja),
            Entity::Thwump(e) => {
                let (ninja, edges) = (&*ctx.ninja, ctx.grid_edges);
                e.think(ninja, edges);
            }
            Entity::ShoveThwump(e) => e.think(ctx),
            Entity::SwitchDoor(e) => e.think(),
            // Death balls are driven by Simulator::death_ball_think, which can
            // reach sibling balls for the repulsion pass.
            Entity::DeathBall(_) => {}
            Entity::Drone(e) => e.think(ctx),
            Entity::FloorGuard(e) => {
                let (ninja, edges) = (&*ctx.ninja, ctx.grid_edges);
                e.think(ninja, edges);
            }
            Entity::EyeBat(e) => e.think(ctx),
            Entity::Rocket(e) => e.think(ctx),
            Entity::Turret(e) => e.think(ctx),
            Entity::DualLaser(e) => e.think(ctx),
            Entity::PcRocket(e) => e.think(ctx),
            // The evil ninja needs the simulator's position tape, which EntityCtx
            // deliberately does not carry; Simulator::evil_ninja_think drives it.
            Entity::EvilNinja(_) => {}
            _ => {}
        }
    }

    /// Move the entity, used by the curriculum manager and vec-env to relocate
    /// entities on a loaded map.
    ///
    /// One implementation for both FFI entry points -- these were two copies of
    /// the same exhaustive match, and every new entity type had to be added to
    /// both. For a locked door this is the SWITCH position; the door barrier
    /// segment is static and is not moved. Callers are responsible for rebuilding
    /// the entity grid afterwards.
    pub fn set_position(&mut self, x: f64, y: f64) {
        match self {
            Entity::ToggleMine(e) => { e.xpos = x; e.ypos = y; }
            Entity::Gold(e) => { e.xpos = x; e.ypos = y; }
            Entity::ExitSwitch(e) => { e.xpos = x; e.ypos = y; }
            Entity::ExitDoor(e) => { e.xpos = x; e.ypos = y; }
            Entity::LockedDoor(e) => { e.xpos = x; e.ypos = y; }
            Entity::OneWay(e) => { e.xpos = x; e.ypos = y; }
            Entity::LaunchPad(e) => { e.xpos = x; e.ypos = y; }
            Entity::BoostPad(e) => { e.xpos = x; e.ypos = y; }
            Entity::Drone(e) => { e.xpos = x; e.ypos = y; }
            Entity::BounceBlock(e) => { e.xpos = x; e.ypos = y; }
            Entity::Thwump(e) => { e.xpos = x; e.ypos = y; }
            Entity::ShoveThwump(e) => { e.xpos = x; e.ypos = y; }
            Entity::SwitchDoor(e) => { e.xpos = x; e.ypos = y; }
            Entity::DeathBall(e) => { e.xpos = x; e.ypos = y; }
            Entity::FloorGuard(e) => { e.xpos = x; e.ypos = y; }
            Entity::EyeBat(e) => { e.xpos = x; e.ypos = y; }
            Entity::Rocket(e) => { e.xpos = x; e.ypos = y; }
            Entity::Turret(e) => { e.xpos = x; e.ypos = y; }
            Entity::DualLaser(e) => { e.xpos = x; e.ypos = y; }
            Entity::EvilNinja(e) => { e.xpos = x; e.ypos = y; }
            Entity::PcRocket(e) => { e.xpos = x; e.ypos = y; }
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
            Entity::LaunchPad(e) => e.reset_state(),
            Entity::BoostPad(e) => e.reset_state(),
            Entity::Drone(e) => e.reset_state(),
            Entity::BounceBlock(e) => e.reset_state(),
            Entity::Thwump(e) => e.reset_state(),
            Entity::ShoveThwump(e) => e.reset_state(),
            Entity::SwitchDoor(e) => e.reset_state(),
            Entity::DeathBall(e) => e.reset_state(),
            Entity::FloorGuard(e) => e.reset_state(),
            Entity::EyeBat(e) => e.reset_state(),
            Entity::Rocket(e) => e.reset_state(),
            Entity::Turret(e) => e.reset_state(),
            Entity::DualLaser(e) => e.reset_state(),
            Entity::EvilNinja(e) => e.reset_state(),
            Entity::PcRocket(e) => e.reset_state(),
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
            Entity::LaunchPad(e) => EntitySnapshot {
                entity_type: 10,
                x: e.xpos,
                y: e.ypos,
                active: true,
                state: 0,
                radius: crate::constants::LAUNCH_PAD_RADIUS,
                // The boost direction -- the only thing that distinguishes one
                // launch pad from another.
                orientation: e.orientation,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::BoostPad(e) => EntitySnapshot {
                entity_type: 24,
                x: e.xpos,
                y: e.ypos,
                active: true,
                // The one-shot latch. Python has no `state` attribute here, so
                // cross-sim comparisons must read `is_touching_ninja` instead.
                state: e.is_touching_ninja as u8,
                radius: crate::constants::BOOST_PAD_RADIUS,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::Drone(e) => EntitySnapshot {
                entity_type: e.kind.type_id(),
                x: e.xpos,
                y: e.ypos,
                active: e.active,
                // Facing direction (0 right, 1 down, 2 left, 3 up) -- the drone's
                // only per-frame observable beyond position.
                state: e.dir,
                radius: e.kind.radius(),
                orientation: e.orientation,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::BounceBlock(e) => EntitySnapshot {
                entity_type: 17,
                x: e.xpos,
                y: e.ypos,
                active: true,
                state: 0,
                radius: crate::constants::BOUNCE_BLOCK_SEMI_SIDE,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::Thwump(e) => EntitySnapshot {
                entity_type: 20,
                x: e.xpos,
                y: e.ypos,
                active: true,
                // Signed state (-1/0/1) folded into 0..2 the same way Python's
                // replay log does (`state % 3`), since the field is unsigned.
                state: e.state.rem_euclid(3) as u8,
                radius: crate::constants::THWUMP_SEMI_SIDE,
                orientation: e.orientation,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::ShoveThwump(e) => EntitySnapshot {
                entity_type: 28,
                x: e.xpos,
                y: e.ypos,
                active: true,
                state: e.state,
                radius: crate::constants::SHOVE_THWUMP_SEMI_SIDE,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::SwitchDoor(e) => EntitySnapshot {
                entity_type: e.type_id(),
                // (x, y) is the SWITCH; the barrier is exposed via door_*, matching
                // the locked-door convention the observation layer already reads.
                x: e.xpos,
                y: e.ypos,
                active: e.active,
                state: 0,
                radius: e.radius(),
                orientation: 0,
                door_x: e.door_xpos,
                door_y: e.door_ypos,
                door_orientation: e.orientation,
                door_closed: e.closed,
            },
            Entity::DeathBall(e) => EntitySnapshot {
                entity_type: 25,
                x: e.xpos,
                y: e.ypos,
                active: true,
                state: 0,
                // The ninja-facing radius; the 8px tile/ball radius is internal.
                radius: crate::constants::DEATH_BALL_RADIUS,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::FloorGuard(e) => EntitySnapshot {
                entity_type: 16,
                x: e.xpos,
                y: e.ypos,
                active: true,
                state: // -1/0/+1 folded into 0..2, as the thwump's signed state is.
                e.dir.rem_euclid(3) as u8,
                radius: crate::constants::FLOOR_GUARD_RADIUS,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::EyeBat(e) => EntitySnapshot {
                entity_type: 27,
                x: e.xpos,
                y: e.ypos,
                active: true,
                state: e.chasing as u8,
                radius: crate::constants::EYE_BAT_RADIUS,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::Rocket(e) => EntitySnapshot {
                entity_type: 18,
                x: e.projectile.xpos,
                y: e.projectile.ypos,
                active: e.state == rocket::STATE_FLYING,
                state: e.state,
                radius: crate::constants::ROCKET_RADIUS,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::Turret(e) => EntitySnapshot {
                entity_type: 19,
                x: e.xpos,
                y: e.ypos,
                active: true,
                state: e.state,
                radius: 0.0,
                orientation: e.orientation,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::DualLaser(e) => EntitySnapshot {
                entity_type: 23,
                x: e.xpos,
                y: e.ypos,
                active: true,
                state: (e.mode == dual_laser::LaserMode::Surface) as u8,
                radius: crate::constants::DUAL_LASER_RADIUS,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::EvilNinja(e) => EntitySnapshot {
                entity_type: 22,
                x: e.xpos,
                y: e.ypos,
                active: e.state == evil_ninja::PLAYBACK,
                state: e.state,
                radius: e.radius,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
            },
            Entity::PcRocket(e) => EntitySnapshot {
                entity_type: 29,
                x: e.projectile.xpos,
                y: e.projectile.ypos,
                active: e.active,
                state: e.state,
                radius: crate::constants::ROCKET_RADIUS,
                orientation: 0,
                door_x: 0.0,
                door_y: 0.0,
                door_orientation: 0,
                door_closed: false,
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
