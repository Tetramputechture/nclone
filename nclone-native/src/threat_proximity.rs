//! The shooter/hazard TELEGRAPH block: `threat_proximity`.
//!
//! A fixed-width scalar vector, per threat SUPER-CLASS, describing the nearest
//! in-range member of that class: where it is, how fast it is closing, whether
//! it is winding up, and -- for the classes that move -- how it is moving.
//!
//! # Why this lives in Rust
//!
//! It did not, and that was a defect rather than a design.
//! `NppEnvironment._compute_threat_proximity` reads `sim.entity_dic`, and on the
//! NATIVE backend `_build_entity_dic` whitelists types
//! `{1, 2, 3, 4, 6, 7, 11, 21}` only -- mines, gold, doors, switches, one-ways.
//! Not one threat type is in that set. Measured on `official/R/027 rear guard`,
//! which carries a rocket (18) and a floor guard (16):
//!
//! | backend | threat types in `entity_dic` |
//! |---|---|
//! | python | `{18: 1}` -- the rocket, at 517 px |
//! | native | **none** |
//!
//! Training runs `--sim-backend native`. So the whole 20-float block was
//! structurally zero in production while reading correctly in every unit test,
//! because the tests exercise the pure `_threat_imminence` mapping and the
//! dev-tool overlay runs on the Python backend.
//!
//! Widening the `entity_dic` whitelist was rejected: that dictionary is consumed
//! by the graph build, the overlays and the replay extractor, and changing what
//! they see as a side effect of adding an observation is how a one-line fix
//! becomes a run. Computing the block in Rust instead follows the rule
//! `threat_field.rs` already sets -- no per-entity data crosses FFI, only the
//! finished fixed-size result -- and it is the only place the rocket's
//! projectile state is available at all.
//!
//! # Layout
//!
//! `THREAT_PROX_CLASSES * THREAT_PROX_FEATURES` floats, class-major:
//!
//! | class | members |
//! |---|---|
//! | 0 contact    | zap (14), mini (26) -- lethal to touch, deterministic patrol |
//! | 1 homing     | chaser (15), death ball (25), eye bat (27), evil ninja (22) |
//! | 2 beam       | turret (19), dual laser (23), laser (13) / chaingun (12) drones |
//! | 3 projectile | rocket (18) |
//!
//! Per class, all eight dims describe the NEAREST in-range member. Selecting by
//! highest imminence instead was considered and rejected: it makes the geometry
//! and the countdown describe different entities on different frames, which is a
//! signal that flickers between objects.

use crate::constants::{
    NINJA_RADIUS, ROCKET_LAUNCH_DELAY, ROCKET_MAX_SPEED, ROCKET_SPEED_MULTIPLIER,
    TURRET_CHARGE_THRESHOLD,
};
use crate::entities::drone::{DroneBehaviour, DroneKind};
use crate::entities::{evil_ninja, rocket, Entity};
use crate::simulator::Simulator;

pub const THREAT_PROX_CLASSES: usize = 4;
/// `[dx_hat, dy_hat, proximity, closing, imminence, vx, vy, spin_up]`.
///
/// Dims 5-7 are the motion block, and they are the reason this is 8 wide rather
/// than the 5 it shipped as. The first five describe a still frame plus a
/// countdown; a rocket projectile 200 px away accelerating straight at the ninja
/// and one drifting past on a tangent produced IDENTICAL vectors, because
/// `closing` was computed from the ninja's own velocity alone and the threat's
/// motion appeared nowhere.
pub const THREAT_PROX_FEATURES: usize = 8;
pub const THREAT_PROXIMITY_DIM: usize = THREAT_PROX_CLASSES * THREAT_PROX_FEATURES;

/// Beyond this a threat is not an immediate concern and the class reads all
/// zero, which is also the "no such threat" reading -- the correct semantics for
/// a threat signal.
pub const THREAT_PROXIMITY_RANGE_PX: f64 = 240.0;
/// Normaliser for every velocity-like dim. A terminal-velocity dive exceeds the
/// horizontal cap, so this is set above both rather than at either.
pub const THREAT_PROXIMITY_SPEED_NORM_PX: f64 = 4.0;

pub const CLASS_CONTACT: usize = 0;
pub const CLASS_HOMING: usize = 1;
pub const CLASS_BEAM: usize = 2;
pub const CLASS_PROJECTILE: usize = 3;

/// One candidate threat, reduced to what the block reports.
struct Candidate {
    class: usize,
    x: f64,
    y: f64,
    /// World-frame velocity, px/frame. Zero for fixtures (turrets, dual lasers)
    /// and for entities whose motion the sim does not expose.
    vx: f64,
    vy: f64,
    /// Normalised telegraph countdown, read from the entity's own state machine.
    imminence: f64,
    /// How much of its speed budget the threat has still to gain, as a fraction
    /// of its own maximum. Non-zero only for the rocket projectile, the one
    /// hazard in the game that accelerates.
    spin_up: f64,
}

/// Speed the rocket projectile will gain NEXT frame, as a fraction of its cap.
///
/// This is the acceleration term, and it is a real signal rather than a derived
/// nicety: a projectile at `speed = 0.05` and one at `speed = 2.28` are 45x
/// apart in how far they travel in the next second, and the still-frame bearing
/// cannot tell them apart. It decays to exactly 0.0 at terminal speed (~24
/// frames after launch), so "still spinning up" and "committed at full speed"
/// are distinguishable.
fn rocket_spin_up(speed: f64, step: f64) -> f64 {
    if speed >= ROCKET_MAX_SPEED {
        return 0.0;
    }
    let next_step = step * ROCKET_SPEED_MULTIPLIER;
    let gain = (speed + next_step).min(ROCKET_MAX_SPEED) - speed;
    (gain / ROCKET_MAX_SPEED).clamp(0.0, 1.0)
}

fn collect_candidates(sim: &Simulator) -> Vec<Candidate> {
    let mut out = Vec::new();
    for entity in &sim.entities {
        match entity {
            Entity::Drone(d) if d.active => {
                let (vx, vy) = d.velocity();
                let (class, imminence) = match d.kind {
                    DroneKind::Zap | DroneKind::Mini => (CLASS_CONTACT, 0.0),
                    DroneKind::Chaser => (CLASS_HOMING, 0.0),
                    // Shooter drones: the BODY is harmless (`body_is_lethal` is
                    // false), so what this reports is the aim point of a beam
                    // that is about to exist. AIMING counts to CHARGE_DURATION,
                    // FIRING is committed, COOLDOWN is a safe window and reads
                    // 0.0 -- which is a positive signal, not an absence.
                    DroneKind::Chaingun | DroneKind::Laser => {
                        let imm = match &d.behaviour {
                            DroneBehaviour::Shooter(s) => {
                                let dur = d.timings.charge.max(1) as f64;
                                match s.state {
                                    1 => (s.timer as f64 / dur).clamp(0.0, 1.0),
                                    2 => 1.0,
                                    _ => 0.0,
                                }
                            }
                            _ => 0.0,
                        };
                        (CLASS_BEAM, imm)
                    }
                };
                out.push(Candidate {
                    class,
                    x: d.xpos,
                    y: d.ypos,
                    vx,
                    vy,
                    imminence,
                    spin_up: 0.0,
                });
            }
            Entity::DeathBall(b) => out.push(Candidate {
                class: CLASS_HOMING,
                x: b.xpos,
                y: b.ypos,
                vx: b.xspeed,
                vy: b.yspeed,
                imminence: 0.0,
                spin_up: 0.0,
            }),
            Entity::EyeBat(b) => out.push(Candidate {
                class: CLASS_HOMING,
                x: b.xpos,
                y: b.ypos,
                vx: b.xspeed,
                vy: b.yspeed,
                imminence: 0.0,
                spin_up: 0.0,
            }),
            // Lethal ONLY in PLAYBACK. A dormant one cannot kill and carries the
            // larger dormant radius, so reporting it would describe a threat
            // that is not there. Its velocity is not exposed by the entity, so
            // the motion dims stay zero rather than being invented.
            Entity::EvilNinja(n) if n.state == evil_ninja::PLAYBACK => out.push(Candidate {
                class: CLASS_HOMING,
                x: n.xpos,
                y: n.ypos,
                vx: 0.0,
                vy: 0.0,
                imminence: 0.0,
                spin_up: 0.0,
            }),
            // Turret: a fixture. TRACKING charges toward the commit; FIRING is
            // live; RECOVERING reads 0.0 and is the single most actionable frame
            // the block has -- it has just fired and cannot fire again for 15
            // frames, so that is exactly when crossing its line is free.
            Entity::Turret(t) => {
                let imm = match t.state {
                    1 => (t.charge / TURRET_CHARGE_THRESHOLD).clamp(0.0, 1.0),
                    2 => 1.0,
                    _ => 0.0,
                };
                out.push(Candidate {
                    class: CLASS_BEAM,
                    x: t.xpos,
                    y: t.ypos,
                    vx: 0.0,
                    vy: 0.0,
                    imminence: imm,
                    spin_up: 0.0,
                });
            }
            // Dual laser: no wind-up at all. It is either sweeping a lethal line
            // or it is not, and the FIELD already carries which.
            Entity::DualLaser(l) => out.push(Candidate {
                class: CLASS_BEAM,
                x: l.xpos,
                y: l.ypos,
                vx: 0.0,
                vy: 0.0,
                imminence: 0.0,
                spin_up: 0.0,
            }),
            Entity::Rocket(r) => match r.state {
                // FLYING: report the PROJECTILE, not the launcher. This is the
                // whole point of the class -- the launcher is a harmless tube
                // bolted to a wall, and once it has fired, the lethal object can
                // be hundreds of px away closing at 2.29 px/frame while the
                // launcher's own bearing says nothing at all.
                rocket::STATE_FLYING => {
                    let p = &r.projectile;
                    out.push(Candidate {
                        class: CLASS_PROJECTILE,
                        x: p.xpos,
                        y: p.ypos,
                        vx: p.xdir * p.speed,
                        vy: p.ydir * p.speed,
                        imminence: 1.0,
                        spin_up: rocket_spin_up(p.speed, p.step),
                    });
                }
                // CHARGING: the launcher IS the right object to report, because
                // the projectile does not exist yet and will appear exactly
                // here. `imminence` is the countdown to that.
                rocket::STATE_CHARGING => out.push(Candidate {
                    class: CLASS_PROJECTILE,
                    x: r.xpos,
                    y: r.ypos,
                    vx: 0.0,
                    vy: 0.0,
                    imminence: (r.charge / ROCKET_LAUNCH_DELAY).clamp(0.0, 1.0),
                    // A launch always starts from a standstill, so the whole
                    // ramp is still ahead of it.
                    spin_up: rocket_spin_up(0.0, crate::constants::ROCKET_INITIAL_STEP),
                }),
                // IDLE acquires by sight in one frame and has committed to
                // nothing. Reporting it would make a harmless fixture read as a
                // permanent threat.
                _ => {}
            },
            _ => {}
        }
    }
    out
}

/// Fill `out` (length [`THREAT_PROXIMITY_DIM`]) from the simulator's live state.
///
/// `closing` is the TRUE relative closing rate, `(v_ninja - v_threat) . u`,
/// positive when the gap is shrinking. The Python original projected the
/// ninja's velocity alone, which is correct for a mine and wrong for everything
/// in this table: a stationary ninja with a rocket inbound at full speed read
/// `closing = 0.0`, i.e. "nothing is happening".
pub fn compute(sim: &Simulator, out: &mut [f32]) {
    debug_assert_eq!(out.len(), THREAT_PROXIMITY_DIM);
    out.fill(0.0);
    let ninja = &sim.ninja;
    let (nx, ny) = (ninja.xpos, ninja.ypos);
    let (nvx, nvy) = (ninja.xspeed, ninja.yspeed);

    // Nearest in-range member per class.
    let mut best: [Option<(f64, usize)>; THREAT_PROX_CLASSES] = [None; THREAT_PROX_CLASSES];
    let candidates = collect_candidates(sim);
    for (i, c) in candidates.iter().enumerate() {
        let d = ((c.x - nx).powi(2) + (c.y - ny).powi(2)).sqrt();
        if d > THREAT_PROXIMITY_RANGE_PX {
            continue;
        }
        match best[c.class] {
            Some((bd, _)) if bd <= d => {}
            _ => best[c.class] = Some((d, i)),
        }
    }

    for (class, slot) in best.iter().enumerate() {
        let Some((d, idx)) = *slot else { continue };
        let c = &candidates[idx];
        let base = class * THREAT_PROX_FEATURES;
        // At d == 0 the bearing is undefined and the ninja is already inside the
        // kill disc: report maximum proximity with a zero bearing.
        let (ux, uy) = if d > 1e-6 {
            ((c.x - nx) / d, (c.y - ny) / d)
        } else {
            (0.0, 0.0)
        };
        let closing = ((nvx - c.vx) * ux + (nvy - c.vy) * uy) / THREAT_PROXIMITY_SPEED_NORM_PX;
        out[base] = ux as f32;
        out[base + 1] = uy as f32;
        out[base + 2] = (1.0 - d / THREAT_PROXIMITY_RANGE_PX).clamp(0.0, 1.0) as f32;
        out[base + 3] = closing.clamp(-1.0, 1.0) as f32;
        out[base + 4] = c.imminence.clamp(0.0, 1.0) as f32;
        out[base + 5] = (c.vx / THREAT_PROXIMITY_SPEED_NORM_PX).clamp(-1.0, 1.0) as f32;
        out[base + 6] = (c.vy / THREAT_PROXIMITY_SPEED_NORM_PX).clamp(-1.0, 1.0) as f32;
        out[base + 7] = c.spin_up.clamp(0.0, 1.0) as f32;
    }
}

/// Reusable buffer so the per-tick path allocates nothing.
pub struct ThreatProximity {
    buf: Vec<f32>,
}

impl Default for ThreatProximity {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreatProximity {
    pub fn new() -> Self {
        ThreatProximity {
            buf: vec![0.0; THREAT_PROXIMITY_DIM],
        }
    }

    pub fn compute(&mut self, sim: &Simulator) -> &[f32] {
        compute(sim, &mut self.buf);
        &self.buf
    }
}

/// Ninja kill radius, re-exported so the Python twin cannot drift from it.
pub const _NINJA_RADIUS: f64 = NINJA_RADIUS;
