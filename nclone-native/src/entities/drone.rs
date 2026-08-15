//! The whole drone family: Zap (14), Mini (26), Chaser (15), Chaingun (12) and
//! Laser (13). Port of nclone/entity_classes/entity_drone_base.py and its five
//! subclasses.
//!
//! A drone walks a grid of cell centres. Each frame it advances `speed` px
//! toward its current target; when it reaches (or overshoots) the target it
//! snaps exactly onto it and picks the next direction by testing the four
//! candidates in an order determined by its patrol mode.
//!
//! The five variants share that whole movement model, so they are one struct
//! here rather than a trait -- Python's split is inheritance, not behaviour.
//! What differs is radius / grid width / speed (see [`DroneKind`]) and what the
//! drone does in the think pass, which is [`DroneBehaviour`]:
//!
//!   * Zap and Mini never think at all -- type 14 moves but is absent from the
//!     think pass entirely (`docs/ENTITY_INTEGRATION_SPEC.md` §4). Contact kills.
//!   * Chaser (15) latches onto a corridor sighting and commits to a direction at
//!     double speed. Contact kills: it reuses the zap drone's logical collision
//!     verbatim (spec §6.6, shared slot1 `0x104007F0`), which is why it gets its
//!     lethality for free.
//!   * Chaingun (12) and Laser (13) run the shared `EntityDroneShooterBase` state
//!     machine and **do not kill on contact** -- their bodies inherit the
//!     `return 0` default (spec §6.6), so only the shot is lethal.
//!
//! Drones do not collide against tile segments; they test half-cell grid edges
//! via [`crate::grid_edges::GridEdges`], which is why that had to land first.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::collision::raycast::{beam_hits_ninja, get_raycast_distance, raycast_vs_player};
use crate::constants::{
    DRONE_CHAINGUN_SPEED, DRONE_CHASER_CORRIDOR_HALF_WIDTH, DRONE_GRID_SIZE, DRONE_LASER_SPEED,
    DRONE_RADIUS, DRONE_ZAP_SPEED, MINI_DRONE_GRID_SIZE, MINI_DRONE_RADIUS, MINI_DRONE_SPEED,
    NINJA_RADIUS,
};
use crate::entities::{EntityCtx, SegmentGrid};
use crate::grid_edges::GridEdges;
use crate::ninja::Ninja;
use crate::segments::Segment;
use crate::types::{clamp_cell, Cell};

/// Loop bound on a corridor cell walk, mirroring the floor guard's `100.0`.
const MAX_CELL_WALK: i32 = 100;

/// Direction index -> unit vector. 0:right 1:down 2:left 3:up.
const DIR_TO_VEC: [(f64, f64); 4] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];

/// Per-mode preference order for the next direction, as *relative* turns
/// (0 keep forward, 1 turn right, 2 go backward, 3 turn left).
/// Modes: 0 follow wall CW, 1 follow wall CCW, 2 wander CW, 3 wander CCW.
const DIR_LIST: [[u8; 4]; 4] = [[1, 0, 3, 2], [3, 0, 1, 2], [0, 1, 3, 2], [0, 3, 1, 2]];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroneKind {
    Zap,
    Mini,
    Chaser,
    Chaingun,
    Laser,
}

impl DroneKind {
    pub fn radius(self) -> f64 {
        match self {
            DroneKind::Mini => MINI_DRONE_RADIUS,
            // The chaser derives from the zap drone and the shooters from the
            // drone base, all of which keep the 7.5 default.
            _ => DRONE_RADIUS,
        }
    }
    pub fn grid_width(self) -> f64 {
        match self {
            DroneKind::Mini => MINI_DRONE_GRID_SIZE,
            _ => DRONE_GRID_SIZE,
        }
    }
    /// Patrol speed. Every one of these is `n/7` except the mini drone's `1.3`
    /// (spec §6.4 -- a decoded speed that is not n/7 should be suspected).
    pub fn speed(self) -> f64 {
        match self {
            DroneKind::Zap => DRONE_ZAP_SPEED,
            DroneKind::Mini => MINI_DRONE_SPEED,
            // The chaser ctor calls the DroneZap ctor and adds no constants.
            DroneKind::Chaser => DRONE_ZAP_SPEED,
            DroneKind::Chaingun => DRONE_CHAINGUN_SPEED,
            DroneKind::Laser => DRONE_LASER_SPEED,
        }
    }
    pub fn type_id(self) -> u8 {
        match self {
            DroneKind::Zap => 14,
            DroneKind::Mini => 26,
            DroneKind::Chaser => 15,
            DroneKind::Chaingun => 12,
            DroneKind::Laser => 13,
        }
    }
    /// Whether contact with the body is lethal.
    ///
    /// True for the zap/mini/chaser line, which all share slot1 `0x104007F0`.
    /// **False for the shooters**: types 12 and 13 inherit the `return 0` default
    /// alongside the turret and dual laser, so a kill-on-touch here would be a
    /// hazard the game does not have (spec §6.6).
    pub fn body_is_lethal(self) -> bool {
        !matches!(self, DroneKind::Chaingun | DroneKind::Laser)
    }
    /// Death-cause string for a contact kill, matching the pure-Python engine's
    /// per-class `KILL_CAUSE` exactly.
    ///
    /// A single `"drone"` for all five kinds is what this used to emit, and it
    /// made the two engines disagree: Python reports `drone_zap` / `mini_drone` /
    /// `drone_chaser`, so a native-backed training run could not distinguish a
    /// deterministic patrol death (a policy failure, avoidable from the forecast
    /// channels) from a chaser death (reactive, a different problem entirely).
    ///
    /// The shooters are `unreachable!` rather than given a string of their own:
    /// `body_is_lethal` is false for them and guards the sole caller two lines
    /// above the call, so a name here would be a cause the pure-Python engine
    /// can never emit -- a phantom TensorBoard row, and a backend divergence of
    /// exactly the kind this method was added to remove. Their kills come from
    /// the discharge (`drone_shot`), never from the body.
    pub fn kill_cause(self) -> &'static str {
        match self {
            DroneKind::Zap => "drone_zap",
            DroneKind::Mini => "mini_drone",
            DroneKind::Chaser => "drone_chaser",
            DroneKind::Chaingun | DroneKind::Laser => {
                unreachable!("shooter drone bodies are harmless; see body_is_lethal")
            }
        }
    }
    pub fn behaviour(self) -> DroneBehaviour {
        match self {
            // Type 14 moves but never thinks; the mini drone likewise.
            DroneKind::Zap | DroneKind::Mini => DroneBehaviour::Patrol,
            DroneKind::Chaser => DroneBehaviour::Chaser(ChaserState::default()),
            DroneKind::Chaingun | DroneKind::Laser => {
                DroneBehaviour::Shooter(ShooterState::default())
            }
        }
    }
}

/// Per-frame charge/fire timings for the two shooter drones.
///
/// Spec §8.3 recovered the state machine, the five hook seam and the field
/// offsets, but the hook bodies and the writers of `+0x70`/`+0x74` are unread,
/// so these were `[I]` and were chosen to match the shipped game's feel. They
/// are now read off the captures' own `+0x7C` state field instead. They match
/// `entity_drone_shooter_base.py` exactly; change both sides together.
#[derive(Debug, Clone, Copy)]
pub struct ShooterTimings {
    pub charge: i32,
    pub fire: i32,
    pub cooldown: i32,
}

impl ShooterTimings {
    pub fn for_kind(kind: DroneKind) -> Self {
        // **Type-specific after all.** An earlier round pooled types 12 and 13
        // into one measurement and concluded they were shared. That pooling was
        // invalid: the structs are different sizes -- 0xE8 for the chaingun,
        // 0xD8 for the laser drone -- so half the samples were reading whatever
        // lived at that offset in the other layout.
        //
        // Split by type, both unanimous across every captured run:
        //
        //     type 12   state 1  53 x78   state 2  80 x78   state 3  90 x78
        //     type 13   state 1  46 x19   state 2  80 x19   state 3  60 x19
        //
        // The type-13 values are correct and OFF by default: 46/60 costs 4 demos
        // (smoke 1381 -> 1377). A shorter charge and cooldown mean more
        // discharges, and this port's shot over-kills by roughly 10x, so more
        // discharges means more false kills. Turn them on with the hit test.
        // `NCLONE_AB_DRONE_TYPE_TIMINGS=1`.
        match kind {
            DroneKind::Laser if crate::ab::ab().drone_type_timings => {
                ShooterTimings { charge: 46, fire: 80, cooldown: 60 }
            }
            _ => ShooterTimings { charge: 53, fire: 80, cooldown: 90 },
        }
    }
}

/// What a drone does during the think pass.
#[derive(Debug, Clone)]
pub enum DroneBehaviour {
    /// Types 14 and 26: absent from the think pass entirely.
    Patrol,
    Chaser(ChaserState),
    Shooter(ShooterState),
}

/// Chaser drone (15) state. Spec §8.4.
///
/// **The chase is bounded by a goal, not by sight.** §8.4 records that acquisition
/// writes `+0x88` (a committed direction) and `+0x80` (a goal, copied into
/// `+0x20`) -- and `+0x20` is the drone base's target position, so acquisition
/// retargets the patrol machinery at the cell the ninja was seen in rather than
/// switching on an endless pursuit. The latch clears on arrival, which re-arms
/// `think`. The alternative reading (latch forever) was measured against the
/// attract corpus and killed the ninja in every demo containing a chaser.
#[derive(Debug, Clone, Default)]
pub struct ChaserState {
    /// `+0x70`. While set, `think` does not run its detection at all.
    pub chasing: bool,
    /// `+0x88`. The direction committed to on acquisition.
    pub chase_dir: Option<u8>,
    /// `+0x80`. Cell-centre goal the chase runs toward.
    pub goal: Option<(f64, f64)>,
}

/// Shooter drone (12/13) state. Spec §8.3.
#[derive(Debug, Clone)]
pub struct ShooterState {
    /// `+0x7C`: 0 idle, 1 aiming, 2 firing, 3 cooldown.
    pub state: u8,
    /// `+0x78`. Charge timer while aiming, cooldown timer while cooling. It is
    /// frozen for the whole of state 2 in every capture -- all 51 state-2 entries
    /// read 53 and it never advances -- so the discharge uses `fire_timer`.
    pub timer: i32,
    /// Counts the discharge out. Whatever the game uses for it is not `+0x78`.
    pub fire_timer: i32,
    /// `+0x80`. The game stores a ninja index with -1 for none.
    pub has_target: bool,
    pub aim_dx: f64,
    pub aim_dy: f64,
    /// Traced shot endpoint, for rendering and for the lethality test.
    pub xend: f64,
    pub yend: f64,
    /// The raycast distance that produced the endpoint. The hit test takes it as
    /// a parameter rather than recomputing it -- see .
    pub beam_len: Option<f64>,
    /// `+0x88` shots fired this discharge and `+0x8C` frames since the last one.
    /// Used only by the decompiled cycle; see `think_shooter_decomp`.
    pub shots: i32,
    pub shot_tick: i32,
    /// `+0x90`/`+0x98`, the discharge's frozen base direction, and
    /// `+0xA0`/`+0xA8`, its signed perpendicular. Both set once, by `fire_setup`.
    pub fire_dx: f64,
    pub fire_dy: f64,
    pub perp_x: f64,
    pub perp_y: f64,
}

// -- the decompiled chaingun weapon cycle ------------------------------------
//
// Read out of the ctor rather than fitted: `FUN_103FE460` calls
// `FUN_104001B0(..., 0x34, 0x5A, 0xC)`, which stores its two ints into `+0x70`
// and `+0x74`, and then writes `+0xB0 <- DAT_10572398 = 3.0`.

/// `+0x70`, compared `+0x70 < timer + 1` -- 53 frames of aiming.
const CHARGE_TICKS: i32 = 52;
/// `+0x74`, compared `+0x74 <= timer`.
const COOLDOWN_TICKS: i32 = 90;
/// `+0xB0`, compared `tick > +0xB0` -- a shot every 4 frames.
const SHOT_PERIOD: f64 = 3.0;
// The laser drone's own three, from `FUN_103FF4C0`:
// `FUN_104001B0(..., 0x2D, 0x3C, 0xD)` then `+0x88 <- 0x50`.
/// 46 frames of charge.
const LASER_CHARGE_TICKS: i32 = 45;
/// 60 frames of cooldown.
const LASER_COOLDOWN_TICKS: i32 = 60;
/// `+0x88`, compared `+0x88 <= +0x8C` after the counter is bumped.
const LASER_FIRE_TICKS: i32 = 80;
/// `if (0x12 < shots) return done`: 19 shots leave, numbered 1..19.
const MAX_SHOTS: i32 = 18;
const SWEEP_DIV: f64 = 18.0;
const SWEEP_SPAN: f64 = 0.3;
/// `DAT_105705C0`. The binary guards every normalisation with
/// `max(|len|, 1.0) * eps < |len|` rather than a bare `len != 0`.
const NORM_EPS: f64 = 2.220446049250313e-16;

impl Default for ShooterState {
    fn default() -> Self {
        ShooterState {
            state: 0,
            timer: 0,
            fire_timer: 0,
            has_target: false,
            aim_dx: 1.0,
            aim_dy: 0.0,
            beam_len: None,
            xend: 0.0,
            yend: 0.0,
            shots: 0,
            shot_tick: 0,
            fire_dx: 1.0,
            fire_dy: 0.0,
            perp_x: 0.0,
            perp_y: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Drone {
    pub kind: DroneKind,
    pub xpos: f64,
    pub ypos: f64,
    pub xtarget: f64,
    pub ytarget: f64,
    pub dir: u8,
    pub dir_old: u8,
    pub orientation: u8,
    pub mode: u8,
    pub active: bool,
    /// Occupancy cell, updated only by `grid_move` -- deliberately allowed to go
    /// stale, see `move_entity`.
    pub cell: Cell,
    /// Current speed. A field rather than `kind.speed()` because the chaser
    /// doubles it on acquisition.
    pub speed: f64,
    pub behaviour: DroneBehaviour,
    pub timings: ShooterTimings,

    initial_xpos: f64,
    initial_ypos: f64,
    initial_dir: u8,
}

impl Drone {
    pub fn new(kind: DroneKind, xcoord: i32, ycoord: i32, orientation: u8, mode: u8) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        // Python: `self.turn(orientation // 2)` before `self.orientation` is set.
        let dir = orientation / 2;
        // Only modes 0-3 exist. A malformed level (procgen has produced these)
        // would KeyError in Python's move(); wrap instead so one bad entity
        // cannot kill a training run. Identity for valid modes.
        let mode = if (mode as usize) < DIR_LIST.len() {
            mode
        } else {
            mode % DIR_LIST.len() as u8
        };
        Drone {
            kind,
            xpos,
            ypos,
            xtarget: xpos,
            ytarget: ypos,
            dir,
            // `turn` sets `dir_old = self.dir or dir`, and on the first call
            // self.dir is None -> dir_old == dir.
            dir_old: dir,
            orientation,
            mode,
            active: true,
            cell: clamp_cell((xpos / 24.0).floor() as i32, (ypos / 24.0).floor() as i32),
            speed: kind.speed(),
            behaviour: kind.behaviour(),
            timings: ShooterTimings::for_kind(kind),
            initial_xpos: xpos,
            initial_ypos: ypos,
            initial_dir: dir,
        }
    }

    /// Whether this drone participates in the think pass at all.
    /// Per-frame velocity, px/frame, in world coordinates.
    ///
    /// A drone always travels along its facing at its current speed -- the
    /// chaser doubles `speed` on acquisition, which is why this reads the field
    /// rather than `kind.speed()`. Exposed for the threat observations, which
    /// need the direction of travel and not just the position: a drone walking
    /// away and a drone walking towards you are the same still frame.
    pub fn velocity(&self) -> (f64, f64) {
        let (vx, vy) = DIR_TO_VEC[(self.dir & 3) as usize];
        (self.speed * vx, self.speed * vy)
    }

    pub fn is_thinkable(&self) -> bool {
        !matches!(self.behaviour, DroneBehaviour::Patrol)
    }

    /// Port of `EntityDroneBase.turn`.
    ///
    /// Python writes `self.dir_old = self.dir or dir`, and `0 or dir` is `dir` --
    /// so turning away from direction 0 (right) records the NEW direction as
    /// `dir_old`, not 0. That is a bug in the original, but `dir_old` feeds the
    /// replay log, so reproducing it keeps exports byte-comparable.
    fn turn(&mut self, dir: u8) {
        self.dir_old = if self.dir != 0 { self.dir } else { dir };
        self.dir = dir;
    }

    /// Port of `EntityDroneBase.move`.
    pub fn move_entity(&mut self, ctx: &mut EntityCtx) {
        if let Some((old_cell, new_cell)) = self.step_motion(ctx.grid_edges, ctx.ninja) {
            ctx.grid_move(old_cell, new_cell);
        }
    }

    /// The motion half of [`Self::move_entity`] -- everything except the
    /// entity-grid bookkeeping, which is the only part that needs `&mut EntityCtx`.
    ///
    /// Split out so the drone hazard forecast can roll a *cloned* drone forward
    /// without mutating the live `entity_grid`. Prediction deliberately shares
    /// this body rather than reimplementing it: a second copy of the patrol
    /// rules would drift from the sim silently, and a forecast that disagrees
    /// with what actually happens is worse than no forecast at all.
    ///
    /// Returns `Some((old_cell, new_cell))` when the drone stepped, `None` when
    /// it declined to (a shooter mid-weapon-cycle) -- so callers reproduce the
    /// binary's "no `grid_move` on that frame" exactly rather than issuing a
    /// same-cell move.
    fn step_motion(&mut self, edges: &GridEdges, ninja: &Ninja) -> Option<(Cell, Cell)> {
        // A shooter drone patrols only while IDLE. It stops dead the frame it
        // acquires and does not move again until the cooldown ends.
        //
        // Measured, not inferred: over both capture batches, 32861 shooter-drone
        // frames, the game moves it on all 21488 frames in state 0 and on NONE
        // of the 11373 frames in states 1, 2 and 3. Not one exception.
        //
        // Patrolling through the weapon cycle put the drone somewhere the real
        // one never was by the time it fired -- on 10131718 the game's drone
        // stops at (518.857, 300) on frame 37 and stays, while this one had
        // walked to (540, 306) by frame 156 and shot from a corridor the real
        // one had already left. `NCLONE_AB_DRONE_PATROL_ALWAYS=1` restores it.
        if let DroneBehaviour::Shooter(s) = &self.behaviour {
            if s.state != 0 && !crate::ab::ab().drone_patrol_always {
                return None;
            }
        }
        let speed = self.speed;
        let (vx, vy) = DIR_TO_VEC[(self.dir & 3) as usize];
        let xspeed = speed * vx;
        let yspeed = speed * vy;
        let dx = self.xtarget - self.xpos;
        let dy = self.ytarget - self.ypos;
        let dist = (dx * dx + dy * dy).sqrt();

        // Reached or overshot the target cell centre?
        let overshoot = dx * (self.xtarget - (self.xpos + xspeed))
            + dy * (self.ytarget - (self.ypos + yspeed));
        if dist < 0.000001 || overshoot < 0.0 {
            self.xpos = self.xtarget;
            self.ypos = self.ytarget;
            if self.choose_next_direction_and_goal(edges, ninja) {
                // `self.speed`, NOT the `speed` cached above: a chaser that
                // arrives at its goal ends the chase inside the call above, which
                // drops it from 16/7 back to 8/7. Python reads the field after
                // the call, so the very step it leaves the goal on is already a
                // patrol step. Caching diverges by exactly 8/7.
                // Clamped at zero, as `FUN_103FADA0`
                // (`docs/decomp/HELPER_dronebase_move_103FADA0.c:56`) does --
                // otherwise a drone whose heading is not the direction of its
                // target steps *backwards* by `dist - speed`. See the Python
                // twin in `entity_drone_base.py`.
                let disp = (self.speed - dist).max(0.0);
                let (nvx, nvy) = DIR_TO_VEC[(self.dir & 3) as usize];
                self.xpos += disp * nvx;
                self.ypos += disp * nvy;
            }
        } else {
            self.xpos += xspeed;
            self.ypos += yspeed;
        }
        // Unconditional, on both branches, matching the single call the binary
        // makes after the if/else.
        let new_cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
        let old_cell = self.cell;
        self.cell = new_cell;
        Some((old_cell, new_cell))
    }

    /// Roll a copy of this drone forward `frames` frames, sampling its position
    /// at each frame index in `sample_at` (which must be sorted ascending).
    ///
    /// Only meaningful for [`DroneBehaviour::Patrol`] -- zap (14) and mini (26).
    /// Their motion is a pure function of the tile grid and the frame counter,
    /// with no ninja dependence, so the roll-out is EXACT rather than an
    /// estimate. The chaser and the shooters both read the ninja, so forecasting
    /// them would be predicting the agent's own future; callers must not.
    pub fn forecast(&self, edges: &GridEdges, ninja: &Ninja, sample_at: &[u32]) -> Vec<(f64, f64)> {
        let mut ghost = self.clone();
        let mut out = Vec::with_capacity(sample_at.len());
        let mut frame = 0u32;
        for &target in sample_at {
            while frame < target {
                ghost.step_motion(edges, ninja);
                frame += 1;
            }
            out.push((ghost.xpos, ghost.ypos));
        }
        out
    }

    /// Try each direction in this mode's preference order; adopt the first that
    /// is not walled off. Returns false if the drone is boxed in.
    fn choose_next_direction_and_goal(&mut self, edges: &GridEdges, ninja: &Ninja) -> bool {
        // `FUN_103FEC60`, the chaser's own override of this slot. Twin of
        // `EntityDroneChaser.choose_next_direction_and_goal`; change both.
        if !crate::ab::ab().chaser_legacy && matches!(self.behaviour, DroneBehaviour::Chaser(_)) {
            let (chasing, mut commit) = match &self.behaviour {
                DroneBehaviour::Chaser(s) => (s.chasing, s.chase_dir),
                _ => (false, None),
            };
            if chasing {
                if let Some(d) = commit {
                    // The game writes `+0x40` here directly, without `turn`.
                    self.dir = d;
                }
                let d = self.dir;
                if self.test_next_direction_and_goal(d, edges) {
                    return true;
                }
                // Blocked. The chase ends AT THE WALL, which is what
                // `chaseend.py` measured: runs end at walls, not at a goal cell.
                let off = crate::ab::ab().drone_dirlist[self.mode as usize][0];
                self.dir = (self.dir + 4 - off) % 4;
                self.speed = self.kind.speed();
                self.behaviour = DroneBehaviour::Chaser(ChaserState {
                    chasing: false,
                    chase_dir: commit,
                    goal: None,
                });
            }
            // The committed direction outlives the latch by exactly this long:
            // it seeds the next search and is then cleared.
            let mut start = self.dir;
            if let Some(d) = commit {
                start = d;
                commit = None;
                self.behaviour = DroneBehaviour::Chaser(ChaserState::default());
            }
            let _ = commit;
            if ninja.is_valid_target() {
                let (ox, oy) = (self.xpos, self.ypos);
                if let Some(d) = self.acquire_from(ox, oy, start, ninja, edges) {
                    self.latch_chase(d, true);
                    // The second `FUN_103FE0D0` call, the one that is not a
                    // test: here the buffer it fills IS `+0x30`/`+0x38`.
                    self.test_next_direction_and_goal(d, edges);
                    return true;
                }
            }
            for i in 0..4usize {
                let new_dir =
                    (self.dir + crate::ab::ab().drone_dirlist[self.mode as usize][i]) % 4;
                if self.test_next_direction_and_goal(new_dir, edges) {
                    self.turn(new_dir);
                    return true;
                }
            }
            return false;
        }
        // A chasing drone steers toward its goal instead of following patrol
        // preference; on arrival the chase ends and `think` can acquire again.
        // Must stay in step with `EntityDroneChaser.choose_next_direction_and_goal`.
        let chase_goal = match &self.behaviour {
            DroneBehaviour::Chaser(ChaserState {
                chasing: true,
                goal: Some(goal),
                ..
            }) => Some(*goal),
            _ => None,
        };
        // Committed run: hold the acquisition heading until blocked, then
        // fall back to patrol. Twin of `_STRAIGHT_CHASE` in the Python chaser.
        if crate::ab::ab().chaser_straight {
            let chase_dir = match &self.behaviour {
                DroneBehaviour::Chaser(ChaserState {
                    chasing: true,
                    chase_dir: Some(d),
                    ..
                }) => Some(*d),
                _ => None,
            };
            if let Some(d) = chase_dir {
                if self.test_next_direction_and_goal(d, edges) {
                    self.turn(d);
                    return true;
                }
                self.end_chase();
            }
        } else if let Some((gx, gy)) = chase_goal {
            let grid_width = self.kind.grid_width();
            let dx = gx - self.xpos;
            let dy = gy - self.ypos;
            // Largest remaining component first, then the other.
            let mut candidates: Vec<u8> = Vec::with_capacity(2);
            if dx.abs() >= grid_width / 2.0 {
                candidates.push(if dx > 0.0 { 0 } else { 2 });
            }
            if dy.abs() >= grid_width / 2.0 {
                candidates.push(if dy > 0.0 { 1 } else { 3 });
            }
            if dy.abs() > dx.abs() {
                candidates.reverse();
            }
            for direction in candidates {
                if self.test_next_direction_and_goal(direction, edges) {
                    self.turn(direction);
                    return true;
                }
            }
            // Arrived, or boxed in on every approach: give up the chase.
            self.end_chase();
        }
        for i in 0..4usize {
            let new_dir =
                (self.dir + crate::ab::ab().drone_dirlist[self.mode as usize][i]) % 4;
            if self.test_next_direction_and_goal(new_dir, edges) {
                self.turn(new_dir);
                return true;
            }
        }
        false
    }

    /// Can the drone reach the adjacent cell centre along `dir`? If so, record it
    /// as the new target. Port of `test_next_direction_and_goal`.
    ///
    /// Note the half-cell (÷12) granularity and that the leading edge is probed
    /// at `pos + dir * RADIUS` while the lateral extent spans `pos ± RADIUS` --
    /// the drone is a circle, tested as a square footprint against grid edges.
    fn test_next_direction_and_goal(&mut self, dir: u8, edges: &GridEdges) -> bool {
        let radius = self.kind.radius();
        let grid_width = self.kind.grid_width();
        let (xdirf, ydirf) = DIR_TO_VEC[(dir & 3) as usize];
        let xtarget = self.xpos + grid_width * xdirf;
        let ytarget = self.ypos + grid_width * ydirf;
        let xdir = xdirf as i32;
        let ydir = ydirf as i32;

        if ydir == 0 {
            let mut cell_x = ((self.xpos + xdirf * radius) / 12.0).floor() as i32;
            let cell_xtarget = ((xtarget + xdirf * radius) / 12.0).floor() as i32;
            let cell_y1 = ((self.ypos - radius) / 12.0).floor() as i32;
            let cell_y2 = ((self.ypos + radius) / 12.0).floor() as i32;
            while cell_x != cell_xtarget {
                if !edges.is_empty_column(cell_x, cell_y1, cell_y2, xdir) {
                    return false;
                }
                cell_x += xdir;
            }
        } else {
            let mut cell_y = ((self.ypos + ydirf * radius) / 12.0).floor() as i32;
            let cell_ytarget = ((ytarget + ydirf * radius) / 12.0).floor() as i32;
            let cell_x1 = ((self.xpos - radius) / 12.0).floor() as i32;
            let cell_x2 = ((self.xpos + radius) / 12.0).floor() as i32;
            while cell_y != cell_ytarget {
                if !edges.is_empty_row(cell_x1, cell_x2, cell_y, ydir) {
                    return false;
                }
                cell_y += ydir;
            }
        }
        self.xtarget = xtarget;
        self.ytarget = ytarget;
        true
    }

    /// Think pass. Only the chaser and the two shooters reach this.
    pub fn think(&mut self, ctx: &mut EntityCtx) {
        match &self.behaviour {
            DroneBehaviour::Patrol => {}
            DroneBehaviour::Chaser(_) => self.think_chaser(ctx.grid_edges, ctx.ninja),
            DroneBehaviour::Shooter(_) => {
                let (segment_grid, segments, edges) =
                    (ctx.segment_grid, ctx.segments, ctx.grid_edges);
                self.think_shooter(segment_grid, segments, edges, ctx.ninja);
            }
        }
    }

    /// Chaser (15): latching corridor detection. Spec §8.4.
    ///
    /// Detection is a **±12px perpendicular corridor, not a cone**, along one of
    /// only **three** directions (`dir-1`, `dir`, `dir+1`) -- a chaser is blind
    /// directly behind itself. Both facts are [V]; the guide's "forward cone"
    /// reading of the abandoned Python draft was close but wrong.
    fn think_chaser(&mut self, edges: &GridEdges, ninja: &Ninja) {
        if !crate::ab::ab().chaser_legacy {
            let chasing = matches!(&self.behaviour,
                                   DroneBehaviour::Chaser(s) if s.chasing);
            if chasing || !ninja.is_valid_target() {
                return;
            }
            // `FUN_103FF1B0` measures from `+0x30`/`+0x38`, not the position,
            // and does NOT turn the drone -- the turn waits for the next cell
            // centre, inside `FUN_103FEC60`.
            let (ox, oy) = (self.xtarget, self.ytarget);
            let start = self.dir;
            if let Some(d) = self.acquire_from(ox, oy, start, ninja, edges) {
                self.latch_chase(d, false);
            }
            return;
        }
        let DroneBehaviour::Chaser(state) = &self.behaviour else {
            return;
        };
        // Twin of `_INTERRUPT` in entity_drone_chaser.py: a running chase
        // can be interrupted by a fresh sighting on another axis.
        if (state.chasing && !crate::ab::ab().chaser_interrupt)
            || !ninja.is_valid_target()
        {
            return;
        }
        for k in -1i32..=1 {
            // (k + 4 + dir) & 3: forward and both sides, never backward.
            let s = ((k + 4 + self.dir as i32) % 4) as u8;
            if !self.ninja_in_corridor(s, ninja, edges) {
                continue;
            }
            let grid_width = self.kind.grid_width();
            // Snap the sighting to a cell centre: the goal feeds the same grid
            // machinery `xtarget`/`ytarget` use, which only holds cell centres.
            let goal = (
                ((ninja.xpos / grid_width).floor() + 0.5) * grid_width,
                ((ninja.ypos / grid_width).floor() + 0.5) * grid_width,
            );
            self.speed = self.kind.speed() * 2.0;
            self.behaviour = DroneBehaviour::Chaser(ChaserState {
                chasing: true,
                chase_dir: Some(s),
                goal: Some(goal),
            });
            return;
        }
    }

    /// The five writes every acquisition site of `FUN_103FF1B0` / `FUN_103FEC60`
    /// shares. `+0x80` is `2 * speed`, straight out of the ctor -- not a goal.
    fn latch_chase(&mut self, direction: u8, turn_now: bool) {
        self.speed = self.kind.speed() * 2.0;
        self.behaviour = DroneBehaviour::Chaser(ChaserState {
            chasing: true,
            chase_dir: Some(direction),
            goal: None,
        });
        if turn_now {
            self.dir = direction;
        }
    }

    /// `dir-1`, `dir`, `dir+1` through all four gates. The direction, or None.
    ///
    /// `ox`/`oy` is the origin the caller measures from: `FUN_103FF1B0` uses
    /// `+0x30`/`+0x38` and `FUN_103FEC60` uses `+0x10`/`+0x18`. They coincide at a
    /// cell centre, which is the only place the second one runs.
    fn acquire_from(
        &mut self,
        ox: f64,
        oy: f64,
        start: u8,
        ninja: &Ninja,
        edges: &GridEdges,
    ) -> Option<u8> {
        for k in -1i32..=1 {
            let s = (((start as i32) + 4 + k) % 4) as u8;
            let (xdir, ydir) = DIR_TO_VEC[s as usize];
            let dx = ninja.xpos - ox;
            let dy = ninja.ypos - oy;
            if !(0.0 < dy * ydir + dx * xdir) {
                continue;
            }
            if (dy * xdir - dx * ydir).abs() > DRONE_CHASER_CORRIDOR_HALF_WIDTH {
                continue;
            }
            let extent = self.extent_from(s, ox, oy, edges);
            let (here, there) = if ydir != 0.0 {
                ((oy / 12.0).floor() as i32, (ninja.ypos / 12.0).floor() as i32)
            } else {
                ((ox / 12.0).floor() as i32, (ninja.xpos / 12.0).floor() as i32)
            };
            if there < here.min(extent) || here.max(extent) < there {
                continue;
            }
            // `FUN_103FE0D0` is a PURE test here -- it writes its goal to a
            // caller-supplied stack buffer, not to `+0x30`. Ours assigns, so
            // save and restore (§64).
            let (sx, sy) = (self.xtarget, self.ytarget);
            let ok = self.test_next_direction_and_goal(s, edges);
            self.xtarget = sx;
            self.ytarget = sy;
            if ok {
                return Some(s);
            }
        }
        None
    }

    /// `FUN_103FF070` (horizontal) / `FUN_103FF100` (vertical), walked from
    /// `(ox, oy)`: step one cell at a time testing a single-cell span, and return
    /// the last cell reached. On overrun the game returns the unmoved start.
    fn extent_from(&self, direction: u8, ox: f64, oy: f64, edges: &GridEdges) -> i32 {
        let (_, ydirf) = DIR_TO_VEC[direction as usize];
        let sign: i32 = if direction < 2 { 1 } else { -1 };
        if ydirf == 0.0 {
            let span = (oy / 12.0).floor() as i32;
            let start = (ox / 12.0).floor() as i32;
            let mut cur = start;
            for _ in 0..MAX_CELL_WALK {
                if !edges.is_empty_column(cur, span, span, sign) {
                    return cur;
                }
                cur += sign;
            }
            start
        } else {
            let span = (ox / 12.0).floor() as i32;
            let start = (oy / 12.0).floor() as i32;
            let mut cur = start;
            for _ in 0..MAX_CELL_WALK {
                if !edges.is_empty_row(span, span, cur, sign) {
                    return cur;
                }
                cur += sign;
            }
            start
        }
    }

    /// Drop back to patrol, which re-arms `think`.
    fn end_chase(&mut self) {
        self.speed = self.kind.speed();
        self.behaviour = DroneBehaviour::Chaser(ChaserState::default());
    }

    /// Is the ninja lined up along `direction`, within the corridor, in sight?
    ///
    /// The drone family's one notion of "I can see something that way", shared
    /// deliberately rather than reimplemented per kind: the chaser runs it over
    /// three directions to decide whether to latch a chase (spec §8.4, [V] -- a
    /// **corridor, not a cone**, and never backward), and the shooters run it over
    /// their single facing direction to decide whether to fire, because a drone's
    /// gun points along the axis it faces. Free-aiming a hitscan at the ninja
    /// instead makes those drones unmissable -- measured against the attract
    /// corpus, it killed the ninja in 43 of 43 demos the real game let through.
    fn ninja_in_corridor(&self, direction: u8, ninja: &Ninja, edges: &GridEdges) -> bool {
        let (xdir, ydir) = DIR_TO_VEC[direction as usize];
        let dx = ninja.xpos - self.xpos;
        let dy = ninja.ypos - self.ypos;
        if dy * ydir + dx * xdir <= 0.0 {
            return false; // behind, or exactly abeam
        }
        if (dy * xdir - dx * ydir).abs() > DRONE_CHASER_CORRIDOR_HALF_WIDTH {
            return false; // outside the corridor
        }
        self.corridor_clear(direction, ninja.xpos, ninja.ypos, edges)
    }

    /// Is the straight run of cells from here to the target unobstructed?
    ///
    /// Stands in for `FUN_103FE0D0` plus the two axis-dependent extent checks
    /// (`FUN_103FF070`, `FUN_103FF100`), all three unread. Their stated job is
    /// "confirm the ninja's cell lies within the clear run", which is
    /// `test_next_direction_and_goal`'s grid-edge probe generalised from one cell
    /// to the whole distance -- so it reuses the same half-cell granularity and
    /// radius-offset footprint rather than inventing a second notion of blocked.
    fn corridor_clear(
        &self,
        direction: u8,
        target_x: f64,
        target_y: f64,
        edges: &GridEdges,
    ) -> bool {
        let radius = self.kind.radius();
        let (xdirf, ydirf) = DIR_TO_VEC[direction as usize];
        let mut steps = 0;
        if ydirf == 0.0 {
            let xdir = xdirf as i32;
            let mut cell_x = ((self.xpos + xdirf * radius) / 12.0).floor() as i32;
            let cell_xtarget = ((target_x + xdirf * radius) / 12.0).floor() as i32;
            let cell_y1 = ((self.ypos - radius) / 12.0).floor() as i32;
            let cell_y2 = ((self.ypos + radius) / 12.0).floor() as i32;
            while cell_x != cell_xtarget {
                if !edges.is_empty_column(cell_x, cell_y1, cell_y2, xdir) {
                    return false;
                }
                cell_x += xdir;
                steps += 1;
                if steps > MAX_CELL_WALK {
                    return false;
                }
            }
        } else {
            let ydir = ydirf as i32;
            let mut cell_y = ((self.ypos + ydirf * radius) / 12.0).floor() as i32;
            let cell_ytarget = ((target_y + ydirf * radius) / 12.0).floor() as i32;
            let cell_x1 = ((self.xpos - radius) / 12.0).floor() as i32;
            let cell_x2 = ((self.xpos + radius) / 12.0).floor() as i32;
            while cell_y != cell_ytarget {
                if !edges.is_empty_row(cell_x1, cell_x2, cell_y, ydir) {
                    return false;
                }
                cell_y += ydir;
                steps += 1;
                if steps > MAX_CELL_WALK {
                    return false;
                }
            }
        }
        true
    }

    /// Shooters (12/13): the shared `EntityDroneShooterBase` state machine.
    ///
    /// Spec §8.3. The four states and their transitions are [V]; the five hook
    /// bodies, the shared target acquisition and the charge/cooldown durations are
    /// all [U], so everything below the control flow is a reconstruction. Keep in
    /// step with `entity_drone_shooter_base.py`.
    fn think_shooter(
        &mut self,
        segment_grid: &SegmentGrid,
        segments: &[Segment],
        edges: &GridEdges,
        ninja: &mut Ninja,
    ) {
        if crate::ab::ab().drone_decomp
            && matches!(self.kind, DroneKind::Chaingun | DroneKind::Laser)
        {
            self.think_shooter_decomp(segment_grid, segments, ninja);
            return;
        }
        let timings = self.timings;
        let (xpos, ypos) = (self.xpos, self.ypos);
        // The firing lane is the drone's facing direction, not a free sightline --
        // see `ninja_in_corridor`. Evaluated before the borrow below because it
        // needs `&self`.
        let in_lane = ninja.is_valid_target() && self.ninja_in_corridor(self.dir, ninja, edges);
        let facing = DIR_TO_VEC[(self.dir & 3) as usize];
        let DroneBehaviour::Shooter(s) = &mut self.behaviour else {
            return;
        };

        let visible = in_lane
            && raycast_vs_player(segment_grid, segments, xpos, ypos, ninja.xpos, ninja.ypos);

        match s.state {
            // IDLE. State 0 has no transition of its own in the decompiled
            // switch, so acquisition has to move it on.
            0 => {
                if visible {
                    s.has_target = true;
                    s.state = 1;
                    s.timer = 0;
                    aim_along_facing(s, facing);
                }
            }
            // AIMING / CHARGING.
            1 => {
                // The charge is NEVER aborted. All 51 completed state-1 runs in
                // the capture corpus are exactly 53 frames; an abort would show
                // as a short one and there is not one. Dropping to cooldown when
                // the target went out of the lane cut a charge off after 9
                // frames on 10132175 that the game ran to completion.
                // `NCLONE_AB_DRONE_ABORT=1` restores it.
                if crate::ab::ab().drone_abort && !visible {
                    s.timer = 0;
                    s.state = 3;
                    s.xend = xpos;
                    s.yend = ypos;
                    s.has_target = false;
                    return;
                }
                aim_along_facing(s, facing);
                s.timer += 1;
                if timings.charge <= s.timer {
                    s.state = 2;
                    // NOT reset: the game carries the charge timer into state 2
                    // and leaves it there. All 51 captured entries read 53.
                    s.fire_timer = 0;
                    trace_and_kill(s, xpos, ypos, segment_grid, segments, ninja);
                }
            }
            // FIRING. `fire()` is vtable[+0x18] and the decompiled state machine
            // calls it ONCE, on the 1 -> 2 transition; state 2 only counts the
            // discharge out. This engine used to re-test the beam on every frame
            // of the burst, which made it kill where the game does not: on
            // `10138394` and `10132692` the game and the Python engine both let
            // the ninja through and this one did not. Python's `think` has never
            // had the re-test, so the two engines disagreed by exactly these two
            // demos. `NCLONE_AB_DRONE_BURST=1` restores it.
            2 => {
                if crate::ab::ab().drone_burst {
                    trace_and_kill(s, xpos, ypos, segment_grid, segments, ninja);
                }
                // Incremented BEFORE the test: the frame the state was
                // entered counts as the first of the discharge, so the dwell is
                // `fire` frames inclusive. Testing first spent 81, not 80.
                s.fire_timer += 1;
                if s.fire_timer >= timings.fire {
                    s.timer = 0;
                    s.state = 3;
                    s.xend = xpos;
                    s.yend = ypos;
                }
            }
            // COOLDOWN.
            _ => {
                s.timer += 1;
                if timings.cooldown <= s.timer {
                    if visible {
                        s.has_target = true;
                        s.state = 1;
                        s.timer = 0;
                        aim_along_facing(s, facing);
                    } else {
                        s.state = 0;
                        s.has_target = false;
                    }
                }
            }
        }
    }

    /// `FUN_104005D0` and the five type-12 hooks, as decompiled.
    ///
    /// The reconstruction above misreads the seam: `vtable[+0x1C]`
    /// (`FUN_103FE790`) is not a done-test, it IS the discharge -- it counts the
    /// shot period, fires, kills, and only reports "done" once the shot counter
    /// passes 18. `vtable[+0x18]` (`FUN_103FE510`) fires nothing; it freezes the
    /// aim. Mirrors `EntityDroneShooterBase._think_decomp`; change both together.
    fn think_shooter_decomp(
        &mut self,
        segment_grid: &SegmentGrid,
        segments: &[Segment],
        ninja: &mut Ninja,
    ) {
        let (xpos, ypos) = (self.xpos, self.ypos);
        let is_laser = matches!(self.kind, DroneKind::Laser);
        let (charge, cooldown) = if is_laser {
            (LASER_CHARGE_TICKS, LASER_COOLDOWN_TICKS)
        } else {
            (CHARGE_TICKS, COOLDOWN_TICKS)
        };
        // `FUN_10400240`: nearest valid ninja, then the slack sightline
        // `FUN_1050E8D0` -- which `raycast_vs_player` already ports exactly.
        // Evaluated before the borrow below because it needs `&self`.
        let visible = ninja.is_valid_target()
            && raycast_vs_player(segment_grid, segments, xpos, ypos, ninja.xpos, ninja.ypos);
        let facing = DIR_TO_VEC[(self.dir & 3) as usize];
        let (nx, ny) = (ninja.xpos, ninja.ypos);
        let (nvx, nvy) = (ninja.xspeed, ninja.yspeed);
        // `FUN_10513BD0`: the charge aborts on a dead or celebrating ninja only.
        let ninja_gone = matches!(ninja.state, 6 | 9);
        let DroneBehaviour::Shooter(s) = &mut self.behaviour else {
            return;
        };

        match s.state {
            0 => {
                if visible {
                    s.has_target = true;
                    s.state = 1;
                    s.timer = 0;
                    aim_along_facing(s, facing);
                    on_acquire(s, is_laser, xpos, ypos, nx, ny);
                }
            }
            1 => {
                if !s.has_target || ninja_gone {
                    s.timer = 0;
                    s.state = 3;
                    return;
                }
                aim_along_facing(s, facing);
                let t = s.timer;
                s.timer = t + 1;
                if charge < t + 1 {
                    s.state = 2;
                    if is_laser {
                        // `FUN_103FF770` only resets the discharge counter; the
                        // aim was frozen at acquisition.
                        s.shot_tick = 0;
                    } else {
                        fire_setup(s, xpos, ypos, nx, ny, nvx, nvy);
                    }
                }
            }
            2 => {
                let done = if is_laser {
                    laser_beam_tick(s, xpos, ypos, segment_grid, segments, ninja)
                } else {
                    discharge_tick(s, xpos, ypos, segment_grid, segments, ninja)
                };
                if done {
                    s.timer = 0;
                    s.state = 3;
                }
            }
            _ => {
                s.timer += 1;
                if cooldown <= s.timer {
                    if visible {
                        s.has_target = true;
                        s.state = 1;
                        s.timer = 0;
                        aim_along_facing(s, facing);
                        on_acquire(s, is_laser, xpos, ypos, nx, ny);
                    } else {
                        s.state = 0;
                        s.has_target = false;
                    }
                }
            }
        }
    }

    /// Lethal on contact -- for the kinds whose bodies collide at all.
    pub fn logical_collision(&self, ninja: &mut Ninja) -> bool {
        if !self.kind.body_is_lethal() || !ninja.is_valid_target() {
            return false;
        }
        if overlap_circle_vs_circle(
            self.xpos,
            self.ypos,
            self.kind.radius(),
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ) {
            ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some(self.kind.kill_cause()));
            return true;
        }
        false
    }

    pub fn reset_state(&mut self) {
        self.xpos = self.initial_xpos;
        self.ypos = self.initial_ypos;
        self.xtarget = self.initial_xpos;
        self.ytarget = self.initial_ypos;
        self.dir = self.initial_dir;
        self.dir_old = self.initial_dir;
        self.active = true;
        self.speed = self.kind.speed();
        self.behaviour = self.kind.behaviour();
        self.cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
    }
}

/// `aim()`, vtable[+0x14]: point the gun along the drone's facing. **[I]** -- unread.
///
/// Axis-aligned, from `DIR_TO_VEC`, because the shot leaves along the lane the
/// target was acquired in. If the drone turns during the charge, the lane turns
/// with it and the target is dropped.
fn aim_along_facing(s: &mut ShooterState, facing: (f64, f64)) {
    s.aim_dx = facing.0;
    s.aim_dy = facing.1;
}

/// `fire()`, vtable[+0x18]: cast the shot against geometry, then test the ninja.
/// **[I]** -- unread.
fn trace_and_kill(
    s: &mut ShooterState,
    xpos: f64,
    ypos: f64,
    segment_grid: &SegmentGrid,
    segments: &[Segment],
    ninja: &mut Ninja,
) {
    match get_raycast_distance(segment_grid, segments, xpos, ypos, s.aim_dx, s.aim_dy) {
        Some(len) => {
            s.beam_len = Some(len);
            s.xend = xpos + s.aim_dx * len;
            s.yend = ypos + s.aim_dy * len;
        }
        None => {
            s.beam_len = None;
            s.xend = xpos;
            s.yend = ypos;
            return;
        }
    }
    if ninja.is_valid_target() && beam_hits_ninja(xpos, ypos, s.xend, s.yend, ninja, s.beam_len) {
        debug_shot("trace_and_kill", xpos, ypos, s, ninja);
        ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("drone_shot"));
    }
}

/// Report a `drone_shot` kill when `NCLONE_DEBUG_DRONE_SHOT=1`.
///
/// The two ports disagree about `drone_shot` on 111 attract demos the real game
/// completes, and native's shooter state is not in the entity snapshot, so from
/// Python there is no way to see which of the two kill sites fired or where it
/// aimed. This is the missing half of that comparison.
fn debug_shot(site: &str, xpos: f64, ypos: f64, s: &ShooterState, ninja: &Ninja) {
    if std::env::var("NCLONE_DEBUG_DRONE_SHOT").as_deref() != Ok("1") {
        return;
    }
    eprintln!(
        "[drone_shot] {site} drone=({xpos:.4},{ypos:.4}) aim=({:+.4},{:+.4}) \
         end=({:.4},{:.4}) len={:?} ninja=({:.4},{:.4})",
        s.aim_dx, s.aim_dy, s.xend, s.yend, s.beam_len, ninja.xpos, ninja.ypos
    );
}

/// vtable `+0x10`, the hook `FUN_10400240` calls once it has a target.
///
/// A no-op for the chaingun, whose aim is frozen later at the 1 -> 2 transition.
/// The laser drone freezes its beam HERE: `+0x90`/`+0x98` is written on the frame
/// it enters state 1 and never again -- one direction through the charge, the
/// whole 80-frame beam and the cooldown -- and it is the bearing to the ninja at
/// that frame, bit-exact against every captured discharge.
fn on_acquire(s: &mut ShooterState, is_laser: bool, xpos: f64, ypos: f64, nx: f64, ny: f64) {
    if !is_laser {
        return;
    }
    let (dx, dy) = (nx - xpos, ny - ypos);
    let dist = (dy * dy + dx * dx).sqrt();
    if dist.abs().max(1.0) * NORM_EPS < dist.abs() {
        s.fire_dx = dx / dist;
        s.fire_dy = dy / dist;
    }
}

/// `FUN_103FFB00`, vtable `+0x1C` for type 13. A CONTINUOUS beam, not a burst.
///
/// Every frame of the discharge it recasts along the frozen direction and runs
/// `FUN_1050B070` -- the ELEVEN-LIMB test, not the chaingun's bare circle --
/// against every ninja, then bumps `+0x8C` and reports done at `+0x88`.
fn laser_beam_tick(
    s: &mut ShooterState,
    xpos: f64,
    ypos: f64,
    segment_grid: &SegmentGrid,
    segments: &[Segment],
    ninja: &mut Ninja,
) -> bool {
    s.aim_dx = s.fire_dx;
    s.aim_dy = s.fire_dy;
    trace_and_kill(s, xpos, ypos, segment_grid, segments, ninja);
    s.shot_tick += 1;
    LASER_FIRE_TICKS <= s.shot_tick
}

/// `FUN_103FE510`, vtable `+0x18`. Freeze the discharge's aim. **No shot leaves.**
///
/// `+0x90`/`+0x98` is the unit vector to the target at this frame and
/// `+0xA0`/`+0xA8` its perpendicular, signed against the target's velocity across
/// the line of fire. Every shot of the burst is built from those two and nothing
/// else; the gun angle `+0x50` is written FROM the shot, not into it.
fn fire_setup(s: &mut ShooterState, xpos: f64, ypos: f64, tx: f64, ty: f64, vx: f64, vy: f64) {
    s.shots = 0;
    s.shot_tick = 0;
    let (mut dx, mut dy) = (tx - xpos, ty - ypos);
    let dist = (dy * dy + dx * dx).sqrt();
    if dist.abs().max(1.0) * NORM_EPS < dist.abs() {
        dx /= dist;
        dy /= dist;
        s.fire_dx = dx;
        s.fire_dy = dy;
        // `if (0.0 <= dir_x*vy - dir_y*vx) negate dir_y else negate dir_x`.
        if 0.0 <= dx * vy - dy * vx {
            s.perp_x = -dy;
            s.perp_y = dx;
        } else {
            s.perp_x = dy;
            s.perp_y = -dx;
        }
        return;
    }
    // Degenerate -- the target is on top of the drone. Fall back to its
    // velocity, then to +x, and sweep nothing.
    s.perp_x = 0.0;
    s.perp_y = 0.0;
    let vlen = (vy * vy + vx * vx).sqrt();
    if vlen.abs().max(1.0) * NORM_EPS < vlen.abs() {
        s.fire_dx = vx / vlen;
        s.fire_dy = vy / vlen;
    } else {
        s.fire_dx = 1.0;
        s.fire_dy = 0.0;
    }
}

/// `FUN_103FE790`, vtable `+0x1C`. Fire the burst; true when it is over.
///
/// Shot `n` (1-based) leaves along `normalise(base + t*perp)` with
/// `t = (n/18 - 0.5) * 0.3`, so shot 9 lies exactly on the base direction and the
/// nineteen shots bracket it eight steps before and ten after. That is a sweep in
/// DIRECTION space, not in angle -- the bearing lands `atan(t)` off the base.
///
/// The kill is `FUN_1050C690`, a bare circle test with no skeleton (§73).
fn discharge_tick(
    s: &mut ShooterState,
    xpos: f64,
    ypos: f64,
    segment_grid: &SegmentGrid,
    segments: &[Segment],
    ninja: &mut Ninja,
) -> bool {
    s.shot_tick += 1;
    if (s.shot_tick as f64) <= SHOT_PERIOD {
        return false;
    }
    let n = s.shots;
    s.shot_tick = 0;
    if MAX_SHOTS < n {
        return true;
    }
    s.shots = n + 1;
    let t = (s.shots as f64 / SWEEP_DIV - 0.5) * SWEEP_SPAN;
    let vx = t * s.perp_x + s.fire_dx;
    let vy = t * s.perp_y + s.fire_dy;
    let m = (vy * vy + vx * vx).sqrt();
    if m == 0.0 {
        return false;
    }
    s.aim_dx = vx / m;
    s.aim_dy = vy / m;
    match get_raycast_distance(segment_grid, segments, xpos, ypos, s.aim_dx, s.aim_dy) {
        Some(len) => {
            s.beam_len = Some(len);
            s.xend = xpos + s.aim_dx * len;
            s.yend = ypos + s.aim_dy * len;
        }
        None => {
            s.beam_len = None;
            s.xend = xpos;
            s.yend = ypos;
            return false;
        }
    }
    if ninja.is_valid_target()
        && crate::collision::raycast::beam_hits_ninja_opt(
            xpos, ypos, s.xend, s.yend, ninja, s.beam_len, false,
        )
    {
        debug_shot("discharge_tick", xpos, ypos, s, ninja);
        ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("drone_shot"));
    }
    false
}
