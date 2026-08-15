//! Temporary A/B switches for the Phase-A collision work.
//!
//! **This module is scaffolding and is meant to be deleted.** Each attract-gate
//! measurement costs ~13 minutes over 8000 demos, so bundling three changes into
//! one build and reading a single aggregate cannot tell you which one moved it --
//! and the first bundle we ran came back net -88. Reading the variant from the
//! environment lets several builds' worth of measurement run concurrently against
//! one `.so` instead of serially against three.
//!
//! Defaults are the faithful-to-the-binary behaviour, so an unset environment is
//! the shipping configuration and nothing here is load-bearing at runtime. Once
//! the corpus has settled each question, fold the winner in and drop the file.

use std::sync::OnceLock;

#[derive(Debug)]
pub struct AbConfig {
    /// Re-gather segments inside the depenetration loop, from the current
    /// position, the way `FUN_1050B910` does. `NCLONE_AB_NO_REGATHER=1` restores
    /// the old gather-once-before-the-loop behaviour.
    pub regather: bool,
    /// Walk gather cells y-outer/x-inner, the way `FUN_1050DDF0` does.
    /// `NCLONE_AB_COLMAJOR=1` restores the old x-outer/y-inner walk.
    pub row_major_gather: bool,
    /// `in_buffer_jump_threshold` / `_wall_` / `_floor_`. These three live in the
    /// game's settings database rather than its code, so no decompile can show
    /// them (`docs/NEXT_STEPS.md` §3.1); the port picked 5 and the launch pad's
    /// sibling bound -- the one that *is* hardcoded -- is `< 4`, which is
    /// circumstantial evidence 5 came from the same place. `NCLONE_AB_BUF_JUMP`
    /// etc. override them so the corpus can settle it. Zero disables that buffer,
    /// which stands in for `option_buffer_input` being off.
    pub buf_jump: i32,
    pub buf_wall: i32,
    pub buf_floor: i32,
    /// Take the wall normal from the FIRST contact in the game's list order
    /// (entities, then tile segments) instead of the sign of their sum.
    /// `Ninja::postCollision` (`docs/decomp/ninja_collide_10513C60.c:354-359`)
    /// copies `list[0]` and zeroes the y component; summing makes two opposite
    /// contacts cancel, which reports `walled == false` inside a one-tile shaft
    /// where the game reports true. **On by default**: it cost 140 demos until we
    /// found that a top/bottom bounce-block contact was pushing a `0.0` onto the
    /// head of the list; with that filtered it is +3 with no losses.
    /// `NCLONE_AB_WN_SUM=1` restores the old sum-and-sign.
    pub wall_normal_first: bool,
    /// With `wall_normal_first`, still reduce the chosen contact to `±1` instead
    /// of storing its actual `dx/dist`. `NCLONE_AB_WN_FIRST` changes *two* things
    /// at once -- which contact wins, and what magnitude is stored -- and the
    /// -140 demos it cost cannot be attributed without separating them. Round 6
    /// established that `+0x110` is read as a full double on the physics path, so
    /// the ~5e-13 difference between `±1` and `±0.99999999999951` is real; this
    /// flag pins the magnitude so the selection can be measured alone.
    /// `NCLONE_AB_WN_SIGN=1`.
    pub wall_normal_sign: bool,

    /// DualLaser (23) convex-corner pivot rate per sub-step. Round 4 derived
    /// `atan2(+0xA0, 24) = 0.0055247…` from the constructor chain; round 5
    /// (`docs/ENTITY_RE_ROUND5_FINDINGS.md` §5.1) says the walk is driven by
    /// `18.1 / 327.62 = 0.0552469…`, ten times larger, and calls it the
    /// highest-value item in the document. The two derivations disagree by
    /// exactly 10×, which is the signature of a misplaced decimal in one of them
    /// -- so the corpus decides rather than either read.
    /// `NCLONE_AB_DL_CORNER_SPEED=<float>`.
    pub dl_corner_speed: f64,
    /// Make `Ninja::kill` a no-op. **Diagnostic only, never a shipping mode.**
    /// It splits the corpus's losses in two: a demo that wins with hazards
    /// disarmed was on the right trajectory and lost to an over-lethal entity; a
    /// demo that still times out was off-trajectory, and no entity work will ever
    /// recover it. Those two buckets need completely different fixes and the
    /// gate cannot tell them apart on its own.
    /// `NCLONE_AB_INVULNERABLE=1`.
    pub invulnerable: bool,
    /// Boost pad (24) magnitude added to the ninja's speed on first contact, and
    /// whether the one-shot latch applies. `FUN_103F3070` computes a magnitude
    /// and calls `Ninja::setVelocity`, but Ghidra lost that call's arguments, so
    /// the `2.0` is **[I]**. Boost-pad maps are SHORTER than the corpus median
    /// (0.95x) and still lose 8.5 points with hazards disarmed, which is the one
    /// gap in the corpus that map difficulty cannot explain.
    /// `NCLONE_AB_BOOST_MAG=<float>`, `NCLONE_AB_BOOST_NOLATCH=1`.
    pub boost_mag: f64,
    pub boost_nolatch: bool,
    /// Restore the old four-way `floor_jump` tree, which discriminated on
    /// `xspeed` instead of the horizontal input and had a "perp jump" branch that
    /// zeroed `xspeed`. The binary has two rows and no such branch.
    /// `NCLONE_AB_LEGACY_FLOOR_JUMP=1`.
    pub legacy_floor_jump: bool,
    /// `DAT_10572138`, the drone direction-preference table -- 16 relative turns,
    /// four per patrol mode, tried in order by `FUN_103FA940`. It decides every
    /// drone's patrol path and ours is **[I]**, inherited from the original
    /// nclone project; only its first two entries are corroborated (see
    /// `docs/ENTITY_RE_ROUND6_REQUEST.md` §1). Override as 16 comma-separated
    /// ints to A/B a candidate table without waiting for the memory dump.
    /// `NCLONE_AB_DRONE_DIRLIST=1,0,3,2,3,0,1,2,0,1,3,2,0,3,1,2`.
    pub drone_dirlist: [[u8; 4]; 4],
    /// Seat a surface dual laser the way the game does: anchor on the closest
    /// point of the winning SEGMENT, body one radius out along that segment's
    /// outward normal. Verified against the round-9 captures -- 19 lasers across
    /// three demos land on the game's exact position, normal and anchor.
    ///
    /// **OFF by default**, and that is measured: switching it on costs the native
    /// gate 92.62% -> 90.81% and takes laser deaths 219 -> 339. Correct placement
    /// makes the beams sweep where the game's sweep, and something downstream then
    /// over-kills. It is a correct fix blocked behind a second bug.
    /// NCLONE_AB_LASER_SEAT=1.
    pub laser_seat: bool,
    /// Restore the old 0/1 edge table and its `2*outer + inner` indexing. The
    /// edge grid is shared by every entity that walks the cell grid -- floor
    /// guards, drones, thwumps -- so this one flag moves all of them at once,
    /// which is why it stays A/B-able. Read from the environment by
    /// `nclone/utils/tile_segment_factory.py` under the same name, so one
    /// variable flips both engines and they cannot drift.
    /// `NCLONE_AB_LEGACY_EDGES=1`.
    pub legacy_edges: bool,
    /// Restore round 9's three-state turret: fire, then drop straight to
    /// SEARCHING and re-latch the aim onto the muzzle. The binary has a fourth
    /// state and the game's own struct shows it (`entities/turret.rs::think`), so
    /// this is kept only so the two halves of the pair can be measured apart.
    /// `NCLONE_AB_TURRET_LEGACY=1`.
    pub turret_legacy: bool,
    /// Route the turret's shot through the 11-limb test instead of
    /// `seg_vs_circle`. Also wrong -- the limb test belongs to the laser family --
    /// but it is the other half of the pair, so it A/Bs separately.
    /// `NCLONE_AB_TURRET_LIMBS=1`.
    pub turret_limbs: bool,
    /// Use the game's real 11-limb beam test (`FUN_1050B070`) rather than the
    /// circle prefilter alone. **On by default**, and read from the environment by
    /// `nclone/physics.py` under the same name, so one variable flips both engines
    /// and they cannot drift apart. It needs the animation skeleton, so it only
    /// takes effect when `Ninja::anim_enabled` is set. Measured against captured ground
    /// truth in `docs/ENTITY_RE_ROUND8_FINDINGS.md` §6.3: the circle test alone is
    /// strictly more lethal than the game and kills demos the game wins.
    /// `NCLONE_AB_NO_LASER_LIMBS=1` restores the circle test.
    pub laser_limbs: bool,
    /// Revert the seating normal to  (63/128 vs 126/128).
    pub no_laser_winding: bool,
    /// Re-test the drone shot on every frame of the discharge, not just on the
    /// 1 -> 2 transition. The Python engine has never done this.
    pub drone_burst: bool,
    /// Restore the ctor branch that read -1 (inside edge) and no-hit as Surface.
    pub laser_empty_surface: bool,
    /// Restore the retired `mode == 1` crawl gate on surface lasers.
    pub laser_mode_gate: bool,
    /// Let a shooter drone keep patrolling through its charge/fire/cooldown.
    pub drone_patrol_always: bool,
    /// Abort a shooter drone charge when the target leaves the lane.
    pub drone_abort: bool,
    /// Restore the three state-dependent toggle mine radii.
    pub mine_radii: bool,
    /// Restore the split of a full tile face into two 12 px half-segments.
    pub half_segments: bool,
    /// Start evil ninjas armed (WAITING) instead of DORMANT.
    pub evil_ninja_armed: bool,
    /// Restore the orientation-ignoring door segment winding.
    pub door_winding: bool,
    /// Walk flush laser seating candidates horizontal-first again.
    pub laser_hfirst: bool,
    /// Restore the fall-through that launched and flew on the same frame.
    pub rocket_fast_launch: bool,
    /// Use the measured per-type shooter timings (laser 46/80/60).
    pub drone_type_timings: bool,
    /// Chase runs until blocked instead of terminating at a goal cell
    /// snapped from the ninja. `chaseend.py`: real runs end at walls.
    pub chaser_straight: bool,
    /// Detect before moving, so a latch applies on the frame it happens.
    pub chaser_early_latch: bool,
    /// A running chase can be interrupted by a fresh sighting on another
    /// axis. Without it a chase is deaf until it ends.
    pub chaser_interrupt: bool,
    /// Run the chaingun's weapon cycle as decompiled rather than as
    /// reconstructed: `vtable[+0x1C]` is the discharge, not a done-test, and
    /// the burst sweeps in direction space off an aim frozen at the transition.
    /// ON -- it takes the chaingun from 72.18% agreement with the captures on
    /// state and 49.55% on position to 99.82% / 98.19%.
    /// `NCLONE_AB_DRONE_RECONSTRUCTED=1` restores the old model. Keep in step
    /// with `entity_drone_shooter_base.py`.
    pub drone_decomp: bool,
    /// Restore the invented goal-chase model for the chaser instead of
    /// `FUN_103FF1B0` + `FUN_103FEC60`. Keep in step with
    /// `entity_drone_chaser.py`'s `NCLONE_AB_CHASER_LEGACY`.
    pub chaser_legacy: bool,
}

static AB: OnceLock<AbConfig> = OnceLock::new();

fn flag(name: &str) -> bool {
    std::env::var(name).map(|v| v == "1").unwrap_or(false)
}

fn float(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Default is the table the ports have always used.
fn dirlist(name: &str) -> [[u8; 4]; 4] {
    let mut out = [[1, 0, 3, 2], [3, 0, 1, 2], [0, 1, 3, 2], [0, 3, 1, 2]];
    if let Ok(v) = std::env::var(name) {
        let parsed: Vec<u8> = v.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if parsed.len() == 16 {
            for (i, chunk) in parsed.chunks(4).enumerate() {
                out[i].copy_from_slice(chunk);
            }
        }
    }
    out
}

fn int(name: &str, default: i32) -> i32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

pub fn ab() -> &'static AbConfig {
    AB.get_or_init(|| AbConfig {
        regather: !flag("NCLONE_AB_NO_REGATHER"),
        row_major_gather: !flag("NCLONE_AB_COLMAJOR"),
        buf_jump: int("NCLONE_AB_BUF_JUMP", 5),
        buf_wall: int("NCLONE_AB_BUF_WALL", 5),
        buf_floor: int("NCLONE_AB_BUF_FLOOR", 5),
        wall_normal_first: !flag("NCLONE_AB_WN_SUM"),
        wall_normal_sign: flag("NCLONE_AB_WN_SIGN"),
        invulnerable: flag("NCLONE_AB_INVULNERABLE"),
        boost_mag: float("NCLONE_AB_BOOST_MAG", 2.0),
        boost_nolatch: flag("NCLONE_AB_BOOST_NOLATCH"),
        legacy_floor_jump: flag("NCLONE_AB_LEGACY_FLOOR_JUMP"),
        drone_dirlist: dirlist("NCLONE_AB_DRONE_DIRLIST"),
        laser_seat: !flag("NCLONE_AB_NO_LASER_SEAT"),
        legacy_edges: flag("NCLONE_AB_LEGACY_EDGES"),
        turret_legacy: flag("NCLONE_AB_TURRET_LEGACY"),
        turret_limbs: flag("NCLONE_AB_TURRET_LIMBS"),
        dl_corner_speed: float(
            "NCLONE_AB_DL_CORNER_SPEED",
            crate::constants::DUAL_LASER_CORNER_SPEED,
        ),
        laser_limbs: !flag("NCLONE_AB_NO_LASER_LIMBS"),
        no_laser_winding: flag("NCLONE_AB_NO_LASER_WINDING"),
        drone_burst: flag("NCLONE_AB_DRONE_BURST"),
        laser_empty_surface: flag("NCLONE_AB_LASER_EMPTY_SURFACE"),
        laser_mode_gate: flag("NCLONE_AB_LASER_MODE_GATE"),
        drone_patrol_always: flag("NCLONE_AB_DRONE_PATROL_ALWAYS"),
        drone_abort: flag("NCLONE_AB_DRONE_ABORT"),
        mine_radii: flag("NCLONE_AB_MINE_RADII"),
        half_segments: flag("NCLONE_AB_HALF_SEGMENTS"),
        evil_ninja_armed: flag("NCLONE_AB_EVIL_NINJA_ARMED"),
        door_winding: flag("NCLONE_AB_DOOR_WINDING"),
        laser_hfirst: flag("NCLONE_AB_LASER_HFIRST"),
        rocket_fast_launch: flag("NCLONE_AB_ROCKET_FAST_LAUNCH"),
        drone_type_timings: flag("NCLONE_AB_DRONE_TYPE_TIMINGS"),
        chaser_straight: flag("NCLONE_AB_CHASER_STRAIGHT"),
        chaser_early_latch: flag("NCLONE_AB_CHASER_EARLY_LATCH"),
        chaser_interrupt: flag("NCLONE_AB_CHASER_INTERRUPT"),
        drone_decomp: !flag("NCLONE_AB_DRONE_RECONSTRUCTED"),
        chaser_legacy: flag("NCLONE_AB_CHASER_LEGACY"),
    })
}
