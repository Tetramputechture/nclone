//! Main Simulator: tick loop with 4 physics substeps.
//! Direct port of nclone/nsim.py.

use crate::collision::spatial::cell_from_pos;
use crate::constants::{MAP_HEIGHT, MAP_WIDTH};
use crate::entities::door_base::door_grid_edges;
use crate::entities::drone::DroneKind;
use crate::entities::{Entity, EntityCtx};
use crate::grid_edges::GridEdges;
use crate::map::loader::{self};
use crate::ninja::Ninja;
use crate::segments::Segment;
use crate::types::{EntitySnapshot, SimulatorState};

/// The main physics simulator.
pub struct Simulator {
    pub frame: u32,
    pub ninja: Ninja,

    // Tile geometry
    pub segments: Vec<Segment>,
    pub segment_grid: [[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],

    // Entities
    pub entities: Vec<Entity>,
    pub entity_grid: [[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],

    /// Indices into `entities` in the order the per-frame `move` and `think`
    /// passes must visit them: ascending entity type, then map-load order.
    ///
    /// This reproduces Python's iteration over `entity_dic`, a dict keyed
    /// `{1: [...], 2: [...], ..., 28: [...]}` (nsim.py:309). `entities` itself is
    /// in map-load order with types interleaved, which is a DIFFERENT order.
    /// It makes no observable difference yet -- toggle mines are the only
    /// thinkers and they do not interact -- but it does as soon as entities
    /// affect each other or the ninja within a pass (death-ball chain bounces,
    /// several bounce blocks accumulating force on the ninja). Built once at
    /// load; entity indices are stable for the lifetime of a Simulator.
    pub entity_order: Vec<usize>,

    /// `entity_order` restricted to the types that take part in the `move` pass.
    /// `Entity::is_movable` keys off the variant alone, so this set is fixed for
    /// the lifetime of the sim; only `active()` varies per frame. Precomputed so
    /// the move pass touches the movers instead of sweeping every entity.
    move_candidates: Vec<usize>,

    /// `entity_order` restricted to the types that can take part in the `think`
    /// pass, pre-sorted by `think_order_rank`. The rank depends only on the
    /// entity type, and a stable sort of the full list filtered afterwards is the
    /// same sequence as filtering first and then stably sorting -- which is what
    /// the pass used to do, per frame. `Entity::is_thinkable` is still evaluated
    /// per frame (the drone arm reads its behaviour), so this is a superset.
    think_candidates: Vec<usize>,

    /// Indices of the door entities (`door_barrier` returns `Some`). The barrier
    /// sync visits these instead of every entity.
    door_indices: Vec<usize>,

    /// Scratch lists for the per-frame move/think passes, kept across frames so
    /// the passes do not allocate. Emptied at the start of every use.
    movable_buf: Vec<usize>,
    thinkable_buf: Vec<usize>,

    /// Scratch lists for `post_collision_with_entities`, kept for the same
    /// reason: both are rebuilt from scratch on every frame.
    scratch_entity_indices: Vec<usize>,
    scratch_collected_doors: Vec<usize>,

    /// Death-ball count from the map header (`map_data[1200]`), which gates the
    /// ball-vs-ball repulsion. Read from the header rather than counted so a
    /// header/entity mismatch behaves the way Python's does.
    pub death_ball_count: usize,

    /// True iff any loaded entity is physically collidable (one-way platform).
    /// When false, `collide_vs_objects` is a no-op, so `pre_collision` skips its
    /// per-frame entity gather. Computed once at load.
    pub has_physical_entities: bool,

    // Tile data for Python access
    pub tile_grid: [[u8; MAP_HEIGHT]; MAP_WIDTH],

    /// Half-cell edge occupancy (static tile layer + dynamic door counters).
    /// Grid-walking entities test this instead of colliding against segments.
    pub grid_edges: GridEdges,

    /// Per-ninja position recorder that the evil ninja (type 22) replays.
    ///
    /// The game keeps 8 ninjas x 18000 records x 0x78 bytes and wraps the write
    /// index at 17999 (`docs/ENTITY_INTEGRATION_SPEC.md` §8.10); only the position
    /// is ever read back, so only the position is stored. Allocated ONLY on maps
    /// that carry an evil ninja -- about 3% of official maps -- because recording
    /// is a write on every tick. Empty means "no recorder", which is the binary's
    /// "buffer gone" branch and makes the entity reset itself.
    pub ninja_tape: Vec<(f64, f64)>,
    pub ninja_tape_index: usize,

    /// Broadphase mask for `kill_on_deadly_mine_contact`: `deadly_mine_zone[cx][cy]`
    /// is true iff a currently-deadly toggle mine (state 0) sits in the 3x3 cell
    /// neighbourhood of cell `(cx, cy)` — i.e. iff a ninja in that cell could
    /// gather a deadly mine.  Computed once at load.  Valid only because the sole
    /// caller (`tick_physics_mines_only`) never mutates mine state, so the
    /// load-time deadly set is constant across an enumeration.
    pub deadly_mine_zone: [[bool; MAP_HEIGHT]; MAP_WIDTH],

    /// Enumeration kernel fast-paths (open-air collision skip + deadly-mine
    /// broadphase).  Disabled when the env var `NCLONE_KERNEL_BASELINE` is set,
    /// which forces the original full-collision behaviour — used to prove the
    /// fast paths are byte-identical via a same-binary A/B.  Read once at load.
    pub kernel_opt: bool,

    // --- Experimental: reachability-flood sizing (off in production) ---
    /// When `Some`, every arc tick records a quantized `(pre-state, input)` key
    /// here.  Counting distinct keys vs `flood_ticks` measures trajectory-
    /// transition redundancy = the transposition-merge speedup ceiling.  Built
    /// only by the `flood_stats` measurement path; `None` (and thus free) in the
    /// production enumeration path.
    pub flood_keys: Option<std::collections::HashSet<u64>>,
    pub flood_ticks: u64,
    pub flood_qpos: f64,
    pub flood_qvel: f64,
    /// Same as `flood_keys`/`flood_ticks` but counts only *collision-relevant*
    /// ticks (a tile segment is within reach this frame).  After K1 the open-air
    /// ticks are already cheap, so this — redundancy among the *expensive* ticks
    /// — is what actually bounds a flood/collision-cache speedup.
    pub flood_keys_coll: Option<std::collections::HashSet<u64>>,
    pub flood_ticks_coll: u64,

    /// Experimental: collision-merge accuracy study.  When `snap_qpos > 0`, after
    /// any tick that did real collision work (floor/wall/ceiling contact) the
    /// ninja pos/vel are snapped to a `(snap_qpos, snap_qvel)` grid.  This forces
    /// the same grain-G equivalence classes a collision cache would create, so
    /// the resulting d* shows the merge's accuracy cost.  A proxy for the cache
    /// (it snaps the post-state rather than applying a cached response), and
    /// conservative on position.  Off (0.0) in production.
    pub snap_qpos: f64,
    pub snap_qvel: f64,
}

/// Projectile blob shared by the rocket (18) and the player-controlled rocket
/// (29): 7 f64 in a fixed order. Both types own their projectile inline rather
/// than spawning an entity (spec §8.8), so it round-trips as part of the launcher.
fn write_projectile(out: &mut Vec<u8>, proj: &crate::entities::rocket::Projectile) {
    for v in [
        proj.xpos, proj.ypos, proj.xdir, proj.ydir, proj.speed, proj.step, proj.travel,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Byte length of the blob `write_projectile` emits.
const PROJECTILE_BYTES: usize = 7 * 8;

/// Position of an entity type in the game's think pass (spec §4). Types absent
/// from the pass never reach here; they sort last, which is harmless because they
/// are filtered out before the sort.
fn think_order_rank(type_id: u8) -> usize {
    const THINK_ORDER: [u8; 17] = [5, 12, 13, 15, 16, 17, 18, 19, 20, 1, 21, 22, 23, 25, 27, 28, 29];
    THINK_ORDER
        .iter()
        .position(|&t| t == type_id)
        .unwrap_or(THINK_ORDER.len())
}

impl Simulator {
    /// Load a map from integer map data.
    pub fn load(map_data: &[i32]) -> Self {
        let data = loader::load_map(map_data);
        let mut ninja = Ninja::new(data.ninja_spawn_x, data.ninja_spawn_y);

        // The animation skeleton is only ever READ by the hitscan beam test, so
        // it is maintained only on maps that have something able to fire one:
        // the turret (19), the dual laser (23) and the two shooter drones
        // (12, 13). Every other map gets the frames back. This does not make the
        // two engines disagree -- `bones` feeds nothing else -- and gating it on
        // the map rather than on a flag means no training configuration can
        // accidentally run the more-lethal circle test.
        if data
            .entities
            .iter()
            .any(|e| matches!(e.entity_type_id(), 12 | 13 | 19 | 23))
        {
            ninja.enable_anim();
        }

        let has_physical_entities = data.entities.iter().any(|e| e.is_physical_collidable());
        let needs_ninja_tape = data
            .entities
            .iter()
            .any(|e| matches!(e, Entity::EvilNinja(_)));

        // Ascending entity type, ties broken by load order (a stable sort over
        // the already load-ordered index list gives exactly that).
        let mut entity_order: Vec<usize> = (0..data.entities.len()).collect();
        entity_order.sort_by_key(|&i| data.entities[i].entity_type_id());

        // Static slices of `entity_order` for the two per-frame passes. See the
        // field docs: membership is decided by the entity's variant, which never
        // changes, so the per-frame work reduces to the `active()`/`is_thinkable()`
        // checks that were already there.
        let move_candidates: Vec<usize> = entity_order
            .iter()
            .copied()
            .filter(|&i| data.entities[i].is_movable())
            .collect();
        let mut think_candidates: Vec<usize> = entity_order
            .iter()
            .copied()
            .filter(|&i| data.entities[i].can_ever_think())
            .collect();
        think_candidates.sort_by_key(|&i| think_order_rank(data.entities[i].entity_type_id()));
        let door_indices: Vec<usize> = (0..data.entities.len())
            .filter(|&i| door_barrier(&data.entities[i]).is_some())
            .collect();
        let n_move = move_candidates.len();
        let n_think = think_candidates.len();

        let mut sim = Simulator {
            frame: 0,
            ninja,
            segments: data.segments,
            segment_grid: data.segment_grid,
            entities: data.entities,
            entity_grid: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
            entity_order,
            move_candidates,
            think_candidates,
            door_indices,
            movable_buf: Vec::with_capacity(n_move),
            thinkable_buf: Vec::with_capacity(n_think),
            scratch_entity_indices: Vec::new(),
            scratch_collected_doors: Vec::new(),
            death_ball_count: if map_data.len() > 1200 {
                map_data[1200].max(0) as usize
            } else {
                0
            },
            has_physical_entities,
            tile_grid: data.tile_grid,
            grid_edges: data.grid_edges,
            // Allocated only when something replays it; see the field docs.
            ninja_tape: if needs_ninja_tape {
                vec![(0.0, 0.0); crate::constants::NINJA_TAPE_LENGTH]
            } else {
                Vec::new()
            },
            ninja_tape_index: 0,
            deadly_mine_zone: [[false; MAP_HEIGHT]; MAP_WIDTH],
            kernel_opt: std::env::var_os("NCLONE_KERNEL_BASELINE").is_none(),
            flood_keys: None,
            flood_ticks: 0,
            flood_qpos: 0.0,
            flood_qvel: 0.0,
            flood_keys_coll: None,
            flood_ticks_coll: 0,
            snap_qpos: 0.0,
            snap_qvel: 0.0,
        };

        // Populate entity grid (skip exit doors -- they're only added when switch is collected)
        sim.rebuild_entity_grid();
        sim.resync_doors();
        sim.compute_deadly_mine_zone();

        sim
    }

    /// Bring both of a door's blocking layers back in line with its `closed`
    /// flag: the collision segment (blocks the ninja) and the half-cell grid
    /// edges (block grid-walking entities). Entities can reach neither, so this
    /// is simulator-side.
    ///
    /// Called after load / `fast_reset` / `restore_full_state` -- anywhere door
    /// state is installed wholesale rather than flipped by a switch. Doors
    /// opened mid-episode adjust both layers incrementally instead (see
    /// `post_collision_with_entities`).
    ///
    /// `restore_full_state` previously resynced neither, so a Go-Explore
    /// checkpoint captured after a door opened would restore with the door's
    /// segment still solid.
    pub fn resync_doors(&mut self) {
        self.grid_edges.clear_door_edges();
        for i in 0..self.entities.len() {
            let Some((seg_idx, dx, dy, orientation, closed)) = door_barrier(&self.entities[i])
            else {
                continue;
            };
            if let Some(seg) = self.segments.get_mut(seg_idx) {
                match seg {
                    Segment::Linear(s) => s.active = closed,
                    Segment::Circular(s) => s.active = closed,
                }
            }
            if closed {
                for (vertical, ex, ey) in door_grid_edges(dx, dy, orientation) {
                    self.grid_edges.adjust_door_edge(vertical, ex, ey, 1);
                }
            }
            set_door_registered(&mut self.entities[i], closed);
        }
    }

    /// Reconcile every door's two blocking layers with its `closed` flag,
    /// applying only the delta for doors that changed since the last call.
    ///
    /// Doors set `closed` from inside `think` / `logical_collision` but cannot
    /// reach `self.segments` or `self.grid_edges`, so the flip happens here. This
    /// must run at BOTH points Python's `change_state` could fire:
    ///   * after the think pass, since a regular door auto-closing has to be
    ///     solid before the ninja's collision substeps run this frame;
    ///   * after the post-collision entity loop, for switch-driven opens/closes.
    ///
    /// O(number of doors), which is a handful even on dense maps -- and regular
    /// doors genuinely toggle every few frames, so an incremental delta beats
    /// rebuilding the whole edge layer.
    fn sync_door_barriers(&mut self) {
        for k in 0..self.door_indices.len() {
            let i = self.door_indices[k];
            let Some((seg_idx, dx, dy, orientation, closed)) = door_barrier(&self.entities[i])
            else {
                continue;
            };
            if door_registered(&self.entities[i]) == closed {
                continue;
            }
            if let Some(seg) = self.segments.get_mut(seg_idx) {
                match seg {
                    Segment::Linear(s) => s.active = closed,
                    Segment::Circular(s) => s.active = closed,
                }
            }
            let delta = if closed { 1 } else { -1 };
            for (vertical, ex, ey) in door_grid_edges(dx, dy, orientation) {
                self.grid_edges.adjust_door_edge(vertical, ex, ey, delta);
            }
            set_door_registered(&mut self.entities[i], closed);
        }
    }

    /// Recompute `deadly_mine_zone` from the current entity set: mark the 3x3 cell
    /// neighbourhood of every currently-deadly toggle mine (state 0).  The 3x3
    /// dilation mirrors `gather_entities_from_neighbourhood`, so a ninja whose
    /// cell is unset cannot possibly gather a deadly mine — letting
    /// `kill_on_deadly_mine_contact` skip its gather + overlap loop on the vast
    /// majority of frames on mine-bearing maps.
    pub fn compute_deadly_mine_zone(&mut self) {
        for col in self.deadly_mine_zone.iter_mut() {
            for c in col.iter_mut() {
                *c = false;
            }
        }
        for entity in &self.entities {
            if let Entity::ToggleMine(mine) = entity {
                if mine.state != 0 {
                    continue;
                }
                let (mx, my) = entity.cell();
                let x0 = mx.saturating_sub(1);
                let x1 = (mx + 1).min(MAP_WIDTH - 1);
                let y0 = my.saturating_sub(1);
                let y1 = (my + 1).min(MAP_HEIGHT - 1);
                for zx in x0..=x1 {
                    for zy in y0..=y1 {
                        self.deadly_mine_zone[zx][zy] = true;
                    }
                }
            }
        }
    }

    /// Broadphase guard for the per-frame tile-collision substeps.  Returns true
    /// iff the swept region covered by this frame's motion (both
    /// `sweep_circle_vs_tiles` at `NINJA_RADIUS*0.5+1` and the depenetration
    /// gather at `NINJA_RADIUS`) could contain an active tile segment.  When
    /// false, every `collide_vs_tiles` substep is a provable no-op (position,
    /// velocity and collision accumulators are all left untouched), so the whole
    /// `0..4` substep block can be skipped — byte-identically.  Requires
    /// `xpos_old`/`ypos_old` to hold the pre-`integrate` position (true between
    /// `integrate` and the substep loop).
    #[inline]
    fn frame_has_tile_segments(&self) -> bool {
        if !self.kernel_opt {
            return true; // baseline: always run the full 4-substep collision
        }
        let n = &self.ninja;
        let pad = crate::constants::NINJA_RADIUS + 1.0;
        crate::collision::spatial::region_has_segments(
            &self.segment_grid,
            &self.segments,
            n.xpos.min(n.xpos_old) - pad,
            n.ypos.min(n.ypos_old) - pad,
            n.xpos.max(n.xpos_old) + pad,
            n.ypos.max(n.ypos_old) + pad,
        )
    }

    /// Rebuild entity grid from current entity positions.
    pub fn rebuild_entity_grid(&mut self) {
        for x in 0..MAP_WIDTH {
            for y in 0..MAP_HEIGHT {
                self.entity_grid[x][y].clear();
            }
        }
        for (idx, entity) in self.entities.iter().enumerate() {
            // Skip exit doors -- only added after switch collection
            if matches!(entity, Entity::ExitDoor(d) if !d.switch_hit) {
                continue;
            }
            if entity.active() || entity.is_logical_collidable() {
                let (cx, cy) = entity.cell();
                self.entity_grid[cx][cy].push(idx);
            }
        }
    }


    /// replay-free Go-Explore checkpoint restore. Layout:
    ///   [frame: u32][ninja_len: u32][ninja bytes][n_entities: u32]
    ///   [per entity: tag u8 + mutable fields]
    /// The ninja portion uses `Ninja::to_bytes` which is exhaustive (all
    /// buffers: jump/floor/wall/launch_pad; all booleans: airborn/airborn_old/
    /// walled/is_jump_useless/last_jump_was_buffered/is_crushable/terminal_impact;
    /// normals, counts, crush, inputs, death state, gold/doors, temporal). The
    /// entity portion captures every per-entity mutable field.
    /// `restore_full_state` must run on a Simulator with the SAME map loaded.
    pub fn save_full_state(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1024);
        out.extend_from_slice(&self.frame.to_le_bytes());
        let nb = self.ninja.to_bytes();
        out.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        out.extend_from_slice(&nb);
        out.extend_from_slice(&(self.entities.len() as u32).to_le_bytes());
        for entity in &self.entities {
            match entity {
                Entity::ToggleMine(e) => {
                    // MUST be the entity type id, not a literal 1. A toggle mine
                    // spawned ARMED reports type 21, and `restore_full_state`
                    // validates `tag == entity_type_id()`. Writing 1 here made
                    // every snapshot of a level containing an armed mine fail
                    // validation, so restore returned false and silently mutated
                    // NOTHING -- turning the incremental save/restore search into
                    // a single drifting trajectory. Official levels are ~20% armed
                    // mines, so this silently broke the fast planner on most of
                    // the real corpus.
                    out.push(e.entity_type);
                    out.push(e.active as u8);
                    out.push(e.state);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                    out.extend_from_slice(&e.radius.to_le_bytes());
                }
                Entity::Gold(e) => {
                    out.push(2);
                    out.push(e.active as u8);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                }
                Entity::ExitSwitch(e) => {
                    out.push(4);
                    out.push(e.active as u8);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                }
                Entity::ExitDoor(e) => {
                    out.push(3);
                    out.push(e.switch_hit as u8);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                }
                Entity::LockedDoor(e) => {
                    out.push(6);
                    out.push(e.active as u8);
                    out.push(e.closed as u8);
                    out.push(e.orientation);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                }
                Entity::OneWay(e) => {
                    // Static: position/orientation never mutate. Round-trip (x, y)
                    // only so the blob stays self-describing and length-validated.
                    out.push(11);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                }
                Entity::LaunchPad(e) => {
                    // Static, like the one-way.
                    out.push(10);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                }
                Entity::Drone(e) => {
                    // Everything the patrol depends on: position, current target
                    // cell centre, facing, and the (stale-by-design) grid cell.
                    out.push(e.kind.type_id());
                    out.push(e.active as u8);
                    out.push(e.dir);
                    out.push(e.dir_old);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                    out.extend_from_slice(&e.xtarget.to_le_bytes());
                    out.extend_from_slice(&e.ytarget.to_le_bytes());
                    out.extend_from_slice(&(e.cell.0 as u32).to_le_bytes());
                    out.extend_from_slice(&(e.cell.1 as u32).to_le_bytes());
                }
                Entity::DeathBall(e) => {
                    out.push(25);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                    out.extend_from_slice(&e.xspeed.to_le_bytes());
                    out.extend_from_slice(&e.yspeed.to_le_bytes());
                    out.extend_from_slice(&(e.cell.0 as u32).to_le_bytes());
                    out.extend_from_slice(&(e.cell.1 as u32).to_le_bytes());
                }
                Entity::SwitchDoor(e) => {
                    out.push(e.type_id());
                    out.push(e.closed as u8);
                    out.push(e.active as u8);
                    out.push(e.registered_closed as u8);
                    out.extend_from_slice(&e.open_timer.to_le_bytes());
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                }
                Entity::BounceBlock(e) => {
                    out.push(17);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                    out.extend_from_slice(&e.xspeed.to_le_bytes());
                    out.extend_from_slice(&e.yspeed.to_le_bytes());
                    out.extend_from_slice(&(e.cell.0 as u32).to_le_bytes());
                    out.extend_from_slice(&(e.cell.1 as u32).to_le_bytes());
                }
                Entity::Thwump(e) => {
                    out.push(20);
                    out.push(e.state as u8); // signed -1/0/1, round-tripped as raw bits
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                    out.extend_from_slice(&(e.cell.0 as u32).to_le_bytes());
                    out.extend_from_slice(&(e.cell.1 as u32).to_le_bytes());
                }
                Entity::ShoveThwump(e) => {
                    out.push(28);
                    out.push(e.state);
                    out.push(e.activated as u8);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                    out.extend_from_slice(&e.xdir.to_le_bytes());
                    out.extend_from_slice(&e.ydir.to_le_bytes());
                    out.extend_from_slice(&(e.cell.0 as u32).to_le_bytes());
                    out.extend_from_slice(&(e.cell.1 as u32).to_le_bytes());
                }
                Entity::FloorGuard(e) => {
                    out.push(16);
                    out.extend_from_slice(&e.dir.to_le_bytes());
                    out.push(e.just_activated as u8);
                    out.extend_from_slice(&e.step2.to_le_bytes());
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                    out.extend_from_slice(&(e.cell.0 as u32).to_le_bytes());
                    out.extend_from_slice(&(e.cell.1 as u32).to_le_bytes());
                }
                Entity::EyeBat(e) => {
                    out.push(27);
                    out.push(e.chasing as u8);
                    out.extend_from_slice(&e.chase_timer.to_le_bytes());
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                    out.extend_from_slice(&e.xspeed.to_le_bytes());
                    out.extend_from_slice(&e.yspeed.to_le_bytes());
                    out.extend_from_slice(&(e.cell.0 as u32).to_le_bytes());
                    out.extend_from_slice(&(e.cell.1 as u32).to_le_bytes());
                }
                Entity::Rocket(e) => {
                    out.push(18);
                    out.push(e.state);
                    out.push(e.prev_state);
                    out.extend_from_slice(&e.charge.to_le_bytes());
                    write_projectile(&mut out, &e.projectile);
                }
                Entity::Turret(e) => {
                    out.push(19);
                    out.push(e.state);
                    out.push(e.has_target as u8);
                    out.extend_from_slice(&(e.band as u32).to_le_bytes());
                    out.extend_from_slice(&e.timer.to_le_bytes());
                    out.extend_from_slice(&e.aim_x.to_le_bytes());
                    out.extend_from_slice(&e.aim_y.to_le_bytes());
                    out.extend_from_slice(&e.charge.to_le_bytes());
                    out.extend_from_slice(&e.xend.to_le_bytes());
                    out.extend_from_slice(&e.yend.to_le_bytes());
                }
                Entity::DualLaser(e) => {
                    // `mode` is decided at construction and never changes, so it
                    // is map state, not episode state, and is not round-tripped.
                    out.push(23);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                    out.extend_from_slice(&e.angle.to_le_bytes());
                    out.extend_from_slice(&e.xnormal.to_le_bytes());
                    out.extend_from_slice(&e.ynormal.to_le_bytes());
                    out.extend_from_slice(&e.xend.to_le_bytes());
                    out.extend_from_slice(&e.yend.to_le_bytes());
                    out.extend_from_slice(&(e.cell.0 as u32).to_le_bytes());
                    out.extend_from_slice(&(e.cell.1 as u32).to_le_bytes());
                }
                Entity::EvilNinja(e) => {
                    out.push(22);
                    out.push(e.state);
                    out.extend_from_slice(&e.timer.to_le_bytes());
                    out.extend_from_slice(&(e.cursor as u32).to_le_bytes());
                    out.extend_from_slice(&e.post_timer.to_le_bytes());
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                    out.extend_from_slice(&e.radius.to_le_bytes());
                    out.extend_from_slice(&e.fade.to_le_bytes());
                    out.extend_from_slice(&(e.cell.0 as u32).to_le_bytes());
                    out.extend_from_slice(&(e.cell.1 as u32).to_le_bytes());
                }
                Entity::PcRocket(e) => {
                    out.push(29);
                    out.push(e.state);
                    out.push(e.controlled as u8);
                    out.push(e.active as u8);
                    out.extend_from_slice(&e.controller.to_le_bytes());
                    write_projectile(&mut out, &e.projectile);
                }
                Entity::BoostPad(e) => {
                    // The one-shot latch IS mutable state: restoring it wrong
                    // either double-boosts or swallows a boost.
                    out.push(24);
                    out.push(e.is_touching_ninja as u8);
                    out.extend_from_slice(&e.xpos.to_le_bytes());
                    out.extend_from_slice(&e.ypos.to_le_bytes());
                }
            }
        }
        out
    }

    /// Restore complete state captured by `save_full_state` (no action replay).
    /// Rebuilds the entity grid afterwards so collision queries reflect the
    /// restored entity states (e.g. an opened exit door re-enters the grid).
    /// Restore complete state from a `save_full_state` blob. Returns `true` on
    /// success, `false` if the blob is INCOMPATIBLE with the currently-loaded
    /// level (different entity count / types, or truncated). On failure NOTHING
    /// is mutated, so the caller can fall back to action replay. This guards
    /// against a Go-Explore checkpoint captured on one level being restored onto
    /// a different level (e.g. 3-entity vs 2-entity), which previously panicked
    /// with an out-of-bounds index and killed the worker.
    pub fn restore_full_state(&mut self, data: &[u8]) -> bool {
        // Expected per-entity field byte count (after the 1-byte tag), and the
        // tag each current entity variant must match.
        fn field_len(tag: u8) -> Option<usize> {
            match tag {
                // ToggleMine: active,state,x,y,radius. Tag 1 = spawned safe,
                // tag 21 = spawned armed (TOGGLE_MINE_TOGGLED); same payload,
                // and both must be accepted or armed-mine levels never restore.
                1 | 21 => Some(1 + 1 + 8 + 8 + 8),
                2 => Some(1 + 8 + 8),         // Gold: active,x,y
                4 => Some(1 + 8 + 8),         // ExitSwitch: active,x,y
                3 => Some(1 + 8 + 8),         // ExitDoor: switch_hit,x,y
                6 => Some(1 + 1 + 1 + 8 + 8), // LockedDoor: active,closed,orient,x,y
                11 => Some(8 + 8),            // OneWay: x,y
                10 => Some(8 + 8),            // LaunchPad: x,y
                24 => Some(1 + 8 + 8),        // BoostPad: is_touching_ninja,x,y
                // Drone: active,dir,dir_old,x,y,xtarget,ytarget,cell.0,cell.1
                14 | 26 => Some(1 + 1 + 1 + 8 + 8 + 8 + 8 + 4 + 4),
                // SwitchDoor: closed,active,registered_closed,open_timer,x,y
                5 | 8 => Some(1 + 1 + 1 + 4 + 8 + 8),
                25 => Some(8 + 8 + 8 + 8 + 4 + 4), // DeathBall: x,y,xspeed,yspeed,cell
                17 => Some(8 + 8 + 8 + 8 + 4 + 4), // BounceBlock: x,y,xspeed,yspeed,cell
                20 => Some(1 + 8 + 8 + 4 + 4),     // Thwump: state,x,y,cell
                // ShoveThwump: state,activated,x,y,xdir,ydir,cell
                28 => Some(1 + 1 + 8 + 8 + 8 + 8 + 4 + 4),
                // FloorGuard: dir,just_activated,step2,x,y,cell
                16 => Some(4 + 1 + 8 + 8 + 8 + 4 + 4),
                // EyeBat: chasing,chase_timer,x,y,xspeed,yspeed,cell
                27 => Some(1 + 4 + 8 + 8 + 8 + 8 + 4 + 4),
                // Rocket: state,prev_state,projectile
                18 => Some(1 + 1 + 8 + PROJECTILE_BYTES),
                // Turret: state,has_target,band,timer,aim,charge,end
                19 => Some(1 + 1 + 4 + 4 + 8 + 8 + 8 + 8 + 8),
                // DualLaser: x,y,angle,normal,end,cell
                23 => Some(8 + 8 + 8 + 8 + 8 + 8 + 8 + 4 + 4),
                // EvilNinja: state,timer,cursor,post_timer,x,y,radius,fade,cell
                22 => Some(1 + 4 + 4 + 4 + 8 + 8 + 8 + 8 + 4 + 4),
                // PcRocket: state,controlled,active,controller,projectile
                29 => Some(1 + 1 + 1 + 4 + PROJECTILE_BYTES),
                _ => None,
            }
        }
        fn expected_tag(e: &Entity) -> u8 {
            // The blob tag IS the entity type id, so this is just the accessor --
            // keeping a second exhaustive match here meant every new entity type
            // had to be added in two places, and forgetting one only showed up as
            // a silently-rejected checkpoint.
            e.entity_type_id()
        }

        // ---- Phase 1: validate WITHOUT mutating ----
        let mut p = 0usize;
        if data.len() < 8 {
            return false;
        }
        p += 4; // frame
        let nlen = u32::from_le_bytes(data[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        if p + nlen + 4 > data.len() {
            return false;
        }
        p += nlen; // ninja bytes
        let n_ent = u32::from_le_bytes(data[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        if n_ent != self.entities.len() {
            return false; // cross-level snapshot — entity count differs
        }
        for i in 0..n_ent {
            if p + 1 > data.len() {
                return false;
            }
            let tag = data[p];
            p += 1;
            if tag != expected_tag(&self.entities[i]) {
                return false; // entity layout differs (different level)
            }
            match field_len(tag) {
                Some(fl) if p + fl <= data.len() => p += fl,
                _ => return false, // unknown tag or truncated
            }
        }

        // ---- Phase 2: apply (validated -> all reads are in bounds) ----
        let mut p = 0usize;
        let read_u32 = |d: &[u8], p: &mut usize| -> u32 {
            let v = u32::from_le_bytes(d[*p..*p + 4].try_into().unwrap());
            *p += 4;
            v
        };
        let read_f64 = |d: &[u8], p: &mut usize| -> f64 {
            let v = f64::from_le_bytes(d[*p..*p + 8].try_into().unwrap());
            *p += 8;
            v
        };
        let read_u8 = |d: &[u8], p: &mut usize| -> u8 {
            let v = d[*p];
            *p += 1;
            v
        };

        let read_i32 = |d: &[u8], p: &mut usize| -> i32 {
            let v = i32::from_le_bytes([d[*p], d[*p + 1], d[*p + 2], d[*p + 3]]);
            *p += 4;
            v
        };
        let read_projectile =
            |d: &[u8], p: &mut usize, proj: &mut crate::entities::rocket::Projectile| {
                proj.xpos = read_f64(d, p);
                proj.ypos = read_f64(d, p);
                proj.xdir = read_f64(d, p);
                proj.ydir = read_f64(d, p);
                proj.speed = read_f64(d, p);
                proj.step = read_f64(d, p);
                proj.travel = read_f64(d, p);
            };

        self.frame = read_u32(data, &mut p);
        let nlen = read_u32(data, &mut p) as usize;
        self.ninja = crate::ninja::Ninja::from_bytes(&data[p..p + nlen]);
        p += nlen;
        let n_ent = read_u32(data, &mut p) as usize;
        for i in 0..n_ent {
            let _tag = read_u8(data, &mut p);
            match &mut self.entities[i] {
                Entity::ToggleMine(e) => {
                    e.active = read_u8(data, &mut p) != 0;
                    e.state = read_u8(data, &mut p);
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                    e.radius = read_f64(data, &mut p);
                }
                Entity::Gold(e) => {
                    e.active = read_u8(data, &mut p) != 0;
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                }
                Entity::ExitSwitch(e) => {
                    e.active = read_u8(data, &mut p) != 0;
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                }
                Entity::ExitDoor(e) => {
                    e.switch_hit = read_u8(data, &mut p) != 0;
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                }
                Entity::LockedDoor(e) => {
                    e.active = read_u8(data, &mut p) != 0;
                    e.closed = read_u8(data, &mut p) != 0;
                    e.orientation = read_u8(data, &mut p);
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                }
                Entity::OneWay(e) => {
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                }
                Entity::LaunchPad(e) => {
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                }
                Entity::Drone(e) => {
                    e.active = read_u8(data, &mut p) != 0;
                    e.dir = read_u8(data, &mut p);
                    e.dir_old = read_u8(data, &mut p);
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                    e.xtarget = read_f64(data, &mut p);
                    e.ytarget = read_f64(data, &mut p);
                    e.cell = (read_u32(data, &mut p) as usize, read_u32(data, &mut p) as usize);
                }
                Entity::DeathBall(e) => {
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                    e.xspeed = read_f64(data, &mut p);
                    e.yspeed = read_f64(data, &mut p);
                    e.cell = (read_u32(data, &mut p) as usize, read_u32(data, &mut p) as usize);
                }
                Entity::SwitchDoor(e) => {
                    e.closed = read_u8(data, &mut p) != 0;
                    e.active = read_u8(data, &mut p) != 0;
                    e.registered_closed = read_u8(data, &mut p) != 0;
                    e.open_timer = read_u32(data, &mut p);
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                }
                Entity::BounceBlock(e) => {
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                    e.xspeed = read_f64(data, &mut p);
                    e.yspeed = read_f64(data, &mut p);
                    e.cell = (read_u32(data, &mut p) as usize, read_u32(data, &mut p) as usize);
                }
                Entity::Thwump(e) => {
                    e.state = read_u8(data, &mut p) as i8;
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                    e.cell = (read_u32(data, &mut p) as usize, read_u32(data, &mut p) as usize);
                }
                Entity::ShoveThwump(e) => {
                    e.state = read_u8(data, &mut p);
                    e.activated = read_u8(data, &mut p) != 0;
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                    e.xdir = read_f64(data, &mut p);
                    e.ydir = read_f64(data, &mut p);
                    e.cell = (read_u32(data, &mut p) as usize, read_u32(data, &mut p) as usize);
                }
                Entity::FloorGuard(e) => {
                    e.dir = read_i32(data, &mut p);
                    e.just_activated = read_u8(data, &mut p) != 0;
                    e.step2 = read_f64(data, &mut p);
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                    e.cell = (read_u32(data, &mut p) as usize, read_u32(data, &mut p) as usize);
                }
                Entity::EyeBat(e) => {
                    e.chasing = read_u8(data, &mut p) != 0;
                    e.chase_timer = read_i32(data, &mut p);
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                    e.xspeed = read_f64(data, &mut p);
                    e.yspeed = read_f64(data, &mut p);
                    e.cell = (read_u32(data, &mut p) as usize, read_u32(data, &mut p) as usize);
                }
                Entity::Rocket(e) => {
                    e.state = read_u8(data, &mut p);
                    e.prev_state = read_u8(data, &mut p);
                    e.charge = read_f64(data, &mut p);
                    read_projectile(data, &mut p, &mut e.projectile);
                }
                Entity::Turret(e) => {
                    e.state = read_u8(data, &mut p);
                    e.has_target = read_u8(data, &mut p) != 0;
                    e.band = read_u32(data, &mut p) as usize;
                    e.timer = read_i32(data, &mut p);
                    e.aim_x = read_f64(data, &mut p);
                    e.aim_y = read_f64(data, &mut p);
                    e.charge = read_f64(data, &mut p);
                    e.xend = read_f64(data, &mut p);
                    e.yend = read_f64(data, &mut p);
                }
                Entity::DualLaser(e) => {
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                    e.angle = read_f64(data, &mut p);
                    e.xnormal = read_f64(data, &mut p);
                    e.ynormal = read_f64(data, &mut p);
                    e.xend = read_f64(data, &mut p);
                    e.yend = read_f64(data, &mut p);
                    e.cell = (read_u32(data, &mut p) as usize, read_u32(data, &mut p) as usize);
                }
                Entity::EvilNinja(e) => {
                    e.state = read_u8(data, &mut p);
                    e.timer = read_i32(data, &mut p);
                    e.cursor = read_u32(data, &mut p) as usize;
                    e.post_timer = read_i32(data, &mut p);
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                    e.radius = read_f64(data, &mut p);
                    e.fade = read_f64(data, &mut p);
                    e.cell = (read_u32(data, &mut p) as usize, read_u32(data, &mut p) as usize);
                }
                Entity::PcRocket(e) => {
                    e.state = read_u8(data, &mut p);
                    e.controlled = read_u8(data, &mut p) != 0;
                    e.active = read_u8(data, &mut p) != 0;
                    e.controller = read_i32(data, &mut p);
                    read_projectile(data, &mut p, &mut e.projectile);
                }
                Entity::BoostPad(e) => {
                    e.is_touching_ninja = read_u8(data, &mut p) != 0;
                    e.xpos = read_f64(data, &mut p);
                    e.ypos = read_f64(data, &mut p);
                }
            }
        }
        self.rebuild_entity_grid();
        // The restored `closed` flags decide which door edges are live; resync
        // rather than trusting whatever the counters held before the restore.
        self.resync_doors();
        true
    }

    /// Execute one full physics frame.
    pub fn tick(&mut self, hor_input: i32, jump_input: i32) {
        self.frame += 1;

        self.ninja.hor_input = hor_input;
        self.ninja.jump_input = jump_input;
        self.ninja.record_action_start_position();

        // Entity move pass, then entity think pass -- both before ninja physics.
        self.run_entity_passes();

        // Ninja physics (only if not disabled)
        if self.ninja.state != 9 {
            if self.ninja.state != 6 {
                // Normal ninja physics
                self.ninja.integrate();
                self.ninja
                    .pre_collision(&self.entity_grid, self.has_physical_entities);

                // 4 physics substeps.
                //
                // Open-air skip, the same one `tick_tiles_only` already uses:
                // with no active segment in the (radius-padded) swept region,
                // `sweep_circle_vs_tiles` returns 1.0 and the depenetration loop
                // breaks on its first iteration, so all four substeps are
                // provably no-ops. Gated additionally on there being no
                // physically-collidable entity on the map, because those move the
                // ninja inside the loop and would invalidate a region test taken
                // before it -- when that flag is false `pre_collision` leaves the
                // entity neighbourhood empty, so `collide_vs_objects` is a no-op
                // too and the loop reduces exactly to the tiles-only case.
                if self.has_physical_entities || self.frame_has_tile_segments() {
                    for _ in 0..4 {
                        self.collide_vs_objects();
                        self.ninja.collide_vs_tiles(&self.segment_grid, &self.segments);
                    }
                }

                // Post-collision with entity handler
                self.post_collision_with_entities();
            }
            // Think
            self.ninja.think();
            // Then the limbs, which is where `nsim.py:527` puts them: a hitscan
            // entity thinking on frame N tests against the pose left at the end
            // of frame N-1.
            self.ninja.update_graphics();
        }

        // The ragdoll pose. Python freezes the skeleton on a fixed frame once the
        // ninja is dead (`nsim.py:529-532`); nothing reads it here -- a dead ninja
        // is not a valid hitscan target -- but keeping it in step costs nothing
        // and keeps the two `update_graphics` call sites aligned.
        if self.ninja.state == 6 && self.ninja.anim_enabled {
            self.ninja.anim_frame = 105;
            self.ninja.anim_state = 7;
            self.ninja.calc_ninja_position();
        }

        // Snapshot the ninja's position for the evil ninja's tape. Last thing in
        // the frame, as in the game's tick -- so an evil ninja thinking on frame N
        // can only read records written on frames 1..N-1.
        self.record_ninja_tape();
    }

    /// Append the ninja's position to the recorder, wrapping the write index with
    /// the binary's `(idx + 1 > 17999) ? 0 : idx + 1` rather than a modulo, so the
    /// index stays comparable across ports.
    fn record_ninja_tape(&mut self) {
        if self.ninja_tape.is_empty() {
            return;
        }
        self.ninja_tape[self.ninja_tape_index] = (self.ninja.xpos, self.ninja.ypos);
        let next = self.ninja_tape_index + 1;
        self.ninja_tape_index = if next > self.ninja_tape.len() - 1 { 0 } else { next };
    }

    /// One evil ninja's `think`. Split out of `Entity::think` because it needs the
    /// simulator's position tape, which `EntityCtx` deliberately does not carry
    /// (the same reason `death_ball_think` is separate).
    fn evil_ninja_think(&mut self, idx: usize) {
        let write_index = self.ninja_tape_index;
        let Simulator {
            entities,
            entity_grid,
            grid_edges,
            segments,
            segment_grid,
            ninja,
            ninja_tape,
            ..
        } = self;
        let mut ctx = EntityCtx {
            index: idx,
            entity_grid,
            grid_edges,
            segments,
            segment_grid,
            ninja,
        };
        if let Entity::EvilNinja(e) = &mut entities[idx] {
            e.think(&mut ctx, ninja_tape, write_index);
        }
    }

    /// One death ball's full `think`: seek + tile collision, then repel every
    /// death ball that comes AFTER it, then update its occupancy cell.
    ///
    /// Split out of `Entity::think` because the repulsion writes to sibling
    /// entities. The ordering is Python's: ball *i* finishes its motion and
    /// pushes balls *i+1..* before ball *i+1* is stepped at all, so a chain of
    /// touching balls resolves front-to-back within a single frame.
    ///
    /// `db_count` comes from the map header (`map_data[1200]`), NOT from the
    /// number of balls actually loaded -- Python reads the header, and a map
    /// whose header undercounts silently skips the repulsion for the tail.
    fn death_ball_think(&mut self, idx: usize) {
        let Simulator {
            entities,
            ninja,
            segment_grid,
            segments,
            ..
        } = self;
        let Entity::DeathBall(ball) = &mut entities[idx] else {
            return;
        };
        // A false return means the depenetration bailed on `dist == 0`, and
        // Python's `return` there abandons the rest of `think`: no ball-vs-ball
        // repulsion and, critically, no `grid_move`. See `death_ball.rs`.
        if !ball.think_motion(ninja, segment_grid, segments) {
            return;
        }
        let (my_index, mx, my) = (ball.index_in_type, ball.xpos, ball.ypos);

        // `death_ball_count` comes from `map_data[1200]`, which is 0 on every
        // map that actually has death balls -- this pass was dead code. Count
        // the balls instead; see entity_death_ball.py for the capture evidence.
        let db_count = self
            .entities
            .iter()
            .filter(|e| matches!(e, Entity::DeathBall(_)))
            .count();
        if my_index + 1 < db_count {
            // Collect first: we need &mut on two distinct entities at once.
            let later: Vec<usize> = self
                .entities
                .iter()
                .enumerate()
                .filter_map(|(i, e)| match e {
                    Entity::DeathBall(b) if b.index_in_type > my_index => Some(i),
                    _ => None,
                })
                .collect();
            let mut self_dvx = 0.0;
            let mut self_dvy = 0.0;
            for other_idx in later {
                let Entity::DeathBall(other) = &mut self.entities[other_idx] else {
                    continue;
                };
                let dx = mx - other.xpos;
                let dy = my - other.ypos;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < 16.0 {
                    let px = dx / dist * 4.0;
                    let py = dy / dist * 4.0;
                    self_dvx += px;
                    self_dvy += py;
                    other.xspeed -= px;
                    other.yspeed -= py;
                }
            }
            if let Entity::DeathBall(ball) = &mut self.entities[idx] {
                ball.xspeed += self_dvx;
                ball.yspeed += self_dvy;
            }
        }

        let Simulator {
            entities,
            entity_grid,
            grid_edges,
            segments,
            segment_grid,
            ninja,
            ..
        } = self;
        let mut ctx = EntityCtx {
            index: idx,
            entity_grid,
            grid_edges,
            segments,
            segment_grid,
            ninja,
        };
        if let Entity::DeathBall(ball) = &mut entities[idx] {
            ball.finish_move(&mut ctx);
        }
    }

    /// Handle physical (depenetration) collisions between the ninja and nearby
    /// entities. Faithful port of `Ninja.collide_vs_objects` (nclone/ninja.py:255):
    /// iterate the entity neighbourhood cached in `pre_collision` and apply each
    /// physically-collidable entity's depenetration to the ninja's position,
    /// crush accumulators, velocity, and floor/ceiling normals.
    ///
    /// This lives on `Simulator` rather than `Ninja` because
    /// `Entity::physical_collision` takes `&mut self`: a bounce block pushes
    /// BACK, absorbing 80% of the depenetration into its own position and
    /// velocity. Driving the loop here lets the entity be borrowed mutably while
    /// the ninja is borrowed immutably (disjoint fields), which a method on
    /// `Ninja` holding `&[Entity]` cannot express.
    ///
    /// Only one-way platforms (type 11) are physically collidable today; the
    /// type-17/20/28 branches are retained verbatim from Python for when bounce
    /// blocks, thwumps and shove thwumps land.
    fn collide_vs_objects(&mut self) {
        let Simulator {
            entities, ninja, ..
        } = self;
        for i in 0..ninja.cached_entity_indices.len() {
            let idx = ninja.cached_entity_indices[i];
            if idx >= entities.len() {
                continue;
            }
            if !entities[idx].is_physical_collidable() {
                continue;
            }
            let etype = entities[idx].entity_type_id();
            let Some(((depen_x, depen_y), (depen_len, _))) =
                entities[idx].physical_collision(ninja)
            else {
                continue;
            };
            let pop_x = depen_x * depen_len;
            let pop_y = depen_y * depen_len;
            ninja.xpos += pop_x;
            ninja.ypos += pop_y;
            // `Ninja::addContact` accumulates the crush vector only when the
            // contact is `isSolid || isMoving`, and each entity states both flags
            // as a store into the same two adjacent bytes of the contact struct:
            // one-way (11) `word 1` -> (solid, !moving); bounce block (17)
            // `word 0` -> neither; thwump (20) `word 0x100` -> (!solid, moving);
            // shove thwump (28) a **BYTE** `0`, one narrower than its siblings,
            // which clears `isSolid` and leaves `isMoving` at whatever the last
            // physical contact wrote. Nothing re-zeroes the struct between
            // entities or between frames. Round 7 §2; see the Python twin.
            match etype {
                11 => {
                    ninja.contact_is_solid = true;
                    ninja.contact_is_moving = false;
                }
                17 => {
                    ninja.contact_is_solid = false;
                    ninja.contact_is_moving = false;
                }
                20 => {
                    ninja.contact_is_solid = false;
                    ninja.contact_is_moving = true;
                }
                // `isMoving` survives
                28 => ninja.contact_is_solid = false,
                _ => {}
            }
            if ninja.contact_is_solid || ninja.contact_is_moving {
                ninja.x_crush += pop_x;
                ninja.y_crush += pop_y;
                ninja.crush_len += depen_len;
            }
            // Ninja can only get crushed by a thwump.
            if etype == 20 {
                ninja.is_crushable = true;
            }
            // Depenetration for bounce blocks, thwumps and shwumps.
            if etype == 17 || etype == 20 || etype == 28 {
                ninja.xspeed += pop_x;
                ninja.yspeed += pop_y;
            }
            // A solid contact PROJECTS the velocity, and only when the ninja is
            // moving into the surface; a non-solid one adds the impulse above.
            //
            // `v -= dot * n`, the same form round 8 §2b.1 established for the tile
            // path -- the one-way was left on the old cross-product-with-the-
            // tangent spelling. Algebraically identical for a unit normal,
            // numerically not: landing flat on a one-way, `nx` is exactly zero, so
            // `yspeed -= dot * ny` cancels to `+0.0` whichever way the ninja is
            // running, while `(vx*ny - vy*nx) * -nx` carries the sign of `vx` into
            // the zero. Ground truth (frame 8 of `10138412`, `10138730` and
            // `10139437`) has `+0.0` in every case.
            if etype == 11 {
                let dot_product = ninja.xspeed * depen_x + ninja.yspeed * depen_y;
                if dot_product < 0.0 {
                    ninja.xspeed -= dot_product * depen_x;
                    ninja.yspeed -= dot_product * depen_y;
                }
            }
            // Adjust ceiling variables on ceiling/wall contact, else floor.
            if depen_y >= -0.0001 {
                ninja.ceiling_count += 1;
                ninja.ceiling_normal_x += depen_x;
                ninja.ceiling_normal_y += depen_y;
            } else {
                ninja.floor_count += 1;
                ninja.floor_normal_x += depen_x;
                ninja.floor_normal_y += depen_y;
            }
        }
    }

    /// Run the per-frame entity `move` pass followed by the `think` pass, in
    /// Python's `entity_dic` order. Port of nsim.py:388-404.
    ///
    /// The active movable/thinkable sets are snapshotted ONCE, before either
    /// pass, exactly as Python's single combined classification loop does. The
    /// difference is observable: an entity deactivated during `move` still gets
    /// its `think` on the same frame.
    fn run_entity_passes(&mut self) {
        // Both lists are drawn from precomputed, correctly ordered candidate
        // slices (see the `move_candidates` / `think_candidates` field docs), and
        // land in buffers owned by the sim, so the passes neither sort nor
        // allocate. The per-frame predicates are unchanged.
        let mut movable = std::mem::take(&mut self.movable_buf);
        let mut thinkable = std::mem::take(&mut self.thinkable_buf);
        movable.clear();
        thinkable.clear();
        for &idx in &self.move_candidates {
            if self.entities[idx].active() {
                movable.push(idx);
            }
        }
        for &idx in &self.think_candidates {
            let entity = &self.entities[idx];
            if entity.active() && entity.is_thinkable() {
                thinkable.push(idx);
            }
        }
        // The think pass is NOT ascending by type. Spec §4 reads both pass orders
        // straight out of the frame loop:
        //
        //   move :  3 4 5 6 8 10 12 13 14 15 16 17 18 19 20 24 25 26   (ascending)
        //   think:  5 12 13 15 16 17 18 19 20 [1] 21 22 23 25 27 28 29
        //
        // Type 1 (mine) is thought TENTH -- after type 20, immediately before
        // type 21, which uses the same function. Both this port and nsim.py used
        // to derive the two lists from ONE ascending classification, so
        // `move order == think order` by construction; the game uses two.
        //
        // Observable when a hazard kills the ninja on the same frame a mine is
        // mid-toggle: `EntityToggleMine.think` reverts to untoggled if it sees a
        // DEAD ninja, so whether the killing think ran first decides the mine's
        // state. Types 18, 19 and 20 all think before type 1 in the game; before
        // 18 and 19 landed, only type 20 sat in that window.
        // `think_candidates` is already in this order -- see its field docs.
        if movable.is_empty() && thinkable.is_empty() {
            self.movable_buf = movable;
            self.thinkable_buf = thinkable;
            return;
        }

        // Destructure so the borrow checker can see that the entity being
        // stepped and the context it is handed are disjoint fields.
        let Simulator {
            entities,
            entity_grid,
            grid_edges,
            segments,
            segment_grid,
            ninja,
            ..
        } = self;

        for &idx in &movable {
            let mut ctx = EntityCtx {
                index: idx,
                entity_grid: &mut *entity_grid,
                grid_edges,
                segments,
                segment_grid,
                ninja: &mut *ninja,
            };
            // Twin of `EntityDroneChaser.move` under NCLONE_AB_CHASER_EARLY_LATCH:
            // detect BEFORE moving, so the frame a chase latches is already run
            // at 16/7. Without it the drone is permanently 8/7 behind the game.
            if crate::ab::ab().chaser_early_latch {
                if let Entity::Drone(d) = &entities[idx] {
                    if matches!(d.behaviour, crate::entities::drone::DroneBehaviour::Chaser(_)) {
                        entities[idx].think(&mut ctx);
                    }
                }
            }
            entities[idx].move_entity(&mut ctx);
        }
        for &idx in &thinkable {
            // Toggle mines are the bulk of the think pass on most maps (they
            // outnumber every other type in the corpus about two to one), and
            // their think reads nothing but the ninja. Dispatching them here
            // skips the 25-arm jump table, the `EntityCtx` build and the door
            // test below -- all of which are no-ops for a mine.
            if let Entity::ToggleMine(mine) = &mut entities[idx] {
                mine.think(ninja);
                continue;
            }
            if matches!(entities[idx], Entity::DeathBall(_) | Entity::EvilNinja(_)) {
                // Handled below: the death ball needs its sibling balls and the
                // evil ninja needs the position tape, neither of which EntityCtx
                // can hand out.
                continue;
            }
            let mut ctx = EntityCtx {
                index: idx,
                entity_grid: &mut *entity_grid,
                grid_edges,
                segments,
                segment_grid,
                ninja: &mut *ninja,
            };
            entities[idx].think(&mut ctx);
            // Immediately, not in the sync pass below: a door that just closed
            // has to be solid to the entities still to think this frame.
            // `sync_one_door` is a no-op for every other variant, so the door
            // test is hoisted here -- it saves a 25-arm dispatch per thinking
            // entity per frame, and toggle mines make that the bulk of the pass.
            if matches!(entities[idx], Entity::SwitchDoor(_) | Entity::LockedDoor(_)) {
                sync_one_door(&mut entities[idx], segments, grid_edges);
            }
        }

        for i in 0..thinkable.len() {
            let idx = thinkable[i];
            if matches!(self.entities[idx], Entity::DeathBall(_)) {
                self.death_ball_think(idx);
            } else if matches!(self.entities[idx], Entity::EvilNinja(_)) {
                self.evil_ninja_think(idx);
            }
        }

        // A regular door that auto-closed during `think` must be solid before the
        // ninja's collision substeps run this frame.
        self.sync_door_barriers();

        self.movable_buf = movable;
        self.thinkable_buf = thinkable;
    }

    /// Execute one physics frame without entity AI (for value iteration).
    pub fn tick_physics_only(&mut self, hor_input: i32, jump_input: i32) {
        self.frame += 1;
        self.ninja.hor_input = hor_input;
        self.ninja.jump_input = jump_input;
        self.ninja.record_action_start_position();

        if self.ninja.state != 9 {
            if self.ninja.state != 6 {
                self.ninja.integrate();
                self.ninja
                    .pre_collision(&self.entity_grid, self.has_physical_entities);

                // Same open-air skip as `tick`; see the comment there.
                if self.has_physical_entities || self.frame_has_tile_segments() {
                    for _ in 0..4 {
                        self.collide_vs_objects();
                        self.ninja.collide_vs_tiles(&self.segment_grid, &self.segments);
                    }
                }

                self.post_collision_with_entities();
            }
            self.ninja.think();
        }
    }

    /// Execute one physics frame with tile collisions only — no entity interactions.
    ///
    /// Used for surface-graph edge enumeration so that exit switches, exit doors, and
    /// toggle mines cannot corrupt simulator state across multiple simulated arcs.
    /// `save_ninja_state` / `restore_ninja_state` only snapshot ninja state, not entity
    /// state, so any entity mutation (switch deactivation, door activation, mine kill)
    /// would persist across arc simulations and break graph connectivity.
    ///
    /// This tick variant is geometrically identical to `tick_physics_only` for levels
    /// that contain only tile geometry — no entity depenetration, no win/death from
    /// entities.  Impact death from tile collisions is still applied.
    pub fn tick_tiles_only(&mut self, hor_input: i32, jump_input: i32) {
        self.frame += 1;
        self.ninja.hor_input = hor_input;
        self.ninja.jump_input = jump_input;
        self.ninja.record_action_start_position();

        if self.ninja.state != 9 {
            if self.ninja.state != 6 {
                self.ninja.integrate();
                // tiles-only: entity depenetration is intentionally excluded, so
                // never gather the physical-entity neighbourhood.
                self.ninja.pre_collision(&self.entity_grid, false);

                // Open-air skip: the 4 substeps are a no-op when no active segment
                // is within the swept region (the common case mid-flight).
                if self.frame_has_tile_segments() {
                    for _ in 0..4 {
                        // Skip collide_vs_objects — no entity depenetration or bounce blocks
                        self.ninja.collide_vs_tiles(&self.segment_grid, &self.segments);
                    }
                }

                self.post_collision_tiles_only();
            }
            self.ninja.think();
        }
    }

    /// Execute one physics frame with tile collisions + toggle-mine DEATH only.
    ///
    /// The ninja trajectory is identical to `tick_tiles_only` (toggle mines exert
    /// no physical force — the ninja passes through them), but the ninja is
    /// additionally killed on contact with a *currently-deadly* toggle mine
    /// (state 0).  Switch/door logical collisions are SKIPPED and, critically,
    /// **no entity state is mutated** — the lethal mine is NOT toggled to safe as
    /// it would be in `tick` (`post_collision_with_entities`).  Because the
    /// entity field is left pristine, `save_ninja_state`/`restore_ninja_state`
    /// (ninja-only) remains valid across the thousands of arcs of surface-graph
    /// edge enumeration: a pruned (lethal) arc does not poison the mine field for
    /// the next arc.
    ///
    /// Use this for entity-aware edge enumeration so arcs are pruned by REAL
    /// physics death (matching the executor) rather than a geometric
    /// waypoint-clearance test, which mismatches the continuous replay near mine
    /// boundaries (a bin-center arc can clear by >14px yet the real trajectory
    /// drifts into the kill zone).
    pub fn tick_physics_mines_only(&mut self, hor_input: i32, jump_input: i32) {
        self.frame += 1;
        self.ninja.hor_input = hor_input;
        self.ninja.jump_input = jump_input;
        self.ninja.record_action_start_position();

        if self.ninja.state != 9 {
            if self.ninja.state != 6 {
                self.ninja.integrate();
                // mines-only: like tiles-only, no entity depenetration.
                self.ninja.pre_collision(&self.entity_grid, false);

                // Open-air skip (see tick_tiles_only): no-op substeps when the swept
                // region holds no active segment.
                if self.frame_has_tile_segments() {
                    for _ in 0..4 {
                        // Tiles only — mines have no physical (depenetration) collision.
                        self.ninja.collide_vs_tiles(&self.segment_grid, &self.segments);
                    }
                }

                self.post_collision_tiles_only();
                self.kill_on_deadly_mine_contact();
            }
            self.ninja.think();
        }
    }

    /// Kill the ninja if it overlaps a currently-deadly toggle mine (state 0),
    /// WITHOUT mutating the mine (no toggle-to-safe).  Entity-state-preserving
    /// companion to `tick_physics_mines_only`; switch/door are ignored.
    fn kill_on_deadly_mine_contact(&mut self) {
        if !self.ninja.is_valid_target() {
            return;
        }
        let (nx, ny) = (self.ninja.xpos, self.ninja.ypos);
        // Broadphase: a deadly mine can only be gathered from cells flagged in
        // `deadly_mine_zone`.  Elsewhere the gather + overlap loop are skipped.
        if self.kernel_opt {
            let (cx, cy) = crate::collision::spatial::cell_from_pos(nx, ny);
            if !self.deadly_mine_zone[cx][cy] {
                return;
            }
        }
        let entity_indices = crate::collision::spatial::gather_entities_from_neighbourhood(
            &self.entity_grid,
            nx,
            ny,
        );
        let mut hit = false;
        for eidx in entity_indices {
            if eidx >= self.entities.len() {
                continue;
            }
            if let Entity::ToggleMine(mine) = &self.entities[eidx] {
                if mine.state == 0
                    && crate::collision::intersection::overlap_circle_vs_circle(
                        mine.xpos,
                        mine.ypos,
                        mine.radius,
                        nx,
                        ny,
                        crate::constants::NINJA_RADIUS,
                    )
                {
                    hit = true;
                    break;
                }
            }
        }
        if hit {
            // No death kinematics needed for enumeration; the arc is simply pruned.
            self.ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("mine"));
        }
    }

    /// Handle post-collision entity interactions.
    fn post_collision_with_entities(&mut self) {
        // Gather entity indices from neighbourhood into a buffer kept across
        // frames, so the per-frame gather does not allocate.
        let mut entity_indices = std::mem::take(&mut self.scratch_entity_indices);
        crate::collision::spatial::gather_entities_from_neighbourhood_into(
            &mut entity_indices,
            &self.entity_grid,
            self.ninja.xpos,
            self.ninja.ypos,
        );

        // Contacts in the game's list order, kept alongside the sum for the
        // `NCLONE_AB_WN_FIRST` variant. Entities come first there, so the tile
        // contacts gathered after the loop go in the second slot. Only the FIRST
        // contact of each list is ever read, so only the first is kept.
        let mut wall_normal = 0.0f64;
        let mut entity_wn: Option<f64> = None;
        let mut tile_wn: Option<f64> = None;

        // Process entity logical collisions.
        //
        // A LIST, not one index: a map can carry several exit pairs (`10131323`
        // has two, and both switches sit on the same tile), so more than one can
        // be collected in a single frame. Keeping only the last one left the other
        // door shut forever -- the ninja then walked through it on frame 392 and
        // native reported a timeout where python won.
        let mut collected_doors = std::mem::take(&mut self.scratch_collected_doors);
        collected_doors.clear();

        for &eidx in &entity_indices {
            if eidx >= self.entities.len() {
                continue;
            }
            match &mut self.entities[eidx] {
                Entity::ToggleMine(mine) => {
                    if self.ninja.is_valid_target() && mine.state == 0 {
                        if crate::collision::intersection::overlap_circle_vs_circle(
                            mine.xpos,
                            mine.ypos,
                            mine.radius,
                            self.ninja.xpos,
                            self.ninja.ypos,
                            crate::constants::NINJA_RADIUS,
                        ) {
                            mine.state = 1;
                            mine.radius = crate::constants::TOGGLE_MINE_RADIUS_UNTOGGLED;
                            self.ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("mine"));
                        }
                    }
                }
                Entity::Gold(gold) => {
                    // Collected by a ninja not in the winning state (state 8);
                    // mirrors entity_gold.py. No physical interaction.
                    if gold.active && self.ninja.state != 8 {
                        if crate::collision::intersection::overlap_circle_vs_circle(
                            gold.xpos,
                            gold.ypos,
                            crate::constants::GOLD_RADIUS,
                            self.ninja.xpos,
                            self.ninja.ypos,
                            crate::constants::NINJA_RADIUS,
                        ) {
                            gold.active = false;
                            self.ninja.gold_collected += 1;
                        }
                    }
                }
                Entity::ExitSwitch(switch) => {
                    if switch.active {
                        if crate::collision::intersection::overlap_circle_vs_circle(
                            switch.xpos,
                            switch.ypos,
                            crate::constants::EXIT_SWITCH_RADIUS,
                            self.ninja.xpos,
                            self.ninja.ypos,
                            crate::constants::NINJA_RADIUS,
                        ) {
                            switch.active = false;
                            collected_doors.push(switch.parent_door_index);
                        }
                    }
                }
                Entity::ExitDoor(door) => {
                    if door.switch_hit {
                        if crate::collision::intersection::overlap_circle_vs_circle(
                            door.xpos,
                            door.ypos,
                            crate::constants::EXIT_DOOR_RADIUS,
                            self.ninja.xpos,
                            self.ninja.ypos,
                            crate::constants::NINJA_RADIUS,
                        ) {
                            self.ninja.win();
                        }
                    }
                }
                Entity::LockedDoor(door) => {
                    // Switch contact (radius 5) opens the door permanently.
                    if door.logical_collision(&mut self.ninja) {
                        self.ninja.doors_opened += 1;
                    }
                }
                Entity::OneWay(ow) => {
                    // Directional barrier: a horizontal-facing one-way contributes a
                    // wall normal (±1) when the ninja is depenetrated against it,
                    // letting it wall-slide/wall-jump. Mirrors the entity-loop branch
                    // of post_collision (ninja.py). Order vs the tile-segment wall_normal
                    // sum above is irrelevant: only the SIGN of the total is consumed.
                    if let Some(n) = ow.logical_collision(&self.ninja) {
                        wall_normal += n;
                        // The game appends a contact only when the logical
                        // collision's `out[1]` -- its y-normal -- is exactly zero
                        // (`ninja_collide_10513C60.c:238`), i.e. only for a
                        // horizontal normal. A top/bottom contact yields
                        // `depen_x == 0.0`, and pushing that puts a zero at the
                        // head of the list. Summing is immune to it; taking
                        // `list[0]` is not. Python drops these implicitly because
                        // `0.0` is falsy.
                        if n != 0.0 {
                            entity_wn.get_or_insert(n);
                        }
                    }
                }
                Entity::LaunchPad(pad) => {
                    // Port of the `entity.type == 10` branch of
                    // Ninja.post_collision (nclone/ninja.py:434-451). The boost
                    // REPLACES velocity rather than adding to it, and is scaled by
                    // 2/3 on the way in.
                    if let Some((xboost_raw, yboost_raw)) = pad.logical_collision(&self.ninja) {
                        let xboost = xboost_raw * 2.0 / 3.0;
                        let yboost = yboost_raw * 2.0 / 3.0;
                        let n = &mut self.ninja;
                        n.xpos += xboost;
                        n.ypos += yboost;
                        n.xspeed = xboost;
                        n.yspeed = yboost;
                        // Launching cancels ground contact outright, so the ninja
                        // cannot floor-jump out of the launch on the same frame.
                        n.floor_count = 0;
                        n.floor_buffer = -1;
                        let boost_scalar = (xboost * xboost + yboost * yboost).sqrt();
                        n.xlp_boost_normalized = xboost / boost_scalar;
                        n.ylp_boost_normalized = yboost / boost_scalar;
                        n.launch_pad_buffer = 0;
                        // A launch out of a jump switches to fall gravity; the
                        // state becomes Falling either way.
                        if n.state == 3 {
                            n.applied_gravity = crate::constants::GRAVITY_FALL;
                        }
                        n.state = 4;
                    }
                }
                Entity::Drone(d) => {
                    // Instantly lethal on contact; no physical interaction.
                    d.logical_collision(&mut self.ninja);
                }
                Entity::BounceBlock(bb) => {
                    // Side contact contributes a wall normal, same as a one-way.
                    if let Some(n) = bb.logical_collision(&self.ninja) {
                        wall_normal += n;
                        // The game appends a contact only when the logical
                        // collision's `out[1]` -- its y-normal -- is exactly zero
                        // (`ninja_collide_10513C60.c:238`), i.e. only for a
                        // horizontal normal. A top/bottom contact yields
                        // `depen_x == 0.0`, and pushing that puts a zero at the
                        // head of the list. Summing is immune to it; taking
                        // `list[0]` is not. Python drops these implicitly because
                        // `0.0` is falsy.
                        if n != 0.0 {
                            entity_wn.get_or_insert(n);
                        }
                    }
                }
                Entity::Thwump(tw) => {
                    // Wall normal from any side, plus death on the charging face.
                    if let Some(n) = tw.logical_collision(&mut self.ninja) {
                        wall_normal += n;
                        // The game appends a contact only when the logical
                        // collision's `out[1]` -- its y-normal -- is exactly zero
                        // (`ninja_collide_10513C60.c:238`), i.e. only for a
                        // horizontal normal. A top/bottom contact yields
                        // `depen_x == 0.0`, and pushing that puts a zero at the
                        // head of the list. Summing is immune to it; taking
                        // `list[0]` is not. Python drops these implicitly because
                        // `0.0` is falsy.
                        if n != 0.0 {
                            entity_wn.get_or_insert(n);
                        }
                    }
                }
                Entity::DeathBall(db) => {
                    db.logical_collision(&mut self.ninja);
                }
                Entity::SwitchDoor(d) => {
                    // Regular: proximity opens. Trap: switch contact shuts it for
                    // good. Either way only `closed` moves here -- the segment and
                    // grid edges are reconciled by sync_door_barriers below.
                    d.logical_collision(&self.ninja);
                }
                Entity::ShoveThwump(st) => {
                    // Latches the shove direction, or kills on the inner core.
                    if let Some(n) = st.logical_collision(&mut self.ninja) {
                        wall_normal += n;
                        // The game appends a contact only when the logical
                        // collision's `out[1]` -- its y-normal -- is exactly zero
                        // (`ninja_collide_10513C60.c:238`), i.e. only for a
                        // horizontal normal. A top/bottom contact yields
                        // `depen_x == 0.0`, and pushing that puts a zero at the
                        // head of the list. Summing is immune to it; taking
                        // `list[0]` is not. Python drops these implicitly because
                        // `0.0` is falsy.
                        if n != 0.0 {
                            entity_wn.get_or_insert(n);
                        }
                    }
                }
                // Boost pads act in the `move` pass, not here.
                Entity::BoostPad(_) => {}
                // The reverse-engineered hazards all report a plain "did I touch
                // the ninja" bool, so they need none of the bespoke handling
                // above (wall normals, switch bookkeeping, recoil) and delegate
                // straight to the shared dispatcher. Turret (19) and DualLaser
                // (23) return false unconditionally -- their bodies do not
                // collide, and their beams kill inside `think` (spec §6.6).
                other @ (Entity::FloorGuard(_)
                | Entity::EyeBat(_)
                | Entity::Rocket(_)
                | Entity::Turret(_)
                | Entity::DualLaser(_)
                | Entity::EvilNinja(_)
                | Entity::PcRocket(_)) => {
                    other.logical_collision(&mut self.ninja);
                }
            }
        }

        // Any door whose `closed` flipped during the entity loop now gets its
        // collision segment and half-cell grid edges brought back in line.
        self.sync_door_barriers();

        // Wall interactions from tile segments -- AFTER the entity loop, because
        // that loop moves the ninja. A launch pad (10) adds its boost straight to
        // `xpos`/`ypos` (`Ninja.post_collision`, nclone/ninja.py:527-528), so
        // gathering the segments first tests the ninja against the wall it was
        // still standing next to and reports `walled == true`. On `10132650` that
        // sent the jump ladder down the wall-jump row instead of `lp_jump` on
        // frame 1, and native died where the game and the python sim win.
        let rad = crate::constants::NINJA_RADIUS + 0.1;
        let (nxpos, nypos) = (self.ninja.xpos, self.ninja.ypos);
        crate::collision::spatial::for_each_segment_in_region(
            &self.segment_grid,
            &self.segments,
            nxpos - rad,
            nypos - rad,
            nxpos + rad,
            nypos + rad,
            |seg_idx| {
                let seg = &self.segments[seg_idx];
                let (_, a, b) = seg.get_closest_point(nxpos, nypos);
                let dx = nxpos - a;
                let dy = nypos - b;
                let dist = (dx * dx + dy * dy).sqrt();
                if dy.abs() < 0.00001 && dist > 0.0 && dist <= rad {
                    wall_normal += dx / dist;
                    tile_wn.get_or_insert(dx / dist);
                }
                true
            },
        );

        // Every switch collected this frame opens its own door and puts it in the
        // entity grid, so the ninja can reach it from the next frame on.
        for idx in collected_doors.drain(..) {
            if idx < self.entities.len() {
                if let Entity::ExitDoor(door) = &mut self.entities[idx] {
                    door.switch_hit = true;
                    let (cx, cy) = cell_from_pos(door.xpos, door.ypos);
                    self.entity_grid[cx][cy].push(idx);
                }
            }
        }

        // Update airborne/walled state
        self.ninja.airborn_old = self.ninja.airborn;
        self.ninja.airborn = true;
        self.ninja.walled = false;
        if crate::ab::ab().wall_normal_first {
            if let Some(n) = entity_wn.or(tile_wn) {
                self.ninja.walled = true;
                self.ninja.wall_normal = if crate::ab::ab().wall_normal_sign {
                    if n > 0.0 { 1.0 } else { -1.0 }
                } else {
                    n
                };
            }
        } else if wall_normal != 0.0 {
            self.ninja.walled = true;
            self.ninja.wall_normal = if wall_normal > 0.0 { 1.0 } else { -1.0 };
        }

        // Floor normal computation
        if self.ninja.floor_count > 0 {
            self.ninja.airborn = false;
            let floor_scalar = (self.ninja.floor_normal_x * self.ninja.floor_normal_x
                + self.ninja.floor_normal_y * self.ninja.floor_normal_y)
                .sqrt();
            if floor_scalar == 0.0 {
                self.ninja.floor_normalized_x = 0.0;
                self.ninja.floor_normalized_y = -1.0;
            } else {
                self.ninja.floor_normalized_x = self.ninja.floor_normal_x / floor_scalar;
                self.ninja.floor_normalized_y = self.ninja.floor_normal_y / floor_scalar;
            }
            // Terminal impact check
            if self.ninja.state != 8 && self.ninja.airborn_old {
                let impact_vel = -(self.ninja.floor_normalized_x * self.ninja.xspeed_old
                    + self.ninja.floor_normalized_y * self.ninja.yspeed_old);
                if impact_vel
                    > crate::constants::MAX_SURVIVABLE_IMPACT
                        - 4.0 / 3.0 * self.ninja.floor_normalized_y.abs()
                {
                    self.ninja.xspeed = self.ninja.xspeed_old;
                    self.ninja.yspeed = self.ninja.yspeed_old;
                    self.ninja.kill(
                        1,
                        self.ninja.xpos,
                        self.ninja.ypos,
                        self.ninja.xspeed * 0.5,
                        self.ninja.yspeed * 0.5,
                        Some("terminal_impact"),
                    );
                    self.ninja.terminal_impact = true;
                }
            }
        }

        // Temporal tracking
        if self.ninja.airborn {
            self.ninja.frames_airborne += 1;
        } else {
            self.ninja.frames_airborne = 0;
        }
        if !self.ninja.airborn {
            self.ninja.consecutive_floor_frames += 1;
        } else {
            self.ninja.consecutive_floor_frames = 0;
        }
        if self.ninja.walled {
            self.ninja.consecutive_wall_frames += 1;
        } else {
            self.ninja.consecutive_wall_frames = 0;
        }

        // Ceiling normal
        if self.ninja.ceiling_count > 0 {
            let ceiling_scalar = (self.ninja.ceiling_normal_x * self.ninja.ceiling_normal_x
                + self.ninja.ceiling_normal_y * self.ninja.ceiling_normal_y)
                .sqrt();
            if ceiling_scalar == 0.0 {
                self.ninja.ceiling_normalized_x = 0.0;
                self.ninja.ceiling_normalized_y = 1.0;
            } else {
                self.ninja.ceiling_normalized_x = self.ninja.ceiling_normal_x / ceiling_scalar;
                self.ninja.ceiling_normalized_y = self.ninja.ceiling_normal_y / ceiling_scalar;
            }
            if self.ninja.state != 8 {
                let impact_vel = -(self.ninja.ceiling_normalized_x * self.ninja.xspeed_old
                    + self.ninja.ceiling_normalized_y * self.ninja.yspeed_old);
                if impact_vel
                    > crate::constants::MAX_SURVIVABLE_IMPACT
                        - 4.0 / 3.0 * self.ninja.ceiling_normalized_y.abs()
                {
                    self.ninja.xspeed = self.ninja.xspeed_old;
                    self.ninja.yspeed = self.ninja.yspeed_old;
                    self.ninja.kill(
                        1,
                        self.ninja.xpos,
                        self.ninja.ypos,
                        self.ninja.xspeed * 0.5,
                        self.ninja.yspeed * 0.5,
                        Some("terminal_impact"),
                    );
                    self.ninja.terminal_impact = true;
                }
            }
        }

        // Crushing check
        if self.ninja.is_crushable && self.ninja.crush_len > 0.0 {
            if (self.ninja.x_crush * self.ninja.x_crush
                + self.ninja.y_crush * self.ninja.y_crush)
                .sqrt()
                / self.ninja.crush_len
                < crate::constants::MIN_SURVIVABLE_CRUSHING
            {
                // `Some("crush")`, not None: the only ninja kill in either backend that
                // passed no cause, so crush deaths folded into `unknown` and were
                // invisible to `death/by_cause/*`. CRUSH is in the taxonomy
                // (CLASS_ENVIRONMENT) and the python twin labels its own kills.
                self.ninja.kill(2, self.ninja.xpos, self.ninja.ypos, 0.0, 0.0,
                                Some("crush"));
            }
        }

        self.scratch_entity_indices = entity_indices;
        self.scratch_collected_doors = collected_doors;
    }

    /// Post-collision handling using tile geometry only — no entity interactions.
    ///
    /// Computes wall normal from tile segments, updates airborne/walled flags, normalises
    /// floor and ceiling normals, and applies terminal-impact death — identical to the
    /// tile-related portion of `post_collision_with_entities` but without the entity
    /// logical-collision loop (mines, switch, door).
    fn post_collision_tiles_only(&mut self) {
        // Wall normal from tile segments only
        let mut wall_normal = 0.0f64;
        let mut tile_wn: Vec<f64> = Vec::new();
        let rad = crate::constants::NINJA_RADIUS + 0.1;
        let seg_indices = crate::collision::spatial::gather_segments_from_region(
            &self.segment_grid,
            &self.segments,
            self.ninja.xpos - rad,
            self.ninja.ypos - rad,
            self.ninja.xpos + rad,
            self.ninja.ypos + rad,
        );
        for seg_idx in seg_indices {
            let seg = &self.segments[seg_idx];
            let (_, a, b) = seg.get_closest_point(self.ninja.xpos, self.ninja.ypos);
            let dx = self.ninja.xpos - a;
            let dy = self.ninja.ypos - b;
            let dist = (dx * dx + dy * dy).sqrt();
            if dy.abs() < 0.00001 && dist > 0.0 && dist <= rad {
                wall_normal += dx / dist;
                tile_wn.push(dx / dist);
            }
        }

        // Update airborne/walled state
        self.ninja.airborn_old = self.ninja.airborn;
        self.ninja.airborn = true;
        self.ninja.walled = false;
        if crate::ab::ab().wall_normal_first {
            if let Some(&n) = tile_wn.first() {
                self.ninja.walled = true;
                self.ninja.wall_normal = if crate::ab::ab().wall_normal_sign {
                    if n > 0.0 { 1.0 } else { -1.0 }
                } else {
                    n
                };
            }
        } else if wall_normal != 0.0 {
            self.ninja.walled = true;
            self.ninja.wall_normal = if wall_normal > 0.0 { 1.0 } else { -1.0 };
        }

        // Floor normal computation and impact check
        if self.ninja.floor_count > 0 {
            self.ninja.airborn = false;
            let floor_scalar = (self.ninja.floor_normal_x * self.ninja.floor_normal_x
                + self.ninja.floor_normal_y * self.ninja.floor_normal_y)
                .sqrt();
            if floor_scalar == 0.0 {
                self.ninja.floor_normalized_x = 0.0;
                self.ninja.floor_normalized_y = -1.0;
            } else {
                self.ninja.floor_normalized_x = self.ninja.floor_normal_x / floor_scalar;
                self.ninja.floor_normalized_y = self.ninja.floor_normal_y / floor_scalar;
            }
            if self.ninja.state != 8 && self.ninja.airborn_old {
                let impact_vel = -(self.ninja.floor_normalized_x * self.ninja.xspeed_old
                    + self.ninja.floor_normalized_y * self.ninja.yspeed_old);
                if impact_vel
                    > crate::constants::MAX_SURVIVABLE_IMPACT
                        - 4.0 / 3.0 * self.ninja.floor_normalized_y.abs()
                {
                    self.ninja.xspeed = self.ninja.xspeed_old;
                    self.ninja.yspeed = self.ninja.yspeed_old;
                    self.ninja.kill(
                        1,
                        self.ninja.xpos,
                        self.ninja.ypos,
                        self.ninja.xspeed * 0.5,
                        self.ninja.yspeed * 0.5,
                        Some("terminal_impact"),
                    );
                    self.ninja.terminal_impact = true;
                }
            }
        }

        // Temporal tracking
        if self.ninja.airborn {
            self.ninja.frames_airborne += 1;
        } else {
            self.ninja.frames_airborne = 0;
        }
        if !self.ninja.airborn {
            self.ninja.consecutive_floor_frames += 1;
        } else {
            self.ninja.consecutive_floor_frames = 0;
        }
        if self.ninja.walled {
            self.ninja.consecutive_wall_frames += 1;
        } else {
            self.ninja.consecutive_wall_frames = 0;
        }

        // Ceiling normal and impact check
        if self.ninja.ceiling_count > 0 {
            let ceiling_scalar = (self.ninja.ceiling_normal_x * self.ninja.ceiling_normal_x
                + self.ninja.ceiling_normal_y * self.ninja.ceiling_normal_y)
                .sqrt();
            if ceiling_scalar == 0.0 {
                self.ninja.ceiling_normalized_x = 0.0;
                self.ninja.ceiling_normalized_y = 1.0;
            } else {
                self.ninja.ceiling_normalized_x = self.ninja.ceiling_normal_x / ceiling_scalar;
                self.ninja.ceiling_normalized_y = self.ninja.ceiling_normal_y / ceiling_scalar;
            }
            if self.ninja.state != 8 {
                let impact_vel = -(self.ninja.ceiling_normalized_x * self.ninja.xspeed_old
                    + self.ninja.ceiling_normalized_y * self.ninja.yspeed_old);
                if impact_vel
                    > crate::constants::MAX_SURVIVABLE_IMPACT
                        - 4.0 / 3.0 * self.ninja.ceiling_normalized_y.abs()
                {
                    self.ninja.xspeed = self.ninja.xspeed_old;
                    self.ninja.yspeed = self.ninja.yspeed_old;
                    self.ninja.kill(
                        1,
                        self.ninja.xpos,
                        self.ninja.ypos,
                        self.ninja.xspeed * 0.5,
                        self.ninja.yspeed * 0.5,
                        Some("terminal_impact"),
                    );
                    self.ninja.terminal_impact = true;
                }
            }
        }
    }

    /// Reset to initial state (same map).
    pub fn fast_reset(&mut self) {
        self.frame = 0;
        self.ninja.reset_state();

        // Reset all entities
        for entity in &mut self.entities {
            entity.reset_state();
        }

        // Rebuild entity grid
        self.rebuild_entity_grid();
        // After reset_state every locked door is closed again; put its segment
        // and half-cell edges back with it.
        self.resync_doors();
    }

    /// Where every patrol drone will be, `horizons` frames from now.
    ///
    /// Returns one `(zap_xy, mini_xy)` pair per horizon, in the order given --
    /// the same shape as the present-frame read, so the observation code treats
    /// "now" and "later" identically.
    ///
    /// Only zap (14) and mini (26) are forecast. Their patrol is a pure function
    /// of the tile grid and the frame counter, so the roll-out is EXACT, not an
    /// estimate: the position returned for horizon `k` is the position the drone
    /// will actually occupy after `k` more ticks. The chaser and the shooter
    /// drones read the ninja, so rolling them forward would amount to predicting
    /// the agent's own future, and they are excluded. See
    /// `Drone::forecast`, which shares the sim's own motion code so the two
    /// cannot drift apart.
    ///
    /// `horizons` must be sorted ascending; the roll-out is a single pass that
    /// samples as it goes, so cost is O(max(horizons) x n_drones), not
    /// O(sum(horizons) x n_drones).
    pub fn forecast_drone_positions(
        &self,
        horizons: &[u32],
    ) -> Vec<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        let mut out: Vec<(Vec<(f64, f64)>, Vec<(f64, f64)>)> =
            horizons.iter().map(|_| (Vec::new(), Vec::new())).collect();
        for entity in &self.entities {
            let drone = match entity {
                Entity::Drone(d) if d.active => d,
                _ => continue,
            };
            let is_mini = match drone.kind {
                DroneKind::Zap => false,
                DroneKind::Mini => true,
                _ => continue,
            };
            let track = drone.forecast(&self.grid_edges, &self.ninja, horizons);
            for (slot, xy) in out.iter_mut().zip(track) {
                if is_mini {
                    slot.1.push(xy);
                } else {
                    slot.0.push(xy);
                }
            }
        }
        out
    }

    /// Get the current simulator state as a snapshot for Python.
    /// Includes tile_grid to avoid separate FFI call for get_tile_grid().
    pub fn get_state(&self) -> SimulatorState {
        let entity_snapshots: Vec<EntitySnapshot> =
            self.entities.iter().map(|e| e.snapshot()).collect();

        let tile_grid: Vec<u8> = (0..MAP_HEIGHT)
            .flat_map(|y| (0..MAP_WIDTH).map(move |x| self.tile_grid[x][y]))
            .collect();

        SimulatorState {
            ninja: self.ninja.snapshot(),
            entities: entity_snapshots,
            frame: self.frame,
            action_mask: self.ninja.get_valid_action_mask(),
            tile_grid,
        }
    }

    /// Like `get_state` but WITHOUT the (static, never-changing) tile grid.
    /// The per-tick hot path only needs ninja + entities + frame + action_mask;
    /// the tile grid is built once at load. Skipping the ~1100-byte grid copy
    /// every tick removes pure marshalling waste (matters most at frame_skip=1,
    /// where the proxy syncs every primitive frame with no amortization).
    pub fn get_state_dynamic(&self) -> SimulatorState {
        let entity_snapshots: Vec<EntitySnapshot> =
            self.entities.iter().map(|e| e.snapshot()).collect();
        SimulatorState {
            ninja: self.ninja.snapshot(),
            entities: entity_snapshots,
            frame: self.frame,
            action_mask: self.ninja.get_valid_action_mask(),
            tile_grid: Vec::new(),
        }
    }

    /// Get a flat tile grid for Python (row-major, 44*25 = 1100 bytes).
    pub fn get_tile_grid_flat(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(MAP_WIDTH * MAP_HEIGHT);
        for y in 0..MAP_HEIGHT {
            for x in 0..MAP_WIDTH {
                result.push(self.tile_grid[x][y]);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::door_base::door_grid_edges;

    /// Minimal map: empty interior (solid border only), ninja spawned exactly on
    /// a locked door's switch so the door opens on the very first tick.
    /// Coordinates are map units; world pixels are 6x that.
    fn locked_door_map(sw: (i32, i32), door: (i32, i32), orientation: i32) -> Vec<i32> {
        let mut md = vec![0i32; 1300];
        // Tiles 184..1150 stay 0 (empty); the loader adds the solid border ring.
        // Ninja entity slot at 1230: [type=0, x, y, orient, mode].
        md[1231] = sw.0;
        md[1232] = sw.1;
        // Locked door at 1235: [6, door_x, door_y, orientation, mode], with the
        // switch coords living at index+6 / index+7 (i.e. inside the following
        // 5-slot entry, which is itself the type-7 phantom switch).
        md[1235] = 6;
        md[1236] = door.0;
        md[1237] = door.1;
        md[1238] = orientation;
        md[1240] = 7;
        md[1241] = sw.0;
        md[1242] = sw.1;
        md
    }

    fn door_edge_values(sim: &Simulator, door_px: (f64, f64), orientation: u8) -> Vec<i16> {
        door_grid_edges(door_px.0, door_px.1, orientation)
            .iter()
            .map(|&(vertical, x, y)| {
                if vertical {
                    sim.grid_edges.door_ver[x][y]
                } else {
                    sim.grid_edges.door_hor[x][y]
                }
            })
            .collect()
    }

    /// A locked door must block BOTH layers while closed and release both when
    /// its switch is collected: the collision segment (ninja) and the half-cell
    /// grid edges (drones/thwumps). Before the grid-edge retrofit only the
    /// segment flipped, so a drone would have walked through an open door's
    /// still-registered edge -- and, worse, through a *closed* door's missing one.
    #[test]
    fn locked_door_flips_segment_and_grid_edges_together() {
        let sw = (50, 50);
        let door = (60, 50);
        let orientation = 0u8; // vertical
        let door_px = ((door.0 * 6) as f64, (door.1 * 6) as f64);
        let mut sim = Simulator::load(&locked_door_map(sw, door, orientation as i32));

        let seg_idx = match &sim.entities[0] {
            Entity::LockedDoor(d) => d.door_segment_index,
            other => panic!("expected a locked door, got {other:?}"),
        };
        let seg_active = |s: &Simulator| match &s.segments[seg_idx] {
            Segment::Linear(l) => l.active,
            Segment::Circular(c) => c.active,
        };

        // Closed: segment solid, both half-edges registered.
        assert!(seg_active(&sim), "closed door segment should be active");
        assert_eq!(door_edge_values(&sim, door_px, orientation), vec![1, 1]);

        // Ninja spawns on the switch, so one tick collects it.
        sim.tick(0, 0);
        assert_eq!(sim.ninja.doors_opened, 1, "switch should have been collected");
        assert!(!seg_active(&sim), "opened door segment should be inactive");
        assert_eq!(
            door_edge_values(&sim, door_px, orientation),
            vec![0, 0],
            "opening the door must drop its half-cell edges too"
        );

        // Reset puts both layers back.
        sim.fast_reset();
        assert!(seg_active(&sim), "reset door segment should be active again");
        assert_eq!(door_edge_values(&sim, door_px, orientation), vec![1, 1]);
    }

    /// `restore_full_state` used to resync neither the door segment nor its grid
    /// edges, so a Go-Explore checkpoint captured after a door opened restored
    /// with the door still solid.
    #[test]
    fn restoring_an_opened_door_checkpoint_keeps_it_open() {
        let sw = (50, 50);
        let door = (60, 50);
        let orientation = 0u8;
        let door_px = ((door.0 * 6) as f64, (door.1 * 6) as f64);
        let map = locked_door_map(sw, door, orientation as i32);

        let mut source = Simulator::load(&map);
        source.tick(0, 0);
        assert_eq!(source.ninja.doors_opened, 1);
        let blob = source.save_full_state();

        // A pristine sim of the same level: door closed, edges registered.
        let mut target = Simulator::load(&map);
        assert_eq!(door_edge_values(&target, door_px, orientation), vec![1, 1]);

        assert!(target.restore_full_state(&blob), "same level should restore");
        let seg_idx = match &target.entities[0] {
            Entity::LockedDoor(d) => d.door_segment_index,
            other => panic!("expected a locked door, got {other:?}"),
        };
        let seg_active = match &target.segments[seg_idx] {
            Segment::Linear(l) => l.active,
            Segment::Circular(c) => c.active,
        };
        assert!(!seg_active, "restored open door must not be solid again");
        assert_eq!(door_edge_values(&target, door_px, orientation), vec![0, 0]);
    }

    /// A level containing an ARMED toggle mine (type 21) must round-trip.
    ///
    /// `save_full_state` used to write a literal tag `1` for every ToggleMine
    /// while `restore_full_state` validates `tag == entity_type_id()`, which is
    /// 21 for a mine spawned armed. Validation therefore failed, restore returned
    /// false and -- by design -- mutated NOTHING. Callers ignore the bool, so the
    /// incremental save/restore search silently degenerated into one drifting
    /// trajectory instead of a tree. Official levels are ~20% armed mines.
    #[test]
    fn armed_toggle_mine_snapshot_round_trips() {
        // Empty room, ninja at (50,50), one ARMED mine (type 21) at (60,50).
        let mut md = vec![0i32; 1300];
        md[1231] = 50;
        md[1232] = 50;
        md[1235] = 21;
        md[1236] = 60;
        md[1237] = 50;

        let mut sim = Simulator::load(&md);
        assert_eq!(
            sim.entities[0].entity_type_id(),
            21,
            "fixture must actually contain an ARMED mine"
        );
        for _ in 0..10 {
            sim.tick(1, 0);
        }
        let (x_before, vx_before) = (sim.ninja.xpos, sim.ninja.xspeed);
        let blob = sim.save_full_state();

        // Perturb hard, then restore: state must come back exactly.
        for _ in 0..10 {
            sim.tick(-1, 0);
        }
        assert_ne!(sim.ninja.xpos, x_before, "perturbation must move the ninja");

        assert!(
            sim.restore_full_state(&blob),
            "a snapshot of an armed-mine level must validate against its own level"
        );
        assert_eq!(sim.ninja.xpos, x_before, "restore must rewind position");
        assert_eq!(sim.ninja.xspeed, vx_before, "restore must rewind velocity");
    }

    /// The safe (type 1) mine case, so a future refactor cannot fix 21 by
    /// breaking 1.
    #[test]
    fn safe_toggle_mine_snapshot_round_trips() {
        let mut md = vec![0i32; 1300];
        md[1231] = 50;
        md[1232] = 50;
        md[1235] = 1;
        md[1236] = 60;
        md[1237] = 50;

        let mut sim = Simulator::load(&md);
        assert_eq!(sim.entities[0].entity_type_id(), 1);
        for _ in 0..10 {
            sim.tick(1, 0);
        }
        let x_before = sim.ninja.xpos;
        let blob = sim.save_full_state();
        for _ in 0..10 {
            sim.tick(-1, 0);
        }
        assert!(sim.restore_full_state(&blob));
        assert_eq!(sim.ninja.xpos, x_before);
    }
}

/// The barrier a door entity owns: `(segment index, door x, door y, orientation,
/// closed)`. `None` for anything that is not a door.
///
/// Free functions rather than `Entity` methods so `sync_door_barriers` can hold
/// `&self.entities[i]` and `&mut self.segments` at once.
fn door_barrier(e: &Entity) -> Option<(usize, f64, f64, u8, bool)> {
    match e {
        Entity::LockedDoor(d) => Some((
            d.door_segment_index,
            d.door_xpos,
            d.door_ypos,
            d.orientation,
            d.closed,
        )),
        Entity::SwitchDoor(d) => Some((
            d.door_segment_index,
            d.door_xpos,
            d.door_ypos,
            d.orientation,
            d.closed,
        )),
        _ => None,
    }
}

/// Flip ONE door's barrier the instant its own `think` changed it.
///
/// Python has no sync pass: `EntityDoorBase.change_state` writes the segment and
/// the grid edges from inside the door's own `think`, so every entity that
/// thinks LATER in the same frame already sees the new state. Native batched the
/// flip into `sync_door_barriers` after the whole think loop, which left a door
/// that auto-closed this frame transparent to every entity behind it in the
/// order -- and every sightline in the game funnels through those segments.
///
/// On `10244473` that is the whole bug: the door at (432, 300) closes on frame
/// 58, Python's chaingun shot stops on it at 12.19 px, and native's ran the full
/// 48.75 px through the stale-open segment and killed a ninja the real game
/// lets live.
fn sync_one_door(entity: &mut Entity, segments: &mut [Segment], grid_edges: &mut GridEdges) {
    let Some((seg_idx, dx, dy, orientation, closed)) = door_barrier(entity) else {
        return;
    };
    if door_registered(entity) == closed {
        return;
    }
    if let Some(seg) = segments.get_mut(seg_idx) {
        match seg {
            Segment::Linear(s) => s.active = closed,
            Segment::Circular(s) => s.active = closed,
        }
    }
    let delta = if closed { 1 } else { -1 };
    for (vertical, ex, ey) in door_grid_edges(dx, dy, orientation) {
        grid_edges.adjust_door_edge(vertical, ex, ey, delta);
    }
    set_door_registered(entity, closed);
}

fn door_registered(e: &Entity) -> bool {
    match e {
        Entity::LockedDoor(d) => d.registered_closed,
        Entity::SwitchDoor(d) => d.registered_closed,
        _ => true,
    }
}

fn set_door_registered(e: &mut Entity, v: bool) {
    match e {
        Entity::LockedDoor(d) => d.registered_closed = v,
        Entity::SwitchDoor(d) => d.registered_closed = v,
        _ => {}
    }
}
