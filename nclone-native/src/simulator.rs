//! Main Simulator: tick loop with 4 physics substeps.
//! Direct port of nclone/nsim.py.

use crate::collision::spatial::cell_from_pos;
use crate::constants::{MAP_HEIGHT, MAP_WIDTH};
use crate::entities::Entity;
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

    // Tile data for Python access
    pub tile_grid: [[u8; MAP_HEIGHT]; MAP_WIDTH],

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

impl Simulator {
    /// Load a map from integer map data.
    pub fn load(map_data: &[i32]) -> Self {
        let data = loader::load_map(map_data);
        let ninja = Ninja::new(data.ninja_spawn_x, data.ninja_spawn_y);

        let mut sim = Simulator {
            frame: 0,
            ninja,
            segments: data.segments,
            segment_grid: data.segment_grid,
            entities: data.entities,
            entity_grid: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
            tile_grid: data.tile_grid,
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
        sim.compute_deadly_mine_zone();

        sim
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

    /// Execute one full physics frame.
    pub fn tick(&mut self, hor_input: i32, jump_input: i32) {
        self.frame += 1;

        self.ninja.hor_input = hor_input;
        self.ninja.jump_input = jump_input;
        self.ninja.record_action_start_position();

        // Entity think (only toggle mines)
        for idx in 0..self.entities.len() {
            if self.entities[idx].is_thinkable() && self.entities[idx].active() {
                // Need to pass ninja state to entity think
                let ninja_snapshot = NinjaRef {
                    xpos: self.ninja.xpos,
                    ypos: self.ninja.ypos,
                    state: self.ninja.state,
                    is_valid: self.ninja.is_valid_target(),
                };
                match &mut self.entities[idx] {
                    Entity::ToggleMine(mine) => {
                        // Inline think to avoid borrow issues
                        if ninja_snapshot.is_valid {
                            if mine.state == 1 {
                                if crate::collision::intersection::overlap_circle_vs_circle(
                                    mine.xpos,
                                    mine.ypos,
                                    mine.radius,
                                    self.ninja.xpos,
                                    self.ninja.ypos,
                                    crate::constants::NINJA_RADIUS,
                                ) {
                                    mine.state = 2;
                                    mine.radius = crate::constants::TOGGLE_MINE_RADIUS_TOGGLING;
                                }
                            } else if mine.state == 2 {
                                if !crate::collision::intersection::overlap_circle_vs_circle(
                                    mine.xpos,
                                    mine.ypos,
                                    mine.radius,
                                    self.ninja.xpos,
                                    self.ninja.ypos,
                                    crate::constants::NINJA_RADIUS,
                                ) {
                                    mine.state = 0;
                                    mine.radius = crate::constants::TOGGLE_MINE_RADIUS_TOGGLED;
                                }
                            }
                        } else if mine.state == 2 && self.ninja.state == 6 {
                            mine.state = 1;
                            mine.radius = crate::constants::TOGGLE_MINE_RADIUS_UNTOGGLED;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Ninja physics (only if not disabled)
        if self.ninja.state != 9 {
            if self.ninja.state != 6 {
                // Normal ninja physics
                self.ninja.integrate();
                self.ninja.pre_collision(&self.entity_grid);

                // 4 physics substeps
                for _ in 0..4 {
                    self.ninja.collide_vs_objects();
                    self.ninja.collide_vs_tiles(&self.segment_grid, &self.segments);
                }

                // Post-collision with entity handler
                self.post_collision_with_entities();
            }
            // Think
            self.ninja.think();
        }
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
                self.ninja.pre_collision(&self.entity_grid);

                for _ in 0..4 {
                    self.ninja.collide_vs_objects();
                    self.ninja.collide_vs_tiles(&self.segment_grid, &self.segments);
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
                self.ninja.pre_collision(&self.entity_grid);

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
                self.ninja.pre_collision(&self.entity_grid);

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
        // Gather entity indices from neighbourhood
        let entity_indices = crate::collision::spatial::gather_entities_from_neighbourhood(
            &self.entity_grid,
            self.ninja.xpos,
            self.ninja.ypos,
        );

        // Check wall interactions from tile segments
        let mut wall_normal = 0.0f64;
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
            }
        }

        // Process entity logical collisions
        let mut switch_collected = false;
        let mut switch_parent_door = 0usize;

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
                            switch_collected = true;
                            switch_parent_door = switch.parent_door_index;
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
            }
        }

        // If switch was collected, activate parent door and add to entity grid
        if switch_collected {
            if switch_parent_door < self.entities.len() {
                if let Entity::ExitDoor(door) = &mut self.entities[switch_parent_door] {
                    door.switch_hit = true;
                    let (cx, cy) = cell_from_pos(door.xpos, door.ypos);
                    self.entity_grid[cx][cy].push(switch_parent_door);
                }
            }
        }

        // Update airborne/walled state
        self.ninja.airborn_old = self.ninja.airborn;
        self.ninja.airborn = true;
        self.ninja.walled = false;
        if wall_normal != 0.0 {
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
                self.ninja.kill(2, self.ninja.xpos, self.ninja.ypos, 0.0, 0.0, None);
            }
        }
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
            }
        }

        // Update airborne/walled state
        self.ninja.airborn_old = self.ninja.airborn;
        self.ninja.airborn = true;
        self.ninja.walled = false;
        if wall_normal != 0.0 {
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

/// Helper struct to avoid borrow issues when passing ninja state to entity think.
#[allow(dead_code)]
struct NinjaRef {
    xpos: f64,
    ypos: f64,
    state: u8,
    is_valid: bool,
}
