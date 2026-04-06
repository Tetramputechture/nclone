//! Ninja physics and state machine.
//! Direct port of nclone/ninja.py (~1,571 lines).
//! Excludes animation/graphics code (Ragdoll, bones, etc.).

use crate::collision::{spatial, sweep};
use crate::constants::*;
use crate::segments::Segment;
use crate::types::NinjaSnapshot;

/// The Ninja physics object.
pub struct Ninja {
    // Position/velocity
    pub xpos: f64,
    pub ypos: f64,
    pub xspeed: f64,
    pub yspeed: f64,
    pub xpos_old: f64,
    pub ypos_old: f64,
    pub xspeed_old: f64,
    pub yspeed_old: f64,

    // Initial spawn (for reset)
    initial_xpos: f64,
    initial_ypos: f64,

    // Physics parameters
    pub applied_gravity: f64,
    pub applied_drag: f64,
    pub applied_friction: f64,

    // State machine
    pub state: u8,
    pub airborn: bool,
    pub airborn_old: bool,
    pub walled: bool,
    pub wall_normal: f64,

    // Jump/buffer state
    pub jump_input_old: i32,
    pub jump_duration: i32,
    pub jump_buffer: i32,
    pub floor_buffer: i32,
    pub wall_buffer: i32,
    pub launch_pad_buffer: i32,
    pub is_jump_useless: bool,
    pub last_jump_was_buffered: bool,

    // Collision normals
    pub floor_normalized_x: f64,
    pub floor_normalized_y: f64,
    pub ceiling_normalized_x: f64,
    pub ceiling_normalized_y: f64,
    pub floor_count: u32,
    pub wall_count: u32,
    pub ceiling_count: u32,
    pub floor_normal_x: f64,
    pub floor_normal_y: f64,
    pub ceiling_normal_x: f64,
    pub ceiling_normal_y: f64,
    pub is_crushable: bool,
    pub x_crush: f64,
    pub y_crush: f64,
    pub crush_len: f64,

    // Inputs
    pub hor_input: i32,
    pub jump_input: i32,

    // Death state
    pub death_xpos: f64,
    pub death_ypos: f64,
    pub death_xspeed: f64,
    pub death_yspeed: f64,
    pub terminal_impact: bool,
    pub death_cause: Option<String>,

    // Position tracking
    pub xpos_at_action: f64,
    pub ypos_at_action: f64,

    // Collectibles
    pub gold_collected: u32,
    pub doors_opened: u32,

    // Temporal tracking
    pub frames_airborne: u32,
    pub state_change_frame: u32,
    pub previous_state: u8,
    pub consecutive_floor_frames: u32,
    pub consecutive_wall_frames: u32,

    // Launch pad boost (stored for lp_jump)
    pub xlp_boost_normalized: f64,
    pub ylp_boost_normalized: f64,

    // Cached entities from pre_collision
    cached_entity_indices: Vec<usize>,
}

impl Ninja {
    /// Create a new Ninja at the given spawn position.
    pub fn new(spawn_x: f64, spawn_y: f64) -> Self {
        Ninja {
            xpos: spawn_x,
            ypos: spawn_y,
            xspeed: 0.0,
            yspeed: 0.0,
            xpos_old: 0.0,
            ypos_old: 0.0,
            xspeed_old: 0.0,
            yspeed_old: 0.0,
            initial_xpos: spawn_x,
            initial_ypos: spawn_y,
            applied_gravity: GRAVITY_FALL,
            applied_drag: DRAG_REGULAR,
            applied_friction: FRICTION_GROUND,
            state: 0, // Immobile
            airborn: false,
            airborn_old: false,
            walled: false,
            wall_normal: 0.0,
            jump_input_old: 0,
            jump_duration: 0,
            jump_buffer: -1,
            floor_buffer: -1,
            wall_buffer: -1,
            launch_pad_buffer: -1,
            is_jump_useless: false,
            last_jump_was_buffered: false,
            floor_normalized_x: 0.0,
            floor_normalized_y: -1.0,
            ceiling_normalized_x: 0.0,
            ceiling_normalized_y: 1.0,
            floor_count: 0,
            wall_count: 0,
            ceiling_count: 0,
            floor_normal_x: 0.0,
            floor_normal_y: 0.0,
            ceiling_normal_x: 0.0,
            ceiling_normal_y: 0.0,
            is_crushable: false,
            x_crush: 0.0,
            y_crush: 0.0,
            crush_len: 0.0,
            hor_input: 0,
            jump_input: 0,
            death_xpos: 0.0,
            death_ypos: 0.0,
            death_xspeed: 0.0,
            death_yspeed: 0.0,
            terminal_impact: false,
            death_cause: None,
            xpos_at_action: spawn_x,
            ypos_at_action: spawn_y,
            gold_collected: 0,
            doors_opened: 0,
            frames_airborne: 0,
            state_change_frame: 0,
            previous_state: 0,
            consecutive_floor_frames: 0,
            consecutive_wall_frames: 0,
            xlp_boost_normalized: 0.0,
            ylp_boost_normalized: 0.0,
            cached_entity_indices: Vec::new(),
        }
    }

    /// Update position and speed by applying drag and gravity.
    pub fn integrate(&mut self) {
        self.xspeed *= self.applied_drag;
        self.yspeed *= self.applied_drag;
        self.yspeed += self.applied_gravity;
        self.xpos_old = self.xpos;
        self.ypos_old = self.ypos;
        self.xpos += self.xspeed;
        self.ypos += self.yspeed;
    }

    /// Reset collision accumulators before collision phase.
    pub fn pre_collision(
        &mut self,
        entity_grid: &[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
    ) {
        self.xspeed_old = self.xspeed;
        self.yspeed_old = self.yspeed;
        self.floor_count = 0;
        self.wall_count = 0;
        self.ceiling_count = 0;
        self.floor_normal_x = 0.0;
        self.floor_normal_y = 0.0;
        self.ceiling_normal_x = 0.0;
        self.ceiling_normal_y = 0.0;
        self.is_crushable = false;
        self.x_crush = 0.0;
        self.y_crush = 0.0;
        self.crush_len = 0.0;
        self.cached_entity_indices =
            spatial::gather_entities_from_neighbourhood(entity_grid, self.xpos, self.ypos);
    }

    /// Handle physical collisions with entities (bounce blocks, thwumps, etc.).
    /// In our scoped port (only toggle mine, exit switch, exit door), none of
    /// these entities have physical collisions, so this is a no-op.
    pub fn collide_vs_objects(&mut self) {
        // Toggle mines, exit switches, and exit doors have no physical collision
        // (is_physical_collidable = false). Only bounce blocks, thwumps, one-way
        // platforms, etc. have physical collisions, and those are out of scope.
        // This method is kept for API compatibility and future expansion.
    }

    /// Handle collisions with tile segments (interpolation + depenetration loop).
    pub fn collide_vs_tiles(
        &mut self,
        segment_grid: &[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
        all_segments: &[Segment],
    ) {
        // Interpolation to prevent going through walls
        let dx = self.xpos - self.xpos_old;
        let dy = self.ypos - self.ypos_old;
        let time = sweep::sweep_circle_vs_tiles(
            segment_grid,
            all_segments,
            self.xpos_old,
            self.ypos_old,
            dx,
            dy,
            NINJA_RADIUS * 0.5,
        );
        self.xpos = self.xpos_old + time * dx;
        self.ypos = self.ypos_old + time * dy;

        let rad = NINJA_RADIUS;

        // Gather segments once for the depenetration loop
        let seg_indices = spatial::gather_segments_from_region(
            segment_grid,
            all_segments,
            self.xpos - rad,
            self.ypos - rad,
            self.xpos + rad,
            self.ypos + rad,
        );

        // Cache into local vars for the tight loop
        let mut xpos = self.xpos;
        let mut ypos = self.ypos;
        let mut xspeed = self.xspeed;
        let mut yspeed = self.yspeed;
        let mut x_crush = self.x_crush;
        let mut y_crush = self.y_crush;
        let mut crush_len = self.crush_len;
        let mut ceiling_count = self.ceiling_count;
        let mut ceiling_normal_x = self.ceiling_normal_x;
        let mut ceiling_normal_y = self.ceiling_normal_y;
        let mut floor_count = self.floor_count;
        let mut floor_normal_x = self.floor_normal_x;
        let mut floor_normal_y = self.floor_normal_y;

        // Depenetration loop (32 iterations max)
        for _ in 0..32 {
            // Find closest point from gathered segments
            let mut shortest_distance: f64 = 9999999.0;
            let mut result = 0i32;
            let mut closest_a = 0.0f64;
            let mut closest_b = 0.0f64;

            for &seg_idx in &seg_indices {
                let seg = &all_segments[seg_idx];
                if !seg.is_active() {
                    continue;
                }
                let (is_back_facing, a, b) = seg.get_closest_point(xpos, ypos);
                let mut distance_sq = (xpos - a) * (xpos - a) + (ypos - b) * (ypos - b);
                if !is_back_facing {
                    distance_sq -= 0.1;
                }
                if distance_sq < shortest_distance {
                    shortest_distance = distance_sq;
                    closest_a = a;
                    closest_b = b;
                    result = if is_back_facing { -1 } else { 1 };
                }
            }

            if result == 0 {
                break;
            }

            let mut dx = xpos - closest_a;
            let dy = ypos - closest_b;

            // Reproduce Python corner cases (left-border positions)
            if dx.abs() <= 0.0000001 {
                dx = 0.0;
                if xpos == 50.51197510492316 || xpos == 49.23232124849253 {
                    dx = -f64::from_bits(0x3D10000000000000u64); // -(2^-47)
                }
                if xpos == 49.153536108584795 {
                    dx = f64::from_bits(0x3D10000000000000u64); // 2^-47
                }
                // Mirror-symmetric counterparts (right-border positions, for horizontally-mirrored levels)
                if xpos == 1005.4880248950768 || xpos == 1006.7676787515074 {
                    dx = f64::from_bits(0x3D10000000000000u64); // +(2^-47)
                }
                if xpos == 1006.8464638914152 {
                    dx = -f64::from_bits(0x3D10000000000000u64); // -(2^-47)
                }
            }

            let dist = (dx * dx + dy * dy).sqrt();
            let depen_len = NINJA_RADIUS - dist * (result as f64);
            if dist == 0.0 || depen_len < 0.0000001 {
                // Write back before returning, matching original's direct self.* updates
                self.xpos = xpos;
                self.ypos = ypos;
                self.xspeed = xspeed;
                self.yspeed = yspeed;
                self.x_crush = x_crush;
                self.y_crush = y_crush;
                self.crush_len = crush_len;
                self.ceiling_count = ceiling_count;
                self.ceiling_normal_x = ceiling_normal_x;
                self.ceiling_normal_y = ceiling_normal_y;
                self.floor_count = floor_count;
                self.floor_normal_x = floor_normal_x;
                self.floor_normal_y = floor_normal_y;
                return;
            }

            let depen_x = dx / dist * depen_len;
            let depen_y = dy / dist * depen_len;
            xpos += depen_x;
            ypos += depen_y;
            x_crush += depen_x;
            y_crush += depen_y;
            crush_len += depen_len;

            // Project velocity onto surface only if moving towards it
            let dot_product = xspeed * dx + yspeed * dy;
            if dot_product < 0.0 {
                let dist_sq = dist * dist;
                let xspeed_new = (xspeed * dy - yspeed * dx) / dist_sq * dy;
                let yspeed_new = (xspeed * dy - yspeed * dx) / dist_sq * (-dx);
                xspeed = xspeed_new;
                yspeed = yspeed_new;
            }

            if dy >= -0.0001 {
                ceiling_count += 1;
                ceiling_normal_x += dx / dist;
                ceiling_normal_y += dy / dist;
            } else {
                floor_count += 1;
                floor_normal_x += dx / dist;
                floor_normal_y += dy / dist;
            }
        }

        // Write back
        self.xpos = xpos;
        self.ypos = ypos;
        self.xspeed = xspeed;
        self.yspeed = yspeed;
        self.x_crush = x_crush;
        self.y_crush = y_crush;
        self.crush_len = crush_len;
        self.ceiling_count = ceiling_count;
        self.ceiling_normal_x = ceiling_normal_x;
        self.ceiling_normal_y = ceiling_normal_y;
        self.floor_count = floor_count;
        self.floor_normal_x = floor_normal_x;
        self.floor_normal_y = floor_normal_y;
    }

    /// Post-collision: entity logical collisions, airborne/walled checks,
    /// floor normals, impact/crush death.
    pub fn post_collision(
        &mut self,
        segment_grid: &[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
        all_segments: &[Segment],
        entity_grid: &[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
        entity_handler: &mut dyn EntityCollisionHandler,
    ) {
        // Logical collisions with entities
        let entity_indices =
            spatial::gather_entities_from_neighbourhood(entity_grid, self.xpos, self.ypos);
        for idx in entity_indices {
            entity_handler.logical_collision(idx, self);
        }

        // Check wall interactions from tile segments
        let mut wall_normal = 0.0f64;
        let rad = NINJA_RADIUS + 0.1;
        let seg_indices = spatial::gather_segments_from_region(
            segment_grid,
            all_segments,
            self.xpos - rad,
            self.ypos - rad,
            self.xpos + rad,
            self.ypos + rad,
        );
        for seg_idx in seg_indices {
            let seg = &all_segments[seg_idx];
            let (_, a, b) = seg.get_closest_point(self.xpos, self.ypos);
            let dx = self.xpos - a;
            let dy = self.ypos - b;
            let dist = (dx * dx + dy * dy).sqrt();
            if dy.abs() < 0.00001 && dist > 0.0 && dist <= rad {
                wall_normal += dx / dist;
            }
        }

        // Check airborne and walled state
        self.airborn_old = self.airborn;
        self.airborn = true;
        self.walled = false;
        if wall_normal != 0.0 {
            self.walled = true;
            self.wall_normal = if wall_normal > 0.0 { 1.0 } else { -1.0 };
        }

        // Calculate combined floor normal
        if self.floor_count > 0 {
            self.airborn = false;
            let floor_scalar =
                (self.floor_normal_x * self.floor_normal_x + self.floor_normal_y * self.floor_normal_y).sqrt();
            if floor_scalar == 0.0 {
                self.floor_normalized_x = 0.0;
                self.floor_normalized_y = -1.0;
            } else {
                self.floor_normalized_x = self.floor_normal_x / floor_scalar;
                self.floor_normalized_y = self.floor_normal_y / floor_scalar;
            }
            // Check terminal impact on floor
            if self.state != 8 && self.airborn_old {
                let impact_vel = -(self.floor_normalized_x * self.xspeed_old
                    + self.floor_normalized_y * self.yspeed_old);
                if impact_vel
                    > MAX_SURVIVABLE_IMPACT - 4.0 / 3.0 * self.floor_normalized_y.abs()
                {
                    self.xspeed = self.xspeed_old;
                    self.yspeed = self.yspeed_old;
                    self.kill(
                        1,
                        self.xpos,
                        self.ypos,
                        self.xspeed * 0.5,
                        self.yspeed * 0.5,
                        Some("terminal_impact"),
                    );
                    self.terminal_impact = true;
                }
            }
        }

        // Track frames airborne
        if self.airborn {
            self.frames_airborne += 1;
        } else {
            self.frames_airborne = 0;
        }

        // Track consecutive contact frames
        if !self.airborn {
            self.consecutive_floor_frames += 1;
        } else {
            self.consecutive_floor_frames = 0;
        }
        if self.walled {
            self.consecutive_wall_frames += 1;
        } else {
            self.consecutive_wall_frames = 0;
        }

        // Calculate combined ceiling normal
        if self.ceiling_count > 0 {
            let ceiling_scalar = (self.ceiling_normal_x * self.ceiling_normal_x
                + self.ceiling_normal_y * self.ceiling_normal_y)
                .sqrt();
            if ceiling_scalar == 0.0 {
                self.ceiling_normalized_x = 0.0;
                self.ceiling_normalized_y = 1.0;
            } else {
                self.ceiling_normalized_x = self.ceiling_normal_x / ceiling_scalar;
                self.ceiling_normalized_y = self.ceiling_normal_y / ceiling_scalar;
            }
            // Check ceiling impact
            if self.state != 8 {
                let impact_vel = -(self.ceiling_normalized_x * self.xspeed_old
                    + self.ceiling_normalized_y * self.yspeed_old);
                if impact_vel
                    > MAX_SURVIVABLE_IMPACT - 4.0 / 3.0 * self.ceiling_normalized_y.abs()
                {
                    self.xspeed = self.xspeed_old;
                    self.yspeed = self.yspeed_old;
                    self.kill(
                        1,
                        self.xpos,
                        self.ypos,
                        self.xspeed * 0.5,
                        self.yspeed * 0.5,
                        Some("terminal_impact"),
                    );
                    self.terminal_impact = true;
                }
            }
        }

        // Check crushing
        if self.is_crushable && self.crush_len > 0.0 {
            if (self.x_crush * self.x_crush + self.y_crush * self.y_crush).sqrt() / self.crush_len
                < MIN_SURVIVABLE_CRUSHING
            {
                self.kill(2, self.xpos, self.ypos, 0.0, 0.0, None);
            }
        }
    }

    /// Floor jump depending on slope angle and direction.
    fn floor_jump(&mut self, was_buffered: bool) {
        self.jump_buffer = -1;
        self.floor_buffer = -1;
        self.launch_pad_buffer = -1;
        if was_buffered {
            self.last_jump_was_buffered = true;
        }
        self.state = 3; // Jumping
        self.applied_gravity = GRAVITY_JUMP;

        let (jx, jy);
        if self.floor_normalized_x == 0.0 {
            // Flat ground jump
            jx = JUMP_FLOOR_X;
            jy = JUMP_FLOOR_Y;
        } else {
            // Slope jump
            let dx = self.floor_normalized_x;
            let dy = self.floor_normalized_y;
            if self.xspeed * dx >= 0.0 {
                // Moving downhill
                if self.xspeed * (self.hor_input as f64) >= 0.0 {
                    jx = JUMP_SLOPE_DOWNHILL_X_MULTIPLIER * dx;
                    jy = JUMP_SLOPE_DOWNHILL_Y_MULTIPLIER * dy;
                } else {
                    jx = 0.0;
                    jy = JUMP_SLOPE_UPHILL_DEFAULT_Y;
                }
            } else {
                // Moving uphill
                if self.xspeed * (self.hor_input as f64) > 0.0 {
                    jx = 0.0;
                    jy = JUMP_SLOPE_UPHILL_DEFAULT_Y;
                } else {
                    self.xspeed = 0.0;
                    jx = JUMP_SLOPE_UPHILL_PERP_X_MULTIPLIER * dx;
                    jy = JUMP_SLOPE_UPHILL_PERP_Y_MULTIPLIER * dy;
                }
            }
        }

        if self.yspeed > 0.0 {
            self.yspeed = 0.0;
        }
        self.xspeed += jx;
        self.yspeed += jy;
        self.xpos += jx;
        self.ypos += jy;
        self.jump_duration = 0;
    }

    /// Wall jump depending on wall normal and sliding state.
    fn wall_jump(&mut self, was_buffered: bool) {
        let (jx, jy);
        if (self.hor_input as f64) * self.wall_normal < 0.0 && self.state == 5 {
            // Slide wall jump
            jx = JUMP_WALL_SLIDE_X_MULTIPLIER;
            jy = JUMP_WALL_SLIDE_Y;
        } else {
            // Regular wall jump
            jx = JUMP_WALL_REGULAR_X_MULTIPLIER;
            jy = JUMP_WALL_REGULAR_Y;
        }
        if was_buffered {
            self.last_jump_was_buffered = true;
        }
        self.state = 3; // Jumping
        self.applied_gravity = GRAVITY_JUMP;
        if self.xspeed * self.wall_normal < 0.0 {
            self.xspeed = 0.0;
        }
        if self.yspeed > 0.0 {
            self.yspeed = 0.0;
        }
        self.xspeed += jx * self.wall_normal;
        self.yspeed += jy;
        self.xpos += jx * self.wall_normal;
        self.ypos += jy;
        self.jump_buffer = -1;
        self.wall_buffer = -1;
        self.launch_pad_buffer = -1;
        self.jump_duration = 0;
    }

    /// Launch pad jump.
    fn lp_jump(&mut self) {
        self.floor_buffer = -1;
        self.wall_buffer = -1;
        self.jump_buffer = -1;
        self.launch_pad_buffer = -1;
        let mut boost_scalar = 2.0 * self.xlp_boost_normalized.abs() + 2.0;
        if boost_scalar == 2.0 {
            boost_scalar = JUMP_LAUNCH_PAD_BOOST_SCALAR;
        }
        self.xspeed += self.xlp_boost_normalized * boost_scalar * JUMP_LAUNCH_PAD_BOOST_FACTOR;
        self.yspeed += self.ylp_boost_normalized * boost_scalar * JUMP_LAUNCH_PAD_BOOST_FACTOR;
    }

    /// Ninja think: handles all state transitions and physics actions.
    pub fn think(&mut self) {
        // Reset buffer tracking
        self.last_jump_was_buffered = false;

        // Track state transitions
        if self.state != self.previous_state {
            self.state_change_frame = 0;
            self.previous_state = self.state;
        } else {
            self.state_change_frame += 1;
        }

        // New jump check
        let new_jump_check = self.jump_input != 0 && self.jump_input_old == 0;
        self.jump_input_old = self.jump_input;

        // Buffer management
        if self.launch_pad_buffer > -1 && self.launch_pad_buffer < 3 {
            self.launch_pad_buffer += 1;
        } else {
            self.launch_pad_buffer = -1;
        }
        let in_lp_buffer = self.launch_pad_buffer > -1 && self.launch_pad_buffer < 4;

        if self.jump_buffer > -1 && self.jump_buffer < 5 {
            self.jump_buffer += 1;
        } else {
            self.jump_buffer = -1;
        }
        let in_jump_buffer = self.jump_buffer > -1 && self.jump_buffer < 5;

        if self.wall_buffer > -1 && self.wall_buffer < 5 {
            self.wall_buffer += 1;
        } else {
            self.wall_buffer = -1;
        }
        let in_wall_buffer = self.wall_buffer > -1 && self.wall_buffer < 5;

        if self.floor_buffer > -1 && self.floor_buffer < 5 {
            self.floor_buffer += 1;
        } else {
            self.floor_buffer = -1;
        }
        let in_floor_buffer = self.floor_buffer > -1 && self.floor_buffer < 5;

        // Initiate jump buffer
        let mut jump_buffer_just_activated = false;
        if new_jump_check && self.airborn {
            self.jump_buffer = 0;
            jump_buffer_just_activated = true;
        }

        // Initiate wall buffer
        if self.walled {
            self.wall_buffer = 0;
        }

        // Initiate floor buffer
        if !self.airborn {
            self.floor_buffer = 0;
        }

        // Track if jump is useless
        if self.jump_input != 0 {
            let has_active_buffer = if jump_buffer_just_activated {
                true
            } else {
                let current_in_jump_buffer = self.jump_buffer > -1 && self.jump_buffer < 5;
                current_in_jump_buffer || in_floor_buffer || in_wall_buffer || in_lp_buffer
            };
            self.is_jump_useless = self.airborn && self.state != 3 && !has_active_buffer;
        } else {
            self.is_jump_useless = false;
        }

        // Special states
        if self.state == 6 || self.state == 9 {
            return;
        }
        if self.state == 7 {
            // Awaiting death -> dead (no ragdoll in native)
            self.state = 6;
            return;
        }
        if self.state == 8 {
            // Celebrating
            self.applied_drag = if self.airborn {
                DRAG_REGULAR
            } else {
                DRAG_SLOW
            };
            return;
        }

        // Grounded logic
        if !self.airborn {
            let xspeed_new = self.xspeed + GROUND_ACCEL * (self.hor_input as f64);
            if xspeed_new.abs() < MAX_HOR_SPEED {
                self.xspeed = xspeed_new;
            }
            if self.state > 2 {
                if self.xspeed * (self.hor_input as f64) <= 0.0 {
                    if self.state == 3 {
                        self.applied_gravity = GRAVITY_FALL;
                    }
                    self.state = 2;
                } else {
                    if self.state == 3 {
                        self.applied_gravity = GRAVITY_FALL;
                    }
                    self.state = 1;
                }
            }
            if !in_jump_buffer && !new_jump_check {
                // Not jumping
                if self.state == 2 {
                    let projection = (self.yspeed * self.floor_normalized_x
                        - self.xspeed * self.floor_normalized_y)
                        .abs();
                    if (self.hor_input as f64) * projection * self.xspeed > 0.0 {
                        self.state = 1;
                        return;
                    }
                    if projection < 0.1 && self.floor_normalized_x == 0.0 {
                        self.state = 0;
                        return;
                    }
                    if self.yspeed < 0.0 && self.floor_normalized_x != 0.0 {
                        // Up slope friction
                        let speed_scalar =
                            (self.xspeed * self.xspeed + self.yspeed * self.yspeed).sqrt();
                        let fric_force = (self.xspeed * (1.0 - FRICTION_GROUND)
                            * self.floor_normalized_y)
                            .abs();
                        let fric_force2 =
                            speed_scalar - fric_force * self.floor_normalized_y * self.floor_normalized_y;
                        self.xspeed = self.xspeed / speed_scalar * fric_force2;
                        self.yspeed = self.yspeed / speed_scalar * fric_force2;
                        return;
                    }
                    self.xspeed *= FRICTION_GROUND;
                    return;
                }
                if self.state == 1 {
                    let projection = (self.yspeed * self.floor_normalized_x
                        - self.xspeed * self.floor_normalized_y)
                        .abs();
                    if (self.hor_input as f64) * projection * self.xspeed > 0.0 {
                        if (self.hor_input as f64) * self.floor_normalized_x >= 0.0 {
                            return;
                        }
                        if xspeed_new.abs() < MAX_HOR_SPEED {
                            let boost = GROUND_ACCEL / 2.0 * (self.hor_input as f64);
                            let xboost =
                                boost * self.floor_normalized_y * self.floor_normalized_y;
                            let yboost = boost * self.floor_normalized_y
                                * (-self.floor_normalized_x);
                            self.xspeed += xboost;
                            self.yspeed += yboost;
                        }
                        return;
                    }
                    self.state = 2;
                } else {
                    // State 0
                    if self.hor_input != 0 {
                        self.state = 1;
                        return;
                    }
                    let projection = (self.yspeed * self.floor_normalized_x
                        - self.xspeed * self.floor_normalized_y)
                        .abs();
                    if projection < 0.1 {
                        self.xspeed *= FRICTION_GROUND_SLOW;
                        return;
                    }
                    self.state = 2;
                }
                return;
            }
            // Jumping from ground
            self.floor_jump(false);
            return;
        }

        // Airborne logic
        let xspeed_new = self.xspeed + AIR_ACCEL * (self.hor_input as f64);
        if xspeed_new.abs() < MAX_HOR_SPEED {
            self.xspeed = xspeed_new;
        }
        if self.state < 3 {
            self.state = 4; // Falling
            return;
        }
        if self.state == 3 {
            self.jump_duration += 1;
            if self.jump_input == 0 || self.jump_duration > MAX_JUMP_DURATION {
                self.applied_gravity = GRAVITY_FALL;
                self.state = 4;
                return;
            }
        }
        if in_jump_buffer || new_jump_check {
            if self.walled || in_wall_buffer {
                self.wall_jump(in_wall_buffer);
                return;
            }
            if in_floor_buffer {
                self.floor_jump(true);
                return;
            }
            if in_lp_buffer && new_jump_check {
                self.lp_jump();
                return;
            }
        }
        if !self.walled {
            if self.state == 5 {
                self.state = 4;
            }
        } else {
            if self.state == 5 {
                if (self.hor_input as f64) * self.wall_normal <= 0.0 {
                    self.yspeed *= FRICTION_WALL;
                } else {
                    self.state = 4;
                }
            } else {
                if self.yspeed > 0.0 && (self.hor_input as f64) * self.wall_normal < 0.0 {
                    if self.state == 3 {
                        self.applied_gravity = GRAVITY_FALL;
                    }
                    self.state = 5; // Wall sliding
                }
            }
        }
    }

    /// Compute valid action mask.
    pub fn get_valid_action_mask(&self) -> [bool; 6] {
        let mut mask = [true; 6];

        let has_active_buffer = (self.jump_buffer > -1 && self.jump_buffer < 5)
            || (self.floor_buffer > -1 && self.floor_buffer < 5)
            || (self.wall_buffer > -1 && self.wall_buffer < 5)
            || (self.launch_pad_buffer > -1 && self.launch_pad_buffer < 4);

        let jump_useless = if self.airborn && self.state != 3 {
            if self.jump_input != 0 {
                !has_active_buffer
            } else {
                false
            }
        } else {
            false
        };

        if jump_useless {
            mask[3] = false;
            mask[4] = false;
            mask[5] = false;
        }

        // Mask directional inputs into walls
        if self.walled {
            if !self.airborn {
                if self.wall_normal > 0.0 {
                    mask[1] = false;
                } else if self.wall_normal < 0.0 {
                    mask[2] = false;
                }
            } else {
                if self.state == 5 {
                    if self.wall_normal > 0.0 {
                        mask[1] = false;
                    } else if self.wall_normal < 0.0 {
                        mask[2] = false;
                    }
                } else {
                    let would_trigger_wall_slide = self.yspeed >= 0.0;
                    if self.wall_normal > 0.0 && !would_trigger_wall_slide {
                        mask[1] = false;
                    } else if self.wall_normal < 0.0 && !would_trigger_wall_slide {
                        mask[2] = false;
                    }
                }
            }
        }

        // Ensure at least one action is valid
        if !mask.iter().any(|&m| m) {
            mask[0] = true;
        }

        mask
    }

    /// Set ninja's state to celebrating (win).
    pub fn win(&mut self) {
        if self.state < 6 {
            if self.state == 3 {
                self.applied_gravity = GRAVITY_FALL;
            }
            self.state = 8;
        }
    }

    /// Set ninja's state to awaiting death (will transition to dead on next think).
    pub fn kill(
        &mut self,
        _death_type: i32,
        xpos: f64,
        ypos: f64,
        xspeed: f64,
        yspeed: f64,
        cause: Option<&str>,
    ) {
        if self.state < 6 {
            self.death_xpos = xpos;
            self.death_ypos = ypos;
            self.death_xspeed = xspeed;
            self.death_yspeed = yspeed;
            self.death_cause = cause.map(|s| s.to_string());
            if self.state == 3 {
                self.applied_gravity = GRAVITY_FALL;
            }
            self.state = 7; // AwaitingDeath
        }
    }

    /// Check if the ninja is a valid target for entity interactions.
    pub fn is_valid_target(&self) -> bool {
        self.state != 6 && self.state != 8 && self.state != 9
    }

    pub fn has_won(&self) -> bool {
        self.state == 8
    }

    pub fn has_died(&self) -> bool {
        self.state == 6 || self.state == 7
    }

    /// Record position at start of action for movement delta detection.
    pub fn record_action_start_position(&mut self) {
        self.xpos_at_action = self.xpos;
        self.ypos_at_action = self.ypos;
    }

    /// Take a snapshot of the ninja state for Python.
    pub fn snapshot(&self) -> NinjaSnapshot {
        NinjaSnapshot {
            xpos: self.xpos,
            ypos: self.ypos,
            xspeed: self.xspeed,
            yspeed: self.yspeed,
            xpos_old: self.xpos_old,
            ypos_old: self.ypos_old,
            xspeed_old: self.xspeed_old,
            yspeed_old: self.yspeed_old,
            state: self.state,
            airborn: self.airborn,
            airborn_old: self.airborn_old,
            walled: self.walled,
            wall_normal: self.wall_normal,
            terminal_impact: self.terminal_impact,
            jump_duration: self.jump_duration,
            jump_input_old: self.jump_input_old,
            jump_buffer: self.jump_buffer,
            floor_buffer: self.floor_buffer,
            wall_buffer: self.wall_buffer,
            launch_pad_buffer: self.launch_pad_buffer,
            last_jump_was_buffered: self.last_jump_was_buffered,
            floor_normalized_x: self.floor_normalized_x,
            floor_normalized_y: self.floor_normalized_y,
            ceiling_normalized_x: self.ceiling_normalized_x,
            ceiling_normalized_y: self.ceiling_normalized_y,
            floor_count: self.floor_count,
            wall_count: self.wall_count,
            ceiling_count: self.ceiling_count,
            applied_gravity: self.applied_gravity,
            applied_drag: self.applied_drag,
            applied_friction: self.applied_friction,
            hor_input: self.hor_input,
            jump_input: self.jump_input,
            frames_airborne: self.frames_airborne,
            state_change_frame: self.state_change_frame,
            previous_state: self.previous_state,
            consecutive_floor_frames: self.consecutive_floor_frames,
            consecutive_wall_frames: self.consecutive_wall_frames,
            has_won: self.has_won(),
            has_died: self.has_died(),
            death_cause: self.death_cause.clone(),
            gold_collected: self.gold_collected,
            doors_opened: self.doors_opened,
        }
    }

    /// Reset ninja to initial spawn state.
    pub fn reset_state(&mut self) {
        self.xpos = self.initial_xpos;
        self.ypos = self.initial_ypos;
        self.xspeed = 0.0;
        self.yspeed = 0.0;
        self.xspeed_old = 0.0;
        self.yspeed_old = 0.0;
        self.xpos_old = 0.0;
        self.ypos_old = 0.0;
        self.applied_gravity = GRAVITY_FALL;
        self.applied_drag = DRAG_REGULAR;
        self.applied_friction = FRICTION_GROUND;
        self.state = 0;
        self.airborn = false;
        self.airborn_old = false;
        self.walled = false;
        self.wall_normal = 0.0;
        self.jump_input_old = 0;
        self.jump_duration = 0;
        self.jump_buffer = -1;
        self.floor_buffer = -1;
        self.wall_buffer = -1;
        self.launch_pad_buffer = -1;
        self.is_jump_useless = false;
        self.last_jump_was_buffered = false;
        self.floor_normalized_x = 0.0;
        self.floor_normalized_y = -1.0;
        self.ceiling_normalized_x = 0.0;
        self.ceiling_normalized_y = 1.0;
        self.floor_count = 0;
        self.wall_count = 0;
        self.ceiling_count = 0;
        self.floor_normal_x = 0.0;
        self.floor_normal_y = 0.0;
        self.ceiling_normal_x = 0.0;
        self.ceiling_normal_y = 0.0;
        self.is_crushable = false;
        self.x_crush = 0.0;
        self.y_crush = 0.0;
        self.crush_len = 0.0;
        self.hor_input = 0;
        self.jump_input = 0;
        self.death_xpos = 0.0;
        self.death_ypos = 0.0;
        self.death_xspeed = 0.0;
        self.death_yspeed = 0.0;
        self.terminal_impact = false;
        self.death_cause = None;
        self.xpos_at_action = self.xpos;
        self.ypos_at_action = self.ypos;
        self.gold_collected = 0;
        self.doors_opened = 0;
        self.frames_airborne = 0;
        self.state_change_frame = 0;
        self.previous_state = 0;
        self.consecutive_floor_frames = 0;
        self.consecutive_wall_frames = 0;
        self.xlp_boost_normalized = 0.0;
        self.ylp_boost_normalized = 0.0;
        self.cached_entity_indices.clear();
    }

    /// Serialize ninja physics state to bytes for save/restore during transition enumeration.
    /// Includes all physics-relevant fields needed for stateless queries.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(512);

        out.extend_from_slice(&self.xpos.to_le_bytes());
        out.extend_from_slice(&self.ypos.to_le_bytes());
        out.extend_from_slice(&self.xspeed.to_le_bytes());
        out.extend_from_slice(&self.yspeed.to_le_bytes());
        out.extend_from_slice(&self.xpos_old.to_le_bytes());
        out.extend_from_slice(&self.ypos_old.to_le_bytes());
        out.extend_from_slice(&self.xspeed_old.to_le_bytes());
        out.extend_from_slice(&self.yspeed_old.to_le_bytes());
        out.extend_from_slice(&self.initial_xpos.to_le_bytes());
        out.extend_from_slice(&self.initial_ypos.to_le_bytes());
        out.extend_from_slice(&self.applied_gravity.to_le_bytes());
        out.extend_from_slice(&self.applied_drag.to_le_bytes());
        out.extend_from_slice(&self.applied_friction.to_le_bytes());
        out.push(self.state);
        out.push(self.airborn as u8);
        out.push(self.airborn_old as u8);
        out.push(self.walled as u8);
        out.extend_from_slice(&self.wall_normal.to_le_bytes());
        out.extend_from_slice(&self.jump_input_old.to_le_bytes());
        out.extend_from_slice(&self.jump_duration.to_le_bytes());
        out.extend_from_slice(&self.jump_buffer.to_le_bytes());
        out.extend_from_slice(&self.floor_buffer.to_le_bytes());
        out.extend_from_slice(&self.wall_buffer.to_le_bytes());
        out.extend_from_slice(&self.launch_pad_buffer.to_le_bytes());
        out.push(self.is_jump_useless as u8);
        out.push(self.last_jump_was_buffered as u8);
        out.extend_from_slice(&self.floor_normalized_x.to_le_bytes());
        out.extend_from_slice(&self.floor_normalized_y.to_le_bytes());
        out.extend_from_slice(&self.ceiling_normalized_x.to_le_bytes());
        out.extend_from_slice(&self.ceiling_normalized_y.to_le_bytes());
        out.extend_from_slice(&self.floor_count.to_le_bytes());
        out.extend_from_slice(&self.wall_count.to_le_bytes());
        out.extend_from_slice(&self.ceiling_count.to_le_bytes());
        out.extend_from_slice(&self.floor_normal_x.to_le_bytes());
        out.extend_from_slice(&self.floor_normal_y.to_le_bytes());
        out.extend_from_slice(&self.ceiling_normal_x.to_le_bytes());
        out.extend_from_slice(&self.ceiling_normal_y.to_le_bytes());
        out.push(self.is_crushable as u8);
        out.extend_from_slice(&self.x_crush.to_le_bytes());
        out.extend_from_slice(&self.y_crush.to_le_bytes());
        out.extend_from_slice(&self.crush_len.to_le_bytes());
        out.extend_from_slice(&self.hor_input.to_le_bytes());
        out.extend_from_slice(&self.jump_input.to_le_bytes());
        out.extend_from_slice(&self.death_xpos.to_le_bytes());
        out.extend_from_slice(&self.death_ypos.to_le_bytes());
        out.extend_from_slice(&self.death_xspeed.to_le_bytes());
        out.extend_from_slice(&self.death_yspeed.to_le_bytes());
        out.push(self.terminal_impact as u8);
        if let Some(ref s) = self.death_cause {
            out.push(1);
            let b = s.as_bytes();
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            out.extend_from_slice(b);
        } else {
            out.push(0);
        }
        out.extend_from_slice(&self.xpos_at_action.to_le_bytes());
        out.extend_from_slice(&self.ypos_at_action.to_le_bytes());
        out.extend_from_slice(&self.gold_collected.to_le_bytes());
        out.extend_from_slice(&self.doors_opened.to_le_bytes());
        out.extend_from_slice(&self.frames_airborne.to_le_bytes());
        out.extend_from_slice(&self.state_change_frame.to_le_bytes());
        out.push(self.previous_state);
        out.extend_from_slice(&self.consecutive_floor_frames.to_le_bytes());
        out.extend_from_slice(&self.consecutive_wall_frames.to_le_bytes());
        out.extend_from_slice(&self.xlp_boost_normalized.to_le_bytes());
        out.extend_from_slice(&self.ylp_boost_normalized.to_le_bytes());

        out
    }

    /// Deserialize ninja state from bytes (from to_bytes).
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut off = 0;
        let read_f64 = |d: &[u8], o: &mut usize| {
            let b: [u8; 8] = d[*o..*o + 8].try_into().unwrap();
            *o += 8;
            f64::from_le_bytes(b)
        };
        let read_u8 = |d: &[u8], o: &mut usize| {
            let v = d[*o];
            *o += 1;
            v
        };
        let read_i32 = |d: &[u8], o: &mut usize| {
            let b: [u8; 4] = d[*o..*o + 4].try_into().unwrap();
            *o += 4;
            i32::from_le_bytes(b)
        };
        let read_u32 = |d: &[u8], o: &mut usize| {
            let b: [u8; 4] = d[*o..*o + 4].try_into().unwrap();
            *o += 4;
            u32::from_le_bytes(b)
        };

        let xpos = read_f64(data, &mut off);
        let ypos = read_f64(data, &mut off);
        let xspeed = read_f64(data, &mut off);
        let yspeed = read_f64(data, &mut off);
        let xpos_old = read_f64(data, &mut off);
        let ypos_old = read_f64(data, &mut off);
        let xspeed_old = read_f64(data, &mut off);
        let yspeed_old = read_f64(data, &mut off);
        let initial_xpos = read_f64(data, &mut off);
        let initial_ypos = read_f64(data, &mut off);
        let applied_gravity = read_f64(data, &mut off);
        let applied_drag = read_f64(data, &mut off);
        let applied_friction = read_f64(data, &mut off);
        let state = read_u8(data, &mut off);
        let airborn = read_u8(data, &mut off) != 0;
        let airborn_old = read_u8(data, &mut off) != 0;
        let walled = read_u8(data, &mut off) != 0;
        let wall_normal = read_f64(data, &mut off);
        let jump_input_old = read_i32(data, &mut off);
        let jump_duration = read_i32(data, &mut off);
        let jump_buffer = read_i32(data, &mut off);
        let floor_buffer = read_i32(data, &mut off);
        let wall_buffer = read_i32(data, &mut off);
        let launch_pad_buffer = read_i32(data, &mut off);
        let is_jump_useless = read_u8(data, &mut off) != 0;
        let last_jump_was_buffered = read_u8(data, &mut off) != 0;
        let floor_normalized_x = read_f64(data, &mut off);
        let floor_normalized_y = read_f64(data, &mut off);
        let ceiling_normalized_x = read_f64(data, &mut off);
        let ceiling_normalized_y = read_f64(data, &mut off);
        let floor_count = read_u32(data, &mut off);
        let wall_count = read_u32(data, &mut off);
        let ceiling_count = read_u32(data, &mut off);
        let floor_normal_x = read_f64(data, &mut off);
        let floor_normal_y = read_f64(data, &mut off);
        let ceiling_normal_x = read_f64(data, &mut off);
        let ceiling_normal_y = read_f64(data, &mut off);
        let is_crushable = read_u8(data, &mut off) != 0;
        let x_crush = read_f64(data, &mut off);
        let y_crush = read_f64(data, &mut off);
        let crush_len = read_f64(data, &mut off);
        let hor_input = read_i32(data, &mut off);
        let jump_input = read_i32(data, &mut off);
        let death_xpos = read_f64(data, &mut off);
        let death_ypos = read_f64(data, &mut off);
        let death_xspeed = read_f64(data, &mut off);
        let death_yspeed = read_f64(data, &mut off);
        let terminal_impact = read_u8(data, &mut off) != 0;
        let death_cause = if read_u8(data, &mut off) != 0 {
            let len = read_u32(data, &mut off) as usize;
            let s = String::from_utf8_lossy(&data[off..off + len]).into_owned();
            off += len;
            Some(s)
        } else {
            None
        };
        let xpos_at_action = read_f64(data, &mut off);
        let ypos_at_action = read_f64(data, &mut off);
        let gold_collected = read_u32(data, &mut off);
        let doors_opened = read_u32(data, &mut off);
        let frames_airborne = read_u32(data, &mut off);
        let state_change_frame = read_u32(data, &mut off);
        let previous_state = read_u8(data, &mut off);
        let consecutive_floor_frames = read_u32(data, &mut off);
        let consecutive_wall_frames = read_u32(data, &mut off);
        let xlp_boost_normalized = read_f64(data, &mut off);
        let ylp_boost_normalized = read_f64(data, &mut off);

        Ninja {
            xpos,
            ypos,
            xspeed,
            yspeed,
            xpos_old,
            ypos_old,
            xspeed_old,
            yspeed_old,
            initial_xpos,
            initial_ypos,
            applied_gravity,
            applied_drag,
            applied_friction,
            state,
            airborn,
            airborn_old,
            walled,
            wall_normal,
            jump_input_old,
            jump_duration,
            jump_buffer,
            floor_buffer,
            wall_buffer,
            launch_pad_buffer,
            is_jump_useless,
            last_jump_was_buffered,
            floor_normalized_x,
            floor_normalized_y,
            ceiling_normalized_x,
            ceiling_normalized_y,
            floor_count,
            wall_count,
            ceiling_count,
            floor_normal_x,
            floor_normal_y,
            ceiling_normal_x,
            ceiling_normal_y,
            is_crushable,
            x_crush,
            y_crush,
            crush_len,
            hor_input,
            jump_input,
            death_xpos,
            death_ypos,
            death_xspeed,
            death_yspeed,
            terminal_impact,
            death_cause,
            xpos_at_action,
            ypos_at_action,
            gold_collected,
            doors_opened,
            frames_airborne,
            state_change_frame,
            previous_state,
            consecutive_floor_frames,
            consecutive_wall_frames,
            xlp_boost_normalized,
            ylp_boost_normalized,
            cached_entity_indices: Vec::new(),
        }
    }

    /// Inject position and velocity for transition enumeration.
    /// Mirrors Python _inject_state() for stateless physics queries.
    pub fn inject_enumeration_state(&mut self, x: f64, y: f64, vx: f64, vy: f64) {
        self.xpos = x;
        self.ypos = y;
        self.xspeed = vx;
        self.yspeed = vy;
        self.jump_duration = 0;

        self.xpos_old = x - vx;
        self.ypos_old = y - vy;
        self.xspeed_old = vx;
        self.yspeed_old = vy;

        self.state = 4; // Falling
        self.applied_gravity = GRAVITY_FALL;
        self.airborn = true;

        self.floor_count = 0;
        self.wall_count = 0;
        self.ceiling_count = 0;
        self.walled = false;
        self.wall_normal = 0.0;

        self.jump_buffer = -1;
        self.floor_buffer = -1;
        self.wall_buffer = -1;
        self.launch_pad_buffer = -1;

        self.is_crushable = false;
        self.x_crush = 0.0;
        self.y_crush = 0.0;
        self.crush_len = 0.0;

        self.terminal_impact = false;
        self.death_cause = None;
    }

    /// Inject full 5D state for value iteration (position, velocity, jump_duration).
    /// Inject grounded floor state for floor jump enumeration.
    ///
    /// Sets state=1 (Running) with active floor contact so that on the very
    /// next tick with jump_input=1, think() calls floor_jump() with the correct
    /// floor normal.  This fixes the airborne-injection bug where
    /// inject_value_iteration_state placed the ninja in state=4 (Falling),
    /// causing floor_jump() to miss the floor-normal-dependent impulse branches
    /// (slope jumps) and making running-edge simulation return after 1 frame.
    ///
    /// floor_normalized_x/y: outward floor normal (unit vector, away from solid).
    /// For flat horizontal ground: floor_normalized_x=0, floor_normalized_y=-1.
    pub fn inject_value_iteration_grounded(
        &mut self,
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        floor_normalized_x: f64,
        floor_normalized_y: f64,
    ) {
        self.xpos = x;
        self.ypos = y;
        self.xspeed = vx;
        self.yspeed = vy;
        self.jump_duration = 0;
        self.applied_gravity = GRAVITY_FALL;
        self.applied_drag = DRAG_REGULAR;

        self.xpos_old = x - vx;
        self.ypos_old = y - vy;
        self.xspeed_old = vx;
        self.yspeed_old = vy;

        self.state = 1; // Running
        self.airborn = false;
        self.airborn_old = false;
        self.walled = false;
        self.wall_normal = 0.0;

        self.floor_count = 1;
        self.floor_normalized_x = floor_normalized_x;
        self.floor_normalized_y = floor_normalized_y;
        self.wall_count = 0;
        self.ceiling_count = 0;
        self.floor_normal_x = floor_normalized_x;
        self.floor_normal_y = floor_normalized_y;
        self.ceiling_normal_x = 0.0;
        self.ceiling_normal_y = 0.0;

        // Active floor buffer so jump fires immediately on the first tick with
        // jump_input=1 (floor_buffer=0 satisfies in_floor_buffer check).
        self.jump_buffer = -1;
        self.floor_buffer = 0;
        self.wall_buffer = -1;
        self.launch_pad_buffer = -1;

        self.is_crushable = false;
        self.x_crush = 0.0;
        self.y_crush = 0.0;
        self.crush_len = 0.0;

        self.terminal_impact = false;
        self.death_cause = None;
    }

    /// Inject wall-slide state for slide-wall-jump enumeration.
    ///
    /// Sets state=5 (WallSliding) with wall contact so that on the next tick
    /// with jump_input=1 and h_input pointing *toward* the wall
    /// (h_input = -sign(wall_normal)), think() calls wall_jump() and satisfies
    /// the `hor_input * wall_normal < 0 && state == 5` condition, producing
    /// the weaker slide impulse (vx = ±0.667, vy = −1.0).
    ///
    /// wall_normal: +1 if the wall's outward normal points right (left-hand wall),
    ///              -1 if it points left (right-hand wall).
    pub fn inject_value_iteration_wall_slide(
        &mut self,
        x: f64,
        y: f64,
        vy: f64,
        wall_normal: f64,
    ) {
        self.xpos = x;
        self.ypos = y;
        self.xspeed = 0.0;
        self.yspeed = vy;
        self.jump_duration = 0;
        self.applied_gravity = GRAVITY_FALL;
        self.applied_drag = DRAG_REGULAR;

        self.xpos_old = x;
        self.ypos_old = y - vy;
        self.xspeed_old = 0.0;
        self.yspeed_old = vy;

        self.state = 5; // WallSliding
        self.airborn = true;
        self.airborn_old = true;
        self.walled = true;
        self.wall_normal = wall_normal;

        self.floor_count = 0;
        self.floor_normalized_x = 0.0;
        self.floor_normalized_y = 0.0;
        self.wall_count = 1;
        self.ceiling_count = 0;
        self.floor_normal_x = 0.0;
        self.floor_normal_y = 0.0;
        self.ceiling_normal_x = 0.0;
        self.ceiling_normal_y = 0.0;

        // wall_buffer is reset to 0 in think() because walled=true, making
        // in_wall_buffer=true so wall_jump() triggers on the very first tick
        // that sees jump_input=1.
        self.jump_buffer = -1;
        self.floor_buffer = -1;
        self.wall_buffer = -1;
        self.launch_pad_buffer = -1;

        self.is_crushable = false;
        self.x_crush = 0.0;
        self.y_crush = 0.0;
        self.crush_len = 0.0;

        self.terminal_impact = false;
        self.death_cause = None;
    }

    /// Unlike inject_enumeration_state, sets jump_duration and gravity based on
    /// whether ninja is actively jumping (reduced gravity) or falling.
    pub fn inject_value_iteration_state(
        &mut self,
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        jump_duration: i32,
    ) {
        self.xpos = x;
        self.ypos = y;
        self.xspeed = vx;
        self.yspeed = vy;
        self.jump_duration = jump_duration;

        self.xpos_old = x - vx;
        self.ypos_old = y - vy;
        self.xspeed_old = vx;
        self.yspeed_old = vy;

        if jump_duration > 0 && jump_duration < MAX_JUMP_DURATION {
            self.state = 3; // Jumping
            self.applied_gravity = GRAVITY_JUMP;
        } else {
            self.state = 4; // Falling
            self.applied_gravity = GRAVITY_FALL;
        }
        self.airborn = true;

        self.floor_count = 0;
        self.wall_count = 0;
        self.ceiling_count = 0;
        self.walled = false;
        self.wall_normal = 0.0;

        self.jump_buffer = -1;
        self.floor_buffer = -1;
        self.wall_buffer = -1;
        self.launch_pad_buffer = -1;

        self.is_crushable = false;
        self.x_crush = 0.0;
        self.y_crush = 0.0;
        self.crush_len = 0.0;

        self.terminal_impact = false;
        self.death_cause = None;
    }
}

/// Trait for entity collision handling.
/// The simulator implements this to dispatch logical collisions to entity instances.
pub trait EntityCollisionHandler {
    fn logical_collision(&mut self, entity_index: usize, ninja: &mut Ninja);
}
