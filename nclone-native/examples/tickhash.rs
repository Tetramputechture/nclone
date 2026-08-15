//! Trajectory fingerprint for the native simulator.
//!
//! Runs deterministic pseudo-random action rollouts on every map in a directory
//! and prints a hash of the full per-frame state (ninja pose + every entity's
//! snapshot). Any change that alters simulation behaviour changes the hash, so
//! this is the gate an optimization has to pass.
//!
//! Usage: cargo run --release --example tickhash -- <maps_dir> [frames] [seeds]

use std::fs;

use nclone_native::simulator::Simulator;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn action(&mut self) -> (i32, i32) {
        let a = self.next();
        (((a % 3) as i32) - 1, ((a >> 8) % 4 == 0) as i32)
    }
}

/// FNV-1a over raw bytes; f64s go in as their exact bit patterns.
struct Hasher(u64);
impl Hasher {
    fn new() -> Self {
        Hasher(0xcbf29ce484222325)
    }
    fn byte(&mut self, b: u8) {
        self.0 ^= b as u64;
        self.0 = self.0.wrapping_mul(0x100000001b3);
    }
    fn u64(&mut self, v: u64) {
        for i in 0..8 {
            self.byte((v >> (i * 8)) as u8);
        }
    }
    fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }
    fn bool(&mut self, v: bool) {
        self.byte(v as u8);
    }
}

/// Run one seeded rollout and fold every frame's full state into `h`.
///
/// `mode` selects the frame variant: 0 = `tick`, 1 = `tick_physics_only`.
fn roll(map_data: &[i32], frames: usize, seed: usize, mode: usize, h: &mut Hasher) {
    let mut sim = Simulator::load(map_data);
    let mut rng = Rng(0x243F6A8885A308D3u64.wrapping_add(seed as u64 * 0x9E3779B97F4A7C15));
    for i in 0..frames {
        let (hor, j) = rng.action();
        if mode == 0 {
            sim.tick(hor, j);
        } else {
            sim.tick_physics_only(hor, j);
        }
        let st = sim.get_state_dynamic();
        let n = &st.ninja;
        h.f64(n.xpos);
        h.f64(n.ypos);
        h.f64(n.xspeed);
        h.f64(n.yspeed);
        h.byte(n.state);
        h.bool(n.airborn);
        h.bool(n.walled);
        h.f64(n.wall_normal);
        h.u64(n.jump_duration as u64);
        h.u64(n.floor_count as u64);
        h.u64(n.wall_count as u64);
        h.u64(n.ceiling_count as u64);
        h.f64(n.floor_normalized_x);
        h.f64(n.floor_normalized_y);
        h.f64(n.ceiling_normalized_x);
        h.f64(n.ceiling_normalized_y);
        h.bool(n.has_won);
        h.bool(n.has_died);
        h.u64(n.gold_collected as u64);
        h.u64(n.doors_opened as u64);
        h.u64(n.frames_airborne as u64);
        h.u64(n.jump_buffer as u64);
        h.u64(n.floor_buffer as u64);
        h.u64(n.wall_buffer as u64);
        h.u64(n.launch_pad_buffer as u64);
        h.f64(n.applied_gravity);
        h.f64(n.applied_drag);
        h.f64(n.applied_friction);
        for m in st.action_mask.iter() {
            h.bool(*m);
        }
        for e in &st.entities {
            h.byte(e.entity_type);
            h.f64(e.x);
            h.f64(e.y);
            h.bool(e.active);
            h.byte(e.state);
            h.f64(e.radius);
            h.byte(e.orientation);
            h.f64(e.door_x);
            h.f64(e.door_y);
            h.byte(e.door_orientation);
            h.bool(e.door_closed);
        }
        if i % 600 == 599 {
            sim.fast_reset();
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| "maps".to_string());
    let frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let seeds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    let mut files: Vec<_> = fs::read_dir(&dir)
        .expect("maps dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "txt").unwrap_or(false))
        .collect();
    files.sort();

    let mut grand = Hasher::new();
    for path in &files {
        let text = fs::read_to_string(path).unwrap();
        let map_data: Vec<i32> = text
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();

        let mut h = Hasher::new();
        // Mode 0 exercises the full frame, mode 1 the physics-only frame used by
        // the enumeration / value-iteration paths. Both take the open-air
        // collision skip, so both are fingerprinted.
        for mode in 0..2usize {
            for seed in 0..seeds {
                roll(&map_data, frames, seed, mode, &mut h);
            }
        }
        println!(
            "{:38} {:016x}",
            path.file_name().unwrap().to_string_lossy(),
            h.0
        );
        grand.u64(h.0);
    }
    println!("\nGRAND {:016x}", grand.0);
}
