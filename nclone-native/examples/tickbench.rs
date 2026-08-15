//! Standalone tick benchmark for profiling the native simulator.
//!
//! Usage: cargo run --release --example tickbench -- <maps_dir> [iters]
//! Map files are whitespace-separated integer map_data dumps.

use std::fs;
use std::time::Instant;

use nclone_native::simulator::Simulator;

/// Deterministic xorshift so runs are comparable across machines.
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "maps".to_string());
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let filter = args.get(3).cloned();

    let mut files: Vec<_> = fs::read_dir(&dir)
        .expect("maps dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "txt").unwrap_or(false))
        .collect();
    files.sort();
    if let Some(f) = &filter {
        files.retain(|p| p.to_string_lossy().contains(f.as_str()));
    }

    let mut grand_ticks = 0u64;
    let mut grand_ns = 0u128;

    for path in &files {
        let text = fs::read_to_string(path).unwrap();
        let map_data: Vec<i32> = text
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();

        let mut sim = Simulator::load(&map_data);
        let n_ent = sim.get_state_dynamic().entities.len();
        let mut rng = Rng(0x243F6A8885A308D3);

        // warmup
        for i in 0..2000 {
            let (h, j) = rng.action();
            sim.tick(h, j);
            if i % 500 == 499 {
                sim.fast_reset();
            }
        }

        let mut rng = Rng(0x243F6A8885A308D3);
        let t0 = Instant::now();
        for i in 0..iters {
            let (h, j) = rng.action();
            sim.tick(h, j);
            if i % 600 == 599 {
                sim.fast_reset();
            }
        }
        let el = t0.elapsed();
        let per = el.as_nanos() as f64 / iters as f64;
        grand_ticks += iters as u64;
        grand_ns += el.as_nanos();
        println!(
            "{:38} ents={:4}  {:8.1} ns/tick  {:9.0} ticks/s",
            path.file_name().unwrap().to_string_lossy(),
            n_ent,
            per,
            1e9 / per
        );
    }

    let per = grand_ns as f64 / grand_ticks as f64;
    println!("\nTOTAL {:.1} ns/tick  ({:.0} ticks/s)", per, 1e9 / per);
}
