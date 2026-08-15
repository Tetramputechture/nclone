//! Standalone replay of the lattice edge-enumeration kernel, for profiling.
//!
//! Consumes a blob written by `scratchpad/dump_enum.py` (the exact arguments a
//! `nclone_native.enumerate_arcs` call would receive for one level) and runs
//! `enumerate_arcs_parallel` on it, so the kernel can be profiled without a
//! Python process in the way.
//!
//! Usage: cargo run --release --example enumbench -- <blob> [n_threads] [reps]

use std::fs;
use std::time::Instant;

use nclone_native::enumerate::{enumerate_arcs_parallel, NodeMapper, NodeSpec, SweepParams};

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn tag_len(&mut self, want: u8) -> usize {
        let kind = self.buf[self.pos];
        assert_eq!(kind, want, "blob array kind mismatch at {}", self.pos);
        let n = u32::from_le_bytes(self.buf[self.pos + 1..self.pos + 5].try_into().unwrap());
        self.pos += 5;
        n as usize
    }
    fn i32s(&mut self) -> Vec<i32> {
        let n = self.tag_len(b'i');
        let out = (0..n)
            .map(|k| {
                let o = self.pos + k * 4;
                i32::from_le_bytes(self.buf[o..o + 4].try_into().unwrap())
            })
            .collect();
        self.pos += n * 4;
        out
    }
    fn f64s(&mut self) -> Vec<f64> {
        let n = self.tag_len(b'd');
        let out = (0..n)
            .map(|k| {
                let o = self.pos + k * 8;
                f64::from_le_bytes(self.buf[o..o + 8].try_into().unwrap())
            })
            .collect();
        self.pos += n * 8;
        out
    }
    fn bools(&mut self) -> Vec<bool> {
        let n = self.tag_len(b'b');
        let out = (0..n).map(|k| self.buf[self.pos + k] != 0).collect();
        self.pos += n;
        out
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: enumbench <blob> [threads] [reps]");
    let n_threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let buf = fs::read(path).expect("blob");
    let mut r = Reader { buf: &buf, pos: 0 };

    let map_data = r.i32s();
    let node_x = r.f64s();
    let node_y = r.f64s();
    let node_vx0 = r.f64s();
    let node_vy0 = r.f64s();
    let node_surf = r.i32s();
    let node_edge = r.bools();
    let node_pt = r.i32s();
    let node_vb = r.i32s();
    let run_durations = r.i32s();
    let jump_holds = r.i32s();
    let h_inputs = r.i32s();
    let post_release = r.i32s();
    let max_arc_frames = r.i32s()[0];
    let max_jump_duration = r.i32s()[0];
    let tick_mines = r.bools()[0];
    let goals_flat = r.f64s();
    let sp_x = r.f64s();
    let sp_y = r.f64s();
    let sp_type = r.i32s();
    let floor_c = r.f64s();
    let wall_c = r.f64s();
    let slide_c = r.f64s();
    let launch_c = r.f64s();
    let cell = r.f64s()[0];
    let max_vel_bins = r.i32s()[0];
    let sp_tan_x = r.f64s();
    let sp_tan_y = r.f64s();
    let tangential = r.bools()[0];

    let nodes: Vec<NodeSpec> = (0..node_x.len())
        .map(|i| NodeSpec {
            src_node: node_pt[i] * max_vel_bins + node_vb[i],
            x: node_x[i],
            y: node_y[i],
            vx0: node_vx0[i],
            vy0: node_vy0[i],
            surface_type: node_surf[i] as u8,
            is_edge: node_edge[i],
        })
        .collect();
    let goals: Vec<(f64, f64, f64)> = goals_flat
        .chunks_exact(3)
        .map(|c| (c[0], c[1], c[2]))
        .collect();
    let to_f32 = |v: Vec<f64>| -> Vec<f32> { v.into_iter().map(|c| c as f32).collect() };
    let mapper = NodeMapper::new(
        &sp_x,
        &sp_y,
        &sp_type,
        &sp_tan_x,
        &sp_tan_y,
        tangential,
        to_f32(floor_c),
        to_f32(wall_c),
        to_f32(slide_c),
        to_f32(launch_c),
        cell,
        max_vel_bins,
    );
    let sw = SweepParams {
        run_durations: &run_durations,
        jump_holds: &jump_holds,
        h_inputs: &h_inputs,
        post_release: &post_release,
        max_arc_frames,
        max_jump_duration,
        tick_mines,
        goals: &goals,
    };

    println!(
        "nodes={} threads={} max_arc_frames={} tangential={}",
        nodes.len(),
        n_threads,
        max_arc_frames,
        tangential
    );
    for rep in 0..reps {
        let t0 = Instant::now();
        let out = enumerate_arcs_parallel(&map_data, &nodes, &sw, &mapper, n_threads, 0.0, 0.0);
        let el = t0.elapsed().as_secs_f64();
        // Fingerprint the arc set so an optimization can be proven to change nothing.
        let mut h: u64 = 0xcbf29ce484222325;
        for a in &out.arcs {
            for v in [
                a.src_node,
                a.dst_node,
                a.frames,
                a.jhold,
                a.h_during,
                a.h_after,
                a.clipped as i32,
            ] {
                h ^= v as u32 as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        let mut hg: u64 = 0xcbf29ce484222325;
        for k in 0..out.gc_node.len() {
            for v in [out.gc_node[k] as u64, out.gc_goal[k] as u64, out.gc_frame[k].to_bits()] {
                hg ^= v;
                hg = hg.wrapping_mul(0x100000001b3);
            }
        }
        let tt: u64 = out.ticks.iter().sum();
        println!(
            "rep{} {:8.2}s  arcs={} gc={} arc_hash={:016x} gc_hash={:016x}",
            rep,
            el,
            out.arcs.len(),
            out.gc_node.len(),
            h,
            hg
        );
        println!(
            "     ticks total={:.1}M  run={:.1}M hold={:.1}M fall={:.1}M   ns/tick={:.0}",
            tt as f64 / 1e6,
            out.ticks[0] as f64 / 1e6,
            out.ticks[1] as f64 / 1e6,
            out.ticks[2] as f64 / 1e6,
            el * 1e9 * n_threads as f64 / tt as f64,
        );
        println!(
            "     arcs  run={} hold={} fall={}   fall_capped={} ({:.1}%)  frames/fall={:.1}",
            out.arc_count[0],
            out.arc_count[1],
            out.arc_count[2],
            out.arc_capped[2],
            100.0 * out.arc_capped[2] as f64 / out.arc_count[2].max(1) as f64,
            out.ticks[2] as f64 / out.arc_count[2].max(1) as f64,
        );
        println!(
            "     depen calls={:.1}M iters={:.1}M  iters/call={:.2}  calls/tick={:.2}",
            out.depen_calls as f64 / 1e6,
            out.depen_iters as f64 / 1e6,
            out.depen_iters as f64 / out.depen_calls.max(1) as f64,
            out.depen_calls as f64 / tt.max(1) as f64,
        );
    }
}
