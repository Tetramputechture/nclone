"""Memory profiling utility for environment memory analysis.

Moved verbatim out of ``test_environment.py`` so the entry point stays thin.
Enabled with ``--profile-memory``.
"""

import gc
import os
import tracemalloc

import numpy as np
import psutil


class MemoryProfiler:
    """Memory profiling utility for environment memory analysis."""

    def __init__(self, snapshot_interval=100):
        self.snapshot_interval = snapshot_interval
        self.process = psutil.Process(os.getpid())
        self.snapshots = []
        self.baseline_snapshot = None
        self.frame_count = 0

    def start(self):
        """Start memory profiling."""
        tracemalloc.start()
        gc.collect()
        self.baseline_snapshot = tracemalloc.take_snapshot()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        print("\n" + "=" * 60)
        print("MEMORY PROFILING ACTIVE")
        print("=" * 60)
        print(f"Baseline memory usage: {self.baseline_memory:.2f} MB")
        print(f"Snapshot interval: {self.snapshot_interval} frames")
        print("=" * 60 + "\n")

    def take_snapshot(self, frame_number, obs=None):
        """Take a memory snapshot and analyze."""
        gc.collect()  # Force garbage collection for accurate measurement
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        snapshot = tracemalloc.take_snapshot()

        # Calculate observation sizes if provided
        obs_sizes = {}
        if obs is not None:
            try:
                if hasattr(obs, "items"):
                    for key, value in obs.items():
                        if isinstance(value, np.ndarray):
                            obs_sizes[key] = {
                                "shape": value.shape,
                                "dtype": str(value.dtype),
                                "size_mb": value.nbytes / 1024 / 1024,
                            }
            except Exception:
                pass

        snapshot_data = {
            "frame": frame_number,
            "memory_mb": current_memory,
            "memory_delta_mb": current_memory - self.baseline_memory,
            "snapshot": snapshot,
            "observation_sizes": obs_sizes,
        }

        self.snapshots.append(snapshot_data)

        # Print snapshot info
        print(f"\n[Frame {frame_number}] Memory Snapshot:")
        print(f"  Total Memory: {current_memory:.2f} MB")
        print(f"  Delta from baseline: {snapshot_data['memory_delta_mb']:+.2f} MB")

        if obs_sizes:
            print("  Observation sizes:")
            total_obs_size = 0
            for key, info in obs_sizes.items():
                print(
                    f"    {key}: {info['shape']} ({info['dtype']}) = {info['size_mb']:.3f} MB"
                )
                total_obs_size += info["size_mb"]
            print(f"  Total observation size: {total_obs_size:.3f} MB")

        # Show top memory allocations
        top_stats = snapshot.statistics("lineno")[:5]
        print("  Top 5 memory allocations:")
        for stat in top_stats:
            print(f"    {stat}")

        return snapshot_data

    def compare_snapshots(self, snapshot_idx1=0, snapshot_idx2=-1):
        """Compare two snapshots to identify memory growth."""
        if len(self.snapshots) < 2:
            print("Not enough snapshots for comparison")
            return

        snap1 = self.snapshots[snapshot_idx1]
        snap2 = self.snapshots[snapshot_idx2]

        print("\n" + "=" * 60)
        print(f"MEMORY COMPARISON: Frame {snap1['frame']} vs Frame {snap2['frame']}")
        print("=" * 60)
        print(f"Memory growth: {snap2['memory_mb'] - snap1['memory_mb']:.2f} MB")

        # Compare tracemalloc snapshots
        top_stats = snap2["snapshot"].compare_to(snap1["snapshot"], "lineno")[:10]
        print("\nTop 10 memory growth sources:")
        for stat in top_stats:
            print(f"  {stat}")
        print("=" * 60)

    def finalize(self):
        """Generate final memory report."""
        print("\n" + "=" * 60)
        print("MEMORY PROFILING REPORT")
        print("=" * 60)

        if self.snapshots:
            first_snap = self.snapshots[0]
            last_snap = self.snapshots[-1]
            total_growth = last_snap["memory_mb"] - first_snap["memory_mb"]
            frames_elapsed = last_snap["frame"] - first_snap["frame"]

            print(f"Total frames profiled: {frames_elapsed}")
            print(f"Starting memory: {first_snap['memory_mb']:.2f} MB")
            print(f"Ending memory: {last_snap['memory_mb']:.2f} MB")
            print(f"Total growth: {total_growth:.2f} MB")
            if frames_elapsed > 0:
                print(f"Growth per frame: {total_growth / frames_elapsed:.4f} MB")

            peak_snap = max(self.snapshots, key=lambda s: s["memory_mb"])
            print(
                f"\nPeak memory: {peak_snap['memory_mb']:.2f} MB at frame {peak_snap['frame']}"
            )

            if len(self.snapshots) >= 2:
                print("\nMemory leak analysis (comparing first and last snapshots):")
                top_stats = last_snap["snapshot"].compare_to(
                    first_snap["snapshot"], "lineno"
                )[:15]
                for i, stat in enumerate(top_stats, 1):
                    print(f"  {i}. {stat}")

        print("=" * 60)

        with open("memory_profiling_report.txt", "w") as f:
            f.write("DETAILED MEMORY PROFILING REPORT\n")
            f.write("=" * 80 + "\n\n")

            if self.snapshots:
                f.write(f"Total snapshots: {len(self.snapshots)}\n")
                f.write(f"Baseline memory: {self.baseline_memory:.2f} MB\n\n")

                for snap_data in self.snapshots:
                    f.write(f"\nFrame {snap_data['frame']}:\n")
                    f.write(f"  Memory: {snap_data['memory_mb']:.2f} MB\n")
                    f.write(f"  Delta: {snap_data['memory_delta_mb']:+.2f} MB\n")

                    if snap_data["observation_sizes"]:
                        f.write("  Observations:\n")
                        for key, info in snap_data["observation_sizes"].items():
                            f.write(
                                f"    {key}: {info['shape']} ({info['dtype']}) = {info['size_mb']:.3f} MB\n"
                            )

                    f.write("  Top allocations:\n")
                    top_stats = snap_data["snapshot"].statistics("lineno")[:10]
                    for stat in top_stats:
                        f.write(f"    {stat}\n")

            if len(self.snapshots) >= 2:
                f.write("\n\nMEMORY LEAK ANALYSIS\n")
                f.write("=" * 80 + "\n")
                first_snap = self.snapshots[0]["snapshot"]
                last_snap = self.snapshots[-1]["snapshot"]
                top_stats = last_snap.compare_to(first_snap, "lineno")[:30]
                for stat in top_stats:
                    f.write(f"{stat}\n")

        print("Detailed report saved to memory_profiling_report.txt")
        tracemalloc.stop()
