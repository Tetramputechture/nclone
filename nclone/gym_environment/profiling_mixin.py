"""Performance profiling mixin for environment bottleneck analysis."""

import time
from collections import defaultdict, deque
from typing import Dict, Optional
import numpy as np


class ProfilingMixin:
    """Mixin for detailed environment performance profiling."""
    
    def _init_profiling(self, profile_enabled: bool = False):
        """Initialize profiling system.
        
        Args:
            profile_enabled: Enable detailed profiling (adds ~5% overhead)
        """
        self.profile_enabled = profile_enabled

        if profile_enabled:
            # Enable feature-level profiling in reachability feature computation
            import nclone.graph.reachability.feature_computation as fc

            fc.ENABLE_FEATURE_PROFILING = True

        if not profile_enabled:
            return

        # Timing storage (rolling window of last 1000 samples)
        self._timing_data = defaultdict(lambda: deque(maxlen=1000))
        self._timing_counts = defaultdict(int)
        
        # Step breakdown tracking
        self._step_start_time = 0.0
        self._last_report_time = time.time()
        self._report_interval = 10.0  # Report every 10 seconds
    
    def _profile_start(self, key: str) -> float:
        """Start timing a section.
        
        Args:
            key: Identifier for this timing section
            
        Returns:
            Start time (pass to _profile_end)
        """
        if not hasattr(self, 'profile_enabled') or not self.profile_enabled:
            return 0.0
        return time.perf_counter()
    
    def _profile_end(self, key: str, start_time: float):
        """End timing a section and record duration.
        
        Args:
            key: Identifier for this timing section
            start_time: Start time from _profile_start
        """
        if not hasattr(self, 'profile_enabled') or not self.profile_enabled or start_time == 0.0:
            return
        
        duration = (time.perf_counter() - start_time) * 1000  # Convert to ms
        self._timing_data[key].append(duration)
        self._timing_counts[key] += 1
        
        # Check if it's time to report
        current_time = time.time()
        if current_time - self._last_report_time >= self._report_interval:
            self._print_profiling_report()
            self._last_report_time = current_time
    
    def _print_profiling_report(self):
        """Print profiling statistics."""
        if not self.profile_enabled or not self._timing_data:
            return
        
        print("\n" + "="*80)
        print("ENVIRONMENT PROFILING REPORT")
        print("="*80)
        
        # Calculate total step time
        step_times = self._timing_data.get("step_total", [])
        if len(step_times) > 0:
            total_avg = np.mean(list(step_times))
            print(f"\nTotal step time: {total_avg:.2f} ms/step ({1000/total_avg:.1f} steps/s)")
        else:
            total_avg = 1.0  # Fallback
        
        # Breakdown by component with hierarchical structure
        print("\nComponent breakdown (avg ± std | min → max | % of total):")
        print("-" * 80)
        
        # Hierarchical component structure
        components = [
            ("step_total", "TOTAL STEP", 0),
            ("physics_total", "  └─ Physics (tick+sync)", 1),
            ("sync_get_state", "       ├─ sync: get_state FFI", 2),
            ("sync_build_tile", "       └─ sync: build_tile_dic (cached)", 2),
            ("observation_get", "  └─ Get observation", 1),
            ("obs_base", "       ├─ Base observation", 2),
            ("obs_reachability", "       ├─ Reachability features", 2),
            ("reach_features", "       │    └─ Feature computation", 3),
            ("obs_graph", "       ├─ Graph observations", 2),
            ("obs_switch_states", "       ├─ Switch states", 2),
            ("proc_spatial_context", "       ├─ Spatial context", 2),
            ("spatial_tiles", "       │    ├─ Tile grid", 3),
            ("spatial_mines", "       │    └─ Mine overlay", 3),
            ("obs_mine_sdf", "       ├─ Mine SDF features", 2),
            ("obs_privileged", "       ├─ Privileged features", 2),
            ("obs_ninja_node", "       └─ Ninja node lookup", 2),
            ("reward_calc", "  └─ Reward calculation", 1),
            ("observation_process", "  └─ Process observation", 1),
            ("info_build", "  └─ Build episode info", 1),
            ("graph_check", "  └─ Graph should_update check", 1),
            ("graph_build", "  └─ Graph build (if updated)", 1),
            ("graph_convert", "       └─ Convert to GraphData", 2),
        ]
        
        for key, label, level in components:
            if key in self._timing_data and len(self._timing_data[key]) > 0:
                times = list(self._timing_data[key])
                avg = np.mean(times)
                std = np.std(times)
                min_t = np.min(times)
                max_t = np.max(times)
                pct = (avg / total_avg) * 100 if total_avg > 0 else 0
                count = len(times)
                
                print(f"{label:<40} {avg:>6.2f}±{std:>5.2f} ms | "
                      f"{min_t:>5.1f}→{max_t:>6.1f} ms | "
                      f"{pct:>5.1f}% | n={count}")
        
        # Graph update statistics
        graph_checks = len(self._timing_data.get("graph_check", []))
        graph_builds = len(self._timing_data.get("graph_build", []))
        if graph_checks > 0:
            update_rate = (graph_builds / graph_checks) * 100
            print(f"\nGraph updates: {graph_builds}/{graph_checks} checks ({update_rate:.1f}% rebuild rate)")

        # Feature-level breakdown (reachability computation sections)
        try:
            from nclone.graph.reachability.feature_computation import (
                get_feature_profiling_stats,
                get_directional_connectivity_cache_stats,
            )

            feature_stats = get_feature_profiling_stats()
            if feature_stats:
                print("\nFeature computation breakdown (avg ms | std | count):")
                print("-" * 60)
                for name, s in sorted(feature_stats.items()):
                    if s["count"] > 0:
                        print(
                            f"  {name:<35} {s['avg_ms']:>6.2f} | "
                            f"std={s['std_ms']:.2f} | n={s['count']}"
                        )
                print("-" * 60)

            # Platform distance cache audit (directional connectivity)
            dc_stats = get_directional_connectivity_cache_stats()
            if dc_stats["total"] > 0:
                print("\nDirectional connectivity cache (platform distances):")
                print(
                    f"  hits={dc_stats['hits']}, misses={dc_stats['misses']}, "
                    f"hit_rate={dc_stats['hit_rate']:.1%}, "
                    f"size={dc_stats['cache_size']}/{dc_stats['max_size']}"
                )
        except ImportError:
            pass

        print("=" * 80 + "\n")
    
    def get_profiling_summary(self) -> Optional[Dict[str, float]]:
        """Get profiling summary as dict.
        
        Returns:
            Dict with average times per component, or None if profiling disabled
        """
        if not self.profile_enabled or not self._timing_data:
            return None
        
        summary = {}
        for key, times in self._timing_data.items():
            if len(times) > 0:
                summary[f"{key}_avg_ms"] = np.mean(list(times))
                summary[f"{key}_std_ms"] = np.std(list(times))
        
        return summary

