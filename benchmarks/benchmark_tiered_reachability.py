"""
Comprehensive performance benchmark for the tiered reachability system.

This module provides detailed performance analysis and optimization
recommendations for all three tiers of the reachability system.
"""

import time
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nclone.graph.reachability.tiered_system import TieredReachabilitySystem
from nclone.graph.reachability.reachability_types import PerformanceTarget
from nclone.graph.reachability.opencv_flood_fill import OpenCVFloodFill


class BenchmarkLevelGenerator:
    """Generate various test levels for comprehensive benchmarking."""
    
    @staticmethod
    def create_simple_level(width: int = 42, height: int = 23) -> np.ndarray:
        """Create a simple open level."""
        return np.zeros((width, height), dtype=int)
    
    @staticmethod
    def create_maze_level(width: int = 42, height: int = 23) -> np.ndarray:
        """Create a maze-like level with walls."""
        level = np.zeros((width, height), dtype=int)
        
        # Add some walls to create a maze pattern
        for i in range(5, width, 8):
            for j in range(2, height - 2):
                if j % 4 != 0:  # Leave gaps for navigation
                    level[i, j] = 1  # Solid wall
        
        for i in range(2, width - 2):
            for j in range(5, height, 6):
                if i % 6 != 0:  # Leave gaps for navigation
                    level[i, j] = 1  # Solid wall
        
        return level
    
    @staticmethod
    def create_complex_level(width: int = 42, height: int = 23) -> np.ndarray:
        """Create a complex level with various tile types."""
        level = np.zeros((width, height), dtype=int)
        
        # Add platforms
        for i in range(5, width, 12):
            for j in range(height // 3, height // 3 + 3):
                level[i:i+6, j] = 1  # Platform
        
        # Add switches
        level[10, 10] = 10  # Switch
        level[25, 15] = 10  # Switch
        
        # Add doors
        level[15, 8] = 20   # Door
        level[30, 18] = 20  # Door
        
        # Add some scattered walls
        np.random.seed(42)  # For reproducible results
        wall_positions = np.random.choice(width * height, size=50, replace=False)
        for pos in wall_positions:
            x, y = pos // height, pos % height
            if level[x, y] == 0:  # Only place walls in empty spaces
                level[x, y] = 1
        
        return level
    
    @staticmethod
    def create_large_level(width: int = 84, height: int = 46) -> np.ndarray:
        """Create a large level for stress testing."""
        level = np.zeros((width, height), dtype=int)
        
        # Add a complex pattern
        for i in range(0, width, 15):
            for j in range(0, height, 10):
                # Create rooms with corridors
                level[i:i+10, j:j+8] = 1  # Room walls
                level[i+2:i+8, j+2:j+6] = 0  # Room interior
                
                # Add corridor connections
                if i + 15 < width:
                    level[i+10:i+15, j+4] = 0  # Horizontal corridor
                if j + 10 < height:
                    level[i+4, j+8:j+10] = 0  # Vertical corridor
        
        return level


class ReachabilityBenchmark:
    """Comprehensive performance benchmark for tiered reachability system."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.tiered_system = TieredReachabilitySystem(debug=False)
        self.benchmark_levels = self._load_benchmark_levels()
        
        # Benchmark configuration
        self.iterations_per_test = 10
        self.ninja_positions = [
            (120, 240),  # Center-left
            (240, 120),  # Top-center
            (360, 360),  # Bottom-right
        ]
        
    def _load_benchmark_levels(self) -> List[Tuple[str, Any]]:
        """Load benchmark levels of varying complexity."""
        generator = BenchmarkLevelGenerator()
        
        return [
            ("Simple", generator.create_simple_level()),
            ("Complex", generator.create_complex_level()),
        ]
    
    def benchmark_all_tiers(self) -> Dict[str, Any]:
        """Run comprehensive benchmark of all tiers."""
        print("Starting Tiered Reachability Benchmark Suite")
        print("=" * 60)
        
        results = {
            'tier1': {'times': [], 'accuracies': [], 'position_counts': []},
            'tier2': {'times': [], 'accuracies': [], 'position_counts': []},
            'summary': {},
            'recommendations': []
        }
        
        # Benchmark each level and position combination
        for level_name, level_data in self.benchmark_levels:
            print(f"\nBenchmarking {level_name} level...")
            
            for ninja_pos in self.ninja_positions:
                switch_states = {}
                
                # Benchmark Tier 1
                tier1_times, tier1_results = self._benchmark_tier(
                    1, level_data, ninja_pos, switch_states
                )
                
                # Benchmark Tier 2
                tier2_times, tier2_results = self._benchmark_tier(
                    2, level_data, ninja_pos, switch_states
                )
                
                # Record results
                results['tier1']['times'].extend(tier1_times)
                results['tier1']['accuracies'].extend([r.confidence for r in tier1_results])
                results['tier1']['position_counts'].extend([len(r.reachable_positions) for r in tier1_results])
                
                results['tier2']['times'].extend(tier2_times)
                results['tier2']['accuracies'].extend([r.confidence for r in tier2_results])
                results['tier2']['position_counts'].extend([len(r.reachable_positions) for r in tier2_results])
        
        # Generate summary statistics
        results['summary'] = self._generate_summary_stats(results)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Print results
        self._print_benchmark_results(results)
        
        return results
    
    def _benchmark_tier(self, tier_num: int, level_data, ninja_pos: Tuple[int, int], 
                       switch_states: Dict[str, bool]) -> Tuple[List[float], List[Any]]:
        """Benchmark a specific tier."""
        times = []
        results = []
        
        for _ in range(self.iterations_per_test):
            start_time = time.perf_counter()
            
            if tier_num == 1:
                result = self.tiered_system.tier1.quick_check(ninja_pos, level_data, switch_states, [])
            elif tier_num == 2:
                result = self.tiered_system.tier2.quick_check(ninja_pos, level_data, switch_states, [])
            else:
                raise ValueError(f"Unsupported tier: {tier_num}")
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            times.append(elapsed_ms)
            results.append(result)
        
        return times, results
    
    def _generate_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for all tiers."""
        summary = {}
        
        for tier_name in ['tier1', 'tier2']:
            metrics = results[tier_name]
            
            if metrics['times']:
                summary[tier_name] = {
                    'avg_time_ms': np.mean(metrics['times']),
                    'p95_time_ms': np.percentile(metrics['times'], 95),
                    'max_time_ms': np.max(metrics['times']),
                    'avg_accuracy': np.mean(metrics['accuracies']),
                    'min_accuracy': np.min(metrics['accuracies']),
                    'avg_positions': np.mean(metrics['position_counts']),
                    'sample_count': len(metrics['times'])
                }
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on results."""
        recommendations = []
        summary = results['summary']
        
        # Check Tier 1 performance
        if 'tier1' in summary:
            tier1_p95 = summary['tier1']['p95_time_ms']
            if tier1_p95 > 1.0:
                recommendations.append(
                    f"Tier 1 p95 time ({tier1_p95:.3f}ms) exceeds 1ms target."
                )
            
            tier1_accuracy = summary['tier1']['avg_accuracy']
            if tier1_accuracy < 0.80:
                recommendations.append(
                    f"Tier 1 accuracy ({tier1_accuracy:.3f}) below 0.80 target."
                )
        
        # Check Tier 2 performance
        if 'tier2' in summary:
            tier2_p95 = summary['tier2']['p95_time_ms']
            if tier2_p95 > 10.0:
                recommendations.append(
                    f"Tier 2 p95 time ({tier2_p95:.3f}ms) exceeds 10ms target."
                )
            
            tier2_accuracy = summary['tier2']['avg_accuracy']
            if tier2_accuracy < 0.90:
                recommendations.append(
                    f"Tier 2 accuracy ({tier2_accuracy:.3f}) below 0.90 target."
                )
        
        if not recommendations:
            recommendations.append("All performance targets met. System is well-optimized.")
        
        return recommendations
    
    def _print_benchmark_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        summary = results['summary']
        
        for tier_name in ['tier1', 'tier2']:
            if tier_name in summary:
                tier_num = tier_name[-1]
                stats = summary[tier_name]
                
                print(f"\nTier {tier_num} Performance:")
                print(f"  Average time:     {stats['avg_time_ms']:.3f}ms")
                print(f"  95th percentile:  {stats['p95_time_ms']:.3f}ms")
                print(f"  Maximum time:     {stats['max_time_ms']:.3f}ms")
                print(f"  Average accuracy: {stats['avg_accuracy']:.3f}")
                print(f"  Minimum accuracy: {stats['min_accuracy']:.3f}")
                print(f"  Avg positions:    {stats['avg_positions']:.1f}")
                print(f"  Sample count:     {stats['sample_count']}")
        
        print("\n" + "-" * 60)
        print("RECOMMENDATIONS:")
        print("-" * 60)
        
        for i, recommendation in enumerate(results['recommendations'], 1):
            print(f"{i}. {recommendation}")
        
        print("\n" + "=" * 60)


def main():
    """Run the complete benchmark suite."""
    benchmark = ReachabilityBenchmark()
    
    # Run benchmarks
    results = benchmark.benchmark_all_tiers()
    
    return results


if __name__ == '__main__':
    main()