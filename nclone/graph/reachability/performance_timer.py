"""
Performance timing utilities for path distance calculations.

Provides lightweight timing instrumentation with minimal overhead (<0.01ms per measurement).
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    
    operation_name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_measurement(self, time_ms: float):
        """Add a timing measurement."""
        self.call_count += 1
        self.total_time_ms += time_ms
        self.min_time_ms = min(self.min_time_ms, time_ms)
        self.max_time_ms = max(self.max_time_ms, time_ms)
        self.recent_times.append(time_ms)
    
    @property
    def avg_time_ms(self) -> float:
        """Get average time in milliseconds."""
        if self.call_count == 0:
            return 0.0
        return self.total_time_ms / self.call_count
    
    @property
    def recent_avg_ms(self) -> float:
        """Get average of recent measurements."""
        if not self.recent_times:
            return 0.0
        return sum(self.recent_times) / len(self.recent_times)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "count": self.call_count,
            "total_ms": self.total_time_ms,
            "avg_ms": self.avg_time_ms,
            "min_ms": self.min_time_ms if self.call_count > 0 else 0.0,
            "max_ms": self.max_time_ms,
            "recent_avg_ms": self.recent_avg_ms,
        }


class PerformanceTimer:
    """
    Lightweight performance timer with context manager support.
    
    Usage:
        timer = PerformanceTimer()
        
        # Context manager (recommended)
        with timer.measure("operation"):
            do_work()
        
        # Manual timing
        timer.start("operation")
        do_work()
        timer.stop("operation")
        
        # Get statistics
        stats = timer.get_stats()
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize performance timer.
        
        Args:
            enabled: Whether timing is enabled (can disable for production)
        """
        self.enabled = enabled
        self.stats: Dict[str, TimingStats] = {}
        self._start_times: Dict[str, float] = {}
    
    def start(self, operation: str):
        """Start timing an operation."""
        if not self.enabled:
            return
        self._start_times[operation] = time.perf_counter()
    
    def stop(self, operation: str) -> float:
        """
        Stop timing an operation and record the measurement.
        
        Args:
            operation: Operation name
            
        Returns:
            Elapsed time in milliseconds
        """
        if not self.enabled:
            return 0.0
            
        if operation not in self._start_times:
            return 0.0
        
        elapsed_s = time.perf_counter() - self._start_times[operation]
        elapsed_ms = elapsed_s * 1000.0
        
        # Record measurement
        if operation not in self.stats:
            self.stats[operation] = TimingStats(operation)
        self.stats[operation].add_measurement(elapsed_ms)
        
        # Clean up
        del self._start_times[operation]
        
        return elapsed_ms
    
    def measure(self, operation: str):
        """
        Context manager for timing an operation.
        
        Usage:
            with timer.measure("operation"):
                do_work()
        """
        return TimingContext(self, operation)
    
    def get_stats(self, operation: Optional[str] = None) -> Dict:
        """
        Get timing statistics.
        
        Args:
            operation: Specific operation name, or None for all operations
            
        Returns:
            Dictionary of statistics
        """
        if operation is not None:
            if operation in self.stats:
                return self.stats[operation].get_summary()
            return {}
        
        # Return all statistics
        return {
            name: stats.get_summary()
            for name, stats in self.stats.items()
        }
    
    def reset(self):
        """Reset all timing statistics."""
        self.stats.clear()
        self._start_times.clear()
    
    def print_summary(self, recent_only: bool = False):
        """
        Print a formatted summary of timing statistics.
        
        Args:
            recent_only: If True, only show recent averages
        """
        if not self.stats:
            print("No timing data available")
            return
        
        print("\n" + "="*80)
        print("Performance Timing Summary")
        print("="*80)
        
        # Sort by total time (most expensive first)
        sorted_stats = sorted(
            self.stats.items(),
            key=lambda x: x[1].total_time_ms,
            reverse=True
        )
        
        for name, stats in sorted_stats:
            print(f"\n{name}:")
            print(f"  Calls: {stats.call_count}")
            
            if recent_only:
                print(f"  Recent Avg: {stats.recent_avg_ms:.3f}ms")
            else:
                print(f"  Total: {stats.total_time_ms:.3f}ms")
                print(f"  Avg: {stats.avg_time_ms:.3f}ms")
                print(f"  Min: {stats.min_time_ms:.3f}ms")
                print(f"  Max: {stats.max_time_ms:.3f}ms")
                print(f"  Recent Avg: {stats.recent_avg_ms:.3f}ms")
        
        print("="*80 + "\n")


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, timer: PerformanceTimer, operation: str):
        """
        Initialize timing context.
        
        Args:
            timer: PerformanceTimer instance
            operation: Operation name
        """
        self.timer = timer
        self.operation = operation
    
    def __enter__(self):
        """Start timing."""
        self.timer.start(self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing."""
        self.timer.stop(self.operation)
        return False  # Don't suppress exceptions


# Global timer instance for convenience
_global_timer = PerformanceTimer(enabled=False)  # Disabled by default


def get_global_timer() -> PerformanceTimer:
    """Get the global performance timer instance."""
    return _global_timer


def enable_global_timing():
    """Enable global performance timing."""
    _global_timer.enabled = True


def disable_global_timing():
    """Disable global performance timing."""
    _global_timer.enabled = False


def reset_global_timing():
    """Reset global timing statistics."""
    _global_timer.reset()


def print_global_timing_summary(recent_only: bool = False):
    """Print global timing summary."""
    _global_timer.print_summary(recent_only=recent_only)

