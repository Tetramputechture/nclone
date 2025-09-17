"""
High-level interface for extracting compact reachability features.

This module provides the ReachabilityFeatureExtractor class, which serves as the
main interface for RL integration. It combines the tiered reachability system
with the compact feature encoder to provide efficient, cached feature extraction
suitable for real-time RL training.

Key features:
- Automatic performance tier selection
- Intelligent caching with TTL
- Graceful error handling
- Performance monitoring
- RL-friendly API design
"""

import time
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .tiered_system import TieredReachabilitySystem
from .reachability_types import PerformanceTarget, ReachabilityResult
from .compact_features import CompactReachabilityFeatures, FeatureConfig


@dataclass
class CacheEntry:
    """Cache entry for feature extraction results."""
    features: np.ndarray
    timestamp: float
    computation_time_ms: float
    performance_target: str
    cache_hits: int = 0


class PerformanceMode(Enum):
    """Performance modes for feature extraction."""
    ULTRA_FAST = "ultra_fast"    # Tier 1: <1ms, 85% accuracy
    FAST = "fast"                # Tier 2: <5ms, 92% accuracy  
    BALANCED = "balanced"        # Tier 2: <5ms, 92% accuracy
    ACCURATE = "accurate"        # Tier 3: <100ms, 99% accuracy
    PRECISE = "precise"          # Tier 3: <100ms, 99% accuracy


class ReachabilityFeatureExtractor:
    """
    High-level interface for extracting compact reachability features.
    
    This class provides the main API for RL integration, combining the tiered
    reachability system with compact feature encoding. It includes intelligent
    caching, performance monitoring, and automatic tier selection to meet
    real-time RL training requirements.
    
    Example usage:
        extractor = ReachabilityFeatureExtractor()
        features = extractor.extract_features(
            ninja_pos, level_data, entities, switch_states, 
            performance_mode=PerformanceMode.FAST
        )
    """
    
    def __init__(
        self,
        tiered_system: Optional[TieredReachabilitySystem] = None,
        feature_config: Optional[FeatureConfig] = None,
        cache_ttl_ms: float = 100.0,
        max_cache_size: int = 1000,
        debug: bool = False
    ):
        """
        Initialize feature extractor.
        
        Args:
            tiered_system: Tiered reachability system (creates default if None)
            feature_config: Feature configuration (uses default if None)
            cache_ttl_ms: Cache time-to-live in milliseconds
            max_cache_size: Maximum number of cached entries
            debug: Enable debug output
        """
        self.tiered_system = tiered_system or TieredReachabilitySystem(debug=debug)
        self.feature_encoder = CompactReachabilityFeatures(config=feature_config, debug=debug)
        self.debug = debug
        
        # Caching configuration
        self.cache_ttl_ms = cache_ttl_ms
        self.max_cache_size = max_cache_size
        self.feature_cache: Dict[str, CacheEntry] = {}
        
        # Performance monitoring
        self.extraction_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance target mapping
        self.performance_target_map = {
            PerformanceMode.ULTRA_FAST: PerformanceTarget.ULTRA_FAST,
            PerformanceMode.FAST: PerformanceTarget.FAST,
            PerformanceMode.BALANCED: PerformanceTarget.BALANCED,
            PerformanceMode.ACCURATE: PerformanceTarget.ACCURATE,
            PerformanceMode.PRECISE: PerformanceTarget.PRECISE,
        }
        
    def extract_features(
        self,
        ninja_position: Tuple[float, float],
        level_data: Any,
        entities: List[Any],
        switch_states: Optional[Dict[str, bool]] = None,
        performance_mode: PerformanceMode = PerformanceMode.FAST
    ) -> np.ndarray:
        """
        Extract compact reachability features for RL integration.
        
        This is the main entry point for feature extraction. It automatically
        handles caching, performance tier selection, and error recovery to
        provide reliable feature vectors for RL training.
        
        Args:
            ninja_position: Current ninja position (x, y)
            level_data: Level data structure
            entities: List of game entities
            switch_states: Current switch activation states
            performance_mode: Performance/accuracy tradeoff
            
        Returns:
            64-dimensional numpy array with encoded features
        """
        start_time = time.perf_counter()
        
        if switch_states is None:
            switch_states = {}
            
        try:
            # Check cache first
            cache_key = self._generate_cache_key(ninja_position, switch_states, level_data, entities)
            cached_entry = self._get_cached_features(cache_key)
            
            if cached_entry is not None:
                self.cache_hits += 1
                cached_entry.cache_hits += 1
                
                if self.debug:
                    print(f"DEBUG: Cache hit for key {cache_key[:16]}...")
                
                return cached_entry.features
            
            # Cache miss - compute features
            self.cache_misses += 1
            
            # Get reachability analysis using appropriate tier
            performance_target = self.performance_target_map[performance_mode]
            reachability_result = self.tiered_system.analyze_reachability(
                level_data=level_data,
                ninja_position=ninja_position,
                switch_states=switch_states,
                performance_target=performance_target
            )
            
            # Encode features
            features = self.feature_encoder.encode_reachability(
                reachability_result=reachability_result,
                level_data=level_data,
                entities=entities,
                ninja_position=ninja_position,
                switch_states=switch_states
            )
            
            # Record timing
            computation_time_ms = (time.perf_counter() - start_time) * 1000
            self.extraction_times.append(computation_time_ms)
            
            # Cache result
            self._cache_features(cache_key, features, computation_time_ms, performance_mode.value)
            
            if self.debug:
                print(f"DEBUG: Feature extraction completed in {computation_time_ms:.2f}ms "
                      f"using {performance_mode.value} mode")
            
            return features
            
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Feature extraction error: {e}")
            
            # Return zero features on error (graceful degradation)
            computation_time_ms = (time.perf_counter() - start_time) * 1000
            self.extraction_times.append(computation_time_ms)
            
            return np.zeros(64, dtype=np.float32)
    
    def extract_features_batch(
        self,
        batch_data: List[Dict[str, Any]],
        performance_mode: PerformanceMode = PerformanceMode.FAST
    ) -> np.ndarray:
        """
        Extract features for a batch of states (for batch RL training).
        
        Args:
            batch_data: List of dicts with keys: ninja_position, level_data, entities, switch_states
            performance_mode: Performance/accuracy tradeoff
            
        Returns:
            Array of shape (batch_size, 64) with encoded features
        """
        batch_size = len(batch_data)
        batch_features = np.zeros((batch_size, 64), dtype=np.float32)
        
        for i, data in enumerate(batch_data):
            batch_features[i] = self.extract_features(
                ninja_position=data['ninja_position'],
                level_data=data['level_data'],
                entities=data['entities'],
                switch_states=data.get('switch_states'),
                performance_mode=performance_mode
            )
        
        return batch_features
    
    def get_feature_names(self) -> List[str]:
        """
        Get human-readable names for each feature dimension.
        
        Returns:
            List of 64 feature names for debugging and analysis
        """
        return self.feature_encoder.get_feature_names()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for monitoring and optimization.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.extraction_times:
            return {
                'total_extractions': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'cache_hit_rate': 0.0,
                'avg_extraction_time_ms': 0.0,
                'max_extraction_time_ms': 0.0,
                'min_extraction_time_ms': 0.0,
                'cache_size': len(self.feature_cache),
                'cache_memory_mb': self._estimate_cache_memory_mb()
            }
        
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'total_extractions': len(self.extraction_times),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'avg_extraction_time_ms': np.mean(self.extraction_times),
            'max_extraction_time_ms': np.max(self.extraction_times),
            'min_extraction_time_ms': np.min(self.extraction_times),
            'cache_size': len(self.feature_cache),
            'cache_memory_mb': self._estimate_cache_memory_mb()
        }
    
    def clear_cache(self):
        """Clear the feature cache."""
        self.feature_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
        if self.debug:
            print("DEBUG: Feature cache cleared")
    
    def optimize_cache(self):
        """Optimize cache by removing old entries."""
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Remove expired entries
        expired_keys = []
        for key, entry in self.feature_cache.items():
            if current_time - entry.timestamp > self.cache_ttl_ms:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.feature_cache[key]
        
        # Remove least recently used entries if cache is too large
        if len(self.feature_cache) > self.max_cache_size:
            # Sort by cache hits (keep most used entries)
            sorted_entries = sorted(
                self.feature_cache.items(),
                key=lambda x: x[1].cache_hits,
                reverse=True
            )
            
            # Keep only the most used entries
            self.feature_cache = dict(sorted_entries[:self.max_cache_size])
        
        if self.debug:
            print(f"DEBUG: Cache optimized, {len(self.feature_cache)} entries remaining")
    
    def validate_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Validate feature vector for debugging and quality assurance.
        
        Args:
            features: 64-dimensional feature vector
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'shape_valid': features.shape == (64,),
            'dtype_valid': features.dtype == np.float32,
            'has_nan': np.isnan(features).any(),
            'has_inf': np.isinf(features).any(),
            'min_value': float(features.min()),
            'max_value': float(features.max()),
            'mean_value': float(features.mean()),
            'std_value': float(features.std()),
            'zero_features': int(np.sum(features == 0.0)),
            'nonzero_features': int(np.sum(features != 0.0))
        }
        
        # Check for reasonable value ranges
        validation['values_in_range'] = np.all((features >= -2.0) & (features <= 2.0))
        
        # Check for sufficient variance
        validation['sufficient_variance'] = validation['std_value'] > 0.01
        
        return validation
    
    def _generate_cache_key(
        self,
        ninja_position: Tuple[float, float],
        switch_states: Dict[str, bool],
        level_data: Any,
        entities: List[Any]
    ) -> str:
        """Generate cache key for current state."""
        # Create a hash of the current state
        state_str = f"{ninja_position}_{sorted(switch_states.items())}"
        
        # Add level data hash (simplified)
        if hasattr(level_data, 'shape'):
            level_hash = f"{level_data.shape}_{hash(level_data.tobytes()) % 10000}"
        else:
            level_hash = f"{getattr(level_data, 'width', 0)}_{getattr(level_data, 'height', 0)}"
        
        # Add entity count (simplified entity hash)
        entity_hash = f"{len(entities)}_{len([e for e in entities if hasattr(e, 'entity_type')])}"
        
        full_state = f"{state_str}_{level_hash}_{entity_hash}"
        
        # Generate SHA256 hash
        return hashlib.sha256(full_state.encode()).hexdigest()
    
    def _get_cached_features(self, cache_key: str) -> Optional[CacheEntry]:
        """Get cached features if valid."""
        if cache_key not in self.feature_cache:
            return None
        
        entry = self.feature_cache[cache_key]
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Check if cache entry is still valid
        if current_time - entry.timestamp > self.cache_ttl_ms:
            del self.feature_cache[cache_key]
            return None
        
        return entry
    
    def _cache_features(
        self,
        cache_key: str,
        features: np.ndarray,
        computation_time_ms: float,
        performance_target: str
    ):
        """Cache computed features."""
        current_time = time.time() * 1000  # Convert to milliseconds
        
        entry = CacheEntry(
            features=features.copy(),
            timestamp=current_time,
            computation_time_ms=computation_time_ms,
            performance_target=performance_target
        )
        
        self.feature_cache[cache_key] = entry
        
        # Optimize cache if it's getting too large
        if len(self.feature_cache) > self.max_cache_size * 1.2:
            self.optimize_cache()
    
    def _estimate_cache_memory_mb(self) -> float:
        """Estimate cache memory usage in MB."""
        if not self.feature_cache:
            return 0.0
        
        # Each feature vector is 64 float32 values = 256 bytes
        # Plus overhead for cache entry structure
        bytes_per_entry = 256 + 100  # Approximate overhead
        total_bytes = len(self.feature_cache) * bytes_per_entry
        
        return total_bytes / (1024 * 1024)  # Convert to MB


class FeatureAnalyzer:
    """
    Utility class for analyzing and visualizing compact features.
    
    This class provides tools for understanding feature behavior,
    debugging encoding issues, and validating feature quality.
    """
    
    def __init__(self, extractor: ReachabilityFeatureExtractor):
        """
        Initialize feature analyzer.
        
        Args:
            extractor: Feature extractor to analyze
        """
        self.extractor = extractor
        self.feature_names = extractor.get_feature_names()
    
    def analyze_feature_distribution(
        self,
        test_states: List[Dict[str, Any]],
        performance_mode: PerformanceMode = PerformanceMode.FAST
    ) -> Dict[str, Any]:
        """
        Analyze feature distribution across multiple states.
        
        Args:
            test_states: List of test states to analyze
            performance_mode: Performance mode for extraction
            
        Returns:
            Dictionary with distribution analysis
        """
        if not test_states:
            return {}
        
        # Extract features for all test states
        features_batch = self.extractor.extract_features_batch(test_states, performance_mode)
        
        analysis = {
            'num_states': len(test_states),
            'feature_stats': {},
            'correlation_analysis': {},
            'outlier_analysis': {}
        }
        
        # Analyze each feature dimension
        for i, feature_name in enumerate(self.feature_names):
            feature_values = features_batch[:, i]
            
            analysis['feature_stats'][feature_name] = {
                'mean': float(np.mean(feature_values)),
                'std': float(np.std(feature_values)),
                'min': float(np.min(feature_values)),
                'max': float(np.max(feature_values)),
                'median': float(np.median(feature_values)),
                'zero_ratio': float(np.sum(feature_values == 0.0) / len(feature_values)),
                'variance': float(np.var(feature_values))
            }
        
        # Correlation analysis
        if len(test_states) > 1:
            correlation_matrix = np.corrcoef(features_batch.T)
            
            # Find highly correlated features
            high_correlations = []
            for i in range(len(self.feature_names)):
                for j in range(i + 1, len(self.feature_names)):
                    corr = correlation_matrix[i, j]
                    if abs(corr) > 0.8:  # High correlation threshold
                        high_correlations.append({
                            'feature1': self.feature_names[i],
                            'feature2': self.feature_names[j],
                            'correlation': float(corr)
                        })
            
            analysis['correlation_analysis'] = {
                'high_correlations': high_correlations,
                'max_correlation': float(np.max(np.abs(correlation_matrix - np.eye(len(self.feature_names))))),
                'mean_abs_correlation': float(np.mean(np.abs(correlation_matrix - np.eye(len(self.feature_names)))))
            }
        
        return analysis
    
    def compare_performance_modes(
        self,
        test_states: List[Dict[str, Any]],
        modes: List[PerformanceMode] = None
    ) -> Dict[str, Any]:
        """
        Compare feature extraction across different performance modes.
        
        Args:
            test_states: List of test states
            modes: Performance modes to compare (uses all if None)
            
        Returns:
            Dictionary with comparison results
        """
        if modes is None:
            modes = list(PerformanceMode)
        
        comparison = {
            'modes': [mode.value for mode in modes],
            'timing_comparison': {},
            'feature_similarity': {},
            'accuracy_analysis': {}
        }
        
        mode_features = {}
        mode_timings = {}
        
        # Extract features using each mode
        for mode in modes:
            start_time = time.perf_counter()
            features = self.extractor.extract_features_batch(test_states, mode)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            mode_features[mode.value] = features
            mode_timings[mode.value] = elapsed_ms / len(test_states)  # Average per state
        
        comparison['timing_comparison'] = mode_timings
        
        # Compare feature similarity between modes
        if len(modes) > 1:
            base_mode = modes[0].value
            base_features = mode_features[base_mode]
            
            for mode in modes[1:]:
                mode_name = mode.value
                mode_feat = mode_features[mode_name]
                
                # Calculate similarity metrics
                mse = np.mean((base_features - mode_feat) ** 2)
                mae = np.mean(np.abs(base_features - mode_feat))
                correlation = np.corrcoef(base_features.flatten(), mode_feat.flatten())[0, 1]
                
                comparison['feature_similarity'][f"{base_mode}_vs_{mode_name}"] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'correlation': float(correlation)
                }
        
        return comparison
    
    def generate_feature_report(
        self,
        test_states: List[Dict[str, Any]],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive feature analysis report.
        
        Args:
            test_states: List of test states to analyze
            output_file: Optional file to save report to
            
        Returns:
            Report as string
        """
        report_lines = [
            "# Compact Reachability Features Analysis Report",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Test states: {len(test_states)}",
            ""
        ]
        
        # Performance statistics
        perf_stats = self.extractor.get_performance_stats()
        report_lines.extend([
            "## Performance Statistics",
            f"- Total extractions: {perf_stats['total_extractions']}",
            f"- Cache hit rate: {perf_stats['cache_hit_rate']:.1%}",
            f"- Average extraction time: {perf_stats['avg_extraction_time_ms']:.2f}ms",
            f"- Max extraction time: {perf_stats['max_extraction_time_ms']:.2f}ms",
            f"- Cache size: {perf_stats['cache_size']} entries",
            f"- Cache memory: {perf_stats['cache_memory_mb']:.2f}MB",
            ""
        ])
        
        # Feature distribution analysis
        if test_states:
            distribution = self.analyze_feature_distribution(test_states)
            
            report_lines.extend([
                "## Feature Distribution Analysis",
                f"- States analyzed: {distribution['num_states']}",
                ""
            ])
            
            # Top features by variance
            feature_stats = distribution['feature_stats']
            sorted_features = sorted(
                feature_stats.items(),
                key=lambda x: x[1]['variance'],
                reverse=True
            )
            
            report_lines.extend([
                "### Top Features by Variance",
                "| Feature | Mean | Std | Min | Max | Zero% |",
                "|---------|------|-----|-----|-----|-------|"
            ])
            
            for feature_name, stats in sorted_features[:10]:
                report_lines.append(
                    f"| {feature_name} | {stats['mean']:.3f} | {stats['std']:.3f} | "
                    f"{stats['min']:.3f} | {stats['max']:.3f} | {stats['zero_ratio']:.1%} |"
                )
            
            report_lines.append("")
            
            # Correlation analysis
            if 'correlation_analysis' in distribution:
                corr_analysis = distribution['correlation_analysis']
                high_corrs = corr_analysis['high_correlations']
                
                if high_corrs:
                    report_lines.extend([
                        "### High Correlations (>0.8)",
                        "| Feature 1 | Feature 2 | Correlation |",
                        "|-----------|-----------|-------------|"
                    ])
                    
                    for corr in high_corrs[:5]:  # Top 5
                        report_lines.append(
                            f"| {corr['feature1']} | {corr['feature2']} | {corr['correlation']:.3f} |"
                        )
                    
                    report_lines.append("")
        
        # Performance mode comparison
        if len(test_states) > 0:
            comparison = self.compare_performance_modes(test_states[:5])  # Use subset for speed
            
            report_lines.extend([
                "## Performance Mode Comparison",
                "| Mode | Avg Time (ms) |",
                "|------|---------------|"
            ])
            
            for mode, timing in comparison['timing_comparison'].items():
                report_lines.append(f"| {mode} | {timing:.2f} |")
            
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report