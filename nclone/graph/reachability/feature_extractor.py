"""
Simplified interface for extracting compact reachability features.

This module provides a simplified ReachabilityFeatureExtractor that uses the
graph-based flood fill reachability system with minimal caching and overhead.
Designed for real-time RL training with <1ms performance targets.
"""

import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .reachability_system import ReachabilitySystem
from .compact_features import CompactReachabilityFeatures, FeatureConfig


class ReachabilityFeatureExtractor:
    """
    Simplified interface for extracting compact reachability features.

    This class provides a streamlined API for RL integration, using ultra-fast
    reachability analysis with minimal overhead. Designed for <1ms performance.

    Example usage:
        extractor = ReachabilityFeatureExtractor()
        features = extractor.extract_features(
            ninja_pos, level_data, entities, switch_states
        )
    """

    def __init__(
        self,
        reachability_system: Optional[ReachabilitySystem] = None,
        feature_config: Optional[FeatureConfig] = None,
        debug: bool = False,
    ):
        """
        Initialize simplified feature extractor.

        Args:
            reachability_system: Ultra-fast reachability system (creates default if None)
            feature_config: Feature configuration (uses default if None)
            debug: Enable debug output
        """
        self.reachability_system = reachability_system or ReachabilitySystem(
            debug=debug
        )
        self.feature_encoder = CompactReachabilityFeatures(
            config=feature_config, debug=debug
        )
        self.debug = debug

        # Minimal performance tracking
        self.extraction_times = []

    def extract_features(
        self,
        ninja_position: Tuple[float, float],
        level_data: Any,
        entities: List[Any],
        switch_states: Optional[Dict[str, bool]] = None,
    ) -> np.ndarray:
        """
        Extract simplified reachability features for RL integration.

        This is the main entry point for feature extraction using ultra-fast
        flood fill analysis with minimal overhead.

        Args:
            ninja_position: Current ninja position (x, y)
            level_data: Level data structure
            entities: List of game entities
            switch_states: Current switch activation states

        Returns:
            8-dimensional numpy array with encoded features
        """
        start_time = time.perf_counter()

        if switch_states is None:
            switch_states = {}

        # Get reachability analysis using ultra-fast system
        reachability_result = self.reachability_system.analyze_reachability(
            level_data=level_data,
            ninja_position=ninja_position,
            switch_states=switch_states,
        )

        # Encode features
        features = self.feature_encoder.encode_reachability(
            reachability_result=reachability_result,
            level_data=level_data,
            entities=entities,
            ninja_position=ninja_position,
            switch_states=switch_states,
        )

        # Record timing
        computation_time_ms = (time.perf_counter() - start_time) * 1000
        self.extraction_times.append(computation_time_ms)
        
        # Keep only recent timing history
        if len(self.extraction_times) > 100:
            self.extraction_times = self.extraction_times[-100:]

        if self.debug:
            print(f"DEBUG: Feature extraction completed in {computation_time_ms:.2f}ms")

        return features

    def extract_features_batch(
        self,
        batch_data: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Extract features for a batch of states (for batch RL training).

        Args:
            batch_data: List of dicts with keys: ninja_position, level_data, entities, switch_states

        Returns:
            Array of shape (batch_size, 8) with encoded features
        """
        batch_size = len(batch_data)
        batch_features = np.zeros((batch_size, 8), dtype=np.float32)

        for i, data in enumerate(batch_data):
            batch_features[i] = self.extract_features(
                ninja_position=data["ninja_position"],
                level_data=data["level_data"],
                entities=data["entities"],
                switch_states=data.get("switch_states"),
            )

        return batch_features

    def get_feature_names(self) -> List[str]:
        """
        Get human-readable names for each feature dimension.

        Returns:
            List of 8 feature names for debugging and analysis
        """
        return self.feature_encoder.get_feature_names()

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get simplified performance statistics.

        Returns:
            Dictionary with basic performance metrics
        """
        if not self.extraction_times:
            return {
                "total_extractions": 0,
                "avg_extraction_time_ms": 0.0,
                "max_extraction_time_ms": 0.0,
                "min_extraction_time_ms": 0.0,
            }

        return {
            "total_extractions": len(self.extraction_times),
            "avg_extraction_time_ms": float(np.mean(self.extraction_times)),
            "max_extraction_time_ms": float(np.max(self.extraction_times)),
            "min_extraction_time_ms": float(np.min(self.extraction_times)),
        }

    def validate_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Validate feature vector for debugging and quality assurance.

        Args:
            features: 8-dimensional feature vector

        Returns:
            Dictionary with validation results
        """
        validation = {
            "shape_valid": features.shape == (8,),
            "dtype_valid": features.dtype == np.float32,
            "has_nan": np.isnan(features).any(),
            "has_inf": np.isinf(features).any(),
            "min_value": float(features.min()),
            "max_value": float(features.max()),
            "mean_value": float(features.mean()),
            "std_value": float(features.std()),
            "zero_features": int(np.sum(features == 0.0)),
            "nonzero_features": int(np.sum(features != 0.0)),
        }

        # Check for reasonable value ranges
        validation["values_in_range"] = np.all((features >= 0.0) & (features <= 1.0))

        # Check for sufficient variance
        validation["sufficient_variance"] = validation["std_value"] > 0.01

        return validation
