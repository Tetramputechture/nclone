"""
Constants for graph visualization and navigation system.

This module contains all configurable constants used throughout the graph
visualization system to avoid magic numbers and provide centralized configuration.
"""

from typing import Tuple

# Import actual physics constants from nclone
from ..constants.physics_constants import (
    GRAVITY_FALL, GRAVITY_JUMP, MAX_HOR_SPEED, MAX_VER_SPEED,
    JUMP_INITIAL_VELOCITY, MAX_JUMP_DISTANCE, MAX_FALL_DISTANCE
)

# =============================================================================
# PATHFINDING CONSTANTS
# =============================================================================

class PathfindingDefaults:
    """Default values for navigation algorithms."""
    
    # Algorithm parameters
    HEURISTIC_WEIGHT: float = 1.0
    MAX_NODES_TO_EXPLORE: int = 10000
    
    # Edge cost weights for physics-accurate navigation
    EDGE_COST_WEIGHTS = {
        'energy_cost': 1.0,        # Weight for energy consumption
        'time_estimate': 0.5,      # Weight for time taken
        'difficulty': 2.0,         # Weight for movement difficulty
        'success_probability': -1.0,  # Negative: higher probability = lower cost
    }
    
    # Physics constants for heuristic calculation (from actual N++ physics)
    GRAVITY_FALL: float = GRAVITY_FALL
    GRAVITY_JUMP: float = GRAVITY_JUMP
    MAX_HORIZONTAL_SPEED: float = MAX_HOR_SPEED
    MAX_VERTICAL_SPEED: float = MAX_VER_SPEED
    JUMP_INITIAL_VELOCITY: float = JUMP_INITIAL_VELOCITY
    MAX_JUMP_DISTANCE: float = MAX_JUMP_DISTANCE
    MAX_FALL_DISTANCE: float = MAX_FALL_DISTANCE


# =============================================================================
# VISUALIZATION CONSTANTS
# =============================================================================

class VisualizationDefaults:
    """Default values for visualization rendering."""
    
    # Rendering sizes
    NODE_SIZE: float = 3.0
    EDGE_WIDTH: float = 1.0
    PATH_WIDTH: float = 3.0
    
    # Transparency
    ALPHA: float = 0.8
    
    # Font sizes
    SMALL_FONT_SIZE: int = 16
    MEDIUM_FONT_SIZE: int = 20
    LARGE_FONT_SIZE: int = 24
    
    # Grid rendering
    GRID_SPACING: int = 50
    GRID_LINE_WIDTH: int = 1


class ColorScheme:
    """Color scheme for visualization components."""
    
    # Background colors (RGB)
    BACKGROUND_COLOR: Tuple[int, int, int] = (20, 20, 30)
    GRID_COLOR: Tuple[int, int, int] = (60, 60, 70)
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
    
    # Node colors by type (RGBA)
    NODE_COLORS = {
        0: (120, 120, 140, 180),  # GRID_CELL
        1: (255, 100, 100, 220),  # ENTITY
        2: (100, 255, 255, 255),  # NINJA
    }
    
    # Edge colors by type (RGBA)
    # Keys align 1:1 with EdgeType in `graph/common.py`:
    # 0: WALK, 1: JUMP, 2: WALL_SLIDE, 3: FALL, 4: ONE_WAY, 5: FUNCTIONAL
    EDGE_COLORS = {
        0: (100, 255, 100, 170),  # WALK (green)
        1: (255, 180, 80, 190),   # JUMP (orange)
        2: (180, 100, 255, 160),  # WALL_SLIDE (purple)
        3: (100, 160, 255, 180),  # FALL (blue)
        4: (255, 120, 200, 180),  # ONE_WAY (pink)
        5: (255, 255, 100, 210),  # FUNCTIONAL (yellow)
    }
    
    # Special visualization colors
    PATH_COLOR: Tuple[int, int, int, int] = (255, 255, 0, 255)  # Yellow path
    GOAL_COLOR: Tuple[int, int, int, int] = (255, 0, 0, 255)    # Red goal
    NINJA_COLOR: Tuple[int, int, int, int] = (0, 255, 0, 255)   # Green ninja


# =============================================================================
# OVERLAY CONSTANTS
# =============================================================================

class OverlayDefaults:
    """Default values for debug overlay system."""
    
    # Overlay transparency
    OVERLAY_ALPHA: float = 0.7
    
    # Text positioning
    TEXT_MARGIN: int = 10
    LINE_HEIGHT: int = 25
    
    # Interactive controls
    MOUSE_CLICK_TOLERANCE: float = 10.0  # Pixels
    
    # Performance monitoring
    PERFORMANCE_HISTORY_SIZE: int = 100
    FPS_UPDATE_INTERVAL: float = 0.5  # Seconds


# =============================================================================
# PHYSICS INTEGRATION CONSTANTS
# =============================================================================

class PhysicsIntegration:
    """Constants for physics integration with npp-rl."""
    
    # Entity caching
    STATIC_ENTITY_TYPES = {1, 2, 3, 4, 5}  # Entity types that don't change state
    DYNAMIC_ENTITY_TYPES = {10, 15, 20}    # Entity types with changing state
    
    # Trajectory validation
    TRAJECTORY_SAMPLE_RATE: float = 0.1    # Time step for trajectory sampling
    COLLISION_TOLERANCE: float = 1.0       # Collision detection tolerance
    
    # Movement classification
    MOVEMENT_ANALYSIS_CACHE_SIZE: int = 1000
    TRAJECTORY_CACHE_SIZE: int = 500


# =============================================================================
# ERROR HANDLING CONSTANTS
# =============================================================================

class ErrorHandling:
    """Constants for error handling and fallback behavior."""
    
    # Timeout values
    PATHFINDING_TIMEOUT: float = 5.0      # Seconds
    RENDERING_TIMEOUT: float = 2.0        # Seconds
    
    # Fallback values
    FALLBACK_NODE_SIZE: float = 2.0
    FALLBACK_EDGE_WIDTH: float = 1.0
    FALLBACK_ALPHA: float = 0.5
    
    # Retry limits
    MAX_PATHFINDING_RETRIES: int = 3
    MAX_RENDERING_RETRIES: int = 2


# =============================================================================
# PERFORMANCE CONSTANTS
# =============================================================================

class PerformanceSettings:
    """Performance-related constants and thresholds."""
    
    # Cache sizes
    GRAPH_CACHE_SIZE: int = 50
    ENTITY_CACHE_SIZE: int = 100
    VISUALIZATION_CACHE_SIZE: int = 20
    
    # Performance thresholds
    LARGE_GRAPH_NODE_THRESHOLD: int = 1000
    LARGE_GRAPH_EDGE_THRESHOLD: int = 5000
    
    # Optimization settings
    ENABLE_HIERARCHICAL_RENDERING: bool = True
    ENABLE_LEVEL_OF_DETAIL: bool = True
    LOD_DISTANCE_THRESHOLD: float = 100.0