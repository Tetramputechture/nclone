"""
Pathfinding module for N++ clone.

This module provides pathfinding utilities that integrate with the existing
simulation infrastructure, eliminating code duplication.
"""

from .surface_parser import Surface, SurfaceType, SurfaceParser
from .navigation_graph import NavigationNode, NavigationGraphBuilder, JumpTrajectory, JumpCalculator
from .astar_pathfinder import PlatformerAStar, MultiObjectivePathfinder
from .dynamic_pathfinding import TemporalNode, EnemyPredictor, DynamicPathfinder
from .path_executor import PathOptimizer, MovementController
from .pathfinding_visualizer import PathfindingVisualizer
from .pathfinding_system import PathfindingSystem
from .pathfinding_tester import PathfindingTester
from .utils import PathfindingUtils, create_enemy_from_entity_data
from .collision import CollisionChecker
from .entity_wrapper import EntityWrapper, EntityManager

# Legacy compatibility
try:
    from .utils import Enemy
except ImportError:
    # Enemy class has been removed, provide a compatibility stub
    class Enemy:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Enemy class is deprecated. Use EntityWrapper instead.")

__all__ = [
    "Surface", "SurfaceType", "SurfaceParser",
    "NavigationNode", "NavigationGraphBuilder", "JumpTrajectory", "JumpCalculator",
    "PlatformerAStar", "MultiObjectivePathfinder",
    "TemporalNode", "EnemyPredictor", "DynamicPathfinder",
    "PathOptimizer", "MovementController",
    "PathfindingVisualizer",
    "PathfindingSystem",
    "PathfindingTester",
    "CollisionChecker", "Enemy",
    "PathfindingUtils",
    "EntityWrapper",
    "EntityManager",
    "create_enemy_from_entity_data"  # Legacy compatibility
]
