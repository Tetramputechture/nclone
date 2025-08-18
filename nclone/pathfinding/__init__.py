"""
Pathfinding module for N++ clone.

This module provides pathfinding utilities that integrate with the existing
simulation infrastructure, eliminating code duplication.
"""

from .surface_parser import Surface, SurfaceType, SurfaceParser
from .navigation_graph import NavigationNode, NavigationGraphBuilder, JumpTrajectory, JumpCalculator
from .astar_pathfinder import PlatformerAStar, MultiObjectivePathfinder
from .dynamic_pathfinding import TemporalNode, EntityPredictor, DynamicPathfinder
from .path_executor import PathOptimizer, MovementController
from .pathfinding_visualizer import PathfindingVisualizer
from .pathfinding_system import PathfindingSystem
from .pathfinding_tester import PathfindingTester
from .utils import PathfindingUtils
from .collision import CollisionChecker
from .entity_wrapper import EntityWrapper, EntityManager

__all__ = [
    "Surface", "SurfaceType", "SurfaceParser",
    "NavigationNode", "NavigationGraphBuilder", "JumpTrajectory", "JumpCalculator",
    "PlatformerAStar", "MultiObjectivePathfinder",
    "TemporalNode", "EntityPredictor", "DynamicPathfinder",
    "PathOptimizer", "MovementController",
    "PathfindingVisualizer",
    "PathfindingSystem",
    "PathfindingTester",
    "CollisionChecker",
    "PathfindingUtils",
    "EntityWrapper",
    "EntityManager",
]
