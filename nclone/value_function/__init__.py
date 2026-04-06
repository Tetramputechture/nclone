"""Surface-graph PBRS: exact cost-to-go via physics-based surface graphs.

This module precomputes optimal cost-to-go from every walkable surface
state to level goals (switch and exit door) using a velocity-augmented
surface graph and Dijkstra's algorithm.  The result is used as a
Potential-Based Reward Shaping (PBRS) signal during RL training.

Architecture overview:
  1. **Surface extraction** -- classify tile collision segments as floor /
     wall, merge into runs, sample SurfacePoint positions.
  2. **Surface graph** -- build velocity-augmented nodes (position x
     discretised velocity), connect via physics-simulated edges.
  3. **Dijkstra** -- backward shortest-path from goal nodes gives exact
     frame-count cost-to-go d*[node].
  4. **Arc cache** -- spatial hash mapping airborne states to precomputed
     cost estimates for O(1) runtime lookup.
  5. **Runtime PBRS** -- per-frame potential Phi(s) = -d*/d_max queried
     by the reward calculator in the gym environment.

Key components:
  - PhysicsStepWrapper: Stateless interface to the Rust physics engine.
  - SurfaceGraphPBRS / PrecomputedValuePBRS: Runtime potential lookup.
  - TwoPhaseValuePBRS / create_two_phase_pbrs: Two-goal (switch + exit) support.
  - extract_surfaces, build_graph, enumerate_edges: Precomputation pipeline.
  - ArcCache, AirborneEvaluator: Airborne state cost estimation.
"""

from .physics_step_wrapper import PhysicsStepWrapper
from .precomputed_pbrs import SurfaceGraphPBRS, PrecomputedValuePBRS
from .two_phase_pbrs import TwoPhaseValuePBRS, create_two_phase_pbrs, create_neural_pbrs
from .surface_extraction import (
    SurfacePoint,
    extract_surfaces,
    SURFACE_FLOOR,
    SURFACE_WALL,
)
from .surface_graph import (
    SurfaceGraph,
    build_graph,
    compute_cost_to_go,
    compute_cost_to_go_with_tree,
    dijkstra_backward_with_tree,
    find_nearest_point,
    find_nearest_node,
    lookup_cost_interpolated,
)
from .edge_enumeration import (
    enumerate_edges,
    populate_graph_edges,
)
from .arc_cache import ArcCache
from .airborne_evaluator import AirborneEvaluator
from .route_extraction import (
    ArcTarget,
    extract_route,
    extract_and_simplify_route,
    merge_running_arcs,
    find_nearest_start_node,
)
from .route_pbrs import RoutePBRS, RouteState, create_route_pbrs

__all__ = [
    # Physics interface
    "PhysicsStepWrapper",
    # Runtime PBRS
    "SurfaceGraphPBRS",
    "PrecomputedValuePBRS",
    "TwoPhaseValuePBRS",
    "create_two_phase_pbrs",
    "create_neural_pbrs",
    # Route-tracking PBRS (new)
    "RoutePBRS",
    "RouteState",
    "create_route_pbrs",
    # Route extraction (new)
    "ArcTarget",
    "extract_route",
    "extract_and_simplify_route",
    "merge_running_arcs",
    "find_nearest_start_node",
    # Surface extraction
    "SurfacePoint",
    "extract_surfaces",
    "SURFACE_FLOOR",
    "SURFACE_WALL",
    # Surface graph
    "SurfaceGraph",
    "build_graph",
    "compute_cost_to_go",
    "compute_cost_to_go_with_tree",
    "dijkstra_backward_with_tree",
    "find_nearest_point",
    "find_nearest_node",
    "lookup_cost_interpolated",
    # Edge enumeration
    "enumerate_edges",
    "populate_graph_edges",
    # Arc cache & airborne
    "ArcCache",
    "AirborneEvaluator",
]
