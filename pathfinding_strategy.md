# NClone Pathfinding Strategy

This document outlines the architecture and strategy for the pathfinding system in NClone, designed to emulate N++ mechanics as closely as possible.

## 1. System Architecture

The pathfinding system is composed of several key components that work in sequence:

1.  **Tile Map Input**: The system takes a 2D NumPy array representing the level's tile map as input.
2.  **Surface Parser (`surface_parser.py`)**:
    *   Parses the tile map to identify all distinct traversable surfaces (floors, ceilings, walls, slopes).
    *   Initially breaks down tiles into small, 8px line segments based on `tile_definitions.py`.
    *   Merges adjacent, co-linear segments of the same type (e.g., multiple small floor segments from adjacent solid tiles) to form larger, continuous surfaces.
3.  **Navigation Graph Builder (`navigation_graph.py`)**:
    *   Constructs a navigation graph (`nx.DiGraph`) based on the parsed surfaces.
    *   **Node Creation**: Places nodes on surfaces, typically at start, end, and intermediate points for longer surfaces.
    *   **Edge Creation**:
        *   **Walk/Run Edges**: Connects nodes on the same surface if they are directly traversable by walking or running.
        *   **Gap-Crossing Edges**: Connects nodes across small horizontal gaps (e.g., <= 8px) between co-linear floor or ceiling surfaces. This helps bridge minor discontinuities that might arise from tile definitions.
        *   **Jump/Fall Edges**: Added by `N2PlusPathfindingSystem` using the `JumpCalculator`. These represent potential jumps or falls between nodes on different surfaces or distant parts of the same surface.
4.  **Jump Calculator (`navigation_graph.py`)**:
    *   Simulates N++ jump physics to determine if a valid jump trajectory exists between two points (graph nodes).
    *   Uses precise N++ physics constants for gravity, acceleration, drag, jump impulses, etc.
    *   Considers different jump strategies (e.g., varying jump hold durations).
    *   Checks for collisions during the simulated jump.
5.  **A\* Pathfinder (`astar_pathfinder.py`)**:
    *   Implements the A\* search algorithm to find the optimal path (sequence of nodes) between a start and goal node in the navigation graph.
    *   Uses a heuristic (e.g., Euclidean distance) and considers edge weights (e.g., travel time or distance).
    *   Can be extended for multi-objective pathfinding (e.g., visiting a switch then an exit).
6.  **Path Optimizer (`path_executor.py`)**:
    *   Takes the raw node path from A\*.
    *   **Smoothing (Line-of-Sight)**: Applies string pulling (LoS checks) to remove redundant waypoints, creating a more direct path.
    *   **Smoothing (Curve - Stubbed)**: Placeholder for Bezier or Catmull-Rom spline smoothing if needed for more fluid movement (currently returns LoS path).
7.  **Movement Controller (`path_executor.py`)**:
    *   Converts the optimized list of world coordinate waypoints into a sequence of frame-by-frame N++ input commands (jump, left, right).
    *   This component is crucial for translating the path into actionable game inputs. Its internal logic for command generation based on N++ physics is currently a mix of stubs and basic implementations.
8.  **Dynamic Pathfinding Components (`dynamic_pathfinding.py`)**:
    *   **Enemy Predictor**: Predicts future enemy positions.
    *   **Dynamic Pathfinder**: Modifies pathfinding considerations based on predicted enemy movements, aiming for safe paths. (Integrates with A* or uses a time-augmented A*).

## 2. N++ Physics Integration

The system strives for accuracy by incorporating N++ physics constants directly into relevant components:

*   **`JumpCalculator`**: Uses constants like `GRAVITY_JUMP`, `GRAVITY_FALL`, `AIR_ACCEL`, `MAX_HOR_SPEED`, `FLOOR_JUMP_VELOCITY_Y`, `WALL_JUMP_VELOCITY_Y`, etc., for realistic jump simulations.
*   **`MovementController`**: Also references these constants (e.g., `NINJA_MAX_SPEED_X_GROUND` (as `ninja_max_speed`)) for estimating frames and generating movement commands.
*   **`CollisionChecker` (`utils.py`)**: Uses tile definitions and N++ collision system logic (e.g. player AABB against tile types).

(A full list of constants can be found in `navigation_graph.py` and `path_executor.py`).

## 3. Key Strategies and Implementations

### 3.1. Surface Parsing and Merging

*   Tiles are initially decomposed into 8px segments based on definitions in an external `tile_definitions.py` file (e.g., a solid tile might yield 3 floor, 3 ceiling, 3 left-wall, and 3 right-wall segments).
*   The `SurfaceParser._merge_adjacent_surfaces` method then groups these segments by type (floor, ceiling, wall) and sorts them.
*   For orthogonal surfaces (floors, ceilings, walls), it iterates through sorted segments and merges them if they are co-linear and directly adjacent (end of one meets start of next). This creates longer, more representative surfaces.
*   **Current Limitation**: The accuracy of merged surfaces heavily depends on the completeness of segment definitions in `tile_definitions.py`. If, for example, a solid tile definition only provides segments covering 0-16px of its 24px width, an 8px gap will remain between surfaces derived from adjacent tiles, preventing a full merge.

### 3.2. Navigation Graph Construction

*   **Nodes**: `NavigationGraphBuilder` places nodes at the start and end of each merged surface. For surfaces longer than a threshold (e.g., 48px), intermediate nodes are also added.
*   **Walk/Run Edges**: Bidirectional edges connect consecutive nodes on the same surface.
*   **Gap-Crossing Edges**: `NavigationGraphBuilder._create_gap_crossing_edges` specifically addresses the fragmentation issue potentially caused by incomplete `tile_definitions.py`. It adds bidirectional "walk_gap" edges between the end node of one floor/ceiling surface and the start node of the next if they are co-linear and separated by a small horizontal gap (e.g., <= 8px).
*   **Jump/Fall Edges**: `N2PlusPathfindingSystem._add_jump_edges_to_graph` iterates through pairs of graph nodes and uses `JumpCalculator.calculate_jump` to find valid jump trajectories. If a valid jump is found, an edge representing this jump (with cost, type, and trajectory data) is added to the graph. This process is computationally intensive (N^2 node pairs) and currently includes heuristics to limit attempts (e.g., max distance).

### 3.3. Pathfinding and Execution

*   **Path Search**: `PlatformerAStar.find_path` searches the `nav_graph`.
*   **Path Smoothing**: `PathOptimizer.smooth_path` uses line-of-sight checks. Bezier smoothing is stubbed.
*   **Command Generation**: `MovementController.generate_commands` takes the smoothed path and edge types to produce a list of input commands. This is currently simplified; for instance, 'walk_gap' edges are treated like 'walk' edges, and the underlying command generation logic for different movement types (walk, jump) is basic.

## 4. Known Issues and Future Work

*   **`tile_definitions.py` Dependency**: The system's ability to form continuous surfaces is highly sensitive to the definitions in `tile_definitions.py`. Incomplete definitions (e.g., not covering the full 24px extent of a tile face) lead to fragmented surfaces. While gap-crossing edges mitigate this for simple cases, it's a fundamental data issue.
*   **`JumpCalculator` Robustness**:
    *   Currently, `_add_jump_edges_to_graph` adds 0 jump edges in basic test cases, with `JumpCalculator` reporting no valid trajectories for many node pairs, including seemingly simple short hops. This needs further investigation and tuning to ensure it can find paths for all N++ feasible jumps.
    *   The N^2 approach for finding jump edges is computationally expensive and needs optimization (e.g., spatial indexing, better heuristics for candidate pairs).
*   **`MovementController` Implementation**:
    *   The methods for generating commands for different movement types (`_generate_walk_run_commands`, `_generate_abstract_jump_commands`, etc.) are largely stubs or highly simplified. They need to be implemented with detailed N++ physics simulations to produce accurate and effective frame-by-frame inputs.
    *   Handling of current velocity and reactive control during path execution is not yet implemented.
*   **`PathOptimizer`**: Bezier smoothing is a stub and needs a proper implementation if desired.
*   **Dynamic Pathfinding**: While placeholders exist, full integration and testing of enemy prediction and dynamic A* adjustments are future work.
*   **Complex Geometries**: Handling of slopes and curved surfaces in `SurfaceParser` merging and `NavigationGraphBuilder` (especially for node placement and edge creation) needs further refinement and testing. Current gap-crossing is only for co-linear horizontal surfaces.
*   **Node/Edge Count Discrepancy (Resolved)**: Initial confusion about node counts was resolved; `_create_surface_nodes` correctly adds 2 nodes for surfaces shorter than the intermediate node threshold.

This strategy provides a foundation for N++ style pathfinding. Future efforts will focus on fleshing out the stubbed components, refining the physics simulations within `JumpCalculator` and `MovementController`, and addressing the `tile_definitions.py` data dependency for robust surface generation.
