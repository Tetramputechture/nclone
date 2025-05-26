# Pathfinding Strategy

## 1. System Architecture

The pathfinding system comprises several key components operating sequentially:

1.  **Tile Map Input**: The system utilizes a 2D NumPy array representing the level\'s tile map.
2.  **Surface Parser (`surface_parser.py`)**:
    *   Identifies distinct traversable surfaces (floors, ceilings, walls, slopes) from the tile map.
    *   Initially decomposes tiles into 8px line segments based on `tile_definitions.py`.
    *   Merges adjacent, co-linear segments of the same type (floors, ceilings, walls, and now co-linear slopes of similar angles) to form continuous surfaces.
3.  **Navigation Graph Builder (`navigation_graph.py`)**:
    *   Constructs a navigation graph (`nx.DiGraph`) from the parsed surfaces.
    *   **Node Creation**: Places nodes on surfaces at start, end, and intermediate points for longer surfaces.
    *   **Edge Creation**:
        *   **Walk/Run Edges**: Connect nodes on the same surface traversable by walking or running. These edges now include an estimated `frames` attribute representing time cost.
        *   **Gap-Crossing Edges**: Connect nodes across small horizontal gaps (e.g., <= 8px) between co-linear floor or ceiling surfaces. These edges also now include an estimated `frames` attribute.
        *   **Jump/Fall Edges**: Added by `N2PlusPathfindingSystem` via the `JumpCalculator`. The process of adding these edges is now spatially optimized to reduce N^2 complexity.
4.  **Jump Calculator (`navigation_graph.py`)**:
    *   Simulates N++ jump physics to determine if a valid jump trajectory exists between two graph nodes.
    *   Utilizes N++ physics constants (gravity, acceleration, drag, jump impulses).
    *   Considers a wider variety of jump strategies:
        *   Differing jump hold durations (including very short "tap" jumps).
        *   Initial horizontal running velocities (standstill, half-speed, full-speed) for floor/slope jumps.
        *   Distinct "normal" and "slide" wall jump mechanics with appropriate X-velocities.
        *   Uses surface normals for jumps off slopes to provide more accurate jump-off angles and velocities.
    *   Performs collision checks during simulated jumps.
5.  **A\* Pathfinder (`astar_pathfinder.py`)**:
    *   Implements the A\* search algorithm to find an optimal path (sequence of nodes) in the navigation graph.
    *   Uses a heuristic (e.g., Euclidean distance) and considers edge weights (e.g., travel time/frames).
    *   Supports extension for multi-objective pathfinding.
6.  **Path Optimizer (`path_executor.py`)**:
    *   Processes the raw node path from A\*.
    *   **Smoothing (Line-of-Sight)**: Applies string pulling (LoS checks) to remove redundant waypoints, creating a more direct path.
    *   **Smoothing (Curve - Bezier)**: Implements Bezier curve smoothing (C2-continuous interpolation) on the LoS path to create smoother trajectories.
7.  **Movement Controller (`path_executor.py`)**:
    *   Converts the optimized list of world coordinate waypoints into a sequence of frame-by-frame N++ input commands (jump, left, right).
    *   Command generation for walk/run segments now incorporates more detailed N++ ground physics (acceleration, friction, max speed) and propagates the agent's kinematic state (position, velocity) between segments.
    *   Command generation for jump and fall segments is still largely abstract and requires further development to precisely follow `JumpTrajectory` data.
8.  **Dynamic Pathfinding Components (`dynamic_pathfinding.py`)**:
    *   **Enemy Predictor**:
        *   Predicts future enemy positions.
        *   Includes more detailed predictive models for Thwumps (cyclic rest-smash-wait-return pattern) and Drones (patrol path following if data is available).
        *   Uses linear interpolation between sampled prediction points for smoother and more accurate enemy tracking.
    *   **Dynamic Pathfinder**:
        *   Modifies pathfinding based on predicted enemy movements for safer paths using temporal A\* search (A\* on a time-expanded graph).
        *   Utilizes time costs (`frames` attribute) from graph edges and tuned parameters for `time_resolution` and heuristics.

## 2. N++ Physics Integration

The system incorporates N++ physics constants into relevant components for accuracy:

*   **`JumpCalculator`**: Uses constants such as `GRAVITY_JUMP`, `GRAVITY_FALL`, `AIR_ACCEL`, `MAX_HOR_SPEED`, `FLOOR_JUMP_VELOCITY_Y`, `WALL_JUMP_VELOCITY_Y`, `WALL_JUMP_VELOCITY_X_NORMAL`, `WALL_JUMP_VELOCITY_X_SLIDE`. Also considers surface normals for slope jumps to align with N++ jump mechanics.
*   **`MovementController`**: References constants (e.g., `GROUND_ACCEL`, `FRICTION_GROUND`, `MAX_HOR_SPEED`) for its enhanced walk/run command generation. Other movement types are planned to use these more deeply.
*   **`CollisionChecker` (`utils.py`)**: Employs tile definitions and N++ collision system logic (e.g., ninja radius).

(A comprehensive list of constants is available in `navigation_graph.py` and `path_executor.py`).

## 3. Core Mechanisms

### 3.1. Surface Parsing and Merging

*   Tiles are decomposed into 8px segments based on definitions in `tile_definitions.py`.
*   `SurfaceParser._merge_adjacent_surfaces` groups segments by type and merges co-linear, adjacent orthogonal segments (floors, ceilings, walls) and co-linear slopes of similar angles.
*   The accuracy of merged surfaces depends on the completeness of segment definitions in `tile_definitions.py`. Incomplete definitions can result in gaps between surfaces derived from adjacent tiles.

### 3.2. Navigation Graph Construction

*   **Nodes**: `NavigationGraphBuilder` places nodes at the start and end of each merged surface, with intermediate nodes for surfaces exceeding a length threshold.
*   **Walk/Run Edges**: Bidirectional edges connect consecutive nodes on the same surface. These edges now store an estimated time cost in `frames`.
*   **Gap-Crossing Edges**: `NavigationGraphBuilder._create_gap_crossing_edges` adds bidirectional "walk_gap" edges between co-linear floor/ceiling surfaces separated by small horizontal gaps. These edges also store an estimated `frames` cost.
*   **Jump/Fall Edges**: `N2PlusPathfindingSystem._add_jump_edges_to_graph` uses `JumpCalculator.calculate_jump` to find valid jump trajectories. This process is now spatially optimized using a grid-based approach to reduce the number of pairwise checks. The `JumpCalculator` itself considers a wider variety of N++ jump mechanics, including initial run velocities and slope-adjusted jumps.

### 3.3. Pathfinding and Execution

*   **Path Search**: `PlatformerAStar.find_path` (and `DynamicPathfinder`) search the `nav_graph`. `DynamicPathfinder` uses edge `frames` for temporal search.
*   **Path Smoothing**: `PathOptimizer.smooth_path` utilizes line-of-sight checks followed by Bezier curve smoothing.
*   **Command Generation**: `MovementController.generate_commands` converts the smoothed path and edge types into input commands. Walk/run command generation is now more physics-aware. Command generation for other movement types (jumps, falls) has an improved structure for state propagation but the detailed input logic is still largely abstract.

## 4. Current Limitations and Potential Enhancements

*   **`tile_definitions.py` Dependency**: The system's surface generation is sensitive to definitions in `tile_definitions.py`. Incomplete definitions can lead to fragmented surfaces, partially mitigated by gap-crossing edges and merging of co-linear slopes. Robust parsing for all tile types (especially curved surfaces) and more advanced merging/repair logic remain areas for improvement.
*   **`JumpCalculator` Robustness**:
    *   The `JumpCalculator` has been enhanced to identify a wider variety of N++ jumps (running starts, slide wall jumps, varied hold times, slope-adjusted jumps), but continuous tuning and testing may be needed for edge cases or highly complex jump scenarios.
    *   The N^2 complexity for finding jump edges has been significantly optimized using spatial indexing in `N2PlusPathfindingSystem`.
*   **`MovementController` Implementation**:
    *   Walk/run command generation now uses more detailed N++ physics for acceleration, friction, and speed limits. Kinematic state (position, velocity) is propagated between segments.
    *   Command generation methods for jump and fall segments are still largely abstract. Future work should focus on implementing controllers that can accurately follow the `JumpTrajectory` data (frame-by-frame positions/velocities) provided by `JumpCalculator` for these aerial maneuvers.
    *   True reactive control during path execution (re-planning based on real-time deviations) remains a future architectural enhancement.
*   **`PathOptimizer`**: Bezier curve smoothing has been implemented. Catmull-Rom splines could be explored as an alternative if Bezier curves prove unsuitable for certain N++ path characteristics.
*   **Dynamic Pathfinding**: `EnemyPredictor` now includes more detailed predictive models for Thwumps (cyclic) and Drones (patrol-based if data is available), and uses interpolation for position queries. `DynamicPathfinder` (temporal A\*) parameters have been tuned, and it utilizes time costs from graph edges. Further refinement of enemy prediction models (e.g., player-aware Death Balls, more drone behaviors) is possible.
*   **Complex Geometries**:
    *   Handling of slopes has been improved: `SurfaceParser` now attempts to merge co-linear slopes of similar angles, and `JumpCalculator` uses slope normals for more accurate jump-off physics.
    *   Parsing and handling of curved surfaces (e.g., quarter-pipes from `TILE_SEGMENT_CIRCULAR_MAP`) in `SurfaceParser` is still basic and requires full implementation.
    *   Node placement, edge creation (e.g., "slide" edges on curves), and movement control on curved surfaces are significant areas for future development.
    *   Gap-crossing logic in `NavigationGraphBuilder` is currently limited to co-linear horizontal surfaces and could be extended to other scenarios (e.g., small vertical steps, gaps between sloped surfaces).

This strategy forms a basis for N++ style pathfinding. Future work includes fully developing the remaining partially implemented components (especially detailed jump/fall execution in `MovementController` and curved surface handling), further refining physics simulations, and continuously improving the data dependency on `tile_definitions.py`.

