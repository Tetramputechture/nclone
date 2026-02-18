"""Simulation-based transition enumeration with multi-velocity sweep.

Enumerates all physics transitions (cell, action, initial_velocity) -> dest_cell
by running the actual physics simulator. Each valid cell is simulated from 5
representative initial velocities x 6 actions x 50 frame durations.

The multi-velocity sweep discovers momentum-dependent edges (running jumps,
momentum carries) that zero-velocity-only initialization would miss. Edges
are merged via minimum weight -- if multiple initial velocities reach the same
destination from the same source, the cheapest frame count wins.

The output is a forward adjacency dict suitable for building a reverse graph
for backward Dijkstra.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

from .grid_classifier import (
    GridClassifier,
    GRID_RESOLUTION,
    GRID_WIDTH,
    GRID_HEIGHT,
)
from ..constants.physics_constants import MAX_HOR_SPEED

logger = logging.getLogger(__name__)

# Maximum number of frames to simulate per (cell, action, velocity) combo.
# 45 frames = MAX_JUMP_DURATION, 50 gives margin for landing.
MAX_FRAMES = 50

# The 6 possible actions: (horizontal_input, jump_input)
# horizontal_input in {-1, 0, 1}, jump_input in {0, 1}
ACTIONS = [
    (0, 0),   # NOOP
    (-1, 0),  # LEFT
    (1, 0),   # RIGHT
    (0, 1),   # JUMP
    (-1, 1),  # JUMP+LEFT
    (1, 1),   # JUMP+RIGHT
]

# Representative initial velocities for multi-velocity sweep.
# These capture the most important momentum-dependent transitions:
# - Rest: baseline (no momentum)
# - Full speed left/right: running jumps, momentum carries
# - Just jumped: upward trajectory from a prior jump
# - Falling fast: entering a cell from above with significant downward speed
INITIAL_VELOCITIES = [
    (0.0, 0.0),                # rest
    (MAX_HOR_SPEED, 0.0),     # full speed right (~3.333 px/frame)
    (-MAX_HOR_SPEED, 0.0),    # full speed left
    (0.0, -2.0),              # just jumped (upward)
    (0.0, 3.0),               # falling fast (survivable)
]

# Type alias for the adjacency structure
# Maps source cell -> list of (dest_cell, frame_count)
AdjacencyDict = Dict[Tuple[int, int], List[Tuple[Tuple[int, int], int]]]


class TransitionEnumerator:
    """Enumerates physics transitions for all valid cells.

    For each valid cell, simulates all combinations of initial velocities,
    actions, and frame durations to discover reachable destination cells
    and their frame costs.

    Uses the simulator's tick_physics_only() for lightweight physics
    evaluation, with ninja.save_state()/restore_state() for stateless queries.
    """

    def __init__(
        self,
        max_frames: int = MAX_FRAMES,
        initial_velocities: Optional[List[Tuple[float, float]]] = None,
    ):
        """Initialize the transition enumerator.

        Args:
            max_frames: Maximum frames to simulate per (cell, action, velocity).
            initial_velocities: List of (vx, vy) initial velocity tuples.
                Defaults to INITIAL_VELOCITIES (5 representative velocities).
        """
        self.max_frames = max_frames
        self.initial_velocities = initial_velocities or INITIAL_VELOCITIES

    def enumerate(
        self,
        sim,
        valid_mask: np.ndarray,
        progress_callback=None,
    ) -> AdjacencyDict:
        """Enumerate all transitions for valid cells via physics simulation.

        For each valid cell, for each initial velocity, for each action:
        simulate up to max_frames frames incrementally, recording the
        destination cell at each frame. Edges are deduplicated by keeping
        the minimum frame count for each (source, dest) pair.

        Args:
            sim: Simulator instance with loaded level (must have ninja).
            valid_mask: Boolean array (GRID_WIDTH, GRID_HEIGHT) from GridClassifier.
            progress_callback: Optional callable(cells_done, total_cells) for progress.

        Returns:
            Forward adjacency dict mapping source cell -> [(dest_cell, frame_count), ...].
            Edges are deduplicated: each (source, dest) pair appears at most once
            with the minimum frame count across all velocities and actions.
        """
        ninja = sim.ninja
        if ninja is None:
            raise ValueError("Simulator must have a ninja (call sim.reset() first)")

        # Collect valid cells
        valid_cells = []
        for ix in range(GRID_WIDTH):
            for iy in range(GRID_HEIGHT):
                if valid_mask[ix, iy]:
                    valid_cells.append((ix, iy))

        total_cells = len(valid_cells)
        logger.info(
            f"Enumerating transitions for {total_cells} valid cells "
            f"({len(self.initial_velocities)} velocities x {len(ACTIONS)} actions "
            f"x {self.max_frames} frames = "
            f"{total_cells * len(self.initial_velocities) * len(ACTIONS) * self.max_frames:,} "
            f"total physics ticks)"
        )

        # Raw edge storage: (src, dst) -> min_frame_count
        raw_edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}

        start_time = time.perf_counter()
        last_log_time = start_time

        for cell_idx, (ix, iy) in enumerate(valid_cells):
            px, py = GridClassifier.cell_to_pixel(ix, iy)

            self._enumerate_cell_transitions(
                sim, ninja, px, py, ix, iy, valid_mask, raw_edges
            )

            # Progress reporting
            if progress_callback and (cell_idx + 1) % 500 == 0:
                progress_callback(cell_idx + 1, total_cells)

            now = time.perf_counter()
            if now - last_log_time > 30.0:
                elapsed = now - start_time
                rate = (cell_idx + 1) / elapsed
                remaining = (total_cells - cell_idx - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  Progress: {cell_idx + 1}/{total_cells} cells "
                    f"({100 * (cell_idx + 1) / total_cells:.1f}%), "
                    f"elapsed={elapsed:.0f}s, ETA={remaining:.0f}s"
                )
                last_log_time = now

        elapsed = time.perf_counter() - start_time

        # Convert raw edges to adjacency dict
        adjacency = self._build_adjacency(raw_edges)

        total_edges = sum(len(neighbors) for neighbors in adjacency.values())
        logger.info(
            f"Transition enumeration complete: {total_cells} cells, "
            f"{total_edges:,} edges, {elapsed:.1f}s"
        )

        return adjacency

    def _enumerate_cell_transitions(
        self,
        sim,
        ninja,
        px: float,
        py: float,
        ix: int,
        iy: int,
        valid_mask: np.ndarray,
        raw_edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int],
    ) -> None:
        """Enumerate all transitions from a single cell.

        Iterates over initial velocities and actions, simulating incrementally
        and recording discovered edges.

        Args:
            sim: Simulator instance
            ninja: Ninja object from sim
            px, py: Pixel center of the cell
            ix, iy: Grid cell indices
            valid_mask: Cell validity mask
            raw_edges: Mutable dict to accumulate edges into
        """
        src_cell = (ix, iy)

        for vx0, vy0 in self.initial_velocities:
            for h_input, jump_input in ACTIONS:
                # Save simulator state
                snapshot = ninja.save_state()

                try:
                    # Inject starting state
                    self._inject_state(sim, ninja, px, py, vx0, vy0)

                    # Track the last recorded destination to avoid duplicate
                    # edges at the same destination for consecutive frames
                    last_dest = None

                    for frame in range(1, self.max_frames + 1):
                        sim.tick_physics_only(h_input, jump_input)

                        # Check for death
                        if ninja.state in (6, 7):
                            break

                        # Get destination cell
                        dest_ix, dest_iy = GridClassifier.pixel_to_cell(
                            ninja.xpos, ninja.ypos
                        )

                        # Check bounds and validity
                        if (
                            0 <= dest_ix < GRID_WIDTH
                            and 0 <= dest_iy < GRID_HEIGHT
                            and valid_mask[dest_ix, dest_iy]
                        ):
                            dest_cell = (dest_ix, dest_iy)

                            # Skip if same as source at frame 1 (no movement)
                            # and skip duplicate consecutive destinations
                            if dest_cell != src_cell and dest_cell != last_dest:
                                edge_key = (src_cell, dest_cell)
                                existing = raw_edges.get(edge_key)
                                if existing is None or frame < existing:
                                    raw_edges[edge_key] = frame
                                last_dest = dest_cell
                        else:
                            # Ninja left the valid area, stop this trajectory
                            break
                finally:
                    ninja.restore_state(snapshot)

    def _inject_state(
        self,
        sim,
        ninja,
        x: float,
        y: float,
        vx: float,
        vy: float,
    ) -> None:
        """Inject position and velocity state into ninja.

        Uses native sim's batch inject when available (faster). Otherwise
        sets Python ninja properties directly.

        Args:
            sim: Simulator instance (Python or NativeSimAdapter)
            ninja: Ninja object (or proxy)
            x, y: Position in pixels
            vx, vy: Velocity in pixels/frame
        """
        if hasattr(sim, "inject_ninja_for_enumeration"):
            sim.inject_ninja_for_enumeration(x, y, vx, vy)
            return

        from ..constants.physics_constants import GRAVITY_FALL

        ninja.xpos = x
        ninja.ypos = y
        ninja.xspeed = vx
        ninja.yspeed = vy
        ninja.jump_duration = 0

        # Previous position (for collision detection)
        ninja.xpos_old = x - vx
        ninja.ypos_old = y - vy
        ninja.xspeed_old = vx
        ninja.yspeed_old = vy

        # Default to falling state -- physics tick will detect ground contact
        ninja.state = 4  # Falling
        ninja.applied_gravity = GRAVITY_FALL
        ninja.airborn = True

        # Reset collision state
        ninja.floor_count = 0
        ninja.wall_count = 0
        ninja.ceiling_count = 0
        ninja.walled = False
        ninja.wall_normal = 0

        # Reset buffers
        ninja.jump_buffer = -1
        ninja.floor_buffer = -1
        ninja.wall_buffer = -1
        ninja.launch_pad_buffer = -1

        # Reset crush state
        ninja.is_crushable = False
        ninja.x_crush = 0
        ninja.y_crush = 0
        ninja.crush_len = 0

        # Reset death tracking
        ninja.terminal_impact = False
        ninja.death_cause = None

    @staticmethod
    def _build_adjacency(
        raw_edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int],
    ) -> AdjacencyDict:
        """Convert raw edge dict to adjacency list format.

        Args:
            raw_edges: Dict mapping (src_cell, dst_cell) -> min_frame_count

        Returns:
            Adjacency dict mapping src_cell -> [(dst_cell, frame_count), ...]
        """
        adjacency: AdjacencyDict = defaultdict(list)

        for (src, dst), frame_count in raw_edges.items():
            adjacency[src].append((dst, frame_count))

        return dict(adjacency)
