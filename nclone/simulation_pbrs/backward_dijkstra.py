"""Backward Dijkstra for computing frame-count distance fields.

Given a forward adjacency graph (source -> [(dest, frame_count)]) produced
by TransitionEnumerator, this module builds the reverse graph and runs
Dijkstra from goal cell(s) to compute the minimum frame cost from every
reachable cell to the goal.

The result is a 2D distance array d[ix, iy] = minimum frames from (ix, iy)
to the goal. Unreachable cells get d = d_max (the maximum finite distance
found), not infinity, following the pbrs-perfect-graph.md recommendation.
"""

import heapq
import logging
import time
from typing import Dict, List, Tuple, Optional

import numpy as np

from .grid_classifier import GRID_WIDTH, GRID_HEIGHT, GridClassifier

logger = logging.getLogger(__name__)

# Type aliases
AdjacencyDict = Dict[Tuple[int, int], List[Tuple[Tuple[int, int], int]]]


def _build_reverse_graph(forward_adj: AdjacencyDict) -> AdjacencyDict:
    """Build reverse adjacency from forward transitions.

    If cell A reaches cell B in k frames (forward edge A->B weight k),
    then in the reverse graph B->A has weight k. This allows Dijkstra
    starting from the goal to compute distances FROM every cell TO the goal.

    Args:
        forward_adj: Forward adjacency dict from TransitionEnumerator.

    Returns:
        Reverse adjacency dict mapping dest_cell -> [(src_cell, frame_count), ...].
    """
    reverse: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], int]]] = {}

    for src, neighbors in forward_adj.items():
        for dest, cost in neighbors:
            if dest not in reverse:
                reverse[dest] = []
            reverse[dest].append((src, cost))

    return reverse


def backward_dijkstra(
    forward_adj: AdjacencyDict,
    goal_pixel: Tuple[float, float],
    valid_mask: np.ndarray,
    search_radius_cells: int = 3,
) -> np.ndarray:
    """Compute minimum frame-count distance from every cell to a goal.

    Builds the reverse graph, finds the goal cell(s), and runs Dijkstra
    to flood-fill distances from the goal outward. Returns a 2D array
    where d[ix, iy] = minimum frames from cell (ix, iy) to the goal.

    Args:
        forward_adj: Forward adjacency from TransitionEnumerator.enumerate().
        goal_pixel: Goal position in pixel coordinates (x, y).
        valid_mask: Boolean array (GRID_WIDTH, GRID_HEIGHT) of valid cells.
        search_radius_cells: Radius (in cells) to search for goal cell if
            the exact cell has no reverse edges.

    Returns:
        Float32 numpy array (GRID_WIDTH, GRID_HEIGHT) of frame-count distances.
        Unreachable cells are set to the maximum finite distance found (d_max).
    """
    start_time = time.perf_counter()

    # Build reverse graph
    reverse_adj = _build_reverse_graph(forward_adj)

    # Find the goal cell
    goal_ix, goal_iy = GridClassifier.pixel_to_cell(goal_pixel[0], goal_pixel[1])

    # Find valid start cells for Dijkstra near the goal position.
    # The goal might not be exactly on a graph node, so search nearby cells.
    start_cells = _find_goal_cells(
        goal_ix, goal_iy, reverse_adj, valid_mask, search_radius_cells
    )

    if not start_cells:
        logger.warning(
            f"No reachable cells found near goal ({goal_pixel[0]:.0f}, "
            f"{goal_pixel[1]:.0f}) -> cell ({goal_ix}, {goal_iy}). "
            f"Distance field will be all-max."
        )
        # Return array filled with a large value
        return np.full((GRID_WIDTH, GRID_HEIGHT), 1e6, dtype=np.float32)

    # Run Dijkstra from goal cells (multi-source)
    dist = _dijkstra(reverse_adj, start_cells)

    # Convert to 2D array
    dist_grid = np.full((GRID_WIDTH, GRID_HEIGHT), np.inf, dtype=np.float32)
    for (ix, iy), d in dist.items():
        if 0 <= ix < GRID_WIDTH and 0 <= iy < GRID_HEIGHT:
            dist_grid[ix, iy] = d

    # Replace inf with d_max (maximum finite distance)
    finite_mask = np.isfinite(dist_grid)
    if finite_mask.any():
        d_max = float(dist_grid[finite_mask].max())
    else:
        d_max = 1e6
    dist_grid[~finite_mask] = d_max

    elapsed = time.perf_counter() - start_time
    reachable_count = int(finite_mask.sum())
    logger.info(
        f"Backward Dijkstra: {reachable_count} reachable cells, "
        f"d_max={d_max:.0f} frames, {elapsed:.3f}s"
    )

    return dist_grid


def _find_goal_cells(
    goal_ix: int,
    goal_iy: int,
    reverse_adj: AdjacencyDict,
    valid_mask: np.ndarray,
    search_radius: int,
) -> List[Tuple[int, int]]:
    """Find valid goal cells near the target position.

    The goal entity position might not align exactly with a cell that
    has reverse edges. Search in expanding radius to find nearby cells
    that exist in the reverse graph.

    Args:
        goal_ix, goal_iy: Target cell indices.
        reverse_adj: Reverse adjacency graph.
        valid_mask: Cell validity mask.
        search_radius: Maximum search radius in cells.

    Returns:
        List of goal cells to seed Dijkstra from. May contain multiple
        cells if the goal straddles cell boundaries.
    """
    # Try exact cell first
    if (goal_ix, goal_iy) in reverse_adj or (
        valid_mask[goal_ix, goal_iy]
        if 0 <= goal_ix < GRID_WIDTH and 0 <= goal_iy < GRID_HEIGHT
        else False
    ):
        return [(goal_ix, goal_iy)]

    # Expand search radius
    for r in range(1, search_radius + 1):
        candidates = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue  # Only check the ring at distance r
                nx, ny = goal_ix + dx, goal_iy + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                    if valid_mask[nx, ny] and (nx, ny) in reverse_adj:
                        candidates.append((nx, ny))
        if candidates:
            return candidates

    # Last resort: any valid cell in reverse_adj near the goal
    best = None
    best_dist = float("inf")
    for cell in reverse_adj:
        dx = cell[0] - goal_ix
        dy = cell[1] - goal_iy
        d = dx * dx + dy * dy
        if d < best_dist:
            best_dist = d
            best = cell
    if best is not None:
        logger.warning(
            f"Goal cell ({goal_ix}, {goal_iy}) not in graph. "
            f"Using nearest cell {best} (dist={best_dist**0.5:.1f} cells)."
        )
        return [best]

    return []


def _dijkstra(
    reverse_adj: AdjacencyDict,
    start_cells: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], float]:
    """Run multi-source Dijkstra on the reverse graph.

    Args:
        reverse_adj: Reverse adjacency dict (dest -> [(src, cost), ...]).
        start_cells: Goal cells to start from (distance 0).

    Returns:
        Dict mapping cell -> minimum distance (in frames) from that cell to goal.
    """
    dist: Dict[Tuple[int, int], float] = {}
    heap = []

    # Seed with goal cells at distance 0
    for cell in start_cells:
        dist[cell] = 0.0
        heapq.heappush(heap, (0.0, cell))

    while heap:
        d, cell = heapq.heappop(heap)

        # Skip if we already found a shorter path
        if d > dist.get(cell, float("inf")):
            continue

        # Expand neighbors in reverse graph
        # In the reverse graph, cell's neighbors are cells that can REACH cell
        # (i.e., cells from which the ninja can transition TO this cell)
        for neighbor, cost in reverse_adj.get(cell, []):
            new_dist = d + cost
            if new_dist < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))

    return dist
