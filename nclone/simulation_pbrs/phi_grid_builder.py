"""Orchestrates the full simulation-based phi grid pipeline.

Pipeline steps:
1. GridClassifier: classify 176x100 cells as valid/invalid
2. TransitionEnumerator: enumerate physics transitions with multi-velocity sweep
3. BackwardDijkstra: compute frame-count distance fields from goals
4. Normalize: convert distances to phi potentials in [-1, 0]

Results are cached as .npz files keyed by a hash of the level geometry
and goal positions.
"""

import hashlib
import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from .grid_classifier import GridClassifier, GRID_WIDTH, GRID_HEIGHT
from .transition_enumerator import TransitionEnumerator
from .backward_dijkstra import backward_dijkstra

logger = logging.getLogger(__name__)


def _compute_cache_key(
    tile_dic: Dict[Tuple[int, int], int],
    switch_pos: Tuple[float, float],
    exit_pos: Tuple[float, float],
) -> str:
    """Compute a deterministic cache key from level data and goal positions.

    Args:
        tile_dic: Simulator tile dictionary {(col, row): tile_type}.
        switch_pos: Exit switch pixel position (x, y).
        exit_pos: Exit door pixel position (x, y).

    Returns:
        A hex string suitable for use as a filename.
    """
    # Deterministic serialization of tile data
    tile_bytes = str(sorted(tile_dic.items())).encode("utf-8")
    goal_bytes = f"{switch_pos[0]:.1f},{switch_pos[1]:.1f},{exit_pos[0]:.1f},{exit_pos[1]:.1f}".encode(
        "utf-8"
    )
    combined = tile_bytes + goal_bytes
    return hashlib.sha256(combined).hexdigest()[:16]


def _normalize_distance_to_phi(dist_grid: np.ndarray) -> np.ndarray:
    """Normalize a frame-count distance grid to phi potentials in [-1, 0].

    phi[ix, iy] = -d[ix, iy] / max_d

    where max_d is the maximum distance across all cells. This produces:
    - phi = 0 at the goal (d=0)
    - phi = -1 at the farthest reachable cell
    - Closer cells have higher (less negative) phi values

    Args:
        dist_grid: Float32 array (GRID_WIDTH, GRID_HEIGHT) of frame-count distances.

    Returns:
        Float32 array (GRID_WIDTH, GRID_HEIGHT) of phi values in [-1, 0].
    """
    max_d = float(dist_grid.max())
    if max_d <= 0:
        return np.zeros_like(dist_grid)
    return -(dist_grid / max_d).astype(np.float32)


class SimulationPhiGridBuilder:
    """Builds simulation-based phi grids for PBRS.

    Orchestrates the full pipeline from level data to normalized phi grids,
    with .npz disk caching for efficient reuse.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_frames: int = 50,
        initial_velocities=None,
    ):
        """Initialize the builder.

        Args:
            cache_dir: Directory to store/load cached phi grids. If None,
                no caching is performed.
            max_frames: Maximum frames per transition simulation.
            initial_velocities: Override for TransitionEnumerator's initial
                velocity set. Defaults to the standard 5-velocity sweep.
        """
        self.cache_dir = cache_dir
        self.enumerator = TransitionEnumerator(
            max_frames=max_frames,
            initial_velocities=initial_velocities,
        )

    def build(
        self,
        sim,
        switch_pos: Tuple[float, float],
        exit_pos: Tuple[float, float],
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build or load phi grids for a level.

        Attempts to load from cache first. If not cached (or force_recompute),
        runs the full pipeline: classify -> enumerate -> dijkstra -> normalize.

        Args:
            sim: Simulator instance with loaded level. Must have ninja
                (call sim.reset() first) and tile_dic populated.
            switch_pos: Exit switch pixel position (x, y).
            exit_pos: Exit door pixel position (x, y).
            force_recompute: If True, ignore cache and recompute.

        Returns:
            Tuple of (phi_switch, phi_exit), each a float32 numpy array
            of shape (GRID_WIDTH, GRID_HEIGHT) with values in [-1, 0].
        """
        cache_key = _compute_cache_key(sim.tile_dic, switch_pos, exit_pos)

        # Try loading from cache
        if not force_recompute and self.cache_dir:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        # Full pipeline
        total_start = time.perf_counter()

        # Step 1: Grid classification
        logger.info("Step 1/4: Classifying grid cells...")
        valid_mask = GridClassifier.classify(sim.tile_dic)
        valid_count = int(valid_mask.sum())
        logger.info(f"  {valid_count}/{GRID_WIDTH * GRID_HEIGHT} cells valid")

        # Step 2: Transition enumeration
        logger.info("Step 2/4: Enumerating transitions...")
        adjacency = self.enumerator.enumerate(sim, valid_mask)

        # Step 3: Backward Dijkstra from each goal
        logger.info("Step 3/4: Running backward Dijkstra from switch...")
        d_switch = backward_dijkstra(adjacency, switch_pos, valid_mask)

        logger.info("Step 3/4: Running backward Dijkstra from exit...")
        d_exit = backward_dijkstra(adjacency, exit_pos, valid_mask)

        # Step 4: Normalize to phi
        logger.info("Step 4/4: Normalizing to phi grids...")
        phi_switch = _normalize_distance_to_phi(d_switch)
        phi_exit = _normalize_distance_to_phi(d_exit)

        total_elapsed = time.perf_counter() - total_start
        logger.info(f"Phi grid build complete in {total_elapsed:.1f}s")

        # Cache to disk
        if self.cache_dir:
            self._save_to_cache(cache_key, phi_switch, phi_exit, d_switch, d_exit)

        return phi_switch, phi_exit

    def _load_from_cache(
        self, cache_key: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Try loading cached phi grids.

        Args:
            cache_key: Cache key string.

        Returns:
            (phi_switch, phi_exit) if found, None otherwise.
        """
        if not self.cache_dir:
            return None

        cache_path = os.path.join(self.cache_dir, f"sim_phi_{cache_key}.npz")
        if not os.path.exists(cache_path):
            return None

        try:
            data = np.load(cache_path)
            phi_switch = data["phi_switch"]
            phi_exit = data["phi_exit"]

            # Validate shape
            if phi_switch.shape != (GRID_WIDTH, GRID_HEIGHT):
                logger.warning(
                    f"Cached phi_switch shape {phi_switch.shape} != "
                    f"expected ({GRID_WIDTH}, {GRID_HEIGHT}), recomputing"
                )
                return None
            if phi_exit.shape != (GRID_WIDTH, GRID_HEIGHT):
                logger.warning(
                    f"Cached phi_exit shape {phi_exit.shape} != "
                    f"expected ({GRID_WIDTH}, {GRID_HEIGHT}), recomputing"
                )
                return None

            logger.info(f"Loaded cached phi grids from {cache_path}")
            return phi_switch, phi_exit
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None

    def _save_to_cache(
        self,
        cache_key: str,
        phi_switch: np.ndarray,
        phi_exit: np.ndarray,
        d_switch: np.ndarray,
        d_exit: np.ndarray,
    ) -> None:
        """Save phi grids to disk cache.

        Also saves raw distance grids for debugging/visualization.

        Args:
            cache_key: Cache key string.
            phi_switch: Normalized phi grid for switch goal.
            phi_exit: Normalized phi grid for exit goal.
            d_switch: Raw frame-count distance grid for switch.
            d_exit: Raw frame-count distance grid for exit.
        """
        if not self.cache_dir:
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"sim_phi_{cache_key}.npz")

        try:
            np.savez_compressed(
                cache_path,
                phi_switch=phi_switch,
                phi_exit=phi_exit,
                d_switch=d_switch,
                d_exit=d_exit,
            )
            logger.info(f"Saved phi grids to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")
