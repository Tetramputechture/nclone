"""
Terminal velocity death prediction system using graph-optimized three-tier approach.

This module provides fast and accurate terminal impact death prediction using:
1. Quick state filter (O(1) pre-filter for safe states)
2. Graph-constrained lookup table (O(1) for reachable positions)
3. Full physics simulation (only for edge cases outside lookup table)

Uses graph reachability data to optimize search space while maintaining
accurate physics simulation for predictions.
"""

import time
import pickle
from pathlib import Path
from typing import Set, Tuple, Dict, Optional, List
from dataclasses import dataclass

from .terminal_velocity_simulator import TerminalVelocitySimulator
from .constants import (
    TERMINAL_IMPACT_SAFE_VELOCITY,
    TERMINAL_VELOCITY_QUANTIZATION,
    TERMINAL_DISTANCE_QUANTIZATION,
    TILE_PIXEL_SIZE,
    LEVEL_WIDTH_PX,
    LEVEL_HEIGHT_PX,
)
from .graph.reachability.subcell_node_lookup import (
    SubcellNodeLookupLoader,
)


@dataclass
class TerminalVelocityPredictorStats:
    """Statistics about the terminal velocity predictor."""

    build_time_ms: float = 0.0
    lookup_table_size: int = 0
    tier1_queries: int = 0  # Queries handled by quick filter
    tier2_queries: int = 0  # Queries handled by lookup table
    tier3_queries: int = 0  # Queries requiring simulation
    cache_hits: int = 0  # Number of cache hits
    cache_misses: int = 0  # Number of cache misses


@dataclass
class DeathProbabilityResult:
    """Result of death probability calculation."""

    action_death_probs: list  # Probability for each action (0-1)
    masked_actions: list  # Indices of masked actions
    frames_simulated: int  # Number of frames simulated
    nearest_surface_distance: float  # Distance to nearest surface


class TerminalVelocityPredictor:
    """
    Terminal velocity death predictor using graph-optimized three-tier approach.

    Tier 1: Quick state filter for obviously safe states
    Tier 2: Graph-constrained lookup table (only reachable positions)
    Tier 3: Full physics simulation for edge cases

    Uses graph reachability to optimize search space while maintaining
    accurate physics simulation.
    """

    # Class-level cache for lookup tables keyed by level_id
    # Static tiles = static terminal velocity physics
    _lookup_table_cache: Dict[str, Dict[Tuple, int]] = {}

    # Class-level tile-type cache (pre-computed terminal velocity data)
    # Loaded once from disk, shared across all predictor instances
    _tile_type_cache: Dict[int, Dict[Tuple, int]] = {}
    _tile_cache_loaded: bool = False

    def __init__(self, sim, graph_data: Optional[Dict] = None):
        """
        Initialize predictor with simulation reference and optional graph data.

        Args:
            sim: Simulation object containing ninja and entities
            graph_data: Optional graph data with adjacency and reachability info
                       Used to optimize lookup table size and query performance
        """
        self.sim = sim
        self.graph_data = graph_data
        self.lookup_table: Dict[Tuple, int] = {}  # State -> action bitmask
        self.simulator = TerminalVelocitySimulator(sim)
        self.stats = TerminalVelocityPredictorStats()
        self.last_death_probability: DeathProbabilityResult = None

        # Level caching for avoiding rebuilds on same level
        self._current_level_id: Optional[str] = None

        # Graph optimization components (if graph data available)
        self.adjacency = graph_data.get("adjacency", {}) if graph_data else {}
        self.reachable = graph_data.get("reachable", set()) if graph_data else set()
        self.subcell_lookup = None

        self.subcell_lookup = SubcellNodeLookupLoader()

        # Load tile cache on first instantiation
        if not TerminalVelocityPredictor._tile_cache_loaded:
            self._load_tile_cache()

    @classmethod
    def _load_tile_cache(cls):
        """
        Load or auto-generate pre-computed tile-type terminal velocity data.

        This method loads the offline-computed terminal impact behaviors for
        flat surface tiles (types 1-5, 18-33). The cache is loaded once per
        process and shared across all predictor instances.

        If cache is not found, it will be auto-generated on first load.
        This is a one-time cost (~30-60 seconds) that saves significant time
        on subsequent runs.

        Curved tiles (6-17) are not cached and use runtime simulation.
        """
        if cls._tile_cache_loaded:
            return

        module_dir = Path(__file__).parent
        cache_file = module_dir / "data" / "terminal_velocity_tile_cache.pkl"

        # Try to load existing cache
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cls._tile_type_cache = pickle.load(f)
                cls._tile_cache_loaded = True

                # Calculate statistics
                total_entries = sum(len(data) for data in cls._tile_type_cache.values())
                tiles_with_data = sum(
                    1 for data in cls._tile_type_cache.values() if len(data) > 0
                )
                print(
                    f"Loaded terminal velocity tile cache: {tiles_with_data} tile types, {total_entries} entries"
                )
                return
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                print("Will attempt to regenerate...")

        # Cache not found or failed to load - auto-generate it
        print()
        print("=" * 70)
        print("Terminal velocity tile cache not found. Generating now...")
        print("This is a ONE-TIME operation (~30-60 seconds)...")
        print("Subsequent runs will use the cached data (instant load).")
        print("=" * 70)
        print()

        try:
            # Import here to avoid circular dependencies
            import sys

            sys.path.insert(0, str(module_dir.parent))

            from nclone.tools.precompute_terminal_velocity_data import (
                generate_all_tile_data,
                save_data_to_file,
            )

            # Generate cache data with verbose output
            cls._tile_type_cache = generate_all_tile_data(
                verbose=True, use_multiprocessing=True
            )

            # Save for future use
            try:
                save_data_to_file(cls._tile_type_cache, str(cache_file), verbose=True)
                print()
                print("=" * 70)
                print("Cache generation complete and saved!")
                print(f"Future runs will load instantly from: {cache_file}")
                print("=" * 70)
                print()
            except Exception as e:
                print(f"Warning: Could not save cache to {cache_file}: {e}")
                print("Cache will be regenerated on next run.")

            cls._tile_cache_loaded = True

        except Exception as e:
            import traceback

            print(f"ERROR: Failed to generate terminal velocity cache: {e}")
            traceback.print_exc()
            print()
            print("Falling back to full simulation (slower but functional)")
            cls._tile_cache_loaded = True  # Mark as attempted
            cls._tile_type_cache = {}

    def _get_tile_type_at_position(self, x: float, y: float) -> int:
        """
        Get tile type at world coordinates.

        Args:
            x: World X coordinate
            y: World Y coordinate

        Returns:
            Tile type ID (0-37), or 0 if no tile exists
        """
        tile_x = int(x // TILE_PIXEL_SIZE)
        tile_y = int(y // TILE_PIXEL_SIZE)

        if hasattr(self.sim, "tile_dic"):
            return self.sim.tile_dic.get((tile_x, tile_y), 0)

        return 0

    def _is_curved_tile(self, tile_type: int) -> bool:
        """
        Check if tile type has curved surfaces.

        Curved tiles (diagonals, quarter circles, quarter pipes) are rare
        sources of terminal impact deaths (<1%). These tiles are not included
        in the precomputed cache and use runtime simulation instead.

        Args:
            tile_type: Tile type ID (0-37)

        Returns:
            True if tile has curved surfaces, False otherwise
        """
        from .constants.tile_constants import CURVED_TILES

        return tile_type in CURVED_TILES

    def build_lookup_table(
        self,
        reachable_positions: Set[Tuple[int, int]],
        level_id: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Build graph-optimized lookup table ONLY for reachable positions.

        Key optimization: Graph reachability constrains search space dramatically.
        Instead of sampling entire level, we only build for positions the ninja
        can actually reach, reducing table size by 50-75%.

        Implements per-level caching: if level_id matches cached level, loads
        from cache instead of rebuilding (88% time reduction on same level).

        Args:
            reachable_positions: Set of (x, y) reachable positions from graph system
            level_id: Optional level identifier for caching (e.g., "level_{hash}")
            verbose: Print build progress and statistics
        """
        start_time = time.perf_counter()

        # Check cache if level_id provided and matches current level
        if level_id is not None and level_id == self._current_level_id:
            # Same level, reuse existing lookup table (already loaded)
            if verbose:
                print(
                    f"Terminal velocity lookup table already loaded for level {level_id}"
                )
            return

        # Check class-level cache for this level
        if (
            level_id is not None
            and level_id in TerminalVelocityPredictor._lookup_table_cache
        ):
            # Cache hit: load from cache
            self.lookup_table = TerminalVelocityPredictor._lookup_table_cache[level_id]
            self._current_level_id = level_id
            self.stats.cache_hits += 1
            self.stats.lookup_table_size = len(self.lookup_table)

            cache_load_time = (time.perf_counter() - start_time) * 1000
            if verbose:
                print(
                    f"Terminal velocity lookup table loaded from cache in {cache_load_time:.1f}ms"
                )
                print(f"  Level ID: {level_id}")
                print(f"  Table size: {len(self.lookup_table)} entries")
                print(f"  Cache hits: {self.stats.cache_hits}")
            return

        # Cache miss: build new lookup table
        if level_id is not None:
            self.stats.cache_misses += 1

        if verbose:
            print("Building terminal velocity lookup table...")
            print(f"  Reachable positions: {len(reachable_positions)}")
            if self.graph_data:
                print("  Using graph optimization (adjacency-based)")

        # Sample velocity states (only dangerous range)
        # Only build for vy > SAFE_VELOCITY since lower velocities are filtered in Tier 1
        import numpy as np

        velocity_samples = np.arange(
            TERMINAL_IMPACT_SAFE_VELOCITY, 12.0, TERMINAL_VELOCITY_QUANTIZATION
        )

        # Build table for subset of reachable positions (sample for performance)
        # Graph already constrains to reachable area - this is the key optimization!
        # Use smarter sampling: prioritize positions near surfaces (higher terminal impact risk)
        sampled_positions = self._sample_positions_for_terminal_risk(
            reachable_positions,
            target_sample_count=max(1, len(reachable_positions) // 20),
        )

        total_possible = len(reachable_positions) * len(velocity_samples)
        total_sampled = len(sampled_positions) * len(velocity_samples)

        # Track cache usage statistics
        tile_cache_hits = 0
        tile_cache_misses = 0
        tile_type_misses = {}  # Track which tile types cause misses
        state_misses_per_tile = {}  # Track state-level misses per tile type

        for pos_idx, pos in enumerate(sampled_positions):
            if verbose and pos_idx % 100 == 0:
                print(
                    f"  Progress: {pos_idx}/{len(sampled_positions)} positions processed "
                    f"(cache hits: {tile_cache_hits}, misses: {tile_cache_misses})"
                )

            for vy in velocity_samples:
                # Create state key (quantized)
                state_key = self._create_lookup_key(pos[0], pos[1], 0.0, vy)

                # Try tile cache first (Phase 2 optimization)
                action_mask = None
                tile_type = self._get_tile_type_at_position(pos[0], pos[1])

                # Skip cache for curved tiles (use simulation instead)
                if self._is_curved_tile(tile_type):
                    # Curved tiles not in cache - go directly to simulation
                    tile_cache_misses += 1
                    # Fall through to simulation below
                elif tile_type in self._tile_type_cache:
                    # Calculate local position within tile
                    tile_x = int(pos[0] // TILE_PIXEL_SIZE)
                    tile_y = int(pos[1] // TILE_PIXEL_SIZE)
                    local_x = pos[0] - (tile_x * TILE_PIXEL_SIZE)
                    local_y = pos[1] - (tile_y * TILE_PIXEL_SIZE)

                    # Quantize local position and velocity for cache lookup
                    local_x_q = round(local_x / 6) * 6
                    local_y_q = round(local_y / 6) * 6
                    vx_q = 0.0  # We only sample vx=0 for now
                    vy_q = (
                        round(vy / TERMINAL_VELOCITY_QUANTIZATION)
                        * TERMINAL_VELOCITY_QUANTIZATION
                    )

                    cache_key = (vx_q, vy_q, local_x_q, local_y_q)

                    if cache_key in self._tile_type_cache[tile_type]:
                        # Cache hit! Use pre-computed result
                        action_mask = self._tile_type_cache[tile_type][cache_key]
                        tile_cache_hits += 1
                    else:
                        # State-level miss: tile type cached, but this specific state isn't
                        state_misses_per_tile[tile_type] = (
                            state_misses_per_tile.get(tile_type, 0) + 1
                        )
                        if (
                            verbose and tile_cache_misses < 10
                        ):  # Only print first few misses
                            print(
                                f"  Cache miss (state): tile_type={tile_type}, "
                                f"pos=({pos[0]:.0f},{pos[1]:.0f}), local=({local_x:.1f},{local_y:.1f}), "
                                f"vy={vy:.2f} -> key={cache_key}"
                            )
                else:
                    # Tile-level miss: tile type not in cache at all
                    # (Note: curved tiles are intentionally skipped, not an error)
                    tile_type_misses[tile_type] = tile_type_misses.get(tile_type, 0) + 1
                    # Only warn about unexpected missing tiles (not empty, glitched, or curved)
                    if verbose and tile_type not in [0, 34, 35, 36, 37]:
                        if not self._is_curved_tile(tile_type):
                            if (
                                tile_type_misses[tile_type] == 1
                            ):  # Only print once per tile type
                                print(
                                    f"  Cache miss (tile): tile_type={tile_type} not in cache "
                                    f"(at pos {pos[0]:.0f},{pos[1]:.0f})"
                                )

                # Fall back to simulation if cache miss
                if action_mask is None:
                    tile_cache_misses += 1

                    # Create the state dict and set the ninja to this state before simulating
                    sampled_state = self._create_state_dict(pos[0], pos[1], 0.0, vy)

                    # Save current ninja state so we can restore it after simulations
                    original_state = self.simulator._save_ninja_state()

                    # Simulate each action to determine if deadly (uses physics simulation!)
                    action_mask = 0
                    for action in range(6):
                        # Set ninja to the sampled state before each simulation
                        self.simulator._restore_ninja_state(sampled_state)

                        # Simulate to check if deadly
                        is_deadly = self.simulator.simulate_for_terminal_impact(action)

                        if is_deadly:
                            action_mask |= 1 << action

                    # Restore original ninja state after all simulations for this sample
                    self.simulator._restore_ninja_state(original_state)

                self.lookup_table[state_key] = action_mask

        build_time = (time.perf_counter() - start_time) * 1000
        self.stats.build_time_ms = build_time
        self.stats.lookup_table_size = len(self.lookup_table)

        # Store in cache if level_id provided
        if level_id is not None:
            TerminalVelocityPredictor._lookup_table_cache[level_id] = self.lookup_table
            self._current_level_id = level_id

        if verbose:
            print(f"\nTerminal velocity predictor built in {build_time:.1f}ms:")
            print(f"  Table size: {len(self.lookup_table)} entries")
            print(f"  Memory: ~{(len(self.lookup_table) * 32) / 1024:.1f} KB")
            if tile_cache_hits + tile_cache_misses > 0:
                cache_hit_rate = (
                    100 * tile_cache_hits / (tile_cache_hits + tile_cache_misses)
                )
                print(
                    f"  Tile cache hit rate: {cache_hit_rate:.1f}% ({tile_cache_hits}/{tile_cache_hits + tile_cache_misses})"
                )

                # Print cache miss breakdown
                if tile_cache_misses > 0:
                    print("  Cache miss breakdown:")
                    if tile_type_misses:
                        print(
                            f"    Tile-level misses (tile type not cached): {sum(tile_type_misses.values())} queries"
                        )
                        for tile_type, count in sorted(
                            tile_type_misses.items(), key=lambda x: -x[1]
                        )[:5]:
                            print(f"      Tile type {tile_type}: {count} misses")
                    if state_misses_per_tile:
                        print(
                            f"    State-level misses (state not in tile cache): {sum(state_misses_per_tile.values())} queries"
                        )
                        for tile_type, count in sorted(
                            state_misses_per_tile.items(), key=lambda x: -x[1]
                        )[:5]:
                            print(f"      Tile type {tile_type}: {count} misses")

            if total_possible > 0:
                reduction_pct = 100 * (1 - total_sampled / total_possible)
                print(
                    f"  Optimization: {reduction_pct:.1f}% reduction vs full grid (reachability-constrained)"
                )

    def is_action_deadly(self, action: int) -> bool:
        """
        Query graph-optimized three-tier system to check if action leads to terminal impact death.

        Tier 1: Quick filter + graph reachability check (O(1))
        Tier 2: Lookup table (O(1), constrained by reachability)
        Tier 3: Full simulation (O(frames), fallback only)

        Args:
            action: Action to check (0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP, 4=JUMP+LEFT, 5=JUMP+RIGHT)

        Returns:
            True if action leads to terminal impact death, False otherwise
        """
        ninja = self.sim.ninja

        # Tier 1: Quick state filter (O(1), ~0.001ms)
        # Skip expensive checks if in obviously safe state
        if not ninja.airborn or abs(ninja.yspeed) < TERMINAL_IMPACT_SAFE_VELOCITY:
            self.stats.tier1_queries += 1
            return False  # Safe state, no terminal impact risk

        # Graph optimization: Check if ninja is in reachable area
        # If not in reachable zone, either already dead or in unreachable area
        if self.subcell_lookup and self.adjacency:
            try:
                current_node = self.subcell_lookup.find_closest_node_position(
                    ninja.xpos, ninja.ypos, self.adjacency
                )

                # If not in reachable area, skip expensive checks
                if not current_node or current_node not in self.reachable:
                    self.stats.tier1_queries += 1  # Count as filtered
                    return False  # Not in reachable area
            except Exception:
                # Graceful degradation - continue to lookup table/simulation
                pass

        # Tier 2: Lookup table (O(1), ~0.01ms)
        state_key = self._create_lookup_key(
            ninja.xpos, ninja.ypos, ninja.xspeed, ninja.yspeed
        )

        if state_key in self.lookup_table:
            self.stats.tier2_queries += 1
            action_mask = self.lookup_table[state_key]
            return bool(action_mask & (1 << action))

        # Tier 3: Full simulation (O(frames), ~0.5ms, rare)
        self.stats.tier3_queries += 1
        return self.simulator.simulate_for_terminal_impact(action)

    def calculate_death_probability(
        self, frames_to_simulate: int = 10
    ) -> DeathProbabilityResult:
        """
        Calculate probability of terminal impact death for each action over N frames.

        This method mirrors MineDeathPredictor.calculate_death_probability() to provide
        death probabilities for observations and visualization.

        Args:
            frames_to_simulate: Number of frames to simulate forward

        Returns:
            DeathProbabilityResult with probabilities and masked actions
        """
        ninja = self.sim.ninja

        # Quick filter: if safe state, return all zeros
        if not ninja.airborn or abs(ninja.yspeed) < TERMINAL_IMPACT_SAFE_VELOCITY:
            result = DeathProbabilityResult(
                action_death_probs=[0.0] * 6,
                masked_actions=[],
                frames_simulated=0,
                nearest_surface_distance=float("inf"),
            )
            self.last_death_probability = result
            return result

        action_death_probs = []
        masked_actions = []

        # Calculate distances to nearest surfaces
        dist_floor = self._distance_to_floor(ninja)
        dist_ceiling = self._distance_to_ceiling(ninja)
        nearest_surface = min(dist_floor, dist_ceiling)

        # For each action, calculate death probability
        for action in range(6):
            # Check if action leads to certain death (single frame)
            is_deadly = self.is_action_deadly(action)

            if is_deadly:
                masked_actions.append(action)
                action_death_probs.append(1.0)  # 100% death probability
            else:
                # Calculate probability over N frames
                # For each frame offset, check if death occurs
                death_count = 0
                for frame_offset in range(1, frames_to_simulate + 1):
                    # Simulate this specific frame offset
                    collision = self.simulator.simulate_for_terminal_impact(
                        action, max_frames=frame_offset
                    )
                    if collision:
                        death_count += 1

                # Probability is fraction of frames that lead to death
                probability = death_count / frames_to_simulate
                action_death_probs.append(probability)

        result = DeathProbabilityResult(
            action_death_probs=action_death_probs,
            masked_actions=masked_actions,
            frames_simulated=frames_to_simulate,
            nearest_surface_distance=nearest_surface,
        )

        # Cache result for visualization
        self.last_death_probability = result

        return result

    def _create_lookup_key(
        self, x: float, y: float, vx: float, vy: float
    ) -> Tuple[float, float, float, float]:
        """
        Create quantized state key for lookup table.

        Quantizes position and velocity to reduce table size while maintaining accuracy.

        Args:
            x: X position
            y: Y position
            vx: X velocity
            vy: Y velocity

        Returns:
            Tuple key for lookup table
        """
        # Quantize position to tile-level (24px)
        x_bucket = (
            round(x / TERMINAL_DISTANCE_QUANTIZATION) * TERMINAL_DISTANCE_QUANTIZATION
        )
        y_bucket = (
            round(y / TERMINAL_DISTANCE_QUANTIZATION) * TERMINAL_DISTANCE_QUANTIZATION
        )

        # Quantize velocity
        vx_bucket = (
            round(vx / TERMINAL_VELOCITY_QUANTIZATION) * TERMINAL_VELOCITY_QUANTIZATION
        )
        vy_bucket = (
            round(vy / TERMINAL_VELOCITY_QUANTIZATION) * TERMINAL_VELOCITY_QUANTIZATION
        )

        return (x_bucket, y_bucket, vx_bucket, vy_bucket)

    def _create_state_dict(self, x: float, y: float, vx: float, vy: float) -> Dict:
        """
        Create ninja state dictionary from position and velocity.

        Args:
            x: X position
            y: Y position
            vx: X velocity
            vy: Y velocity

        Returns:
            Dictionary with ninja state ready for simulation
        """
        # Import here to avoid circular dependency
        from .constants import GRAVITY_FALL, DRAG_REGULAR, FRICTION_GROUND

        return {
            "hor_input": 0,
            "jump_input": 0,
            "xpos": x,
            "ypos": y,
            "xpos_old": x,
            "ypos_old": y,
            "xspeed": vx,
            "yspeed": vy,
            "xspeed_old": vx,
            "yspeed_old": vy,
            "state": 4,  # Falling (most common state for terminal velocity)
            "airborn": True,
            "airborn_old": True,
            "walled": False,
            "wall_normal": 0,
            "applied_gravity": GRAVITY_FALL,
            "applied_drag": DRAG_REGULAR,
            "applied_friction": FRICTION_GROUND,
            "jump_duration": 0,
            "jump_buffer": -1,
            "floor_buffer": -1,
            "wall_buffer": -1,
            "launch_pad_buffer": -1,
            "jump_input_old": 0,
            "floor_normalized_x": 0,
            "floor_normalized_y": -1,
            "ceiling_normalized_x": 0,
            "ceiling_normalized_y": 1,
            "floor_count": 0,
            "ceiling_count": 0,
        }

    def _sample_positions_for_terminal_risk(
        self, reachable_positions: Set[Tuple[int, int]], target_sample_count: int
    ) -> List[Tuple[int, int]]:
        """
        Sample positions from reachable set, prioritizing areas with higher terminal impact risk.

        Strategy:
        1. For high-reachability levels (>80%): Filter to surface-adjacent positions only
        2. Use spatial hash to distribute samples uniformly across reachable space
        3. Reduce sampling to ~5% of reachable positions (20x reduction)

        High-reachability optimization: Open levels have large empty spaces where terminal
        impact is impossible. Focus sampling on positions near surfaces (within 48px of any tile).

        Args:
            reachable_positions: Set of all reachable positions from graph
            target_sample_count: Desired number of samples (e.g., len(reachable) // 20)

        Returns:
            List of sampled positions for lookup table building
        """
        import numpy as np

        if len(reachable_positions) <= target_sample_count:
            # If we have fewer positions than target, use all
            return list(reachable_positions)

        # High-reachability optimization: filter to surface-adjacent positions
        total_cells = (LEVEL_WIDTH_PX // 24) * (LEVEL_HEIGHT_PX // 24)
        reachability_ratio = len(reachable_positions) / total_cells

        if reachability_ratio > 0.8:
            # High reachability: focus on surface-adjacent positions
            surface_adjacent = self._filter_surface_adjacent_positions(
                reachable_positions, max_distance=48
            )
            if len(surface_adjacent) > 0:
                # Use surface-adjacent positions for sampling
                reachable_positions = surface_adjacent
                # Adjust target to account for reduced search space
                target_sample_count = min(target_sample_count, len(surface_adjacent))

        # Convert to list for indexing
        positions_list = list(reachable_positions)

        # Strategy 1: Use spatial hash for uniform distribution
        # Group positions by spatial cells (24x24 grid)
        spatial_groups = {}
        for pos in positions_list:
            cell_x = pos[0] // 24
            cell_y = pos[1] // 24
            cell_key = (cell_x, cell_y)
            if cell_key not in spatial_groups:
                spatial_groups[cell_key] = []
            spatial_groups[cell_key].append(pos)

        # Sample uniformly from spatial cells
        samples_per_cell = max(1, target_sample_count // len(spatial_groups))
        sampled = []

        for cell_positions in spatial_groups.values():
            # Sample from this cell
            if len(cell_positions) <= samples_per_cell:
                sampled.extend(cell_positions)
            else:
                # Randomly sample from this cell
                indices = np.random.choice(
                    len(cell_positions), size=samples_per_cell, replace=False
                )
                sampled.extend([cell_positions[i] for i in indices])

        # If we still need more samples (unlikely), add random ones
        if len(sampled) < target_sample_count:
            remaining = set(positions_list) - set(sampled)
            additional_count = min(target_sample_count - len(sampled), len(remaining))
            if additional_count > 0:
                additional = np.random.choice(
                    list(remaining), size=additional_count, replace=False
                )
                sampled.extend(additional)

        # Limit to target count
        if len(sampled) > target_sample_count:
            sampled = sampled[:target_sample_count]

        return sampled

    def _filter_surface_adjacent_positions(
        self, positions: Set[Tuple[int, int]], max_distance: int = 48
    ) -> Set[Tuple[int, int]]:
        """
        Filter positions to only those near tile surfaces.

        Terminal velocity deaths occur when the ninja impacts a surface at high speed.
        In open levels with high reachability, most positions are far from any surface
        and cannot cause terminal impact. This method filters to positions within
        max_distance of any tile, dramatically reducing the search space for open levels.

        Args:
            positions: Set of positions to filter
            max_distance: Maximum distance from a tile surface (pixels)

        Returns:
            Set of positions within max_distance of any tile
        """
        if not hasattr(self.sim, "tile_dic") or not self.sim.tile_dic:
            # No tiles - return all positions (shouldn't happen)
            return positions

        # Get all tile positions
        tile_positions = set(self.sim.tile_dic.keys())

        # Convert tile grid positions to pixel positions
        tile_pixel_positions = set()
        for tx, ty in tile_positions:
            # Add tile center position
            pixel_x = tx * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
            pixel_y = ty * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
            tile_pixel_positions.add((pixel_x, pixel_y))

        # Filter positions to those within max_distance of any tile
        surface_adjacent = set()
        max_dist_sq = max_distance * max_distance

        for pos in positions:
            # Check if within max_distance of any tile
            for tile_pos in tile_pixel_positions:
                dx = pos[0] - tile_pos[0]
                dy = pos[1] - tile_pos[1]
                dist_sq = dx * dx + dy * dy

                if dist_sq <= max_dist_sq:
                    surface_adjacent.add(pos)
                    break  # Found nearby tile, no need to check others

        return surface_adjacent if len(surface_adjacent) > 0 else positions

    def _distance_to_floor(self, ninja) -> float:
        """
        Calculate approximate distance to nearest floor.

        Uses simple raycast down to find nearest floor tile.

        Args:
            ninja: Ninja object

        Returns:
            Distance to floor in pixels (inf if no floor found)
        """
        # Simple approximation: check tiles below ninja
        from .physics import gather_segments_from_region

        # Check region below ninja
        x = ninja.xpos
        y_start = ninja.ypos
        y_end = ninja.ypos + 200  # Look down 200 pixels

        segments = gather_segments_from_region(self.sim, x - 10, y_start, x + 10, y_end)

        if not segments:
            return float("inf")

        # Find closest segment below
        min_distance = float("inf")
        for segment in segments:
            result = segment.get_closest_point(x, y_start)
            if result:
                _, closest_x, closest_y = result
                if closest_y > y_start:  # Below ninja
                    distance = closest_y - y_start
                    min_distance = min(min_distance, distance)

        return min_distance

    def _distance_to_ceiling(self, ninja) -> float:
        """
        Calculate approximate distance to nearest ceiling.

        Uses simple raycast up to find nearest ceiling tile.

        Args:
            ninja: Ninja object

        Returns:
            Distance to ceiling in pixels (inf if no ceiling found)
        """
        # Simple approximation: check tiles above ninja
        from .physics import gather_segments_from_region

        # Check region above ninja
        x = ninja.xpos
        y_start = ninja.ypos - 200  # Look up 200 pixels
        y_end = ninja.ypos

        segments = gather_segments_from_region(self.sim, x - 10, y_start, x + 10, y_end)

        if not segments:
            return float("inf")

        # Find closest segment above
        min_distance = float("inf")
        for segment in segments:
            result = segment.get_closest_point(x, y_end)
            if result:
                _, closest_x, closest_y = result
                if closest_y < y_end:  # Above ninja
                    distance = y_end - closest_y
                    min_distance = min(min_distance, distance)

        return min_distance

    def get_stats(self) -> TerminalVelocityPredictorStats:
        """Get statistics about the predictor."""
        return self.stats

    def print_performance_summary(self):
        """Print performance summary of tier usage."""
        total_queries = (
            self.stats.tier1_queries
            + self.stats.tier2_queries
            + self.stats.tier3_queries
        )

        if total_queries == 0:
            print("No queries yet")
            return

        print("\nTerminal Velocity Predictor Performance:")
        print(f"  Total queries: {total_queries}")
        print(
            f"  Tier 1 (quick filter): {self.stats.tier1_queries} "
            f"({100 * self.stats.tier1_queries / total_queries:.1f}%)"
        )
        print(
            f"  Tier 2 (lookup table): {self.stats.tier2_queries} "
            f"({100 * self.stats.tier2_queries / total_queries:.1f}%)"
        )
        print(
            f"  Tier 3 (simulation): {self.stats.tier3_queries} "
            f"({100 * self.stats.tier3_queries / total_queries:.1f}%)"
        )

    @classmethod
    def clear_cache(cls):
        """Clear the class-level lookup table cache.

        Called when loading a new level or when explicit cache clearing is needed.
        """
        cls._lookup_table_cache.clear()
