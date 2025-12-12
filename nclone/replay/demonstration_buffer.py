"""Demonstration buffer for mixing expert data with on-policy training.

Loads expert demonstrations from compact replay files and provides
(observation, action) pairs for behavioral cloning / demonstration augmentation.

Key Features:
- Loads compact binary replay files (.npp format)
- Simulates replays using ReplayExecutor to extract observations
- Architecture-aware observation filtering (matches policy modalities)
- Provides sampling interface compatible with SB3 rollout buffers
- Parallel loading with multiprocessing for faster ingestion

IMPORTANT: Demonstration Cache Invalidation
When observation space changes (e.g., privileged features 5→18 dims), cached
demonstrations become incompatible. Clear cache before training:
    rm -rf <replay_dir>/cache/*.npz
    rm -rf ../nclone/bc_replays/cache/*.npz

Usage:
    demo_buffer = DemonstrationBuffer(
        replay_paths,
        max_demos=100,
        architecture_config=arch_config,  # Optional: filters by modalities
        num_workers=4,  # Parallel loading with 4 workers
    )
    demo_batch = demo_buffer.sample_batch(batch_size=64)
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

# Import ReplayExecutor for simulating replays to extract observations
try:
    from nclone.replay.replay_executor import ReplayExecutor

    REPLAY_EXECUTOR_AVAILABLE = True
except ImportError:
    REPLAY_EXECUTOR_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default observation keys matching environment output
# These are all keys that ReplayExecutor and NppEnvironment can provide
DEFAULT_OBSERVATION_KEYS = [
    "game_state",
    "reachability_features",
    "switch_states",
    "graph_node_feats",
    "graph_edge_index",
    "graph_node_mask",
    "graph_edge_mask",
    "action_mask",
    "spatial_context",  # Graph-free spatial features (112 dims with velocity)
    "mine_sdf_features",  # CRITICAL: 3 dims for actor safety awareness (MISSING caused shape mismatch!)
]


# Maximum workers to prevent memory exhaustion (each worker creates a ReplayExecutor)
# Keep low because each worker holds observations in memory before returning
MAX_DEFAULT_WORKERS = 2

# Maximum replays per batch to limit memory usage in each worker
# Each replay can generate hundreds of transitions with large observation tensors
MAX_REPLAYS_PER_BATCH = 5


def _load_replay_batch_worker(
    replay_paths: List[str],
    observation_keys: List[str],
    max_transitions_per_demo: int,
    simulate_replays: bool,
    frame_skip: int = 1,
) -> List[Dict[str, Any]]:
    """Worker function for parallel replay loading (batch mode).

    This function runs in a separate process and creates ONE ReplayExecutor
    instance that is reused across all replays in the batch. This significantly
    reduces memory overhead compared to creating one executor per replay.

    Args:
        replay_paths: List of paths to replay files (as strings for pickling)
        observation_keys: Keys to extract from observations
        max_transitions_per_demo: Maximum transitions per demo
        simulate_replays: Whether to simulate replays for observations

    Returns:
        List of dicts with episode data (None entries for failed loads)
    """
    # Import here to avoid issues with multiprocessing
    from pathlib import Path

    from nclone.replay.gameplay_recorder import CompactReplay
    from nclone.replay.replay_executor import ReplayExecutor

    results = []

    # Create ONE executor for the entire batch (memory efficient)
    executor = None
    if simulate_replays:
        try:
            executor = ReplayExecutor(
                observation_config={},
                render_mode="grayscale_array",
                enable_rendering=False,
                frame_skip=frame_skip,
            )
        except Exception:
            executor = None

    for replay_path in replay_paths:
        replay_file = Path(replay_path)

        try:
            # Load binary replay
            with open(replay_file, "rb") as f:
                data = f.read()

            replay = CompactReplay.from_binary(data, episode_id=replay_file.stem)

            # Only load successful demonstrations
            if not replay.success:
                continue

            # Prepare result dict
            result = {
                "level_id": replay.level_id,
                "success": replay.success,
                "total_frames": len(replay.input_sequence),
                "map_data": replay.map_data,
                "input_sequence": replay.input_sequence,
                "transitions": [],
                "replay_name": replay_file.name,
            }

            # Simulate replay if executor is available
            if executor is not None:
                try:
                    # Simulate replay to extract observations
                    obs_results = executor.execute_replay(
                        map_data=replay.map_data,
                        input_sequence=replay.input_sequence,
                    )

                    # Create transitions with full observations
                    for frame_idx, obs_result in enumerate(obs_results):
                        if frame_idx >= max_transitions_per_demo:
                            break

                        # Extract and filter observation
                        obs = obs_result.get("observation", {})
                        action = obs_result.get("action", 0)

                        # Filter observation to requested keys
                        filtered_obs = {}
                        for key in observation_keys:
                            if key in obs:
                                value = obs[key]
                                if isinstance(value, np.ndarray):
                                    filtered_obs[key] = value.copy()
                                else:
                                    filtered_obs[key] = np.array(value)

                        result["transitions"].append(
                            {
                                "observation": filtered_obs,
                                "action": action,
                                "frame_idx": frame_idx,
                                "done": (frame_idx == len(obs_results) - 1),
                            }
                        )

                except Exception:
                    # If simulation fails, transitions list stays empty
                    result["transitions"] = []

            # If no transitions from simulation, store action-only transitions
            if not result["transitions"]:
                for frame_idx, action in enumerate(replay.input_sequence):
                    if frame_idx >= max_transitions_per_demo:
                        break

                    result["transitions"].append(
                        {
                            "observation": {},
                            "action": action,
                            "frame_idx": frame_idx,
                            "done": (frame_idx == len(replay.input_sequence) - 1),
                        }
                    )

            results.append(result)

        except Exception:
            # Skip failed replays
            continue

    # Cleanup executor
    if executor is not None:
        try:
            executor.close()
        except Exception:
            pass

    return results


@dataclass
class DemonstrationTransition:
    """A single demonstration transition (s, a, s', done)."""

    observation: Dict[str, np.ndarray]
    action: int
    next_observation: Optional[Dict[str, np.ndarray]] = None
    done: bool = False

    # Metadata
    episode_idx: int = 0
    frame_idx: int = 0
    level_id: Optional[str] = None


@dataclass
class DemonstrationEpisode:
    """A complete demonstration episode."""

    transitions: List[DemonstrationTransition]
    level_id: Optional[str]
    success: bool
    total_frames: int

    # Map data for potential replay
    map_data: Optional[bytes] = None
    input_sequence: Optional[List[int]] = None


class DemonstrationBuffer:
    """Buffer storing expert demonstrations for training augmentation.

    Workflow:
    1. Load compact replay files (.npp binary format)
    2. Simulate replays using ReplayExecutor to extract observations
    3. Filter observations by architecture modalities (if config provided)
    4. Store (observation, action) pairs for sampling
    5. Provide batch sampling interface for training

    Integration with PPO:
    - Sampled via DemoAugmentationCallback which applies BC auxiliary loss
    - Helps bootstrap learning of physics primitives (jumping, wall-jumping)
    """

    def __init__(
        self,
        replay_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        max_demos: int = 100,
        max_transitions_per_demo: int = 2000,
        observation_keys: Optional[List[str]] = None,
        architecture_config: Optional[Any] = None,
        simulate_replays: bool = True,
        num_workers: Optional[int] = None,
        frame_stack_config: Optional[Dict[str, Any]] = None,
        level_filter: Optional[str] = None,
        frame_skip: int = 1,
    ):
        """Initialize demonstration buffer.

        Args:
            replay_paths: Path(s) to replay files or directories
            max_demos: Maximum number of demonstrations to load
            max_transitions_per_demo: Maximum transitions per demonstration
            observation_keys: Keys to extract from observations. If None,
                determined from architecture_config or uses DEFAULT_OBSERVATION_KEYS.
            architecture_config: Optional ArchitectureConfig for modality filtering.
                If provided, observation_keys are determined by enabled modalities.
            simulate_replays: If True, simulate replays to extract full observations.
                             If False, only store actions (lighter but incomplete).
            num_workers: Number of parallel workers for loading replays.
                        If None, defaults to max(1, cpu_count - 1).
                        Set to 1 for sequential loading.
            frame_stack_config: Frame stacking configuration dict with:
                - enable_state_stacking: bool
                - state_stack_size: int (default 4)
                When enabled, sample_batch returns properly stacked observations
                using consecutive frames from episodes.
            level_filter: Optional level name to filter demonstrations by.
                If provided, only loads demos matching this level name (exact match).
            frame_skip: Number of physics ticks per observation (must match training).
                When > 1, ReplayExecutor samples observations at this frequency.
        """
        self.max_demos = max_demos
        self.max_transitions_per_demo = max_transitions_per_demo
        self.architecture_config = architecture_config
        self.simulate_replays = simulate_replays
        self.frame_stack_config = frame_stack_config or {}
        self.level_filter = level_filter
        self.frame_skip = frame_skip

        # Determine number of workers for parallel loading
        # Cap at MAX_DEFAULT_WORKERS to prevent memory exhaustion
        if num_workers is None:
            cpu_count = os.cpu_count() or 1
            self.num_workers = min(MAX_DEFAULT_WORKERS, max(1, cpu_count - 1))
        else:
            self.num_workers = max(1, num_workers)

        # Determine observation keys from architecture config or defaults
        self.observation_keys = self._resolve_observation_keys(
            observation_keys, architecture_config
        )

        # Storage
        self.episodes: List[DemonstrationEpisode] = []
        self.transitions: List[DemonstrationTransition] = []

        # Statistics
        self.total_episodes_loaded = 0
        self.total_transitions = 0
        self.successful_episodes = 0

        # Initialize replay executor for simulating demonstrations (sequential mode only)
        # In parallel mode, each worker creates its own executor
        self._replay_executor: Optional[Any] = None
        if self.num_workers == 1 and simulate_replays and REPLAY_EXECUTOR_AVAILABLE:
            try:
                self._replay_executor = ReplayExecutor(
                    observation_config={},
                    render_mode="grayscale_array",
                    enable_rendering=False,
                    frame_skip=self.frame_skip,
                )
                logger.info(
                    f"ReplayExecutor initialized for demonstration loading "
                    f"(frame_skip={self.frame_skip})"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ReplayExecutor: {e}")
                self._replay_executor = None
        elif simulate_replays and not REPLAY_EXECUTOR_AVAILABLE:
            logger.warning(
                "ReplayExecutor not available. Demonstrations will have empty observations. "
                "Install nclone to enable full demonstration loading."
            )

        # Load replays if paths provided
        if replay_paths is not None:
            self._load_replays(replay_paths)

    def _resolve_observation_keys(
        self,
        explicit_keys: Optional[List[str]],
        architecture_config: Optional[Any],
    ) -> List[str]:
        """Resolve observation keys from config or explicit list.

        Priority:
        1. Explicit observation_keys if provided
        2. Architecture config modalities if provided
        3. DEFAULT_OBSERVATION_KEYS

        Args:
            explicit_keys: Explicitly provided observation keys
            architecture_config: Optional architecture config with modalities

        Returns:
            List of observation keys to extract
        """
        # Use explicit keys if provided
        if explicit_keys is not None:
            return explicit_keys

        # Derive from architecture config modalities
        if architecture_config is not None and hasattr(
            architecture_config, "modalities"
        ):
            modalities = architecture_config.modalities
            keys = []

            # Core state observations (always included if available)
            if getattr(modalities, "use_game_state", True):
                keys.append("game_state")
            if getattr(modalities, "use_reachability", True):
                keys.append("reachability_features")

            # Switch states (useful for levels with locked doors)
            keys.append("switch_states")

            # Graph observations
            if getattr(modalities, "use_graph", False):
                keys.extend(
                    [
                        "graph_node_feats",
                        "graph_edge_index",
                        "graph_node_mask",
                        "graph_edge_mask",
                    ]
                )

            # Spatial context (graph-free local geometry features)
            if getattr(modalities, "use_spatial_context", False):
                keys.append("spatial_context")
                keys.append("mine_sdf_features")  # Always include with spatial_context

            # Action mask (always useful)
            keys.append("action_mask")

            logger.info(f"DemonstrationBuffer using architecture-filtered keys: {keys}")
            return keys

        # Fall back to defaults
        return DEFAULT_OBSERVATION_KEYS.copy()

    def _matches_level_filter(self, replay_file: Path) -> bool:
        """Check if replay file matches the level filter.

        Parses filename and does exact matching:
        - YYYYMMDD_HHMMSS_level_name.replay → extracts "level_name"
        - level_name.npp → extracts "level_name"

        Args:
            replay_file: Path to replay file

        Returns:
            True if matches filter or no filter set, False otherwise
        """
        if self.level_filter is None:
            return True

        # Normalize filter (unify _ and - separators)
        filter_normalized = self.level_filter.lower().replace("_", "-")

        # Parse filename: YYYYMMDD_HHMMSS_level_name.replay or level_name.npp
        file_stem = replay_file.stem
        parts = file_stem.split("_", 2)  # Split on first 2 underscores

        if len(parts) >= 3:
            # Timestamp format: extract level name after timestamp
            file_level_name = parts[2].lower().replace("_", "-")
        else:
            # No timestamp: entire filename is level name
            file_level_name = file_stem.lower().replace("_", "-")

        matches = file_level_name == filter_normalized

        if matches:
            logger.debug(f"Demo {replay_file.name} matches filter '{self.level_filter}'")

        return matches

    def _load_replays(self, paths: Union[str, Path, List[Union[str, Path]]]) -> int:
        """Load replay files from paths.

        Uses parallel processing when num_workers > 1.

        Args:
            paths: Path(s) to replay files or directories

        Returns:
            Number of replays loaded
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]

        # First, collect all replay file paths (with level filtering)
        replay_files: List[Path] = []

        for path in paths:
            path = Path(path)

            if path.is_file():
                if self._matches_level_filter(path):
                    replay_files.append(path)
            elif path.is_dir():
                # Collect all replay files in directory
                for ext in ["*.npp", "*.bin", "*.replay"]:
                    for file in path.glob(ext):
                        if self._matches_level_filter(file):
                            replay_files.append(file)

        # Limit to max_demos
        replay_files = replay_files[: self.max_demos]

        if not replay_files:
            if self.level_filter:
                logger.warning(
                    f"No replay files found matching level filter '{self.level_filter}'"
                )
            else:
                logger.warning("No replay files found to load")
            return 0

        if self.level_filter:
            logger.info(
                f"Found {len(replay_files)} replay files matching level '{self.level_filter}', "
                f"loading with {self.num_workers} worker(s)"
            )
        else:
            logger.info(
                f"Found {len(replay_files)} replay files (no filter), "
                f"loading with {self.num_workers} worker(s)"
            )

        # Use parallel loading if num_workers > 1
        if self.num_workers > 1:
            loaded_count = self._load_replays_parallel(replay_files)
        else:
            loaded_count = self._load_replays_sequential(replay_files)

        logger.info(
            f"Loaded {loaded_count} demonstrations with "
            f"{self.total_transitions} total transitions"
        )

        return loaded_count

    def _load_replays_sequential(self, replay_files: List[Path]) -> int:
        """Load replays sequentially (single worker).

        Args:
            replay_files: List of replay file paths

        Returns:
            Number of replays loaded
        """
        loaded_count = 0

        for replay_file in replay_files:
            if self._load_single_replay(replay_file):
                loaded_count += 1

        return loaded_count

    def _load_replays_parallel(self, replay_files: List[Path]) -> int:
        """Load replays in parallel using ProcessPoolExecutor with batch processing.

        Each worker processes a small batch of replays with a single ReplayExecutor
        instance, reducing memory overhead. Batch size is limited to prevent
        workers from accumulating too much data in memory.

        Args:
            replay_files: List of replay file paths

        Returns:
            Number of replays loaded
        """
        loaded_count = 0
        total_files = len(replay_files)

        # Use small fixed batch size to limit memory per worker
        # Each batch creates one ReplayExecutor but limits accumulated data
        batch_size = min(MAX_REPLAYS_PER_BATCH, max(1, total_files // self.num_workers))
        batches = [
            [str(f) for f in replay_files[i : i + batch_size]]
            for i in range(0, total_files, batch_size)
        ]

        logger.info(
            f"Splitting {total_files} replays into {len(batches)} batches "
            f"({batch_size} replays per batch, {self.num_workers} workers)"
        )

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit batch tasks
                future_to_batch_idx = {
                    executor.submit(
                        _load_replay_batch_worker,
                        batch,
                        self.observation_keys,
                        self.max_transitions_per_demo,
                        self.simulate_replays,
                        self.frame_skip,
                    ): idx
                    for idx, batch in enumerate(batches)
                }

                # Process batch results as they complete
                for future in as_completed(future_to_batch_idx):
                    batch_idx = future_to_batch_idx[future]

                    try:
                        batch_results = future.result()

                        for result in batch_results:
                            if result is not None:
                                # Convert result dict to episode and transitions
                                self._integrate_worker_result(result)
                                loaded_count += 1

                        logger.info(
                            f"Batch {batch_idx + 1}/{len(batches)} complete: "
                            f"{len(batch_results)} replays, "
                            f"total {loaded_count}/{total_files}"
                        )

                    except Exception as e:
                        logger.warning(f"Worker batch {batch_idx} failed: {e}")

        except Exception as e:
            logger.error(
                f"Parallel loading failed: {e}. Falling back to sequential loading."
            )
            # Fall back to sequential loading
            return self._load_replays_sequential(replay_files)

        return loaded_count

    def _integrate_worker_result(self, result: Dict[str, Any]) -> None:
        """Integrate worker result into the buffer.

        Converts serializable dict from worker to proper dataclass instances.

        Args:
            result: Dict with episode data from worker
        """
        # Create episode
        episode = DemonstrationEpisode(
            transitions=[],
            level_id=result.get("level_id"),
            success=result.get("success", True),
            total_frames=result.get("total_frames", 0),
            map_data=result.get("map_data"),
            input_sequence=result.get("input_sequence"),
        )

        # Create transitions from result
        episode_idx = len(self.episodes)
        has_observations = False

        for trans_data in result.get("transitions", []):
            transition = DemonstrationTransition(
                observation=trans_data.get("observation", {}),
                action=trans_data.get("action", 0),
                episode_idx=episode_idx,
                frame_idx=trans_data.get("frame_idx", 0),
                level_id=result.get("level_id"),
                done=trans_data.get("done", False),
            )
            episode.transitions.append(transition)
            self.transitions.append(transition)

            if transition.observation:
                has_observations = True

        # Log appropriately based on whether observations were loaded
        replay_name = result.get("replay_name", "unknown")
        if has_observations:
            logger.debug(
                f"Loaded replay {replay_name} with simulation: "
                f"{len(episode.transitions)} transitions with observations"
            )
        else:
            logger.debug(
                f"Loaded replay {replay_name} without observations: "
                f"{len(episode.transitions)} transitions"
            )

        # Update statistics
        self.episodes.append(episode)
        self.total_episodes_loaded += 1
        self.total_transitions += len(episode.transitions)
        self.successful_episodes += 1

    def _load_single_replay(self, replay_file: Path) -> bool:
        """Load a single replay file.

        If ReplayExecutor is available and simulate_replays=True, the replay
        is simulated to extract full observations. Otherwise, only actions
        are stored with empty observations.

        Args:
            replay_file: Path to replay file

        Returns:
            True if successfully loaded
        """
        try:
            from nclone.replay.gameplay_recorder import CompactReplay

            # Load binary replay
            with open(replay_file, "rb") as f:
                data = f.read()

            replay = CompactReplay.from_binary(data, episode_id=replay_file.stem)

            # Only load successful demonstrations
            if not replay.success:
                logger.debug(f"Skipping unsuccessful replay: {replay_file}")
                return False

            # Create episode from replay
            episode = DemonstrationEpisode(
                transitions=[],
                level_id=replay.level_id,
                success=replay.success,
                total_frames=len(replay.input_sequence),
                map_data=replay.map_data,
                input_sequence=replay.input_sequence,
            )

            # If we have a replay executor, simulate to get observations
            if self._replay_executor is not None and self.simulate_replays:
                try:
                    # Simulate replay to extract observations
                    obs_results = self._replay_executor.execute_replay(
                        map_data=replay.map_data,
                        input_sequence=replay.input_sequence,
                    )

                    # Create transitions with full observations
                    for frame_idx, obs_result in enumerate(obs_results):
                        if frame_idx >= self.max_transitions_per_demo:
                            break

                        # Extract and filter observation
                        obs = obs_result.get("observation", {})
                        action = obs_result.get("action", 0)
                        filtered_obs = self._filter_observation(obs)

                        transition = DemonstrationTransition(
                            observation=filtered_obs,
                            action=action,
                            episode_idx=len(self.episodes),
                            frame_idx=frame_idx,
                            level_id=replay.level_id,
                            done=(frame_idx == len(obs_results) - 1),
                        )
                        episode.transitions.append(transition)
                        self.transitions.append(transition)

                    logger.debug(
                        f"Loaded replay {replay_file.name} with simulation: "
                        f"{len(episode.transitions)} transitions with observations"
                    )

                except Exception as sim_error:
                    logger.warning(
                        f"Failed to simulate replay {replay_file.name}: {sim_error}. "
                        f"Loading without observations."
                    )
                    # Fall through to load without observations
                    episode.transitions.clear()

            # If no observations were loaded (no executor or simulation failed),
            # store transitions with empty observations
            if not episode.transitions:
                for frame_idx, action in enumerate(replay.input_sequence):
                    if frame_idx >= self.max_transitions_per_demo:
                        break

                    transition = DemonstrationTransition(
                        observation={},  # Empty - sampling will skip these
                        action=action,
                        episode_idx=len(self.episodes),
                        frame_idx=frame_idx,
                        level_id=replay.level_id,
                        done=(frame_idx == len(replay.input_sequence) - 1),
                    )
                    episode.transitions.append(transition)
                    self.transitions.append(transition)

                logger.error(
                    f"Loaded replay {replay_file.name} without observations: "
                    f"{len(episode.transitions)} transitions"
                )

            self.episodes.append(episode)
            self.total_episodes_loaded += 1
            self.total_transitions += len(episode.transitions)
            self.successful_episodes += 1

            return True

        except Exception as e:
            logger.warning(f"Failed to load replay {replay_file}: {e}")
            return False

    def add_episode(
        self,
        observations: List[Dict[str, np.ndarray]],
        actions: List[int],
        success: bool = True,
        level_id: Optional[str] = None,
    ) -> None:
        """Add a demonstration episode from observations and actions.

        Args:
            observations: List of observation dicts
            actions: List of actions
            success: Whether episode was successful
            level_id: Optional level identifier
        """
        if len(observations) != len(actions):
            raise ValueError(
                f"Observations ({len(observations)}) and actions ({len(actions)}) "
                "must have same length"
            )

        episode = DemonstrationEpisode(
            transitions=[],
            level_id=level_id,
            success=success,
            total_frames=len(actions),
        )

        for frame_idx, (obs, action) in enumerate(zip(observations, actions)):
            transition = DemonstrationTransition(
                observation=self._filter_observation(obs),
                action=action,
                episode_idx=len(self.episodes),
                frame_idx=frame_idx,
                level_id=level_id,
                done=(frame_idx == len(actions) - 1),
            )
            episode.transitions.append(transition)
            self.transitions.append(transition)

        self.episodes.append(episode)
        self.total_episodes_loaded += 1
        self.total_transitions += len(episode.transitions)

        if success:
            self.successful_episodes += 1

    def _filter_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Filter observation to include only specified keys.

        Handles backward compatibility for old spatial_context format (96 dims → 112 dims).

        Args:
            obs: Full observation dict

        Returns:
            Filtered observation dict
        """
        filtered = {}

        for key in self.observation_keys:
            if key in obs:
                value = obs[key]
                if isinstance(value, np.ndarray):
                    filtered[key] = value.copy()
                else:
                    filtered[key] = np.array(value)

                # BACKWARD COMPATIBILITY: Upgrade old 96-dim spatial_context to 112-dim
                if key == "spatial_context" and filtered[key].shape[-1] == 96:
                    filtered[key] = self._upgrade_spatial_context_with_velocity(
                        filtered[key], obs
                    )
                    logger.debug(
                        "Upgraded spatial_context from 96 to 112 dims (computed velocity features)"
                    )

        return filtered

    def _upgrade_spatial_context_with_velocity(
        self, old_spatial: np.ndarray, obs: Dict[str, Any]
    ) -> np.ndarray:
        """Upgrade old 96-dim spatial_context to 112-dim with computed velocity features.

        Properly calculates velocity_dot_direction and distance_rate for each mine
        using ninja velocity from game_state.

        Args:
            old_spatial: Old 96-dim spatial_context
            obs: Full observation dict (contains game_state for velocity)

        Returns:
            Upgraded 112-dim spatial_context with computed velocity features
        """
        # Extract components
        tile_grid = old_spatial[:64]  # Unchanged
        old_mine_overlay = old_spatial[64:]  # 32 dims (8 mines × 4 features)

        # Extract ninja velocity from game_state
        ninja_vx, ninja_vy = 0.0, 0.0
        if "game_state" in obs:
            game_state = obs["game_state"]
            # game_state format: [vel_mag, vel_dir_x, vel_dir_y, ...]
            if len(game_state) >= 3:
                vel_mag = float(game_state[0])  # Normalized magnitude [0, 1]
                vel_dir_x = float(game_state[1])  # Unit vector [-1, 1]
                vel_dir_y = float(game_state[2])  # Unit vector [-1, 1]

                # Reconstruct actual velocity in px/frame
                # Velocity is stored normalized, so multiply by MAX_HOR_SPEED
                from nclone.constants.physics_constants import MAX_HOR_SPEED

                ninja_vx = vel_mag * vel_dir_x * MAX_HOR_SPEED
                ninja_vy = vel_mag * vel_dir_y * MAX_HOR_SPEED

        # Expand mine overlay from 4 to 6 features with computed velocity terms
        new_mine_overlay = np.zeros(48, dtype=np.float32)

        for i in range(8):  # 8 nearest mines
            old_base = i * 4
            new_base = i * 6

            # Copy old 4 features: relative_x, relative_y, state, radius
            relative_x = old_mine_overlay[old_base + 0]  # [-1, 1]
            relative_y = old_mine_overlay[old_base + 1]  # [-1, 1]
            state = old_mine_overlay[old_base + 2]
            radius = old_mine_overlay[old_base + 3]

            new_mine_overlay[new_base + 0] = relative_x
            new_mine_overlay[new_base + 1] = relative_y
            new_mine_overlay[new_base + 2] = state
            new_mine_overlay[new_base + 3] = radius

            # Calculate velocity features
            # relative_x and relative_y are already normalized to [-1, 1]
            # Denormalize to pixels for distance calculation
            from nclone.constants.physics_constants import (
                LEVEL_WIDTH_PX,
                LEVEL_HEIGHT_PX,
            )

            dx_px = relative_x * LEVEL_WIDTH_PX
            dy_px = relative_y * LEVEL_HEIGHT_PX
            distance = np.sqrt(dx_px**2 + dy_px**2)

            if distance > 1e-6:
                # Normalized direction from ninja to mine
                dir_x = dx_px / distance
                dir_y = dy_px / distance

                # Velocity dot direction: ninja_velocity · direction_to_mine
                # Positive = approaching, negative = moving away
                # Normalized by MAX_HOR_SPEED for consistent scale [-1, 1]
                from nclone.constants.physics_constants import MAX_HOR_SPEED

                velocity_dot = (ninja_vx * dir_x + ninja_vy * dir_y) / MAX_HOR_SPEED

                # Distance rate: velocity component toward mine
                # Negative = getting closer (collision risk)
                distance_rate = -velocity_dot

                new_mine_overlay[new_base + 4] = np.clip(velocity_dot, -1.0, 1.0)
                new_mine_overlay[new_base + 5] = np.clip(distance_rate, -1.0, 1.0)
            else:
                # At mine position - velocity doesn't matter
                new_mine_overlay[new_base + 4] = 0.0
                new_mine_overlay[new_base + 5] = 0.0

        # Combine into new 112-dim format
        return np.concatenate([tile_grid, new_mine_overlay])

    def sample_transitions(self, batch_size: int) -> List[DemonstrationTransition]:
        """Sample random transitions from buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of sampled transitions
        """
        if not self.transitions:
            return []

        indices = np.random.choice(
            len(self.transitions),
            size=min(batch_size, len(self.transitions)),
            replace=False,
        )

        return [self.transitions[i] for i in indices]

    def sample_batch(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Sample a batch of (observations, actions) for training.

        Only includes transitions that have complete observations.
        If demonstrations were loaded without simulation, this may return
        fewer samples than requested.

        When frame_stack_config has enable_state_stacking=True, this method
        returns properly stacked game_state observations using consecutive
        frames from each episode.

        Args:
            batch_size: Number of samples

        Returns:
            (observations_dict, actions_array) tuple
        """
        transitions = self.sample_transitions(batch_size)

        if not transitions:
            return {}, np.array([])

        # Filter to transitions with complete observations
        valid_transitions = [
            t
            for t in transitions
            if t.observation
            and all(key in t.observation for key in self.observation_keys)
        ]

        if not valid_transitions:
            logger.warning(
                "No valid transitions with observations found. "
                "Demonstrations may have been loaded without simulation. "
                "Enable simulate_replays=True and ensure ReplayExecutor is available."
            )
            return {}, np.array([])

        # Check if state stacking is enabled
        enable_state_stacking = self.frame_stack_config.get(
            "enable_state_stacking", False
        )
        state_stack_size = self.frame_stack_config.get("state_stack_size", 4)

        # Collect observations and actions
        observations = {key: [] for key in self.observation_keys}
        actions = []

        for trans in valid_transitions:
            for key in self.observation_keys:
                if key == "game_state" and enable_state_stacking:
                    # Get stacked game_state from consecutive frames
                    stacked_state = self._get_stacked_game_state(
                        trans, state_stack_size
                    )
                    observations[key].append(stacked_state)
                else:
                    observations[key].append(trans.observation[key])
            actions.append(trans.action)

        # Stack into arrays
        obs_arrays = {}
        for key, values in observations.items():
            if values:
                obs_arrays[key] = np.stack(values)

        return obs_arrays, np.array(actions)

    def _get_stacked_game_state(
        self, transition: DemonstrationTransition, stack_size: int
    ) -> np.ndarray:
        """Get stacked game_state from consecutive frames in the episode.

        Retrieves the current frame and (stack_size - 1) previous frames from
        the same episode. For early frames (frame_idx < stack_size - 1), the
        first frame is repeated to pad the stack.

        Args:
            transition: Current transition to stack from
            stack_size: Number of frames to stack

        Returns:
            Stacked game_state array [stack_size, state_dim] matching rollout buffer format
        """
        episode_idx = transition.episode_idx
        frame_idx = transition.frame_idx

        if episode_idx >= len(self.episodes):
            # Fallback: repeat current frame
            current_state = transition.observation.get("game_state")
            if current_state is None:
                return np.zeros((stack_size, 41), dtype=np.float32)
            # Stack the same frame stack_size times: (state_dim,) -> (stack_size, state_dim)
            return np.tile(current_state[np.newaxis, :], (stack_size, 1))

        episode = self.episodes[episode_idx]
        episode_transitions = episode.transitions

        # Collect frames: [t - (stack_size-1), ..., t-1, t]
        stacked_frames = []
        for offset in range(stack_size - 1, -1, -1):
            target_frame_idx = frame_idx - offset

            if target_frame_idx < 0:
                # Pad with first frame of episode
                target_frame_idx = 0

            # Find the transition with this frame_idx
            frame_state = None
            if target_frame_idx < len(episode_transitions):
                frame_trans = episode_transitions[target_frame_idx]
                if frame_trans.observation and "game_state" in frame_trans.observation:
                    frame_state = frame_trans.observation["game_state"]

            if frame_state is None:
                # Fallback to current frame's state
                frame_state = transition.observation.get(
                    "game_state", np.zeros(41, dtype=np.float32)
                )

            stacked_frames.append(frame_state)

        # Stack: [frame_t-3, frame_t-2, frame_t-1, frame_t] -> (stack_size, state_dim)
        return np.stack(stacked_frames, axis=0)

    def get_episode(self, idx: int) -> Optional[DemonstrationEpisode]:
        """Get episode by index.

        Args:
            idx: Episode index

        Returns:
            Episode or None if index out of range
        """
        if 0 <= idx < len(self.episodes):
            return self.episodes[idx]
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_episodes": len(self.episodes),
            "total_transitions": self.total_transitions,
            "successful_episodes": self.successful_episodes,
            "avg_episode_length": (self.total_transitions / max(1, len(self.episodes))),
            "observation_keys": self.observation_keys,
        }

    def __len__(self) -> int:
        """Return number of transitions in buffer."""
        return len(self.transitions)

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.transitions) == 0


class DemonstrationAugmentedTrainer:
    """Helper class for mixing demonstrations with on-policy training.

    Usage with PPO:
        augmenter = DemonstrationAugmentedTrainer(demo_buffer, demo_ratio=0.2)

        # During training:
        combined_obs, combined_actions = augmenter.mix_with_rollouts(
            rollout_obs, rollout_actions, batch_size
        )
    """

    def __init__(
        self,
        demo_buffer: DemonstrationBuffer,
        demo_ratio: float = 0.2,
        decay_steps: int = 500_000,
        min_ratio: float = 0.05,
    ):
        """Initialize demonstration augmenter.

        Args:
            demo_buffer: Buffer containing demonstrations
            demo_ratio: Initial fraction of batch that should be demos (0.0-1.0)
            decay_steps: Steps over which to decay demo ratio
            min_ratio: Minimum demo ratio after decay
        """
        self.demo_buffer = demo_buffer
        self.initial_ratio = demo_ratio
        self.current_ratio = demo_ratio
        self.decay_steps = decay_steps
        self.min_ratio = min_ratio
        self.total_steps = 0

    def update_ratio(self, timesteps: int) -> None:
        """Update demo ratio based on training progress.

        Args:
            timesteps: Current training timesteps
        """
        self.total_steps = timesteps

        if timesteps >= self.decay_steps:
            self.current_ratio = self.min_ratio
        else:
            # Linear decay
            progress = timesteps / self.decay_steps
            self.current_ratio = (
                self.initial_ratio * (1.0 - progress) + self.min_ratio * progress
            )

    def get_demo_batch_size(self, total_batch_size: int) -> int:
        """Calculate number of demo samples for a batch.

        Args:
            total_batch_size: Total batch size

        Returns:
            Number of demo samples to include
        """
        if self.demo_buffer.is_empty():
            return 0

        return int(total_batch_size * self.current_ratio)

    def mix_with_rollouts(
        self,
        rollout_obs: Dict[str, np.ndarray],
        rollout_actions: np.ndarray,
        total_batch_size: int,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Mix demonstration samples with rollout data.

        Args:
            rollout_obs: Observations from rollout buffer
            rollout_actions: Actions from rollout buffer
            total_batch_size: Total desired batch size

        Returns:
            (combined_obs, combined_actions) tuple
        """
        demo_size = self.get_demo_batch_size(total_batch_size)
        rollout_size = total_batch_size - demo_size

        if demo_size == 0 or self.demo_buffer.is_empty():
            return rollout_obs, rollout_actions

        # Sample from demonstrations
        demo_obs, demo_actions = self.demo_buffer.sample_batch(demo_size)

        if len(demo_actions) == 0:
            return rollout_obs, rollout_actions

        # Sample from rollouts
        rollout_indices = np.random.choice(
            len(rollout_actions),
            size=min(rollout_size, len(rollout_actions)),
            replace=False,
        )

        # Combine observations
        combined_obs = {}
        for key in rollout_obs:
            rollout_samples = rollout_obs[key][rollout_indices]
            if key in demo_obs:
                combined_obs[key] = np.concatenate(
                    [
                        demo_obs[key],
                        rollout_samples,
                    ]
                )
            else:
                combined_obs[key] = rollout_samples

        # Combine actions
        combined_actions = np.concatenate(
            [
                demo_actions,
                rollout_actions[rollout_indices],
            ]
        )

        return combined_obs, combined_actions

    def get_statistics(self) -> Dict[str, Any]:
        """Get augmentation statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "current_ratio": self.current_ratio,
            "total_steps": self.total_steps,
            "demo_buffer_size": len(self.demo_buffer),
            **self.demo_buffer.get_statistics(),
        }

