"""Demonstration buffer for mixing expert data with on-policy training.

Loads compact binary replay files (.npp format), simulates them through
``ReplayExecutor`` to regenerate Layer 1 observations, and exposes
``(observation, action)`` sampling for behavioral cloning / demonstration
augmentation.

Layer 1 observation contract (fixed set):
    - tile_grid
    - entity_overlays
    - hazard_map
    - game_state
    - action_mask

Usage:
    demo_buffer = DemonstrationBuffer(
        replay_paths,
        max_demos=100,
        num_workers=2,
    )
    demo_obs, demo_actions = demo_buffer.sample_batch(batch_size=64)
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from nclone.replay.replay_executor import ReplayExecutor

    REPLAY_EXECUTOR_AVAILABLE = True
except ImportError:
    REPLAY_EXECUTOR_AVAILABLE = False

logger = logging.getLogger(__name__)

# Layer 1 policy observation contract — fixed set of keys.
DEFAULT_OBSERVATION_KEYS: List[str] = [
    "tile_grid",
    "entity_overlays",
    "hazard_map",
    "game_state",
    "action_mask",
]

# Maximum workers to prevent memory exhaustion (each worker creates a ReplayExecutor).
MAX_DEFAULT_WORKERS = 2

# Maximum replays per worker batch to cap peak memory.
MAX_REPLAYS_PER_BATCH = 5


def _load_replay_batch_worker(
    replay_paths: List[str],
    observation_keys: List[str],
    max_transitions_per_demo: int,
    simulate_replays: bool,
    frame_skip: int = 1,
    obs_exposure: Optional[Dict[str, bool]] = None,
) -> List[Dict[str, Any]]:
    """Worker for parallel batch loading.

    A single ``ReplayExecutor`` is shared across the batch to amortize env
    setup cost. ``obs_exposure`` forwards ReachabilityConfig obs-exposure flags so
    the replayed BC obs match the policy obs the training run was built with.
    """
    from pathlib import Path

    from nclone.replay.gameplay_recorder import CompactReplay
    from nclone.replay.replay_executor import ReplayExecutor

    results: List[Dict[str, Any]] = []
    executor = None
    if simulate_replays:
        try:
            executor = ReplayExecutor(frame_skip=frame_skip, obs_exposure=obs_exposure)
        except Exception:
            executor = None

    for replay_path in replay_paths:
        replay_file = Path(replay_path)
        try:
            with open(replay_file, "rb") as f:
                data = f.read()

            replay = CompactReplay.from_binary(data, episode_id=replay_file.stem)
            if not replay.success:
                continue

            result: Dict[str, Any] = {
                "level_id": replay.level_id,
                "success": replay.success,
                "total_frames": len(replay.input_sequence),
                "map_data": replay.map_data,
                "input_sequence": replay.input_sequence,
                "transitions": [],
                "replay_name": replay_file.name,
            }

            if executor is not None:
                try:
                    obs_results = executor.execute_replay(
                        map_data=replay.map_data,
                        input_sequence=replay.input_sequence,
                    )

                    for frame_idx, obs_result in enumerate(obs_results):
                        if frame_idx >= max_transitions_per_demo:
                            break

                        obs = obs_result.get("observation", {})
                        action = obs_result.get("action", 0)

                        filtered_obs: Dict[str, np.ndarray] = {}
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
                    result["transitions"] = []

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
            continue

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

    map_data: Optional[bytes] = None
    input_sequence: Optional[List[int]] = None


class DemonstrationBuffer:
    """Buffer storing expert demonstrations for training augmentation.

    Workflow:
        1. Load compact replay files (``.npp`` binary format).
        2. Simulate replays with ``ReplayExecutor`` to extract Layer 1 obs.
        3. Store ``(observation, action)`` pairs for sampling.

    Sampled batches integrate with PPO via ``DemoAugmentationCallback`` which
    applies a BC auxiliary loss.
    """

    def __init__(
        self,
        replay_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        max_demos: int = 100,
        max_transitions_per_demo: int = 2000,
        observation_keys: Optional[List[str]] = None,
        simulate_replays: bool = True,
        num_workers: Optional[int] = None,
        level_filter: Optional[str] = None,
        frame_skip: int = 1,
        obs_exposure: Optional[Dict[str, bool]] = None,
    ):
        self.max_demos = max_demos
        self.max_transitions_per_demo = max_transitions_per_demo
        self.simulate_replays = simulate_replays
        self.level_filter = level_filter
        self.frame_skip = frame_skip
        # ReachabilityConfig obs-exposure flags (use_jump_potential /
        # expose_cost_to_go_map / ...) so the replayed BC obs match the policy obs
        # the training run was built with. None = strict Layer-1 contract.
        self.obs_exposure = obs_exposure

        if num_workers is None:
            cpu_count = os.cpu_count() or 1
            self.num_workers = min(MAX_DEFAULT_WORKERS, max(1, cpu_count - 1))
        else:
            self.num_workers = max(1, num_workers)

        self.observation_keys = (
            list(observation_keys)
            if observation_keys is not None
            else DEFAULT_OBSERVATION_KEYS.copy()
        )

        self.episodes: List[DemonstrationEpisode] = []
        self.transitions: List[DemonstrationTransition] = []

        self.total_episodes_loaded = 0
        self.total_transitions = 0
        self.successful_episodes = 0

        self._replay_executor: Optional[Any] = None
        if self.num_workers == 1 and simulate_replays and REPLAY_EXECUTOR_AVAILABLE:
            try:
                self._replay_executor = ReplayExecutor(
                    frame_skip=self.frame_skip, obs_exposure=self.obs_exposure
                )
                logger.info(
                    "ReplayExecutor initialized for demonstration loading "
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

        if replay_paths is not None:
            self._load_replays(replay_paths)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _matches_level_filter(self, replay_file: Path) -> bool:
        """Exact match against ``level_filter`` (separators normalized)."""
        if self.level_filter is None:
            return True

        filter_normalized = self.level_filter.lower().replace("_", "-")
        file_stem = replay_file.stem
        parts = file_stem.split("_", 2)
        if len(parts) >= 3:
            file_level_name = parts[2].lower().replace("_", "-")
        else:
            file_level_name = file_stem.lower().replace("_", "-")
        return file_level_name == filter_normalized

    def _load_replays(
        self,
        paths: Union[str, Path, List[Union[str, Path]]],
    ) -> int:
        if isinstance(paths, (str, Path)):
            paths = [paths]

        replay_files: List[Path] = []
        for path in paths:
            path = Path(path)
            if path.is_file():
                if self._matches_level_filter(path):
                    replay_files.append(path)
            elif path.is_dir():
                for ext in ("*.npp", "*.bin", "*.replay"):
                    for file in path.glob(ext):
                        if self._matches_level_filter(file):
                            replay_files.append(file)

        replay_files = replay_files[: self.max_demos]
        if not replay_files:
            if self.level_filter:
                logger.warning(
                    f"No replay files found matching level filter '{self.level_filter}'"
                )
            else:
                logger.warning("No replay files found to load")
            return 0

        logger.info(
            f"Found {len(replay_files)} replay files, "
            f"loading with {self.num_workers} worker(s)"
        )

        if self.num_workers > 1:
            loaded = self._load_replays_parallel(replay_files)
        else:
            loaded = self._load_replays_sequential(replay_files)

        logger.info(
            f"Loaded {loaded} demonstrations with "
            f"{self.total_transitions} total transitions"
        )
        return loaded

    def _load_replays_sequential(self, replay_files: List[Path]) -> int:
        loaded = 0
        for f in replay_files:
            if self._load_single_replay(f):
                loaded += 1
        return loaded

    def _load_replays_parallel(self, replay_files: List[Path]) -> int:
        loaded = 0
        total_files = len(replay_files)
        batch_size = min(MAX_REPLAYS_PER_BATCH, max(1, total_files // self.num_workers))
        batches = [
            [str(f) for f in replay_files[i : i + batch_size]]
            for i in range(0, total_files, batch_size)
        ]

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
                future_to_batch_idx = {
                    pool.submit(
                        _load_replay_batch_worker,
                        batch,
                        self.observation_keys,
                        self.max_transitions_per_demo,
                        self.simulate_replays,
                        self.frame_skip,
                        self.obs_exposure,
                    ): idx
                    for idx, batch in enumerate(batches)
                }

                for future in as_completed(future_to_batch_idx):
                    batch_idx = future_to_batch_idx[future]
                    try:
                        batch_results = future.result()
                        for result in batch_results:
                            if result is not None:
                                self._integrate_worker_result(result)
                                loaded += 1
                        logger.info(
                            f"Batch {batch_idx + 1}/{len(batches)} complete "
                            f"(total {loaded}/{total_files})"
                        )
                    except Exception as e:
                        logger.warning(f"Worker batch {batch_idx} failed: {e}")

        except Exception as e:
            logger.error(
                f"Parallel loading failed: {e}. Falling back to sequential loading."
            )
            return self._load_replays_sequential(replay_files)

        return loaded

    def _integrate_worker_result(self, result: Dict[str, Any]) -> None:
        episode = DemonstrationEpisode(
            transitions=[],
            level_id=result.get("level_id"),
            success=result.get("success", True),
            total_frames=result.get("total_frames", 0),
            map_data=result.get("map_data"),
            input_sequence=result.get("input_sequence"),
        )

        episode_idx = len(self.episodes)
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

        self.episodes.append(episode)
        self.total_episodes_loaded += 1
        self.total_transitions += len(episode.transitions)
        self.successful_episodes += 1

    def _load_single_replay(self, replay_file: Path) -> bool:
        try:
            from nclone.replay.gameplay_recorder import CompactReplay

            with open(replay_file, "rb") as f:
                data = f.read()

            replay = CompactReplay.from_binary(data, episode_id=replay_file.stem)
            if not replay.success:
                logger.debug(f"Skipping unsuccessful replay: {replay_file}")
                return False

            episode = DemonstrationEpisode(
                transitions=[],
                level_id=replay.level_id,
                success=replay.success,
                total_frames=len(replay.input_sequence),
                map_data=replay.map_data,
                input_sequence=replay.input_sequence,
            )

            if self._replay_executor is not None and self.simulate_replays:
                try:
                    obs_results = self._replay_executor.execute_replay(
                        map_data=replay.map_data,
                        input_sequence=replay.input_sequence,
                    )

                    for frame_idx, obs_result in enumerate(obs_results):
                        if frame_idx >= self.max_transitions_per_demo:
                            break

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

                except Exception as sim_error:
                    logger.warning(
                        f"Failed to simulate replay {replay_file.name}: {sim_error}. "
                        "Loading without observations."
                    )
                    episode.transitions.clear()

            if not episode.transitions:
                for frame_idx, action in enumerate(replay.input_sequence):
                    if frame_idx >= self.max_transitions_per_demo:
                        break
                    transition = DemonstrationTransition(
                        observation={},
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

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def add_episode(
        self,
        observations: List[Dict[str, np.ndarray]],
        actions: List[int],
        success: bool = True,
        level_id: Optional[str] = None,
    ) -> None:
        """Append a demonstration episode built in-memory."""
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
        """Return a copy of ``obs`` restricted to ``observation_keys``."""
        filtered: Dict[str, np.ndarray] = {}
        for key in self.observation_keys:
            if key in obs:
                value = obs[key]
                if isinstance(value, np.ndarray):
                    filtered[key] = value.copy()
                else:
                    filtered[key] = np.array(value)
        return filtered

    def sample_transitions(self, batch_size: int) -> List[DemonstrationTransition]:
        if not self.transitions:
            return []
        indices = np.random.choice(
            len(self.transitions),
            size=min(batch_size, len(self.transitions)),
            replace=False,
        )
        return [self.transitions[i] for i in indices]

    def sample_batch(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Sample a batch of (obs, actions) restricted to Layer 1 keys."""
        transitions = self.sample_transitions(batch_size)
        if not transitions:
            return {}, np.array([])

        valid_transitions = [
            t
            for t in transitions
            if t.observation
            and all(key in t.observation for key in self.observation_keys)
        ]
        if not valid_transitions:
            logger.warning(
                "No valid transitions with observations found. "
                "Demonstrations may have been loaded without simulation."
            )
            return {}, np.array([])

        observations: Dict[str, List[np.ndarray]] = {key: [] for key in self.observation_keys}
        actions: List[int] = []
        for trans in valid_transitions:
            for key in self.observation_keys:
                observations[key].append(trans.observation[key])
            actions.append(trans.action)

        obs_arrays = {
            key: np.stack(values) for key, values in observations.items() if values
        }
        return obs_arrays, np.array(actions)

    def get_episode(self, idx: int) -> Optional[DemonstrationEpisode]:
        if 0 <= idx < len(self.episodes):
            return self.episodes[idx]
        return None

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_episodes": len(self.episodes),
            "total_transitions": self.total_transitions,
            "successful_episodes": self.successful_episodes,
            "avg_episode_length": (
                self.total_transitions / max(1, len(self.episodes))
            ),
            "observation_keys": self.observation_keys,
        }

    def __len__(self) -> int:
        return len(self.transitions)

    def is_empty(self) -> bool:
        return len(self.transitions) == 0


class DemonstrationAugmentedTrainer:
    """Helper for mixing demonstrations with on-policy training.

    Usage with PPO::

        augmenter = DemonstrationAugmentedTrainer(demo_buffer, demo_ratio=0.2)
        combined_obs, combined_actions = augmenter.mix_with_rollouts(
            rollout_obs, rollout_actions, batch_size,
        )
    """

    def __init__(
        self,
        demo_buffer: DemonstrationBuffer,
        demo_ratio: float = 0.2,
        decay_steps: int = 500_000,
        min_ratio: float = 0.05,
    ):
        self.demo_buffer = demo_buffer
        self.initial_ratio = demo_ratio
        self.current_ratio = demo_ratio
        self.decay_steps = decay_steps
        self.min_ratio = min_ratio
        self.total_steps = 0

    def update_ratio(self, timesteps: int) -> None:
        self.total_steps = timesteps
        if timesteps >= self.decay_steps:
            self.current_ratio = self.min_ratio
        else:
            progress = timesteps / self.decay_steps
            self.current_ratio = (
                self.initial_ratio * (1.0 - progress) + self.min_ratio * progress
            )

    def get_demo_batch_size(self, total_batch_size: int) -> int:
        if self.demo_buffer.is_empty():
            return 0
        return int(total_batch_size * self.current_ratio)

    def mix_with_rollouts(
        self,
        rollout_obs: Dict[str, np.ndarray],
        rollout_actions: np.ndarray,
        total_batch_size: int,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        demo_size = self.get_demo_batch_size(total_batch_size)
        rollout_size = total_batch_size - demo_size

        if demo_size == 0 or self.demo_buffer.is_empty():
            return rollout_obs, rollout_actions

        demo_obs, demo_actions = self.demo_buffer.sample_batch(demo_size)
        if len(demo_actions) == 0:
            return rollout_obs, rollout_actions

        rollout_indices = np.random.choice(
            len(rollout_actions),
            size=min(rollout_size, len(rollout_actions)),
            replace=False,
        )

        combined_obs: Dict[str, np.ndarray] = {}
        for key in rollout_obs:
            rollout_samples = rollout_obs[key][rollout_indices]
            if key in demo_obs:
                combined_obs[key] = np.concatenate([demo_obs[key], rollout_samples])
            else:
                combined_obs[key] = rollout_samples

        combined_actions = np.concatenate(
            [demo_actions, rollout_actions[rollout_indices]]
        )
        return combined_obs, combined_actions

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "current_ratio": self.current_ratio,
            "total_steps": self.total_steps,
            "demo_buffer_size": len(self.demo_buffer),
            **self.demo_buffer.get_statistics(),
        }
