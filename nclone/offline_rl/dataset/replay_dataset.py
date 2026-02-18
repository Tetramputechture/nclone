"""HDF5-backed PyTorch Dataset for offline RL training.

Stores transitions in HDF5 files. On load, all arrays are read into RAM
(~3GB for 9M transitions) to avoid gzip-decompression overhead during
random-access batch sampling. Supports single-level and cross-level (UVFA) modes.
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .transition_extractor import (
    AGENT_STATE_DIM,
    MINE_FEATURE_DIM,
    TILE_GRID_H,
    TILE_GRID_W,
    EpisodeTransitions,
)

logger = logging.getLogger(__name__)


class OfflineRLDataset(Dataset):
    """HDF5-backed dataset for offline RL training.

    Per-transition arrays:
        states (N, 11), next_states (N, 11), actions (N,), rewards (N,),
        dones (N,), level_ids (N,), mine_features (N, 24), next_mine_features (N, 24)

    Per-level arrays:
        tile_grids (L, 42, 23), switch_overlays (L, 42, 23), exit_overlays (L, 42, 23)
    """

    def __init__(self, h5_path: str, device: str = "cpu"):
        """Load dataset from HDF5 file into RAM.

        Reads all arrays into contiguous numpy arrays at init time so that
        sample_batch uses fast numpy indexing instead of HDF5 random I/O.

        Args:
            h5_path: Path to the HDF5 dataset file.
            device: Device for returned tensors ('cpu' or 'cuda').
        """
        self.h5_path = h5_path
        self.device = device
        self._h5 = None

        import time
        t0 = time.time()
        logger.info("Loading dataset into RAM from %s ...", h5_path)

        with h5py.File(h5_path, "r") as f:
            self._states = f["states"][:].astype(np.float32)
            self._next_states = f["next_states"][:].astype(np.float32)
            self._actions = f["actions"][:].astype(np.int64)
            self._rewards = f["rewards"][:].astype(np.float32)
            self._dones = f["dones"][:].astype(np.float32)
            self._level_ids = f["level_ids"][:].astype(np.int64)
            self._mine_features = f["mine_features"][:].astype(np.float32)
            self._next_mine_features = f["next_mine_features"][:].astype(np.float32)

            self._tile_grids = f["tile_grids"][:].astype(np.int64)
            self._switch_overlays = f["switch_overlays"][:].astype(np.float32)
            self._exit_overlays = f["exit_overlays"][:].astype(np.float32)

        self._n_transitions = self._states.shape[0]
        self._n_levels = self._tile_grids.shape[0]

        ram_mb = sum(
            a.nbytes for a in [
                self._states, self._next_states, self._actions,
                self._rewards, self._dones, self._level_ids,
                self._mine_features, self._next_mine_features,
                self._tile_grids, self._switch_overlays, self._exit_overlays,
            ]
        ) / 1e6

        logger.info(
            "Loaded dataset: %d transitions, %d levels (%.0f MB RAM) in %.1fs",
            self._n_transitions,
            self._n_levels,
            ram_mb,
            time.time() - t0,
        )

    def __len__(self) -> int:
        return self._n_transitions

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        level_id = int(self._level_ids[idx])
        return {
            "state": torch.from_numpy(self._states[idx].copy()),
            "next_state": torch.from_numpy(self._next_states[idx].copy()),
            "action": torch.tensor(self._actions[idx], dtype=torch.long),
            "reward": torch.tensor(self._rewards[idx], dtype=torch.float32),
            "done": torch.tensor(self._dones[idx], dtype=torch.float32),
            "mine_features": torch.from_numpy(self._mine_features[idx].copy()),
            "next_mine_features": torch.from_numpy(self._next_mine_features[idx].copy()),
            "tile_grid": torch.from_numpy(self._tile_grids[level_id].copy()),
            "switch_overlay": torch.from_numpy(self._switch_overlays[level_id].copy()),
            "exit_overlay": torch.from_numpy(self._exit_overlays[level_id].copy()),
        }

    def sample_batch(
        self, batch_size: int, pin_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions.

        All data is in RAM so this is pure numpy fancy indexing (~1ms for 1024).

        Args:
            batch_size: Number of transitions to sample.
            pin_memory: If True, allocate tensors in pinned (page-locked) memory
                for faster CPU-to-GPU transfers. Only useful when device is 'cpu'
                and data will later be moved to CUDA.

        Returns:
            Dictionary of batched tensors on the configured device.
        """
        idx = np.random.randint(0, self._n_transitions, size=batch_size)

        level_ids = self._level_ids[idx]

        batch = {
            "state": torch.from_numpy(self._states[idx].copy()),
            "next_state": torch.from_numpy(self._next_states[idx].copy()),
            "action": torch.from_numpy(self._actions[idx].copy()),
            "reward": torch.from_numpy(self._rewards[idx].copy()),
            "done": torch.from_numpy(self._dones[idx].copy()),
            "mine_features": torch.from_numpy(self._mine_features[idx].copy()),
            "next_mine_features": torch.from_numpy(self._next_mine_features[idx].copy()),
            "tile_grid": torch.from_numpy(self._tile_grids[level_ids].copy()),
            "switch_overlay": torch.from_numpy(self._switch_overlays[level_ids].copy()),
            "exit_overlay": torch.from_numpy(self._exit_overlays[level_ids].copy()),
        }

        if pin_memory:
            batch = {k: v.pin_memory() for k, v in batch.items()}

        if self.device != "cpu":
            batch = {k: v.to(self.device) for k, v in batch.items()}

        return batch

    @property
    def n_transitions(self) -> int:
        return self._n_transitions

    @property
    def n_levels(self) -> int:
        return self._n_levels

    def close(self):
        """Release memory (HDF5 file is already closed after init)."""
        self._states = None
        self._next_states = None


_SENTINEL = None


class PrefetchingBatchIterator:
    """Background-thread batch prefetcher for GPU training.

    Prepares the next batch on CPU (with optional pin_memory) in a background
    thread while the current batch is being processed on GPU. This overlaps
    numpy indexing + torch conversion + CPU-to-GPU transfer with GPU compute.

    Usage::

        dataset = OfflineRLDataset("data.h5", device="cpu")
        prefetcher = PrefetchingBatchIterator(dataset, batch_size=2048, device="cuda")
        for step in range(num_steps):
            batch = next(prefetcher)  # returns batch already on GPU
    """

    def __init__(
        self,
        dataset: OfflineRLDataset,
        batch_size: int,
        device: str = "cuda",
        prefetch_count: int = 2,
    ):
        self._dataset = dataset
        self._batch_size = batch_size
        self._device = torch.device(device)
        self._use_pin = self._device.type == "cuda"

        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_count)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def _producer(self) -> None:
        """Background thread that continuously produces batches."""
        while not self._stop_event.is_set():
            batch = self._dataset.sample_batch(
                self._batch_size, pin_memory=self._use_pin,
            )
            try:
                self._queue.put(batch, timeout=1.0)
            except queue.Full:
                if self._stop_event.is_set():
                    break

    def __next__(self) -> Dict[str, torch.Tensor]:
        batch = self._queue.get()
        if self._device.type != "cpu":
            batch = {
                k: v.to(self._device, non_blocking=True)
                for k, v in batch.items()
            }
        return batch

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return self

    def stop(self) -> None:
        """Signal the background thread to stop and drain the queue."""
        self._stop_event.set()
        # Drain remaining items so the producer thread can exit
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5.0)


def build_dataset(
    episodes: List[EpisodeTransitions],
    output_path: str,
    compression: str = "gzip",
) -> str:
    """Build an HDF5 dataset from extracted episode transitions.

    Args:
        episodes: List of EpisodeTransitions from TransitionExtractor.
        output_path: Path for the output HDF5 file.
        compression: HDF5 compression algorithm ('gzip', 'lzf', or None).

    Returns:
        Path to the created HDF5 file.
    """
    if not episodes:
        raise ValueError("No episodes to save")

    # Assign level IDs: deduplicate by level_id string
    level_id_map: Dict[str, int] = {}
    level_statics: List = []
    for ep in episodes:
        lid = ep.level_static.level_id
        if lid not in level_id_map:
            level_id_map[lid] = len(level_id_map)
            level_statics.append(ep.level_static)

    # Compute total transitions
    total = sum(len(ep.states) for ep in episodes)
    n_levels = len(level_id_map)

    logger.info("Building dataset: %d transitions, %d levels -> %s", total, n_levels, output_path)

    # Concatenate all transitions
    all_states = np.zeros((total, AGENT_STATE_DIM), dtype=np.float32)
    all_next_states = np.zeros((total, AGENT_STATE_DIM), dtype=np.float32)
    all_actions = np.zeros(total, dtype=np.int64)
    all_rewards = np.zeros(total, dtype=np.float32)
    all_dones = np.zeros(total, dtype=np.bool_)
    all_level_ids = np.zeros(total, dtype=np.int64)
    all_mine_feats = np.zeros((total, MINE_FEATURE_DIM), dtype=np.float32)
    all_next_mine_feats = np.zeros((total, MINE_FEATURE_DIM), dtype=np.float32)

    offset = 0
    for ep in episodes:
        n = len(ep.states)
        lid = level_id_map[ep.level_static.level_id]

        all_states[offset : offset + n] = ep.states
        all_next_states[offset : offset + n] = ep.next_states
        all_actions[offset : offset + n] = ep.actions
        all_rewards[offset : offset + n] = ep.rewards
        all_dones[offset : offset + n] = ep.dones
        all_level_ids[offset : offset + n] = lid
        all_mine_feats[offset : offset + n] = ep.mine_features
        all_next_mine_feats[offset : offset + n] = ep.next_mine_features
        offset += n

    # Build per-level arrays
    all_tile_grids = np.zeros((n_levels, TILE_GRID_W, TILE_GRID_H), dtype=np.int8)
    all_switch_overlays = np.zeros((n_levels, TILE_GRID_W, TILE_GRID_H), dtype=np.float32)
    all_exit_overlays = np.zeros((n_levels, TILE_GRID_W, TILE_GRID_H), dtype=np.float32)

    for ls in level_statics:
        lid = level_id_map[ls.level_id]
        all_tile_grids[lid] = ls.tile_grid
        all_switch_overlays[lid] = ls.switch_overlay
        all_exit_overlays[lid] = ls.exit_overlay

    # Write HDF5
    comp = compression if compression else None
    with h5py.File(output_path, "w") as f:
        f.create_dataset("states", data=all_states, compression=comp)
        f.create_dataset("next_states", data=all_next_states, compression=comp)
        f.create_dataset("actions", data=all_actions, compression=comp)
        f.create_dataset("rewards", data=all_rewards, compression=comp)
        f.create_dataset("dones", data=all_dones, compression=comp)
        f.create_dataset("level_ids", data=all_level_ids, compression=comp)
        f.create_dataset("mine_features", data=all_mine_feats, compression=comp)
        f.create_dataset("next_mine_features", data=all_next_mine_feats, compression=comp)
        f.create_dataset("tile_grids", data=all_tile_grids, compression=comp)
        f.create_dataset("switch_overlays", data=all_switch_overlays, compression=comp)
        f.create_dataset("exit_overlays", data=all_exit_overlays, compression=comp)

        # Store metadata
        f.attrs["n_transitions"] = total
        f.attrs["n_levels"] = n_levels
        f.attrs["agent_state_dim"] = AGENT_STATE_DIM
        f.attrs["mine_feature_dim"] = MINE_FEATURE_DIM
        f.attrs["level_id_map"] = str(level_id_map)

    logger.info("Dataset saved: %s (%.1f MB)", output_path, Path(output_path).stat().st_size / 1e6)
    return output_path


# HDF5 chunk size for resizable datasets (balances write throughput vs. granularity)
_TRANSITION_CHUNK = 8192


class StreamingDatasetBuilder:
    """Incrementally builds an HDF5 dataset without holding all data in memory.

    Uses resizable HDF5 datasets (maxshape=(None, ...)) and appends episodes
    in batches. Suitable for 100K+ files / 50M+ transitions that would OOM
    with the in-memory ``build_dataset()`` approach.

    Usage::

        builder = StreamingDatasetBuilder("dataset.h5")
        for episode in extract_episodes(...):
            builder.add_episode(episode)
        builder.finalize()
    """

    def __init__(self, output_path: str, compression: Optional[str] = "gzip"):
        self._path = output_path
        self._comp = compression if compression else None
        self._h5 = h5py.File(output_path, "w")
        self._offset = 0  # next write position in transition arrays
        self._level_id_map: Dict[str, int] = {}
        self._level_statics: List = []
        self._n_episodes = 0
        self._n_wins = 0

        # Create resizable per-transition datasets (initial size 0)
        kw = dict(compression=self._comp, chunks=True)
        self._ds_states = self._h5.create_dataset(
            "states", shape=(0, AGENT_STATE_DIM), maxshape=(None, AGENT_STATE_DIM),
            dtype=np.float32, **kw,
        )
        self._ds_next_states = self._h5.create_dataset(
            "next_states", shape=(0, AGENT_STATE_DIM), maxshape=(None, AGENT_STATE_DIM),
            dtype=np.float32, **kw,
        )
        self._ds_actions = self._h5.create_dataset(
            "actions", shape=(0,), maxshape=(None,), dtype=np.int64, **kw,
        )
        self._ds_rewards = self._h5.create_dataset(
            "rewards", shape=(0,), maxshape=(None,), dtype=np.float32, **kw,
        )
        self._ds_dones = self._h5.create_dataset(
            "dones", shape=(0,), maxshape=(None,), dtype=np.bool_, **kw,
        )
        self._ds_level_ids = self._h5.create_dataset(
            "level_ids", shape=(0,), maxshape=(None,), dtype=np.int64, **kw,
        )
        self._ds_mine_feats = self._h5.create_dataset(
            "mine_features", shape=(0, MINE_FEATURE_DIM), maxshape=(None, MINE_FEATURE_DIM),
            dtype=np.float32, **kw,
        )
        self._ds_next_mine_feats = self._h5.create_dataset(
            "next_mine_features", shape=(0, MINE_FEATURE_DIM), maxshape=(None, MINE_FEATURE_DIM),
            dtype=np.float32, **kw,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_episode(self, ep: EpisodeTransitions) -> None:
        """Append one episode's transitions to the HDF5 file."""
        n = len(ep.states)
        if n == 0:
            return

        # Register level
        lid_str = ep.level_static.level_id
        if lid_str not in self._level_id_map:
            self._level_id_map[lid_str] = len(self._level_id_map)
            self._level_statics.append(ep.level_static)
        lid = self._level_id_map[lid_str]

        # Resize datasets
        new_size = self._offset + n
        self._ds_states.resize(new_size, axis=0)
        self._ds_next_states.resize(new_size, axis=0)
        self._ds_actions.resize(new_size, axis=0)
        self._ds_rewards.resize(new_size, axis=0)
        self._ds_dones.resize(new_size, axis=0)
        self._ds_level_ids.resize(new_size, axis=0)
        self._ds_mine_feats.resize(new_size, axis=0)
        self._ds_next_mine_feats.resize(new_size, axis=0)

        s = self._offset
        self._ds_states[s:new_size] = ep.states
        self._ds_next_states[s:new_size] = ep.next_states
        self._ds_actions[s:new_size] = ep.actions
        self._ds_rewards[s:new_size] = ep.rewards
        self._ds_dones[s:new_size] = ep.dones
        self._ds_level_ids[s:new_size] = lid
        self._ds_mine_feats[s:new_size] = ep.mine_features
        self._ds_next_mine_feats[s:new_size] = ep.next_mine_features

        self._offset = new_size
        self._n_episodes += 1
        if ep.dones[-1] and ep.rewards[-1] == 0.0:
            self._n_wins += 1

    def add_episodes(self, episodes: List[EpisodeTransitions]) -> None:
        """Append a batch of episodes."""
        for ep in episodes:
            self.add_episode(ep)

    def finalize(self) -> str:
        """Write per-level arrays and metadata, then close the file.

        Returns:
            Path to the completed HDF5 dataset.
        """
        n_levels = len(self._level_id_map)

        # Build and write per-level arrays
        tile_grids = np.zeros((n_levels, TILE_GRID_W, TILE_GRID_H), dtype=np.int8)
        switch_overlays = np.zeros((n_levels, TILE_GRID_W, TILE_GRID_H), dtype=np.float32)
        exit_overlays = np.zeros((n_levels, TILE_GRID_W, TILE_GRID_H), dtype=np.float32)

        for ls in self._level_statics:
            lid = self._level_id_map[ls.level_id]
            tile_grids[lid] = ls.tile_grid
            switch_overlays[lid] = ls.switch_overlay
            exit_overlays[lid] = ls.exit_overlay

        self._h5.create_dataset("tile_grids", data=tile_grids, compression=self._comp)
        self._h5.create_dataset("switch_overlays", data=switch_overlays, compression=self._comp)
        self._h5.create_dataset("exit_overlays", data=exit_overlays, compression=self._comp)

        # Metadata
        self._h5.attrs["n_transitions"] = self._offset
        self._h5.attrs["n_levels"] = n_levels
        self._h5.attrs["n_episodes"] = self._n_episodes
        self._h5.attrs["n_wins"] = self._n_wins
        self._h5.attrs["agent_state_dim"] = AGENT_STATE_DIM
        self._h5.attrs["mine_feature_dim"] = MINE_FEATURE_DIM
        self._h5.attrs["level_id_map"] = str(self._level_id_map)

        self._h5.close()
        self._h5 = None

        size_mb = Path(self._path).stat().st_size / 1e6
        logger.info(
            "Dataset finalized: %s (%.1f MB, %d transitions, %d levels, %d episodes, %d wins)",
            self._path, size_mb, self._offset, n_levels, self._n_episodes, self._n_wins,
        )
        return self._path

    @property
    def n_transitions(self) -> int:
        return self._offset

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    @property
    def n_levels(self) -> int:
        return len(self._level_id_map)
