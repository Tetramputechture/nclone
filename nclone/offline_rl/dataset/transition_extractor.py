"""Extract (s, a, r, s', done) transitions from expert demonstrations.

Replays npp_attract demo files and .replay files through the Rust physics
simulator to produce IQL-compatible transition tuples with 11D ninja state
and nearest-8-mine features.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...constants.physics_constants import (
    LEVEL_HEIGHT_PX,
    LEVEL_WIDTH_PX,
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
    NINJA_STATE_CELEBRATING,
)
from ...replay.gameplay_recorder import CompactReplay
from ...replay.npp_attract_decoder import NppAttractDecoder
from ...replay.replay_executor import decode_input_to_controls, map_input_to_action

logger = logging.getLogger(__name__)

NUM_NEAREST_MINES = 8
AGENT_STATE_DIM = 11
MINE_FEATURE_DIM = NUM_NEAREST_MINES * 3  # 8 mines * (rel_x, rel_y, state)
TILE_GRID_W = 42
TILE_GRID_H = 23
TILE_SIZE_PX = 24


@dataclass
class LevelStaticData:
    """Static per-level data extracted once from the demo file."""

    tile_grid: np.ndarray  # (42, 23) int8
    switch_overlay: np.ndarray  # (42, 23) float32
    exit_overlay: np.ndarray  # (42, 23) float32
    level_id: str = ""


@dataclass
class EpisodeTransitions:
    """All transitions extracted from a single demonstration episode."""

    states: np.ndarray  # (T, 11) float32
    actions: np.ndarray  # (T,) int64
    rewards: np.ndarray  # (T,) float32
    next_states: np.ndarray  # (T, 11) float32
    dones: np.ndarray  # (T,) bool
    mine_features: np.ndarray  # (T, 24) float32
    next_mine_features: np.ndarray  # (T, 24) float32
    level_static: LevelStaticData = field(default_factory=LevelStaticData)


def _build_entity_overlay(
    entities: list, entity_type: int, grid_w: int = TILE_GRID_W, grid_h: int = TILE_GRID_H
) -> np.ndarray:
    """Build a (grid_w, grid_h) overlay with 1.0 at cells containing the given entity type."""
    overlay = np.zeros((grid_w, grid_h), dtype=np.float32)
    for ent in entities:
        if ent.entity_type == entity_type:
            cx = int(ent.x) // TILE_SIZE_PX
            cy = int(ent.y) // TILE_SIZE_PX
            cx = max(0, min(cx, grid_w - 1))
            cy = max(0, min(cy, grid_h - 1))
            overlay[cx, cy] = 1.0
    return overlay


def _is_switch_collected(entities: list) -> bool:
    """Check if exit switch has been collected (active=False means collected)."""
    for ent in entities:
        if ent.entity_type == 4:  # exit switch
            return not ent.active
    return False


def _extract_ninja_state(sim_state) -> np.ndarray:
    """Extract 11D ninja state vector from a SimulatorState snapshot.

    Components: x, y, vx, vy, on_ground, wall_sliding,
                jump_frames_remaining, switch_collected,
                floor_normal_x, floor_normal_y, wall_normal
    """
    ninja = sim_state.ninja
    state = np.zeros(AGENT_STATE_DIM, dtype=np.float32)

    state[0] = ninja.xpos / LEVEL_WIDTH_PX
    state[1] = ninja.ypos / LEVEL_HEIGHT_PX
    state[2] = ninja.xspeed / MAX_HOR_SPEED
    state[3] = ninja.yspeed / MAX_HOR_SPEED
    state[4] = 1.0 if not ninja.airborn else 0.0  # on_ground
    state[5] = 1.0 if ninja.state == 5 else 0.0  # wall_sliding (state 5)
    state[6] = ninja.jump_duration / MAX_JUMP_DURATION if ninja.state == 3 else 0.0
    state[7] = 1.0 if _is_switch_collected(sim_state.entities) else 0.0
    state[8] = ninja.floor_normalized_x if not ninja.airborn else 0.0
    state[9] = ninja.floor_normalized_y if not ninja.airborn else 0.0
    state[10] = ninja.wall_normal if ninja.walled else 0.0

    return state


def _extract_mine_features(
    entities: list,
    ninja_x: float,
    ninja_y: float,
) -> np.ndarray:
    """Extract nearest-8-mine features relative to ninja position.

    For each of the 8 closest toggle mines (entity types 1 and 21):
        (relative_x, relative_y, state_value)

    State encoding: -1.0=deadly (toggled/state 0), 0.5=transitioning (state 2),
                    1.0=safe (untoggled/state 1), 0.0=padding (no mine)
    """
    mine_state_map = {0: -1.0, 1: 1.0, 2: 0.5}

    mines = []
    for ent in entities:
        if ent.entity_type in (1, 21):
            dx = ent.x - ninja_x
            dy = ent.y - ninja_y
            dist_sq = dx * dx + dy * dy
            state_val = mine_state_map.get(ent.state, 0.0)
            mines.append((dist_sq, dx, dy, state_val))

    mines.sort(key=lambda m: m[0])

    features = np.zeros(MINE_FEATURE_DIM, dtype=np.float32)
    for i, (_, dx, dy, sv) in enumerate(mines[:NUM_NEAREST_MINES]):
        base = i * 3
        features[base] = dx / LEVEL_WIDTH_PX
        features[base + 1] = dy / LEVEL_HEIGHT_PX
        features[base + 2] = sv

    return features


# Nclone map layout: tiles at bytes 184-1149 (966 bytes)
_MAP_TILES_OFFSET = 184
_MAP_TILES_LEN = 966


class TransitionExtractor:
    """Extract RL transitions from npp_attract and .replay demonstration files.

    Uses the Rust Simulator for deterministic replay of expert demonstrations,
    collecting (state, action, reward, next_state, done) tuples with 11D
    ninja state and nearest-8-mine features at each physics frame.
    """

    def __init__(self, max_episode_frames: int = 18000):
        """Initialize the extractor.

        Args:
            max_episode_frames: Maximum frames per episode before forced termination.
                Default 18000 = 5 minutes at 60fps.
        """
        self.decoder = NppAttractDecoder()
        self.max_episode_frames = max_episode_frames

    def extract_from_file(self, demo_path: str) -> Optional[EpisodeTransitions]:
        """Extract transitions from a single demo file.

        Supports two formats (detected by extension):
        - .replay: CompactReplay format from test_environment recording
        - No extension: npp_attract format

        Args:
            demo_path: Path to a demo file (.replay or no extension).

        Returns:
            EpisodeTransitions with all frame-by-frame transitions,
            or None if the file cannot be decoded.
        """
        try:
            import nclone_native
        except ImportError:
            raise ImportError(
                "nclone_native is required for transition extraction. "
                "Build it with: cd nclone-native && maturin develop --release"
            )

        path = Path(demo_path)
        if path.suffix == ".replay":
            # Load .replay format (test_environment recording)
            try:
                with open(demo_path, "rb") as f:
                    data = f.read()
                replay = CompactReplay.from_binary(data, episode_id=path.stem)
            except Exception as e:
                logger.warning("Failed to load replay file %s: %s", demo_path, e)
                return None
            if not replay.success:
                logger.warning("Replay %s did not end in success, skipping", demo_path)
                return None
            demo_inputs = replay.input_sequence
            map_bytes = replay.map_data
            if len(map_bytes) < _MAP_TILES_OFFSET + _MAP_TILES_LEN:
                logger.warning(
                    "Replay %s map_data too short (%d bytes), skipping",
                    demo_path,
                    len(map_bytes),
                )
                return None
            tiles_raw = list(
                map_bytes[_MAP_TILES_OFFSET : _MAP_TILES_OFFSET + _MAP_TILES_LEN]
            )
        else:
            # Load npp_attract format (no extension)
            decoded = self.decoder.decode_npp_attract_file(demo_path)
            demo_inputs = decoded.get("demo_data", [])
            map_bytes = self.decoder.create_nclone_map(demo_path)
            tiles_raw = decoded["tiles"]

        if not demo_inputs:
            logger.warning("No demo data in %s, skipping", demo_path)
            return None

        map_data = [int(b) for b in map_bytes]
        sim = nclone_native.Simulator.load(map_data)
        sim_state = sim.get_state()

        # Extract static per-level data
        tile_grid = (
            np.array(tiles_raw, dtype=np.int8).reshape(TILE_GRID_H, TILE_GRID_W).T
        )
        switch_overlay = _build_entity_overlay(sim_state.entities, entity_type=4)
        exit_overlay = _build_entity_overlay(sim_state.entities, entity_type=3)

        level_static = LevelStaticData(
            tile_grid=tile_grid,
            switch_overlay=switch_overlay,
            exit_overlay=exit_overlay,
            level_id=path.stem,
        )

        # Pre-allocate arrays
        n_frames = min(len(demo_inputs), self.max_episode_frames)
        states = np.zeros((n_frames, AGENT_STATE_DIM), dtype=np.float32)
        next_states = np.zeros((n_frames, AGENT_STATE_DIM), dtype=np.float32)
        actions = np.zeros(n_frames, dtype=np.int64)
        rewards = np.zeros(n_frames, dtype=np.float32)
        dones = np.zeros(n_frames, dtype=np.bool_)
        mine_feats = np.zeros((n_frames, MINE_FEATURE_DIM), dtype=np.float32)
        next_mine_feats = np.zeros((n_frames, MINE_FEATURE_DIM), dtype=np.float32)

        # Get initial state
        current_state = _extract_ninja_state(sim_state)
        ninja_x, ninja_y = sim_state.ninja.xpos, sim_state.ninja.ypos
        current_mine_feat = _extract_mine_features(sim_state.entities, ninja_x, ninja_y)

        actual_frames = 0
        episode_ended_in_win = False
        for frame_idx in range(n_frames):
            input_byte = demo_inputs[frame_idx]
            hor_input, jump_input = decode_input_to_controls(input_byte)
            action = map_input_to_action(input_byte)

            # Store current state
            states[frame_idx] = current_state
            actions[frame_idx] = action
            mine_feats[frame_idx] = current_mine_feat

            # Advance simulation
            sim.tick(hor_input, jump_input)
            sim_state = sim.get_state()

            # Check termination: has_won() and celebrating state (state==8) both indicate win
            won = sim.has_won() or sim_state.ninja.state == NINJA_STATE_CELEBRATING
            died = sim.has_died()
            done = won or died

            # Reward: -1 per frame, 0 at terminal win
            rewards[frame_idx] = 0.0 if won else -1.0

            # Extract next state
            next_s = _extract_ninja_state(sim_state)
            ninja_x, ninja_y = sim_state.ninja.xpos, sim_state.ninja.ypos
            next_mf = _extract_mine_features(sim_state.entities, ninja_x, ninja_y)

            next_states[frame_idx] = next_s
            next_mine_feats[frame_idx] = next_mf
            dones[frame_idx] = done

            actual_frames = frame_idx + 1

            if done:
                episode_ended_in_win = won
                break

            # Prepare for next iteration
            current_state = next_s
            current_mine_feat = next_mf

        # Reject episodes that don't end in a win (death, timeout, etc.)
        if not episode_ended_in_win:
            return None

        # Trim to actual length
        return EpisodeTransitions(
            states=states[:actual_frames],
            actions=actions[:actual_frames],
            rewards=rewards[:actual_frames],
            next_states=next_states[:actual_frames],
            dones=dones[:actual_frames],
            mine_features=mine_feats[:actual_frames],
            next_mine_features=next_mine_feats[:actual_frames],
            level_static=level_static,
        )

    def extract_from_directory(
        self, demo_dir: str, max_files: Optional[int] = None
    ) -> List[EpisodeTransitions]:
        """Extract transitions from all demo files in a directory.

        Args:
            demo_dir: Path to directory containing npp_attract demo files.
            max_files: Maximum number of files to process (None = all).

        Returns:
            List of EpisodeTransitions, one per successfully processed demo.
        """
        demo_path = Path(demo_dir)
        if not demo_path.is_dir():
            raise FileNotFoundError(f"Demo directory not found: {demo_dir}")

        # Collect demo files: attract (no extension) or .replay from test_environment
        def _is_demo_file(f: Path) -> bool:
            if not f.is_file() or f.name.startswith("."):
                return False
            return f.suffix == "" or f.suffix == ".replay"

        all_files = sorted([f for f in demo_path.iterdir() if _is_demo_file(f)])
        if max_files is not None:
            all_files = all_files[:max_files]

        logger.info("Found %d files in %s", len(all_files), demo_dir)

        episodes = []
        for i, fpath in enumerate(all_files):
            if (i + 1) % 100 == 0:
                logger.info("Processing file %d/%d: %s", i + 1, len(all_files), fpath.name)

            ep = self.extract_from_file(str(fpath))
            if ep is not None and len(ep.states) > 0:
                episodes.append(ep)

        logger.info(
            "Extracted %d episodes, %d total transitions",
            len(episodes),
            sum(len(ep.states) for ep in episodes),
        )
        return episodes
