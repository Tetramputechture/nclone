"""NativeGymVecEnv: in-process vectorized environment using Rust NativeVecEnv.

Uses NativeVecEnv for batch physics; observation and reward computed in Python
per state. Single process, no IPC - beneficial for small N (e.g. 4-8) vs SubprocVecEnv.
"""

import os
import tempfile
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from .config import EnvironmentConfig
from .npp_environment import NppEnvironment


# Action mapping: discrete 0-5 -> (hor_input, jump_input)
_ACTION_TO_INPUT = [
    (0, 0),   # 0: NOOP
    (-1, 0),  # 1: Left
    (1, 0),   # 2: Right
    (0, 1),   # 3: Jump
    (-1, 1),  # 4: Jump+Left
    (1, 1),   # 5: Jump+Right
]


def _create_native_vecenv(map_data: List[int], num_envs: int):
    """Create NativeVecEnv from map data."""
    import nclone_native
    return nclone_native.NativeVecEnv.create(map_data, num_envs)


class NativeGymVecEnv(VectorEnv):
    """Vectorized environment using Rust NativeVecEnv for batch physics.

    Holds N simulators in Rust; each step runs batch physics then computes
    observation and reward in Python per env. No IPC - single process.
    """

    def __init__(
        self,
        map_data: List[int],
        num_envs: int,
        config: Optional[EnvironmentConfig] = None,
        frame_skip: int = 4,
        map_path: Optional[str] = None,
    ):
        if config is None:
            config = EnvironmentConfig.for_training()
        config.use_native_physics = True
        config.frame_skip = frame_skip
        # Use temp file for map if no path given (needed for graph build)
        if map_path is None:
            fd, map_path = tempfile.mkstemp(suffix=".nmap")
            os.write(fd, bytes(map_data))
            os.close(fd)
            self._temp_map_path = map_path
        else:
            self._temp_map_path = None
        config.custom_map_path = map_path

        self._vecenv = _create_native_vecenv(map_data, num_envs)
        self._frame_skip = frame_skip

        # Create template env for observation/reward logic (shared graph, reward calc)
        self._template_env = NppEnvironment(config)
        # Replace headless with vecenv-backed one
        self._template_env.nplay_headless = _NativeVecEnvBackedHeadless(
            self._vecenv, num_envs
        )
        self._template_env.nplay_headless.sim.gym_env = self._template_env
        self._template_env.nplay_headless.current_map_data = map_data
        # Sync proxy with initial state so map_data is available for level_data
        self._template_env.nplay_headless.set_active_and_sync(0)

        # Trigger graph build via reset
        self._template_env.reset()

        # Per-env state
        self._prev_obs: List[Optional[Dict]] = [None] * num_envs
        self._states = None  # Filled after step

        obs_space = self._template_env.observation_space
        if isinstance(obs_space, spaces.Dict):
            batch_obs_space = spaces.Dict({
                k: self._batch_space(v, num_envs)
                for k, v in obs_space.spaces.items()
            })
        else:
            batch_obs_space = self._batch_space(obs_space, num_envs)

        # VectorEnv base: set attributes directly (no __init__ params)
        self.num_envs = num_envs
        self.observation_space = batch_obs_space
        self.single_observation_space = obs_space
        self.action_space = spaces.MultiDiscrete([6] * num_envs)
        self.single_action_space = spaces.Discrete(6)

    def _batch_space(self, space: spaces.Space, n: int) -> spaces.Space:
        """Create batched space."""
        if isinstance(space, spaces.Box):
            return spaces.Box(
                low=np.broadcast_to(space.low, (n,) + space.shape),
                high=np.broadcast_to(space.high, (n,) + space.shape),
                shape=(n,) + space.shape,
                dtype=space.dtype,
            )
        return space

    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Step all environments."""
        # Convert discrete actions to (hor, jump) and run batch physics
        actions_list = [
            _ACTION_TO_INPUT[int(a)] for a in actions
        ]
        for _ in range(self._frame_skip - 1):
            self._vecenv.step(actions_list)
        self._states = self._vecenv.step(actions_list)

        # Compute obs and reward for each env
        headless = self._template_env.nplay_headless
        obs_list = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i in range(self.num_envs):
            headless.set_active_and_sync(i)
            state = self._states[i]
            prev_obs = self._prev_obs[i]
            if prev_obs is None:
                prev_obs = self._template_env._get_observation()

            obs = self._template_env._get_observation()
            reward = self._template_env.reward_calculator.calculate_reward(
                obs, prev_obs, int(actions[i]),
                frames_executed=self._frame_skip,
                curriculum_manager=getattr(
                    self._template_env, "goal_curriculum_manager", None
                ),
            )
            terminated = obs.get("player_won", False) or obs.get("player_dead", False)
            truncated = self._template_env._check_curriculum_aware_truncation()

            obs_list.append(self._template_env._process_observation(obs))
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            self._prev_obs[i] = obs

            if terminated or truncated:
                self._vecenv.reset_env(i)
                headless.set_active_and_sync(i)
                self._prev_obs[i] = self._template_env._get_observation()

            info = self._template_env._build_episode_info(
                obs.get("player_won", False), terminated, truncated
            )
            infos.append(info)

        batched_obs = self._batch_obs(obs_list)
        return (
            batched_obs,
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            infos,
        )

    def _batch_obs(self, obs_list: List[Dict]) -> Dict[str, np.ndarray]:
        """Stack list of dict obs into batched arrays."""
        batched = {}
        for key in obs_list[0]:
            arr = obs_list[0][key]
            if isinstance(arr, np.ndarray):
                batched[key] = np.stack([o[key] for o in obs_list])
            else:
                batched[key] = np.array([o[key] for o in obs_list])
        return batched

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset all environments."""
        self._vecenv.reset_all()
        headless = self._template_env.nplay_headless
        obs_list = []
        for i in range(self.num_envs):
            headless.set_active_and_sync(i)
            obs = self._template_env._get_observation()
            self._prev_obs[i] = obs
            obs_list.append(self._template_env._process_observation(obs))
        return self._batch_obs(obs_list), {}

    def close(self):
        """Cleanup."""
        self._template_env.close()
        if self._temp_map_path and os.path.exists(self._temp_map_path):
            try:
                os.unlink(self._temp_map_path)
            except OSError:
                pass


class _NativeVecEnvBackedHeadless:
    """Headless that reads from NativeVecEnv by index. No tick - physics in batch."""

    def __init__(self, vecenv, num_envs: int):
        self._vecenv = vecenv
        self._num_envs = num_envs
        self._active_index = 0
        self.current_map_data = None
        self.sim = _VecEnvSimProxy(self)

    def set_active_and_sync(self, index: int):
        """Set active env index and sync proxy from its state."""
        self._active_index = index
        state = self._vecenv.get_state(index)
        self.sim._update_from_state(state)

    def tick(self, horizontal_input: int, jump_input: int):
        """No-op - physics runs in NativeGymVecEnv.step batch."""
        pass

    def load_map_from_map_data(self, map_data: List[int]):
        self.current_map_data = map_data
        self._vecenv.load_all(map_data)

    def reset(self):
        self._vecenv.reset_all()

    def fast_reset(self):
        self._vecenv.reset_all()

    def get_ninja_state(self):
        """Get ninja state from current proxy (matches NativePlayHeadless)."""
        n = self.sim.ninja
        from ..constants.physics_constants import (
            MAX_HOR_SPEED, GRAVITY_JUMP, GRAVITY_FALL,
            DRAG_REGULAR, DRAG_SLOW, FRICTION_GROUND, FRICTION_GROUND_SLOW,
            MAX_JUMP_DURATION,
        )
        from .constants import LEVEL_WIDTH, LEVEL_HEIGHT
        state = []
        state.extend([np.clip(n.xspeed / MAX_HOR_SPEED, -1, 1), np.clip(n.yspeed / MAX_HOR_SPEED, -1, 1)])
        state.extend([1.0 if n.state in (0, 1, 2) else -1.0, 1.0 if n.state in (3, 4) else -1.0,
                     1.0 if n.state == 5 else -1.0, 1.0 if n.state in (6, 7, 8, 9) else -1.0])
        state.append(1.0 if n.airborn else -1.0)
        buffer_window = 5.0
        state.extend([(max(n.jump_buffer, 0) / buffer_window) * 2 - 1,
                     (max(n.floor_buffer, 0) / buffer_window) * 2 - 1,
                     (max(n.wall_buffer, 0) / buffer_window) * 2 - 1])
        jdr = max(0, MAX_JUMP_DURATION - n.jump_duration) / MAX_JUMP_DURATION if n.state == 3 else 0.0
        state.append(jdr)
        state.extend([(min(n.floor_count, 1) * 2) - 1, (min(n.wall_count, 1) * 2) - 1,
                     (min(n.ceiling_count, 1) * 2) - 1])
        state.extend([n.floor_normalized_x, n.floor_normalized_y,
                     float(n.wall_normal) if n.wall_count > 0 else 0.0])
        state.append(1.0 if n.walled else -1.0)
        gravity_range = GRAVITY_FALL - GRAVITY_JUMP
        state.append((n.applied_gravity - GRAVITY_JUMP) / gravity_range * 2 - 1
                     if gravity_range > 1e-6 else 0.0)
        drag_range = DRAG_REGULAR - DRAG_SLOW
        state.append((n.applied_drag - DRAG_SLOW) / drag_range * 2 - 1 if drag_range > 1e-6 else 0.0)
        friction_range = FRICTION_GROUND - FRICTION_GROUND_SLOW
        state.append((n.applied_friction - FRICTION_GROUND_SLOW) / friction_range * 2 - 1
                     if friction_range > 1e-6 else 0.0)
        state.extend([n.xpos / LEVEL_WIDTH, n.ypos / LEVEL_HEIGHT])
        return state

    def get_valid_action_mask(self):
        return list(self.sim._snapshot.action_mask) if self.sim._snapshot else [True] * 6

    def ninja_has_won(self):
        return self.sim.ninja.has_won()

    def ninja_has_died(self):
        return self.sim.ninja.has_died()

    def ninja_position(self):
        return self.sim.ninja.xpos, self.sim.ninja.ypos

    def ninja_velocity(self):
        return self.sim.ninja.xspeed, self.sim.ninja.yspeed

    def ninja_velocity_old(self):
        return self.sim.ninja.xspeed_old, self.sim.ninja.yspeed_old

    def ninja_airborn_old(self):
        return self.sim.ninja.airborn_old

    def ninja_death_cause(self):
        return self.sim.ninja.death_cause

    def get_ninja_terminal_impact(self):
        return self.sim.ninja.terminal_impact

    def exit_switch_activated(self):
        _, switch_active, _ = self._get_switch_door_from_snapshot()
        return not switch_active

    def exit_switch_position(self):
        switch_pos, _, _ = self._get_switch_door_from_snapshot()
        return switch_pos

    def exit_door_position(self):
        _, _, door_pos = self._get_switch_door_from_snapshot()
        return door_pos

    def _get_switch_door_from_snapshot(self):
        from ..native_headless import _get_switch_door_from_entities
        if self.sim._snapshot is None:
            return (0.0, 0.0), True, (0.0, 0.0)
        (switch_pos, switch_active), door_pos = _get_switch_door_from_entities(
            self.sim._snapshot.entities
        )
        return switch_pos, switch_active, door_pos

    def locked_doors(self):
        return []

    def locked_door_switches(self):
        return []

    def get_mine_entities(self):
        self.sim._ensure_entity_dic()
        return self.sim._entity_dic.get(1, []), self.sim._entity_dic.get(21, [])

    def get_tile_data(self):
        return self.sim.tile_dic

    def get_goal_positions_for_features(self):
        return {
            "switch_pos": self.exit_switch_position(),
            "exit_pos": self.exit_door_position(),
            "switch_activated": self.exit_switch_activated(),
            "locked_doors": self.locked_doors(),
            "locked_door_switches": self.locked_door_switches(),
            "mine_entities": self.get_mine_entities(),
        }

    def render(self, debug_info=None):
        return None

    def get_action_mask(self):
        return self.get_valid_action_mask()

    sim_renderer = None
    render_mode = "grayscale_array"
    enable_rendering = False


class _VecEnvSimProxy:
    """Sim proxy for VecEnv-backed headless. Updated from state, not native_sim."""

    def __init__(self, headless: _NativeVecEnvBackedHeadless):
        from ..native_headless import _SimProxy, _NinjaProxy, _SimConfigProxy
        self.ninja = _NinjaProxy()
        self.frame = 0
        self.sim_config = _SimConfigProxy()
        self._entity_dic = {}
        self.tile_dic = {}
        self.grid_entity = {}
        self.gym_env = None
        self._snapshot = None
        self._headless = headless
        self.map_data = None
        self._entity_dic_valid = False
        self._tile_dic_cache_valid = False

    def _update_from_state(self, state):
        """Update from SimulatorState (has tile_grid)."""
        from ..native_headless import _build_entity_dic, _build_tile_dic_from_flat
        self._snapshot = state
        self.frame = state.frame
        self.ninja._update_from_snapshot(state.ninja)
        self._entity_dic_valid = False
        if hasattr(state, "tile_grid") and not self._tile_dic_cache_valid:
            self.tile_dic = _build_tile_dic_from_flat(state.tile_grid)
            self._tile_dic_cache_valid = True
        if self._headless and self._headless.current_map_data is not None:
            self.map_data = self._headless.current_map_data

    def _ensure_entity_dic(self):
        if self._entity_dic_valid and self._snapshot:
            return
        if self._snapshot is None:
            self._entity_dic = {}
            return
        from ..native_headless import _build_entity_dic
        self._entity_dic = _build_entity_dic(self._snapshot.entities)
        self._entity_dic_valid = True

    @property
    def entity_dic(self):
        self._ensure_entity_dic()
        return self._entity_dic
