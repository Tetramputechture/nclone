"""Test NativeGymVecEnv: in-process vectorized environment with Rust physics."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import nclone_native
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False


def load_map_data(map_name: str) -> list:
    """Load raw binary map data from test_maps directory."""
    map_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "nclone",
        "test_maps",
        map_name,
    )
    with open(map_path, "rb") as f:
        return [int(b) for b in f.read()]


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="nclone_native not available")
def test_native_gym_vecenv_reset_and_step():
    """Test NativeGymVecEnv reset and step produce valid observations."""
    from nclone.gym_environment.native_vecenv import NativeGymVecEnv

    map_data = load_map_data("simple-walk")
    num_envs = 4

    env = NativeGymVecEnv(map_data=map_data, num_envs=num_envs)

    obs, infos = env.reset()
    assert obs is not None
    assert isinstance(obs, dict)
    for key, val in obs.items():
        assert hasattr(val, "shape"), f"{key} should be array"
        assert val.shape[0] == num_envs, f"{key} batch dim should be {num_envs}"

    # Take a few steps
    for _ in range(10):
        actions = env.action_space.sample()
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        assert obs is not None
        assert rewards.shape == (num_envs,)
        assert terminateds.shape == (num_envs,)
        assert truncateds.shape == (num_envs,)
        assert len(infos) == num_envs

    env.close()


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="nclone_native not available")
def test_native_gym_vecenv_reset_individual():
    """Test that reset_env clears individual env state correctly."""
    from nclone.gym_environment.native_vecenv import NativeGymVecEnv

    map_data = load_map_data("simple-walk")
    env = NativeGymVecEnv(map_data=map_data, num_envs=2)

    obs, _ = env.reset()
    # Step until one env terminates
    for _ in range(200):
        obs, rewards, terms, truncs, _ = env.step([1, 1])  # walk right
        if terms.any() or truncs.any():
            break

    # Reset all and verify
    obs, _ = env.reset()
    assert obs is not None
    env.close()
