"""v24: fine ego MINE-SDF crop (ego_crop_mines_fine) obs key.

The global hazard_map is 6px but gets average-pooled to a ~7-tile embedding
before the policy sees it, so the lethal 14px mine disc is unresolvable. This
crop slices the 6px hazard SDF ninja-centered so the policy gets sub-tile mine
geometry locally. These tests pin: (1) the key + space are emitted when the flag
is on, (2) the slice co-registers with the global hazard_map, (3) off-grid cells
pad to +1.0 (the SDF "safe" asymptote).
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np

from nclone.gym_environment.config import EnvironmentConfig, ReachabilityConfig
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.map_generation.map_jump_platforms import MapJumpPlatforms


def _make_env(size=21):
    cfg = EnvironmentConfig.for_training(
        frame_skip=1,
        sim_backend="python",
        reachability=ReachabilityConfig(
            enable_ego_crop=True, ego_crop_size=size,
            expose_ego_crop_mines_fine=True,
        ),
    )
    return NppEnvironment(config=cfg)


def test_ego_crop_mines_fine_present_and_shaped():
    env = _make_env(21)
    md = list(MapJumpPlatforms(seed=3).generate(seed=3).map_data())
    obs, _ = env.reset(options={"map_data": md, "map_name": "jp3", "new_level": True})
    assert "ego_crop_mines_fine" in obs
    em = obs["ego_crop_mines_fine"]
    assert em.shape == (2, 21, 21)
    assert em.dtype == np.float32
    sp = env.observation_space.spaces["ego_crop_mines_fine"]
    assert sp.shape == (2, 21, 21)
    # SDF is bounded [-1, 1].
    assert float(em.min()) >= -1.0 - 1e-6 and float(em.max()) <= 1.0 + 1e-6


def test_ego_crop_mines_fine_co_registered_with_hazard_map():
    """Centering the window on a 6px cell must reproduce the global hazard SDF
    there (the crop is a plain co-registered slice at 6px)."""
    env = _make_env(21)
    md = list(MapJumpPlatforms(seed=3).generate(seed=3).map_data())
    env.reset(options={"map_data": md, "map_name": "jp3", "new_level": True})
    hazard = env._compute_hazard_map()  # (2, W, H) 6px playable frame
    # Pick a deadly-mine-proximate cell (most negative SDF = inside a kill disc).
    ch0 = hazard[0]
    hx, hy = np.unravel_index(int(np.argmin(ch0)), ch0.shape)
    # 6px playable cell (hx,hy) -> world px = (cell + border_cells) * 6, border=4.
    cell6 = 6.0
    nx = float((hx + 4) * cell6)
    ny = float((hy + 4) * cell6)
    crop = env._compute_ego_crop_mines_fine(nx, ny)
    c = crop.shape[1] // 2
    # int(nx/6)-4 == hx by construction, so the center cell equals hazard[hx,hy].
    assert abs(float(crop[0, c, c]) - float(ch0[hx, hy])) < 1e-5
    assert float(crop[0].min()) < 1.0  # some real (non-padding) mine signal present


def test_ego_crop_mines_fine_pads_offgrid_to_one():
    env = _make_env(21)
    md = list(MapJumpPlatforms(seed=1).generate(seed=1).map_data())
    env.reset(options={"map_data": md, "map_name": "jp1", "new_level": True})
    # Ninja far off the top-left: whole window overhangs -> all +1.0 (safe).
    crop = env._compute_ego_crop_mines_fine(nx=-1000.0, ny=-1000.0)
    assert crop.shape == (2, 21, 21)
    assert float(crop.min()) == 1.0 and float(crop.max()) == 1.0
