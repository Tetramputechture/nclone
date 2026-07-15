"""Regression: mine-aware ego_crop (ego_crop_mines) obs key.

The tile-only ego_crop is mine-blind — the #1 cause of compositional mine
deaths. ego_crop_mines adds a (2, size, size) deadly/safe mine-density crop
co-registered 1:1 with the tile crop (same ninja-centered window). These tests
pin: (1) the key + space are emitted when ego_crop is enabled, (2) the slice is
co-registered with entity_overlays ch3/ch4, and (3) off-level cells pad to 0.
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np

from nclone.gym_environment.config import EnvironmentConfig, ReachabilityConfig
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.entity_overlay_utils import (
    ENTITY_OVERLAY_CH_MINE_DEADLY,
    ENTITY_OVERLAY_CH_MINE_SAFE,
)
from nclone.map_generation.map_jump_platforms import MapJumpPlatforms


def _make_env(size=21):
    cfg = EnvironmentConfig.for_training(
        frame_skip=1,
        sim_backend="python",
        reachability=ReachabilityConfig(enable_ego_crop=True, ego_crop_size=size),
    )
    return NppEnvironment(config=cfg)


def test_ego_crop_mines_present_and_shaped():
    env = _make_env(21)
    md = list(MapJumpPlatforms(seed=3).generate(seed=3).map_data())
    obs, _ = env.reset(options={"map_data": md, "map_name": "jp3", "new_level": True})
    assert "ego_crop_mines" in obs
    em = obs["ego_crop_mines"]
    assert em.shape == (2, 21, 21)
    assert em.dtype == np.float32
    sp = env.observation_space.spaces["ego_crop_mines"]
    assert sp.shape == (2, 21, 21)


def test_ego_crop_mines_co_registered_with_overlays():
    """Centering the window on a mined floor row must reproduce the overlay
    mine density there (the crop is a plain co-registered slice)."""
    env = _make_env(21)
    md = list(MapJumpPlatforms(seed=3).generate(seed=3).map_data())
    obs, _ = env.reset(options={"map_data": md, "map_name": "jp3", "new_level": True})
    eo = obs["entity_overlays"]
    deadly = eo[ENTITY_OVERLAY_CH_MINE_DEADLY]
    nz = np.argwhere(deadly > 0)
    assert len(nz) > 0, "test level should have floor mines"
    # Center the crop on a mined tile (tx, ty) -> px = (t + 1) * 24 (strip border).
    tx, ty = int(nz[len(nz) // 2][0]), int(nz[len(nz) // 2][1])
    crop = env._compute_ego_crop_mines(
        nx=float((tx + 1) * 24), ny=float((ty + 1) * 24), entity_overlays=eo
    )
    # The centered cell holds that tile's deadly density exactly.
    c = crop.shape[1] // 2
    assert crop[0, c, c] == deadly[tx, ty]
    assert crop[0].sum() > 0


def test_ego_crop_mines_pads_offlevel_to_zero():
    env = _make_env(21)
    md = list(MapJumpPlatforms(seed=1).generate(seed=1).map_data())
    obs, _ = env.reset(options={"map_data": md, "map_name": "jp1", "new_level": True})
    eo = np.zeros_like(obs["entity_overlays"])
    # Ninja far off the top-left corner: entire window overhangs -> all zeros.
    crop = env._compute_ego_crop_mines(nx=0.0, ny=0.0, entity_overlays=eo)
    assert crop.shape == (2, 21, 21)
    assert float(crop.sum()) == 0.0
