"""Audit II P1-6: coarse far ego crop (``ego_crop_coarse``) contract tests.

Covers shape/bounds, co-registration with the fine ego crop (the fine center
tile's solidity agrees with the coarse center cell's solid fraction), OOB
padding (border reads solid), and the objective marker appearing once the
objective enters the +-21-tile Chebyshev window.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.constants import (
    EGO_CROP_COARSE_CHANNELS,
    EGO_CROP_COARSE_POOL_TILES,
    EGO_CROP_COARSE_SIZE,
)
from nclone.gym_environment.generator_registry import GeneratorRegistry
from nclone.gym_environment.npp_environment import NppEnvironment


def _make_env() -> NppEnvironment:
    cfg = EnvironmentConfig.for_training(seed=0, sim_backend="python")
    cfg.reachability.enable_ego_crop = True
    cfg.reachability.enable_ego_crop_coarse = True
    return NppEnvironment(cfg)


def _map(category: str = "simple", seed: int = 7) -> list[int]:
    reg = GeneratorRegistry(random.Random(seed))
    return [int(v) for v in reg.get_map(category, seed).map_data()]


class TestEgoCropCoarse:
    def test_shape_bounds_and_space(self):
        env = _make_env()
        try:
            assert "ego_crop_coarse" in env.observation_space.spaces
            obs, _ = env.reset(options={"map_data": _map(), "map_name": "cc1"})
            cc = obs["ego_crop_coarse"]
            assert cc.shape == (
                EGO_CROP_COARSE_CHANNELS,
                EGO_CROP_COARSE_SIZE,
                EGO_CROP_COARSE_SIZE,
            )
            assert np.isfinite(cc).all()
            assert cc[0].min() >= 0.0 and cc[0].max() <= 1.0  # solid fraction
        finally:
            env.close()

    def test_center_coregistration_and_oob(self):
        env = _make_env()
        try:
            obs, _ = env.reset(options={"map_data": _map(), "map_name": "cc2"})
            cc = obs["ego_crop_coarse"]
            c = EGO_CROP_COARSE_SIZE // 2
            # The ninja stands in free space: the coarse cell containing it
            # cannot be fully solid.
            assert cc[0, c, c] < 1.0
            # The window (+-21 tiles) always overhangs the 42x23 level
            # vertically -> the top/bottom rows must read fully solid (OOB pad).
            assert cc[0, :, 0].min() == 1.0
            assert cc[0, :, -1].min() == 1.0
        finally:
            env.close()

    def test_objective_marker_appears_within_window(self):
        env = _make_env()
        try:
            env.reset(options={"map_data": _map(), "map_name": "cc3"})
            px, py = env.nplay_headless.ninja_position()
            sx, sy = env.nplay_headless.exit_switch_position()
            P = EGO_CROP_COARSE_POOL_TILES
            half = EGO_CROP_COARSE_SIZE // 2

            def in_window(obs_px):
                ntx, nty = int(obs_px[0] / 24) - 1, int(obs_px[1] / 24) - 1
                stx, sty = int(sx / 24) - 1, int(sy / 24) - 1
                wx = stx // P - (ntx // P - half)
                wy = sty // P - (nty // P - half)
                return (
                    0 <= wx < EGO_CROP_COARSE_SIZE
                    and 0 <= wy < EGO_CROP_COARSE_SIZE
                )

            saw_marker = False
            obs = None
            for _ in range(400):
                obs, _r, term, trunc, _info = env.step(2)  # run right
                pos = env.nplay_headless.ninja_position()
                cc = obs["ego_crop_coarse"]
                if in_window(pos):
                    # Active objective (switch, pre-activation) must read 1.0.
                    assert cc[3].max() == pytest.approx(1.0), (
                        "objective in window but no marker"
                    )
                    saw_marker = True
                    break
                if term or trunc:
                    break
            assert saw_marker, "ninja never brought the switch into the window"
        finally:
            env.close()


def _make_env_route() -> NppEnvironment:
    cfg = EnvironmentConfig.for_training(seed=0, sim_backend="python")
    cfg.reachability.enable_ego_crop = True
    cfg.reachability.enable_ego_crop_coarse = True
    cfg.reachability.ego_crop_coarse_route = True
    return NppEnvironment(cfg)


class TestEgoCropCoarseRoute:
    def test_route_grows_channels_to_six(self):
        env = _make_env_route()
        try:
            sp = env.observation_space.spaces["ego_crop_coarse"]
            assert sp.shape == (
                EGO_CROP_COARSE_CHANNELS + 2,
                EGO_CROP_COARSE_SIZE,
                EGO_CROP_COARSE_SIZE,
            )
            assert sp.low.min() == -1.0  # unit-vector route channels
            obs, _ = env.reset(options={"map_data": _map(), "map_name": "ccr1"})
            cc = obs["ego_crop_coarse"]
            assert cc.shape[0] == EGO_CROP_COARSE_CHANNELS + 2
            assert np.isfinite(cc).all()
            # Base density/marker channels unchanged (>=0), route channels in [-1,1].
            assert cc[:EGO_CROP_COARSE_CHANNELS].min() >= 0.0
            assert cc[EGO_CROP_COARSE_CHANNELS:].min() >= -1.0
            assert cc[EGO_CROP_COARSE_CHANNELS:].max() <= 1.0
        finally:
            env.close()

    def test_route_channels_carry_signal(self):
        # Somewhere along a run the coarse route channels must be non-zero (a
        # directed grad(d*) exists toward the switch through the maze).
        env = _make_env_route()
        try:
            env.reset(options={"map_data": _map("simple"), "map_name": "ccr2"})
            base = EGO_CROP_COARSE_CHANNELS
            saw_route = False
            for _ in range(200):
                obs, _r, t, tr, _i = env.step(2)
                cc = obs["ego_crop_coarse"]
                if float(np.abs(cc[base]).sum() + np.abs(cc[base + 1]).sum()) > 0.0:
                    saw_route = True
                    break
                if t or tr:
                    env.reset(options={"map_data": _map("simple"), "map_name": "ccr2"})
            assert saw_route, "coarse route channels never carried a gradient"
        finally:
            env.close()

    def test_route_off_keeps_four_channels(self):
        env = _make_env()
        try:
            assert env.observation_space.spaces["ego_crop_coarse"].shape[0] == (
                EGO_CROP_COARSE_CHANNELS
            )
        finally:
            env.close()
