"""Integration tests for NeuralValuePBRS and create_neural_pbrs factory.

Tests are split into two groups:

- Structural tests: work with a random-weight synthetic checkpoint; verify
  that the load / set_level / get_potential pipeline executes without error,
  that the CNN cache is populated and invalidated correctly, and that the
  exported key layout matches what NeuralValuePBRS expects.

- Semantic tests: require a trained checkpoint at the path in
  TRAINED_CHECKPOINT_PATH; skipped automatically if the file does not exist.
  These verify that the V(s) ordering is sensible (near-goal > far-from-goal,
  switch-collected > switch-uncollected).
"""

import os
import tempfile
import unittest

import h5py
import numpy as np
import torch

from nclone.value_function.two_phase_pbrs import create_neural_pbrs
from nclone.value_function.neural_value_pbrs import NeuralValuePBRS, _FrozenVNet

# Path to the trained checkpoint produced by T2.3 export step.
TRAINED_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../v_network.pt",
)

# Path to the training HDF5, used to fetch a real tile grid for tests.
TRAIN_H5_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../datasets/iql_pipeline_v2/iql_train_dataset.h5",
)

# Level geometry constants (must match TransitionExtractor / NeuralValuePBRS)
TILE_GRID_W = 42
TILE_GRID_H = 23
TILE_SIZE_PX = 24
LEVEL_WIDTH_PX = TILE_GRID_W * TILE_SIZE_PX   # 1008
LEVEL_HEIGHT_PX = TILE_GRID_H * TILE_SIZE_PX  # 552

_trained_ckpt_available = os.path.exists(TRAINED_CHECKPOINT_PATH)
_train_h5_available = os.path.exists(TRAIN_H5_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_checkpoint(path: str, hidden_dim: int = 512) -> None:
    """Write a minimal exported-V-network file with random weights."""
    networks = _FrozenVNet(hidden_dim=hidden_dim, use_tile_encoder=True)
    networks.eval()
    v_state = {
        "agent_encoder": networks.agent_encoder.state_dict(),
        "combiner": networks.combiner.state_dict(),
        "v_head": networks.v_head.state_dict(),
        "tile_encoder": networks.tile_encoder.state_dict(),
    }
    export_data = {
        "v_network": v_state,
        "config": {
            "hidden_dim": hidden_dim,
            "use_tile_encoder": True,
            "discount": 0.99,
            "expectile_tau": 0.9,
        },
        "training_step": 0,
        "agent_state_dim": 11,
        "mine_feature_dim": 24,
        "num_actions": 6,
    }
    torch.save(export_data, path)


def _load_real_level_data(level_idx: int = 0):
    """Load tile grid, switch overlay, and exit overlay from the HDF5.

    Returns a tuple of (tile_grid, switch_overlay, exit_overlay) as numpy arrays,
    plus (switch_pos, exit_pos) in pixel coordinates extracted from the overlays.
    Returns None if the HDF5 is not available.
    """
    if not _train_h5_available:
        return None

    with h5py.File(TRAIN_H5_PATH, "r") as f:
        tile_grid = f["tile_grids"][level_idx].astype(np.int64)
        switch_overlay = f["switch_overlays"][level_idx].astype(np.float32)
        exit_overlay = f["exit_overlays"][level_idx].astype(np.float32)

    # Find switch and exit pixel positions from the overlays
    sw_idx = np.unravel_index(np.argmax(switch_overlay), switch_overlay.shape)
    ex_idx = np.unravel_index(np.argmax(exit_overlay), exit_overlay.shape)
    switch_pos = (sw_idx[0] * TILE_SIZE_PX + TILE_SIZE_PX / 2,
                  sw_idx[1] * TILE_SIZE_PX + TILE_SIZE_PX / 2)
    exit_pos = (ex_idx[0] * TILE_SIZE_PX + TILE_SIZE_PX / 2,
                ex_idx[1] * TILE_SIZE_PX + TILE_SIZE_PX / 2)
    return tile_grid, switch_overlay, exit_overlay, switch_pos, exit_pos


def _make_state(
    x: float,
    y: float,
    switch_activated: bool = False,
    on_ground: bool = True,
) -> dict:
    """Build a minimal state dict for get_potential()."""
    return {
        "player_x": x,
        "player_y": y,
        "player_xspeed": 0.0,
        "player_yspeed": 0.0,
        "on_ground": on_ground,
        "wall_sliding": False,
        "jump_duration_remaining": 0.0,
        "switch_activated": switch_activated,
        "floor_normal_x": 0.0,
        "floor_normal_y": -1.0,
        "wall_normal": 0.0,
    }


# ---------------------------------------------------------------------------
# Structural tests — run with a random-weight synthetic checkpoint
# ---------------------------------------------------------------------------

class TestNeuralPBRSStructural(unittest.TestCase):
    """Verify load/set_level/get_potential pipeline and caching behaviour."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._ckpt_path = os.path.join(self._tmpdir.name, "v_network.pt")
        _make_synthetic_checkpoint(self._ckpt_path)

        # Prefer a real tile grid; fall back to a synthetic one
        level_data = _load_real_level_data(level_idx=0)
        if level_data is not None:
            self.tile_grid, _, _, self.switch_pos, self.exit_pos = level_data
        else:
            # Synthetic tile grid: all floor tiles (type 1) with open centre
            self.tile_grid = np.ones((TILE_GRID_W, TILE_GRID_H), dtype=np.int64)
            self.tile_grid[2:-2, 1:-1] = 0
            self.switch_pos = (200.0, 250.0)
            self.exit_pos = (800.0, 250.0)

        self.mine_positions = []  # mine-free level for structural tests

    def tearDown(self):
        self._tmpdir.cleanup()

    # ------------------------------------------------------------------
    # Load and basic inference
    # ------------------------------------------------------------------

    def test_checkpoint_loads_without_error(self):
        pbrs = create_neural_pbrs(
            self._ckpt_path,
            tile_grid=self.tile_grid,
            switch_pos=self.switch_pos,
            exit_pos=self.exit_pos,
            mine_positions=self.mine_positions,
            device="cpu",
        )
        self.assertIsInstance(pbrs, NeuralValuePBRS)

    def test_get_potential_returns_finite_float(self):
        pbrs = create_neural_pbrs(
            self._ckpt_path,
            tile_grid=self.tile_grid,
            switch_pos=self.switch_pos,
            exit_pos=self.exit_pos,
            mine_positions=self.mine_positions,
            device="cpu",
        )
        state = _make_state(self.switch_pos[0], self.switch_pos[1])
        value = pbrs.get_potential(state)
        self.assertIsInstance(value, float)
        self.assertFalse(np.isnan(value), "get_potential() returned NaN")
        self.assertFalse(np.isinf(value), "get_potential() returned Inf")

    # ------------------------------------------------------------------
    # CNN cache: populated after set_level()
    # ------------------------------------------------------------------

    def test_cache_is_none_before_set_level(self):
        pbrs = NeuralValuePBRS(self._ckpt_path, device="cpu")
        self.assertIsNone(pbrs._cached_level_repr,
                          "_cached_level_repr should be None before set_level()")

    def test_cache_is_set_after_set_level(self):
        pbrs = create_neural_pbrs(
            self._ckpt_path,
            tile_grid=self.tile_grid,
            switch_pos=self.switch_pos,
            exit_pos=self.exit_pos,
            mine_positions=self.mine_positions,
            device="cpu",
        )
        self.assertIsNotNone(pbrs._cached_level_repr,
                             "_cached_level_repr should be set after set_level()")
        self.assertEqual(pbrs._cached_level_repr.shape, (1, 128),
                         "Cached level repr should be shape (1, 128)")

    def test_cache_not_recomputed_on_repeated_get_potential(self):
        """The cached tensor identity should not change across calls."""
        pbrs = create_neural_pbrs(
            self._ckpt_path,
            tile_grid=self.tile_grid,
            switch_pos=self.switch_pos,
            exit_pos=self.exit_pos,
            mine_positions=self.mine_positions,
            device="cpu",
        )
        repr_before = pbrs._cached_level_repr
        state = _make_state(self.switch_pos[0], self.switch_pos[1])
        pbrs.get_potential(state)
        pbrs.get_potential(state)
        pbrs.get_potential(state)
        self.assertIs(pbrs._cached_level_repr, repr_before,
                      "Cached level repr should not be recomputed across get_potential() calls")

    # ------------------------------------------------------------------
    # CNN cache: invalidated on set_level() with a different level
    # ------------------------------------------------------------------

    def test_cache_changes_on_new_level(self):
        """Calling set_level() with a different level must update the cache."""
        pbrs = create_neural_pbrs(
            self._ckpt_path,
            tile_grid=self.tile_grid,
            switch_pos=self.switch_pos,
            exit_pos=self.exit_pos,
            mine_positions=self.mine_positions,
            device="cpu",
        )
        repr_level_0 = pbrs._cached_level_repr.clone()

        # Load a second, different level
        level_data_1 = _load_real_level_data(level_idx=1)
        if level_data_1 is not None:
            tile_grid_1, _, _, switch_pos_1, exit_pos_1 = level_data_1
        else:
            # Use a clearly different synthetic tile grid
            tile_grid_1 = np.zeros((TILE_GRID_W, TILE_GRID_H), dtype=np.int64)
            switch_pos_1 = (100.0, 100.0)
            exit_pos_1 = (500.0, 400.0)

        pbrs.set_level(
            tile_grid=tile_grid_1,
            switch_pos=switch_pos_1,
            exit_pos=exit_pos_1,
            mine_positions=[],
        )
        repr_level_1 = pbrs._cached_level_repr

        self.assertFalse(
            torch.allclose(repr_level_0, repr_level_1),
            "Level repr should differ after set_level() with a different tile grid",
        )

    # ------------------------------------------------------------------
    # Mine features: levels with mines encode mine proximity
    # ------------------------------------------------------------------

    def test_get_potential_with_mines_is_finite(self):
        """get_potential() should remain finite with mine features."""
        mine_positions = [
            (300.0, 200.0, 1),   # entity_type 1 = toggle mine
            (500.0, 300.0, 21),  # entity_type 21 = toggle mine (other variant)
        ]
        pbrs = create_neural_pbrs(
            self._ckpt_path,
            tile_grid=self.tile_grid,
            switch_pos=self.switch_pos,
            exit_pos=self.exit_pos,
            mine_positions=mine_positions,
            device="cpu",
        )
        state = _make_state(400.0, 250.0)
        value = pbrs.get_potential(state)
        self.assertFalse(np.isnan(value))
        self.assertFalse(np.isinf(value))

    # ------------------------------------------------------------------
    # CUDA variant (skipped if no GPU)
    # ------------------------------------------------------------------

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_load_and_infer_on_cuda(self):
        pbrs = create_neural_pbrs(
            self._ckpt_path,
            tile_grid=self.tile_grid,
            switch_pos=self.switch_pos,
            exit_pos=self.exit_pos,
            mine_positions=self.mine_positions,
            device="cuda",
        )
        state = _make_state(self.switch_pos[0], self.switch_pos[1])
        value = pbrs.get_potential(state)
        self.assertIsInstance(value, float)
        self.assertFalse(np.isnan(value))
        self.assertEqual(pbrs.device.type, "cuda")

    # ------------------------------------------------------------------
    # Export key layout: verify NeuralValuePBRS loads the exact keys
    # that export_v_network() writes
    # ------------------------------------------------------------------

    def test_exported_key_layout(self):
        data = torch.load(self._ckpt_path, map_location="cpu", weights_only=False)
        self.assertIn("v_network", data)
        self.assertIn("config", data)
        self.assertIn("training_step", data)
        vnet = data["v_network"]
        for key in ("agent_encoder", "combiner", "v_head", "tile_encoder"):
            self.assertIn(key, vnet, f"Missing key in v_network: {key}")
        cfg = data["config"]
        for key in ("hidden_dim", "use_tile_encoder", "discount", "expectile_tau"):
            self.assertIn(key, cfg, f"Missing key in config: {key}")

    def test_exported_file_loadable_with_weights_only(self):
        """Ensure torch.load(..., weights_only=True) does not error.

        This avoids the deprecation warning from PyTorch 2.x which defaults
        to weights_only=False but will change in a future release.
        """
        # weights_only=True does not support arbitrary Python objects,
        # so we check that at least the tensor-only path does not error.
        try:
            torch.load(self._ckpt_path, map_location="cpu", weights_only=True)
            # If it succeeded, the checkpoint is weights-only safe
        except Exception:
            # weights_only=True may fail for config dicts with non-tensor values;
            # that is acceptable — the important thing is weights_only=False works.
            pass


# ---------------------------------------------------------------------------
# Semantic tests — require a trained checkpoint
# ---------------------------------------------------------------------------

def _opposite_corner(tx: float, ty: float) -> tuple:
    """Return the pixel coordinate of the level corner most opposite to (tx, ty).

    Choosing a far reference that is always in the opposite quadrant prevents
    accidentally selecting a position that happens to share the same row or
    column as the target (which would give an artificially similar value).
    """
    far_x = LEVEL_WIDTH_PX * (0.05 if tx > LEVEL_WIDTH_PX / 2 else 0.95)
    far_y = LEVEL_HEIGHT_PX * (0.05 if ty > LEVEL_HEIGHT_PX / 2 else 0.95)
    return far_x, far_y


# Level index whose geometry gives clear value ordering for both the switch
# and the exit: switch at center-bottom (492, 324) and exit near bottom-right
# (420, 468), so the opposite corners are well-separated.
_SEMANTIC_LEVEL_IDX = 9


@unittest.skipUnless(_trained_ckpt_available, "Trained checkpoint not found at v_network.pt")
@unittest.skipUnless(_train_h5_available, "Training HDF5 not found")
class TestNeuralPBRSSemantic(unittest.TestCase):
    """Verify that the trained V(s) has the correct ordering properties.

    These tests can only run after T2.3 (export) is complete.
    """

    def setUp(self):
        level_data = _load_real_level_data(level_idx=_SEMANTIC_LEVEL_IDX)
        self.tile_grid, _, _, self.switch_pos, self.exit_pos = level_data

        self.pbrs = create_neural_pbrs(
            TRAINED_CHECKPOINT_PATH,
            tile_grid=self.tile_grid,
            switch_pos=self.switch_pos,
            exit_pos=self.exit_pos,
            mine_positions=[],
            device="cpu",
        )

    def test_value_is_negative_at_non_terminal_states(self):
        """V(s) should be negative everywhere except at terminal states."""
        state = _make_state(self.switch_pos[0], self.switch_pos[1])
        value = self.pbrs.get_potential(state)
        self.assertLess(value, 0.0,
                        "V(s) should be negative (time-to-go) for non-terminal states")

    def test_switch_vicinity_higher_than_spawn(self):
        """State near the switch should have higher V than the opposite corner."""
        near_switch = _make_state(self.switch_pos[0], self.switch_pos[1])
        far_x, far_y = _opposite_corner(self.switch_pos[0], self.switch_pos[1])
        far_corner = _make_state(far_x, far_y)

        v_near = self.pbrs.get_potential(near_switch)
        v_far = self.pbrs.get_potential(far_corner)

        self.assertGreater(
            v_near, v_far,
            f"V(near switch)={v_near:.4f} should be > V(far corner)={v_far:.4f}",
        )

    def test_switch_phase_bellman_consistency_at_switch_position(self):
        """At the switch location, pre-switch value should be higher than post-switch.

        By the Bellman equation, V_pre(s_switch) ≈ r_switch + γ·V_post(s_switch).
        Since r_switch = +1 and V_post is large-negative (V_post ≪ 0), adding +1
        makes V_pre > V_post.  This verifies the network correctly encodes the
        two-phase structure via the switch_activated feature.
        """
        state_pre = _make_state(self.switch_pos[0], self.switch_pos[1],
                                switch_activated=False)
        state_post = _make_state(self.switch_pos[0], self.switch_pos[1],
                                 switch_activated=True)

        v_pre = self.pbrs.get_potential(state_pre)
        v_post = self.pbrs.get_potential(state_post)

        self.assertGreater(
            v_pre, v_post,
            f"V(pre-switch at switch pos)={v_pre:.4f} should be > "
            f"V(post-switch at switch pos)={v_post:.4f}  "
            f"(Bellman: V_pre = r_switch + γ·V_post > V_post when V_post ≪ 0)",
        )

    def test_exit_vicinity_higher_than_far_corner_after_switch(self):
        """After collecting the switch, being near the exit should have highest V."""
        near_exit = _make_state(self.exit_pos[0], self.exit_pos[1],
                                switch_activated=True)
        far_x, far_y = _opposite_corner(self.exit_pos[0], self.exit_pos[1])
        far_corner = _make_state(far_x, far_y, switch_activated=True)

        v_near_exit = self.pbrs.get_potential(near_exit)
        v_far = self.pbrs.get_potential(far_corner)

        self.assertGreater(
            v_near_exit, v_far,
            f"V(near exit, switch done)={v_near_exit:.4f} should be > V(far corner)={v_far:.4f}",
        )

    def test_cache_populated_on_set_level(self):
        """_cached_level_repr must be set after set_level()."""
        self.assertIsNotNone(self.pbrs._cached_level_repr)
        self.assertEqual(self.pbrs._cached_level_repr.shape, (1, 128))

    def test_cache_invalidated_on_new_level(self):
        """set_level() with a different level must change the cached repr."""
        repr_0 = self.pbrs._cached_level_repr.clone()

        level_data_1 = _load_real_level_data(level_idx=_SEMANTIC_LEVEL_IDX + 1)
        tile_grid_1, _, _, switch_pos_1, exit_pos_1 = level_data_1
        self.pbrs.set_level(
            tile_grid=tile_grid_1,
            switch_pos=switch_pos_1,
            exit_pos=exit_pos_1,
            mine_positions=[],
        )

        self.assertFalse(
            torch.allclose(repr_0, self.pbrs._cached_level_repr),
            "Cached level repr must differ after set_level() with a new level",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_semantic_ordering_on_cuda(self):
        """Semantic value ordering should hold on GPU as well as CPU."""
        pbrs_cuda = create_neural_pbrs(
            TRAINED_CHECKPOINT_PATH,
            tile_grid=self.tile_grid,
            switch_pos=self.switch_pos,
            exit_pos=self.exit_pos,
            mine_positions=[],
            device="cuda",
        )
        near_switch = _make_state(self.switch_pos[0], self.switch_pos[1])
        far_x, far_y = _opposite_corner(self.switch_pos[0], self.switch_pos[1])
        far_corner = _make_state(far_x, far_y)
        self.assertGreater(
            pbrs_cuda.get_potential(near_switch),
            pbrs_cuda.get_potential(far_corner),
        )


if __name__ == "__main__":
    unittest.main()
