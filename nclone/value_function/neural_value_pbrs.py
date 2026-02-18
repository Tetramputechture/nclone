"""Neural value function PBRS adapter using IQL-trained V(s).

Drop-in replacement for SurfaceGraphPBRS/TwoPhaseValuePBRS. Uses a frozen
V-network from IQL training as the potential function Phi(s) for PBRS.

The CNN output (level geometry + goal overlays) is static per-level and cached
after set_level(). Per-step inference only runs the MLP (35D input) + V head.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..constants.physics_constants import (
    LEVEL_HEIGHT_PX,
    LEVEL_WIDTH_PX,
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
)

logger = logging.getLogger(__name__)

NUM_NEAREST_MINES = 8
AGENT_STATE_DIM = 11
MINE_FEATURE_DIM = NUM_NEAREST_MINES * 3
TILE_GRID_W = 42
TILE_GRID_H = 23
TILE_SIZE_PX = 24


class NeuralValuePBRS:
    """PBRS potential from IQL-trained V(s) network.

    Implements the same get_potential() interface as SurfaceGraphPBRS,
    making it a drop-in replacement in the PBRS pipeline.

    Architecture:
        - CNN tile encoder (static per-level, cached at set_level)
        - Agent state MLP: 11D ninja physics + 24D nearest-8-mines = 35D
        - V-head: combined representation -> scalar value

    The CNN output is computed once per level and cached, so per-step
    cost is only the MLP forward pass (~35D -> 512D -> 1D).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        """Load frozen V-network from exported IQL checkpoint.

        Args:
            checkpoint_path: Path to exported V-network file (.pt).
            device: Torch device for inference.
        """
        self.device = torch.device(device)

        data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = data["config"]

        # Reconstruct V-network architecture
        from ..offline_rl.networks.value_networks import IQLNetworks

        self._networks = IQLNetworks(
            hidden_dim=config["hidden_dim"],
            use_tile_encoder=config.get("use_tile_encoder", True),
        ).to(self.device)

        # Load V-network component weights
        v_state = data["v_network"]
        self._networks.agent_encoder.load_state_dict(v_state["agent_encoder"])
        self._networks.combiner.load_state_dict(v_state["combiner"])
        self._networks.v_head.load_state_dict(v_state["v_head"])
        if "tile_encoder" in v_state and self._networks.tile_encoder is not None:
            self._networks.tile_encoder.load_state_dict(v_state["tile_encoder"])

        self._networks.eval()

        # Cached level encoding (computed once per level in set_level)
        self._cached_level_repr: Optional[torch.Tensor] = None
        self._cached_tile_grid: Optional[torch.Tensor] = None
        self._cached_switch_overlay: Optional[torch.Tensor] = None
        self._cached_exit_overlay: Optional[torch.Tensor] = None

        # Mine positions cached at set_level time
        self._mine_positions: List[Tuple[float, float]] = []
        self._mine_initial_types: List[int] = []  # entity_type: 1 or 21

        # Reference to headless simulator for live state queries
        self._headless = None

        logger.info(
            "Loaded NeuralValuePBRS from %s (device=%s, tile_encoder=%s, step=%d)",
            checkpoint_path,
            device,
            config.get("use_tile_encoder", True),
            data.get("training_step", 0),
        )

    def set_level(
        self,
        tile_grid: np.ndarray,
        switch_pos: Tuple[float, float],
        exit_pos: Tuple[float, float],
        mine_positions: List[Tuple[float, float, int]],
        headless: Any = None,
    ):
        """Cache static level data and precompute CNN encoding.

        Called once when a new level is loaded or changed.

        Args:
            tile_grid: (42, 23) array of tile type IDs (0-37).
            switch_pos: (x, y) pixel position of exit switch.
            exit_pos: (x, y) pixel position of exit door.
            mine_positions: List of (x, y, entity_type) for all toggle mines.
            headless: Optional reference to NativePlayHeadless for live state queries.
        """
        self._headless = headless

        # Cache mine positions
        self._mine_positions = [(mx, my) for mx, my, _ in mine_positions]
        self._mine_initial_types = [mt for _, _, mt in mine_positions]

        # Build switch/exit overlays
        switch_overlay = np.zeros((TILE_GRID_W, TILE_GRID_H), dtype=np.float32)
        sx = max(0, min(int(switch_pos[0]) // TILE_SIZE_PX, TILE_GRID_W - 1))
        sy = max(0, min(int(switch_pos[1]) // TILE_SIZE_PX, TILE_GRID_H - 1))
        switch_overlay[sx, sy] = 1.0

        exit_overlay = np.zeros((TILE_GRID_W, TILE_GRID_H), dtype=np.float32)
        ex = max(0, min(int(exit_pos[0]) // TILE_SIZE_PX, TILE_GRID_W - 1))
        ey = max(0, min(int(exit_pos[1]) // TILE_SIZE_PX, TILE_GRID_H - 1))
        exit_overlay[ex, ey] = 1.0

        # Convert to tensors (batch dim = 1)
        self._cached_tile_grid = (
            torch.from_numpy(tile_grid.astype(np.int64)).unsqueeze(0).to(self.device)
        )
        self._cached_switch_overlay = (
            torch.from_numpy(switch_overlay).unsqueeze(0).to(self.device)
        )
        self._cached_exit_overlay = (
            torch.from_numpy(exit_overlay).unsqueeze(0).to(self.device)
        )

        # Precompute CNN encoding (static per-level)
        if self._networks.tile_encoder is not None:
            with torch.no_grad():
                self._cached_level_repr = self._networks.tile_encoder(
                    self._cached_tile_grid,
                    self._cached_switch_overlay,
                    self._cached_exit_overlay,
                )  # (1, 128)
        else:
            self._cached_level_repr = None

        logger.info(
            "NeuralValuePBRS set_level: %d mines, switch=(%d,%d), exit=(%d,%d)",
            len(mine_positions),
            int(switch_pos[0]),
            int(switch_pos[1]),
            int(exit_pos[0]),
            int(exit_pos[1]),
        )

    def get_potential(self, state: Dict[str, Any]) -> float:
        """Compute PBRS potential Phi(s) from the IQL V-network.

        Compatible with the SurfaceGraphPBRS/TwoPhaseValuePBRS interface.

        Args:
            state: Game state dictionary with keys:
                - player_x, player_y: Position in pixels.
                - player_xspeed, player_yspeed: Velocity in px/frame.
                - switch_activated: bool (phase indicator).
                - jump_duration_remaining: float (normalized).
                Optional (from headless or caller):
                - on_ground: bool
                - wall_sliding: bool
                - floor_normal_x, floor_normal_y: float
                - wall_normal: float
                - entity_states: dict of mine entity states

        Returns:
            Potential value (negative time-to-go, typically in [-1, 0]).
            Returns 0.0 for terminal states.
        """
        # Extract 11D ninja state
        ninja_state = self._extract_ninja_state(state)

        # Compute nearest-8-mine features
        mine_features = self._compute_mine_features(
            state.get("player_x", 0.0),
            state.get("player_y", 0.0),
            state,
        )

        # Build batch dict for network
        state_tensor = torch.from_numpy(ninja_state).unsqueeze(0).to(self.device)
        mine_tensor = torch.from_numpy(mine_features).unsqueeze(0).to(self.device)

        batch = {
            "state": state_tensor,
            "mine_features": mine_tensor,
        }

        # Add level encoding
        if self._networks.use_tile_encoder:
            if self._cached_level_repr is not None:
                # Use cached CNN output directly (skip CNN forward pass)
                batch["tile_grid"] = self._cached_tile_grid
                batch["switch_overlay"] = self._cached_switch_overlay
                batch["exit_overlay"] = self._cached_exit_overlay

        # Forward pass (only MLP + V-head when level repr is cached)
        with torch.no_grad():
            v_value = self._networks.value(batch)

        return float(v_value.item())

    def _extract_ninja_state(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract 11D normalized ninja state from game state dict.

        Falls back to headless queries for physics fields not in the state dict.
        """
        s = np.zeros(AGENT_STATE_DIM, dtype=np.float32)

        px = state.get("player_x", 0.0)
        py = state.get("player_y", 0.0)
        vx = state.get("player_xspeed", 0.0) or 0.0
        vy = state.get("player_yspeed", 0.0) or 0.0

        s[0] = px / LEVEL_WIDTH_PX
        s[1] = py / LEVEL_HEIGHT_PX
        s[2] = vx / MAX_HOR_SPEED
        s[3] = vy / MAX_HOR_SPEED
        s[4] = 1.0 if state.get("on_ground", False) else 0.0
        s[5] = 1.0 if state.get("wall_sliding", False) else 0.0
        s[6] = state.get("jump_duration_remaining", 0.0)
        s[7] = 1.0 if state.get("switch_activated", False) else 0.0
        s[8] = state.get("floor_normal_x", 0.0)
        s[9] = state.get("floor_normal_y", 0.0)
        s[10] = state.get("wall_normal", 0.0)

        # Try to fill missing fields from headless if available
        if self._headless is not None:
            try:
                sim_state = self._headless.sim.get_state()
                ninja = sim_state.ninja

                if "on_ground" not in state:
                    s[4] = 1.0 if not ninja.airborn else 0.0
                if "wall_sliding" not in state:
                    s[5] = 1.0 if ninja.state == 5 else 0.0
                if "jump_duration_remaining" not in state:
                    s[6] = ninja.jump_duration / MAX_JUMP_DURATION if ninja.state == 3 else 0.0
                if "floor_normal_x" not in state:
                    s[8] = ninja.floor_normalized_x if not ninja.airborn else 0.0
                if "floor_normal_y" not in state:
                    s[9] = ninja.floor_normalized_y if not ninja.airborn else 0.0
                if "wall_normal" not in state:
                    s[10] = ninja.wall_normal if ninja.walled else 0.0
            except Exception:
                pass  # Use defaults if headless query fails

        return s

    def _compute_mine_features(
        self,
        ninja_x: float,
        ninja_y: float,
        state: Dict[str, Any],
    ) -> np.ndarray:
        """Compute nearest-8-mine features with current entity states.

        Mine positions are cached from set_level(). Mine states are read
        from the headless simulator or entity_states in the state dict
        to capture dynamic toggling during episodes.
        """
        if not self._mine_positions:
            return np.zeros(MINE_FEATURE_DIM, dtype=np.float32)

        # Get current mine states
        mine_states = self._get_current_mine_states(state)

        # Compute relative positions and sort by distance
        mine_state_map = {0: -1.0, 1: 1.0, 2: 0.5}
        mines = []
        for i, (mx, my) in enumerate(self._mine_positions):
            dx = mx - ninja_x
            dy = my - ninja_y
            dist_sq = dx * dx + dy * dy
            raw_state = mine_states[i] if i < len(mine_states) else 1
            state_val = mine_state_map.get(raw_state, 0.0)
            mines.append((dist_sq, dx, dy, state_val))

        mines.sort(key=lambda m: m[0])

        features = np.zeros(MINE_FEATURE_DIM, dtype=np.float32)
        for i, (_, dx, dy, sv) in enumerate(mines[:NUM_NEAREST_MINES]):
            base = i * 3
            features[base] = dx / LEVEL_WIDTH_PX
            features[base + 1] = dy / LEVEL_HEIGHT_PX
            features[base + 2] = sv

        return features

    def _get_current_mine_states(self, state: Dict[str, Any]) -> List[int]:
        """Get current toggle mine states from headless or state dict.

        Returns list of state codes (0=toggled/deadly, 1=untoggled/safe, 2=transitioning).
        """
        # Try explicit entity_states from state dict
        entity_states = state.get("entity_states", None)
        if entity_states is not None:
            return entity_states

        # Try headless simulator
        if self._headless is not None:
            try:
                sim_state = self._headless.sim.get_state()
                states = []
                for ent in sim_state.entities:
                    if ent.entity_type in (1, 21):
                        states.append(ent.state)
                return states
            except Exception:
                pass

        # Fallback: use initial types as proxy
        # type 1 = untoggled (safe=1), type 21 = toggled (deadly=0)
        return [1 if t == 1 else 0 for t in self._mine_initial_types]

    def get_gradient(
        self, state: Dict[str, Any], epsilon: float = 6.0,
    ) -> Tuple[float, float]:
        """Estimate potential gradient via finite differences."""
        phi0 = self.get_potential(state)

        sp_x = dict(state)
        sp_x["player_x"] = state.get("player_x", 0.0) + epsilon
        phi_x = self.get_potential(sp_x)

        sp_y = dict(state)
        sp_y["player_y"] = state.get("player_y", 0.0) + epsilon
        phi_y = self.get_potential(sp_y)

        return (
            (phi_x - phi0) / epsilon,
            (phi_y - phi0) / epsilon,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics for logging."""
        total_params = sum(p.numel() for p in self._networks.parameters())
        return {
            "type": "neural_iql",
            "total_params": total_params,
            "device": str(self.device),
            "n_mines_cached": len(self._mine_positions),
            "level_repr_cached": self._cached_level_repr is not None,
        }
