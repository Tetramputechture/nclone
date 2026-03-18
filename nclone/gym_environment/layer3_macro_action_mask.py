"""Layer 3 FiGAR macro-action safety masking helpers.

Computes a boolean safety mask over (action, duration) candidates by forward
simulating the current ninja state against currently deadly mine positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from nclone.constants.physics_constants import NINJA_RADIUS, TOGGLE_MINE_RADIUS_TOGGLED
from nclone.mine_physics_simulator import MinePhysicsSimulator


_NUM_ACTIONS = 6
_DEFAULT_KILL_RADIUS = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED


@dataclass(frozen=True)
class Layer3MaskResult:
    """Layer 3 output for a single decision point."""

    action_mask: np.ndarray
    duration_mask: np.ndarray
    masked_actions: int
    masked_candidates: int
    mask_overrides: int
    all_masked_event: bool


@dataclass(frozen=True)
class _NativeNinjaState:
    """Minimal runtime ninja state needed for FiGAR forward simulation."""

    x: float
    y: float
    vx: float
    vy: float
    jump_duration: int


class Layer3MacroActionMasker:
    """Build mine-only FiGAR macro-action masks from the current env state."""

    def __init__(
        self,
        duration_bins: Sequence[int],
        fallback_mode: str = "noop",
        kill_radius: float = _DEFAULT_KILL_RADIUS,
    ) -> None:
        self.duration_bins = tuple(int(duration) for duration in duration_bins)
        self.fallback_mode = fallback_mode
        self.kill_radius = float(kill_radius)
        self._native_sim = None
        self._native_map_data = None

    def compute(
        self,
        headless,
        base_action_mask: np.ndarray,
        actions_to_execute: Callable[[int], tuple[int, int]],
    ) -> Layer3MaskResult:
        """Compute Layer 3 masks for the current state."""
        base_mask = np.asarray(base_action_mask, dtype=np.bool_).copy()
        duration_mask = np.zeros(
            (_NUM_ACTIONS, len(self.duration_bins)),
            dtype=np.bool_,
        )
        deadly_mines = self._get_deadly_mine_positions(headless)

        if not deadly_mines:
            duration_mask[:] = base_mask[:, None]
            return self._finalize(base_mask, duration_mask, base_mask)

        native_state = self._get_native_state(headless)
        if native_state is not None:
            duration_mask = self._compute_with_native_clone(
                headless=headless,
                deadly_mines=deadly_mines,
                base_action_mask=base_mask,
                current_state=native_state,
                actions_to_execute=actions_to_execute,
            )
        else:
            duration_mask = self._compute_with_python_simulator(
                headless=headless,
                deadly_mines=deadly_mines,
                base_action_mask=base_mask,
            )

        return self._finalize(base_mask, duration_mask, base_mask)

    def _finalize(
        self,
        base_mask: np.ndarray,
        duration_mask: np.ndarray,
        original_action_mask: np.ndarray,
    ) -> Layer3MaskResult:
        duration_mask = np.asarray(duration_mask, dtype=np.bool_)
        action_mask = np.logical_and(base_mask, duration_mask.any(axis=1))
        all_masked = not action_mask.any()

        if all_masked and self.fallback_mode == "noop":
            action_mask = action_mask.copy()
            duration_mask = duration_mask.copy()
            action_mask[0] = True
            duration_mask[0, 0] = True

        masked_actions = int((~action_mask).sum())
        masked_candidates = int((~duration_mask).sum())
        mask_overrides = int(np.logical_and(original_action_mask, ~action_mask).sum())
        return Layer3MaskResult(
            action_mask=action_mask.astype(np.int8),
            duration_mask=duration_mask.astype(np.int8),
            masked_actions=masked_actions,
            masked_candidates=masked_candidates,
            mask_overrides=mask_overrides,
            all_masked_event=all_masked,
        )

    def _get_deadly_mine_positions(self, headless) -> list[tuple[float, float]]:
        """Return mine centers that are currently deadly in the live env."""
        toggle_mines, toggled_mines = headless.get_mine_entities()
        deadly_mines: list[tuple[float, float]] = []
        for mine in (*toggle_mines, *toggled_mines):
            if getattr(mine, "state", 1) == 0:
                deadly_mines.append((float(mine.xpos), float(mine.ypos)))
        return deadly_mines

    def _get_native_state(self, headless) -> Optional[_NativeNinjaState]:
        """Extract the current runtime state for native forward simulation."""
        native_sim = getattr(headless, "_native_sim", None)
        if native_sim is None or not hasattr(native_sim, "get_ninja_jump_duration"):
            return None

        x, y = headless.ninja_position()
        vx, vy = headless.ninja_velocity()
        return _NativeNinjaState(
            x=float(x),
            y=float(y),
            vx=float(vx),
            vy=float(vy),
            jump_duration=int(native_sim.get_ninja_jump_duration()),
        )

    def _compute_with_native_clone(
        self,
        headless,
        deadly_mines: list[tuple[float, float]],
        base_action_mask: np.ndarray,
        current_state: _NativeNinjaState,
        actions_to_execute: Callable[[int], tuple[int, int]],
    ) -> np.ndarray:
        """Use a dedicated native simulator clone to score macro-actions."""
        sim = self._ensure_native_clone(headless)
        duration_mask = np.zeros((_NUM_ACTIONS, len(self.duration_bins)), dtype=np.bool_)

        for action in range(_NUM_ACTIONS):
            if not base_action_mask[action]:
                continue

            hor_input, jump_input = actions_to_execute(action)
            for duration_idx, duration in enumerate(self.duration_bins):
                sim.fast_reset()
                sim.inject_ninja_for_value_iteration(
                    current_state.x,
                    current_state.y,
                    current_state.vx,
                    current_state.vy,
                    current_state.jump_duration,
                )

                safe = True
                for _ in range(duration):
                    sim.tick_physics_only(hor_input, jump_input)
                    x, y = sim.get_ninja_pos()
                    if self._hits_deadly_mine(x, y, deadly_mines):
                        safe = False
                        break

                duration_mask[action, duration_idx] = safe

        return duration_mask

    def _ensure_native_clone(self, headless):
        """Return a reusable native simulator clone for Layer 3 rollouts."""
        current_map_data = getattr(headless, "current_map_data", None)
        if current_map_data is None:
            raise RuntimeError("Layer 3 native masking requires current_map_data")

        if self._native_sim is not None and self._native_map_data == current_map_data:
            return self._native_sim

        import nclone_native

        map_data = [int(value) for value in current_map_data]
        self._native_sim = nclone_native.Simulator.load(map_data)
        self._native_map_data = list(map_data)
        return self._native_sim

    def _compute_with_python_simulator(
        self,
        headless,
        deadly_mines: list[tuple[float, float]],
        base_action_mask: np.ndarray,
    ) -> np.ndarray:
        """Fallback path for the pure-Python simulator."""
        duration_mask = np.zeros((_NUM_ACTIONS, len(self.duration_bins)), dtype=np.bool_)
        simulator = MinePhysicsSimulator(headless.sim)
        initial_state = headless.sim.ninja.save_state()

        for action in range(_NUM_ACTIONS):
            if not base_action_mask[action]:
                continue

            for duration_idx, duration in enumerate(self.duration_bins):
                collision = simulator.simulate_action_sequence(
                    initial_state=initial_state,
                    action=action,
                    mine_positions=deadly_mines,
                    num_frames=duration,
                )
                duration_mask[action, duration_idx] = not collision

        return duration_mask

    def _hits_deadly_mine(
        self,
        x: float,
        y: float,
        deadly_mines: Sequence[tuple[float, float]],
    ) -> bool:
        """Return True when the position is inside any currently deadly mine."""
        for mine_x, mine_y in deadly_mines:
            if math.hypot(x - mine_x, y - mine_y) < self.kill_radius:
                return True
        return False
