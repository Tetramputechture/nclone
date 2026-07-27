"""Layer 3 FiGAR macro-action safety masking helpers.

Computes a boolean safety mask over (action, duration) candidates by forward
simulating the current ninja state and rejecting the ones that end in death.

v26 W6b — the fatality predicate used to be **mine-only**. That was the single
largest gap in the safety stack: mine deaths are ~60% of v25's losses and impact
deaths ~40%, yet every *hard* guarantee the agent had was about mines. A macro
that committed to a fatal fall was never masked, and ``--takeoff-arc-gap-mask``
(the other commitment lever) was keyed on mines too. The forward simulator was
already running the real physics from the real ninja state — only the question it
asked at each frame was wrong.

Two knobs generalize it:

* ``include_lethal_impact`` — also reject a candidate whose simulated execution
  ends in ANY physics death, which ``tick_physics_only`` already reports through
  ``has_died()`` (terminal impact on floors and ceilings alike).
* ``lethal_impact_lookahead`` — keep simulating past the macro's own duration
  with neutral input. This is what catches *commitment*: a 12-frame macro that
  merely steps off a ledge is survivable for all 12 of its frames and fatal 30
  frames later, and the macro-length horizon alone can never see that.

The native path also simulates each action ONCE to the longest duration bin
rather than re-simulating per bin. Both predicates are prefix-monotone in
duration (a candidate that dies at frame f is fatal for every bin >= f and safe
for every bin < f), so this is exactly equivalent to the per-bin loop and ~15x
cheaper — which is what makes the extra lookahead affordable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from nclone.constants.physics_constants import NINJA_RADIUS, TOGGLE_MINE_RADIUS_TOGGLED
from nclone.mine_physics_simulator import MinePhysicsSimulator


logger = logging.getLogger(__name__)

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
    # v26 W6b: how many candidates each fatality cause vetoed, so the two can be
    # told apart in telemetry. Without this split an impact-mask regression
    # (agent refuses to jump -> timeouts) is indistinguishable from the mine
    # mask simply firing more.
    masked_by_mine: int = 0
    masked_by_impact: int = 0


@dataclass(frozen=True)
class _NativeNinjaState:
    """Minimal runtime ninja state needed for FiGAR forward simulation."""

    x: float
    y: float
    vx: float
    vy: float
    jump_duration: int


class Layer3MacroActionMasker:
    """Build FiGAR macro-action safety masks from the current env state.

    Mine-only by default (the pre-v26 contract). ``include_lethal_impact`` adds
    unsurvivable impacts to the fatality predicate — see the module docstring for
    why that was the stack's largest gap.
    """

    def __init__(
        self,
        duration_bins: Sequence[int],
        fallback_mode: str = "noop",
        kill_radius: float = _DEFAULT_KILL_RADIUS,
        include_lethal_impact: bool = False,
        lethal_impact_lookahead: int = 0,
    ) -> None:
        self.duration_bins = tuple(int(duration) for duration in duration_bins)
        self.fallback_mode = fallback_mode
        self.kill_radius = float(kill_radius)
        self.include_lethal_impact = bool(include_lethal_impact)
        self.lethal_impact_lookahead = max(0, int(lethal_impact_lookahead))
        self._max_duration = max(self.duration_bins) if self.duration_bins else 0
        self._native_sim = None
        self._native_map_data = None
        self._warned_python_impact = False

    def compute(
        self,
        headless,
        base_action_mask: np.ndarray,
        actions_to_execute: Callable[[int], tuple[int, int]],
        fall_risk_nearby: bool = True,
    ) -> Layer3MaskResult:
        """Compute Layer 3 masks for the current state.

        Args:
            fall_risk_nearby: cheap caller-supplied gate. Impact masking removes
                the "no mines -> nothing to check" early-out, which would make
                every decision point on a mine-free level pay a full forward
                simulation. The caller (which holds the cached fall-lethality
                field) can answer "is there a fatal drop anywhere the ninja could
                reach this macro" for free; ``False`` restores the early-out.
                Defaults to ``True`` so an uninformed caller stays conservative.
        """
        base_mask = np.asarray(base_action_mask, dtype=np.bool_).copy()
        duration_mask = np.zeros(
            (_NUM_ACTIONS, len(self.duration_bins)),
            dtype=np.bool_,
        )
        deadly_mines = self._get_deadly_mine_positions(headless)
        check_impact = self.include_lethal_impact and fall_risk_nearby

        if not deadly_mines and not check_impact:
            duration_mask[:] = base_mask[:, None]
            return self._finalize(base_mask, duration_mask, base_mask)

        native_state = self._get_native_state(headless)
        if native_state is not None:
            duration_mask, by_mine, by_impact = self._compute_with_native_clone(
                headless=headless,
                deadly_mines=deadly_mines,
                base_action_mask=base_mask,
                current_state=native_state,
                actions_to_execute=actions_to_execute,
                check_impact=check_impact,
            )
        else:
            if check_impact and not self._warned_python_impact:
                self._warned_python_impact = True
                logger.warning(
                    "Layer 3 lethal-impact masking needs the native simulator "
                    "(has_died() after tick_physics_only); the pure-Python "
                    "fallback stays mine-only for this env."
                )
            duration_mask = self._compute_with_python_simulator(
                headless=headless,
                deadly_mines=deadly_mines,
                base_action_mask=base_mask,
            )
            by_mine = int((~duration_mask).sum())
            by_impact = 0

        return self._finalize(
            base_mask, duration_mask, base_mask,
            masked_by_mine=by_mine, masked_by_impact=by_impact,
        )

    def _finalize(
        self,
        base_mask: np.ndarray,
        duration_mask: np.ndarray,
        original_action_mask: np.ndarray,
        masked_by_mine: int = 0,
        masked_by_impact: int = 0,
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
            masked_by_mine=int(masked_by_mine),
            masked_by_impact=int(masked_by_impact),
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
        check_impact: bool = False,
    ) -> tuple[np.ndarray, int, int]:
        """Use a dedicated native simulator clone to score macro-actions.

        One rollout per ACTION, not per (action, duration): both fatality
        predicates are prefix-monotone in duration, so the first fatal frame
        determines the whole duration column. Exactly equivalent to the old
        per-bin loop, ~15x cheaper.

        Returns ``(duration_mask, masked_by_mine, masked_by_impact)``.
        """
        sim = self._ensure_native_clone(headless)
        duration_mask = np.zeros((_NUM_ACTIONS, len(self.duration_bins)), dtype=np.bool_)
        bins = np.asarray(self.duration_bins, dtype=np.int64)
        masked_by_mine = 0
        masked_by_impact = 0

        # Past the macro's own frames, the commitment question is "does this
        # still kill me if I stop steering", so the lookahead runs neutral input.
        lookahead = self.lethal_impact_lookahead if check_impact else 0
        horizon = self._max_duration + lookahead

        for action in range(_NUM_ACTIONS):
            if not base_action_mask[action]:
                continue

            hor_input, jump_input = actions_to_execute(action)
            sim.fast_reset()
            sim.inject_ninja_for_value_iteration(
                current_state.x,
                current_state.y,
                current_state.vx,
                current_state.vy,
                current_state.jump_duration,
            )

            # 1-indexed frame at which this action first becomes fatal; the macro
            # is safe for every bin strictly shorter than it.
            fatal_frame = None
            fatal_is_impact = False
            for frame in range(1, horizon + 1):
                if frame <= self._max_duration:
                    sim.tick_physics_only(hor_input, jump_input)
                else:
                    sim.tick_physics_only(0, 0)
                if check_impact and sim.has_died():
                    fatal_frame = frame
                    fatal_is_impact = True
                    break
                x, y = sim.get_ninja_pos()
                if deadly_mines and self._hits_deadly_mine(x, y, deadly_mines):
                    fatal_frame = frame
                    break

            if fatal_frame is None:
                duration_mask[action, :] = True
                continue

            # A death in the LOOKAHEAD window is not attributable to any one
            # duration — every candidate committed to the same doomed trajectory,
            # so the whole action is vetoed. A death inside the macro itself only
            # condemns the bins long enough to reach it.
            if fatal_frame > self._max_duration:
                safe = np.zeros(len(self.duration_bins), dtype=np.bool_)
            else:
                safe = bins < fatal_frame
            duration_mask[action, :] = safe
            n_masked = int((~safe).sum())
            if fatal_is_impact:
                masked_by_impact += n_masked
            else:
                masked_by_mine += n_masked

        return duration_mask, masked_by_mine, masked_by_impact

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
