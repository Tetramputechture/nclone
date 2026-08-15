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

The native path shares the HELD-INPUT PREFIX across duration bins: every bin
holds the same input over its own duration, so the trunk is simulated once and
each bin branches off it (save/restore) to run its own neutral lookahead. Cost is
``max_duration + n_bins * lookahead`` instead of
``n_bins * (max_duration + lookahead)``.

It used to share more than that -- one rollout per action, input held to the
LONGEST bin and then neutral, with that single post-maximum window reused for
every bin. Prefix monotonicity justifies sharing only while the same input is
still held; the continuation after RELEASE differs per bin. So a death in the
lookahead window masked every bin, including short ones that would have released
early and survived, and a short bin's own lookahead was never simulated, so it
could be certified safe while being fatal. Both directions are now evaluated on
the candidate's real continuation.

What this is and is not
-----------------------
This is a PREDICTIVE HAZARD MASK, not an exact model-predictive shield, and it
does not inherit the guarantees of Alshiekh et al. shielding:

* the native clone is reset from map data and injected with ninja position,
  velocity and jump duration only -- not a full live simulator snapshot, so
  entity state in the clone is the level's initial state, not the current one;
* ``tick_physics_only`` does not advance entity AI, and the set of currently
  deadly mines is snapshotted once per decision point, so a mine that arms or
  disarms mid-macro is not modelled;
* the lookahead is finite, so it certifies "does not die within the horizon",
  never "is recoverable";
* when every candidate is masked the fallback enables a one-frame NOOP, which is
  a least-bad default and is NOT proven safe. ``Layer3MaskResult.fallback_fatal``
  reports whether that NOOP was itself evaluated as fatal.
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
    # True when the all-masked NOOP fallback fired AND that NOOP candidate was
    # itself evaluated as fatal, i.e. the mask is knowingly emitting a lethal
    # action because it has nothing better. Distinguishes "no safe option was
    # found" from "we picked a safe default", which the all_masked_event flag
    # alone cannot.
    fallback_fatal: bool = False


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
        self._warned_restore_failed = False

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

        fallback_fatal = False
        if all_masked and self.fallback_mode == "noop":
            # The NOOP candidate we are about to force may itself have been
            # evaluated and rejected; record that rather than presenting the
            # fallback as a safe choice.
            fallback_fatal = bool(base_mask[0]) and not bool(duration_mask[0, 0])
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
            fallback_fatal=fallback_fatal,
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

    def _fatal_now(
        self,
        sim,
        deadly_mines: Sequence[tuple[float, float]],
        check_impact: bool,
    ) -> tuple[bool, bool]:
        """``(is_fatal, is_impact)`` for the simulator's CURRENT frame."""
        if check_impact and sim.has_died():
            return True, True
        x, y = sim.get_ninja_pos()
        if deadly_mines and self._hits_deadly_mine(x, y, deadly_mines):
            return True, False
        return False, False

    def _neutral_lookahead_fatal(
        self,
        sim,
        deadly_mines: Sequence[tuple[float, float]],
        check_impact: bool,
        frames: int,
    ) -> tuple[bool, bool, bool]:
        """Release the input HERE and coast ``frames`` frames on neutral input.

        Runs from the simulator's current state and restores it afterwards, so the
        caller's held-input trunk can carry on undisturbed.

        Returns ``(is_fatal, is_impact, trunk_intact)``. ``trunk_intact`` is False
        if the state could not be restored, which invalidates the shared trunk.
        """
        snapshot = sim.save_full_state()
        fatal = False
        is_impact = False
        for _ in range(frames):
            sim.tick_physics_only(0, 0)
            fatal, is_impact = self._fatal_now(sim, deadly_mines, check_impact)
            if fatal:
                break
        trunk_intact = bool(sim.restore_full_state(snapshot))
        return fatal, is_impact, trunk_intact

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

        Each (action, duration) candidate is evaluated on ITS OWN continuation:
        hold the action's input for exactly ``duration`` frames, then coast on
        neutral input for the lookahead window. That is what the candidate will
        actually do, and it is not shared across durations -- releasing the input
        earlier changes the landing, so a shorter bin can survive a trajectory
        that a longer one dies on, and vice versa.

        The frames BEFORE the release are shared: every bin holds the same input
        over its prefix, so the held-input trunk is simulated once and each bin
        branches off it via save/restore. Cost is
        ``max_duration + n_bins * lookahead`` rather than
        ``n_bins * (max_duration + lookahead)``, and unlike the previous
        single-rollout scheme it is exact.

        The previous version simulated ONE rollout per action -- input held to the
        longest bin, then neutral -- and reused that single post-maximum window for
        every bin. Prefix monotonicity does justify sharing while the same input is
        still held, but not the continuation after release, so a death in the
        lookahead window masked EVERY bin (including short ones that would have
        released early and lived) and a short bin's own lookahead was never
        simulated at all (so it could be certified safe while being fatal).

        Returns ``(duration_mask, masked_by_mine, masked_by_impact)``.
        """
        sim = self._ensure_native_clone(headless)
        n_bins = len(self.duration_bins)
        duration_mask = np.zeros((_NUM_ACTIONS, n_bins), dtype=np.bool_)
        bins = np.asarray(self.duration_bins, dtype=np.int64)
        # Branch off the trunk in ascending duration order so each bin's release
        # point is reached by advancing the trunk, never by rewinding it.
        order = np.argsort(bins, kind="stable")
        masked_by_mine = 0
        masked_by_impact = 0

        # Past the macro's own frames, the commitment question is "does this
        # still kill me if I stop steering", so the lookahead runs neutral input.
        lookahead = self.lethal_impact_lookahead if check_impact else 0

        def _inject() -> None:
            sim.fast_reset()
            sim.inject_ninja_for_value_iteration(
                current_state.x,
                current_state.y,
                current_state.vx,
                current_state.vy,
                current_state.jump_duration,
            )

        for action in range(_NUM_ACTIONS):
            if not base_action_mask[action]:
                continue

            hor_input, jump_input = actions_to_execute(action)
            _inject()

            safe = np.zeros(n_bins, dtype=np.bool_)
            by_impact = np.zeros(n_bins, dtype=np.bool_)

            trunk_frame = 0          # frames of held input simulated so far
            trunk_fatal = False      # the held-input prefix already killed us
            trunk_fatal_is_impact = False

            for idx in order:
                duration = int(bins[idx])

                # Advance the shared held-input trunk to this bin's release point.
                while not trunk_fatal and trunk_frame < duration:
                    sim.tick_physics_only(hor_input, jump_input)
                    trunk_frame += 1
                    trunk_fatal, trunk_fatal_is_impact = self._fatal_now(
                        sim, deadly_mines, check_impact
                    )

                if trunk_fatal:
                    # Died at or before this bin's release, so every bin from here
                    # up (the order is ascending) is fatal for the same reason.
                    by_impact[idx] = trunk_fatal_is_impact
                    continue

                if lookahead <= 0:
                    safe[idx] = True
                    continue

                fatal, is_impact, trunk_intact = self._neutral_lookahead_fatal(
                    sim, deadly_mines, check_impact, lookahead
                )
                safe[idx] = not fatal
                by_impact[idx] = is_impact
                if not trunk_intact:
                    # The branch could not be rolled back, so the trunk no longer
                    # represents "input held for `trunk_frame` frames". Rebuild it
                    # from the injected state rather than scoring the rest of the
                    # bins against a corrupted simulator.
                    if not self._warned_restore_failed:
                        self._warned_restore_failed = True
                        logger.warning(
                            "Layer 3 lookahead could not restore the native clone; "
                            "rebuilding the held-input trunk per duration bin. "
                            "Masks stay exact, the fast path is lost."
                        )
                    _inject()
                    trunk_frame = 0
                    for _ in range(duration):
                        sim.tick_physics_only(hor_input, jump_input)
                        trunk_frame += 1
                        trunk_fatal, trunk_fatal_is_impact = self._fatal_now(
                            sim, deadly_mines, check_impact
                        )
                        if trunk_fatal:
                            break

            duration_mask[action, :] = safe
            n_impact = int((~safe & by_impact).sum())
            masked_by_impact += n_impact
            masked_by_mine += int((~safe).sum()) - n_impact

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
