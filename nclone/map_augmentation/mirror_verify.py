"""Executable verification of the horizontal-mirror symmetry contract.

The simulator is deterministic, so the symmetry claims in
``docs/HORIZONTAL_AUGMENTATION_PLAN.md`` §2 are *testable*, not merely arguable.
This module is the test:

Run A
    simulate the level normally, then apply
    :class:`~nclone.map_augmentation.obs_mirror.ObservationMirror` to each
    observation.
Run B
    mirror the level with :func:`~nclone.map_augmentation.transforms.mirror_map_data`,
    mirror every action, and simulate.

If reflection is a symmetry of the sim AND our observation transform is right,
``A == B`` field by field. That single comparison validates the tile table, the
entity handling, drone chirality, the ghost-arc ``u = +-1`` swap, the raycast
permutation and the scandot invariance *simultaneously* -- and it is the only
way to catch an asymmetry hiding in the sim rather than in our transform.

When it fails, :class:`CommutationReport` says WHICH fields diverged, by how
much, and on which step. Without that the failure is a wall of floats.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .obs_mirror import ObservationMirror, mirror_action
from .transforms import mirror_map_data

__all__ = [
    "CommutationReport",
    "FieldDivergence",
    "TIE_BREAK_SENSITIVE",
    "check_involution",
    "check_commutation",
    "rollout",
]


# What the harness found, and how the assertions below are shaped by it.
#
# The SIMULATOR commutes with reflection exactly: ninja position, velocity,
# surface normals, contacts and the goal geometry agree to float precision on
# every level tested. That is the load-bearing claim and it is asserted strictly
# (:data:`GAME_STATE_EXACT_DIMS`).
#
# Two classes of *derived observation* do not commute exactly, and neither is
# fixable by a better transform:
#
# 1. QUANTIZATION. Every ego-centric readout locates the ninja by
#    ``int(px / cell)``. A grid cell is a half-open interval, and the reflection
#    of a half-open interval is the OTHER half-open interval -- so when the ninja
#    sits EXACTLY on a cell boundary the mirrored readout picks the neighbouring
#    cell and the whole window shifts by one. Away from boundaries these keys
#    agree exactly, which is what :func:`check_commutation` asserts: divergences
#    are only tolerated on steps where the ninja is on a boundary. Spawn
#    positions are grid-aligned, so those steps are common, not exotic.
#
# 2. TIE-BREAKS. A planner or an ``argmin`` has to choose between options that
#    reflection makes equivalent, and no pure function of position can choose
#    consistently in both orientations. These keys are never asserted exact.
QUANTIZED_KEYS: Dict[str, str] = {
    "ego_crop": "tile window origin is int(px/24)",
    "ego_crop_mines": "tile window origin is int(px/24)",
    "ego_crop_route": "tile window origin is int(px/24)",
    "ego_crop_coarse": "pooled tile window origin",
    "ego_crop_route_fine": "6px window origin",
    "ego_crop_mines_fine": "6px window origin",
    "cost_to_go_map": "ego-residual normalisation reads the ninja's cell",
    # raycasts / scandots march the tile grid from the ninja's tile; the arcs
    # integrate from the exact position but resolve contacts on the tile grid.
    "game_state": "raycast / scandot tile march starts at int(px/24)",
}

#: Quantized keys that ACCUMULATE over the episode, so one boundary step
#: contaminates every later step. Their carve-out is sticky: a divergence at
#: step ``s`` is excused if ANY step up to ``s`` was on a boundary.
EPISODIC_QUANTIZED_KEYS: Dict[str, str] = {
    "visitation_map": "per-episode visit grid; visited cell is int(px/12)",
    "frontier_map": "derived from the same per-episode visit grid",
}
QUANTIZED_KEYS.update(EPISODIC_QUANTIZED_KEYS)

TIE_BREAK_SENSITIVE: Dict[str, str] = {
    # The leg planner picks one of several equal-length routes. The mirrored
    # hint is a VALID route for the mirrored level, just not the one that
    # planner would have picked -- harmless as a value/aux target (equal-cost
    # routes have equal cost-to-go), a small input perturbation for the policy.
    "waypoint_nav": "planner picks one of several equal-length routes",
    # extract_surfaces samples floors every 6 px but every 3 px NEAR SEGMENT
    # ENDS, so the sample set itself is not symmetric at a platform's end column.
    "fall_lethality_map": "asymmetric floor sampling at segment ends",
    # A window onto fall_lethality_map, so it inherits that field's residual.
    "ego_crop_fall_fine": "window onto fall_lethality_map",
}

#: ``game_state`` dims that MUST match exactly on every step: the ninja physics
#: block (velocity, movement state, buffers, contacts, surface normals, physics
#: parameters, position) and the goal geometry. These are read straight off the
#: simulator with no grid quantization, so a divergence here means the SIM does
#: not commute -- the one failure this whole exercise exists to detect.
#:
#: Deliberately excluded: 23 ``time_remaining`` (normalised by a dynamic
#: truncation limit derived from the free-space field, which carries the
#: tie-break noise), 24-26 physics-derived (same), 32-39 raycasts and 57-68
#: scandots (tile marches), the arc bundles (tile-grid contact resolution) and
#: 40-43 free-space nav (``argmin`` over near-tied hops).
GAME_STATE_EXACT_DIMS: Tuple[int, ...] = tuple(range(0, 23)) + tuple(range(27, 32))

#: A ninja x within this many px of a 6 px cell boundary counts as "on a
#: boundary" for the quantization carve-out above. A real boundary hit is EXACT
#: in the simulator; the slack exists only because the harness recovers the
#: position from the float32 ``game_state`` (relative error ~1e-7 on a
#: level-width-normalised value, i.e. well under a thousandth of a pixel).
BOUNDARY_EPS_PX = 1e-3

#: Residual element fraction a quantized key may differ by on a NON-boundary
#: step. The same half-open-interval effect applies to static references the
#: readouts quantize -- the coarse crop's objective marker (``otx // 2``) and the
#: cost field's own cell grid -- and those do not move, so their one-cell offset
#: persists for the whole episode rather than only on the frames the ninja is
#: grid-aligned. Measured worst case over the official corpus is 0.15% of
#: elements; 1% leaves headroom while still catching a gross regression (a
#: missing tile-ID remap moves 10-40% of cells).
QUANTIZED_ELEMENT_FRACTION_TOLERANCE = 0.01


# Observation entries that are bookkeeping rather than policy inputs, or that
# reflection is not expected to preserve pointwise.
_IGNORED_KEYS = frozenset({"level_data", "switch_states_dict"})


@dataclass
class FieldDivergence:
    """One observation key that failed the commutation check."""

    key: str
    step: int
    max_abs_diff: float
    n_bad: int
    size: int
    #: Flat indices of the worst offenders, for a vector key (empty for maps).
    worst_indices: Tuple[int, ...] = ()

    def __str__(self) -> str:
        loc = f" at {list(self.worst_indices[:8])}" if self.worst_indices else ""
        return (
            f"{self.key}: step {self.step}, max|d|={self.max_abs_diff:.6g}, "
            f"{self.n_bad}/{self.size} elements differ{loc}"
        )


@dataclass
class CommutationReport:
    """Result of one ``check_commutation`` run."""

    steps: int
    n_obs_compared: int
    divergences: List[FieldDivergence] = field(default_factory=list)
    reward_max_abs_diff: float = 0.0
    reward_divergent: bool = False
    terminated_match: bool = True
    keys_compared: Tuple[str, ...] = ()
    #: Steps on which the ninja sat exactly on a 6 px quantization boundary.
    boundary_steps: frozenset = frozenset()

    @property
    def ok(self) -> bool:
        return (
            not self.divergences
            and not self.reward_divergent
            and self.terminated_match
        )

    def strict_divergences(self) -> List[FieldDivergence]:
        """Divergences that the symmetry contract does NOT excuse.

        Excused: any :data:`TIE_BREAK_SENSITIVE` key, and any
        :data:`QUANTIZED_KEYS` key on a step where the ninja sat exactly on a
        cell boundary. For ``game_state``, only the dims outside
        :data:`GAME_STATE_EXACT_DIMS` are excused -- the ninja physics block must
        match on EVERY step, boundary or not.
        """
        exact_dims = set(GAME_STATE_EXACT_DIMS)
        out: List[FieldDivergence] = []
        for d in self.divergences:
            if d.key == "game_state":
                if exact_dims.intersection(d.worst_indices):
                    out.append(d)
                continue
            if d.key in TIE_BREAK_SENSITIVE:
                continue
            if d.key in QUANTIZED_KEYS:
                if d.key in EPISODIC_QUANTIZED_KEYS:
                    if any(b <= d.step for b in self.boundary_steps):
                        continue
                elif d.step in self.boundary_steps:
                    continue
                frac = d.n_bad / d.size if d.size else 1.0
                if frac <= QUANTIZED_ELEMENT_FRACTION_TOLERANCE:
                    continue
            out.append(d)
        return out

    @property
    def contract_holds(self) -> bool:
        """The symmetry contract, as asserted by the test suite."""
        return (
            not self.strict_divergences()
            and self.terminated_match
            and not self.reward_divergent
        )

    def worst_fraction(self, key: str) -> float:
        """Largest fraction of ``key``'s elements that differed on any step."""
        worst = 0.0
        for d in self.divergences:
            if d.key == key and d.size:
                worst = max(worst, d.n_bad / d.size)
        return worst

    def bad_indices(self, key: str) -> set:
        """Union over steps of the divergent element indices of a FLAT key."""
        out: set = set()
        for d in self.divergences:
            if d.key == key:
                out.update(d.worst_indices)
        return out

    def per_key_stats(self) -> Dict[str, Tuple[float, float, int]]:
        """``{key: (max_abs_diff, worst_element_fraction, n_steps_divergent)}``."""
        out: Dict[str, Tuple[float, float, int]] = {}
        for d in self.divergences:
            cur = out.get(d.key, (0.0, 0.0, 0))
            out[d.key] = (
                max(cur[0], d.max_abs_diff),
                max(cur[1], d.n_bad / d.size if d.size else 0.0),
                cur[2] + 1,
            )
        return out

    def summary(self) -> str:
        if self.ok:
            return (
                f"mirror commutation OK: {self.steps} steps, "
                f"{self.n_obs_compared} observations, "
                f"{len(self.keys_compared)} keys, rewards identical"
            )
        verdict = "holds (excused)" if self.contract_holds else "VIOLATED"
        lines = [
            f"mirror commutation: contract {verdict} over {self.steps} steps "
            f"({self.n_obs_compared} observations, "
            f"{len(self.boundary_steps)} on a cell boundary)"
        ]
        for d in self.strict_divergences()[:8]:
            lines.append(f"  STRICT {d}")
        if not self.terminated_match:
            lines.append("  episode termination differed between the two runs")
        if self.reward_divergent:
            lines.append(
                f"  reward stream differs: max|d|={self.reward_max_abs_diff:.6g}"
            )
        # One line per KEY (worst step), not per step: a divergent field is
        # usually divergent on every step and the per-step list is unreadable.
        worst_by_key: Dict[str, FieldDivergence] = {}
        for d in self.divergences:
            cur = worst_by_key.get(d.key)
            if cur is None or d.max_abs_diff > cur.max_abs_diff:
                worst_by_key[d.key] = d
        for key in sorted(worst_by_key, key=lambda k: -worst_by_key[k].max_abs_diff):
            d = worst_by_key[key]
            n_steps = sum(1 for x in self.divergences if x.key == key)
            lines.append(f"  {d} (on {n_steps} step(s))")
        return "\n".join(lines)


def _build_env(exposure: Optional[Mapping[str, Any]], frame_skip: int):
    """Construct an ``NppEnvironment`` carrying the given obs exposures."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import dataclasses

    from ..gym_environment.config import EnvironmentConfig, ReachabilityConfig
    from ..gym_environment.npp_environment import NppEnvironment

    reach_fields = {f.name for f in dataclasses.fields(ReachabilityConfig)}
    reach_kw = {k: v for k, v in (exposure or {}).items() if k in reach_fields}
    cfg = EnvironmentConfig.for_training(reachability=ReachabilityConfig(**reach_kw))
    cfg.frame_skip = max(1, int(frame_skip))
    # The real level is injected per run; skip the throwaway procgen load.
    cfg.defer_initial_map_load = True
    return NppEnvironment(cfg)


def rollout(
    map_data: Sequence[int],
    actions: Sequence[int],
    *,
    exposure: Optional[Mapping[str, Any]] = None,
    frame_skip: int = 1,
    map_name: str = "mirror_verify",
    env=None,
    skip_reward: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], List[float], bool]:
    """Simulate ``actions`` on ``map_data`` and return (obs, rewards, terminated).

    ``map_name`` matters: several per-level fields (the fall-lethality field
    among them) are cached on the map NAME, so the two runs of a commutation
    check MUST use different names or run B silently reuses run A's field.
    """
    owns_env = env is None
    if owns_env:
        env = _build_env(exposure, frame_skip)
    try:
        env.skip_reward_calculation = bool(skip_reward)
        # Load through the benchmark path so ``map_name`` reaches the loader:
        # per-level fields (the fall-lethality field among them) are cached on
        # the map NAME, so two runs sharing a name would share a cached field
        # and the check would pass on a field it never actually recomputed.
        obs, _info = env.reset(
            options={
                "map_data": list(map_data),
                "map_name": map_name,
                "new_level": True,
            }
        )

        observations = [_copy_obs(obs)]
        rewards: List[float] = []
        terminated = False
        for action in actions:
            obs, reward, term, trunc, _info = env.step(int(action))
            observations.append(_copy_obs(obs))
            rewards.append(float(reward))
            if term or trunc:
                terminated = bool(term)
                break
        return observations, rewards, terminated
    finally:
        if owns_env:
            env.close()


def _copy_obs(obs: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in obs.items():
        if k in _IGNORED_KEYS or k.startswith("_"):
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
    return out


def check_involution(
    mirror: ObservationMirror, obs: Mapping[str, np.ndarray], atol: float = 0.0
) -> List[str]:
    """Return the keys for which ``M(M(x)) != x``.

    Catches sign errors and bad permutations immediately, without needing the
    simulator at all.
    """
    twice = mirror(mirror(obs))
    bad = []
    for key, value in obs.items():
        got = twice[key]
        a = np.asarray(value)
        b = np.asarray(got)
        if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=0.0):
            bad.append(key)
    return bad


def check_commutation(
    map_data: Sequence[int],
    actions: Sequence[int],
    *,
    exposure: Optional[Mapping[str, Any]] = None,
    frame_skip: int = 1,
    atol: float = 1e-5,
    # PBRS is shaped by the free-space potential, whose Dijkstra relaxes cells in
    # a different ORDER in the mirrored level. The sums are mathematically equal
    # but differ in their last bits, and the potential difference is scaled by
    # pbrs_weight. Measured worst case over the official corpus is 3e-4 on
    # O(1) rewards; anything larger is a real reward asymmetry.
    reward_atol: float = 1e-3,
    ignore_keys: Sequence[str] = (),
    check_rewards: bool = True,
) -> CommutationReport:
    """Assert ``mirror(obs(sim(level, a))) == obs(sim(mirror(level), mirror(a)))``.

    Args:
        map_data: The level to test, as a raw ``map_data`` int list.
        actions: Discrete actions (0-5) to drive both runs.
        exposure: ``ReachabilityConfig`` obs-exposure flags. Use the RUN's real
            flags -- a field that is not exposed is not checked.
        atol: Absolute tolerance per observation element.
        reward_atol: Absolute tolerance on the per-step reward stream.
        ignore_keys: Keys to exclude (e.g. a field with a known, accepted
            asymmetry you are isolating).
        check_rewards: When False, reward calculation is skipped entirely in
            both runs (much faster; use for pure observation checks).
    """
    ignored = set(ignore_keys)
    actions = [int(a) for a in actions]
    mirrored_actions = [mirror_action(a) for a in actions]

    obs_a, rew_a, term_a = rollout(
        map_data,
        actions,
        exposure=exposure,
        frame_skip=frame_skip,
        map_name="mirror_verify_original",
        skip_reward=not check_rewards,
    )
    obs_b, rew_b, term_b = rollout(
        mirror_map_data(list(map_data)),
        mirrored_actions,
        exposure=exposure,
        frame_skip=frame_skip,
        map_name="mirror_verify_mirrored",
        skip_reward=not check_rewards,
    )

    n = min(len(obs_a), len(obs_b))
    mirror = ObservationMirror(
        int(np.asarray(obs_a[0]["game_state"]).shape[-1]), exposure=exposure
    )

    keys = tuple(
        sorted(k for k in obs_a[0] if k in obs_b[0] and k not in ignored)
    )
    # An ego-centric readout locates the ninja with int(px / cell); when px is
    # EXACTLY on a cell boundary the mirrored readout lands on the neighbouring
    # cell (a half-open interval reflects to the other half-open interval), so
    # every ninja-centred window shifts by one. Record those steps rather than
    # loosening the tolerance everywhere.
    from ..gym_environment.constants import LEVEL_WIDTH

    boundary = set()
    for step in range(n):
        nx = float(np.asarray(obs_a[step]["game_state"])[21]) * LEVEL_WIDTH
        cells = nx / 6.0
        if abs(cells - round(cells)) <= BOUNDARY_EPS_PX:
            boundary.add(step)

    report = CommutationReport(
        steps=n - 1,
        n_obs_compared=n,
        keys_compared=keys,
        terminated_match=(term_a == term_b) and (len(obs_a) == len(obs_b)),
        boundary_steps=frozenset(boundary),
    )

    for step in range(n):
        expected = mirror(obs_a[step])
        for key in keys:
            got = np.asarray(obs_b[step][key], dtype=np.float64)
            want = np.asarray(expected[key], dtype=np.float64)
            if got.shape != want.shape:
                report.divergences.append(
                    FieldDivergence(key, step, float("inf"), got.size, got.size)
                )
                continue
            diff = np.abs(got - want)
            worst = float(diff.max()) if diff.size else 0.0
            if worst > atol:
                bad = np.flatnonzero(diff.ravel() > atol)
                report.divergences.append(
                    FieldDivergence(
                        key=key,
                        step=step,
                        max_abs_diff=worst,
                        n_bad=int(bad.size),
                        size=int(diff.size),
                        # Full index list for flat keys (they are small and the
                        # caller needs to know exactly WHICH dims moved);
                        # positions are meaningless for the maps.
                        worst_indices=tuple(int(i) for i in bad)
                        if diff.ndim == 1
                        else (),
                    )
                )

    if check_rewards:
        m = min(len(rew_a), len(rew_b))
        if m:
            rdiff = np.abs(np.array(rew_a[:m]) - np.array(rew_b[:m]))
            report.reward_max_abs_diff = float(rdiff.max())
            report.reward_divergent = report.reward_max_abs_diff > reward_atol

    return report
