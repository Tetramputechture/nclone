"""Offline validation of the scale-invariant free-space PBRS reward.

This de-risks the long generalist training run WITHOUT a GPU by driving the real
``RewardCalculator`` with the *recommended* config — free-space PBRS, fixed
reference ``K = _pbrs_d0_ref_override`` — over synthetic distance trajectories and
asserting the properties that were BROKEN before (per-episode D0 made the per-step
gradient scale 1/route-length, fading on long/open levels and inconsistent across
the curriculum -> overfitting; see the long-horizon-distance-dropoff finding).

Calibration over the 840-level mechanic_benchmark_v2 (5 movement buckets + the
FAR_OBJECTIVE bucket) found spawn cost-to-completion p50=67, p99=428, max=909, with
free-space available on 100% of levels. We pick K=1000 (round, ~10% headroom over
max) so EVERY level — including the FAR_OBJECTIVE long-horizon ones the agent must
generalize to — has a live (non-clamped) potential gradient from frame 0.

Properties asserted:
  1. Scale-invariance: a fixed cost decrement earns the SAME PBRS on a short and an
     open level (the fix). The per-episode-D0 path is shown to violate this.
  2. No clamp at K: the longest level's Phi(spawn) > -1 (gradient live at spawn).
  3. Telescoping / policy-invariance: total PBRS is path-independent — wandering
     earns the same as a direct route, so the agent cannot farm the dense term.
  4. Directionality: toward-goal steps earn PBRS>0, away-from-goal earn PBRS<0.
  5. Alignment: a winning trajectory earns >> a timeout/death, and with the
     recommended clawback=1.0 the death payoff is location-INDEPENDENT (no
     dive-to-death incentive on the geometric free-space potential), while the
     per-step gradient (4) still guides exploration toward the goal.

Run standalone for a readable report::

    ../nclone/.venv/bin/python -m nclone.tests.test_scale_invariant_pbrs
"""
from __future__ import annotations

from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.main_reward_calculator import (
    RewardCalculator,
)

K = 1000.0  # fixed PBRS reference (> max spawn cost-to-completion 909 on v2)
W = 2.0  # pbrs_weight (gentle: longest-level dense total = W*1.0=2.0 vs completion 5.5)
GAMMA = 1.0  # shaping gamma=1.0 -> strict telescoping + scale-invariance
CLAWBACK = 1.0  # recommended for the GEOMETRIC free-space potential (no dive-to-death)


def _cfg(*, fixed_ref: bool = True, weight: float = W, clawback: float = CLAWBACK) -> RewardConfig:
    return RewardConfig(
        enable_lattice_pbrs=True,
        _pbrs_weight_override=weight,
        _pbrs_gamma_override=GAMMA,
        _pbrs_d0_ref_override=(K if fixed_ref else None),
        pbrs_death_clawback=clawback,
    )


def _run(cfg, costs, *, switch_at=None, terminal="none"):
    """Drive RewardCalculator over a cost-to-completion trajectory.

    ``costs`` is the per-step free-space cost-to-completion (we set switch and exit
    distances equal so the potential basis is one continuous field across switch
    activation). ``terminal`` in {none, win, death, timeout} fires on the last step.
    Returns (total_reward, total_pbrs, per_step_components, calc).
    """
    rc = RewardCalculator(reward_config=cfg)
    rc.reset()
    prev = {"player_x": 50.0, "player_y": 50.0, "switch_activated": False}
    total = 0.0
    total_pbrs = 0.0
    comps = []
    n = len(costs)
    for i, c in enumerate(costs):
        last = i == n - 1
        sw = switch_at is not None and i >= switch_at
        obs = {
            "player_x": 50.0,
            "player_y": 50.0,
            "switch_x": 200.0,
            "switch_y": 200.0,
            "exit_door_x": 300.0,
            "exit_door_y": 300.0,
            "switch_activated": sw,
            "player_dead": last and terminal == "death",
            "player_won": last and terminal == "win",
            "_cached_switch_distance": float(c),
            "_cached_exit_distance": float(c),
        }
        trunc = last and terminal == "timeout"
        r = rc.calculate_reward(obs, prev, frames_executed=1, truncated=trunc)
        total += r
        total_pbrs += rc.last_components.get("pbrs_reward", 0.0)
        comps.append(rc.last_components)
        prev = obs
    return total, total_pbrs, comps, rc


# ---------------------------------------------------------------------------


def test_scale_invariance():
    """A fixed 6-unit cost decrement earns the SAME PBRS on a short (D0=67) and a
    far (D0=909) level under fixed K — and DIFFERENT PBRS under per-episode D0."""
    # Fixed K: per-step PBRS = W * (delta / K), independent of the level's D0.
    _, _, cs_short, _ = _run(_cfg(fixed_ref=True), [67.0, 61.0])
    _, _, cs_open, _ = _run(_cfg(fixed_ref=True), [909.0, 903.0])
    p_short = cs_short[1]["pbrs_reward"]
    p_open = cs_open[1]["pbrs_reward"]
    expected = W * 6.0 / K
    assert abs(p_short - expected) < 1e-9, (p_short, expected)
    assert abs(p_open - expected) < 1e-9, (p_open, expected)
    assert abs(p_short - p_open) < 1e-9, "fixed K must be route-length invariant"

    # Per-episode D0 (the OLD broken path): PBRS = W*delta/D0 -> scales 1/D0.
    _, _, bs_short, _ = _run(_cfg(fixed_ref=False), [67.0, 61.0])
    _, _, bs_open, _ = _run(_cfg(fixed_ref=False), [909.0, 903.0])
    b_short = bs_short[1]["pbrs_reward"]
    b_open = bs_open[1]["pbrs_reward"]
    ratio = b_short / b_open
    assert ratio > 4.0, (
        f"per-episode D0 should be many-x stronger on the short level "
        f"(short={b_short:.4f} open={b_open:.4f} ratio={ratio:.2f})"
    )
    return p_short, p_open, b_short, b_open, ratio


def test_no_clamp_at_K():
    """The longest level (D0=max=909) must have Phi(spawn) > -1 with K=1000, so its
    gradient is live from frame 0 (a too-small K, e.g. 500, would clamp it to -1)."""
    _, _, comps, rc = _run(_cfg(fixed_ref=True), [909.0])
    phi_spawn = rc.last_components["potential"]
    assert phi_spawn > -1.0 + 1e-6, f"longest level clamps at spawn: {phi_spawn}"
    assert abs(phi_spawn - (-909.0 / K)) < 1e-9, phi_spawn

    # Contrast: a too-small K=500 would clamp the same far level to exactly -1.
    cfg500 = RewardConfig(
        enable_lattice_pbrs=True,
        _pbrs_weight_override=W,
        _pbrs_gamma_override=GAMMA,
        _pbrs_d0_ref_override=500.0,
    )
    _, _, _, rc500 = _run(cfg500, [909.0])
    assert abs(rc500.last_components["potential"] - (-1.0)) < 1e-9
    return phi_spawn


def test_telescoping_policy_invariance():
    """Total PBRS is path-independent: a direct descent and a wandering route to the
    same terminal earn the SAME dense total (= W*(Phi_T - Phi_0)). The agent cannot
    farm the dense reward by dawdling."""
    direct = [909.0, 740.0, 560.0, 380.0, 200.0, 60.0, 0.0]
    wander = [909.0, 820.0, 870.0, 500.0, 640.0, 120.0, 0.0]  # same endpoints
    _, pbrs_direct, _, _ = _run(_cfg(), direct, switch_at=4, terminal="win")
    _, pbrs_wander, _, _ = _run(_cfg(), wander, switch_at=4, terminal="win")
    expected = W * (0.0 - (-909.0 / K))  # W*(Phi_T - Phi_0)
    assert abs(pbrs_direct - pbrs_wander) < 1e-9, (pbrs_direct, pbrs_wander)
    assert abs(pbrs_direct - expected) < 1e-9, (pbrs_direct, expected)
    return pbrs_direct, pbrs_wander, expected


def test_directionality():
    """Toward-goal steps earn PBRS>0; away-from-goal earn PBRS<0."""
    _, _, comps, _ = _run(_cfg(), [150.0, 120.0, 150.0])  # toward, then away
    toward = comps[1]["pbrs_reward"]
    away = comps[2]["pbrs_reward"]
    assert toward > 0, toward
    assert away < 0, away
    assert abs(toward + away) < 1e-9, "symmetric: equal-and-opposite displacement"
    return toward, away


def test_alignment_win_beats_timeout_and_death():
    """Completing earns >> timing out or dying; with clawback=1.0 the death payoff is
    location-INDEPENDENT (no dive-to-death on the geometric potential)."""
    descent = [909.0, 740.0, 560.0, 380.0, 200.0, 60.0, 0.0]
    win_total, _, _, _ = _run(_cfg(), descent, switch_at=4, terminal="win")

    idle = [150.0] * 7
    timeout_total, _, _, _ = _run(_cfg(), idle, terminal="timeout")

    death_spawn_total, _, _, _ = _run(_cfg(), [909.0] * 7, terminal="death")
    death_deep_total, _, _, _ = _run(
        _cfg(), [909.0, 740.0, 560.0, 380.0, 200.0, 90.0, 40.0], terminal="death"
    )

    assert win_total > 5.0, win_total  # ~R_SWITCH+R_EXIT+dense - time
    assert win_total > timeout_total + 5.0, (win_total, timeout_total)
    assert win_total > death_deep_total + 4.0, (win_total, death_deep_total)
    # clawback=1.0 -> dying close vs far pays the SAME (strict policy-invariance):
    # the agent cannot farm reward by diving toward the goal and dying.
    assert abs(death_deep_total - death_spawn_total) < 1e-6, (
        f"clawback=1.0 must make death location-independent "
        f"(deep={death_deep_total:.4f} spawn={death_spawn_total:.4f})"
    )
    return win_total, timeout_total, death_spawn_total, death_deep_total


def main():
    print("=" * 72)
    print(f"Scale-invariant free-space PBRS validation  (K={K:.0f}, W={W}, gamma={GAMMA})  [v2 curriculum]")
    print("=" * 72)

    p_s, p_o, b_s, b_o, ratio = test_scale_invariance()
    print("\n[1] SCALE-INVARIANCE (fixed cost decrement of 6 units)")
    print(f"    fixed K   : short PBRS={p_s:.5f}  far PBRS={p_o:.5f}  -> IDENTICAL ✓")
    print(f"    per-ep D0 : short PBRS={b_s:.5f}  far PBRS={b_o:.5f}  -> {ratio:.2f}x skew ✗ (the bug)")

    phi = test_no_clamp_at_K()
    print("\n[2] NO CLAMP AT K")
    print(f"    longest level (D0=909) Phi(spawn)={phi:.4f} > -1  -> gradient live ✓")
    print(f"    (a too-small K=500 would clamp it to -1.0 -> flat on the far levels ✗)")

    pd, pw, exp = test_telescoping_policy_invariance()
    print("\n[3] TELESCOPING / POLICY-INVARIANCE")
    print(f"    direct route PBRS total = {pd:.5f}")
    print(f"    wander route PBRS total = {pw:.5f}  -> path-independent ✓ (= W*(Phi_T-Phi_0)={exp:.4f})")

    tw, aw = test_directionality()
    print("\n[4] DIRECTIONALITY")
    print(f"    toward-goal PBRS = {tw:+.5f} (>0)   away-from-goal PBRS = {aw:+.5f} (<0)  ✓")

    win, to, ds, dd = test_alignment_win_beats_timeout_and_death()
    print("\n[5] ALIGNMENT (cumulative episode reward)")
    print(f"    WIN          = {win:+.3f}")
    print(f"    TIMEOUT-idle = {to:+.3f}")
    print(f"    DEATH@spawn  = {ds:+.3f}")
    print(f"    DEATH@deep   = {dd:+.3f}   -> win >> timeout/death; death location-independent (no dive-to-death) ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED — reward is scale-invariant, non-clamping, "
          "policy-invariant,\n  directional, and aligned across the curriculum.")
    print("=" * 72)


if __name__ == "__main__":
    main()
