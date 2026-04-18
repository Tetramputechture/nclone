"""L1.5: HWM shaping progress is non-negative; cumulative normalized reward <= 1 per phase."""

import math

from nclone.gym_environment.reward_calculation.hwm_shaping import EpisodeRewardState


def test_hwm_non_negative_and_bounded_per_phase():
    state = EpisodeRewardState()
    max_sw, max_ex = 100.0, 80.0
    state.reset(
        initial_dist_switch=100.0,
        max_dist_switch=max_sw,
        initial_dist_exit=80.0,
        max_dist_exit=max_ex,
    )

    switch_phase_dists = [100.0, 90.0, 95.0, 70.0, 70.0, 50.0]
    total_norm = 0.0
    for d in switch_phase_dists:
        p = state.step(
            dist_switch=d, dist_exit=999.0, switch_activated=False
        )
        assert p >= 0.0
        assert not math.isnan(p)
        total_norm += p
    assert total_norm <= 1.0 + 1e-6

    p_tr = state.step(
        dist_switch=30.0, dist_exit=60.0, switch_activated=True
    )
    assert p_tr == 0.0

    exit_phase_dists = [60.0, 55.0, 55.0, 20.0]
    total_exit = 0.0
    for d in exit_phase_dists:
        p = state.step(
            dist_switch=10.0, dist_exit=d, switch_activated=True
        )
        assert p >= 0.0
        total_exit += p
    assert total_exit <= 1.0 + 1e-6
