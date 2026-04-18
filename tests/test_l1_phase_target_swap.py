"""L1.5: EpisodeRewardState uses dist_switch pre-switch and dist_exit post-switch."""

from nclone.gym_environment.reward_calculation.hwm_shaping import EpisodeRewardState


def test_phase_uses_switch_distance_then_exit_distance():
    s = EpisodeRewardState()
    s.reset(
        initial_dist_switch=100.0,
        max_dist_switch=100.0,
        initial_dist_exit=200.0,
        max_dist_exit=200.0,
    )

    p0 = s.step(dist_switch=90.0, dist_exit=5.0, switch_activated=False)
    assert p0 > 0.0

    p_ignore_exit = s.step(dist_switch=85.0, dist_exit=999.0, switch_activated=False)
    assert p_ignore_exit > 0.0

    s.step(dist_switch=10.0, dist_exit=150.0, switch_activated=True)

    p_sw_only = s.step(dist_switch=0.0, dist_exit=150.0, switch_activated=True)
    assert p_sw_only == 0.0

    p_ex = s.step(dist_switch=0.0, dist_exit=120.0, switch_activated=True)
    assert p_ex > 0.0
