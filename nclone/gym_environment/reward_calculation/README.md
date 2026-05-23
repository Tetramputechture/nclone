# Reward calculation (unified path)

The environment emits one reward per `env.step()`, produced by
`RewardCalculator.calculate_reward` in `main_reward_calculator.py`. There is no
longer a per-physics-frame primitive-reward loop or a precomputed PBRS branch.
BSRS shaping is applied on the trainer side (`MaskedPPO` / `FiGAR-PPO`).

## Formula

```
r = r_sparse                                    # terminal events (switch, exit, death, truncation)
  + beta(t) * alpha * hwm_progress              # A* HWM shaping, always non-negative
  + time_penalty_per_step * frames_executed     # small per-frame cost
  + BACKTRACK_PENALTY * frames_executed         # only when A* distance to goal strictly increases
```

| Component               | Source                                                   | Notes                                                         |
|-------------------------|----------------------------------------------------------|---------------------------------------------------------------|
| `r_sparse`              | `RewardConfig` (`R_SWITCH`, `R_EXIT`, `R_DEATH[_MINE]`, `TRUNCATION_PENALTY`) | Fires once per event inside a step.                            |
| `hwm_progress`          | `EpisodeRewardState.step` in `hwm_shaping.py`            | Two-phase (switch -> exit), normalized against initial distance. |
| `alpha`                 | `RewardConfig.hwm_alpha`                                 | Scales HWM before the blend.                                  |
| `beta(t)`               | `RewardConfig.beta_blend()`                              | HWM anneal; complementary BSRS term `(1 - beta) * eta * (gamma V' - V)` is added by the trainer. |
| `time_penalty_per_step` | `RewardConfig.time_penalty_per_step`                     | `-0.001` per frame (configurable).                            |
| `BACKTRACK_PENALTY`     | `reward_constants.BACKTRACK_PENALTY`                     | `-0.0004` per frame while moving away from current goal.      |

`GLOBAL_REWARD_SCALE` is applied once to the final float before returning.

## Trainer contract

`collect_rollouts` in FiGAR-PPO (and MaskedPPO) consumes the env step reward
directly as the primitive reward `r_t`. When BSRS is enabled, the learner adds
the bootstrapped shaping term without touching the environment:

```
r_t <- r_t + (1 - beta(t)) * eta * (gamma * V(s_{t+1}) - V(s_t))
```

This keeps the env deterministic and free of value-function dependencies while
still scheduling the dense signal from pure HWM (beta=1) to pure BSRS (beta=0)
over `bsrs_blend_horizon_fraction * total_timesteps`.

## Per-step diagnostics

`RewardCalculator.last_components` is populated every step with:

- `hwm_reward`, `hwm_progress`, `beta_blend_hwm`
- `time_penalty`, `backtrack_penalty`
- `total_reward`, `scaled_reward`
- `distance_to_switch`, `distance_to_exit`
- `watermark_switch`, `watermark_exit`, `current_phase`
- `closest_distance_episode`, `movement_distance`
- `is_terminal`, `truncated`

Episode-level rollups are in `get_episode_reward_metrics` (consumed by
`PBRSLoggingCallback` in `npp-rl`): `hwm_total`, `time_penalty_total`,
`terminal_reward`, `forward_steps`, `backtrack_steps`, `path_optimality`, etc.

## Deleted / offline

- `primitive_reward_calculator.py`, `reward_machine.py`, `adaptive_pbrs.py`,
  `sparse_reward_calculator.py`, `pbrs_potentials.py` -- removed.
- `nclone/simulation_pbrs/` -- removed (phi-grid precompute is no longer used
  at runtime).
- `nclone/value_function/` -- retained strictly as an **offline** curriculum
  tool (route extraction, surface graph, `visualize_route_interactive.py`).
  It is never imported by the runtime reward path.
