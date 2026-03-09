Here's the complete breakdown of all TensorBoard stats, organized by namespace and source:

---

## `train/` — Core Training Metrics
Logged every `log_interval` (default: 10k steps) once the buffer is full.

### Episode Outcomes
| Stat | Description |
|---|---|
| `train/ep_reward_mean50` | Mean episode reward over last 50 episodes |
| `train/ep_length_mean50` | Mean episode length (frames) over last 50 |
| `train/ep_reward_std50` | Std dev of episode reward (requires ≥2 episodes) |
| `train/success_rate50` | Fraction of last 50 episodes agent succeeded (goal reached) |
| `train/death_rate50` | Fraction of last 50 episodes ninja died (mine/impact/OOB) |
| `train/timeout_rate50` | Fraction of last 50 episodes that truncated at time limit |

### Losses
| Stat | Description |
|---|---|
| `train/q_loss` | Mean combined Q-network loss across UTD updates |
| `train/q1_loss` | Mean Q1 critic loss (logged separately when available) |
| `train/q2_loss` | Mean Q2 critic loss (logged separately when available) |
| `train/policy_loss` | Mean policy network loss across UTD updates |
| `train/alpha_loss` | Temperature (entropy coeff) loss |

### Temperature / Entropy
| Stat | Description |
|---|---|
| `train/alpha` | Current temperature value |
| `train/log_alpha` | Raw `log_alpha` parameter |
| `train/policy_entropy` | Mean policy entropy from diagnostic batch |
| `train/target_entropy` | Target entropy threshold |

### Q-Value Diagnostics
| Stat | Description |
|---|---|
| `train/mean_q` | Mean Q-value across actions in diagnostic batch |
| `train/max_q` | Max Q-value — monitors overestimation |
| `train/min_q` | Min Q-value |
| `train/q1_q2_divergence` | Twin-critic divergence; sustained high = one critic drifting |
| `train/q_target_mean` | Mean Q-target value |
| `train/q_target_clipped_frac` | Fraction of Q-targets that hit the clamp bound; >0.30 triggers a warning, >0.60 indicates a degenerate value function |

### Gradient Health
| Stat | Description |
|---|---|
| `train/q_grad_norm` | Gradient norm for Q-networks (clips at `max_grad_norm=1.0`) |
| `train/policy_grad_norm` | Gradient norm for policy network |

### Training State / Throughput
| Stat | Description |
|---|---|
| `train/fps` | Training throughput (env steps / second) |
| `train/effective_utd` | Actual UTD ratio (halved to `utd_ratio//2` during first 500k steps) |
| `train/buffer_size` | Online replay buffer fill level |
| `train/demo_buffer_size` | Demo replay buffer size |
| `train/demo_ratio` | Current demo fraction in mixed batches (decays after 1M steps) |

### Action Distribution / SMDP Health
| Stat | Description |
|---|---|
| `train/jump_action_fraction` | Fraction of recent actions that were jump-type (base actions 1–3) |
| `train/action_duration_mean` | Mean macro-action duration in frames (rolling 2000-step window) |

---

## `obs/` — Observation Space Diagnostics
Logged every `log_interval`.

| Stat | Description |
|---|---|
| `obs/path_dist_to_current_goal_mean` | Mean of reachability feat 7 (path dist to current goal, normalized 0–1) |
| `obs/path_dist_to_current_goal_min` | Min across diagnostic batch |
| `obs/path_dist_to_current_goal_max` | Max across diagnostic batch |
| `obs/reach_0_path_dist_switch_mean` | Reachability feat 0: path distance to switch |
| `obs/reach_1_path_dist_exit_mean` | Reachability feat 1: path distance to exit |
| `obs/reach_2_switch_activated_mean` | Reachability feat 2: switch activated flag |
| `obs/reach_3_next_hop_dir_x_mean` | Reachability feat 3: next-hop direction X |
| `obs/reach_4_next_hop_dir_y_mean` | Reachability feat 4: next-hop direction Y |
| `obs/reach_5_platform_ahead_mean` | Reachability feat 5: platform-ahead indicator |
| `obs/reach_6_platform_below_mean` | Reachability feat 6: platform-below indicator |

---

## `diag/` — Reachability Normalization Diagnostics
Logged every `log_interval`. Sourced from `get_reachability_diagnostics()`.

| Stat | Description |
|---|---|
| `diag/reach_area_scale` | Level surface area scale factor used for normalization |
| `diag/reach_surface_area` | Raw level surface area |
| `diag/reach_switch_dist_raw` | Raw (unnormalized) graph distance to switch; -1 if unreachable |
| `diag/reach_exit_dist_raw` | Raw graph distance to exit; -1 if unreachable |
| `diag/reach_inf_fraction` | Fraction of graph nodes with infinite path distance |

---

## `curriculum/` — Per-Level RFCL Progress (Fast)
Logged every `log_interval` alongside the main stats (the "fast" curriculum view).

Per level `{lid}`:
| Stat | Description |
|---|---|
| `curriculum/level_{lid}_p0_frontier_success` | Phase 0 frontier success rate (triggers advance when ≥ `frontier_advance_threshold`) |
| `curriculum/level_{lid}_p1_frontier_success` | Phase 1 frontier success rate |
| `curriculum/level_{lid}_p0_frontier_idx` | Current Phase 0 frontier snapshot index (progress through reverse curriculum) |
| `curriculum/level_{lid}_spawn_success_rate` | Full-level success rate from spawn (the ultimate metric) |

---

## `rfcl/` — Full RFCL Diagnostics (Slow)
Logged every `rfcl_log_interval` (default: 50k steps) by `RFCLDiagnostics`.

Per level `{lid}`:
| Stat | Description |
|---|---|
| `rfcl/level_{lid}_phase0_frontier_idx` | Phase 0 frontier snapshot index |
| `rfcl/level_{lid}_phase0_frontier_success` | Phase 0 frontier success rate |
| `rfcl/level_{lid}_phase0_complete` | Phase 0 complete flag (0/1) |
| `rfcl/level_{lid}_phase1_frontier_idx` | Phase 1 frontier snapshot index |
| `rfcl/level_{lid}_phase1_frontier_success` | Phase 1 frontier success rate |
| `rfcl/level_{lid}_phase1_complete` | Phase 1 complete flag (0/1) |
| `rfcl/level_{lid}_p0_stage` | 0 = still in reverse curriculum, 1 = advanced to forward |
| `rfcl/level_{lid}_spawn_success_rate` | Success rate from spawn |
| `rfcl/level_{lid}_fully_complete` | Both phases complete (0/1) |

Scheduler aggregates:
| Stat | Description |
|---|---|
| `rfcl/n_active_levels` | Number of levels currently being trained |
| `rfcl/n_solved_levels` | Number of fully solved levels |
| `rfcl/n_stalled_levels` | Number of stalled levels (no frontier progress) |
| `rfcl/n_total_levels` | Total level count |
| `rfcl/fraction_solved` | `n_solved / n_total` |

---

## `eval/` — Spawn-to-Exit Evaluation
Logged every `eval_interval` (default: 50k steps) by `SACEvaluator`, using deterministic policy from spawn (no curriculum resets).

Per level `{lid}`:
| Stat | Description |
|---|---|
| `eval/level_{lid}_spawn_to_switch_rate` | Fraction of eval episodes reaching the switch |
| `eval/level_{lid}_spawn_to_exit_rate` | Fraction completing the full level (the ground-truth capability metric) |
| `eval/level_{lid}_mean_completion_frames` | Mean frames for successful completions; `max_frames` if none succeeded |

---

## `dense_reward/` — Dense Reward Component Breakdown
Logged every `log_interval`, only when `use_dense_rewards=True`. Rolling mean over last 200 completed episodes.

| Stat | Description |
|---|---|
| `dense_reward/dense_pbrs_total_mean` | Mean PBRS (potential-based reward shaping) per episode |
| `dense_reward/dense_vel_total_mean` | Mean velocity-alignment reward per episode |
| `dense_reward/dense_milestone_total_mean` | Mean distance-milestone bonus per episode |
| `dense_reward/dense_hazard_total_mean` | Mean mine hazard avoidance reward per episode |
| `dense_reward/dense_total_mean` | Mean total dense shaping per episode |
| `dense_reward/fraction_of_episode_reward` | Dense shaping as a fraction of total episode reward (sanity-check that shaping isn't dominating) |