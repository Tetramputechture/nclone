# Epic 4: Online Multi-Level FiGAR-PPO Training

## Goal

Train a FiGAR-PPO agent simultaneously across 200+ levels using the universal PBRS reward signal from Epic 3. The agent must develop a generalized policy that can solve unseen levels at a meaningful zero-shot rate by the end of training.

## Background

FiGAR-PPO (Fine-Grained Action Repetition PPO) is the preferred online RL algorithm because:
- Its factored action + duration heads `P(action) * P(duration)` explicitly learn how long to hold each action, well-suited to N++'s momentum-based movement
- Contact-interrupt stops macro-actions when the ninja's movement regime changes (grounded / airborne / wall-sliding), preventing over-commitment at key physics transitions
- Duration bins: `[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 45, 64, 90]` frames (13 bins, log-spaced)
- Base actions: 6 (NOOP, LEFT, RIGHT, JUMP, JUMP+LEFT, JUMP+RIGHT)
- Buffer: `PrimitiveSMDPRolloutBuffer` stores per-primitive-frame data with decision-point flags

**Key gap: FiGAR-PPO currently lacks a level-context encoder.** SAC's `SACFeatureExtractor` has a dedicated `LevelContextCNN` (42×23 tile grid → 768D CNN → 64D MLP), but the PPO path's `ConfigurableMultimodalExtractor` has no equivalent tile-encoding branch. Without level context, the FiGAR-PPO policy cannot distinguish between levels, which prevents generalization.

**Second gap: no multi-level scheduler for PPO.** SAC has `LevelScheduler` for assigning environments to levels and rotating stalled ones. PPO's `architecture_trainer.py` supports a single-level `CurriculumEnv` but not a rotating multi-level scheduler.

**Third gap: demo augmentation is not FiGAR-aware.** `DemoAugmentationCallback` currently injects demo transitions into the rollout buffer without setting `duration_indices` or `old_duration_log_prob`. This breaks `PrimitiveSMDPRolloutBuffer` consistency for the duration head.

## Success Criteria

- FiGAR-PPO trains stably across 200+ levels for 100M steps without NaN or policy collapse
- Per-level solve rate on training levels: ≥70% at 100M steps
- Zero-shot solve rate on held-out levels: ≥40% (no per-level fine-tuning)
- Level-context encoding is verifiably used: two policies trained with and without level context show measurable solve rate difference on held-out levels
- Demo augmentation produces valid `duration_indices` in the buffer (no assertion errors in `PrimitiveSMDPRolloutBuffer`)

## Key Files

| File | Role |
|------|------|
| `npp_rl/agents/figar_ppo.py` | `FiGARPPO` — macro-action collection, factored loss |
| `npp_rl/agents/figar_policy.py` | `FiGARActorCriticPolicy` — action head + duration head |
| `npp_rl/agents/primitive_smdp_rollout_buffer.py` | Per-primitive-frame buffer with decision-point tracking |
| `npp_rl/training/architecture_trainer.py` | PPO training loop, curriculum, callbacks |
| `npp_rl/training/environment_factory.py` | `NppEnvironment` creation with `EnvironmentConfig` |
| `npp_rl/feature_extractors/configurable_extractor.py` | Multimodal feature extractor (add level-context branch) |
| `npp_rl/callbacks/demo_callback.py` | Demo augmentation (fix FiGAR duration handling) |

---

## Tickets

### T4.1 — Add level-context encoding to the PPO feature extractor

**What:** Add a level-context branch to `ConfigurableMultimodalExtractor` (or create a new `FiGARLevelContextExtractor`) that encodes the 42×23 tile grid into a fixed-size vector and fuses it with the existing physics/spatial features.

**Why:** Without level context, the policy receives identical feature representations on two different levels with the same ninja state but different tile layouts. The policy cannot learn level-specific routing, platform recognition, or mine avoidance patterns that depend on level geometry.

**Acceptance criteria:**
- A new feature branch accepts `tile_grid` from the environment observation dict: `obs["tile_grid"]` with shape `(B, 42, 23)` int64
- The branch runs a CNN encoder: 42×23 tile grid → embedding lookup (38 types → 16D) → 3 conv layers (16→32→64→64) → global average pool → linear → 64D output
- The CNN output is cached per level — `level_id` is read from the observation or passed as a side-channel; the CNN only re-runs when `level_id` changes between steps
- The 64D level encoding is concatenated with the existing features before the final fusion MLP
- The branch is controlled by a new `ModalityConfig` flag `enable_level_context: bool = True`
- When `enable_level_context = False`, the extractor behaves identically to the current implementation (backward compatible)

**Key implementation notes:**
- The tile grid must be added to the `NppEnvironment` observation dict. Check whether `env.get_tile_grid()` already exists (it does on `SACEnvironment`); add it to the `NppEnvironment` observation pipeline if missing
- Caching strategy: maintain a `Dict[int, torch.Tensor]` mapping `level_id → cached_encoding`; invalidate on `env.reset()` with a new level
- The `level_id` field must be included in the observation dict (or in `env.info`); verify `NppEnvironment` exposes it
- Keep the CNN small: global avg pool after the conv layers eliminates spatial dimensions and produces a translation-invariant level encoding; this follows the same design as the IQL `TileEncoder`

**Relevant files:** `npp_rl/feature_extractors/configurable_extractor.py`, `nclone/gym_environment/npp_environment.py`, `npp_rl/training/architecture_configs.py`

---

### T4.2 — Build the multi-level FiGAR-PPO training script

**What:** Create `npp_rl/scripts/train_ppo_multilevel.py`, a training entry point that accepts a level directory, assigns levels to parallel environments, and trains FiGAR-PPO with the neural PBRS reward.

**Why:** The existing PPO training scripts (`train_and_compare.py`) target single-level or fixed-curriculum setups. A dedicated multi-level script keeps the training loop clean and avoids coupling multi-level logic to the comparison infrastructure.

**Acceptance criteria:**
- Script accepts `--level-dir` (directory of level files), `--checkpoint` (neural PBRS checkpoint path), `--n-envs` (default 32), `--total-steps` (default 100M)
- Each of the `n_envs` environments is initialized with a level drawn from the level directory and the neural PBRS checkpoint from Epic 3
- FiGAR-PPO is instantiated with `FiGARActorCriticPolicy` and the level-context extractor from T4.1
- Training runs and logs to TensorBoard: per-level solve rate, mean episode reward, policy entropy (action and duration separately), alpha_shaping value
- Checkpoint saved every 1M steps to `--output-dir`
- Script exits cleanly on `KeyboardInterrupt` and saves a final checkpoint

**Key implementation notes:**
- Use `EnvironmentConfig.for_training()` with `neural_pbrs_checkpoint` set; each environment gets the same checkpoint but maintains its own `NeuralValuePBRS` instance (no shared state)
- `NppEnvironment` observations are dicts; ensure `FiGARActorCriticPolicy` uses a dict-obs-compatible feature extractor (`ConfigurableMultimodalExtractor` or the new `FiGARLevelContextExtractor`)
- Set `EnvironmentConfig.max_time_in_frames = 10000` as the episode time limit; truncation (not termination) on timeout
- The number of environments should be sized to fill available CPU cores; use `SubprocVecEnv` for parallelism

**Relevant files:** `npp_rl/training/architecture_trainer.py`, `npp_rl/training/environment_factory.py`, `nclone/gym_environment/config.py`

---

### T4.3 — Implement multi-level scheduler for PPO

**What:** Create `npp_rl/training/ppo_level_scheduler.py`, a scheduler that manages which levels are assigned to which environments and rotates stalled levels out of the pool.

**Why:** During training, some levels will be solved quickly (easy layouts) while others stall (complex tile geometry, dense mines). The scheduler ensures training time is allocated proportionally to unsolved levels, preventing the agent from plateauing by over-training on solved levels.

**Acceptance criteria:**
- `PPOLevelScheduler` tracks per-level: episode count, solve count, last solve step, stall flag
- Initialization: assign levels round-robin across `n_envs` environments
- Advancement: when a level reaches ≥80% solve rate over a rolling 50-episode window, it is marked "solved" and deprioritized
- Stall detection: when a level has not been solved in the last 500 episodes, it is marked "stalled"
- On environment episode end: if the assigned level is solved or stalled, the scheduler assigns a new unsolved level to that environment
- Level selection for reassignment: sample from unsolved levels, weighted by inverse solve rate (harder levels get more time)
- Statistics logged every 10K steps: number of solved/stalled/active levels, global solve rate

**Key implementation notes:**
- The scheduler is called from the training loop's episode-end hook; it should be lightweight (no blocking I/O or heavy computation)
- Level files are loaded lazily: the level path is passed to `EnvironmentConfig`; the environment loads the level file on `reset()`, not before
- The scheduler does not need to be aware of the level content — only the level ID and solve/stall statistics
- Keep state serializable (plain dicts or dataclasses) so the scheduler state can be checkpointed and restored

**Relevant files:** `npp_rl/training/level_scheduler.py` (reference for interface), `npp_rl/training/architecture_trainer.py`

---

### T4.4 — Fix FiGAR demo augmentation for duration handling

**What:** Update `DemoAugmentationCallback` to correctly populate `duration_indices` and `old_duration_log_prob` when injecting demo transitions into `PrimitiveSMDPRolloutBuffer`.

**Why:** `PrimitiveSMDPRolloutBuffer` is used by FiGAR-PPO and stores per-primitive-frame data including `duration_indices` (which duration bin was chosen) and `old_duration_log_prob` (for PPO clipping). Demo transitions currently injected by `DemoAugmentationCallback` do not set these fields, causing the buffer to use uninitialized values (zeros or garbage), which distorts the duration head's gradient.

**Acceptance criteria:**
- Demo transitions injected into the buffer have valid `duration_indices` derived from the demo's run-length-encoded macro-action durations
- `old_duration_log_prob` is computed by calling `FiGARActorCriticPolicy.evaluate_actions(obs, actions, duration_indices)` on the injected demos
- The buffer passes its own consistency check after demo injection (no assertion errors, no NaN in `old_duration_log_prob`)
- The callback correctly handles the case where demo durations exceed the max duration bin (clamp to the nearest bin)
- Demo-augmented training runs without PPO clipping ratio errors (`ratio > 100`) in the duration head

**Key implementation notes:**
- Demo macro-actions have durations in raw frames (1–90); map to the 13 duration bins `[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 45, 64, 90]` using nearest-bin lookup
- The `DemoAugmentationCallback` uses `DemonstrationBuffer` which stores replay transitions; duration information is available from the run-length encoding in `DemoConverter`
- `FiGARActorCriticPolicy.evaluate_actions()` accepts an optional `duration_indices` tensor; when provided, it computes `duration_log_prob` and adds it to the returned log probability

**Relevant files:** `npp_rl/callbacks/demo_callback.py`, `npp_rl/agents/figar_policy.py`, `npp_rl/agents/primitive_smdp_rollout_buffer.py`

---

### T4.5 — Training diagnostics and monitoring

**What:** Add per-level training diagnostics to the FiGAR-PPO training loop and document the expected training curve shapes for the multi-level setting.

**Why:** Multi-level PPO training is harder to debug than single-level: a policy collapse on one difficult level may not show up in the mean reward until many other levels have also degraded. Per-level tracking makes regressions visible immediately.

**Acceptance criteria:**
- TensorBoard logs the following per-level metrics (updated every 10 episodes per level):
  - `levels/{level_id}/solve_rate` — rolling 50-episode solve rate
  - `levels/{level_id}/mean_episode_length` — rolling mean in frames
  - `levels/{level_id}/mean_reward` — rolling mean episode reward
- Global metrics logged every 1K steps:
  - `train/solve_rate_all` — mean across all active levels
  - `train/policy_entropy_action` — entropy of the action distribution
  - `train/policy_entropy_duration` — entropy of the duration distribution
  - `train/alpha_shaping` — current PBRS shaping weight from T3.4
  - `train/n_solved_levels`, `train/n_stalled_levels`, `train/n_active_levels`
- A summary document `docs/generalizable_agent_v2/04-online-multi-level-figar-ppo/expected-training-curves.md` describes what healthy vs. unhealthy training looks like: expected timeline, warning signs (entropy collapse, solve rate plateau, duration bin saturation)

**Key implementation notes:**
- Per-level metrics can be tracked in the `PPOLevelScheduler` from T4.3 and logged by a lightweight callback
- Duration entropy collapsing to near-zero means the agent is always choosing the same duration regardless of context — this is an early warning of policy collapse
- A healthy training curve should show: early fast progress on easy (no-mine) levels, then gradual improvement on mine-heavy levels as the value function guides the agent to avoid hazards

**Relevant files:** `npp_rl/training/architecture_trainer.py`, `npp_rl/callbacks/`
