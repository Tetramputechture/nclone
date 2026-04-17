# Linear Plan Audit: N++ Generalizable Agent

## Scope

This audit is based on:

- The Linear project `N++ Generalizable Agent` and its current issue breakdown
- `docs/sim_mechanics_doc.md`
- `docs/STRATEGY_EPIC.md`
- Existing code and planning artifacts in `npp-rl`, including the already-present `minimal_mlp` baseline and `scripts/mpc_controller.py`
- External grounding from platformer AI and RL literature:
  - classical forward-model platformer planners such as Mario A* agents
  - platform-graph / surface-graph pathfinding approaches for 2D platformers
  - Procgen-style results showing that architecture/curriculum/generalization matter more than extremely elaborate hand-designed shaping

## Executive Verdict

The plan is directionally good, but not yet sound end to end.

The strongest parts are:

- moving away from runtime surface-graph PBRS
- keeping the surface graph as an offline curriculum/demonstration tool
- using global structured observations rather than raw pixels
- treating A* shaping as an early bootstrap only
- removing frame stacking and recurrent state if the observation is truly Markov

The main problems are:

1. `NPP-14` is not mathematically or algorithmically sound as written.
2. The revised observation simplification in `NPP-37` / `NPP-38` drops important mechanics from `sim_mechanics_doc.md`.
3. `NPP-26` / `NPP-27` overclaim a "true regret oracle" on levels where the planner does not model the real mechanics.
4. Layer 3 is still too ambitious too early. A simpler benchmark-and-baseline track should come before a large synthetic generator + ACCEL pipeline.

My recommendation is:

- keep the overall direction
- rewrite the BSRS task
- narrow the observation redesign so it is simpler but still mechanically complete
- add first-class baseline and benchmark issues
- reduce Layer 3 scope until Layer 1/2 beat simpler baselines on real mechanic buckets

If those changes are made, the plan can produce a good agent. If not, there is a real risk of building an elegant but weak stack.

## Scorecard

| Dimension | Verdict | Why |
|---|---|---|
| Accuracy | Mixed | The broad strategy is plausible, but several issue details do not match the mechanics or the cited theory. |
| Coverage | Incomplete | Some critical N++ mechanics are missing or underrepresented in the revised observation plan. |
| Correctness | Weak in a few key places | BSRS and FiGAR interaction are underspecified; "true regret" is overstated. |
| Simplicity | Better than the original epic, still too ambitious | The move from entity attention to rasterization is good, but Layer 3 and the universal hazard map still overshoot what is needed first. |

## Ground Truth From The Sim

The most important constraints from `sim_mechanics_doc.md` are:

- N++ is a deterministic, frame-sensitive physics platformer with only 6 primitive actions but very high timing sensitivity.
- Momentum matters. Wide gaps often require speed buildup away from the goal before a successful jump.
- The state is close to Markov if and only if the observation exposes the true physics state and all important dynamic entity state.
- Mechanics that matter are not just mines and geometry. They include:
  - jump/floor/wall/launch-pad buffers
  - wall interaction and wall-jump timing
  - slopes
  - toggle mine state
  - launch pads
  - bounce blocks
  - one-way platforms
  - doors
  - thwumps
  - drones
  - death balls

That means any "simple" plan still has to preserve enough information for these mechanics, or the agent is not actually seeing an MDP.

## What Is Sound

### 1. Runtime A* should stay simple

Using 6px A* HWM as an early bootstrap is reasonable. The strategy doc is right that geometric distance is useful on easy locomotion levels and not worth turning into a giant runtime planner.

This is also consistent with platformer AI practice: classical forward-model agents often use geometric or graph guidance to bootstrap action choice, then rely on simulation or policy learning for the hard timing.

### 2. Surface graph should be offline-only

Keeping the velocity-augmented surface graph for:

- offline expert routes
- curriculum scoring
- optional demonstrations

is a good trade. It preserves the value of the existing planner without forcing it into the hot loop.

### 3. Removing entity attention is probably the right simplification

Canceling `NPP-11` was a good move. For this domain, most hazards are fundamentally spatial. A rasterized representation is more natural than an O(n^2) attention stack and scales much better to mine-heavy levels.

### 4. Feedforward is probably enough

`NPP-21` is directionally correct. Given explicit velocity, jump phase, contact normals, and buffer state, this should be a feedforward problem. That is much simpler than carrying recurrent state.

But this is only true if the observation really includes all important dynamic entity state. Right now that is not yet true.

## Major Findings

### High: `NPP-14` is inconsistent as written

The ticket says:

- `gamma = 0.99`
- `eta_bsrs = 1.0`
- cites a convergence bound from SLOPE: `eta < (1-gamma)/(1+gamma)`

Those numbers do not match. With `gamma = 0.99`, that bound is about `0.005`, not `1.0`.

So at least one of these must change:

- the claimed theorem should be removed from the ticket
- `eta_bsrs` should be drastically reduced
- or the implementation should be explicitly framed as a heuristic, not a theoretically justified setting

This is the clearest correctness problem in the current plan.

### High: BSRS is underspecified for FiGAR / SMDP transitions

The current BSRS formula uses:

`gamma * V(s') - V(s)`

But the plan is built around FiGAR action-duration outputs. If the policy acts in macro-steps of duration `k`, then the correct SMDP-style shaping term at decision boundaries is duration-aware, i.e. it depends on `gamma^k`, not just `gamma`.

If shaping is applied per primitive frame, the ticket should say that. If it is applied per FiGAR decision, the current formula is wrong or at least incomplete.

This needs to be resolved before implementation.

### High: the revised observation plan dropped real mechanics

The revised Layer 1 is simpler than the epic, but it drops too much.

Problems in `NPP-37` / `NPP-38`:

- doors are missing
  - `sim_mechanics_doc.md` includes regular doors, locked doors, and trap doors
  - the revised observation plan does not clearly encode them
- safe toggle mines are missing
  - the old mine map had a safe-mine channel
  - the revised plan removes safe mines from the hazard representation
  - that loses information about a real interactive mechanic
- one-way platform directionality is underspecified
  - `dynamic_surface` alone is not enough
  - one-way behavior depends on approach direction
- bounce blocks are treated as plain "dynamic surface"
  - that is probably too lossy for a spring-like object with distinct contact dynamics
- the ticket mentions moving platforms and laser drones
  - these are not grounded in `sim_mechanics_doc.md`
  - they also do not appear to be first-class mechanics in the current codebase search

So the revised observation plan is simpler, but not yet mechanically complete.

### High: "true regret oracle" is overstated

`NPP-26` and `NPP-27` call the planner-based expert gap a true regret oracle. That is only true on the subset of levels where the offline planner correctly models the real transition dynamics.

It does not appear true for:

- drones
- death balls
- thwumps with reactive triggering
- doors with timing/state
- any mechanic not represented in the planner

For those levels, this is not a true regret oracle. It is at best:

- an approximate expert gap
- or a supported-mechanics expert gap

That distinction matters because otherwise the curriculum will be optimized against a false teacher signal.

### Medium: A* reachability is too weak as the main validator for generated levels

`NPP-25` and `NPP-29` validate generated or mutated levels with 6px A* reachability.

That is okay for Phase 1 locomotion-only levels.

It is not enough for levels where success depends on:

- speed buildup
- wall-jumps
- launch pads
- one-way platforms
- doors
- dynamic hazards

So the validator is fine for a small templated early curriculum, but not as the main correctness gate for a large synthetic Phase 2/3 generator.

### Medium: the current plan lacks a first-class benchmark/baseline track

This is the biggest planning gap from a "will this produce a good agent?" perspective.

The project should have explicit issues for:

- a minimal RL baseline
- a non-ML baseline
- a mechanic-bucket benchmark suite on real levels

This matters because platformers are one of the domains where strong non-ML baselines are real. Classical forward-model A* planners have already won Mario-style platformer benchmarks, and platform-graph search is a standard approach for 2D platforming.

If a simple planner or MPC controller is already competitive on a subset of levels, that should shape the RL plan immediately.

### Medium: Layer 3 is too large too early

The current Layer 3 combines:

- synthetic level generation
- SFL / PLR
- expert route integration
- curriculum phases
- ACCEL mutations

That is too much before you have:

- a reliable benchmark
- a good baseline
- evidence that a simpler curated curriculum is insufficient

For N++, making good synthetic levels is itself a difficult mechanics problem. It should not be assumed to be cheaper than using a curated real level set plus a small number of templates.

### Medium: per-level A* should be cached globally, not recomputed on every load

`NPP-17` still implies A* is run at level load. For a reused pool, that work should be shared and cached per level, not repeated independently by every environment reset.

The right model is:

- lazy compute once per level
- store in a shared cache
- invalidate only on level mutation or a state-dependent topology change you truly care about

## Coverage Audit Against Mechanics

### Covered well

- global geometry via tile tensor
- switch and exit phase split
- explicit velocity and local physics state in `game_state`
- A* as a bootstrap rather than a full dynamics oracle
- wall-jump / speed-buildup learnability as a policy/value problem rather than a reward-engineering problem

### Missing or too weak

- door state and door timing
- safe toggle mine presence
- one-way platform orientation
- bounce block state richness
- reactive entity state needed to make the observation truly Markov
- slope-specific validation
- buffer/timing-specific validation
- real-level benchmark buckets for each mechanic family

## Overengineering Audit

## Good simplifications already made

- `NPP-11` canceled
- surface graph moved offline
- feedforward policy instead of recurrence
- fewer observation streams

## Still overengineered

### 1. Universal hazard/surface map as a single giant L1 blocker

The revised map is appealing, but it tries to solve too much at once:

- all hazard types
- all dynamic surfaces
- velocity fields
- incremental Rust updates
- caching logic
- unlimited entity counts

A simpler sequence is better:

1. Keep mines and clearly deadly moving hazards in the raster map.
2. Keep a very small structured channel for mechanics that are not well represented as generic hazard fields.
3. Add dynamic surfaces only after proving they matter on benchmark levels.

### 2. Synthetic generator + ACCEL too early

This should be cut down to:

- a tiny templated Phase 1 generator
- curated real levels for everything else

Only add ACCEL after there is evidence that the simpler curriculum saturates.

### 3. Future layers are correctly marked as future, but should stay that way

`NPP-33` through `NPP-36` are fine as parked ideas. They should not be pulled forward until the base stack is already competitive:

- DHRL
- entity-aware graph extensions for more dynamic entities
- exploration bonuses
- quasimetric potentials

These are all "optimize success after success exists" features, not "get to first good agent" features.

## Recommended Issue-Level Changes

| Issue | Recommendation | Why |
|---|---|---|
| `NPP-5` | Keep, but narrow and resequence | Good top-level direction; too much hidden risk inside revised Layer 1. |
| `NPP-13` | Keep | A* HWM bootstrap is reasonable. |
| `NPP-14` | Rewrite before implementation | Current theorem/hyperparameter story is inconsistent; FiGAR interaction is underspecified. |
| `NPP-17` | Keep, add shared cache requirement | Do not recompute A* maps per reset if the same level is reused. |
| `NPP-18` | Expand | Add doors, one-way platforms, slopes, buffer timing, and baseline comparisons. |
| `NPP-19` | Keep, but non-gating | Plain CNN baseline should exist before this becomes the required path. |
| `NPP-20` | Keep only if observation redesign survives audit | Useful if the hazard map stays; otherwise this may change shape. |
| `NPP-21` | Keep with caveat | Feedforward is right only after observation completeness is fixed. |
| `NPP-22` | Keep, but lower priority than correctness/benchmark work | Decoupling is sensible but not the first bottleneck. |
| `NPP-23` | Keep | Fine once Layer 2 exists. |
| `NPP-24` | Expand | Validate behavior, not just value ordering. |
| `NPP-25` | Shrink | Make this a Phase 1 template generator only. Do not promise broad synthetic coverage yet. |
| `NPP-26` | Keep, rename expert metric | Do not call it true regret unless planner support is complete. |
| `NPP-27` | Keep, but scope to supported mechanics | Strong for static/simple subsets; approximate elsewhere. |
| `NPP-28` | Keep | Phase structure is fine. Gates should be benchmark-driven. |
| `NPP-29` | Defer | ACCEL is too early. |
| `NPP-30` | Keep | Cleanup is valuable. |
| `NPP-31` | Keep | Offline-only contract is exactly right. |
| `NPP-32` | Keep as optional | Good optional upgrade for supported entity-rich expert routes. |
| `NPP-37` | Split into smaller issues | Current scope is too broad and drops important semantics. |
| `NPP-38` | Modify | Add back missing semantic channels or a small structured supplement. |
| `NPP-33` to `NPP-36` | Leave deferred | Not needed for first strong agent. |

## Issues That Should Be Added

### 1. Benchmark suite over real mechanic buckets

Add a new issue for a held-out benchmark suite organized by mechanic family:

- flat run / basic jump
- wall-jump
- slopes
- dense mines
- launch pads
- one-way platforms
- doors
- thwumps
- drones
- death balls

Success should be measured on these buckets, not just on synthetic aggregate metrics.

### 2. Baseline track

Add a new issue for baseline agents:

- minimal RL baseline using the existing `minimal_mlp`
- non-ML baseline using the existing `scripts/mpc_controller.py`
- optional direct A* / heuristic controller

This is the fastest way to learn whether the problem is actually architecture-limited or mostly a planning/evaluation problem.

### 3. Demonstration / BC track

The current plan says BC pretraining is optional, but the repo already has BC machinery and the offline planner already exists.

Add a first-class issue to:

- generate demonstrations on supported levels
- pretrain the policy
- use BC or DAPG as a sample-efficiency accelerator

That is simpler and more defensible than jumping quickly to exploration bonuses or quasimetric research.

### 4. Observation completeness audit

Add an issue specifically to verify that the final observation exposes every dynamic variable required for a Markov policy:

- door state
- toggle mine state
- one-way direction
- thwump state
- any timed entity state that affects next-state dynamics

Without this, `NPP-21` rests on a claim that has not been proved.

## Recommended Simpler Plan

If the goal is "a good agent as fast as possible", this is the sequence I would use.

### Stage 0: Make success measurable

- build the real-level benchmark suite by mechanic family
- wire in baseline evaluation

### Stage 1: Prove simple baselines

- run the existing `minimal_mlp`
- run the existing MPC baseline
- add a very simple heuristic baseline if useful

If these are already strong on some buckets, use them immediately for demonstrations and sanity checks.

### Stage 2: Build the smallest credible learned agent

- tile grid
- mechanically complete game state
- simple hazard representation
- A* HWM bootstrap
- no recurrence
- no expert-route coupling in runtime

This is where the first strong RL agent should come from.

### Stage 3: Add BSRS carefully

- use a detached or lagged value estimate
- make the shaping duration-aware for FiGAR
- keep the BSRS weight small at first
- treat it as an empirical improvement, not a theorem-backed free win

### Stage 4: Upgrade architecture only if needed

- plain CNN baseline first
- then Impoola + FiLM if the simpler model saturates
- keep policy/value decoupling if it gives measurable gains

### Stage 5: Use curriculum conservatively

- curated real levels
- tiny Phase 1 templates
- PLR/SFL
- offline expert routes only on supported subsets

### Stage 6: Only then consider bigger research layers

- ACCEL
- broader synthetic generation
- DHRL
- exploration bonuses
- quasimetric potentials

## Final Recommendation

The project should not be scrapped. The core strategic direction is good.

But the current Linear breakdown should be edited before more implementation work proceeds:

- fix `NPP-14`
- narrow `NPP-37`
- repair the observation coverage holes
- stop calling planner-based regret "true" on unsupported mechanics
- add a benchmark/baseline track
- reduce Layer 3 to a smaller, safer first version

With those changes, the plan is sound enough to produce a good agent.

Without those changes, the most likely failure mode is not "the ideas are bad." It is "the project spends too much effort on elegant infrastructure before proving the simple path."
