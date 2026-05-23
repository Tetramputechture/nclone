# N++ Generalizable Agent: Strategy Epic

**Document Version:** 2.1  
**Date:** April 21, 2026  
**Scope:** Observation architecture, reward pipeline, policy architecture, and curriculum system for training a generalizable N++ RL agent.  
**Key change from v2.0:** Layer 1 narrowed to **toggle mines + exit switch + exit door only**. Non-mine entities (bounce blocks, thwumps, launch pads, locked doors, drones, death balls, one-way platforms) are deferred. The observation space and model are sized accordingly — see *Layer 1 implementation reality* immediately below before reading Section 3, which preserves the original long-term design.

---

## 1a. Layer 1 implementation reality (April 2026)

The long-form design in Section 3 describes the eventual target. The live code in `nclone` + `npp-rl` implements a strict subset of it, scoped to toggle mines / exit switch / exit door:

| Key | Shape | Source |
|---|---|---|
| `tile_grid` | `(42, 23)` int8, values 0–33 | `base_environment._get_obs` |
| `entity_overlays` | `(5, 42, 23)` float32 | `entity_overlay_utils.compute_entity_overlays` |
| `hazard_map` | `(1, 168, 92)` float32 | `MineThreatMap` (Rust) — deadly-mine SDF only |
| `game_state` | `(28,)` float32 | `observation_processor` |

Active entity overlay channels (5): ninja position splat, switch-not-collected indicator at switch tile, exit-door tile, deadly-mine density, safe-mine density. Mines are split by live `state` (0 = deadly, 1 = safe) via `classify_mines_by_state`, so the hazard map and overlay never disagree.

Deferred — **not** in the observation, the model, or the reward:

- 3-channel mine threat map (`safe_sdf`, `passability`); only the 1-channel deadly SDF (`hazard_map`) is live.
- Entity attention (`entity_list`, `entity_mask`) — non-mine entities are out of scope.
- Bounce-block, thwump, launch-pad overlays — out of scope.
- Locked-door surface nodes and locked-door-switch goals — explicitly excluded from `extract_goal_positions` and `observation_processor`.
- Recurrent policies (LSTM / GRU / Mamba) — purged; policy is feedforward.

Reward pipeline actually shipped (see `main_reward_calculator.py`):

```
r = r_sparse
  + beta * alpha_astar * hwm_progress              # NPP-42 HWM on A* cost-to-go
  + time_penalty_per_step * frames                 # TIME_PENALTY_PER_STEP = -0.001
  + BACKTRACK_PENALTY * frames [if d(goal) ↑]      # = -0.0004
  # BSRS rollout term (1 - beta) * eta * (gamma * V(s') - V(s)) is added on the
  # training side in FiGARPPO / MaskedPPO, not inside the env reward calculator.
```

Terminals add `R_SWITCH`, `R_EXIT`, `R_DEATH` (+ `R_DEATH_MINE`), or `TRUNCATION_PENALTY`. `beta` anneals 1 → 0 over `bsrs_blend_horizon_fraction * total_timesteps` (see `RewardConfig.beta_blend`).

**Single runtime reward path (NPP-30 / NPP-31 closed, April 2026).** The primitive-frame reward loop, the precomputed-PBRS branch, `PrimitiveRewardCalculator`, `pbrs_potentials.py`, `simulation_pbrs/`, and the runtime route overlay are deleted. Every `env.step()` calls `RewardCalculator.calculate_reward` exactly once. `nclone/value_function/` is retained strictly as offline curriculum tooling (route extraction, surface graph, expert-route visualization). See `nclone/gym_environment/reward_calculation/README.md` for the full contract.

Everything below this block is preserved as the multi-layer strategic target. Layer 1 acceptance is defined by the table above, not by the Section 3 tables.

---

## 1. Vision and Core Insight

The agent is trained with a **three-component reward signal: 6px A\* geometric distance (HWM-style) + Bootstrapped Reward Shaping (BSRS) + sparse terminals.** The A\* HWM provides a fast, mine-aware directional signal that bootstraps early training on simple levels where geometric distance IS the physics-optimal path. BSRS uses the agent's own evolving value function V\_θ as a dynamic potential, taking over from the external signal after ~40% of training as V\_θ learns actual physics. The architecture (Impoola-CNN + FiLM), multi-level training (500+ levels), and curriculum (SFL scoring, ACCEL mutations) do the heavy lifting; the reward function's job is simply to **not be misleading on easy levels** — and geometric A\* distance is not misleading on easy levels.

**The velocity-augmented surface graph is retained as an offline curriculum tool** for computing frame-optimal expert routes on levels selected by the curriculum. These expert routes serve as the regret oracle for PLR scoring (expert\_return − agent\_return) and as demonstration data for optional behavioral cloning pre-training. The surface graph does **not** contribute to per-frame reward computation or observation channels.

**Key departures from previous strategy:**

- **Surface graph PBRS dropped from per-frame reward.** The velocity-augmented Dijkstra cost-to-go was previously proposed as the per-frame PBRS potential. This role is replaced by simple 6px A\* HWM (early training bootstrap) + BSRS (after value function matures). The surface graph is retained as an offline curriculum tool only.
- **Route overlays and waypoint vector dropped from observations.** Spatial overlay channels derived from the surface graph (cost-to-go field, direction field, route mask, velocity targets) and the Dijkstra waypoint vector are removed from the observation space. The CNN receives tile embeddings, entity overlays, and the mine threat map. The architecture learns routing from tile geometry and experience.
- **SAC/RFCL replaced by FiGAR-PPO.** The training algorithm is FiGAR-PPO with dual (action, duration) heads and sparse terminal rewards. SAC-based approaches are archived.
- **Flat observation vector replaced by structured multi-modal input.** The 153-dim flat vector is replaced by tile embeddings, a mine threat map, entity attention, and a game-state vector — each processed by a stream suited to its structure.

---

## 2. Research Foundation

### 2.1 Simple Distance Shaping: Why A\* HWM Works for Early Training

**The reward function's job is to bootstrap learning on easy levels, not to be physics-perfect.** The curriculum ensures the agent faces trivial levels first (flat corridors, small gaps, simple jumps) before encountering complex momentum puzzles. On trivial levels, geometric distance IS the physics-optimal path — A\* distance to the goal is not misleading. The agent learns basic locomotion. The value function begins learning that speed is useful. BSRS gradually takes over as V\_θ matures.

**Ranked impact on agent performance:**

1. **Curriculum design** (which levels, in what order) — the biggest lever
2. **Architecture** (Impoola-CNN + FiLM) — learns physics from experience
3. **Multi-level training** (500+ levels) — prevents memorization, forces generalizable representations
4. **BSRS** (value function as dynamic shaping) — adapts to true dynamics, handles momentum
5. **Go-Explore** (cross-episode memory) — discovers non-obvious routes
6. **Simple distance metric** (6px A\* HWM) — bootstraps early training

The surface graph would address item 6 at 100× the engineering complexity. Items 1–5 handle physics-awareness through learning rather than engineering.

**Why geometric distance is sufficient for early training.** The curriculum phases difficulty:
- **Phase 1 (trivial levels):** Small gaps, flat floors. A\* distance IS the optimal path. The agent learns basic locomotion — running, jumping, landing. The value function learns "speed is good." A\* HWM works perfectly.
- **Phase 2 (moderate levels):** Larger gaps requiring speed buildup. A\* points toward the gap; the agent approaches at low speed, fails, accidentally builds more speed, crosses. The value function begins learning "approaching gaps with speed is better." BSRS now provides physics-aware shaping because V\_θ has learned the speed-gap relationship. A\* becomes less important.
- **Phase 3 (hard levels):** Complex routing, backtracking, entity interactions. BSRS dominates. A\* provides only a coarse directional hint. The value function handles the rest.

**The 6px A\* graph is the right choice for the distance metric.** The existing `graph_builder.py` and `path_distance_calculator.py` provide 6px resolution (vs. 24px for tile BFS), diagonal/air connectivity, and mine-aware routing. It computes in under a second per level and the code already exists. Use HWM-style (non-negative progress only — reward only when achieving a new minimum distance to goal) to avoid penalizing beneficial retreat.

| Metric | Tile BFS (42×23) | 6px A\* (existing) | Velocity-augmented surface graph |
|---|---|---|---|
| Computation | ~10μs | ~100ms | ~60–120 seconds |
| Code complexity | ~30 lines | ~500 lines (existing) | ~6,000 lines |
| Mine-aware | Easy (recompute) | Easy (existing) | Hard (entity extensions) |
| Physics-aware | No | No | Yes |
| Accurate on easy levels | Yes | Yes | Yes |
| Accurate on hard levels | No | No | Yes |
| Needed after BSRS takes over | No | No | No |

**PBRS theory grounding for BSRS.** Grzes & Kudenko (2009) and Canonaco et al. (2024) prove that Φ = V\* is the optimal potential function — when the potential equals the true value function, the shaped MDP has zero incremental reward along optimal trajectories. BSRS implements exactly this: as V\_θ converges toward the true value function through experience, the BSRS potential approaches V\* without any precomputed graph. The agent learns the physics; the value function encodes it.

---

### 2.2 Why BSRS Handles What Geometric Distance Cannot

**HWM on geometric distance has one known limitation: it cannot account for momentum.** For a wide gap, the agent must build speed to the left before jumping right. A\* HWM rewards moving toward the gap at any speed — it does not distinguish arriving at the ledge fast vs. slow. Randløv & Alstrøm's bicycle agent learned to ride in circles precisely because non-potential distance rewards created exploitable loops.

**BSRS resolves this through learning rather than precomputation.** As the agent encounters momentum-dependent routes (Phase 2+ of curriculum), V\_θ learns that "at this ledge position with low horizontal velocity, the future return is low." The BSRS term γV\_θ(s') − V\_θ(s) becomes negative when the agent is at the ledge with insufficient speed — exactly the physics-correct signal. Crucially, BSRS also values states where the agent has retreated to build speed: V\_θ(farther back but fast) > V\_θ(closer but slow), producing positive shaping for building momentum. This is the signal that A\* HWM alone cannot provide and that a precomputed surface graph would provide through engineering — but BSRS provides it through experience.

**The transition is guaranteed by curriculum design.** The agent never faces momentum-critical levels until after extensive training on simpler levels. By the time momentum matters, ~40% of training has elapsed and V\_θ has learned meaningful physics correlations. BSRS has already taken partial control of the shaping signal.

**When geometric distance is genuinely misleading.** Two practical failure modes exist for A\* HWM in N++:
1. **Gap with required momentum buildup:** A\* says "move right toward gap" but physics requires left-first sprint. Handled by BSRS as described above.
2. **Entity-dependent routes:** A\* may propose a path blocked without a bounce block or launch pad. The entity attention module gives the CNN visibility into these entities; V\_θ learns entity-dependent value through experience.

**Low-cost mitigations if needed.** BiPaRS (Hu et al., NeurIPS 2020) learns a state-dependent weighting function that can down-weight the A\* HWM signal near chasms and entity-dependent areas. Müller & Kudenko (AAMAS 2025) show a constant potential bias can correct scale mismatches between the external signal and Q-initialization.

---

### 2.3 Bootstrapped Reward Shaping (BSRS) for Dynamic Blending

**BSRS eliminates dependence on any external heuristic.** Bootstrapped Reward Shaping (Adamczyk et al., AAAI 2025) uses the agent's own evolving value function estimate V\_θ(s) as the PBRS potential, producing shaped reward F(s,s') = γV\_θ(s') − V\_θ(s). The key theoretical result: BSRS with shape-scale η is equivalent to rescaling the learning rate by (1+η) and rescaling the reward — it amplifies the TD learning signal without requiring any domain knowledge. BSRS requires changing only a single line of code and was validated across the Atari suite.

**SLOPE** (Li et al., 2026) demonstrated that bootstrapped potential construction outperforms TD-MPC2, DreamerV3, and SAC across 30+ manipulation tasks, with convergence guaranteed when η < (1−γ)/(1+γ).

**The recommended hybrid approach:** Initialize with 6px A\* HWM as the external distance signal (non-negative progress reward when achieving a new minimum distance to goal). This provides excellent early-training gradient on simple levels where geometric distance is accurate. As the value function matures (~40% through training), blend in BSRS which captures physics-aware and entity-aware dynamics that A\* HWM misses. After ~70% of training, BSRS dominates and the A\* HWM term contributes negligibly. This is implemented via `beta_blend` annealing from 1.0 → 0.0 (see Section 3.8 for implementation details).

---

### 2.4 Racing Line and Momentum Planning

**The optimal path through an N++ level is not the geometric shortest path but the time-optimal trajectory given momentum constraints — a "racing line" in the autonomous racing sense.** TU Munich's global racetrajectory optimization framework demonstrates the structural parallel: output is a sequence of waypoints with (position, heading, curvature, target\_velocity, target\_acceleration) — directly analogous to the (surface\_position, velocity\_bin) waypoint representation in the N++ surface graph.

**Kinodynamic planning provides theoretical grounding.** Webb & van den Berg's Kinodynamic RRT\* (2012) proves asymptotic optimality for systems with linear differential constraints. N++ physics approximate a double integrator with gravity and drag — the nilpotent case where closed-form optimal connections between states exist. However, the structured nature of platformer levels (discrete surfaces, known geometry) makes the graph-based approach more efficient than sampling-based methods: the ~45K-node surface graph exhaustively covers reachable states.

**Legged locomotion validates the momentum-planning paradigm.** Cheng et al.'s Extreme Parkour (ICRA 2024) demonstrated RL policies for quadruped parkour that must plan momentum generation before obstacles. The key insight: parkour requires operating at actuator limits and planning momentum buildup well before the maneuver — directly paralleling N++ where the agent must start accelerating many frames before a critical jump. DTC (Deep Tracking Control, Science Robotics 2023) takes the hybrid approach: trajectory optimization provides footholds while RL provides robustness — the same architecture proposed here.

**Tool-assisted speedrun optimization is the closest direct analog.** TAS methodology faces the same exponential input space — even a 4-second track has ~440 input slots with 4^439 combinations. The velocity-augmented surface graph compresses this exponential space into a tractable ~45K-node graph by exploiting surface geometry and velocity discretization.

---

### 2.5 Architecture: Impoola-CNN + FiLM + Global Average Pooling

**Upgrading from simple CNN to IMPALA-CNN is essential.** Cobbe et al. (ICML 2020) showed that Nature-CNN "almost completely fails to train" on Procgen while IMPALA-CNN (15-layer residual network) significantly improves both sample efficiency and generalization.

**Impoola-CNN** (Janer et al., RLC 2025) improves on IMPALA-CNN by replacing the flatten+linear layer with Global Average Pooling, achieving 17% higher generalization with 35% fewer parameters on Procgen. GAP reduces translation sensitivity — an essential inductive bias for N++'s global (non-ego-centric) view, teaching the network to recognize "there's a gap ahead" rather than "there's a gap at pixel (120,50)." Impoola-CNN's biggest gains occurred precisely in environments without agent-centered observations, matching N++'s architecture.

**FiLM conditioning** (Perez et al., AAAI 2018) was validated for physics-conditioned RL in quadrotor control (Bauersfeld et al., ICRA 2023), where a single FiLM-conditioned policy achieved within 0.6–2% of per-setting-tuned policies across the entire flight envelope. This directly applies to N++: at high velocity, distant obstacles and walls become more salient because the agent cannot stop quickly; at low velocity, nearby platforms matter more for precision jumps. The architecture pattern π₂(FiLM(π₁(observation), physics\_state)) maps directly to N++.

**Late policy-value decoupling** (IDAAC, Raileanu & Fergus, ICML 2021) provides a crucial insight: the value function requires more level-specific information than the policy, and shared representations cause the policy to overfit to value-function-specific features. When training on a single level, V(s) degenerates into an A\* distance lookup for that geometry. Separate heads with a shared encoder correctly captures this asymmetry.

---

### 2.6 Learnability-Scored Curriculum (SFL + PLR + ACCEL)

**Sampling For Learnability** (Rutherford et al., NeurIPS 2024) is the current best curriculum approach, scoring levels by p·(1−p) where p is the agent's success rate. This paper showed that existing regret approximations used by PLR and ACCEL (Positive Value Loss, MaxMC) do not actually correlate with regret but instead with success rate, causing wasted training on already-solved levels. SFL significantly outperforms PLR, ACCEL, and domain randomization on MiniGrid, XLand-MiniGrid, and JaxNav benchmarks.

**SFL's learnability scoring naturally avoids three failure modes** for N++: levels where A\* shaping is accurate and the agent already succeeds (p ≈ 1, low learnability), levels where A\* shaping is completely misleading and the agent always fails (p ≈ 0, low learnability), and levels at the agent's frontier where training is most productive (p ≈ 0.5, maximum learnability).

**True regret oracle.** The availability of computed expert routes in N++ provides an additional advantage: true regret can be computed as expert\_return − agent\_return, bypassing the flawed approximations that SFL was designed to address. The "No Regrets" paper showed that oracle regret is the single most important factor in curriculum effectiveness.

**ACCEL** (Parker-Holder et al., ICML 2022) extends PLR with evolutionary level mutations. For N++'s 42×23 tile grid, mutations (adding/removing platforms, adjusting gap widths, repositioning hazards) are natural operations that create an expanding frontier of difficulty.

**Multi-level training forces physics-aware generalization.** Cobbe et al. (2019, 2020) showed that agents trained on fewer than 100 levels memorize level-specific trajectories, while 500+ levels force the value function to learn generalizable features. FiLM conditioning amplifies this effect: the same geometry has different values depending on physics state, making per-level memorization impossible.

---

### 2.7 DHRL Hierarchy and Timed Subgoals (Future Layer)

**DHRL** (Lee et al., NeurIPS 2022 Oral) provides the most directly applicable hierarchical architecture: a three-level hierarchy where a graph-level planner runs Dijkstra to produce waypoints, and a low-level policy executes primitive actions to reach each waypoint. For N++, the precomputed graph can replace the high-level policy entirely for entity-free navigation — the Dijkstra optimal path IS the optimal subgoal sequence.

**Timed subgoals** (HiTS, Gürtler et al., NeurIPS 2021) should be used rather than fixed intervals. In momentum-based domains, fixed k-step subgoal intervals poorly match variable-duration physics maneuvers — a wall-jump takes ~20 frames while a long arc takes ~60. HiTS's timed subgoals emit (target\_node, time\_budget) pairs, giving the low-level policy a precise temporal constraint aligning with FiGAR's variable-duration macro-actions.

**The recommended three-level architecture (future):**
- **Level 3 (Entity-aware replanner):** Modifies graph edge costs based on entity positions, runs Dijkstra to produce waypoint sequence, replans on failure or entity state change.
- **Level 2 (Waypoint tracker):** Selects next waypoint from path, emits (target\_node, time\_budget) timed subgoals, monitors progress, advances or aborts based on landing/timeout/infeasibility events.
- **Level 1 (FiGAR-PPO executor):** Receives subgoal (target position + required landing velocity), outputs (action, duration) pairs, terminates on surface landing, timeout, or momentum loss.

**Replanning is trivially cheap.** The key insight is that backward Dijkstra already computes cost-to-go at ALL ~45K nodes. When the agent deviates from the planned route, it simply queries the nearest graph node for the optimal action from its current state — O(1) replanning. Hatch & Boots (2023) formalize this: a planning tree with cost-to-go annotations serves as a terminal cost function, providing graceful degradation with zero replanning latency.

---

### 2.8 Entity-Aware Graph Extensions

**The recommended approach is tiered: bake static entities into the graph, handle cyclic entities with time-expanded subgraphs, and defer reactive entities to the RL agent.**

**Static entities (bounce blocks, one-way platforms, launch pads) should be integrated immediately.** These add ~1–5K nodes and require only extending the surface extraction phase. Bounce blocks become new surface nodes with velocity-reflection edges. Launch pads become deterministic trajectory edges. One-way platforms are surface nodes with directional edge constraints.

**Moving platforms require SIPP-inspired time-expanded subgraphs.** Safe Interval Path Planning (Phillips & Likhachev, CMU 2011) is the gold standard: rather than expanding every timestep, group contiguous collision-free timesteps into "safe intervals." For cyclic platforms with period T discretized into K=10 phase bins, create boarding, riding, and dismounting edges.

**Thwumps are the hardest case.** A thwump's position depends on the agent's path (it triggers when approached), but the optimal path depends on the thwump's position. The practical solution is spatial independence factoring: most thwumps have non-overlapping zones of influence. For ≤3 interacting thwumps, enumerate the 4–8 combined states. For complex interactions, defer to the RL agent with the graph providing a pessimistic plan.

| Entity type | Graph treatment | Additional nodes | Precompute time |
|---|---|---|---|
| Bounce blocks | Static surface nodes + reflection edges | +500–2K | +10–20 sec |
| One-way platforms | Directional surface nodes | +100–500 | +5–10 sec |
| Launch pads | Deterministic trajectory edges | +100–500 | +5–15 sec |
| Moving platforms | SIPP time-expanded subgraphs | +500–2K | +20–60 sec |
| Thwumps (≤3 interacting) | Factored conditional edges | +200–1K per state | +30–120 sec |
| Thwumps (complex) | Deferred to RL agent | 0 | 0 |

---

### 2.9 Exploration Bonuses (Future Layer)

**Episodic and global exploration bonuses should be combined multiplicatively.** Henaff et al. (ICML 2023) established that the combination of episodic and global intrinsic bonuses multiplicatively produces the most robust exploration in procedurally generated environments.

**E3B** (Henaff et al., NeurIPS 2022) provides elliptical episodic bonuses in a learned embedding space using inverse dynamics models — superior to count-based methods in procedural environments. **NovelD** (Zhang et al., NeurIPS 2021) focuses reward on frontier states at the boundary between explored and unexplored regions.

**Distributional RND** (DRND, Yang et al., ICML 2024) improves on standard RND by distilling a distribution of random networks, providing more discriminative bonuses that better distinguish truly novel states from merely different-looking states across levels.

**DuRND** (Ma et al., ICML 2025) is particularly relevant, using dual RND modules — one updated on successful trajectory states, one on failed trajectory states — to jointly produce novelty reward and contribution reward assessing progress toward task completion.

**ACWI** (2025) uses a Beta Network to predict intrinsic reward weight β(s) based on correlation with future extrinsic returns — allowing high exploration when far from the goal and low exploration near the goal.

---

### 2.10 Quasimetric Learned Potentials (Future Layer)

**QRL** (Wang et al., ICML 2023) learns a quasimetric distance function d\_θ(s,g) that enforces the triangle inequality as an architectural constraint while allowing asymmetry (d(A,B) ≠ d(B,A)). This is precisely the structure needed for N++: jumping up costs more than falling down, and momentum-dependent traversal makes distance asymmetric. QRL achieved 4.9× better sample efficiency on robotics tasks and was validated on MountainCar — a physics-asymmetric domain analogous to N++.

**QPRL** (Hossain & Roy, ICML 2025) decomposes asymmetric traversal costs into a path-independent potential Φ (capturing height and position relative to the goal, analogous to A\* distance) and a path-dependent residual Ψ (capturing irreversible, momentum-dependent costs). This Φ/Ψ decomposition maps perfectly to N++: Φ encodes static geometry cost while Ψ encodes whether the agent can physically traverse a gap given current velocity.

**Temporal Metric Distillation** (TMD, Myers et al., NeurIPS 2025) unified contrastive and quasimetric approaches, enabling training from suboptimal offline data with stitching capabilities — critical since N++ demonstration routes may not be perfectly optimal.

**Practical pipeline:** (1) Collect demonstration trajectories across diverse synthetic levels, (2) Pre-train d\_θ(s,g) offline using QRL or TMD, (3) Use Φ(s) = −d\_θ(s, g\_current) as the PBRS potential, switching g from switch location to exit door upon collection.

---

## 3. Layer 1: Structured Observations + A\* HWM + BSRS + Multi-Level Training

### 3.1 Structured Tile Tensor

#### Tile Type Encoding

Use a learned embedding per tile type (34 types, types 0–33):

```python
tile_embedding = nn.Embedding(num_embeddings=34, embedding_dim=8)

# Input: tile_grid (batch, 42, 23), dtype=int64, values in [0, 33]
embedded = tile_embedding(tile_grid)           # (batch, 42, 23, 8)
embedded = embedded.permute(0, 3, 1, 2)        # (batch, 8, 42, 23)
```

**Observation space:** `'tile_grid': Box(low=0, high=33, shape=(42, 23), dtype=np.int8)`  
**Rollout storage:** int8, 966 bytes/step. Embedding applied on-the-fly during training.  
**Source:** Raw tile type array from `nplay_headless` / Rust simulator. Computed once at level load.

#### Entity Overlay Channels (8 channels on 42×23 grid)

| Channel | Content | Source |
|---------|---------|--------|
| `ninja_pos` | Bilinear splat at ninja's tile-space position | `ninja.xpos / 24, ninja.ypos / 24` |
| `switch_pos` | 1.0 at exit switch tile if not yet collected | Entity type 4 |
| `exit_pos` | 1.0 at exit door tile | Entity type 3 |
| `mine_deadly_density` | Count of deadly mines overlapping each tile, normalized /8 | Entity type 1/21, state==toggled |
| `mine_safe_density` | Count of safe mines overlapping each tile, normalized /8 | Entity type 1/21, state==untoggled |
| `bounce_block` | 1.0 at each bounce block's tile | Entity type 17 |
| `thwump` | 1.0 at each thwump's tile | Entity type 20 |
| `launch_pad` | 1.0 at each launch pad's tile | Entity type 10 |

A tile "overlaps" a mine's kill zone if any point in the tile is within 14px (ninja\_radius 10px + mine\_radius 4px) of the mine center. The density channels replace the previous binary mine channels — with 256 mines, a single tile can be covered by multiple overlapping kill zones, and the density conveys how heavily mined the area is.

**Observation space:** `'entity_overlays': Box(low=0.0, high=1.0, shape=(8, 42, 23), dtype=np.float32)`  
**Recomputed every frame** (mines toggle, thwumps move, ninja moves). Tile grid is static per level.

---

### 3.2 Mine Threat Map (High-Resolution SDF)

#### Rationale

The 42×23 tile grid (24px per cell) is too coarse to resolve mine passability. Two mines 20px apart (passable — the ninja fits between their kill zones) vs. 28px apart (impassable) differ by only ~0.33 tiles. A dedicated mine map at 6px resolution (matching the reachability graph) resolves the 14px kill zone as ~2.3 cells.

#### Channels (3 × 168 × 92)

| Channel | Content | Range |
|---------|---------|-------|
| `deadly_sdf` | Signed distance to nearest deadly mine, normalized | [-1, 1] |
| `safe_sdf` | Signed distance to nearest safe mine, normalized | [-1, 1] |
| `passability` | Soft passability score (sigmoid of deadly SDF) | [0, 1] |

**SDF computation:** `sdf = clip((min_distance_to_mine_center - 14px) / 24px, -1, 1)`. Positive outside kill zone, negative inside, zero at kill boundary. The SDF implicitly encodes relationships between mines — two close mines create a narrow positive corridor between negative regions.

**Passability:** `sigmoid((deadly_sdf_value × 24) / 3)` — smooth gradient from 1.0 (safe) to 0.0 (lethal), with transition width ~6px.

**Performance:** Vectorized NumPy over 256 mines × 15,456 cells: ~2ms. Too slow at 256 envs. **Implement in Rust** (`nclone-native`) using spatial hash pruning (only mines within 38px of a cell contribute). Target: ~50μs/env. Existing `mine_sdf.rs` provides the infrastructure pattern.

**Incremental updates:** Cache the threat map; recompute only on mine state change (toggle events are ~1–3 per episode). Full recomputation is rare.

**Observation space:** `'mine_threat_map': Box(low=-1.0, high=1.0, shape=(3, 168, 92), dtype=np.float32)`

**Rollout storage optimization:** Store only `mine_states` array (256 × 1 byte = 256 bytes) + mine positions (static per level, stored once). Recompute the 168×92 map during training forward passes. This avoids storing 185 KB/step.

---

### 3.3 Entity Attention Module (Non-Mine Entities Only)

Mines are handled spatially (overlays + threat map). The entity attention module processes entities with complex individual state:

**Entity types in the attention module:** bounce blocks, thwumps, launch pads, drones, death balls, doors, one-way platforms, exit switch, exit door. **Mines are excluded.**

**Max entities:** 48 (typical non-mine entity count is < 30).

**Per-entity feature vector (16 dims):** type\_onehot(8), rel\_x(1), rel\_y(1), distance\_norm(1), state\_code(1), orientation(2), velocity\_x(1), velocity\_y(1).

**Architecture:** `EntityAttention` module (2-layer self-attention, 64-dim embeddings, 2 heads, mean pool → 64-dim output). O(48²) = 2,304 attention operations — negligible.

**Observation space:**
```python
'entity_list': Box(low=-1.0, high=1.0, shape=(48, 16), dtype=np.float32)
'entity_mask': Box(low=0, high=1, shape=(48,), dtype=np.bool_)
```

---

### 3.4 Game State Vector (27 dims)

Keep the existing 27-dim `game_state` unchanged.

**Phase indicator:** `switch_activated` moves from the old reachability features into the game\_state vector as a 28th dim (or keep it in a separate 1-dim key).

---

### 3.5 Surface Graph: Offline Curriculum Tool

The velocity-augmented surface graph (existing `value_function/` pipeline) is retained exclusively as an **offline curriculum tool**. It is not used for per-frame reward or observations during rollout. Its outputs serve two purposes:

1. **Regret oracle for PLR curriculum scoring:** The Dijkstra-optimal path through the velocity-augmented graph gives the frame-optimal route, enabling true regret computation as `expert_return − agent_return`. This is the highest-quality curriculum signal available (see Section 2.6).
2. **Demonstration data for optional behavioral cloning pre-training:** Expert routes provide high-quality trajectory data for initializing the policy on simple maneuvers before RL begins.

**The surface graph runs offline on levels selected by the curriculum, not on every level in the pool and not on every training step.** It is invoked by `npp_rl/curriculum/expert_routes.py` when PLR promotes a level to the active pool or when a mutation produces a new level requiring expert route scoring.

#### Entity Surface Extensions (for Expert Route Quality)

For the expert routes to be accurate on entity-rich levels, the graph should include static entity surfaces. These extensions are **optional and incremental** — the base tile-only graph already produces valid expert routes for entity-free levels, which cover Phase 1 and most of Phase 2 of the curriculum.

**Bounce blocks (type 17):** Add surface segments to the graph for levels where the optimal route uses a bounce block. Estimated addition: **+500–2K nodes, +10–20 seconds** per level.

**Launch pads (type 10):** Simulate launch trajectories for nodes within contact range. Estimated addition: **+100–500 nodes, +5–15 seconds** per level.

**One-way platforms (type 11):** Directional surface nodes with edge constraints.

**Drones, death balls:** Not modeled in graph. Deferred to the RL agent.

#### Precomputation Budget (offline, on-demand)

| Component | Time (8 cores) |
|-----------|----------------|
| Surface extraction + velocity graph | ~40 sec |
| Entity surface extensions (optional) | ~30–60 sec |
| Dijkstra (×2 phases) | ~2 sec |
| Route extraction | ~3 sec |
| **Total per level** | **~1.5–2 minutes** |

This runs only for levels actively promoted by PLR, not for the entire level pool upfront.

#### Files (existing, used as curriculum tool)

- **`nclone/value_function/surface_extraction.py`**: Extend with entity surfaces for promoted levels.
- **`nclone/value_function/edge_enumeration.py`**: Use `tick()` near entities, `tick_tiles_only()` elsewhere.
- **`nclone/value_function/precompute_cli.py`**: Called by `expert_routes.py` on PLR-promoted levels.

---

### 3.6 Reward: A\* HWM + BSRS + Sparse Terminals

#### Rationale

The reward signal has three components: a non-negative geometric progress signal (A\* HWM) that bootstraps early training, a bootstrapped self-shaping signal (BSRS) that learns physics through experience and dominates after ~40% of training, and sparse terminal rewards. A time penalty discourages stalling.

- **A\* HWM (external bootstrap):** Uses the existing 6px A\* distance to the goal (switch, then exit). Reward is only positive, issued when the agent achieves a new minimum A\* distance (high-watermark). This avoids penalizing beneficial retreat while providing directional signal on easy levels.
- **BSRS (dynamic physics-aware shaping):** γV\_θ(s') − V\_θ(s) from the agent's own value head. Requires no precomputation. Captures momentum, entity interactions, and routing learned from experience. Grows in importance as V\_θ matures.
- **beta\_blend annealing:** Starts at 1.0 (pure A\* HWM), linearly decays to 0.0 (pure BSRS) over the first ~70% of training.

#### Implementation

```python
def compute_reward(
    r_sparse: float,          # +1 switch, +5 exit, -0.5 death
    astar_progress: float,    # HWM-style: positive delta only when new min A* dist achieved
    v_current: float,         # V_θ(s) from value head
    v_next: float,            # V_θ(s') from value head
    gamma: float = 0.99,
    alpha_astar: float = 1.0,     # A* HWM shaping weight
    eta_bsrs: float = 1.0,        # BSRS shape-scale
    beta_blend: float = 1.0,      # anneals from 1.0 → 0.0
    time_penalty: float = -0.002, # per-step cost
) -> float:
    r_astar = alpha_astar * astar_progress
    r_bsrs = eta_bsrs * (gamma * v_next - v_current)
    r_shaping = beta_blend * r_astar + (1.0 - beta_blend) * r_bsrs
    return r_sparse + r_shaping + time_penalty
```

**A\* distance computation:** At level load, run the 6px A\* once (~100ms) using `path_distance_calculator.py`. Cache the distance map. On mine toggle events, recompute (~100ms, infrequent). Track the per-episode minimum distance achieved; `astar_progress = max(0, prev_min - current_distance)`.

**Phase transition:** On switch collection, swap the A\* target from switch to exit door. Issue a phase-transition bonus `C_PHASE = 0.5` to reinforce switch collection explicitly.

**beta\_blend annealing schedule:** Linear decay from 1.0 at step 0 to 0.0 at step `0.7 × total_steps`. After that point, BSRS is the sole shaping component and the external A\* distance plays no role.

#### Sparse Terminal Rewards

| Event | Reward |
|-------|--------|
| Switch collected | +1.0 |
| Exit reached | +5.0 |
| Death (mine, enemy, fall) | −0.5 |

---

### 3.7 Complete Observation Space (Layer 1)

```python
observation_space = gym.spaces.Dict({
    'tile_grid':         Box(low=0, high=33, shape=(42, 23), dtype=np.int8),
    'entity_overlays':   Box(low=0.0, high=1.0, shape=(8, 42, 23), dtype=np.float32),
    'mine_threat_map':   Box(low=-1.0, high=1.0, shape=(3, 168, 92), dtype=np.float32),
    'entity_list':       Box(low=-1.0, high=1.0, shape=(48, 16), dtype=np.float32),
    'entity_mask':       Box(low=0, high=1, shape=(48,), dtype=np.bool_),
    'game_state':        Box(low=-1.0, high=1.0, shape=(27,), dtype=np.float32),
})
```

**Rollout buffer (recommended compact storage):**

| Component | Stored form | Bytes/step | Reconstructed at training |
|-----------|------------|------------|--------------------------|
| tile\_grid | int8 array | 966 | Embedding applied |
| entity\_overlays | Entity positions + states | ~400 | Rasterized on-the-fly |
| mine\_threat\_map | mine\_states array | 256 | Recomputed via Rust SDF |
| entity\_list | Raw entity data | ~768 | Feature vectors computed |
| game\_state | float32 | 108 | Direct |
| **Total** | | **~2.5 KB/step** | |

Compare to v1.0's 34 KB/step — a **14× reduction** via on-the-fly reconstruction. Route overlays and waypoint vector are not stored (removed from observation space).

---

### 3.8 Layer 1 Feature Extractor (Transitional)

```python
class StructuredTileExtractorV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.tile_embedding = nn.Embedding(34, 8)
        
        # Main spatial CNN: tile embeds (8) + entity overlays (8) = 16 channels
        self.cnn = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )  # Output: 64
        
        # Mine threat CNN: 3 channels on 168×92
        self.mine_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),  # → 84×46
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(), # → 42×23
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(), # → 21×12
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )  # Output: 32
        
        self.entity_attn = EntityAttention(entity_dim=16, embed_dim=64, num_heads=2)  # Output: 64
        
        self.game_state_mlp = nn.Sequential(
            nn.Linear(27, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
        )  # Output: 128
        
        # Fusion: 64 + 32 + 64 + 128 = 288 → 256
        self.fusion = nn.Sequential(nn.Linear(288, 256), nn.ReLU(), nn.Dropout(0.1))
        self.features_dim = 256
    
    def forward(self, obs):
        tile_embed = self.tile_embedding(obs['tile_grid'].long()).permute(0, 3, 1, 2)
        spatial_input = torch.cat([tile_embed, obs['entity_overlays']], dim=1)
        
        cnn_features = self.cnn(spatial_input)
        mine_features = self.mine_cnn(obs['mine_threat_map'])
        entity_features = self.entity_attn(obs['entity_list'], obs['entity_mask'])
        state_features = self.game_state_mlp(obs['game_state'])
        
        combined = torch.cat([cnn_features, mine_features, entity_features, state_features], dim=1)
        return self.fusion(combined)
```

---

### 3.9 Multi-Level Training Infrastructure

`LevelPool` with per-environment level assignment, per-level success tracking, uniform random sampling (upgraded to PLR in Layer 3). Each environment must load different levels via `set_level(level_id)` at reset. **At level load, run the 6px A\* distance computation (~100ms) and cache the distance map.** The velocity-augmented surface graph is not required at load time — it is computed offline by the curriculum tool for PLR-promoted levels only.

---

### 3.10 Layer 1 Validation

**Test 1 — A\* HWM correctness:** On a test level, verify the A\* distance map decreases monotonically along the shortest path to the switch/exit. Verify `astar_progress` is non-negative. Verify the distance map updates correctly on mine toggle.

**Test 2 — BSRS signal:** After initial training on a simple level, verify that V\_θ(at-gap-with-high-speed) > V\_θ(at-gap-with-low-speed). Verify the BSRS term provides positive shaping for speed-building retreats on gap levels.

**Test 3 — Mine threat map:** On a level with 100+ mines, verify the SDF correctly identifies passable corridors. Verify passability channel has smooth gradients at kill zone boundaries.

**Test 4 — Single-level convergence:** Train on a moderate level (some gaps, a walljump). Verify the reward signal correctly guides the agent — A\* HWM dominates early, BSRS term grows in magnitude as V\_θ learns physics.

**Test 5 — Multi-level training:** Train on 50 levels simultaneously. After 5M steps, measure switch collection rate on held-out levels.

---

### 3.11 Files to Modify

- **`nclone/gym_environment/base_environment.py`**: New observation space (drop `route_overlays`, `waypoint_vector`). New `_get_obs()`. Load A\* distance map at level init. Compute `astar_progress` per frame for HWM reward. Remove graph PBRS loading path.
- **`nclone/gym_environment/observation_processor.py`**: New functions: `compute_entity_overlays()`, `compute_entity_list()` (excluding mines), `compute_mine_threat_map()` (calls Rust SDF). Remove `compute_route_overlays()` and `compute_waypoint_vector()`.
- **`nclone/gym_environment/reward_calculation/main_reward_calculator.py`**: Replace old HWM reward with A\* HWM + BSRS blending formula from Section 3.6. Accept `v_current` and `v_next` from the value head for BSRS.
- **`nclone-native/src/mine_sdf.rs`**: Repurpose for high-resolution mine threat map computation (168×92 grid with 3 channels). Add incremental update on mine toggle.
- **`npp_rl/feature_extractors/configurable_extractor.py`**: Replace with `StructuredTileExtractorV1` (16 input channels, no waypoint stream).
- **`nclone/gym_environment/spatial_context.py`**: **Deprecate.**
- **`nclone/graph/reachability/feature_computation.py`**: **Deprecate** (route overlays and waypoints removed from observations).

---

## 4. Layer 2: Impoola-CNN + FiLM + Policy-Value Decoupling

### 4.1 Impoola-CNN with FiLM Velocity Conditioning

Upgrade from the transitional Layer 1 extractor to `ImpoolaCNNWithFiLM` (15-layer residual network with Global Average Pooling). Input channels remain **16** (8 tile embed + 8 entity overlays) — route overlays have been removed from the observation space.

```python
self.spatial_cnn = ImpoolaCNNWithFiLM(
    in_channels=16,  # 8 tile embed + 8 entity overlays
    state_dim=27,    # game_state for FiLM conditioning
)
```

**Spatial dimensions:**
```
Input:       (B, 16, 42, 23)
After seq 0: (B, 32, 21, 11)
After seq 1: (B, 64, 10, 5)
After seq 2: (B, 64, 5, 2)
After GAP:   (B, 64)
```

---

### 4.2 Mine Threat Encoder with FiLM

```python
class MineThreatEncoder(nn.Module):
    def __init__(self, in_channels=3, state_dim=27):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.film = nn.Linear(state_dim, 2 * 32)
        self.output_dim = 32
    
    def forward(self, mine_map, game_state):
        x = self.conv(mine_map)
        film_params = self.film(game_state)
        gamma, beta = film_params.chunk(2, dim=-1)
        x = (gamma.unsqueeze(-1).unsqueeze(-1) + 1.0) * x + beta.unsqueeze(-1).unsqueeze(-1)
        return self.gap(x).flatten(1)
```

FiLM conditioning makes physical sense here: at high velocity, the agent needs wider mine clearance (less reaction time), so the network can learn that the same SDF value is "safe" at low speed but "dangerously close" at high speed.

---

### 4.3 Feedforward Fusion (No Temporal Backbone)

The N++ environment, as observed through the game\_state vector, is a **true MDP** — every variable that determines the next physics state is present in the current observation (`vx`, `vy`, `airborne`, `walled`, `state`, buffer values, `jump_duration_remaining`, etc.). No hidden state exists and no temporal memory is needed:

- **Velocity tracking:** `(vx, vy)` are explicit in game\_state. Acceleration is fully determined by `(vx, airborne, hor_input)`.
- **Speed buildup memory:** The current velocity IS the summary statistic of acceleration history. `vx = 3.2` tells the agent everything "40 frames of running right" would tell it.
- **Jump phase tracking:** `jump_duration_remaining`, `airborne`, and buffer states fully capture jump phase.
- **Path memory:** Level geometry is visible in the tile grid; mine state is reflected in the mine threat map. Go-Explore handles cross-episode exploration memory.

**Frame stacking removal:** Remove `FrameStackWrapper`. It is not needed.

**game\_state remains at 27 dims.** The `horizontal_range`, `max_jump_height`, and `time_to_ground` physics-derived features are retained — they remain useful FiLM conditioning signals even without an LSTM.

**Fallback if oscillation is observed:** If empirical results show the agent oscillating between equally-valued directions, append `(previous_action, previous_duration, frames_since_last_decision)` as 3 additional game\_state features. This is 3 floats instead of a 256-dim LSTM with full BPTT infrastructure.

---

### 4.4 Late Policy-Value Decoupling

`DecoupledHeads` architecture — shared encoder, separate policy head and value head. The value function requires more level-specific information than the policy; sharing representations causes the policy to overfit to value-function-specific features.

---

### 4.5 Complete Model Architecture (Layer 2)

```python
class NPPActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.tile_embedding = nn.Embedding(34, 8)
        self.spatial_cnn = ImpoolaCNNWithFiLM(in_channels=16, state_dim=27)   # 64
        self.mine_encoder = MineThreatEncoder(in_channels=3, state_dim=27)    # 32
        self.entity_attn = EntityAttention(entity_dim=16, embed_dim=64)       # 64
        self.game_state_mlp = nn.Sequential(
            nn.Linear(27, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())   # 128
        
        # Fusion: 64 + 32 + 64 + 128 = 288 → 256
        self.fusion = nn.Sequential(nn.Linear(288, 256), nn.ReLU(), nn.Dropout(0.1))
        self.heads = DecoupledHeads(features_dim=256)
    
    def forward(self, obs):
        tile_embed = self.tile_embedding(obs['tile_grid'].long()).permute(0, 3, 1, 2)
        spatial_input = torch.cat([tile_embed, obs['entity_overlays']], dim=1)
        
        spatial_features = self.spatial_cnn(spatial_input, obs['game_state'])
        mine_features = self.mine_encoder(obs['mine_threat_map'], obs['game_state'])
        entity_features = self.entity_attn(obs['entity_list'], obs['entity_mask'])
        state_features = self.game_state_mlp(obs['game_state'])
        
        fused = self.fusion(torch.cat([
            spatial_features, mine_features, entity_features, state_features
        ], dim=-1))
        
        action_logits, duration_logits, value, intrinsic_value = self.heads(fused)
        return action_logits, duration_logits, value, intrinsic_value
```

**Parameter estimate:** ~1.2M (Impoola-CNN ~350K, mine encoder ~80K, entity attention ~100K, MLPs ~170K, heads ~400K, embeddings ~50K, FiLM generators ~120K). Removing the LSTM saves ~500K parameters and eliminates BPTT.

---
 
### 4.6 Layer 2 Validation

**Test 1 — FiLM sensitivity:** Verify V(x, y, vx=high) > V(x, y, vx=low) near gaps requiring running jumps.

**Test 2 — Velocity awareness via value function:** Verify V(x, y, vx=high) > V(x, y, vx=low) near gaps requiring running jumps. Since `vx` is explicit in game\_state and fed through game\_state\_mlp, the policy has direct access to velocity. Confirm agent builds speed before gaps.

**Test 3 — Mine navigation:** On a dense-mine level (100+ mines), verify the agent threads passable corridors identified by the threat map. Compare to agent without threat map.

**Test 4 — Entity-aware routing:** On a level where the optimal path goes through a bounce block, verify the agent discovers and uses the entity-dependent route through exploration + value function learning, without any precomputed graph guidance.

**Test 5 — Generalization:** Held-out level success rate improvement over Layer 1.

---

### 4.7 Files to Modify

- **`npp_rl/feature_extractors/configurable_extractor.py`**: Replace with `ImpoolaCNNWithFiLM`, `MineThreatEncoder`, `EntityAttention`, fusion.
- **`npp_rl/agents/figar_ppo.py`**: Use feedforward policy (no hidden state management, no BPTT).
- **`nclone/gym_environment/frame_stack_wrapper.py`**: **Deprecate.**

---

## 5. Layer 3: Learnability-Scored Curriculum with Synthetic Levels

### 5.1 Synthetic Level Generator

Controllable difficulty parameters, difficulty tiers 0.0–1.0, reachability validation.

**Expert route computation (offline, on-demand):** When PLR promotes a level to the active pool, run the velocity-augmented surface graph planner via `npp_rl/curriculum/expert_routes.py`. The planner produces the frame-optimal Dijkstra route, which serves as:
- **Regret oracle** for curriculum scoring (expert\_return − agent\_return).
- **Demonstration data** for optional behavioral cloning pre-training.

The surface graph is **not** used as a reward source or observation source at runtime. Levels where the planner fails (graph not yet computed or entity-rich levels without entity-aware extension) fall back to SFL learnability scoring (p·(1−p)) instead of true regret scoring.

---

### 5.2 Prioritized Level Replay with Learnability Scoring

`PrioritizedLevelReplay` with SFL learnability scoring (p·(1−p)) replacing naive regret approximations. Updated per-level diagnostic metrics:

```python
# Per-level metrics
level_success_rate          # p for SFL learnability scoring
level_best_astar_dist       # Best A* distance achieved this episode (closest to switch/exit)
level_mean_shaping          # Mean per-episode total shaping reward (A* HWM + BSRS)
level_expert_regret         # expert_return − agent_return (when expert route available)
level_entity_route_used     # Whether agent discovered entity-dependent routes
```

---

### 5.3 Curriculum Progression

Three-phase structure (Foundation → Intermediate → Advanced + Real). The velocity-augmented surface graph is computed on-demand for PLR-promoted levels, not batch-precomputed for the entire pool.

**Phase 1 (Foundation):** 50 synthetic levels, simple geometry, mines only. A\* HWM dominates reward signal. Validate basic locomotion convergence.

**Phase 2 (Intermediate):** 500 synthetic levels, entity-enabled. A\* HWM → BSRS blend transition occurs during this phase. Validate entity-aware routing through exploration.

**Phase 3 (Advanced + Real):** 2K–5K+ synthetic + real levels. BSRS dominates. Full ACCEL mutations. PLR with SFL scoring and true regret oracle for promoted levels.

---

### 5.4 ACCEL-Style Level Mutations

After mutation, verify reachability using the 6px A\* graph (~100ms). Re-run the velocity-augmented surface graph precomputation (~2 min) only if PLR promotes the mutated level to the active pool and true regret scoring is required. Natural tile mutations for N++'s 42×23 grid: adding/removing platforms, adjusting gap widths, repositioning hazards.

---

### 5.5 Files to Create

- `npp_rl/curriculum/level_generator.py` — synthetic level generator with difficulty parameters
- `npp_rl/curriculum/plr.py` — PLR with SFL learnability scoring
- `npp_rl/curriculum/expert_routes.py` — interfaces with entity-aware surface graph planner for regret oracle
- `npp_rl/training/train.py` — PLR integration
- `nclone/gym_environment/base_environment.py` — dynamic level loading

---

## Appendix A: Deprecation List

**Replaced by structured observations + A\* HWM reward:**
- `nclone/gym_environment/spatial_context.py` — replaced by tile tensor + entity overlays
- `nclone/gym_environment/observation_processor.py` — global\_view rendering path
- `nclone/graph/reachability/feature_computation.py` — replaced by entity overlays + mine threat map; route overlays and waypoints removed from observation space
- `nclone/graph/reachability/directional_connectivity.py` — no longer called for observations
- `nclone/gym_environment/reward_calculation/hwm_shaping.py` — **replaced by A\* HWM + BSRS reward formula (Section 3.6)**

**Replaced by feedforward architecture (frame stacking not needed):**
- `nclone/gym_environment/frame_stack_wrapper.py`

**Repurposed (not removed):**
- `nclone-native/src/mine_sdf.rs` — repurposed for high-res mine threat map (168×92)

**Retained for reward computation (A\* HWM):**
- `nclone/graph/reachability/graph_builder.py` — 6px grid used for A\* distance map at level load
- `nclone/graph/reachability/path_distance_calculator.py` — computes A\* distance map for HWM reward

**Retained as offline curriculum tool (expert routes for PLR regret oracle):**
- `nclone/value_function/surface_extraction.py` — extended with entity surfaces for promoted levels
- `nclone/value_function/surface_graph.py` — unchanged
- `nclone/value_function/edge_enumeration.py` — extended with entity physics
- `nclone/value_function/precompute_cli.py` — invoked by `expert_routes.py` on PLR-promoted levels
- `nclone/value_function/arc_cache.py` — retained for expert route computation

**Removed from runtime pipeline (no longer used for per-frame reward or observations):**
- `nclone/gym_environment/reward_calculation/primitive_reward_calculator.py` — the legacy per-frame primitive reward loop (NPP-30)
- `nclone/gym_environment/reward_calculation/pbrs_potentials.py` — `PBRSCalculator` wrapper and static `PBRSPotentials`
- `nclone/gym_environment/reward_calculation/reward_machine.py`, `adaptive_pbrs.py`, `sparse_reward_calculator.py` — unused alternative reward paths
- `nclone/simulation_pbrs/` — precomputed per-level Φ grids consumed only by the deleted primitive path (NPP-31)
- Runtime route overlay (`_debug_route_overlay`, `set_route_overlay_enabled`) — removed from `debug_mixin.py`; offline viewer in `nclone/value_function/visualize_route_interactive.py` is unaffected
- `env.step()` info keys `primitive_rewards` and `precomputed_pbrs_active` — no longer emitted
- `base_environment.py` kwargs `value_function_dir`, `use_precomputed_pbrs`, `simulation_phi_dir`, `use_reachability_pbrs`, `iql_checkpoint_path` — removed
- Route overlay channel generation code — removed from `observation_processor.py`
- Waypoint vector generation code — removed from `observation_processor.py`

---

## Appendix B: Hyperparameter Summary

| Parameter | Layer 1 | Layer 2 | Layer 3 | Notes |
|-----------|---------|---------|---------|-------|
| `tile_embed_dim` | 8 | 8 | 8 | Learned per tile type |
| `entity_overlay_channels` | 8 | 8 | 8 | Density-based mine channels |
| `mine_threat_channels` | 3 | 3 | 3 | deadly\_sdf, safe\_sdf, passability |
| `mine_threat_resolution` | 168×92 | 168×92 | 168×92 | 6px per cell |
| `entity_attn_dim` | 64 | 64 | 64 | Non-mine entities only |
| `max_entities` | 48 | 48 | 48 | Excludes mines |
| `cnn_channels` | [32,64,64] | [32,64,64] | [32,64,64] | Impoola-CNN |
| `cnn_in_channels` | 16 | 16 | 16 | tile(8)+entity(8) |
| `features_dim` | 256 | 256 | 256 | Fused output |
| `alpha_astar_hwm` | 1.0 | 1.0 | 1.0 | A\* HWM shaping weight |
| `eta_bsrs` | 1.0 | 1.0 | 1.0 | BSRS shape-scale |
| `beta_blend_start` | 1.0 | 1.0 | 1.0 | A\* HWM/BSRS blend, anneals to 0 at 70% training |
| `time_penalty` | −0.002 | −0.002 | −0.002 | Per-step cost |
| `C_PHASE` | 0.5 | 0.5 | 0.5 | Phase transition bonus (switch collection) |
| `gamma` | 0.99 | 0.99 | 0.99 | Discount factor |
| `learning_rate` | 3e-4 | 2.5e-4 | 2.5e-4 | |
| `entropy_coeff` | 0.01 | 0.01 | 0.01 | |
| `num_envs` | 256 | 256 | 256 | |
| `plr_temperature` | N/A | N/A | 0.3 | |
| `num_training_levels` | 50 | 500 | 2K–5K+ | |
| `astar_compute_time` | ~100ms/level | ~100ms/level | ~100ms/level | At level load |

---

## Appendix C: Memory Budget

| Component | Stored | Layer 1 | Layer 2 | Notes |
|-----------|--------|---------|---------|-------|
| Obs per step (compact) | Positions + states | ~2.5 KB | ~2.5 KB | Reconstructed at training |
| Rollout buffer (64K steps) | | ~160 MB | ~160 MB | Compact storage; no LSTM hidden states |
| Model parameters | | ~0.9M | ~1.2M | LSTM removed saves ~500K params |
| A\* distance map (per level) | float32 array | ~4 KB | ~4 KB | 168×92 cells, loaded at level init |
| Surface graph cache (per promoted level) | .npz file | — | — | ~5 MB, offline only, curriculum use |
| Mine SDF compute | Per frame | ~50μs | ~50μs | Rust, incremental |

---

## Appendix D: Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│              OFFLINE CURRICULUM TOOL (on PLR promotion)         │
│  tile geometry + entity surfaces → velocity-augmented graph     │
│  → Dijkstra expert route for switch & exit                     │
│  → expert_return for regret oracle; demo data for BC           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               LEVEL LOAD (~100ms, at set_level())               │
│  6px A* distance map to switch / exit                          │
│  → cached for HWM reward; recomputed on mine toggle            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ loaded at env.set_level()
┌───────────────────────────▼─────────────────────────────────────┐
│                     RUNTIME (per frame)                         │
│                                                                 │
│  Observations:                                                  │
│  ├─ tile_grid (42×23 int8) ──────────────┐                     │
│  ├─ entity_overlays (8×42×23) ───────────┤ → Impoola-CNN       │
│  │                                       └── + FiLM → 64       │
│  ├─ mine_threat_map (3×168×92, Rust SDF) ──→ MineCNN+FiLM → 32│
│  ├─ entity_list (48×16, non-mine) ─────────→ Attention → 64   │
│  └─ game_state (27 dims) ──────────────────→ MLP → 128        │
│                                                                 │
│  Fusion: [64+32+64+128] = 288 → MLP → 256                     │
│  Heads: policy(action+duration), value, intrinsic_value        │
│                                                                 │
│  Reward:                                                        │
│  ├─ sparse: R_SWITCH(+1), R_EXIT(+5), R_DEATH(−0.5)           │
│  ├─ A* HWM: α·astar_progress (non-negative, β_blend weighted) │
│  ├─ BSRS: η(γV_θ(s') − V_θ(s)) ((1−β_blend) weighted)       │
│  └─ time_penalty: −0.002/step                                  │
│     β_blend: 1.0 → 0.0 over first 70% of training             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix E: Key Research References

| Paper | Venue | Relevance |
|-------|-------|-----------|
| Ng, Harada & Russell — Policy Invariance Under Reward Transformations | ICML 1999 | PBRS theoretical foundation (BSRS) |
| Grzes & Kudenko — Theoretical and Empirical Analysis of PBRS | AAMAS 2009 | Φ = V\* is optimal potential; motivates BSRS |
| Wiewiora — Potential-Based Shaping and Q-Value Initialization | JAIR 2003 | PBRS ≡ Q-initialization; risk of misleading external potentials |
| Cobbe et al. — Leveraging Procedural Generation | ICML 2020 | IMPALA-CNN, multi-level training |
| Perez et al. — FiLM: Visual Reasoning with General Conditioning | AAAI 2018 | Feature-wise linear modulation |
| Raileanu & Fergus — Decoupling Value and Policy | ICML 2021 (IDAAC) | Late policy-value decoupling |
| Lee et al. — DHRL: Disentangled Hierarchical RL | NeurIPS 2022 Oral | Graph-guided hierarchy |
| Gürtler et al. — HiTS: Timed Subgoals | NeurIPS 2021 | Timed subgoals for DHRL |
| Parker-Holder et al. — ACCEL | ICML 2022 | Evolutionary level mutations |
| Rutherford et al. — SFL: Sampling For Learnability | NeurIPS 2024 | Learnability-scored curriculum |
| Adamczyk et al. — BSRS: Bootstrapped Reward Shaping | AAAI 2025 | Self-bootstrapped PBRS |
| Li et al. — SLOPE | 2026 | Bootstrapped potential convergence bounds |
| Wang et al. — QRL: Quasimetric RL | ICML 2023 | Asymmetric physics-aware distance |
| Hossain & Roy — QPRL | ICML 2025 | Φ/Ψ decomposition for irreversible costs |
| Myers et al. — TMD: Temporal Metric Distillation | NeurIPS 2025 | Quasimetric from suboptimal data |
| Henaff et al. — E3B: Elliptical Episodic Bonuses | NeurIPS 2022 | Episodic exploration |
| Henaff et al. — Episodic × Global exploration | ICML 2023 | Multiplicative bonus combination |
| Yang et al. — DRND: Distributional RND | ICML 2024 | Global exploration in procedural envs |
| Ma et al. — DuRND | ICML 2025 | Dual RND with contribution signal |
| Janer et al. — Impoola-CNN | RLC 2025 | GAP + residual CNN for Procgen |
| Bauersfeld et al. — FiLM for quadrotor control | ICRA 2023 | Physics-conditioned FiLM validation |
| Cheng et al. — Extreme Parkour | ICRA 2024 | Momentum planning in locomotion (motivates BSRS learning physics) |
| Canonaco et al. — PBRS with approximate potentials | arXiv 2024 | BSRS convergence guarantees even with imperfect external signal |
| Müller & Kudenko — Potential bias intervention | AAMAS 2025 | Low-cost PBRS improvement |
| Hu et al. — BiPaRS | NeurIPS 2020 | Learned shaping weight function |
| Mediratta et al. — Generalization in online RL | ICLR 2024 | Multi-level training vs offline |
