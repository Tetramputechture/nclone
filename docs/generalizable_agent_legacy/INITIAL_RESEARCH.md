# Dense generalizable reward functions for momentum-based platformer RL

**The most promising path to cross-level generalization in N++ is a quasimetric value function pre-trained on expert demonstrations, used as potential-based reward shaping (PBRS) during online RL across many levels simultaneously.** Your prior approaches failed because both the A\* high-water-mark shaping and RFCL curriculum are inherently level-specific — they encode individual layout structure rather than generalizable platforming knowledge. The solution is to learn a universal value function from your 100k+ expert demos that operates on local, physics-aware features and transfers zero-shot to unseen levels. This report lays out the complete architecture, algorithm, and training pipeline to achieve this.

---

## Quasimetric learning captures momentum physics better than any alternative

The core insight driving the top recommendation is that **N++'s momentum-based physics create strongly asymmetric cost landscapes** — reaching a platform above via wall-jumping costs far more actions than falling down, and horizontal movement with momentum is dramatically cheaper than from rest. Standard value functions learn this implicitly through Bellman backups, but Quasimetric RL (QRL, Wang et al. ICML 2023) encodes this asymmetry as a structural prior.

QRL exploits the fact that the negative optimal value function **-V\*(s, g) is a quasimetric**: it satisfies non-negativity, identity, and the triangle inequality, but crucially *allows asymmetry* (d(s₁, s₂) ≠ d(s₂, s₁)). QRL parameterizes this distance with interval quasimetric embeddings and trains it to match local transition costs while respecting the triangle inequality globally. On MountainCar (another asymmetric physics domain), QRL learns optimal value functions much earlier than standard Q-learning, and the authors specifically note that "contrastive RL only inaccurately estimates the on-policy values."

For N++, decompose the task into two sub-goals: **d_θ(s, s_switch)** for phase 1 and **d_θ(s, s_door)** for phase 2. The quasimetric structure naturally encodes that going up costs more than going down, that momentum reduces traversal cost, and that the triangle inequality constrains generalization to unseen states. No reward labels are needed — only transition data with known goal locations from your 100k+ demonstrations.

**The practical fallback is goal-conditioned action-free IQL** (the HIQL variant from Park et al., NeurIPS 2023). This learns V(s, g) via expectile regression without actions or rewards: `L(ψ) = E[L_τ(γV(s',g) - V(s,g))]`. In deterministic environments like N++, this is unbiased. Set the **expectile parameter τ = 0.9** — with near-optimal expert data, the upper expectile identifies the best demonstrated strategies while providing smoothing. IQL is simpler, faster (~2-3x vs CQL), better understood, and has a mature ecosystem. Conservative Q-Learning (CQL) is unnecessarily pessimistic for expert-only data and produces overly flat value landscapes that make poor PBRS potentials.

Among the remaining alternatives: **Contrastive RL** (Eysenbach et al., NeurIPS 2022) excels at learning transferable representations from demonstrations via InfoNCE without rewards, but its symmetric inner product struggles with asymmetric physics. **Successor features** provide principled transfer via the GPI theorem but require a restrictive linear reward decomposition. **Bootstrapped Reward Shaping** (Adamczyk et al., AAAI 2025) is complementary — it accelerates online training by using the agent's own evolving V(s) as a dynamic potential with a shape-scale η ≈ 2, and can be combined with a pre-trained value function from the offline phase.

| Method | Physics asymmetry | Cross-level transfer | Works without rewards | PBRS quality | Complexity |
|--------|:-:|:-:|:-:|:-:|:-:|
| **QRL** | ★★★★★ | ★★★★ | ★★★★★ | ★★★★★ | Medium-high |
| **IQL (goal-cond.)** | ★★★ | ★★★ | ★★★★ | ★★★★ | Low |
| Contrastive RL | ★★ | ★★★★ | ★★★★★ | ★★★ | Low-medium |
| Successor features | ★★ | ★★★★★ | ★★ | ★★★ | Medium |
| CQL | ★★★ | ★★ | ★★ | ★★ | Medium |

---

## Why your previous approaches failed — and what fixes the generalization gap

Both FiGAR-PPO with HWM A\* shaping and SAC-RFCL fail for the same fundamental reason: **they encode level-specific structure into the reward signal rather than learning generalizable platforming knowledge.**

The A\* high-water-mark computes shortest paths on each level's specific tile layout. The agent learns to follow that gradient for individual levels rather than learning velocity management, jump timing, or hazard avoidance as transferable skills. Moreover, HWM's non-negative-only signal provides zero gradient during backtracking phases — and in N++, backing up to build momentum before a long jump is routinely optimal. RFCL is even more level-specific: its reverse curriculum traces states from particular level demonstrations, and its "forward curriculum" expands coverage within a single environment, not across layouts. The RFCL paper (Tao et al., ICLR 2024) benchmarks on MetaWorld and Adroit — single-task mastery environments, not multi-level generalization.

**The fix is a level-agnostic dense reward function that operates on local/relative features.** If V(s) processes a local tile window around the agent, relative direction and distance to the current objective, velocity state, and contact information — rather than global coordinates or level-specific A\* paths — it naturally transfers. Research on CoinRun and Procgen confirms that RL agents routinely overfit to training level instances. Even with 20,000 training levels, significant generalization gaps persist. The Procgen literature shows that the remedies are: (1) training across many diverse levels simultaneously, (2) data augmentation, (3) architectural choices that reduce translation sensitivity, and (4) exploration-driven curricula like PLR. A universal learned reward function is the mechanism that makes all of these work together.

**RFCL becomes unnecessary with a good dense reward.** RFCL primarily solves the sparse-reward exploration problem. With dense shaping from a pre-trained value function, the agent receives meaningful gradient signal from any state, making reverse curriculum redundant. Drop it in favor of multi-level training with Prioritized Level Replay.

---

## The right architecture: IMPALA-CNN with FiLM velocity conditioning and cross-attention

The architecture must take a 42×23 tile grid plus agent physics state and output a value estimate that generalizes across unseen levels. Based on extensive Procgen and NetHack literature, the recommended design combines four components:

**Level encoder.** Encode the tile map as a multi-channel tensor `(C × 42 × 23)` where each channel is a binary mask for one entity type (walls, slopes at 4 orientations, mines, bounce pads, switches, doors). Process with a **scaled IMPALA-CNN** (Espeholt et al., 2018) — 3 residual blocks with increasing channels (32→64→64), 2× width scaling. The IMPALA-CNN is the established gold standard for procedurally generated environments: Cobbe et al. (2020) showed that the small Nature-CNN completely fails on Procgen while IMPALA-CNN with width scaling dramatically improves both sample efficiency and generalization. Recent work (Janer et al., 2025) shows replacing max pooling with **global average pooling** ("Impoola-CNN") yields 17% better generalization with 35% fewer parameters by reducing translation sensitivity.

**Physics state encoder with derived features.** Rather than feeding raw `(vx, vy)`, compute physics-derived quantities that make the value function's job easier:

- Kinetic energy: `0.5 * (vx² + vy²)`
- Momentum direction: `atan2(vy, vx)`
- Estimated time to ground: from current `y`, `vy`, and gravity
- Maximum remaining jump height: from `jump_frames_remaining` and `vy`
- Horizontal range at current speed: from `vx`, time-to-ground, and drag coefficient

These derived features let the network directly compare horizontal range against gap width, or jump height against platform offset, rather than learning these physics relationships from scratch.

**FiLM conditioning for velocity-aware spatial features.** This is the key architectural innovation for momentum-based platformers. Use Feature-wise Linear Modulation (Perez et al., AAAI 2018) to let the velocity state modulate the level's spatial features: `H_modulated = γ(v) * H_level + β(v)`, where `γ, β = FiLM_generator(physics_state)`. This enables the network to learn that "this gap is easy at high speed but lethal at low speed" by having velocity state reshape the spatial value landscape. FiLM has been validated in AlphaStar, quadrotor control (within 0.6-2% of hand-tuned policies), and meta-learning settings.

**Cross-attention between agent and level.** After FiLM modulation, apply cross-attention where the agent state forms the query and the spatial level features form keys and values. This naturally learns to focus on the relevant level regions (the next gap, the nearest mine, the objective) based on the agent's current position and momentum. Combine with an **11×11 local tile crop** around the agent for fine-grained nearby geometry and objective-relative vectors `(switch_pos - agent_pos, door_pos - agent_pos)` for goal awareness.

**Separate value and policy heads** — shared encoder is fine, but separating the heads prevents value function overfitting to level-specific cues from contaminating the policy (validated on Procgen).

---

## PPO with demonstrations beats SAC-Discrete for this domain

SAC-Discrete has **known failure modes** that are particularly dangerous for N++. The SD-SAC analysis (TMLR 2024) identifies three issues: entropy target instability causing near-random or prematurely deterministic policies, double-Q underestimation suppressing exploration, and poor performance in long-term decision-making. The authors explicitly note "discrete SAC still performs poorly in instances involving long-term decision-making" — directly relevant to N++'s multi-step momentum management.

**PPO is the safer, better-tested choice for platformer RL.** Every major platformer generalization result (CoinRun, Procgen, Obstacle Tower) uses PPO as the base algorithm. It handles discrete actions natively, provides stable monotonic improvement, and integrates cleanly with PBRS.

To leverage your 100k+ demonstrations with PPO, use **DAPG-style augmentation** (Rajeswaran et al., RSS 2018): behavioral cloning pre-training followed by an auxiliary demonstration-tracking loss during online RL that decays over training. DAPG showed **30× speedup** with only 25 demonstrations; with 100k+, the effect should be substantial.

If sample efficiency is paramount, consider **Advantage-Weighted Regression (AWR)** (Peng et al., 2019) as a hybrid approach. AWR performs two supervised learning steps — regress value function on returns, then regress policy on advantage-weighted actions — and naturally handles both offline demos and online data. It's more stable than SAC, works with discrete actions, and inherently bridges on-policy stability with off-policy data utilization.

**Keep FiGAR temporal abstraction.** The 48-action macro-action space (6 base × 8 durations) is essential for momentum physics. Single-frame policies must independently re-derive "hold left for N frames" from scratch, while macro-actions encode the action-persistence structure inherent in momentum-based movement. A good value function does not eliminate this need — it makes credit assignment easier but doesn't solve the structural mismatch between frame-level decisions and momentum-scale physics.

If staying with SAC, increase the **RLPD demo ratio from 15% to 50%** — Ball et al. (ICML 2023) found 50% optimal, with robustness from 15-75%, and use the SD-SAC entropy-penalty variant for stability.

---

## Practical implementation: the complete training pipeline

### Phase 1: Offline value function pre-training

Extract `(state, next_state, phase)` tuples from all 100k+ demonstrations. Assign sparse rewards: **+1 for hitting switch, +1 for reaching exit, -1 per timestep** (encouraging time-optimal behavior). Train two separate value functions:

For the QRL path: learn `d_θ(s, g_switch)` and `d_θ(s, g_door)` using quasimetric embeddings with the triangle inequality constraint. The state `s` includes the full physics state plus a CNN encoding of the local tile context, conditioned on the global level embedding via FiLM.

For the IQL fallback: train goal-conditioned IQL with **τ = 0.9**, learning rate 3e-4, batch size 256, for ~300K gradient steps. Monitor validation TD error on held-out demonstrations. Use **layer normalization** throughout — this is critical for offline-to-online stability (recommended by both RLPD and Cal-QL literature).

### Phase 2: PBRS construction with reward machine

Define a **3-state reward machine** encoding the sequential objective:
- State u₀: seeking switch → shape with Φ₀(s) = -d_θ(s, g_switch) or V_switch(s)  
- State u₁: seeking exit → shape with Φ₁(s) = -d_θ(s, g_door) or V_exit(s)  
- State u₂: success (terminal)

Apply standard PBRS: `R_shaped = R_env + α × [γΦ(s') - Φ(s)]`. Critical implementation details: **set Φ = 0 at all terminal states** (death and success), match γ exactly with the online RL discount factor, clip |γΦ(s') - Φ(s)| to ±2.0, and **freeze the pre-trained value function** — updating Φ during training violates the static potential assumption and creates feedback loops.

Use **Counterfactual Experiences for Reward Machines** (CRM, Icarte et al. 2022): when replaying any transition, also update Q-functions for hypothetical RM state transitions, providing **3× data efficiency** (one real experience generates updates for all three RM states).

### Phase 3: Online multi-level training with PLR

Train PPO (or AWR) across hundreds of diverse levels simultaneously using **Prioritized Level Replay** (Jiang et al., ICML 2021). PLR scores each level by the average |GAE| of recent trajectories, then prioritizes levels with highest estimated learning potential. Key hyperparameters from Procgen validation: **buffer size 4000, replay probability 0.95, sampling temperature 0.3, staleness coefficient 0.3**. Start with the easiest 20% of levels (shortest expert completion times), then let PLR naturally expand to harder levels as the agent improves.

**Linearly anneal the shaping weight α from 1.0 → 0.1** over the first 60% of training. Under exact PBRS, annealing is theoretically unnecessary, but with function approximation the learned potential introduces subtle biases. Keeping a residual α = 0.1 maintains exploration guidance without dominating the true reward signal.

Apply **UCB-DrAC data augmentation** (Raileanu et al., NeurIPS 2021) — automatic selection of the best augmentation (crop, translate, color jitter) per training phase, with policy and value function regularization ensuring invariance to augmentations. This yielded ~40% improvement in Procgen test performance over standard PPO.

### Phase 4: Optional BSRS acceleration

During online training, layer **Bootstrapped Reward Shaping** (η ≈ 2) on top of the pre-trained PBRS. This uses the agent's own evolving value estimate as an additional dynamic potential, handling the distribution mismatch between demonstration states and the agent's actual exploration. BSRS proved convergent in the tabular case and improved 21/40 Atari environments by >10%.

---

## What the Procgen frontier tells us about N++ generalization

No published RL work exists specifically on N++, making this a genuinely novel problem setting. The closest benchmarks are Procgen and CoinRun, where the current state of the art reveals several directly applicable insights.

The biggest recent breakthrough is architectural: **Jesson & Jiang (October 2024) reduced the Procgen optimality gap by 37.9%** (from 0.58 to 0.36) through simple changes — frame stacking, replacing 2D convolutions with 3D convolutions, and scaling kernel counts. This is orthogonal to algorithmic improvements and should be adopted for N++, where temporal information (velocity trajectories across frames) is crucial for momentum-based gameplay.

**EDE** (Jiang et al., NeurIPS 2023) demonstrated that exploration is critical for *generalization*, not just training performance. Using deep ensembles with distributional RL to separate epistemic and aleatoric uncertainty, EDE became the first value-based method to achieve SOTA on both Procgen and Crafter. For N++, this suggests that uncertainty-driven exploration may help discover generalizable physics-exploitation strategies.

On the imitation learning frontier, **NitroGen** (NVIDIA et al., January 2026) trained a 493M-parameter vision transformer on 40,000 hours of gameplay across 1,000+ games via pure behavior cloning, achieving **52% relative improvement** on unseen games. While N++ has a much narrower domain, NitroGen validates that large-scale imitation from demonstrations can produce competent platformer gameplay without any RL — suggesting that behavioral cloning pre-training from your 100k+ demos is a strong foundation.

For world models, **DIAMOND** (NeurIPS 2024 Spotlight) achieved 1.46 mean human-normalized score on Atari 100k using a diffusion-based world model. However, N++'s deterministic physics present an even better opportunity: **use the exact physics simulator as the world model** (zero model error) and generate imagined trajectories for value function training. This "imagination-augmented" training with perfect forward simulation provides unlimited free data for refining the value function.

---

## Conclusion: a unified strategy for generalizable N++ agents

The core failure mode of prior N++ approaches was encoding level-specific information into the reward signal. The solution requires three fundamental shifts: **(1) learn a physics-aware value function from demonstrations** using QRL or goal-conditioned IQL that operates on local, velocity-conditioned features rather than global layout coordinates; **(2) use this value function as a frozen PBRS potential** within a reward machine encoding the switch→exit structure, providing dense guidance without per-level engineering; and **(3) train across hundreds of levels simultaneously** with PLR curriculum and DrAC augmentation, letting the universal reward signal drive generalizable skill acquisition.

The most novel architectural insight is **FiLM velocity conditioning on spatial level features** — this lets the same tile pattern produce different value estimates depending on the agent's current momentum, which is essential for a physics game where position alone is nearly meaningless. Combined with the quasimetric distance function's structural encoding of physics asymmetry, this produces a value landscape that respects the actual costs of movement in momentum space.

The recommended algorithm is **PPO with DAPG-style demonstration augmentation** rather than SAC-Discrete, based on SAC-Discrete's documented instability in long-horizon discrete domains and PPO's dominance in every platformer generalization benchmark. Keep FiGAR macro-actions — momentum physics demands temporal abstraction. If you need sample efficiency, AWR offers a stable hybrid alternative that naturally incorporates offline demonstrations.

The most impactful quick wins, in order of expected return on implementation effort: (1) replace HWM A\* shaping with IQL-learned PBRS potential (eliminates level-specific reward); (2) add FiLM velocity conditioning to the level encoder (makes value physics-aware); (3) train across 200+ levels with PLR instead of per-level training (enables generalization); (4) upgrade to IMPALA-CNN 2× with 3D convolutions and frame stacking (37.9% gap reduction on Procgen). These four changes address the fundamental bottlenecks while remaining implementable within the existing codebase.