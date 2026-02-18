# Dense value learning from millions of demonstrations for physics-based platformer RL

**IQL-based offline value learning combined with potential-based reward shaping is the strongest approach for your N++ agent.** Implicit Q-Learning can train on your millions of expert demos to produce dense V(s) and Q(s,a) estimates for every state — including mid-air — that directly replace the brittle surface-graph precomputation. When used as a PBRS potential, IQL's learned value function provides the dense reward signal needed for complex multi-hop routes while mathematically preserving optimal policy guarantees. This approach generalizes across level layouts when paired with a Universal Value Function Approximator (UVFA) architecture that conditions on the tile grid, and handles the two-phase objective by encoding switch-collection status in the state. The rest of this report details the full landscape of applicable methods, ranked by relevance and practicality.

---

## IQL is the clear first choice for offline value learning

**Implicit Q-Learning** (Kostrikov et al., ICLR 2022) emerges as the top candidate across every evaluation criterion. IQL learns both V(s) and Q(s,a) via expectile regression — approximating the value of the best in-distribution actions without ever querying out-of-distribution actions. This produces **neural network value functions that generalize to any state through interpolation**, including the airborne states between surfaces where your current graph-based approach fails.

The critical capability for N++ is **trajectory stitching**. IQL performs iterative Bellman backups that can combine segments from different demonstrations to discover novel routes. On AntMaze benchmarks — long-horizon navigation tasks requiring stitching suboptimal trajectory fragments, directly analogous to N++ multi-hop platforming — IQL massively outperforms methods lacking this capability. The expectile parameter τ controls stitching aggressiveness: at **τ = 0.9** (recommended for N++), IQL aggressively combines trajectory segments, enabling it to discover multi-jump routes even if no single demonstration shows the exact sequence.

For **discrete action spaces with 6 actions**, IQL is a natural fit. The advantage-weighted regression policy extraction simply computes A(s,a) = Q(s,a) − V(s) for all 6 actions and weights behavioral cloning loss by exp(β·A). No continuous optimization or sampling is needed. Implementation is straightforward: the official JAX implementation exists, and d3rlpy provides `DiscreteCQL` out of the box (with DiscreteIQL being added). Training is standard mini-batch SGD that scales linearly with your millions of transitions.

The **two-phase objective** integrates cleanly: encode a binary `switch_collected` flag in the state representation. IQL's value function will learn dramatically different value landscapes for pre-switch and post-switch states, since the reward structure changes completely. With millions of demos showing both phases, the value function learns this transition naturally.

**CQL (Conservative Q-Learning)** serves as a strong alternative if IQL's values prove insufficiently conservative. CQL's discrete variant adds a log-sum-exp regularizer that pushes down Q-values for out-of-distribution actions — trivially computable for 6 actions with no sampling required. CQL provides **guaranteed lower bounds** on true Q-values, preventing dangerous overestimation in regions far from demonstration data. On Atari benchmarks (discrete actions up to 18), CQL outperformed prior methods by 5–36×.

---

## PBRS with learned potentials preserves optimality guarantees

The safest way to inject learned value estimates into your existing FiGAR-PPO pipeline is **Potential-Based Reward Shaping** with IQL's V(s) as the potential function. PBRS defines the shaping reward as F(s, s') = γ·Φ(s') − Φ(s), and Ng et al. (1999) proved this is the **only** form of reward shaping that preserves optimal policies — regardless of potential quality. An imperfect potential won't change what's optimal; it just determines how much the shaping helps convergence.

The practical implementation is a one-line change to your reward function:

```
shaped_reward = env_reward + gamma * V_iql(next_state) - V_iql(current_state)
```

Several key considerations make this robust in practice. **Scaling** matters: normalize V(s) values and introduce a tunable weight α (start at 0.1–1.0) to prevent the shaping signal from dominating the environment reward. **Terminal states** must have Φ(s_terminal) = 0 for episodic tasks. **Annealing** α toward zero over training allows the agent's own value estimates to take over as they improve. Wiewiora (2003) showed PBRS is mathematically equivalent to Q-value initialization, so this approach effectively warm-starts your PPO agent's understanding of each level.

This PBRS approach avoids the instabilities of direct offline-to-online fine-tuning. Research shows that directly fine-tuning IQL or CQL Q-networks online causes a "recalibration" phase where values diverge and policies collapse. PBRS sidesteps this entirely by treating the offline V(s) as a frozen, separate shaping signal rather than network weights to be fine-tuned.

For a **more aggressive integration**, Cal-QL (Nakamoto et al., NeurIPS 2023) provides calibrated Q-values specifically designed for offline-to-online transition, preventing the unlearning phenomenon with a one-line modification to CQL. If you later want to move beyond PBRS to direct fine-tuning, Cal-QL is the method to use.

---

## Hierarchical and goal-conditioned methods solve the two-phase structure

**HIQL (Hierarchical Implicit Q-Learning)** is the strongest approach specifically for N++'s long-horizon, two-phase structure. HIQL trains a single action-free goal-conditioned value function V(s, φ(g)) via IQL, then extracts a two-level hierarchy: a high-level policy that predicts latent subgoals k steps ahead, and a low-level policy that executes primitive actions to reach each subgoal. On OGBench's **80 navigation tasks**, HIQL dominates — it achieves the highest success rates on AntMaze-large, HumanoidMaze, and pixel-based manipulation benchmarks.

The hierarchical decomposition maps naturally to N++: the high-level policy selects which platform or surface to reach next (in learned representation space, not explicit coordinates), while the low-level policy handles the frame-by-frame jump mechanics. The **subgoal horizon k ≈ 25 steps** is a critical hyperparameter — tune it to match typical jump sequence lengths. Only the low-level policy requires action labels; the value function and high-level policy can learn from action-free trajectory data.

**Successor Features** offer an elegant alternative specifically for the two-phase objective. The decomposition Q(s,a) = ψ^π(s,a) · w separates dynamics (successor features ψ, which capture expected future state occupancy) from rewards (weight vector w). Since N++'s physics remain identical across both phases — only the reward target changes from switch to exit — you can **learn ψ once from all demonstrations, then simply swap w** to switch objectives. Generalized Policy Improvement (GPI) then selects the best action across both phases. This decomposition has provable convergence guarantees via SF-DQN (ICLR 2024) and works natively with discrete actions.

**Contrastive RL and quasimetric methods** represent the current state of the art for learning temporal distances from demonstrations. **Temporal Metric Distillation (TMD, 2025)** combines contrastive successor features with quasimetric network parameterization to achieve SOTA on OGBench across 80 tasks. The quasimetric structure is particularly valuable for platformers because distances are inherently **asymmetric** — jumping up to a platform takes more time/effort than falling down, and quasimetrics encode this directionality via the triangle inequality without assuming symmetry. TMD learns V*(s, g) for any state-goal pair from offline trajectory data through contrastive learning (no reward labels needed), producing dense distance estimates for all states including mid-air.

**Contrastive Difference Predictive Coding (CDPC)** specifically achieves 2× median improvement in success rates over prior contrastive methods by introducing a temporal-difference version that "stitches" trajectory segments — achieving the multi-hop route composition that your surface graph fails at, but through learned representations rather than explicit graph construction.

---

## Reward learning from demonstrations scales to millions of trajectories

If you want to learn a reward function rather than a value function, **T-REX (Trajectory-ranked Reward Extrapolation)** is the most scalable option. It trains a reward network r_θ(s) from ranked trajectory pairs using a Bradley-Terry pairwise loss — pure supervised learning with no MDP solving in the inner loop. With millions of N++ demonstrations naturally rankable by completion time or score, T-REX training is highly efficient. The learned reward provides dense signal for all states and can achieve **better-than-demonstrator performance** by extrapolating the reward function beyond the demonstration distribution. D-REX automates the ranking by using BC policies with varying noise levels.

**AIRL (Adversarial Inverse RL)** recovers a state-only reward r(s) that is disentangled from dynamics, theoretically enabling transfer across level layouts. However, AIRL requires on-policy training (expensive) and adversarial optimization (unstable). Recent work (2024) shows that using SAC as the inner RL algorithm breaks AIRL's reward disentangling due to the entropy term; a PPO-AIRL hybrid preserves it.

**GAIL and its successors** provide dense reward through a discriminator D(s,a), but with important limitations. DRAIL (NeurIPS 2024, NVIDIA) replaces GAIL's discriminator with a diffusion classifier, producing smoother and more precise rewards. C-GAIL (NeurIPS 2024) adds control-theoretic regularization for stability. However, all GAIL variants require on-policy data collection as the computational bottleneck and cap at demonstration quality. **IQ-Learn** (NeurIPS 2021 Spotlight) offers a non-adversarial alternative that learns a single Q-function encoding both reward and policy, recovering rewards with **0.99 Pearson correlation** to ground truth on Hopper.

For the specific goal of **temporal progress estimation**, Rank2Reward (ICRA 2024) and TimeRewarder (2025) learn monotonically increasing progress scores from demonstration temporal ordering. TimeRewarder estimates frame-wise temporal distances and produces rewards that can be **frozen during RL** — no online updates needed. This naturally handles mid-air states (they receive progress scores based on trajectory position) and the two-phase objective (switch collection causes a natural progress jump).

---

## A universal value model can generalize across level layouts

Cross-level generalization requires a **UVFA-style architecture** that conditions the value function on both agent state and level geometry. The recommended architecture:

- **Tile embedding layer**: `nn.Embedding(38, dim=8-16)` maps each of 38 tile types to a learned vector, capturing semantic similarity (all solid tiles cluster together). Applied to the 42×23 grid, this produces a (42, 23, embed_dim) spatial tensor
- **Level encoder CNN**: 2-3 convolutional layers with 3×3 or 5×5 kernels extract spatial patterns from the embedded grid. Receptive fields should cover platform-relevant distances (~5 tiles of context)
- **Agent state MLP**: encodes (x, y, vx, vy, on_ground, switch_collected) into a compact vector
- **Combined network**: concatenate or FiLM-condition the CNN output with agent state, feed through MLP to produce V(s, level) or Q(s, a, level) for all 6 actions

Research on Procgen (Cobbe et al., ICML 2020) — the most relevant benchmark for procedurally-generated platformer generalization — establishes that agents need **500–1,000+ training levels** before generalizing reliably. Larger models significantly outperform smaller ones. A 2024 study found that replacing 2D convolutions with **3D convolutions over stacked frames** reduced the optimality gap by 37.9% across all Procgen environments. Separate policy and value networks also improve generalization by reducing interference.

A critical finding from "The Generalization Gap in Offline RL" (ICLR 2024, Facebook Research) is that **behavioral cloning often outperforms offline RL methods for cross-level generalization**, and data diversity matters more than data quality. This suggests training on demonstrations from the maximum diversity of levels, even if some are suboptimal, and potentially using BC as the generalization backbone with offline RL providing per-level value refinement.

---

## The recommended implementation pipeline

Based on the full research landscape, here is a practical ranked pipeline:

**Stage 1 — Offline value learning (highest priority)**. Train IQL on your demonstration dataset with the UVFA architecture described above. Use expectile τ = 0.85–0.9 for aggressive stitching, discount γ = 0.99 for long episodes, batch size 1024, learning rate 3e-4, 3-layer MLPs with 512 hidden units and LayerNorm. Include the phase flag in the state. This produces V(s, level) and Q(s, a, level) that replace your surface graph entirely. Use d3rlpy's DiscreteCQL as a starting point, or implement DiscreteIQL from the IQL paper's straightforward equations.

**Stage 2 — PBRS integration**. Freeze the IQL value network and inject it as a PBRS potential into your existing FiGAR-PPO training: `r_shaped = r_env + α·(γ·V_iql(s') − V_iql(s))` with α starting at 0.5 and annealed over training. Enforce V(s_terminal) = 0. This gives your PPO agent dense reward signal across all states — including mid-air — with mathematical guarantees that the optimal policy is preserved.

**Stage 3 — Hierarchical extension (if needed)**. If pure PBRS shaping is insufficient for very long levels, add HIQL-style hierarchical policy extraction. The high-level policy selects subgoal representations every k=25 steps; the low-level policy (your FiGAR-PPO agent) executes. The HIQL value function provides the subgoal selection signal, while the shaped reward guides low-level execution.

**Stage 4 — Online fine-tuning (optional)**. For levels where offline performance plateaus, use Cal-QL for stable offline-to-online transition, or RLPD's remarkably simple approach: mix 50% offline and 50% online data in the replay buffer with high update-to-data ratio (20×) and LayerNorm. For discrete actions, substitute DQN for SAC in RLPD's framework, applying the same symmetric sampling principle.

---

## Conclusion

The core insight unifying all the best approaches is that **your millions of expert demonstrations contain dense temporal distance information for every state, including airborne states — and this information can be extracted by offline TD-learning methods that stitch trajectory fragments**. IQL's expectile regression extracts this cleanly without the brittleness of explicit graph construction. The surface-graph approach fails because it attempts to enumerate connectivity explicitly; learned value functions succeed because they implicitly capture connectivity through Bellman backup propagation across your vast demonstration dataset.

Three genuinely novel considerations deserve emphasis. First, **quasimetric value functions** (TMD, QRL) are uniquely suited to platformers because movement costs are inherently asymmetric — climbing costs more than falling — and quasimetrics encode this directionality structurally rather than hoping the network learns it. Since N++ is deterministic, QRL's known optimistic bias in stochastic environments is a non-issue. Second, **successor features** provide the most elegant two-phase handling: learn dynamics once, swap reward weights instantly between switch-seeking and exit-seeking, with provable transfer guarantees. Third, the finding that **BC outperforms offline RL for cross-level generalization** (ICLR 2024) suggests a split strategy: use BC with the UVFA architecture for the generalizable foundation model, then use IQL/PBRS for per-level value refinement where you need dense reward shaping beyond what BC can provide. This hybrid captures the best of both paradigms — BC's generalization with IQL's stitching and dense value estimates.