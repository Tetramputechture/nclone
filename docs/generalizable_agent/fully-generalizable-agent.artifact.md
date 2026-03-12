# A blueprint for a generalizable N++ reinforcement learning agent

A pre-trained policy combined with test-time MCTS planning over the exact physics simulator — augmented by hierarchical subgoal decomposition and entity-aware attention — represents the most promising architecture for solving arbitrary N++ levels. This system leverages every unique advantage of the N++ setting: 100k+ expert demonstrations for strong offline priors, a deterministic simulator for zero-error planning, known entity physics for analytical feature engineering, and the existing surface graph for natural subgoal structure. The proposed three-phase pipeline (offline pre-training → zero-shot inference → per-level adaptation) draws on breakthroughs from 2024–2025 in test-time compute scaling, diffusion-based trajectory planning, and offline-to-online RL transfer to create a system that should achieve near-expert performance on unseen levels within minutes of adaptation.

---

## The core architectural insight: simulator-augmented Gumbel AlphaZero

The single most impactful design decision is treating N++'s deterministic physics simulator as a **perfect world model for test-time tree search**. Research on MuZero versus AlphaZero reveals a critical finding: MuZero's learned dynamics model fails catastrophically for deep planning in complex environments (Zhu et al., 2023 showed "free search with the learned model fails completely" in Lunar Lander and Breakout), whereas a ground-truth simulator enables arbitrarily deep, error-free search. Since N++ provides this ground-truth simulator, the agent bypasses the primary bottleneck of learned-model MCTS entirely.

**Gumbel MuZero** (Danihelka et al., 2022) emerges as the optimal MCTS variant for this setting. It uses the Gumbel-Top-k trick to sample actions from the policy prior without replacement, then applies sequential halving to allocate simulation budget across candidates. The result is **guaranteed policy improvement with as few as 2 simulations** — a property validated on 9×9 Go where standard MuZero fails below 16 simulations. With N++'s 48 macro-actions, sample k=16–24 candidates per node, apply sequential halving with 50–200 simulations per decision step, and use the pre-trained value network for leaf evaluation.

The planning operates as Model Predictive Control: at each decision point, run MCTS for a horizon of **15–25 macro-actions** (roughly 30–200 frames of look-ahead), execute the first action, then replan from the new state. This rolling-horizon approach provides robustness through continuous replanning while the value network handles beyond-horizon estimation. With simulator steps at ~0.1ms, a 10-minute per-level budget yields **6 million simulator steps** — enough for deep search across every decision point in a level.

An important efficiency optimization comes from **ReZero** (2024), which reuses MCTS subtrees between consecutive decision steps, reducing redundant computation by 30–50%. Additionally, **SPO** (Sequential Monte Carlo Policy Optimization, 2024) offers a GPU-parallelizable alternative to traditional MCTS that achieves competitive quality while being embarrassingly parallel — valuable if batch simulation is available.

For implementation, the **LightZero** framework (NeurIPS 2023 Spotlight) provides a unified, open-source implementation of MuZero, EfficientZero, Gumbel MuZero, and Sampled EfficientZero, and explicitly notes that "employing AlphaZero directly is advantageous when an environment simulator is available."

---

## Pre-training on 100k+ demonstrations establishes a strong prior

A critical finding from Mediratta et al. (ICLR 2024) upends conventional wisdom: **behavioral cloning outperforms all offline RL methods and Decision Transformers for cross-level generalization** in procedurally generated environments. On ProcGen, BC achieves stronger transfer to unseen levels than CQL, IQL, or sequence models. This directly informs the pre-training strategy — the 100k+ expert demonstrations should be leveraged primarily through BC, not offline RL.

The pre-training architecture should incorporate four key innovations from recent literature:

**Entity-aware attention** (following Zambaldi et al., ICLR 2019, and ECRL, ICLR 2024 Spotlight). Each dynamic entity gets a typed embedding: `e_i = TypeEncoder(type_onehot, x, y, vx, vy, phase, state, dist_to_ninja, angle_to_ninja)`. Two to three layers of multi-head self-attention over all entity embeddings discover relevant interactions without hard-coding. ECRL demonstrated compositional generalization from 3 to 10+ objects — exactly the scaling property needed when levels vary from 0 to 15+ entities.

**Level conditioning via GNN or CNN encoder with FiLM modulation**. The 42×23 tile map is processed by either a graph neural network (tiles as nodes, adjacency as edges) or a small CNN into a 128-dimensional level embedding. This embedding modulates the policy's hidden layers via Feature-wise Linear Modulation: `γ(e_level) * h + β(e_level)`. GNN-based structural encoding enables generalization across arbitrary tile arrangements, while FiLM conditioning has proven effective for task-conditioned policies. Research on GNN-RL (Almasan et al., IEEE TNNLS 2024) shows GNN-based agents generalizing over 232 unseen topologies.

**Frame stacking with 3D convolutions** for momentum understanding. Jesson et al. (October 2024) achieved a **37.9% reduction in the ProcGen optimality gap** through three simple changes: frame stacking, 3D convolutions replacing 2D, and scaling kernel count. For N++, where momentum is everything, temporal context from 3–4 stacked frames allows the network to infer velocity and acceleration implicitly.

**Physics-informed observation features** that extend the existing 153-dimensional space. For each drone: `[sin(2π·phase/period), cos(2π·phase/period), time_to_safe_window, safe_window_duration]`. For thwumps: `[is_triggered, charge_progress, can_ride, line_of_sight]`. For death balls: `[relative_vel, bearing, time_to_intercept, evasion_direction]`. Dynamic SDFs: `[min_dynamic_dist, gradient_x, gradient_y, time_to_collision]`. These features are computable analytically from known entity physics and extend the existing mine SDF infrastructure at negligible computational cost (~0.01ms per frame).

---

## Hierarchical decomposition through the surface graph

The existing surface graph and reachability analysis provide a natural hierarchical structure that maps precisely to recent advances in hierarchical RL. The recommended architecture is a **three-layer hierarchy** inspired by HIQL (Park et al., NeurIPS 2023), HAC (Levy et al., 2019), and subgoal graph planning.

**Layer 3 — Route planner (graph search).** Convert the tile map into a surface graph where nodes represent landable surfaces and edges represent physically achievable transitions. Annotate edges with costs derived from: (a) the pre-trained value function's estimated traversal difficulty, (b) expert demonstration frequency (edges traversed more often are easier), and (c) timing constraints from dynamic entities. Run **A\* search** over this annotated graph to find the optimal surface sequence from spawn → exit switch → exit door. For dynamic entities, model edge costs as time-dependent: `cost(edge, t)` depends on entity positions at time t. Replan at each subgoal based on current timing state. Graph construction and A\* are CPU-only operations completing in under one second per level.

**Layer 2 — Subgoal policy (HIQL-style).** **HIQL** is the strongest offline hierarchical method for this setting. It trains a goal-conditioned value function V(s, g) on all demonstration data using Implicit Q-Learning, then extracts a two-level policy where the high-level predicts subgoal representations k steps ahead and the low-level produces actions to reach them. The key insight is that hierarchical decomposition improves "signal-to-noise ratio" — for distant goals, optimal and suboptimal actions become indistinguishable in the flat value function, but hierarchy reduces effective planning horizons at each level. Recent extensions include **NF-HIQL** (February 2026, normalizing flow policies for multimodal behavior), **Eik-HIQL** (2025, Eikonal regularization for physics-informed learning), and the **OGBench** suite (December 2024) providing clean implementations.

**Layer 1 — Motor controller (goal-conditioned skill library).** Learn 10–20 skill policies from expert demonstrations using trajectory segmentation. **CompILE** (Kipf et al., 2019) segments trajectories into sub-tasks via a VAE with learned boundaries, while physics-based segmentation at surface transition events (landing/takeoff) provides reliable boundaries without learning. **DADS** (Sharma et al., 2020) discovers skills whose outcomes are predictable by simultaneously learning a skill-conditioned dynamics model — yielding skills where landing positions and velocities can be anticipated, enabling model-based planning over the skill space. Each skill policy takes (current_state, target_surface) as input and outputs FiGAR macro-actions.

The asymmetric reachability structure of N++ (going from A to B may be trivially easy while B to A is impossible) makes **ProQ** (Projective Quasimetric Planning, 2025) particularly relevant — it learns asymmetric distance functions from offline trajectories and uses them to build navigation graphs with sparse keypoints, naturally handling the one-way nature of many platformer transitions.

---

## Per-level adaptation combines fine-tuning with search

The per-level adaptation budget of minutes to hours enables a powerful three-stage adaptation pipeline, drawing on the latest offline-to-online transfer research.

**Stage 1: Zero-shot inference (seconds).** Encode the new level's tile map through the GNN/CNN encoder → generate level embedding → condition the pre-trained policy → attempt the level. The diffusion trajectory model (see below) generates 10–100 candidate trajectories conditioned on the level layout. The physics simulator verifies each candidate, filtering to valid completions. For many moderate-difficulty levels, this stage alone may succeed.

**Stage 2: Warm-start fine-tuning (minutes).** Following **WSRL** (ICLR 2025), the most effective offline-to-online transfer method avoids retaining offline data entirely. Instead: (a) collect 50–100 rollouts using the pre-trained policy on the new level to seed the replay buffer (the "warmup phase"), (b) run PPO fine-tuning with **L2 regularization** toward pre-trained weights (λ||θ − θ_pretrained||², λ ≈ 0.01–0.1) and a **KL penalty** (β·KL(π_new || π_pretrained)) to prevent catastrophic forgetting, (c) use a low learning rate (1e-5 to 5e-5). Research confirms that standard knowledge retention techniques (EWC, L2, KL) significantly help RL fine-tuning, which is "secretly a forgetting mitigation problem." An alternative is **ResFiT** (2025), which freezes the BC policy entirely and learns a small residual correction — achieving **200× better sample efficiency** than on-policy PPO while guaranteeing performance never degrades below the base policy.

**Stage 3: Expert Iteration loop (minutes to hours).** Apply the **MuZero Unplugged / Reanalyse** framework: for states encountered during rollouts, run Gumbel AlphaZero MCTS with the perfect simulator to generate improved policy and value targets. Train the network on these MCTS-improved targets. Iterate 2–5 times. Because the simulator is perfect (unlike MuZero's learned model), Reanalyse targets are far more accurate. MuZero Unplugged demonstrated a "20-fold increase in score" over data-generating policies on some Atari games — with a perfect simulator, improvements should be even larger. This Expert Iteration loop (Anthony et al., 2017) systematically improves beyond demonstration quality: the neural network (apprentice) provides fast intuition, MCTS (expert) provides slow deliberation, and iterative distillation transfers MCTS improvements into the network.

---

## Handling dynamic entities requires attention and analytical features

Dynamic entities represent the hardest generalization challenge because their behavior creates time-dependent constraints that vary across levels. The solution combines three complementary approaches.

**Entity Interaction Transformer architecture.** Following ECRL (ICLR 2024 Spotlight), process all entities through a permutation-equivariant transformer. Each entity gets a structured representation vector encoding its type, kinematic state, behavioral phase, and threat level. Self-attention discovers which entity-entity and entity-ninja interactions matter for current decisions. The agent learns to attend strongly to nearby death balls and drones whose patrol paths must be crossed, while ignoring distant mines. With ~10 entities per level and 2–3 attention layers, computational overhead is under 0.1ms per forward pass.

**Analytical physics features for known behaviors.** Since all entity physics are deterministic and known, pre-compute features that would otherwise require learning. For **periodic drones**: safe window timing computed from patrol path geometry, encoded as cyclic features. For **thwumps**: explicit state machine encoding (idle/charging/retracting) plus line-of-sight indicators and ride-ability flags. For **death balls**: relative velocity vectors, bearing angles, and time-to-intercept computed from the known seeking algorithm. For **bounce blocks**: analytical launch trajectories computed from spring physics, with reachable tiles from each launch encoded as features. This physics-informed feature engineering converts hard-to-learn timing relationships into directly observable inputs.

**Safety critic for hazard awareness.** Train a separate Q_safe(s, a) network estimating death probability over the next K frames. During action selection in both the base policy and MCTS, reject actions where the safety critic exceeds a threshold. Train from expert demonstrations (which demonstrate successful hazard avoidance) and from negative examples generated by the simulator. Dynamic SDFs extending the existing mine SDF infrastructure provide smooth gradient fields pointing away from all hazards — the composite `SDF_composite(x) = min(SDF_entity_1(x), ..., SDF_entity_n(x))` gives distance to the nearest threat of any type.

For **death balls specifically**, the pursuit-evasion literature (AMS-DRL, IEEE 2024) shows that agents develop sophisticated strategies including flanking, driving, and ambush maneuvers. The key insight for N++ is that the agent can learn to **manipulate death ball positions** — luring them one direction before reversing — which requires the MCTS planner to evaluate trajectories over horizons long enough to capture these multi-step strategies (15–25 macro-actions is sufficient).

For **thwumps as platforms**, hierarchical decomposition helps most. The high-level planner can include "trigger and ride thwump" as an explicit subgoal when the surface graph indicates that a thwump-assisted path is shorter. The motor controller then executes the timing-critical trigger-ride-dismount sequence. Expert demonstrations containing thwump-riding examples train this skill implicitly.

---

## Diffusion trajectory planning offers a powerful complement

**Diffusion-based trajectory generation** (Janner et al., ICML 2022; LatentDiffuser, ICLR 2024) provides a fundamentally different approach that complements MCTS. Rather than searching action-by-action, a diffusion model generates entire trajectories by iteratively denoising from random noise, conditioned on the start state and goal. This approach naturally handles **multimodal solutions** (multiple valid paths through a level), performs implicit **trajectory stitching** (combining fragments from different demonstrations), and bypasses the bootstrapping problems that plague value-based methods on long horizons.

For N++, train a conditional diffusion model on expert trajectory data with level tile-map conditioning. At test time, condition on the unseen level's layout and denoise to generate 10–100 candidate trajectory plans. Verify each against the perfect simulator (filtering physically impossible trajectories). **LatentDiffuser** (ICLR 2024) operates in a learned continuous latent action space, reducing the dimensionality of the planning problem while preserving expressiveness. The trajectory generation process takes seconds per candidate, and with 100 candidates and simulator verification, the probability of finding at least one valid solution is high for moderate-difficulty levels.

This serves as both a **zero-shot solution generator** (before any per-level adaptation) and a **warm-start for MCTS** (initial trajectories from diffusion provide promising subtrees for tree search to refine). The combination addresses a weakness of pure MCTS: tree search struggles with globally coordinated multi-step plans (like setting up a long momentum chain), whereas diffusion generates globally coherent trajectories that can then be locally refined.

---

## The complete system: a three-phase pipeline

**Phase 1 — Offline pre-training (one-time, ~48 GPU-hours on A100)**

Train four components simultaneously on the 100k+ expert demonstrations:

- **BC policy network** with entity attention, FiLM-conditioned level encoder, and frame stacking. Architecture: GNN level encoder (128-dim) → FiLM modulation of a 512-unit, 3-layer MLP with entity attention block. Output: 48-action FiGAR distribution + value estimate + safety score. Total parameters: ~5–10M (scaling up from typical RL networks, per Scaled Q-Learning findings that larger models transfer better).
- **HIQL goal-conditioned value function** on all demonstrations, learning which states are reachable from which — effectively internalizing the physics. Extract high-level subgoal policy and low-level motor controller.
- **Conditional diffusion trajectory model** mapping (level_encoding, start_state) → full trajectory. Train on demonstration trajectories conditioned on tile-map embeddings.
- **Skill library** via CompILE trajectory segmentation at surface transitions, yielding 10–20 reusable movement primitives with DADS-style predictable outcomes.

**Phase 2 — Zero-shot inference on unseen levels (seconds)**

For a new level:
1. Encode tile map → level embedding via GNN encoder
2. Build surface graph → annotate with value function estimates → A\* route planning
3. Generate 50–100 candidate trajectories via diffusion model
4. Verify candidates in simulator, rank by completion time
5. If a valid trajectory is found, execute it (done in seconds)

**Phase 3 — Per-level adaptation (minutes to hours, as needed)**

If zero-shot fails or near-optimal performance is desired:
1. **WSRL warmup**: 100 rollouts with pre-trained policy → seed replay buffer
2. **PPO fine-tuning**: 5,000–50,000 gradient steps with L2/KL regularization toward pre-trained weights. Alternatively, ResFiT residual learning on frozen BC policy for guaranteed non-degradation.
3. **Expert Iteration**: Run Gumbel AlphaZero MCTS at each decision point during rollouts → collect MCTS-improved policy/value targets → train network on improved targets → repeat 2–5 times.
4. **Adaptive compute allocation**: allocate more MCTS budget to high-uncertainty segments (near dynamic entities, tight platforming sections) and less to straightforward sections, following test-time compute scaling principles (Snell et al., 2024).

---

## Concrete recommendations ranked by expected impact

The following prioritization reflects both research evidence strength and implementation feasibility:

- **Gumbel AlphaZero with the perfect simulator** is the highest-impact single component. The zero-model-error advantage over learned dynamics is enormous, and Gumbel's guaranteed policy improvement with minimal simulations makes it practical for real-time or near-real-time planning. Start here.
- **BC pre-training with entity attention and level conditioning** establishes the policy prior that makes everything else work. Without a strong prior, MCTS branching is intractable and per-level adaptation is slow. The entity attention module and physics-informed features for dynamic entities are essential for generalization.
- **HIQL + surface graph hierarchy** provides the structural backbone for long-horizon reasoning. The surface graph is already partially built; HIQL adds offline learning of subgoal policies from demonstrations. This converts the flat 1000+ frame planning problem into a sequence of 5–15 subgoal transitions.
- **WSRL/ResFiT per-level fine-tuning** is the most practical adaptation mechanism. ResFiT's 200× sample efficiency gain and non-degradation guarantee make it particularly attractive. WSRL's warmup phase prevents the Q-function collapse that typically plagues offline-to-online transfer.
- **Diffusion trajectory generation** provides a complementary planning modality. Where MCTS excels at local refinement, diffusion excels at global trajectory coherence. The combination covers both local precision and global strategy.
- **Expert Iteration loop** pushes performance beyond demonstration quality. This is the mechanism for achieving super-expert performance on individual levels, using MCTS-improved targets to systematically distill search improvements into the network.

The system's key architectural advantage is that every component reinforces the others: BC pre-training produces the MCTS policy prior; MCTS produces improved training targets for Expert Iteration; the hierarchy reduces the horizon that MCTS must search; entity attention enables physics-informed features that make both BC and MCTS more effective; and diffusion provides global trajectory candidates that MCTS can locally refine. This is not a collection of independent techniques — it is an integrated system where each component addresses a specific weakness of the others.

## Conclusion

The proposed architecture exploits N++'s unique properties — deterministic physics, perfect simulator access, rich demonstrations, and discrete tile structure — more thoroughly than any single existing method. The critical insight is that **a known simulator transforms the planning problem fundamentally**: where MuZero's learned models fail at depth, a perfect simulator enables unbounded accurate search, making Gumbel AlphaZero-style MCTS the natural backbone. The pre-train-then-adapt paradigm, validated by NitroGen (2025) for games and WSRL/ResFiT (2025) for offline-to-online transfer, provides the practical framework. Hierarchical decomposition through the surface graph reduces the effective horizon from thousands of frames to tens of subgoals. Entity-aware attention with analytical physics features solves the dynamic entity problem without requiring per-entity-type engineering. And diffusion-based trajectory generation adds a global planning capability that pure tree search lacks. No existing system combines all of these elements — the novelty lies in their integration into a coherent pipeline specifically designed for N++'s structure. The expected outcome is an agent that can zero-shot solve moderate levels in seconds and adapt to the hardest levels within minutes, approaching or exceeding expert demonstration quality through the Expert Iteration loop's systematic improvement.