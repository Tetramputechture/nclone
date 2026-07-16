The honest answer depends on which levels you're targeting. Here is a calibrated take.

## The short answer

For entity-free and static-entity levels (tiles, mines, launch pads, bounce blocks, locked doors, one-ways), a purely deterministic non-ML approach will work, and probably work well. Your simulator is 1 µs/frame, deterministic, fully observable, and the Dijkstra solution through the velocity-augmented graph is the optimal policy for these levels, not an approximation of it.

For levels with drones, death balls, or charged thwumps, non-ML runs into a structural wall. These entities create time-dependent constraints the graph cannot model, and the workarounds are either intractable or brittle.

The real question is where we draw the line between the two. Your own research already has the answer embedded in it: Tier 1/2 entities are deterministic-solvable, Tier 3 needs learning or search.

## What your research actually says

Looking across the documents, there are three converging claims about what the simulator's determinism buys you.

First, `VALUE_PRECOMPUTATION_SPEC` says the graph's `d*` is the exact optimal value function for the entity-free subproblem, not an approximation. That's already a non-ML solution sitting inside an ML pipeline.

Second, Hatch & Boots (2023), cited throughout your Physics-Aware Route Planning doc, formalize the architecture where "the precomputed graph IS a value function over the entire state space" and "when perturbed, the controller queries existing tree nodes rather than replanning." That's textbook non-ML closed-loop control.

Third, your Blueprint doc makes the "perfect simulator" argument central: Gumbel MuZero beats TD-MPC2/Dreamer here because the simulator is ground truth, and learned dynamics models are strictly worse when ground truth is available at 1 µs/step. That argument cuts deeper than the doc acknowledges. If a learned world model is worse than the simulator, then in regimes where the simulator alone suffices, the policy doesn't need to be learned either.

## Four non-ML architectures worth considering

Ordered by sophistication, each exploits more of the simulator's determinism.

1. Pure open-loop replay. Dijkstra on the velocity-augmented graph gives you a waypoint sequence. For each arc, the edge enumeration already recorded the exact `(h_input, jump_duration, post_release_input)` tuple that produced it. Concatenate those input schedules and replay: no RL, no runtime logic, no policy, just a tape. This works for any level the graph can solve without dynamic entities. Speedrun tool-assisted runs (TAS) for momentum platformers are literally this.

2. Closed-loop graph tracking. Same graph, but at each frame: (a) look up your current state's nearest graph node, (b) execute the "best outgoing edge" from that node's stored predecessor pointer, (c) re-query if you deviate. This is Hatch & Boots 2023's MPC-over-planning-tree. It handles small physics perturbations, off-by-one-frame errors, and minor binning snap-through without replanning. Still pure table lookup.

3. Graph + on-demand simulator MCTS. Use the graph `d*` as a leaf value, run Gumbel AlphaZero-style search at decision points with the native simulator, n=16-32. No policy network is needed: a uniform prior is fine because the leaf value is near-optimal. This handles drones and timing-sensitive segments, since MCTS can forward-simulate the drone's deterministic patrol and find safe windows without any learned component. Still no ML.

4. Graph + width-based planning (RolloutIW). Same structure as (3), with novelty-based search (Lipovetzky-Geffner) replacing MCTS. Factored state (pos, vel, entity states) is exactly the regime where IW(1)/RolloutIW matches or beats UCT on Atari. This is arguably the single most underrated non-ML direction for your setup and deserves direct experimentation.

Architectures (3) and (4) can likely handle drones and thwumps. They're deterministic, the simulator can roll them forward exactly, and MCTS/IW just needs to find timing. Death balls are the borderline case because they're agent-reactive over long horizons, which forces deeper search trees.

## Where non-ML breaks, and why

Three failure modes matter.

Velocity-bin snap-through at contact thresholds. The research report I compiled noted this. N++ is a hybrid system, and `V*` is only piecewise Lipschitz across mode boundaries (contact vs airborne, jump-hold vs released). Whether a velocity bin rounds to "can reach wall-jump window" vs "cannot" is binary. Pure graph lookup will occasionally pick an edge that works on the lattice but not in the continuous simulator. You'd see this as "plan says this is 15 frames away, agent dies on frame 14." Architectures (2), (3), (4) handle this by re-querying or searching when deviation grows; pure (1) doesn't.

Edge-set incompleteness. Your `(3 × 11)`, `(3 × 12 × 3)`, `(2 × 3 × 12)` sweeps won't cover every frame-perfect trick. For Tier 1/2 levels this is usually fine because Dijkstra composes atomic edges and most levels don't need frame-perfect tricks. For hard levels it bites. The ML-free fix is to add custom primitives for specific gaps. The ML fix is a learned policy that finds the tricks during online rollouts.

Agent-reactive entities (death balls, thwump timing in tight spots). Death balls apply `a_x, a_y` toward the ninja with a max-speed clamp, so their trajectory is a function of your policy, independent of it. MCTS can handle this if horizons are ≥15-25 macro-actions (your Blueprint doc's number, backed by AMS-DRL 2024 pursuit-evasion literature), but the search branches can't be amortized across levels the way a learned policy can. At procgen scale this becomes expensive.

## So which do you pick?

Given what you've already built, think about it as three questions in sequence rather than a single binary choice.

Question 1: are you targeting a bounded level set (say <1000 hand-designed levels) or an open-ended one (procgen, user-generated, 10k+)? If bounded and mostly Tier 1/2, pure non-ML architecture (3), graph + simulator MCTS at decision points, is probably the right terminal state and will beat an ML approach in reliability, explainability, and debugging. Your research already endorses Gumbel MuZero with perfect simulator over learned alternatives for this exact reason; you can go one step further and skip the learned policy prior too, because a uniform prior plus near-optimal `d*` leaf value is strong enough.

Question 2: how many levels feature drones, death balls, or thwumps in agent-reactive configurations? If it's a small fraction, handle those levels with per-level MCTS deeper search. If it's most levels, the amortization argument for ML gets stronger, because learning drone-timing priors across levels is exactly what a learned policy does well and what MCTS has to rediscover each time.

Question 3: do you need sub-second inference at deployment, or is multi-second-per-decision fine? Pure ML is the only answer if you need tight inference latency across many levels. Non-ML + MCTS is fine if you have budget.

## My actual recommendation

Build the non-ML stack first. Measure where it breaks.

Concretely: implement architecture (3), graph cost-to-go + closed-loop tracking + Gumbel MCTS at dynamic-entity segments, with zero learned components. Your simulator is 1 µs/frame, your graph is already built, and Gumbel-Top-k + sequential halving is ~200 lines of code. Run it on your full level distribution. You'll likely find it solves 70-90% of levels outright. The levels it fails on are your actual ML budget, and you'll know why they fail, which lets you choose the ML method that addresses that specific failure mode (policy prior for MCTS branching, learned safety critic for death balls, quasimetric V for procgen generalization) rather than committing to a large pipeline up front and discovering later which parts were unnecessary.

This is also the Go-Explore playbook (Ecoffet et al. Nature 2021), which your Blueprint doc cites: solve first by any deterministic means, robustify second via imitation only if needed. Your existing infrastructure is already 80% of the deterministic-solve phase. Finishing it before committing to RL training costs you weeks and saves you from discovering at 200 GPU-hours in that you overbuilt.

One caveat: if the near-term goal is a generalizable agent for open-ended procgen levels with heavy Tier 3 entity density, then ML is required. That's a specific constraint. The default for a deterministic simulator with known physics and mostly-static entities is that ML is optional, and your research, read carefully, agrees.
