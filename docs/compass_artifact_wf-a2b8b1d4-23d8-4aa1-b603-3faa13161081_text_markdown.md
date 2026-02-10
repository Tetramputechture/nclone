# Semi-Lagrangian value iteration solves the slow-dynamics problem

**The core issue — value functions going flat within 12px cells because the agent moves only 1.5px/frame — is a well-understood problem in numerical optimal control, and semi-Lagrangian (SL) methods solve it directly.** Rather than snapping transitions to grid nodes (which makes them stay in the same cell for ~8 frames), SL methods compute exact continuous-space departure points using the simulator and interpolate the value function at those points. This captures sub-grid variation naturally, producing meaningful gradients for PBRS even within coarse cells. At 15.4M states (60MB), the problem is trivially GPU-feasible with tabular methods, making neural function approximation unnecessary complexity. The recommended approach — SL value iteration in JAX with multilinear interpolation — should converge in minutes to tens of minutes per level on a consumer GPU.

---

## Why standard grid VI fails and semi-Lagrangian fixes it

Standard grid-based value iteration discretizes both states *and transitions*. When the agent moves 1.5px/frame on a 12px grid, the discretized transition maps most state-action pairs back to the same cell. The Bellman backup becomes V(s) ← max_a [R + γV(s)], producing a flat value landscape within cells — zero gradient for PBRS across ~8 consecutive frames.

Semi-Lagrangian value iteration takes a fundamentally different approach. For each grid point **x_i** and each action **a**, it:

1. Computes the exact next state x' = f(x_i, a) using the physics simulator in **continuous space**
2. Interpolates V at x' from surrounding grid values (since x' generally falls between grid points)
3. Sets V(x_i) = max_a [R(x_i, a) + γ · Interpolate(V, x')]

The critical insight is that even a 1.5px displacement is captured precisely as a fractional position within the cell. **Multilinear interpolation then weights the surrounding grid values differently depending on the exact sub-grid position**, so the Bellman backup produces distinct values for neighboring grid points even when their transitions land in the same cell. Over successive iterations, this sub-grid information propagates through the entire value function, creating meaningful within-cell gradients.

This is the standard numerical method for Hamilton-Jacobi-Bellman equations in optimal control, extensively analyzed by Falcone & Ferretti (SIAM, 2014). A key theoretical property: SL schemes are **unconditionally stable** regardless of the ratio between grid spacing and movement speed (no CFL condition required), making them ideal for exactly this "slow dynamics" regime.

---

## The practical recipe: tabular SL-VI in JAX with 4D interpolation

The implementation is straightforward because the problem has favorable structure. The **jump_frames_remaining** dimension is discrete with small cardinality (~10–20 values) and transitions deterministically (decrements by 1 while jumping, resets otherwise). This means no interpolation is needed in that dimension — store V as a stack of 4D arrays indexed by jump state. The continuous interpolation operates only over **(x, y, vx, vy)**, requiring **2⁴ = 16 grid lookups per interpolation** rather than 2⁵ = 32.

The per-iteration computation for 15.4M states × 6 actions involves ~92M physics evaluations and ~92M 4D multilinear interpolations (each just 16 lookups plus a weighted sum). On a modern GPU using JAX's `vmap` and `jit`, this runs in **1–5 seconds per iteration**. With γ = 0.99, convergence typically requires **500–1500 iterations**, putting total compute at roughly **10–120 minutes per level** — well within budget.

**Memory is not a constraint.** The value function occupies ~60MB at float32. Even storing all 6 Q-values simultaneously requires only ~360MB. A consumer GPU with 8GB has ample headroom.

The main engineering challenge is reimplementing the N++ physics simulator as a pure JAX function (no Python control flow, using `jax.lax.cond` for branches). Collision detection with level geometry should use pre-computed collision maps stored as arrays for vectorized lookup. Once this is done, the entire VI loop can be JIT-compiled into a single GPU kernel.

**Multilinear interpolation preserves the contraction property of the Bellman operator**, guaranteeing convergence to V* — a result proven by Judd and by Falcone & Ferretti. Higher-order cubic interpolation achieves O(Δx⁴) accuracy versus O(Δx²) for linear, but **cubic interpolation violates monotonicity** and can create oscillations near value function discontinuities (which occur at obstacle boundaries). For a platformer with sharp geometric features, multilinear is the safer and recommended choice.

---

## Discretization strategy: finer position, coarser velocity

The "slow dynamics" problem is primarily a **position resolution** issue. The agent's position changes slowly (~1.5px/frame), but velocity changes rapidly — gravity alone adds ~0.5–1 px/frame² per frame, and friction/drag create fast velocity dynamics. This asymmetry means position dimensions need finer resolution than velocity dimensions.

A highly effective rebalancing: **6px position resolution with 1.0 px/frame velocity resolution**. This yields ~4 frames per cell crossing (2× better gradient signal) while the coarser velocity grid (2× fewer bins per velocity axis) keeps total states manageable. The position grid grows from 88×50 = 4,400 to 176×100 = 17,600 cells (4×), while the velocity grid shrinks from 41×41 ≈ 1,681 to 21×21 = 441 bins (0.26×). Net effect: roughly the same total state count (~77M states, ~300MB) but with dramatically better position-dependent gradients.

| Configuration | Position cells | Velocity bins | Jump | Total states | Memory |
|---|---|---|---|---|---|
| Current: 12px / 0.5 vel | 4,400 | 1,681 | 10 | ~15.4M* | ~60 MB |
| **Recommended: 6px / 1.0 vel** | 17,600 | 441 | 10 | **~77M** | **~300 MB** |
| Aggressive: 3px / 1.0 vel | 70,400 | 441 | 10 | ~310M | ~1.2 GB |
| Maximum: 3px / 0.5 vel | 70,400 | 1,681 | 10 | ~1.18B | ~4.5 GB |

*Your reported 15.4M suggests velocity ranges narrower than ±10 px/frame; scale accordingly.

Even the 3px/1.0vel configuration at ~310M states and ~1.2GB is GPU-feasible on a 24GB card, giving ~2 frames per cell crossing. But the 6px configuration offers the best complexity-to-gradient tradeoff and is the recommended starting point. With SL interpolation, even 12px resolution provides non-zero within-cell gradients — the finer grid simply makes them more accurate.

---

## Accelerating convergence: coarse-to-fine initialization and prioritized sweeping

A critical but often overlooked issue: **information propagation speed in value iteration**. Value information advances one transition per iteration. For a level 1056px wide with a 1.5px/frame agent, the value signal from the goal must propagate ~704 frames worth of transitions to reach the far end. This means **at least ~700 iterations** regardless of grid resolution — it's a physics-limited quantity.

Two techniques dramatically reduce the iteration count:

**Coarse-to-fine (one-way multigrid)** solves VI on a coarse grid first (12px position, ~15M states, converges in minutes), then uses multilinear interpolation to initialize the fine grid (6px, ~77M states). The fine-grid VI then starts very close to V* and needs only ~50–200 iterations to converge rather than ~700+. Chow & Tsitsiklis (1991) proved this approach is asymptotically optimal in computational cost. Alla & Falcone (2013) reported **4–8× speedups** using a hybrid where the fine-grid phase uses policy iteration instead of value iteration.

**Prioritized sweeping** updates states in order of largest Bellman residual rather than sweeping the entire grid uniformly. For sparse-reward problems (like reaching a goal), this focuses computation on the expanding frontier of useful value information. Combined with coarse-to-fine, this can reduce total compute by another order of magnitude.

For the PBRS application specifically, the coarse-grid value function itself can serve as an initial shaping potential while the fine-grid solution computes — Grześ & Kudenko (2008) demonstrated this multigrid reward shaping approach explicitly.

---

## When to consider alternatives beyond tabular SL-VI

**Adaptive mesh refinement** becomes worthwhile if uniform grids are too large. Munos & Moore (2002) developed the definitive framework: start coarse, solve, then selectively refine cells where the **Stdev-Inf criterion** (product of discretization-induced variance and influence on other states) is highest. This targets resolution where it simultaneously improves accuracy and matters for the policy — not just where the value function has large gradients. In practice, AMR achieves comparable accuracy to uniform fine grids with **3–10× fewer cells**. The tradeoff is implementation complexity: kd-tree data structures, Kuhn triangulations for interpolation (5! = 120 simplices per 5D cell), and incremental MDP rebuilding. For your problem, the 6px uniform grid is likely sufficient, making AMR a "nice to have" rather than a necessity.

**Fitted value iteration with neural networks** is the right tool if the state space grows beyond ~1B states (e.g., adding enemy positions or global timer dimensions). Ernst et al.'s Extra-Trees approach is notably more stable than neural networks — tree ensembles partition the state space adaptively and don't suffer from the "deadly triad" of function approximation + bootstrapping. For your current 5D problem, FVI is unnecessary overhead. Where it shines: if you later want to generalize across levels or add continuous dimensions.

**DeepReach** (Bansal & Tomlin, 2021) uses sinusoidal neural networks to solve Hamilton-Jacobi equations without grids, demonstrated up to 10D. It can handle your 5D problem but requires reformulating from discounted infinite-horizon (what you need) to finite-time reachability (what DeepReach solves). The accuracy degrades in higher dimensions (~18–19% boundary violation rate at 9D). For 5D with a tractable grid, it offers no advantage over exact tabular methods.

**Kernel/RBF methods** (Ormoneit & Sen, 2002) provide the strongest convergence guarantees among function approximators but scale as O(n²) in sample count — impractical at millions of states. They're best suited as the regression engine within FVI using ~10K–100K representative states, not for full-resolution computation.

---

## PBRS gradient quality: interpolation is sufficient

The shaping reward F(s, s') = γΦ(s') − Φ(s) with Φ = V* only provides a directional signal when Φ(s') ≠ Φ(s)/γ. With flat tabular values (no interpolation), states within the same cell produce F = (γ−1)V — a constant negative bias with no directional information. This is the fundamental problem.

**Multilinear interpolation of the tabular V* solves this completely.** At any continuous-space query point, the interpolated value is a weighted combination of 16 surrounding grid values (in 4D continuous space). Moving 1.5px within a 12px cell shifts the interpolation weights, producing a **piecewise-linear gradient** with error O(h²) relative to the true value function gradient. This gradient points toward higher-value regions (closer to goal, better momentum states) at every frame, not just at cell boundaries.

Empirically, PBRS only needs to provide a **directional hint**, not exact gradients. Ng et al. (1999) showed that even rough potential functions dramatically accelerate learning. Wiewiora (2003) proved that PBRS with Φ = V* is equivalent to initializing Q-values to V* — so the question reduces to "is linearly interpolated V* a good Q-value initialization?" For smooth physics with smooth value functions, the answer is yes.

If sharper gradients are needed, two options exist: (1) use a finer grid (6px or 3px), or (2) after computing tabular V*, fit a small neural network to the table for smooth analytical gradients during RL training. The network training is a simple supervised regression on 15M labeled points — fast, stable, and doesn't involve any Bellman bootstrapping.

---

## Conclusion

The recommended implementation path has three tiers of increasing complexity, each sufficient on its own:

**Tier 1 (simplest, start here):** Keep the current 12px/0.5vel grid but switch from discrete-transition VI to semi-Lagrangian VI with 4D multilinear interpolation. Use the exact physics simulator for continuous-space transitions. Implement in JAX with `vmap`/`jit`. This alone eliminates the flat-gradient problem. Expected compute: ~30–60 minutes per level.

**Tier 2 (recommended):** Rebalance to 6px position / 1.0 velocity resolution (~77M states). Use coarse-to-fine initialization: solve at 12px first, interpolate to initialize the 6px grid, then converge. This gives 2× better gradients with faster convergence. Expected compute: ~1–2 hours per level.

**Tier 3 (if needed):** Add adaptive mesh refinement using Munos & Moore's framework, concentrating 3px resolution near obstacles and policy boundaries while keeping 12px in open space. Or fit a neural network to the converged tabular V* for smoother PBRS queries.

The key technical insight is that this problem sits in a sweet spot: large enough that naive grid methods produce poor gradients, but small enough that exact tabular methods with proper interpolation vastly outperform approximate function-approximation methods. Semi-Lagrangian value iteration is the textbook solution — it was designed precisely for dynamics that don't align neatly with grid spacing.