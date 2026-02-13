This is the right realization. Analytical edge weights try to model physics you already have a perfect simulator for — that's backwards. Since you can run the simulator offline, **let the simulator itself be the edge-weight oracle.**

The cleanest approach is to drop analytical physics modeling entirely and compute a **frame-count distance field** directly from exhaustive simulation.

## Why frame count IS energy cost

Frames are the universal currency in this game. Every physics interaction — gravity work, friction loss, acceleration effort, wall-jump setup time — is paid in frames. A 200-frame path is strictly more expensive than a 100-frame path regardless of which mechanics are involved. You don't need to model gravity work or friction analytically; the simulator already does that. You just need to ask: "from here, what's the minimum number of frames to reach the goal?"

This also sidesteps the velocity-dependence problem entirely for the potential function. The potential Φ(x,y) provides directional gradient — "go this way, it gets cheaper." The agent learns to exploit momentum, wall-jump chains, and launch pads through RL training itself. The potential doesn't need to encode those exploits; it just needs to not point the wrong direction.

## The approach: simulation-based transition enumeration + backward Dijkstra

**Step 1: Discretize the level into a position grid.**

Pick a resolution of ~6px (quarter-tile, about half the ninja radius). This gives a grid of 176×100 ≈ 17,600 cells over the 1056×600 pixel level. Each cell is either "inside solid geometry" (invalid) or "open space" (valid). Use the known tile collision data to classify cells.

**Step 2: For every valid cell, simulate all 6 actions for 1 frame and record the destination cell.**

Place the ninja at the cell center with zero velocity. For each of the 6 actions (left, right, neutral × jump, no-jump), run exactly 1 physics step. Record which cell the ninja lands in. This builds an empirical transition table:

```
transitions: Dict[(cell, action)] → cell'
```

This is 17,600 valid cells × 6 actions = ~105,000 simulator calls. At your reported ~0.3ms per step, that's about 30 seconds. Trivial.

For cells in mid-air (not on a surface), the simulator handles it correctly — the ninja falls under gravity, which IS the real transition from that position. Airborne cells naturally chain downward until the ninja lands or leaves the grid.

**Step 3: Build the empirical directed graph and run backward Dijkstra from each goal.**

The transition table IS your graph. Each cell is a node, each (cell, action) → cell' is a directed edge with weight 1 (one frame). Run backward Dijkstra from the switch position and from the exit position, giving you:

```
d_switch[cell] = minimum frames from cell to switch
d_exit[cell]   = minimum frames from cell to exit
```

Dijkstra on a graph with ~105k edges runs in milliseconds.

**Step 4: Store as Φ lookup grids.**

```python
phi_switch[ix, iy] = -d_switch[(ix, iy)] / max_d_switch  # normalized to [-1, 0]
phi_exit[ix, iy]   = -d_exit[(ix, iy)] / max_d_exit
```

Runtime lookup is O(1): `ix, iy = int(x / 6), int(y / 6)`.

## Handling the zero-velocity limitation

The main objection is: transitions computed from zero velocity miss momentum-dependent paths. A cell at the top of a wall-jump corridor might show d*=500 frames (requiring a full climb from rest) when the agent could arrive with upward momentum and reach the goal in 50 frames.

This matters less than it seems, for three reasons:

**The gradient direction is still correct.** Even if the absolute distance values are pessimistic, the *relative ordering* between neighboring cells is almost always right. A cell closer to the goal along a viable path will have a lower d* than a cell farther away, regardless of assumed velocity. PBRS only uses the *difference* Φ(s') − Φ(s), so absolute accuracy doesn't matter — directional accuracy does.

**The agent learns velocity exploitation through RL.** The potential says "go this direction." The agent learns "go this direction AND carry momentum." These are complementary. The potential provides the navigation gradient; the policy learns the physics exploitation.

**You can improve it with a multi-velocity pass if needed.** If the zero-velocity field proves too coarse, run the transition enumeration with a small set of representative initial velocities per cell — say (0,0), (±MAX_HOR_SPEED, 0), (0, ±3.0). Take the minimum d* across all initial velocities for each cell. This multiplies compute by 5× (still ~2.5 minutes) and captures the most important momentum-dependent transitions. This is a refinement, not a prerequisite — start with zero-velocity and measure whether the gradient is adequate before adding complexity.

## Multi-step transitions for jump discovery

One-frame transitions won't discover jumps — a jump takes multiple frames to execute and land. Extend the approach:

**Run multi-step simulations (k = 1, 2, 3, ... up to ~50 frames) with constant action held.** For each cell and each of the 6 actions, simulate holding that action for k frames and record the destination cell. Edge weight = k frames.

Since MAX_JUMP_DURATION is 45 frames, k=50 captures a full jump arc. With 6 actions × 50 durations × 17,600 cells = 5.28 million simulator calls ≈ 26 minutes. Still very feasible for offline precomputation.

This naturally discovers: jumps (hold jump for 1-45 frames), falls (hold any direction without jump), wall-slide entries (hold toward wall while falling), and running across flat surfaces (hold direction).

**For action sequences (not just constant hold):** The most important compound action is "jump then release jump at frame k" (variable jump height). Simulate these too: for each cell, simulate jump+direction for k frames then direction-only for (50-k) frames, across a sweep of k values. This roughly triples the compute but captures variable-height jumps, which are critical for platformer navigation.

## What about wall-jump chains and launch pads?

Wall-jump chains and launch pad combos are multi-step strategies that won't be discovered by single-action simulations from zero velocity at arbitrary positions. This is fine. These are exactly the physics exploits the RL agent should discover. The distance field provides the "coarse navigation mesh" — roughly correct direction and distance to goals — and the agent refines it with learned strategies.

If specific exploits are critical for level completion (some levels might be unsolvable without wall-jump chains), the distance field will show those cells as d* = ∞ (unreachable from rest). When the agent discovers the chain during training and reaches those cells, PBRS kicks in with the gradient from whatever reachable cells are nearby. The potential field doesn't need to encode every exploit — it needs to not actively mislead, and "unreachable = Φ_min" is a safe, honest signal.

## Updated implementation plan delta

Replace "Component 1: Energy-Cost Reachability Graph" in the implementation plan with:

```
PRECOMPUTATION (once per level, offline):

1. Grid setup:
   - Resolution: 6px per cell (176×100 grid)
   - Classify cells as valid/invalid using tile collision geometry
   
2. Transition enumeration:
   - For each valid cell, for each of 6 actions, for k in [1..50]:
     - Place ninja at cell center, zero velocity
     - Simulate k frames holding that action
     - Record destination cell
     - Edge: (source_cell, action, k) → dest_cell, weight = k
   - Optional: also simulate variable-jump-height sequences
   
3. Backward Dijkstra:
   - From switch position cell → d_switch[cell] for all cells
   - From exit position cell → d_exit[cell] for all cells
   - Unreachable cells get d = d_max (not infinity)
   
4. Normalize and store:
   - phi_switch[cell] = -d_switch[cell] / max(d_switch)
   - phi_exit[cell] = -d_exit[cell] / max(d_exit)
   - Φ lookup is O(1) at runtime: ix = x // 6, iy = y // 6

Total precompute time: ~30 minutes per level (parallelizable across levels)
Runtime cost per step: one array lookup
```

Everything downstream (reward machine, PBRS formula, FiGAR-PPO integration) stays exactly the same. The only change is *how* the Φ grid is computed — simulation-based instead of analytical.