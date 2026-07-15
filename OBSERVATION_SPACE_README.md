# Observation Space Reference

The policy ("Layer 1") observation is a `gymnasium` `Dict` of **four structured
tensors plus control masks**. It is scoped to the generalist agent's mechanics:
**toggle mines, one exit switch, one exit door, one player spawn, and any tile
layout** (levels are a fixed 42×23 inner-tile size). Everything else (gold, locked
doors, one-way platforms, drones, launch pads, bounce blocks, thwumps) is out of
scope and not represented.

> Graph tensors, `reachability_features`, `spatial_context`, and visual frames
> described in older revisions of this file are **gone** — they are not produced
> for the policy. Reward-side infrastructure still builds a graph internally for
> PBRS, but it never reaches the rollout buffer.

## Contract

```python
observation_space = Dict({
    "tile_grid":       Box(0, 37, (42, 23), int8),       # raw tile IDs, axis order (W, H)
    "entity_overlays": Box(0, 1,  (5, 42, 23), float32), # (C, W, H)
    "hazard_map":      Box(-1, 1, (2, 168, 92), float32),# (C, W, H), 6px grid
    "game_state":      Box(-1, 1, (183,), float32),      # GAME_STATE_CHANNELS
    "action_mask":     Box(0, 1, (6,), int8),
    # "duration_mask": Box(0, 1, (6, n_durations), int8)  # only with FiGAR macro mask
    # --- optional exposure keys (flag-gated; see ReachabilityConfig) ---
    # "cost_to_go_map":  Box(0, 1, (3, 84, 46), float32)   # expose_cost_to_go_map
    # "waypoint_nav"/"lattice_nav": Box(-1, 1, (4,), f32)  # expose_*_features
    # "ego_crop":        Box(0, 37, (21, 21), int8)        # enable_ego_crop
    # "ego_crop_mines":  Box(0, inf, (2, 21, 21), f32)     # enable_ego_crop
    # "ego_crop_route":  Box(-1, 1, (2, 21, 21), f32)      # enable_ego_crop (v14 B2)
    # "ego_crop_coarse": Box(0, inf, (4, 21, 21), f32)     # enable_ego_crop_coarse (P1-6)
})
```

All spatial tensors share the **(W, H)** axis convention (`W=42/168`, `H=23/92`),
so `tile_grid`, `entity_overlays`, and an upsampled `hazard_map` co-register
without a transpose.

### `tile_grid` — `(42, 23)` int8, `[0, 37]`

Raw N++ tile IDs for the inner playable grid (38 categorical types: empty, full,
half tiles, slopes, curves, …). These are **categorical**, not ordinal — embed
them (e.g. `nn.Embedding(38, d)`) in the policy; do not feed as a scalar channel.

### `entity_overlays` — `(5, 42, 23)` float32, `[0, 1]`

Per-tile entity overlay grid. Channels:

| ch | name | encoding |
|----|------|----------|
| 0 | `ninja_pos`          | bilinear sub-tile splat at the ninja position (sums to 1) |
| 1 | `switch_pos`         | bilinear sub-tile splat at the exit switch, **omitted once activated** |
| 2 | `exit_pos`           | bilinear sub-tile splat at the exit door |
| 3 | `mine_deadly_density`| count of deadly (toggled) mines whose 14px kill-zone overlaps the tile, `/8` clamp |
| 4 | `mine_safe_density`  | count of safe (untoggled/toggling) mines per tile, `/8` clamp |

Channels 0–2 use a **bilinear sub-tile splat**: static entities sit on a 6px
grid, so a 24px tile holds 16 distinguishable positions. The splat spreads unit
mass over the four surrounding cells in proportion, preserving the exact sub-tile
location that a single-cell write would quantize away.

### `hazard_map` — `(2, 168, 92)` float32, `[-1, 1]`

Two mine signed-distance fields on a **6px grid** over the 1008×552px playable
area. Each cell holds `clip((d - 14) / 24, -1, 1)` where `d` is the px distance to
the nearest mine of that class:

| ch | name | mines |
|----|------|-------|
| 0 | `deadly_sdf` | toggled / deadly mines |
| 1 | `safe_sdf`   | untoggled / safe mines |

Positive outside the 14px kill zone (`= ninja radius 10 + mine radius 4`, the true
death-contact boundary), zero at the boundary, negative inside. Recomputed only
when a mine set changes (e.g. a toggle); mines sealed from reachable space are
blinded out. The two channels give the policy precise (6px) geometry for both mine
states, not just the coarse 24px density in `entity_overlays`.

### `game_state` — `(183,)` float32, `[-1, 1]`

Flat ninja-physics + goal-geometry + execution vector. See
`nclone.gym_environment.constants` for the named indices
(`GAME_STATE_CHANNELS == 183` as of the v14 tap-arc + wall-jump-now rows; the
v13 ghost-arc blocks — committed + jump-now arcs + arc-phi route-progress
deltas — end at 128).

| idx | feature |
|-----|---------|
| 0–1 | velocity (x, y) / MAX_HOR_SPEED |
| 2–5 | movement-state one-hot-ish (ground / air / wall / special) |
| 6 | airborne |
| 7–9 | jump / floor / wall buffers |
| 10 | jump_duration_remaining |
| 11–13 | floor / wall / ceiling contact |
| 14–16 | floor normal (x, y), wall normal |
| 17 | walled |
| 18–20 | applied gravity / drag / friction (normalized) |
| 21–22 | ninja position (x, y) / level extent |
| 23 | time_remaining (curriculum-aware truncation) |
| 24–26 | physics-derived: horizontal_range, max_jump_height, time_to_ground |
| 27 | `switch_activated` phase bit (**critical for the Markov two-phase task**) |
| 28–29 | ninja-relative vector to the exit **switch**, `(dx, dy)` / level extent, clipped |
| 30–31 | ninja-relative vector to the exit **door**, `(dx, dy)` / level extent, clipped |
| 32–39 | surface ray-casts: normalized distance `[0,1]` to the nearest solid surface in 8 compass directions `[E, NE, N, NW, W, SW, S, SE]` (0 = adjacent, 1 = ≥500px) |
| 40–43 | free-space nav hint `[dx_hop, dy_hop, cost_to_go_norm, phase]` |
| 44 | nearest deadly-mine distance, normalized `[0,1]` (0 = touching, 1 = none within 168px) |
| 45 | closing speed toward the current objective, `dot(velocity, goal_dir)/MAX_HOR_SPEED` clipped `[-1,1]` |
| 46–48 | jump-execution: jump_held, jump_useless, launch-pad buffer window |
| 49–51 | look-ahead landing surface: normal (nx, ny) + normalized distance |
| 52 | signed vertical terminal-velocity fraction |
| 53–56 | execution timing: time-to-contact (straight-ray τ/60), frames airborne, per-frame accel (ax, ay) |
| 57–68 | forward-biased terrain scandots: (floor-gap, ceiling-gap) per column for 6 columns |
| 69–95 | **committed ghost-arc bundle (Audit II P0-1)**: for u ∈ {−1, 0, +1}, 9 dims per arc — `[dx/500, dy/500, nx, ny, mine_flag, none_flag, clear_width/9, impact_ratio/2, tau/60]` from the native ballistic projection of the real airborne recurrence to first contact. The ±1 arcs are the bang-bang reachable-landing interval endpoints; `clear_width` is the contiguous mine-free standable extent at the contact; grounded frames use the fast-path fill (zeros + ny = −1). |
| 96–122 | **jump-now ghost-arc bundle (v13)**: same 3×9 layout, but the *counterfactual full-hold takeoff* while GROUNDED — flat-floor jump impulse exactly as `Ninja.floor_jump` applies it (`yspeed = min(yspeed,0) + JUMP_FLOOR_Y`, position displaced by the impulse), then `MAX_JUMP_DURATION` frames of reduced gravity. Answers "if I jump right now, where do each of the three steering arcs land, and is it survivable?" — takeoff-decision information the committed bundle (airborne-only) cannot provide. Zeros while airborne. |
| 123–128 | **arc route-progress deltas (v13)**: per arc, `tanh((Φ(ninja) − Φ(landing)) / 150)` from the committed free-space/jump cost-to-completion source — positive = that arc's landing makes route progress, negative = it flies away from the route. Slots 0–2 = committed arcs (airborne), slots 3–5 = jump-now arcs (grounded); inactive/mine/no-contact/inf-cost slots emit 0. Links the geometric arc bundle to the *reward's own* cost-to-go field, so "which arc is the route" is directly observable instead of learned. |
| 129–155 | **tap-jump arc bundle (v14 B1)**: same 3×9 layout as the jump-now row, but the takeoff is *released immediately* (`jf = 0` → full gravity after the impulse). The v13 jump-now arcs are full-hold only; under a low ceiling (e.g. LOOP_CORRIDOR's 48px mined-walkway headroom) all three slam the ceiling, so the move that actually clears a mine hop — a tap jump — was invisible to the policy. Zeros while airborne. |
| 156–182 | **wall-jump-now arc bundle (v14)**: same 3×9 layout — the counterfactual *regular wall-jump kick this frame* while AIRBORNE + WALLED (`Ninja.wall_jump`: xspeed zeroed if into the wall, downward yspeed zeroed, `+1·wall_normal / −1.4` impulse, position displaced, full hold). The committed bundle on wall-slide frames shows only the slide/fall continuation; this row makes the kick affordance visible — the third contact regime's takeoff, targeting LOOP_CORRIDOR shaft chains / VERTICAL_SHAFT. Zeros when grounded or un-walled. |

Indices 28–31 are exact on the 6px entity grid and give the policy precise goal
geometry that the 24px overlay channels cannot. Both goal vectors are always
present (continuous across the switch→exit transition); the `switch_activated` bit
selects which is the live objective.

Indices 32–39 (**surface ray-casts**) march the tile grid to the nearest solid
surface per compass direction — wall sensing and blind-jump / gap verification
beyond the local tile view, in microseconds.

Indices 40–43 (**free-space nav hint**) are a physics-free steepest-descent over a
mine-aware free-space cost-to-go toward the current objective: `dx_hop, dy_hop` is
the next-step direction that **rounds obstacles** (vs the straight-line goal
vectors), `cost_to_go_norm` is the normalized remaining cost, `phase` mirrors
`switch_activated`. It is an **observation only** — never reuse it as a reward
potential (that would double-count against the PBRS term). Zeros when the spawn
has no finite free-space route (severed / airborne levels).

## Validation

```bash
.venv/bin/python -m pytest tests/test_entity_overlays.py tests/test_hazard_map.py \
    tests/test_l1_game_state_switch_activated.py tests/test_layer1_mines_fixtures.py
```
