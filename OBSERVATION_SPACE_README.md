# Observation Space Reference

The policy ("Layer 1") observation is a `gymnasium` `Dict` of four structured
tensors plus control masks. It is scoped to the generalist agent's mechanics:
toggle mines, one exit switch, one exit door, one player spawn, and any tile
layout (levels are a fixed 42×23 inner-tile size). Everything else (gold, locked
doors, one-way platforms, drones, launch pads, bounce blocks, thwumps) is out of
scope and not represented.

> Graph tensors, `reachability_features`, `spatial_context`, and visual frames
> described in older revisions of this file are gone. They are not produced
> for the policy. Reward-side infrastructure still builds a graph internally for
> PBRS, but it never reaches the rollout buffer.

## Contract

```python
observation_space = Dict({
    "tile_grid":       Box(0, 37, (42, 23), int8),       # raw tile IDs, axis order (W, H)
    "entity_overlays": Box(0, 1,  (5, 42, 23), float32), # (C, W, H)
    "hazard_map":      Box(-1, 1, (2, 168, 92), float32),# (C, W, H), 6px grid
    "game_state":      Box(-1, 1, (261,), float32),      # GAME_STATE_CHANNELS
    "action_mask":     Box(0, 1, (6,), int8),
    # "duration_mask": Box(0, 1, (6, n_durations), int8)  # only with FiGAR macro mask
    # --- optional exposure keys (flag-gated; see ReachabilityConfig) ---
    # "cost_to_go_map":  Box(0, 1, (3, 84, 46), float32)   # expose_cost_to_go_map
    # "waypoint_nav"/"lattice_nav": Box(-1, 1, (4,), f32)  # expose_*_features
    # "ego_crop":        Box(0, 37, (21, 21), int8)        # enable_ego_crop
    # "ego_crop_mines":  Box(0, inf, (2, 21, 21), f32)     # enable_ego_crop
    # "ego_crop_route":  Box(-1, 1, (2, 21, 21), f32)      # enable_ego_crop (v14 B2)
    # "ego_crop_coarse": Box(0, inf, (4, 21, 21), f32)     # enable_ego_crop_coarse (P1-6)
    # "drone_hazard_map": Box(-1, 1, (4, 84, 46), f32)     # enable_drone_hazard
})
```

All spatial tensors share the (W, H) axis convention (`W=42/168`, `H=23/92`),
so `tile_grid`, `entity_overlays`, and an upsampled `hazard_map` co-register
without a transpose.

### `tile_grid`: `(42, 23)` int8, `[0, 37]`

Raw N++ tile IDs for the inner playable grid (38 categorical types: empty, full,
half tiles, slopes, curves, and so on). These are categorical, not ordinal. Embed
them (e.g. `nn.Embedding(38, d)`) in the policy; do not feed as a scalar channel.

### `entity_overlays`: `(5, 42, 23)` float32, `[0, 1]`

Per-tile entity overlay grid. Channels:

| ch | name | encoding |
|----|------|----------|
| 0 | `ninja_pos`          | bilinear sub-tile splat at the ninja position (sums to 1) |
| 1 | `switch_pos`         | bilinear sub-tile splat at the exit switch, omitted once activated |
| 2 | `exit_pos`           | bilinear sub-tile splat at the exit door |
| 3 | `mine_deadly_density`| count of deadly (toggled) mines whose 14px kill-zone overlaps the tile, `/8` clamp |
| 4 | `mine_safe_density`  | count of safe (untoggled/toggling) mines per tile, `/8` clamp |

Channels 0-2 use a bilinear sub-tile splat: static entities sit on a 6px
grid, so a 24px tile holds 16 distinguishable positions. The splat spreads unit
mass over the four surrounding cells in proportion, preserving the exact sub-tile
location that a single-cell write would quantize away.

### `hazard_map`: `(2, 168, 92)` float32, `[-1, 1]`

Two mine signed-distance fields on a 6px grid over the 1008×552px playable
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
states, beyond the coarse 24px density in `entity_overlays`.

### `drone_hazard_map`: `(4, 84, 46)` float32, `[-1, 1]` — optional

Off by default; enabled with `ReachabilityConfig.enable_drone_hazard`. Signed
distance to the nearest lethal patrol drone on the same 6px field as
`hazard_map` (emitted min-pooled to 12px like it), so the two co-register and
share units — now, and at each forecast horizon:

| ch | name | encoding |
|----|------|----------|
| 0 | `drone_sdf` | `clip((d - kill_radius) / 24, -1, 1)` to the nearest zap (type 14) or mini (type 26) drone, **this frame** |
| 1 | `drone_sdf_t12` | the same field 12 sim frames from now |
| 2 | `drone_sdf_t24` | the same field 24 sim frames from now |
| 3 | `drone_sdf_t48` | the same field 48 sim frames from now |

Channel count is `1 + len(DRONE_HAZARD_HORIZONS)`; the horizons are the
authoritative list in `constants.py`.

The kill radius is per class — 17.5px for a zap drone (`NINJA_RADIUS 10` + 7.5),
14px for a mini (10 + 4) — so the two classes are evaluated separately and merged
by element-wise **minimum**, which keeps the worst case rather than averaging a
threat away. The same reasoning drives the min-pooling to the policy grid.

Without this key a patrolling drone is indistinguishable from empty space: the
rest of the hazard pipeline is mine-shaped, and the native backend's
`entity_dic` deliberately whitelists only static entity types, so drones are
read from the live sim snapshot instead (`get_drone_positions`).

Unlike `hazard_map` there is **no change-gate** — mines only enter or leave the
deadly set on a `state` toggle, but drones move every frame. The engine bounds
cost by scattering each drone into its `kill_radius + 24px` influence box
(beyond it the SDF is exactly 1.0) rather than querying all 15,456 cells:
~20µs/frame per channel on a 28-drone level, ~200µs for the whole key including
the roll-out and the pooling.

#### The forecast channels are exact, not predicted

Channel 0 alone says only "a drone is there now". It cannot distinguish a
corridor whose drone is walking away from one whose drone is walking in, and
those demand opposite actions from an identical frame — so timing is not
learnable from it.

Zap and mini drones patrol a fixed grid with **no ninja dependence**, so their
future positions are a lookup, not a prediction: channel `i+1` is bit-for-bit
what channel 0 will read `DRONE_HAZARD_HORIZONS[i]` frames later, pinned by
`test_forecast_matches_the_sim_it_predicts` against the simulator itself. Mines
have no analogue of this. The roll-out drives the simulator's own motion code
(`Drone::forecast` / `EntityDroneBase.predict_positions`) rather than a second
copy of the patrol rules, which would drift silently.

Horizons are in **sim frames**, not env steps: at the training `frame_skip` of 4
they are 3, 6 and 12 actions ahead. A zap drone covers a 24px cell in 21 frames,
so 12/24/48 span roughly half a cell, one cell and two cells, while the ninja at
full run speed covers ~6.7 cells in 48 frames — the range over which a route
commitment is still revocable.

The chaser drone (type 15) is deliberately excluded from both the present-frame
read and the forecast. Its movement depends on the ninja, so unlike types 14/26
it is not a function of the frame counter; forward-rolling it would amount to
predicting the agent's own future.

> **Do not combine this key with observation-synthesising mirror augmentation
> yet.** The level mirror does not reflect drone trajectories (measured
> 2026-08-03): spawn positions mirror exactly, but paths diverge from frame 1.
> A synthesised mirrored observation would therefore place drones where the
> mirrored level does not have them. This is a pre-existing defect in
> `map_augmentation/transforms.py` — see the caveat on the `drone_hazard_map`
> entry in `obs_mirror.py` for the diagnosis. Mirroring at the LEVEL level is
> unaffected in kind (it still produces a valid level, just not a true
> reflection).

### `game_state`: `(261,)` float32, `[-1, 1]`

Flat ninja-physics + goal-geometry + execution vector. `GAME_STATE_CHANNELS == 261`
(the v22 training config adds one blocked-goal scalar via `--expose-blocked-goal`,
giving 262). `nclone.gym_environment.constants` holds the named block indices and is
the authoritative source. The table below is accurate for indices 0-68; each arc
block is 45-dim (3 steering arcs). Full per-dim arc fields are defined in
constants.py and the arc feature builder.

| idx | feature |
|-----|---------|
| 0-1 | velocity (x, y) / MAX_HOR_SPEED |
| 2-5 | movement-state one-hot-ish (ground / air / wall / special) |
| 6 | airborne |
| 7-9 | jump / floor / wall buffers |
| 10 | jump_duration_remaining |
| 11-13 | floor / wall / ceiling contact |
| 14-16 | floor normal (x, y), wall normal |
| 17 | walled |
| 18-20 | applied gravity / drag / friction (normalized) |
| 21-22 | ninja position (x, y) / level extent |
| 23 | time_remaining (curriculum-aware truncation) |
| 24-26 | physics-derived: horizontal_range, max_jump_height, time_to_ground |
| 27 | `switch_activated` phase bit (critical for the Markov two-phase task) |
| 28-29 | ninja-relative vector to the exit switch, `(dx, dy)` / level extent, clipped |
| 30-31 | ninja-relative vector to the exit door, `(dx, dy)` / level extent, clipped |
| 32-39 | surface ray-casts: normalized distance `[0,1]` to the nearest solid surface in 8 compass directions `[E, NE, N, NW, W, SW, S, SE]` (0 = adjacent, 1 = ≥500px) |
| 40-43 | free-space nav hint `[dx_hop, dy_hop, cost_to_go_norm, phase]` |
| 44 | nearest deadly-mine distance, normalized `[0,1]` (0 = touching, 1 = none within 168px) |
| 45 | closing speed toward the current objective, `dot(velocity, goal_dir)/MAX_HOR_SPEED` clipped `[-1,1]` |
| 46-48 | jump-execution: jump_held, jump_useless, launch-pad buffer window |
| 49-51 | look-ahead landing surface: normal (nx, ny) + normalized distance |
| 52 | signed vertical terminal-velocity fraction |
| 53-56 | execution timing: time-to-contact (straight-ray τ/60), frames airborne, per-frame accel (ax, ay) |
| 57-68 | forward-biased terrain scandots: (floor-gap, ceiling-gap) per column for 6 columns |
| 69-113 | committed ghost-arc bundle (Audit II P0-1): for u ∈ {−1, 0, +1}, one arc each (15 dims per arc), core fields `[dx/500, dy/500, nx, ny, mine_flag, none_flag, clear_width/9, impact_ratio/2, tau/60]` plus extended fields (see constants.py), from the native ballistic projection of the real airborne recurrence to first contact. The ±1 arcs are the bang-bang reachable-landing interval endpoints; `clear_width` is the contiguous mine-free standable extent at the contact; grounded frames use the fast-path fill (zeros + ny = −1). |
| 114-158 | jump-now ghost-arc bundle (v13): same 3-arc layout for the counterfactual full-hold takeoff while GROUNDED. Flat-floor jump impulse exactly as `Ninja.floor_jump` applies it (`yspeed = min(yspeed,0) + JUMP_FLOOR_Y`, position displaced by the impulse), then `MAX_JUMP_DURATION` frames of reduced gravity. Answers "if I jump right now, where do each of the three steering arcs land, and is it survivable?", takeoff-decision information the airborne-only committed bundle cannot provide. Zeros while airborne. |
| 159-164 | arc route-progress deltas (v13): per arc, `tanh((Φ(ninja) − Φ(landing)) / 150)` from the committed free-space/jump cost-to-completion source. Positive means that arc's landing makes route progress; negative means it flies away from the route. Slots 0-2 = committed arcs (airborne), slots 3-5 = jump-now arcs (grounded); inactive/mine/no-contact/inf-cost slots emit 0. Links the geometric arc bundle to the reward's own cost-to-go field, so "which arc is the route" is directly observable. |
| 165-209 | tap-jump arc bundle (v14 B1): same 3-arc layout as the jump-now row, but the takeoff is released immediately (`jf = 0` → full gravity after the impulse). The v13 jump-now arcs are full-hold only, so under a low ceiling (e.g. LOOP_CORRIDOR's 48px mined-walkway headroom) all three slam the ceiling and the move that actually clears a mine hop (a tap jump) was invisible to the policy. Zeros while airborne. |
| 210-254 | wall-jump-now arc bundle (v14): same 3-arc layout, the counterfactual regular wall-jump kick this frame while AIRBORNE + WALLED (`Ninja.wall_jump`: xspeed zeroed if into the wall, downward yspeed zeroed, `+1·wall_normal / −1.4` impulse, position displaced, full hold). On wall-slide frames the committed bundle shows only the slide/fall continuation; this row makes the kick affordance visible, the third contact regime's takeoff, targeting LOOP_CORRIDOR shaft chains / VERTICAL_SHAFT. Zeros when grounded or un-walled. |
| 255-260 | passive-arc bundle (6): the no-input continuation arc. See constants.py (`GAME_STATE_PASSIVE_ARC_IDX`) for the exact fields. |

Indices 28-31 are exact on the 6px entity grid and give the policy precise goal
geometry that the 24px overlay channels cannot. Both goal vectors are always
present (continuous across the switch→exit transition); the `switch_activated` bit
selects which is the live objective.

Indices 32-39 (surface ray-casts) march the tile grid to the nearest solid
surface per compass direction: wall sensing and blind-jump / gap verification
beyond the local tile view, in microseconds.

Indices 40-43 (free-space nav hint) are a physics-free steepest-descent over a
mine-aware free-space cost-to-go toward the current objective. `dx_hop, dy_hop` is
the next-step direction that rounds obstacles (vs the straight-line goal
vectors), `cost_to_go_norm` is the normalized remaining cost, `phase` mirrors
`switch_activated`. It is observation only: never reuse it as a reward
potential, which would double-count against the PBRS term. Zeros when the spawn
has no finite free-space route (severed / airborne levels).

## Validation

```bash
.venv/bin/python -m pytest tests/test_entity_overlays.py tests/test_hazard_map.py \
    tests/test_drone_hazard_map.py \
    tests/test_l1_game_state_switch_activated.py tests/test_layer1_mines_fixtures.py
```
