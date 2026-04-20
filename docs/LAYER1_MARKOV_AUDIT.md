# Layer 1 Markov Audit (toggle-mines chunk)

**Issue:** [NPP-42](https://linear.app/npp-rl/issue/NPP-42/new-d-observation-completeness-audit-toggle-mines-markov-gate)
**Scope:** Layer 1, toggle-mines-only (no launch pads, bounce blocks, thwumps, drones, death balls, one-way platforms, doors, moving platforms).
**Status:** Draft sign-off for in-chunk consumers (NPP-38, NPP-15, NPP-16, NPP-21).

Companion to [MECHANIC_SUPPORT_MATRIX.md](MECHANIC_SUPPORT_MATRIX.md) (row-level in-scope / waived column).

## Goal

Prove that the Layer 1 observation dict is **Markov-complete over every transition variable required by in-scope mechanics**, so a feed-forward policy (no frame stack, no recurrence) can in principle match the non-recurrent optimal.

If a transition variable is not captured by any observation key, it is either

1. Added in-chunk (by NPP-38, NPP-46, or NPP-12 / NPP-13 game_state changes), or
2. Waived because its source mechanic is out of scope.

## Layer 1 observation keys (final shapes)

| Key | Shape | dtype | Source |
|----|------|-------|--------|
| `tile_grid` | `(42, 23)` | `int8` | `nclone.gym_environment.observation_processor.tile_grid` |
| `entity_overlays` | `(5, 42, 23)` | `float32` | `nclone.gym_environment.entity_overlay_utils.compute_entity_overlays` (NPP-38) |
| `hazard_map` | `(1, 168, 92)` | `float32` | `nclone-native::mine_sdf` → `nclone.gym_environment.observation_processor.compute_hazard_map` (NPP-46) |
| `game_state` | `(GAME_STATE_CHANNELS,)` | `float32` | `NinjaStateObservation` + physics-derived + `switch_activated` (see `nclone/gym_environment/constants.py`) |

`entity_overlays` channels: `[ninja_pos, switch_pos, exit_pos, mine_deadly_density, mine_safe_density]`.

`hazard_map` channel 0: `deadly_sdf = clip((min_dist_to_deadly_mine_center - 14) / 24, -1, 1)` at 6px per cell.

`game_state` features (ninja_state 24 + physics-derived 3 + switch_activated 1 = `GAME_STATE_CHANNELS`, see `nclone/gym_environment/constants.py`): velocity (2), movement_state (4), airborne (1), buffers (3), jump_duration_remaining (1), contacts (3), normals (3), walled (1), physics range/jump/time-to-ground (3), position (2), time_remaining (1), switch_activated (1).

## In-scope rows (must be Markov-complete)

| Row | Transition variables | Covered by |
|---|---|---|
| Flat run / walk | `pos, vx, vy, airborne, walled, floor_buffer` | `tile_grid`, `game_state` (velocity, airborne, walled, buffers, position) |
| Basic jump / gap / fall | `pos, vx, vy, jump_duration_remaining, airborne, floor_buffer, jump_buffer` | `tile_grid`, `game_state` (velocity, airborne, buffers, jump_duration_remaining, position) |
| Wall jump | `pos, vx, vy, walled, wall_normal, wall_buffer, jump_buffer` | `tile_grid`, `game_state` (velocity, walled, contacts, normals, buffers, position) |
| Slopes / ramps | `pos, vx, vy, airborne, contact_normal` | `tile_grid`, `game_state` (velocity, airborne, contacts, normals, physics-derived: horizontal_range / time_to_ground) |
| Toggle mines | `mine_center_pos, mine_radius (static), mine_state ∈ {safe, toggled, deadly}` | `entity_overlays[mine_deadly_density]`, `entity_overlays[mine_safe_density]` (tile-space density), `hazard_map[deadly_sdf]` (sub-tile SDF, 6px) |
| Exit switch | `switch_pos, switch_activated` | `entity_overlays[switch_pos]` (zeroed after collection), `game_state.switch_activated` |
| Exit door | `exit_pos, active_when_switch_collected` | `entity_overlays[exit_pos]`, `game_state.switch_activated` (phase indicator) |

### Notes on toggle-mine Markov-ness

Toggle mines have three states in the simulator: `safe` (0), `toggled` (1), `deadly` (2). The observation must disambiguate `safe` from `deadly` (both visible as "not immediately lethal by touch", but the transition graph differs because entering a `toggled` mine is fatal and a `safe` one arms when the ninja steps out). The pair `(mine_deadly_density, mine_safe_density)` is sufficient because:

- `mine_safe_density` covers both `safe` and `toggled` (non-deadly today, but dangerous on exit or on second contact).
- `mine_deadly_density` is 0 for any tile containing only non-deadly mines.
- Sub-tile disambiguation (is the ninja currently inside a `toggled` mine vs. a `safe` one) is carried by `hazard_map[deadly_sdf]` becoming negative when the ninja is inside the kill zone of a mine that has since been armed; this is sufficient at 6px resolution for the ninja's 10px radius.

**Open consideration (documented, not blocking):** distinguishing `safe` vs `toggled` at sub-tile resolution requires either (a) a third raster channel (untoggled_safe_sdf), or (b) a dedicated bit per tile in `entity_overlays`. This is not required for the toggle-mines chunk because once a mine is in `toggled` state, it will either revert to `safe` when the ninja leaves it (observed via `hazard_map` change the next step) or kill the ninja. The Markov decision the policy must make is "am I about to exit a toggled mine this tick", which is fully captured by the ninja's position and the `hazard_map` footprint. No additional observation channel is required for the current scope.

## Waived rows (out of scope for this chunk)

Each of the following is waived from the Layer 1 Markov gate because the underlying mechanic is explicitly out of scope for the toggle-mines chunk:

| Row | Waiver reason |
|---|---|
| Locked / regular doors | Out of scope for Layer 1 toggle-mines chunk. Bring back with a dedicated Markov-audit row before adding door state to `game_state` or `entity_overlays`. |
| Trap doors | Out of scope. |
| One-way platforms | Out of scope (would require a directed-surface channel; covered by the cancelled NPP-47). |
| Launch pads | Out of scope. `launch_pad_pos` and `launch_pad_direction` channels removed from `entity_overlays` in NPP-38. |
| Bounce blocks | Out of scope. Dynamic-surface / surface-velocity channels removed with NPP-47 cancellation. |
| Thwumps | Out of scope. Thwump safe-face and velocity channels are not modeled. |
| Drones | Out of scope. No hazard-velocity channel. |
| Death balls | Out of scope. |
| Moving platforms | Out of scope. |

If any waived mechanic is brought back, the reintroduction ticket must:

1. Add a new row to this audit with concrete transition variables and candidate observation key(s).
2. Update [MECHANIC_SUPPORT_MATRIX.md](MECHANIC_SUPPORT_MATRIX.md) `layer1_scope` column.
3. Land observation-space extensions (extra channels) **before** policy changes that depend on them.

## Fixture walkthrough

Three fixture maps confirm populated keys and correctness (see `nclone/tests/test_layer1_mines_fixtures.py` under NPP-18 T6/T8):

1. **Flat corridor** — `tile_grid` covers floor / walls; `game_state` reports `walled=0, airborne=0, vy≈0` during run; `entity_overlays[switch_pos, exit_pos]` populated; `hazard_map` is all `+1.0` (no mines).
2. **Mines-dense** — 100 toggle mines; `hazard_map[deadly_sdf]` shows negative regions around currently-deadly mines and `0` boundary; flipping a mine from safe→deadly makes the corresponding SDF region go negative within one step; `entity_overlays[mine_deadly_density]` and `[mine_safe_density]` track tile-level counts.
3. **Slopes** — `tile_grid` encodes slope tile-ids (class `SLOPE_45_*` etc.); `game_state.contacts` / `game_state.normals` / `game_state.physics` (horizontal_range, time_to_ground) cover slope transitions.

## Dependent issues

The following issues can cite this audit as Markov sign-off for Layer 1 (toggle-mines scope):

- [NPP-38](https://linear.app/npp-rl/issue/NPP-38) — shrink entity overlays to 5 channels.
- [NPP-15](https://linear.app/npp-rl/issue/NPP-15) — 4-key observation dict.
- [NPP-16](https://linear.app/npp-rl/issue/NPP-16) — `StructuredTileExtractorV1`.
- [NPP-21](https://linear.app/npp-rl/issue/NPP-21) — remove frame-stack requirement.

## Sign-off

- [x] In-scope rows list complete.
- [x] Each in-scope row cites at least one observation key per transition variable.
- [x] Waived rows enumerated and justified.
- [x] Plan cross-reference: `.cursor/plans/layer_1_toggle-mines_chunk_*.plan.md`.
