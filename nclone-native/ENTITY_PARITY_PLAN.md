# Rust entity parity plan

> **Status (2026-07-29): ALL PHASES DONE (0-6).** `nclone-native` now implements
> every entity the Python reference sim has. Official-map coverage 492 -> 1540
> (13.8% -> 43.2%).
>
> **Headline result, measured end-to-end:** re-simulating 8000 ATTRACT demos and
> keeping only those that still WIN goes from **3171 (39.6%) to 4271 (53.4%)** --
> +1100 demos, +35% relative. Extrapolated over the 250,468-demo corpus that is
> roughly **99k -> 134k importable expert demos**. The before-number was measured
> against a HEAD baseline build, same 8000 files, same converter.
>
> Beyond this point the ceiling is entities absent from BOTH sims -- turrets,
> floor guard, evil ninja, chaser/laser/chaingun drones. Those need reverse
> engineering: see `docs/GHIDRA_ENTITY_RE_GUIDE.md`.
>
> Per-phase notes are inline below.


Goal: bring `nclone-native` up to the entity coverage of the Python reference sim
(`nclone/entity_classes/`), so the training pipeline can run on the full official
map corpus rather than the mine/gold/exit subset.

Measured against the 3568 official maps in `nclone/nclone/maps/official`
(entity-type subset + exactly one exit door and one exit switch, i.e.
`levelgen.labeling.subset_ok` generalised to a larger allowed set):

| allowed entity set | compatible maps |
| --- | --- |
| current training manifest `{1,2,3,4,21}` | **492** (13.8%) |
| what the Rust sim already implements `{1,2,3,4,6,7,11,21}` | **541** (15.2%) |
| full Python parity | **1540** (43.2%) |

Two separate facts fall out of that table:

1. **49 maps are already free.** The manifest
   (`scripts/enumerate_compatible_official.py`, `IN_SCOPE = {1,2,3,4,21}`) is
   stale — the Rust sim gained locked doors (6/7) and one-ways (11) after it was
   written. Widening `IN_SCOPE` needs no simulator work.
2. Full parity roughly **triples** the addressable corpus, 492 → 1540.

Beyond 1540 the ceiling is Python's own coverage: rocket turret (18, blocks 908
maps), floor guard (16, 518), laser turret (23, 503), gauss turret (19, 425),
evil ninja (22, 280), drone chaser (15, 75), laser drone (13, 31), chaingun
drone (12, 23) have no implementation on *either* side and are out of scope here.

---

## 1. What is missing

Implemented in Rust (`src/entities/`): toggle mine (1/21), gold (2), exit door
(3), exit switch (4), locked door (6/7), one-way platform (11).

Missing, with the marginal map unlock each provides when added on top of the
current Rust set in greedy order:

| # | type | Python source | maps blocked | greedy cumulative |
| --- | --- | --- | --- | --- |
| 1 | 14 drone zap | `entity_drone_zap.py` + `entity_drone_base.py` | 835 | 905 |
| 2 | 26 mini drone | `entity_mini_drone.py` | 601 | 1051 |
| 3 | 17 bounce block | `entity_bounce_block.py` | 509 | 1150 |
| 4 | 10 launch pad | `entity_launch_pad.py` | 425 | 1242 |
| 5 | 25 death ball | `entity_death_ball.py` | 204 | 1323 |
| 6 | 20 thwump | `entity_thwump.py` | 163 | 1396 |
| 7 | 24 boost pad | `entity_boost_pad.py` | 155 | 1437 |
| 8 | 5 regular door | `entity_door_regular.py` | 76 | 1460 |
| 9 | 28 shove thwump | `entity_shove_thwump.py` | 41 | 1476 |
| 10 | 8/9 trap door | `entity_door_trap.py` | 356 | 1540 |

Trap door shows a `+0` marginal until its switch (type 9) is counted with it —
they must land together, and as a pair they are the single largest late win.

---

## 2. Infrastructure gaps — the actual work

The ten entities are mostly small ports. What blocks them is five pieces of
engine machinery the Rust sim has never needed, because every entity it
implements today is either static or state-only.

### 2.1 No `move()` phase in the tick loop

`Simulator::tick` (`src/simulator.rs:385`) runs a think pass (hardcoded to
toggle mines, inlined to dodge the borrow checker) and then ninja physics.
Python's `Simulator.tick` (`nclone/nsim.py:388-404`) runs **two** passes before
ninja physics:

```python
for entity in active_movable_entities: entity.move()
for entity in active_thinkable_entities: entity.think()
```

Both are needed by drones, bounce blocks, thwumps, boost pads (movable) and by
death balls, thwumps, shove thwumps, regular doors (thinkable).

**Iteration order is part of the contract.** Python iterates
`entity_dic` = `{1: [...], 2: [...], ..., 28: [...]}` — ascending type, then
map-load order within a type. Rust's `entities: Vec<Entity>` is pure map-load
order with types interleaved. Today that is unobservable (mines don't interact),
but death-ball chain bounces and multi-bounce-block force accumulation are
order-sensitive. Build a `type_ordered: Vec<usize>` index at load and drive both
passes from it.

### 2.2 `physical_collision` is `&self`

`Entity::physical_collision(&self, ninja: &Ninja)` (`src/entities/mod.rs:97`)
is read-only because one-way platforms are static. Bounce blocks push back
(`self.xpos -= …`, `self.xspeed -= …`), and shove thwumps mutate state from
`logical_collision`. Both need `&mut self`, which collides with
`Ninja::collide_vs_objects(&mut self, entities: &[Entity])`
(`src/ninja.rs:223`) holding a shared borrow of the entity slice across the
4-substep loop.

Fix: move the substep loop's entity pass into `Simulator`, index one entity at a
time out of `self.entities`, and pass `(&mut entity, &mut self.ninja)` — or
split the ninja out with `std::mem::take`. This is the largest single
refactor in the plan and should land on its own commit, verified byte-identical
on the existing one-way maps before any new entity is added.

The good news: `collide_vs_objects` already carries the type-17/20/28 branches
verbatim from Python (`src/ninja.rs:240-253`), and `Ninja` already has
`is_crushable`, `x_crush`, `crush_len`, `launch_pad_buffer`,
`xlp_boost_normalized` / `ylp_boost_normalized`. The consuming side is done.

### 2.3 No half-cell grid-edge dictionaries

`is_empty_row` / `is_empty_column` (`nclone/physics.py:158-187`) are how every
grid-walking entity — drones, thwumps, shove thwumps — tests whether it may
enter the next half-cell. They read four dictionaries on the sim:

- `hor_grid_edge_dic`, `ver_grid_edge_dic` — static, built from tiles
- `door_hor_grid_edge_dic`, `door_ver_grid_edge_dic` — dynamic, incremented and
  decremented by `EntityDoorBase.change_state` (`entity_door_base.py:104-112`)

Rust builds **none** of them. `TILE_GRID_EDGE_MAP` is already transcribed in
`src/map/tiles.rs:11` but has zero call sites — the loader only accumulates the
*segment* dictionaries (`create_segments_two_phase`, `src/map/loader.rs:84`).

Work:
- Add `hor_grid_edge: [[u8; 51]; 88]` and `ver_grid_edge: [[u8; 50]; 89]` to
  `MapData`/`Simulator`, filled in the tile loop using the Python *toggle* rule
  (`tile_segment_factory.py:276-297`): `dic[key] = new_val if dic[key] == 0 else 0`
  — note this is a toggle, **not** the `+=` accumulation the segment dicts use.
  Borders are 1 (`nsim.py:333-341`).
- Add the two `door_*` counters as `i8` grids.
- Port `is_empty_row` / `is_empty_column` into `src/collision/spatial.rs`.
- Retrofit the existing locked door: it currently flips only its segment
  (`simulator.rs:747`) and never touches a door edge grid, which is invisible
  today but would make drones walk through closed locked doors the moment 2.3
  lands. Route all four door types through one shared `set_door_closed()`.

### 2.4 Entities never change grid cell

`Entity.grid_move()` (`nclone/entities.py:257`) removes an entity from its old
`grid_entity` cell and inserts it into the new one whenever its centre crosses a
cell boundary. Rust has `rebuild_entity_grid()` but only calls it at load, reset,
and state restore. Every movable entity calls `grid_move()` each frame, so
`Simulator` needs an incremental `entity_grid_move(idx)`.

Also revisit the two load-time caches that assume a static entity field:
`has_physical_entities` (`simulator.rs:81`) and `deadly_mine_zone`
(`simulator.rs:117`). The first just needs the new physical types folded in; the
second stays valid (its only caller never mutates mines), but its doc comment
should say so explicitly once movable entities exist.

### 2.5 Missing physics helpers

Absent from Rust entirely (verified by grep over `nclone-native/src`):

| helper | Python | needed by |
| --- | --- | --- |
| `penetration_square_vs_point` | `physics.py:131` | bounce block, thwump, shove thwump |
| `overlap_circle_vs_segment` | `physics.py:409` | thwump lethal face |
| `map_vector_to_orientation` | `physics.py:190` | shove thwump logging only |
| `is_empty_row` / `is_empty_column` | `physics.py:158` | drones, thwumps |

Already present and reusable: `overlap_circle_vs_circle`, `sweep_circle_vs_tiles`,
`get_single_closest_point` (`src/collision/sweep.rs`, needed by the death ball),
and `map_orientation_to_vector` (currently living in
`src/entities/one_way_platform.rs:18` — promote it to a shared module).

### 2.6 Snapshot / serialization surface

Three places enumerate every entity variant and must grow with each addition:

- `EntitySnapshot` (`src/types.rs:269`) — currently
  `{type, x, y, active, state, radius, orientation, door_*}`. Moving entities
  need velocity for the observation layer (death ball, bounce block); drones need
  `dir`/`mode`; thwumps and shove thwumps use `state` with different semantics.
  Add fields rather than overloading, and keep the Python-side reader
  (`nclone/gym_environment/entity_extractor.py`) in step.
- `save_full_state` / `restore_full_state` (`simulator.rs:197`, `:262`) — the
  Go-Explore checkpoint blob. Each new variant needs a tag and a `field_len`
  entry; the validation phase already rejects mismatched layouts, so a missed
  entry fails closed rather than corrupting. Note this makes existing
  checkpoints unreadable — bump a blob version byte.
- `set_entity_position` (`src/lib.rs:101`) — exhaustive match used by the
  curriculum manager.

---

## 3. Proposed sequencing

Each phase ends green against the parity harness (§4) before the next starts.

**Phase 0 — free coverage, no sim changes. ✅ DONE.**
`scripts/enumerate_compatible_official.py` now takes `--scope`
(`minimal` / `native` / `python`) instead of a hardcoded `IN_SCOPE`, defaulting
to `native`. Confirmed 492 / 541 / 1540; `minimal` reproduces the old manifest
exactly. Written to `datasets/official_compatible_manifest_native.json` rather
than overwriting `official_compatible_manifest.json` — the latter is the live
det-eval target, and silently moving it would invalidate comparisons against the
v28 baseline. Switching the training/eval target to the new manifest is a
separate, deliberate call.

**Phase 1 — engine machinery. ✅ DONE.** No new entity types; verified
byte-identical against the (unchanged) Python sim rather than merely
self-consistent.

- §2.5 helpers → `src/geometry.rs` (`map_orientation_to_vector`, which also
  de-duplicates the two copies that had drifted into `one_way_platform.rs` and
  `locked_door.rs`, plus `map_vector_to_orientation` with Python's half-to-even
  rounding) and `src/collision/intersection.rs`
  (`penetration_square_vs_point`, `overlap_circle_vs_segment`). Checked against
  24 Python-generated reference cases, exact equality.
- §2.3 grid edges → `src/grid_edges.rs`, filled by the loader's tile walk, with
  the shared door geometry in `src/entities/door_base.rs`. All four grids match
  Python cell-for-cell on 90 official maps including every locked-door map.
- §2.1 move/think → `Simulator::run_entity_passes`, driven by a type-ordered
  `entity_order` index. The inlined toggle-mine think is gone; mines go through
  `Entity::think` again. §2.4 `grid_move` lives on `EntityCtx`.
- §2.2 → `Entity::physical_collision` now takes `&mut self`, and the loop moved
  from `Ninja::collide_vs_objects` to `Simulator::collide_vs_objects`.

Two pre-existing bugs surfaced and were fixed on the way, both about door state
installed wholesale rather than flipped by a switch:
`restore_full_state` resynced neither the door's collision segment nor (once
they existed) its grid edges, so a Go-Explore checkpoint captured after a door
opened restored with the door solid again. `fast_reset` and `restore_full_state`
now share `Simulator::resync_doors`.

Verification added: `tests/test_grid_edge_parity.py` (73 cases),
`tests/test_sim_differential_parity.py` (18 cases, ~4.8k simulated frames,
exact f64 equality on ninja + entity state every frame), and 12 Rust unit tests.
The differential harness was itself checked by injecting a divergence and
confirming it fires. Full nclone suite: 760 passed, 4 pre-existing failures
(reproduced identically on a HEAD baseline build, all in map generation /
route generators, none touching the sim).

**Phase 2 — static/simple entities. ✅ DONE.** Launch pad (10) in
`launch_pad.rs`, boost pad (24) in `boost_pad.rs`. The launch-pad boost lives in
`post_collision_with_entities` (it rewrites ninja state wholesale rather than
contributing a wall normal); the boost pad is the first user of the move pass.
Differential: launch pad 38/39 maps exact, boost pad 12/12.

**Phase 3 — drones. ✅ DONE.** One `drone.rs` covering zap (14) and mini (26) —
they differ only in radius, grid width and speed, so Python's inheritance split
is not worth reproducing. First real user of §2.3. Two originals faithfully
preserved: the out-of-range `mode` wrap (procgen emits bad modes, and Python
would `KeyError`), and `turn`'s `self.dir or dir`, which records the NEW
direction as `dir_old` when turning away from direction 0.
Differential: zap 28/30, mini 29/30 (all diffs the known ulp issue, never a
drone position or facing).

**Phase 4 — square physics entities. ✅ DONE.** Bounce block (17), thwump (20),
shove thwump (28). Only the PHYSICS of `entity_bounce_block.py` was ported — its
`_update_data_tracking` / `nearby_bounce_blocks` / `clearance_directions`
machinery is RL telemetry, not simulation, and would have cost more than the
physics it surrounds. Note the shove thwump is driven from `think`, not the move
pass, and its `state == 1` branch deliberately falls through so it launches on
the same frame it leaves the activated state.
Differential: bounce block 24/25, thwump 24/25, shwump 14/14.

**Phase 5 — remaining doors. ✅ DONE.** Regular (5) and trap (8 + phantom switch
9) share `switch_door.rs`. The locked-door special case in the simulator is gone:
all three door types now go through `sync_door_barriers`, which reconciles the
collision segment AND the half-cell grid edges from each door's `closed` flag,
applying only deltas. It runs twice per frame — after `think` (a regular door
that auto-closed must be solid before the ninja's collision substeps) and after
the post-collision entity loop.
Differential: regular 22/22, trap 24/25.

**Phase 6 — death ball (25). ✅ DONE.** The only entity needing its own sweep +
16-iteration depenetration loop, and the only one that mutates a SIBLING
(ball-vs-ball repulsion). `EntityCtx` deliberately cannot express that, so
`Simulator::death_ball_think` drives it: ball *i* completes its motion and pushes
balls *i+1..* before ball *i+1* is stepped, matching Python. `db_count` comes
from the map header (`map_data[1200]`), not the loaded entity count.
Differential: 24/25 maps exact, all 108 death-ball maps now in scope.

---

## 3a. Parity bugs found by the full-corpus differential sweep

Running `scripts/diffcheck_entity_parity.py` over every entity type (not just the
ones in the old `SUPPORTED` list) turned up four real divergences. All four are
fixed; every type now passes 20/20 frame-exact except the known 1-ulp drift in §3b.

1. **`segment_dic` was a `HashMap` — the native sim was not reproducible.**
   `create_segments_two_phase` emitted one segment per non-zero accumulator entry
   by iterating a `std::collections::HashMap`, whose order is randomised **per
   process**. That order becomes the order segments sit in inside
   `segment_grid[cell]`, which is what breaks ties between two equidistant
   segments — so the same map, same tape, same binary gave different results in
   different processes. It surfaced as a dual-laser surface walk drifting by
   0.1px, but it was never laser-specific.
   Fixed by replacing both accumulators with dense arrays iterated x-outer /
   y-inner, which is exactly Python's order: `nsim.py:426-443` pre-fills **every**
   key (zeros included) in that order, and the tile pass only updates existing
   keys, so its dict order is fully determined.

2. **`get_single_closest_point` had an AABB rejection Python does not have.**
   `gather_segments_from_region` returns everything registered in the overlapped
   *cells*, and Python then considers all of them — a segment can sit well outside
   the query box and still be the closest point returned. The native fast-path
   rejected those, so a door segment 36px from a dual laser was invisible to it and
   the laser picked spinner mode where Python picked surface. Also aligned the
   initial `shortest_distance` bound with Python's `9999999`.

3. **Trap-door segments were deactivated too late.** The loader pushed the segment
   active and left `resync_doors` to open it after the whole map had loaded.
   Python's `EntityDoorTrap.__init__` calls `change_state(closed=False)` during
   construction, so every entity built later in the stream sees the segment already
   gone. The dual laser picks its mode by querying nearby geometry inside its own
   ctor, so any trap door preceding it in the entity stream flipped its mode.

4. **The drone `move` cached `speed` across `choose_next_direction_and_goal`.**
   A chaser that reaches its goal ends the chase *inside* that call, dropping from
   16/7 to 8/7. Python reads `self.speed` after the call, so the step it leaves the
   goal on is already a patrol step. Caching diverged by exactly 8/7 px.

## 3b. Known defect, NOT introduced by this work

Native and Python drift by exactly **1 ulp in `ninja.xspeed`** on roughly 6% of
maps (9 of 150 gold-only official maps over 1500 frames each). It reproduces on
maps with no pads, drones or doors at all, so it predates every phase here.

Ruled out so far: the physics constants (bit-identical both sides), `integrate`,
the ground-friction formula, and the `abs(dx) <= 1e-7` snap with its hardcoded
`2^-47` nudges (Rust has those). It enters during the collision phase of a
landing frame. Next suspect is the segment gather ORDER feeding
`get_single_closest_point` — Python gathers once before the 32-iteration
depenetration loop — producing a different tie-break between equidistant
segments.

Reproduce with `scratchpad/diffcheck.py gold 150 1500`. It matters because it
compounds: a handful of long attract demos fail to win for this reason alone, so
compare deltas rather than absolutes when measuring entity work.

## 4. Verification

Bit-exact parity with the Python sim is the only acceptable bar; the training
pipeline already treats trajectory divergence as a correctness bug.

- **Differential harness.** Load the same map in both sims, drive an identical
  action sequence, assert ninja pos/vel/state and every entity's mutable fields
  match exactly (`==` on f64, not epsilon) every frame. Seed from official maps
  containing the entity under test, plus random action sequences. This should be
  built in Phase 1, before any new entity, using existing maps.
- **Phase 1 regression.** Re-run the existing test suite plus a
  `NCLONE_KERNEL_BASELINE` A/B; Phase 1 changes must not move a single bit.
- **Checkpoint compatibility.** Round-trip `save_full_state`/`restore_full_state`
  on a map with each new entity type, and confirm a stale blob is rejected rather
  than misparsed.
- **Corpus sweep.** After each phase, load every newly-compatible official map
  and assert no panic and no NaN — cheap, and catches malformed-level cases like
  the drone mode issue.
