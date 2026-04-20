# Layer 1 Local Validation Report (toggle-mines chunk)

**Issue:** [NPP-18](https://linear.app/npp-rl/issue/NPP-18/l110-layer-1-validation-suite-local-subset)
**Scope:** Layer 1, toggle-mines-only (no launch pads, bounce blocks, thwumps, drones, one-ways, doors, moving platforms).
**Status:** Local subset (T1 + T3 + T6-mines + T8) passed. T2/T4/T5/T7 deferred per rescoped plan.

Companion documents:
- Markov completeness sign-off: [`docs/LAYER1_MARKOV_AUDIT.md`](LAYER1_MARKOV_AUDIT.md) (NPP-42).
- Support matrix: [`docs/MECHANIC_SUPPORT_MATRIX.md`](MECHANIC_SUPPORT_MATRIX.md).
- Plan: `.cursor/plans/layer_1_toggle-mines_chunk_*.plan.md`.

## Environment

| Field | Value |
|---|---|
| Timestamp (UTC) | 2026-04-19T20:09:15Z |
| Host kernel | `Linux 6.6.87.2-microsoft-standard-WSL2 x86_64` |
| Python | 3.12.11 |
| PyTorch | 2.5.1+cu121 |
| CUDA available | False (CPU-only pytest, per NPP-18 AC) |
| Pytest | 9.0.2 |
| Hardware | dev-3080ti-smoke host (CPU used for these suites) |

All four local tests are CPU-only; `cloud-scale` validation (T5) is explicitly deferred.

## Command invocations

```bash
cd nclone
source .venv/bin/activate
pytest tests/test_astar_hwm.py tests/test_hazard_map.py tests/test_layer1_mines_fixtures.py
```

## Result summary

**Overall: 16 passed in 10.15 s (CPU).**

| Deliverable | File | Tests | Status |
|---|---|---|---|
| T1 — A* HWM monotonicity + mine-toggle recompute | `nclone/tests/test_astar_hwm.py` | 4 | PASS |
| T3 — `hazard_map` correctness on mine-dense fixture | `nclone/tests/test_hazard_map.py` | 6 | PASS |
| T6 (mines-only) — fixture maps: mine toggles, switch-phase swap, exit reach | `nclone/tests/test_layer1_mines_fixtures.py` (3 tests) | 3 | PASS |
| T8 — Markov spot-check on 3 toggle-mine fixtures | `nclone/tests/test_layer1_mines_fixtures.py::test_markov_layer1_contract_per_fixture[seed=18001/18002/18003]` | 3 | PASS |

Per-test breakdown (verbatim pytest output):

```
tests/test_astar_hwm.py::test_hwm_monotonic_non_negative_bounded_per_phase PASSED
tests/test_astar_hwm.py::test_hwm_no_reward_for_regression PASSED
tests/test_astar_hwm.py::test_astar_cache_invalidation_on_mine_state_change PASSED
tests/test_astar_hwm.py::test_cached_path_distance_calculator_rebuilds_on_mine_flip PASSED
tests/test_hazard_map.py::test_hazard_map_shape_and_dtype_empty PASSED
tests/test_hazard_map.py::test_hazard_map_constants_match_rust PASSED
tests/test_hazard_map.py::test_hazard_map_sign_convention_single_mine PASSED
tests/test_hazard_map.py::test_hazard_map_toggle_flips_sign PASSED
tests/test_hazard_map.py::test_hazard_map_rust_python_parity PASSED
tests/test_hazard_map.py::test_hazard_map_if_changed_returns_none_on_repeat PASSED
tests/test_layer1_mines_fixtures.py::test_mines_toggle_flips_hazard_sign PASSED
tests/test_layer1_mines_fixtures.py::test_switch_collection_flag_layout PASSED
tests/test_layer1_mines_fixtures.py::test_exit_and_switch_channels_populated_at_reset PASSED
tests/test_layer1_mines_fixtures.py::test_markov_layer1_contract_per_fixture[18001] PASSED
tests/test_layer1_mines_fixtures.py::test_markov_layer1_contract_per_fixture[18002] PASSED
tests/test_layer1_mines_fixtures.py::test_markov_layer1_contract_per_fixture[18003] PASSED
```

## T1 — A* HWM monotonicity + mine-toggle recompute

Two sub-invariants are validated:

1. **HWM monotonicity.** `EpisodeRewardState` emits non-negative shaping per
   step and the cumulative normalised reward per phase is bounded by 1. This
   guarantees the High-Water-Mark term is a valid potential-style signal.
   Regressions (moving further from target) emit zero.
2. **Mine-toggle recompute.** A toggle-mine state flip invalidates the
   shared A* distance-map cache ([NPP-44](https://linear.app/npp-rl/issue/NPP-44))
   and forces the next `get_or_compute` call to re-run the loader. The same
   invariant holds at the `CachedPathDistanceCalculator` level: a mine flip
   forces `build_level_cache` to rebuild; an unchanged state short-circuits.

## T3 — `hazard_map` correctness

The 1-channel `hazard_map` exposed by
`nclone.gym_environment.npp_environment.NppEnvironment` is validated against
a Python NumPy SDF reference on a 100-mine fixture:

- Shape `(HAZARD_MAP_CHANNELS, HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH) = (1, 92, 168)`, `float32`.
- Sign convention: negative inside the 14 px kill zone, `0` at boundary,
  clamped to `+1` at full SDF scale.
- Toggling a mine deadly -> safe flips the SDF from negative to non-negative
  at the mine center.
- Max `|Δ|` between the Rust engine and the Python reference is `<= 1e-5`.
- `compute_hazard_map_if_changed` returns `None` when the input list is
  unchanged (cache hit) and a fresh buffer otherwise.

## T6 (mines-only subset) — fixture behaviour

Three fixtures audit the behaviour the in-scope Markov rows require:

1. **Mine toggle fixture (`test_mines_toggle_flips_hazard_sign`).** Loads a
   single deadly mine, verifies the `hazard_map` SDF at the mine center is
   negative. Drops the mine from the deadly set; the SDF at that cell
   becomes non-negative.
2. **Switch-phase swap fixture (`test_switch_collection_flag_layout`).**
   `game_state[GAME_STATE_SWITCH_ACTIVATED_IDX]` is `0.0` at reset and is
   laid out as the final `game_state` slot (matches
   `docs/LAYER1_MARKOV_AUDIT.md` and
   `nclone/gym_environment/constants.py`).
3. **Exit-reach fixture (`test_exit_and_switch_channels_populated_at_reset`).**
   `entity_overlays[switch_pos]` and `entity_overlays[exit_pos]` both carry
   non-zero mass at reset so the policy has targets to navigate toward. The
   ninja splat channel is also non-empty.

The remaining T6 rows (doors, one-ways, slope-buffer, launch pads, bounce
blocks, thwumps, drones, moving platforms) are explicitly **waived** via
[NPP-42](https://linear.app/npp-rl/issue/NPP-42/new-d-observation-completeness-audit-toggle-mines-markov-gate).
See [`docs/LAYER1_MARKOV_AUDIT.md` § "Waived rows"](LAYER1_MARKOV_AUDIT.md#waived-rows-out-of-scope-for-this-chunk)
for per-row justifications.

## T8 — Markov spot-check on three toggle-mine fixtures

For each of three fixture seeds (`18001`, `18002`, `18003`) the observation
dict returned by `NppEnvironment.reset` is audited against the contract
declared in [`LAYER1_MARKOV_AUDIT.md`](LAYER1_MARKOV_AUDIT.md) §
"Layer 1 observation keys":

| Key | Expected shape | Expected dtype | Result |
|---|---|---|---|
| `tile_grid` | `(42, 23)` | `int8` | PASS |
| `entity_overlays` | `(5, 42, 23)` | `float32` | PASS |
| `hazard_map` | `(1, 92, 168)` | `float32` | PASS |
| `game_state` | `(GAME_STATE_CHANNELS,) = (28,)` | `float32` | PASS |

Additional invariants checked per fixture:

- `hazard_map` values are clipped to `[-1, 1]` (guarantees the Rust SDF
  respects the `SDF_SCALE = 24` budget on all samples).
- `entity_overlays[ninja_pos]` channel has non-zero total mass (ninja splat
  populated on every reset).

This satisfies the T8 acceptance criterion "Markov spot-check on 3
toggle-mine fixtures citing NPP-42 sign-off": every in-scope transition
variable listed in the audit maps to an observation key that is present,
correctly shaped and correctly dtyped on every tested fixture.

## Deferred / out-of-scope (follow-up)

Per the rescoped [NPP-18](https://linear.app/npp-rl/issue/NPP-18) body the
following tests remain deferred until their dependencies land:

| Deliverable | Deferred because | Unblocker |
|---|---|---|
| **T2** — BSRS / value ordering | BSRS reward not yet integrated | [NPP-14](https://linear.app/npp-rl/issue/NPP-14) + [NPP-45](https://linear.app/npp-rl/issue/NPP-45) |
| **T4** — Single-level overnight train | Requires BSRS shaping to be meaningful | follows T2 |
| **T5** — 256×50 multi-level cloud run | Needs cloud infra (A100/H100) | cloud chunk |
| **T7** — Per-bucket baseline comparison | Requires benchmark harness | [NPP-39/40/41](https://linear.app/npp-rl/issue/NPP-39) |

These will be picked up in a subsequent chunk; the Layer 1 toggle-mines
policy can train against the local subset without them.

## Acceptance checklist ([NPP-18](https://linear.app/npp-rl/issue/NPP-18))

- [x] T1, T3, T6-mines, T8 implemented and runnable via `pytest`.
- [x] `docs/layer1_local_report.md` summarises pass/fail per test with
  command invocations and timestamps.
- [x] All four local tests pass on `dev-3080ti-smoke` hardware (CPU-only
  for pytest parts).
- [x] T6 explicitly points at [NPP-42](https://linear.app/npp-rl/issue/NPP-42)
  waivers for the skipped rows (see "T6 (mines-only subset)" above).
- [x] T8 cites [NPP-42](https://linear.app/npp-rl/issue/NPP-42) sign-off
  comment via `docs/LAYER1_MARKOV_AUDIT.md`.

Parent gate [NPP-5](https://linear.app/npp-rl/issue/NPP-5) — toggle-mines
chunk — is satisfied for the Layer 1 local-validation deliverable.
