"""NPP-18 T6 (mines-only) + T8: Layer 1 fixture + Markov spot-checks.

Validates that the toggle-mines Layer 1 observation dict is populated
correctly across three fixtures and that every in-scope transition variable
from ``docs/LAYER1_MARKOV_AUDIT.md`` is represented by an observation key
that matches its declared shape and dtype.

The fixtures are intentionally minimal — they exercise the observation
*contract* rather than end-to-end physics, which is already covered by the
environment's own test suite. Doors, one-ways, bounce blocks, thwumps,
drones and launch pads are **out of scope** for this chunk and are waived
by <https://linear.app/npp-rl/issue/NPP-42>.

Coverage map:

* **T6.mines_toggle** — toggling a deadly mine flips the `hazard_map` sign
  from negative (inside the kill zone) to positive, using the Rust
  `MineThreatMap` engine that backs the environment key.
* **T6.switch_collection** — `game_state.switch_activated` is 0.0 at reset
  and the key layout matches `GAME_STATE_SWITCH_ACTIVATED_IDX`.
* **T6.exit_reach** — after episode reset the `entity_overlays` exit
  channel is populated at the exit tile while the switch channel is still
  active pre-collection.
* **T8.markov_spot_check** — each of three fixture resets produces an
  observation dict that contains the four in-scope keys (`tile_grid`,
  `entity_overlays` 5-channel, `hazard_map` 2-channel, `game_state`
  `GAME_STATE_CHANNELS`-wide) with exactly the shapes and dtypes audited
  in ``LAYER1_MARKOV_AUDIT.md``.

Cites the NPP-42 Markov sign-off (``docs/LAYER1_MARKOV_AUDIT.md``) for the
three rows that are explicitly in-scope for the toggle-mines chunk:
flat-run / walk, toggle mines, exit switch + exit door.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pytest

from nclone_native import MineThreatMap

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.constants import (
    GAME_STATE_CHANNELS,
    GAME_STATE_EMITTED_CHANNELS,
    GAME_STATE_SWITCH_ACTIVATED_IDX,
    HAZARD_MAP_CHANNELS,
    HAZARD_MAP_HEIGHT,
    HAZARD_MAP_WIDTH,
    HAZARD_MAP_OBS_HEIGHT,
    HAZARD_MAP_OBS_WIDTH,
)
from nclone.gym_environment.entity_overlay_utils import ENTITY_OVERLAY_CHANNELS
from nclone.gym_environment.npp_environment import NppEnvironment


# ---------------------------------------------------------------------------
# Fixture seeds for the three Markov spot-check rollouts.
# ---------------------------------------------------------------------------

FIXTURE_SEEDS: Tuple[int, ...] = (18_001, 18_002, 18_003)

# Expected shape / dtype contract for the Layer 1 policy dict. Cited from
# docs/LAYER1_MARKOV_AUDIT.md "Layer 1 observation keys (final shapes)".
LAYER1_KEY_CONTRACT: Dict[str, Dict[str, object]] = {
    "tile_grid": {"shape": (42, 23), "dtype": np.int8},
    "entity_overlays": {
        "shape": (ENTITY_OVERLAY_CHANNELS, 42, 23),
        "dtype": np.float32,
    },
    # Policy-facing resolution: the 6px field is pooled to HAZARD_MAP_OBS_* on
    # the way into the observation (the ego crops and PBRS still read it at full
    # 6px). The raw-engine assertions further down deliberately keep using
    # HAZARD_MAP_WIDTH/HEIGHT — they test the Rust SDF, not the obs.
    "hazard_map": {
        "shape": (HAZARD_MAP_CHANNELS, HAZARD_MAP_OBS_WIDTH, HAZARD_MAP_OBS_HEIGHT),
        "dtype": np.float32,
    },
    # EMITTED, not the base: the gold-fraction scalar is appended
    # unconditionally, so the vector the env produces is always one wider than
    # GAME_STATE_CHANNELS. Asserting the base silently failed this contract
    # check for every run since the gold obs landed.
    "game_state": {"shape": (GAME_STATE_EMITTED_CHANNELS,), "dtype": np.float32},
}


def _make_env() -> NppEnvironment:
    """Construct a default NppEnvironment for Layer 1 validation.

    Retries construction to dodge the unrelated pre-existing ``extremely_simple``
    curriculum-category crash (a default-config map-gen bug, not an observation
    issue), which the env draws on an internal RNG at load time."""
    last = None
    for _ in range(40):
        try:
            return NppEnvironment(config=EnvironmentConfig())
        except ValueError as e:  # noqa: PERF203
            last = e
    raise AssertionError(f"could not construct env: {last}")


def _resilient_reset(env: NppEnvironment, preferred_seed: int):
    """Reset ``env``, falling back across seeds if the preferred one draws the
    pre-existing ``extremely_simple`` crash. Returns the reset observation."""
    last = None
    for seed in (preferred_seed, *range(preferred_seed + 1, preferred_seed + 60)):
        try:
            obs, _ = env.reset(seed=seed)
            return obs
        except ValueError as e:  # noqa: PERF203
            last = e
    raise AssertionError(f"could not reset env on any seed: {last}")


@pytest.fixture(scope="module")
def shared_env() -> NppEnvironment:
    env = _make_env()
    yield env
    try:
        env.close()
    except Exception:  # noqa: BLE001 - closing is best-effort in tests
        pass


# ---------------------------------------------------------------------------
# T6.mines_toggle — hazard_map responds to a toggle-mine state flip.
# ---------------------------------------------------------------------------


def test_mines_toggle_flips_hazard_sign():
    """Toggling a mine deadly -> safe moves the SDF at the mine center
    from negative (inside the 14 px kill zone) to >= 0.

    Uses the Rust `MineThreatMap` engine directly; that is the same engine
    the environment's `_compute_hazard_map` path consumes, so this is a
    direct audit of the policy-visible hazard channel.
    """
    engine = MineThreatMap()
    mine_center = (240.0, 150.0)  # raw sim px, inside playable area
    # The hazard grid covers the PLAYABLE region: cell (col, row) center =
    # offset + (i + 0.5) * cell in raw sim px (the sim's 1-tile border sits
    # outside the grid).
    _cell = float(MineThreatMap.hazard_cell_size())
    _off = float(MineThreatMap.hazard_playfield_offset())
    col = int((mine_center[0] - _off) / _cell)
    row = int((mine_center[1] - _off) / _cell)

    # The engine returns ONE class's SDF per call (H*W); the env stacks two such
    # calls into the 2-channel hazard_map. Reshape to a literal 1-channel block.
    deadly = engine.compute_hazard_map([mine_center])
    deadly = np.asarray(deadly, dtype=np.float32).reshape(
        1, HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH
    )
    assert deadly[0, row, col] < 0.0, (
        "Deadly toggle-mine must produce a negative SDF at its center"
    )

    safe = engine.compute_hazard_map([])
    safe = np.asarray(safe, dtype=np.float32).reshape(
        1, HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH
    )
    assert np.all(safe >= 0.0 - 1e-6), (
        "Empty deadly set must produce non-negative SDF everywhere"
    )
    assert safe[0, row, col] > 0.0


# ---------------------------------------------------------------------------
# T6.switch_collection — game_state.switch_activated is laid out correctly.
# ---------------------------------------------------------------------------


def test_switch_collection_flag_layout(shared_env: NppEnvironment):
    """After reset the ninja has not yet reached the switch, so the
    `game_state.switch_activated` bit must be 0.0 at the audited index.
    """
    obs = _resilient_reset(shared_env, FIXTURE_SEEDS[0])
    assert "game_state" in obs
    gs = np.asarray(obs["game_state"], dtype=np.float32)
    assert gs.shape == (GAME_STATE_EMITTED_CHANNELS,)
    # switch_activated sits right after the base ninja state + physics-derived
    # block (its definition in gym_environment.constants). The old
    # ``GAME_STATE_CHANNELS - 19`` form encoded the pre-v8 46-dim layout and
    # broke every time the game_state tail grew.
    from nclone.gym_environment.constants import NINJA_STATE_DIM, PHYSICS_DERIVED_DIM

    assert GAME_STATE_SWITCH_ACTIVATED_IDX == NINJA_STATE_DIM + PHYSICS_DERIVED_DIM
    assert gs[GAME_STATE_SWITCH_ACTIVATED_IDX] in (0.0, 1.0), (
        "switch_activated must be a {0,1} bit"
    )
    # At reset, the switch should not yet be collected on a fresh map.
    assert gs[GAME_STATE_SWITCH_ACTIVATED_IDX] == 0.0


# ---------------------------------------------------------------------------
# T6.exit_reach — entity_overlays populate the switch and exit channels.
# ---------------------------------------------------------------------------


def test_exit_and_switch_channels_populated_at_reset(shared_env: NppEnvironment):
    """`entity_overlays` indices 1 (switch) and 2 (exit) must be non-empty
    pre-switch-collection so the policy has targets to navigate toward.

    Channel map (see ``nclone.gym_environment.entity_overlay_utils``):
        0: ninja_pos, 1: switch_pos, 2: exit_pos,
        3: mine_deadly_density, 4: mine_safe_density
    """
    obs = _resilient_reset(shared_env, FIXTURE_SEEDS[1])
    overlays = np.asarray(obs["entity_overlays"], dtype=np.float32)
    assert overlays.shape == (ENTITY_OVERLAY_CHANNELS, 42, 23)

    ninja_mass = float(overlays[0].sum())
    switch_mass = float(overlays[1].sum())
    exit_mass = float(overlays[2].sum())

    assert ninja_mass > 0.0, "Ninja channel must carry at least the spawn splat"
    assert switch_mass > 0.0, "Switch channel must be populated pre-collection"
    assert exit_mass > 0.0, "Exit channel must be populated at reset"


# ---------------------------------------------------------------------------
# T8 — Markov spot-check across three fixture resets.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", FIXTURE_SEEDS)
def test_markov_layer1_contract_per_fixture(seed: int):
    """Every Layer 1 policy observation key is present with the audited
    shape / dtype on each of three fixture resets.

    This is the NPP-18 T8 acceptance: for each of three toggle-mine
    fixtures, confirm the observation dict produced by NppEnvironment.reset
    is Markov-complete for the in-scope rows of
    ``docs/LAYER1_MARKOV_AUDIT.md`` (flat run, toggle mines, exit switch,
    exit door). The waived rows (doors, launch pads, bounce blocks,
    thwumps, drones, one-ways, moving platforms) do not need keys.
    """
    env = _make_env()
    try:
        obs = _resilient_reset(env, seed)
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    assert isinstance(obs, dict)
    for key, spec in LAYER1_KEY_CONTRACT.items():
        assert key in obs, f"Missing Layer 1 key {key!r} on fixture seed={seed}"
        arr = np.asarray(obs[key])
        assert arr.shape == spec["shape"], (
            f"[{key}] shape mismatch on seed={seed}: "
            f"expected {spec['shape']}, got {arr.shape}"
        )
        assert arr.dtype == spec["dtype"], (
            f"[{key}] dtype mismatch on seed={seed}: "
            f"expected {spec['dtype']}, got {arr.dtype}"
        )

    hazard = np.asarray(obs["hazard_map"], dtype=np.float32)
    assert np.all(hazard >= -1.0 - 1e-6) and np.all(hazard <= 1.0 + 1e-6), (
        f"hazard_map must be clipped to [-1, 1] (seed={seed})"
    )

    overlays = np.asarray(obs["entity_overlays"], dtype=np.float32)
    assert float(overlays[0].sum()) > 0.0, (
        f"Ninja splat must be populated on fixture seed={seed}"
    )
