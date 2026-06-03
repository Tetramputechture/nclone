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
  `entity_overlays` 5-channel, `hazard_map` 1-channel, `game_state`
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
    GAME_STATE_SWITCH_ACTIVATED_IDX,
    HAZARD_MAP_CHANNELS,
    HAZARD_MAP_HEIGHT,
    HAZARD_MAP_WIDTH,
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
    "hazard_map": {
        "shape": (HAZARD_MAP_CHANNELS, HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH),
        "dtype": np.float32,
    },
    "game_state": {"shape": (GAME_STATE_CHANNELS,), "dtype": np.float32},
}


def _make_env() -> NppEnvironment:
    """Construct a default NppEnvironment for Layer 1 validation."""
    return NppEnvironment(config=EnvironmentConfig())


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
    mine_center = (240.0, 150.0)  # inside playable area
    col = int(mine_center[0] / float(MineThreatMap.hazard_cell_size()))
    row = int(mine_center[1] / float(MineThreatMap.hazard_cell_size()))

    deadly = engine.compute_hazard_map([mine_center])
    deadly = np.asarray(deadly, dtype=np.float32).reshape(
        HAZARD_MAP_CHANNELS, HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH
    )
    assert deadly[0, row, col] < 0.0, (
        "Deadly toggle-mine must produce a negative SDF at its center"
    )

    safe = engine.compute_hazard_map([])
    safe = np.asarray(safe, dtype=np.float32).reshape(
        HAZARD_MAP_CHANNELS, HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH
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
    obs, _ = shared_env.reset(seed=FIXTURE_SEEDS[0])
    assert "game_state" in obs
    gs = np.asarray(obs["game_state"], dtype=np.float32)
    assert gs.shape == (GAME_STATE_CHANNELS,)
    # switch_activated is the second-to-last channel; gold-remaining is last.
    assert GAME_STATE_SWITCH_ACTIVATED_IDX == GAME_STATE_CHANNELS - 2
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
    obs, _ = shared_env.reset(seed=FIXTURE_SEEDS[1])
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
        obs, info = env.reset(seed=seed)
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
