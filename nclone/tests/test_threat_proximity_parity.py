"""Rust `threat_proximity` vs its pure-Python twin, on real levels.

`threat_proximity_reference.py` says in its own docstring that this file pins it.
Until 2026-08-13 this file did not exist, and the claim was the only thing
standing behind a second implementation.

That matters more here than for a typical duplicate. The block's original defect
was precisely a backend split: it was computed in Python off `sim.entity_dic`,
which on the NATIVE backend whitelists `{1,2,3,4,6,7,11,21}` and contains no
threat type at all. Training runs native, so the whole vector was structurally
zero in production for an entire run while every unit test passed — the tests
exercised the pure `_threat_imminence` mapping, and the dev overlay runs on the
Python backend. Nothing compared the two.

So this test asserts two separate things, and BOTH are load-bearing:

1. The two implementations agree dim for dim on real levels.
2. At least some of those levels produce a NON-ZERO vector. Without this, a
   regression to structural zero passes trivially — zero agrees with zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone.gym_environment.constants import THREAT_PROXIMITY_DIM

TICKS = 60
# Beyond this the two sims' ninja positions have drifted enough that they are no
# longer being asked the same question -- see the known 1-ulp xspeed divergence
# on ~6% of maps. Such a level is skipped rather than compared, because a
# position mismatch would be reported as a threat_proximity mismatch.
MAX_POS_DRIFT_PX = 0.5
ATOL = 1e-4


def _official_maps(limit):
    from pathlib import Path

    root = Path("nclone/maps/official")
    if not root.is_dir():
        pytest.skip("official maps not present in this checkout")
    # Map files are extensionless, but so is `.gitkeep` -- Path(".gitkeep").suffix
    # is "". Feeding it to the loader panics the Rust side outright
    # (map/loader.rs: index out of bounds), so exclude dotfiles explicitly.
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix == "" and not p.name.startswith(".")
    )[:limit]


def _native_vec(map_path):
    from nclone.native_headless import NativePlayHeadless

    s = NativePlayHeadless()
    s.load_map(str(map_path))
    for _ in range(TICKS):
        s.tick(0, 0)
    v = np.frombuffer(s._native_sim.threat_proximity(), dtype=np.float32)
    return np.array(v, copy=True), s.ninja_position()


def _reference_vec(map_path):
    from nclone.gym_environment.threat_proximity_reference import (
        compute_threat_proximity_reference,
    )
    from nclone.nplay_headless import NPlayHeadless

    h = NPlayHeadless(render_mode="rgb_array")
    h.load_map(str(map_path))
    for _ in range(TICKS):
        h.tick(0, 0)
    nx, ny = h.ninja_position()
    return compute_threat_proximity_reference(h, nx, ny), (nx, ny)


@pytest.mark.slow
def test_native_matches_the_python_twin_on_official_levels():
    maps = _official_maps(40)
    compared = 0
    live = 0
    drifted = 0
    for m in maps:
        try:
            nat, npos = _native_vec(m)
            ref, rpos = _reference_vec(m)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{m.name}: {type(exc).__name__}: {exc}")
        if max(abs(npos[0] - rpos[0]), abs(npos[1] - rpos[1])) > MAX_POS_DRIFT_PX:
            drifted += 1
            continue
        assert nat.shape == (THREAT_PROXIMITY_DIM,)
        assert ref.shape == (THREAT_PROXIMITY_DIM,)
        np.testing.assert_allclose(
            nat, ref, atol=ATOL, err_msg=f"backends disagree on {m.name}"
        )
        compared += 1
        if np.abs(nat).max() > 0:
            live += 1

    assert compared >= 20, f"only {compared} levels comparable ({drifted} drifted)"
    # The structural-zero guard. If this fires, the block is dead again and the
    # agreement assertion above became vacuous.
    assert live > 0, (
        f"all {compared} levels produced an all-zero vector -- "
        "threat_proximity is structurally dead, as it was in production"
    )
