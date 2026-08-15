"""Half-cell grid-edge parity between the Python and native simulators.

Grid-walking entities (drones, thwumps, shove thwumps) never collide against
tile segments -- they ask whether the half-cell edge they are about to cross is
solid, which is answered by four grids on the simulator:

    hor_grid_edge_dic / ver_grid_edge_dic            static, built from tiles
    door_hor_grid_edge_dic / door_ver_grid_edge_dic  dynamic, per closed door

The native sim grew these in the entity-parity work (see
nclone-native/ENTITY_PARITY_PLAN.md 2.3). They must match Python EXACTLY: the
build rule is a *toggle* (`dic[k] = new if dic[k] == 0 else 0`) rather than the
accumulation the segment dictionaries use, so empty tiles clear edges their
neighbours wrote and tile iteration order is load-bearing. A single wrong cell
silently reroutes a drone patrol.

These tests are the gate on that, checked over real official maps rather than
synthetic ones so the awkward tile combinations are actually covered. The
incremental open/close path is covered by Rust unit tests in
`nclone-native/src/simulator.rs` (a switch collection has to actually happen,
which needs a driven ninja rather than a loaded map).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

nclone_native = pytest.importorskip("nclone_native")

from nclone.nsim import Simulator  # noqa: E402
from nclone.sim_config import SimConfig  # noqa: E402

# Dimensions of the four dictionaries, mirroring nsim.reset_map_tile_data.
HOR_W, HOR_H = 88, 51
VER_W, VER_H = 89, 50

OFFICIAL_ROOT = Path(__file__).resolve().parents[1] / "nclone" / "maps" / "official"
MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "npp-rl"
    / "datasets"
    / "official_compatible_manifest_native.json"
)

# Enough maps to cover the tile vocabulary without making the suite slow. Every
# locked-door map is added on top, since those are the only ones exercising the
# door layer at load.
N_SAMPLED = 40


def _map_paths() -> list[Path]:
    if MANIFEST.exists():
        comp = json.loads(MANIFEST.read_text())["compatible"]
        paths = [Path(c["path"]) for c in comp]
        locked = [Path(c["path"]) for c in comp if "locked_door" in c["entities"]]
    elif OFFICIAL_ROOT.exists():
        paths = [p for p in OFFICIAL_ROOT.rglob("*") if p.is_file() and p.stat().st_size > 1235]
        locked = []
    else:
        pytest.skip("no official maps or manifest available")
    paths = [p for p in paths if p.exists()]
    if not paths:
        pytest.skip("manifest paths do not resolve on this machine")
    sampled = random.Random(20260729).sample(paths, min(N_SAMPLED, len(paths)))
    # dict.fromkeys keeps order and de-dupes.
    return list(dict.fromkeys(sampled + [p for p in locked if p.exists()]))


def _load_both(map_data: list[int]):
    py = Simulator(SimConfig())
    py.load(map_data)
    return py, nclone_native.Simulator.load(map_data)


def _diff_grids(py, native) -> list[tuple]:
    """Return every (grid, x, y, python_value, native_value) that disagrees."""
    hor, ver, door_hor, door_ver = native.get_grid_edges()
    bad = []
    for x in range(HOR_W):
        for y in range(HOR_H):
            i = x * HOR_H + y
            if int(py.hor_grid_edge_dic[(x, y)]) != hor[i]:
                bad.append(("hor", x, y, py.hor_grid_edge_dic[(x, y)], hor[i]))
            if int(py.door_hor_grid_edge_dic[(x, y)]) != door_hor[i]:
                bad.append(("door_hor", x, y, py.door_hor_grid_edge_dic[(x, y)], door_hor[i]))
    for x in range(VER_W):
        for y in range(VER_H):
            i = x * VER_H + y
            if int(py.ver_grid_edge_dic[(x, y)]) != ver[i]:
                bad.append(("ver", x, y, py.ver_grid_edge_dic[(x, y)], ver[i]))
            if int(py.door_ver_grid_edge_dic[(x, y)]) != door_ver[i]:
                bad.append(("door_ver", x, y, py.door_ver_grid_edge_dic[(x, y)], door_ver[i]))
    return bad


@pytest.mark.parametrize("map_path", _map_paths(), ids=lambda p: p.name)
def test_grid_edges_match_python_on_load(map_path: Path):
    """All four grids agree cell-for-cell straight after loading a map."""
    map_data = [int(b) for b in map_path.read_bytes()]
    py, native = _load_both(map_data)
    bad = _diff_grids(py, native)
    assert not bad, f"{len(bad)} differing cells, first 5: {bad[:5]}"


def test_grid_edges_survive_fast_reset():
    """fast_reset must restore the door layer, not just the entity state."""
    paths = _map_paths()
    map_data = [int(b) for b in paths[0].read_bytes()]
    py, native = _load_both(map_data)
    before = native.get_grid_edges()
    native.fast_reset()
    assert native.get_grid_edges() == before, "fast_reset perturbed the grid edges"
    assert not _diff_grids(py, native)
