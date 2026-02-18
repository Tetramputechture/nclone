"""Adapter for nclone_native Rust simulator for use with TransitionEnumerator.

Wraps the Rust Simulator with the interface expected by SimulationPhiGridBuilder:
- sim.ninja with save_state/restore_state and xpos/ypos/state
- sim.tick_physics_only(h_input, jump_input)
- sim.tile_dic for GridClassifier and cache key
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Entity type constants (match nclone-native types.rs)
_ENTITY_EXIT_DOOR = 3
_ENTITY_EXIT_SWITCH = 4


def _get_switch_door_from_entities(entities) -> Tuple[Tuple[Tuple[float, float], bool], Tuple[float, float]]:
    """Get (switch_pos, switch_active), door_pos from entity list."""
    switch_pos, switch_active = (0.0, 0.0), True
    door_pos = (0.0, 0.0)
    for e in entities:
        if e.entity_type == _ENTITY_EXIT_SWITCH:
            switch_pos = (e.x, e.y)
            switch_active = e.active
        elif e.entity_type == _ENTITY_EXIT_DOOR:
            door_pos = (e.x, e.y)
    return (switch_pos, switch_active), door_pos


class NativeNinjaProxy:
    """Duck-typed proxy for ninja, backed by Rust simulator.

    Used by TransitionEnumerator which expects ninja.save_state(), restore_state(),
    xpos, ypos, state. The _inject_state path uses sim.inject_ninja_for_enumeration()
    directly (batch call) instead of individual property writes.
    """

    def __init__(self, rust_sim):
        self._sim = rust_sim

    def save_state(self):
        return self._sim.save_ninja_state()

    def restore_state(self, snapshot):
        self._sim.restore_ninja_state(snapshot)

    @property
    def state(self):
        return self._sim.get_ninja_state_code()

    @property
    def xpos(self):
        return self._sim.get_ninja_pos()[0]

    @property
    def ypos(self):
        return self._sim.get_ninja_pos()[1]


class NativeSimAdapter:
    """Wraps nclone_native.Simulator for use with TransitionEnumerator.

    Provides the same interface as NPlayHeadless.sim: ninja, tick_physics_only,
    tile_dic. The tile_dic is built from Rust get_tile_dict() for GridClassifier
    and cache key computation.
    """

    def __init__(self, map_data):
        """Create adapter from raw map data (list of ints)."""
        from nclone_native import Simulator

        self._sim = Simulator.load(map_data)
        self.ninja = NativeNinjaProxy(self._sim)
        self.tile_dic = self._build_tile_dic()

    def _build_tile_dic(self):
        """Build {(col, row): tile_type} from Rust tile grid."""
        return dict(self._sim.get_tile_dict())

    def tick_physics_only(self, h_input: int, jump_input: int) -> None:
        """Execute one physics frame without entity AI."""
        self._sim.tick_physics_only(h_input, jump_input)

    def inject_ninja_for_enumeration(self, x: float, y: float, vx: float, vy: float) -> None:
        """Inject position/velocity for transition enumeration (batch call)."""
        self._sim.inject_ninja_for_enumeration(x, y, vx, vy)


def create_native_sim_from_map_path(map_path: str) -> Tuple[NativeSimAdapter, Tuple[float, float], Tuple[float, float]]:
    """Create NativeSimAdapter and extract switch/exit positions from a map file.

    Returns:
        Tuple of (adapter, switch_pos, exit_pos).
    """
    with open(map_path, "rb") as f:
        map_data = [int(b) for b in f.read()]

    adapter = NativeSimAdapter(map_data)
    state = adapter._sim.get_state()
    (switch_pos, _), exit_pos = _get_switch_door_from_entities(state.entities)

    if switch_pos == (0.0, 0.0):
        logger.warning(f"Switch position is (0,0) for {map_path} - entities may not be loaded")
    if exit_pos == (0.0, 0.0):
        logger.warning(f"Exit position is (0,0) for {map_path} - entities may not be loaded")

    return adapter, switch_pos, exit_pos
