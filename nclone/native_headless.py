"""NativePlayHeadless: Python wrapper around Rust native physics engine.

Implements the same API as NPlayHeadless but delegates physics to nclone_native.
No rendering support -- this is for training only.
"""

import os
import time
import numpy as np
from typing import Optional, List

import nclone_native
from nclone_native import constants as nc

from .constants.physics_constants import (
    MAX_HOR_SPEED,
    GRAVITY_JUMP,
    GRAVITY_FALL,
    DRAG_REGULAR,
    DRAG_SLOW,
    FRICTION_GROUND,
    FRICTION_GROUND_SLOW,
    MAX_JUMP_DURATION,
)


class NativePlayHeadless:
    """Drop-in replacement for NPlayHeadless that uses native Rust physics.

    This class implements the identical API surface used by the gym environment:
    - load_map / load_map_from_map_data
    - tick / tick_physics_only
    - reset / fast_reset
    - get_ninja_state
    - sim attribute (with ninja, frame, entity_dic, tile_dic, etc.)

    No rendering support. For rendering, use the original NPlayHeadless.
    """

    def __init__(
        self,
        render_mode: str = "grayscale_array",
        enable_animation: bool = False,
        enable_logging: bool = False,
        enable_debug_overlay: bool = False,
        seed: Optional[int] = None,
        enable_rendering: bool = False,
        debug: bool = False,
    ):
        self.render_mode = render_mode
        self.enable_rendering = False  # Never render in native mode
        self.sim_renderer = (
            None  # No rendering in native mode (hasattr checks will skip)
        )
        self.debug = debug
        self.current_tick = 0
        self.current_map_data = None
        self._map_just_loaded = False
        self._native_sim = None  # nclone_native.Simulator

        # Create a lightweight sim proxy for code that accesses self.sim.ninja etc.
        self.sim = _SimProxy(headless=self)

    def load_map_from_map_data(self, map_data: List[int]):
        """Load a map from integer map data."""
        map_data_int = [int(x) for x in map_data]
        self._native_sim = nclone_native.Simulator.load(map_data_int)
        self.current_map_data = map_data_int
        self._map_just_loaded = True
        self.current_tick = 0
        self.sim._tile_dic_cache_valid = False
        # A NEW level invalidates any relevant-mine cull installed for the
        # previous one. Without this, a cull leaked from a mine-dense level
        # silently drops the new level's mines from every derived view (obs
        # overlays, hazard map, graph) while the sim still kills on them.
        # Also resets the cached entity-build index plan.
        self.sim.set_relevant_mine_keys(None)
        self._sync_proxy()

    def load_map(self, map_path: str):
        """Load a map from a binary file."""
        with open(map_path, "rb") as f:
            map_data = [int(b) for b in f.read()]
        self.load_map_from_map_data(map_data)

    def reset(self):
        """Full reset (reloads entities from current map data)."""
        if self.current_map_data is not None:
            self._native_sim = nclone_native.Simulator.load(self.current_map_data)
            self.current_tick = 0
            self.sim._tile_dic_cache_valid = False
            self._sync_proxy()

    def fast_reset(self):
        """Fast reset for same-level training."""
        if self._native_sim is not None:
            self._native_sim.fast_reset()
            self.current_tick = 0
            # Tile grid unchanged on fast_reset (same level)
            self._sync_proxy()

    def tick(self, horizontal_input: int, jump_input: int):
        """Execute one physics frame."""
        self.current_tick += 1
        self._native_sim.tick(horizontal_input, jump_input)
        self._sync_proxy()

    def tick_physics_only(self, horizontal_input: int, jump_input: int):
        """Execute one physics frame without entity AI."""
        self.current_tick += 1
        self._native_sim.tick_physics_only(horizontal_input, jump_input)
        self._sync_proxy()

    def _sync_proxy(self):
        """Sync the sim proxy from the native state snapshot."""
        if self._native_sim is None:
            return
        env = getattr(self.sim, "gym_env", None) if self.sim else None
        profile = env and getattr(env, "profile_enabled", False)

        if profile:
            t0 = time.perf_counter()
        # The tile grid is static per level. Pay for the full snapshot only on the
        # first sync after a (re)load to populate tile_dic; every subsequent tick
        # uses the lean snapshot that omits the ~1100-byte tile grid copy. At
        # frame_skip=1 (FiGAR-PPO) this is the per-primitive-frame hot path.
        if self.sim is not None and self.sim._tile_dic_cache_valid:
            state = self._native_sim.get_state_dynamic()
        else:
            state = self._native_sim.get_state()
        if profile:
            env._profile_end("sync_get_state", t0)
        self.sim._update_from_snapshot(
            state, self._native_sim, profile_env=env if profile else None
        )

    def get_state_snapshot(self):
        """Capture complete simulation state for replay-free checkpoint restore.

        Returns a backend-tagged, picklable blob ``("native", bytes)`` covering
        the frame counter, the full ninja (all buffers + booleans via
        ``Ninja::to_bytes``) and every entity's mutable state.
        """
        if self._native_sim is None:
            return None
        return ("native", bytes(self._native_sim.save_full_state()))

    def set_state_snapshot(self, blob):
        """Restore a snapshot from :meth:`get_state_snapshot` (no action replay)."""
        backend, data = blob
        if backend != "native":
            raise ValueError(
                f"Cannot restore a {backend!r} snapshot into the native backend"
            )
        if self._native_sim is None:
            raise RuntimeError("native simulator not loaded")
        # Returns False if the snapshot is incompatible with the loaded level
        # (e.g. a Go-Explore checkpoint from a different level / entity count).
        # Raise so the env's _restore_from_snapshot falls back to action replay.
        if not self._native_sim.restore_full_state(list(data)):
            raise ValueError("snapshot incompatible with current level")
        self._sync_proxy()

    def get_ninja_state(self):
        """Get ninja state as a list of 23 normalized floats matching NPlayHeadless exactly.

        Returns 23 features (time_remaining appended separately in base_environment.py).
        """
        n = self.sim.ninja  # _NinjaProxy from snapshot
        from .gym_environment.constants import LEVEL_WIDTH, LEVEL_HEIGHT

        state = []

        # Velocity (2)
        velocity_x = np.clip(n.xspeed / MAX_HOR_SPEED, -1, 1)
        velocity_y = np.clip(n.yspeed / MAX_HOR_SPEED, -1, 1)
        state.extend([velocity_x, velocity_y])

        # Movement state categories (4)
        ground_movement = 1.0 if n.state in (0, 1, 2) else -1.0
        air_movement = 1.0 if n.state in (3, 4) else -1.0
        wall_interaction = 1.0 if n.state == 5 else -1.0
        special_states = 1.0 if n.state in (6, 7, 8, 9) else -1.0
        state.extend([ground_movement, air_movement, wall_interaction, special_states])

        # Airborne (1)
        state.append(1.0 if n.airborn else -1.0)

        # Buffers (4). (b + 1)/(window + 1) so INACTIVE (-1) -> -1.0 and every
        # ACTIVE value 0..window is distinct in (-1, 1] (b=0 "just armed" no
        # longer aliases inactive). Must match nplay_headless exactly.
        buffer_window = 5.0
        state.append(((max(n.jump_buffer, -1) + 1) / (buffer_window + 1)) * 2 - 1)
        state.append(((max(n.floor_buffer, -1) + 1) / (buffer_window + 1)) * 2 - 1)
        state.append(((max(n.wall_buffer, -1) + 1) / (buffer_window + 1)) * 2 - 1)

        # Jump duration remaining (1)
        if n.state == 3:
            jdr = max(0, MAX_JUMP_DURATION - n.jump_duration) / MAX_JUMP_DURATION
        else:
            jdr = 0.0
        state.append(jdr)

        # Surface contact (3). Wall contact gates on the live ``walled`` flag —
        # ``wall_count`` is never incremented during simulation (matches
        # nplay_headless).
        state.append((min(n.floor_count, 1) * 2) - 1)
        state.append(1.0 if n.walled else -1.0)
        state.append((min(n.ceiling_count, 1) * 2) - 1)

        # Surface normals (3). wall_normal gated on ``walled`` (see above).
        state.append(n.floor_normalized_x)
        state.append(n.floor_normalized_y)
        state.append(float(n.wall_normal) if n.walled else 0.0)

        # Walled (1)
        state.append(1.0 if n.walled else -1.0)

        # Physics parameters (3)
        gravity_range = GRAVITY_FALL - GRAVITY_JUMP
        state.append(
            (n.applied_gravity - GRAVITY_JUMP) / gravity_range * 2 - 1
            if gravity_range > 1e-6
            else 0.0
        )
        drag_range = DRAG_REGULAR - DRAG_SLOW
        state.append(
            (n.applied_drag - DRAG_SLOW) / drag_range * 2 - 1
            if drag_range > 1e-6
            else 0.0
        )
        friction_range = FRICTION_GROUND - FRICTION_GROUND_SLOW
        state.append(
            (n.applied_friction - FRICTION_GROUND_SLOW) / friction_range * 2 - 1
            if friction_range > 1e-6
            else 0.0
        )

        # Position (2)
        state.append(n.xpos / LEVEL_WIDTH)
        state.append(n.ypos / LEVEL_HEIGHT)

        return state

    def render(self, debug_info=None):
        """No rendering in native mode."""
        return None

    def get_valid_action_mask(self):
        """Get valid action mask from cached snapshot when available.

        The proxy is synced after each tick; use cached action_mask to avoid
        redundant get_state() FFI call during observation building.
        """
        if self.sim._snapshot is not None:
            return list(self.sim._snapshot.action_mask)
        state = self._native_sim.get_state()
        return list(state.action_mask)

    def get_action_mask(self):
        """Alias for get_valid_action_mask (NPlayHeadless API)."""
        return self.get_valid_action_mask()

    def ninja_has_won(self):
        return self.sim.ninja.has_won()

    def ninja_has_died(self):
        return self.sim.ninja.has_died()

    def ninja_position(self):
        return self.sim.ninja.xpos, self.sim.ninja.ypos

    def ninja_velocity(self):
        return self.sim.ninja.xspeed, self.sim.ninja.yspeed

    def ninja_velocity_old(self):
        return self.sim.ninja.xspeed_old, self.sim.ninja.yspeed_old

    def ninja_airborn_old(self):
        return self.sim.ninja.airborn_old

    def ninja_death_cause(self):
        return self.sim.ninja.death_cause

    def get_ninja_terminal_impact(self):
        return self.sim.ninja.terminal_impact

    def _sim_exit_switch(self):
        """Get exit switch proxy. For compatibility - reads from snapshot when available."""
        switch_pos, switch_active, _ = self._get_switch_door_from_snapshot()
        if switch_pos == (0.0, 0.0):
            return None
        proxy = _EntityExitSwitchProxy(
            switch_pos[0], switch_pos[1], switch_active, 0, _ENTITY_EXIT_SWITCH
        )
        return proxy

    def _get_switch_door_from_snapshot(self):
        """Read switch/door from the per-tick cache (scanned once in _update_from_snapshot)."""
        cache = getattr(self.sim, "_switch_door_cache", None)
        if cache is None:
            return (0.0, 0.0), True, (0.0, 0.0)
        (switch_pos, switch_active), door_pos = cache
        return switch_pos, switch_active, door_pos

    def exit_switch_activated(self):
        """True if switch collected (active=False)."""
        _, switch_active, _ = self._get_switch_door_from_snapshot()
        return not switch_active

    def exit_switch_position(self):
        """Return (x, y) of exit switch or (0, 0)."""
        switch_pos, _, _ = self._get_switch_door_from_snapshot()
        return switch_pos

    def _sim_exit_door(self):
        """Get exit door. Reads from snapshot when available."""
        _, _, door_pos = self._get_switch_door_from_snapshot()
        if door_pos != (0.0, 0.0):
            return _EntityExitProxy(
                door_pos[0], door_pos[1], True, 0, _ENTITY_EXIT_DOOR
            )
        self.sim._ensure_entity_dic()
        for e in self.sim.entity_dic.get(3, []):
            if isinstance(e, _EntityExitProxy):
                return e
        return None

    def exit_door_position(self):
        """Return (x, y) of exit door or (0, 0)."""
        _, _, door_pos = self._get_switch_door_from_snapshot()
        return door_pos

    def locked_doors(self):
        """Return locked door entities (type 6), incl. their switch coords.

        Mirrors ``NPlayHeadless.locked_doors``; the native engine supports locked
        doors and ``_build_entity_dic`` builds type-6 ``_EntityDoorLockedProxy``
        objects exposing xpos/ypos (switch), ``segment`` (door barrier), and
        active/closed state.
        """
        return list(self.sim.entity_dic.get(6, []))

    def locked_door_switches(self):
        """Return locked-door switch entities (type 7).

        Mirrors ``NPlayHeadless.locked_door_switches``. In nclone the switch is
        embedded in the type-6 locked-door entity, so type 7 is unused for locked
        doors and this is empty (matching the Python backend).
        """
        return list(self.sim.entity_dic.get(7, []))

    def get_mine_entities(self):
        """Return (toggle_mines, toggled_mines) for spatial context computation."""
        return (
            self.sim.entity_dic.get(1, []),
            self.sim.entity_dic.get(21, []),
        )

    def set_relevant_mine_keys(self, keys):
        """Install the relevant-mine cull at the entity_dic source so every
        downstream consumer (entity proxies, mine-state scan, obs, graph) sees
        only the relevant subset on mine-dense levels."""
        self.sim.set_relevant_mine_keys(keys)

    def get_all_mine_positions(self):
        """Raw (x, y, state, type) for every toggle mine (bypasses the cull)."""
        return self.sim.get_all_mine_positions()

    def get_all_mine_data_for_visualization(self) -> List[dict]:
        """Get all mine data for route visualization.

        Returns data for both entity types 1 (untoggled mines) and 21 (toggled mines).
        Mirrors NPlayHeadless API for compatibility with _build_episode_info.
        """
        from .constants.physics_constants import TOGGLE_MINE_RADII

        mines = []
        entity_dic = self.sim.entity_dic

        for entity_type, default_state in [(1, 1), (21, 0)]:
            if entity_type not in entity_dic:
                continue
            for mine in entity_dic[entity_type]:
                if hasattr(mine, "xpos") and hasattr(mine, "ypos"):
                    state = getattr(mine, "state", default_state)
                    radius = TOGGLE_MINE_RADII.get(state, 4.0)
                    mines.append(
                        {
                            "x": float(mine.xpos),
                            "y": float(mine.ypos),
                            "state": int(state),
                            "radius": float(radius),
                        }
                    )
        return mines

    def get_tile_data(self):
        """Get tile data from simulator proxy."""
        return self.sim.tile_dic

    def get_goal_positions_for_features(self):
        """Return pre-computed goal positions for reachability feature computation (O(1) lookups)."""
        return {
            "switch_pos": self.exit_switch_position(),
            "exit_pos": self.exit_door_position(),
            "switch_activated": self.exit_switch_activated(),
            "locked_doors": self.locked_doors(),
            "locked_door_switches": self.locked_door_switches(),
            "mine_entities": (
                self.sim.entity_dic.get(1, []),
                self.sim.entity_dic.get(21, []),
            ),
        }


class _EntityProxy:
    """Minimal proxy for entity_dic compatibility. Needs xpos, ypos, active, state, cell."""

    __name__ = "Generic"

    def __init__(self, x: float, y: float, active: bool, state: int, entity_type: int):
        self.xpos = x
        self.ypos = y
        self.active = active
        self.state = state
        self.entity_type = entity_type
        self.parent = None
        # cell used for grid_entity lookup (24px tiles)
        self.cell = (int(x // 24), int(y // 24))


class _EntityExitSwitchProxy(_EntityProxy):
    """Proxy for exit switch. parent points to door."""

    pass


_EntityExitSwitchProxy.__name__ = "EntityExitSwitch"


class _EntityExitProxy(_EntityProxy):
    """Proxy for exit door."""

    pass


_EntityExitProxy.__name__ = "EntityExit"


class _DoorSegmentProxy:
    """Minimal door-segment proxy for the door barrier endpoints.

    Exposes ``x1/y1/x2/y2`` (read by the entity extractor and base_environment to
    recover the door midpoint) plus ``p1/p2`` tuples, mirroring the Python door
    segment interface.
    """

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.x1, self.y1 = p1
        self.x2, self.y2 = p2


class _EntityDoorLockedProxy(_EntityProxy):
    """Proxy for a locked door (type 6).

    Mirrors the Python EntityDoorLocked interface that downstream code reads:
    xpos/ypos and sw_xpos/sw_ypos are the SWITCH position; `segment` exposes the
    door barrier endpoints; `closed`/`active` track door and switch state.
    """

    def __init__(self, sw_x, sw_y, door_x, door_y, orientation, active, closed):
        # Entity position == switch position (matches Python EntityDoorBase, which
        # resets xpos/ypos to the switch after building the barrier segment, and
        # exposes the barrier ONLY via `segment` — NOT a door_xpos attribute. We
        # mirror that exactly so the shared entity overlay / extractor produce
        # byte-identical output across backends.
        super().__init__(sw_x, sw_y, active, 0, 6)
        self.sw_xpos = sw_x
        self.sw_ypos = sw_y
        self.orientation = orientation
        self.closed = closed
        # Door endpoints: 24px segment, vertical iff orientation in (0, 4).
        if orientation in (0, 4):
            self.segment = _DoorSegmentProxy(
                (door_x, door_y - 12.0), (door_x, door_y + 12.0)
            )
        else:
            self.segment = _DoorSegmentProxy(
                (door_x - 12.0, door_y), (door_x + 12.0, door_y)
            )


_EntityDoorLockedProxy.__name__ = "EntityDoorLocked"


class _EntityOneWayProxy(_EntityProxy):
    """Proxy for a one-way platform (type 11).

    Mirrors the Python ``EntityOneWayPlatform`` attributes read downstream: xpos/
    ypos plus the orientation-derived blocking normal (normal_x, normal_y). The
    normal is computed via the same ``map_orientation_to_vector`` as Python so the
    shared entity-overlay builder is byte-identical across backends.
    """

    def __init__(self, x: float, y: float, orientation: int):
        super().__init__(x, y, True, 0, 11)
        from .physics import map_orientation_to_vector

        self.orientation = orientation
        self.normal_x, self.normal_y = map_orientation_to_vector(orientation)


_EntityOneWayProxy.__name__ = "EntityOneWayPlatform"


# Entity type constants from nclone
_ENTITY_TOGGLE_MINE = 1
_ENTITY_EXIT_DOOR = 3
_ENTITY_EXIT_SWITCH = 4
_ENTITY_LOCKED_DOOR = 6
_ENTITY_ONE_WAY = 11
_ENTITY_TOGGLE_MINE_TOGGLED = 21


def _get_switch_door_from_entities(entities) -> tuple:
    """Get (switch_x, switch_y, switch_active), (door_x, door_y) from entity list.

    Reads directly from EntitySnapshot list - used during frame loop to avoid
    building full entity_dic. Returns ((0,0), True), ((0,0), (0,0)) if not found.
    """
    switch_pos, switch_active = (0.0, 0.0), True
    door_pos = (0.0, 0.0)
    for e in entities:
        if e.entity_type == _ENTITY_EXIT_SWITCH:
            switch_pos = (e.x, e.y)
            switch_active = e.active
        elif e.entity_type == _ENTITY_EXIT_DOOR:
            door_pos = (e.x, e.y)
    return (switch_pos, switch_active), door_pos


def _build_entity_dic(entities) -> dict:
    """Build entity_dic from Rust EntitySnapshot list. Matches Python entity_dic layout.
    entity_dic[3] contains BOTH EntityExitSwitch and EntityExit (door) per Python convention.

    ``entities`` may be the full snapshot list or a pre-selected subset (the
    relevant-mine cull is applied upstream via the cached index plan in
    ``_ensure_entity_dic`` so this hot path never re-scans culled mines).
    """
    entity_dic = {1: [], 3: [], 4: [], 6: [], 7: [], 11: [], 21: []}
    exit_switches = []
    exit_doors = []

    for e in entities:
        t = e.entity_type
        if t == 1:  # ToggleMine (untoggled)
            entity_dic[1].append(_EntityProxy(e.x, e.y, e.active, e.state, t))
        elif t == 21:  # ToggleMineToggled
            entity_dic[21].append(_EntityProxy(e.x, e.y, e.active, e.state, t))
        elif t == 11:  # OneWay: static directional barrier; orientation -> normal.
            entity_dic[11].append(_EntityOneWayProxy(e.x, e.y, e.orientation))
        elif t == 4:  # ExitSwitch
            proxy = _EntityExitSwitchProxy(e.x, e.y, e.active, e.state, t)
            exit_switches.append(proxy)
        elif t == 3:  # ExitDoor
            proxy = _EntityExitProxy(e.x, e.y, True, e.state, t)
            exit_doors.append(proxy)
        elif t == 6:  # LockedDoor: (e.x, e.y) is the switch; door_* is the barrier.
            entity_dic[6].append(
                _EntityDoorLockedProxy(
                    e.x,
                    e.y,
                    e.door_x,
                    e.door_y,
                    e.door_orientation,
                    e.active,
                    e.door_closed,
                )
            )

    # entity_dic[3] = doors + switches (Python convention). Link switch.parent -> door.
    for door in exit_doors:
        entity_dic[3].append(door)
    for sw in exit_switches:
        entity_dic[3].append(sw)
        if exit_doors:
            sw.parent = exit_doors[0]

    return entity_dic


def _build_tile_dic_from_flat(flat: list, profile_env=None) -> dict:
    """Build tile_dic from flat tile grid (row-major, 44*25).
    Uses tile_grid from SimulatorState when available (single FFI call).
    """
    profile = profile_env and getattr(profile_env, "profile_enabled", False)
    if profile:
        t0 = time.perf_counter()
    # MAP_WIDTH=44, MAP_HEIGHT=25, row-major
    tile_dic = {}
    for y in range(25):
        for x in range(44):
            tile_dic[(x, y)] = flat[y * 44 + x]
    if profile:
        profile_env._profile_end("sync_build_tile", t0)
    return tile_dic


class _SimProxy:
    """Lightweight proxy that mimics the Simulator + Ninja interface for Python code.

    Updated each tick from the native SimulatorState snapshot.
    """

    def __init__(self, headless=None):
        self.ninja = _NinjaProxy()
        self.frame = 0
        self.sim_config = _SimConfigProxy()
        self._entity_dic = {}
        self.tile_dic = {}
        self.grid_entity = {}
        self.gym_env = None
        self._snapshot = None
        self._switch_door_cache = None  # ((switch_pos, switch_active), door_pos) per tick
        self._headless = headless
        self.map_data = None
        # Parity with nclone.nsim.Simulator: action-masking knobs the env reads/writes.
        # Native physics ignores these; kept so base_environment writes and the debug
        # mixin read don't depend on backend.
        self._action_masking_mode = None
        self._straightness_masking_debug = False
        # Tile grid is static per level - cache to avoid rebuilding every tick
        self._tile_dic_cache_valid = False
        # Entity dic built lazily on first access (after frame loop)
        self._entity_dic_valid = False
        # Optional relevant-mine cull: set of (round(x), round(y)) for mines that
        # border reachable space. When set, _build_entity_dic skips the rest.
        self._relevant_mine_keys = None
        # Cached entity-build plan: (snapshot_len, [indices to build]). Entity
        # positions/types are static per level, so the set of indices we keep
        # (all non-mine entities + relevant mines) is invariant — computed once
        # rather than re-scanning all entities (and round()-checking each) every
        # tick. Only mutable state (read fresh from the snapshot) changes.
        self._entity_build_plan = None

    @property
    def entity_dic(self):
        """Lazy-load entity_dic on first access (e.g. during observation build)."""
        self._ensure_entity_dic()
        return self._entity_dic

    def set_relevant_mine_keys(self, keys):
        """Install (or clear, with ``None``) the relevant-mine key set and force a
        rebuild so the next entity_dic reflects the cull."""
        self._relevant_mine_keys = keys
        self._entity_build_plan = None  # recompute the index plan against new keys
        self._entity_dic_valid = False

    def get_all_mine_positions(self):
        """Raw (x, y, state, type) for EVERY toggle mine in the snapshot, bypassing
        the relevant-mine cull. The env uses this to compute the relevant set from
        the full mine population (avoids a circular dependency on the filtered dic)."""
        if self._snapshot is None:
            return []
        return [
            (e.x, e.y, e.state, e.entity_type)
            for e in self._snapshot.entities
            if e.entity_type in (1, 21)
        ]

    def get_tile_grid(self):
        """Return the 966-element flat tile slice (map_data[184:1150]).

        Matches ``nclone.nsim.Simulator.get_tile_grid`` byte-for-byte (column-major
        ``tile_data[x + y * 42]``), so callers reshape identically regardless of backend.
        """
        map_data = self.map_data
        if map_data is None and self._headless is not None:
            map_data = self._headless.current_map_data
        if map_data is None:
            return []
        return map_data[184:1150]

    def _update_from_snapshot(self, state, native_sim, profile_env=None):
        """Update proxy from a SimulatorState snapshot."""
        self._snapshot = state
        self.frame = state.frame
        self.ninja._update_from_snapshot(state.ninja)
        self._entity_dic_valid = False  # Invalidate - entities may have changed
        # Scan entities for switch/door ONCE per tick and cache. Goal accessors
        # (exit_switch_position/activated, exit_door_position) and the reward's
        # goal-position lookup hit these many times per step; scanning every call
        # was ~11 entity sweeps/step.
        self._switch_door_cache = _get_switch_door_from_entities(state.entities)
        profile = profile_env and getattr(profile_env, "profile_enabled", False)
        # Tile grid is static per level - only rebuild when cache invalidated
        # Uses state.tile_grid (bundled in get_state) - no separate get_tile_grid() FFI
        if not self._tile_dic_cache_valid:
            flat = getattr(state, "tile_grid", None)
            if flat is None:
                # Fallback for older nclone_native without bundled tile_grid
                flat = native_sim.get_tile_grid()
            self.tile_dic = _build_tile_dic_from_flat(flat, profile_env=profile_env)
            self._tile_dic_cache_valid = True
        if self._headless and self._headless.current_map_data is not None:
            self.map_data = self._headless.current_map_data

    def _ensure_entity_dic(self):
        """Build entity_dic on first access (lazy). Called when observation needs it."""
        if self._entity_dic_valid and self._snapshot is not None:
            return
        if self._snapshot is None:
            self._entity_dic = {}
            return
        entities = self._snapshot.entities
        if self._relevant_mine_keys is not None:
            # Build (once) the static index plan: every non-mine entity plus the
            # relevant mines. Reused across ticks since entity positions/types
            # don't change — avoids re-scanning all entities + round()-checking
            # each mine on the per-tick hot path.
            plan = self._entity_build_plan
            if plan is None or plan[0] != len(entities):
                keys = self._relevant_mine_keys
                idxs = [
                    i
                    for i, e in enumerate(entities)
                    if e.entity_type not in (1, 21)
                    or (round(e.x), round(e.y)) in keys
                ]
                self._entity_build_plan = (len(entities), idxs)
                plan = self._entity_build_plan
            self._entity_dic = _build_entity_dic([entities[i] for i in plan[1]])
        else:
            self._entity_dic = _build_entity_dic(entities)
        self._entity_dic_valid = True


class _NinjaProxy:
    """Mimics the Ninja object interface using NinjaSnapshot data."""

    def __init__(self):
        # Initialize all fields to defaults
        self.xpos = 0.0
        self.ypos = 0.0
        self.xspeed = 0.0
        self.yspeed = 0.0
        self.xpos_old = 0.0
        self.ypos_old = 0.0
        self.xspeed_old = 0.0
        self.yspeed_old = 0.0
        self.state = 0
        self.airborn = False
        self.airborn_old = False
        self.walled = False
        self.wall_normal = 0.0
        self.terminal_impact = False
        self.jump_duration = 0
        self.jump_input_old = 0
        self.jump_buffer = -1
        self.floor_buffer = -1
        self.wall_buffer = -1
        self.launch_pad_buffer = -1
        self.last_jump_was_buffered = False
        self.floor_normalized_x = 0.0
        self.floor_normalized_y = -1.0
        self.ceiling_normalized_x = 0.0
        self.ceiling_normalized_y = 1.0
        self.floor_count = 0
        self.wall_count = 0
        self.ceiling_count = 0
        self.applied_gravity = 0.0
        self.applied_drag = 0.0
        self.applied_friction = 0.0
        self.hor_input = 0
        self.jump_input = 0
        self.frames_airborne = 0
        self.state_change_frame = 0
        self.previous_state = 0
        self.consecutive_floor_frames = 0
        self.consecutive_wall_frames = 0
        self.gold_collected = 0
        self.doors_opened = 0
        self.death_cause = None
        # Derived
        self.is_jump_useless = False

    def _update_from_snapshot(self, snap):
        """Update all fields from NinjaSnapshot."""
        self.xpos = snap.xpos
        self.ypos = snap.ypos
        self.xspeed = snap.xspeed
        self.yspeed = snap.yspeed
        self.xpos_old = snap.xpos_old
        self.ypos_old = snap.ypos_old
        self.xspeed_old = snap.xspeed_old
        self.yspeed_old = snap.yspeed_old
        self.state = snap.state
        self.airborn = snap.airborn
        self.airborn_old = snap.airborn_old
        self.walled = snap.walled
        self.wall_normal = snap.wall_normal
        self.terminal_impact = snap.terminal_impact
        self.jump_duration = snap.jump_duration
        self.jump_input_old = snap.jump_input_old
        self.jump_buffer = snap.jump_buffer
        self.floor_buffer = snap.floor_buffer
        self.wall_buffer = snap.wall_buffer
        self.launch_pad_buffer = snap.launch_pad_buffer
        self.last_jump_was_buffered = snap.last_jump_was_buffered
        self.floor_normalized_x = snap.floor_normalized_x
        self.floor_normalized_y = snap.floor_normalized_y
        self.ceiling_normalized_x = snap.ceiling_normalized_x
        self.ceiling_normalized_y = snap.ceiling_normalized_y
        self.floor_count = snap.floor_count
        self.wall_count = snap.wall_count
        self.ceiling_count = snap.ceiling_count
        self.applied_gravity = snap.applied_gravity
        self.applied_drag = snap.applied_drag
        self.applied_friction = snap.applied_friction
        self.hor_input = snap.hor_input
        self.jump_input = snap.jump_input
        self.frames_airborne = snap.frames_airborne
        self.state_change_frame = snap.state_change_frame
        self.previous_state = snap.previous_state
        self.consecutive_floor_frames = snap.consecutive_floor_frames
        self.consecutive_wall_frames = snap.consecutive_wall_frames
        self.gold_collected = snap.gold_collected
        self.doors_opened = snap.doors_opened
        self.death_cause = snap.death_cause

    def has_won(self):
        return self.state == 8

    def has_died(self):
        return self.state == 6 or self.state == 7

    def is_valid_target(self):
        return self.state not in (6, 8, 9)

    def get_valid_action_mask(self):
        """This would need the native sim to compute, so use proxy data."""
        # The action mask is computed during get_state() in native
        # Callers should use NativePlayHeadless.get_valid_action_mask() instead
        raise NotImplementedError(
            "Use NativePlayHeadless.get_valid_action_mask() instead"
        )


class _SimConfigProxy:
    """Minimal SimConfig proxy."""

    def __init__(self):
        self.basic_sim = True
        self.full_export = False
        self.tolerance = 1.0
        self.enable_anim = False
        self.log_data = False
        self.debug = False
