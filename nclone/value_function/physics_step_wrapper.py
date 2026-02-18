"""Physics step wrapper for stateless simulation queries.

This module provides PhysicsStepWrapper, which wraps the nclone_native Rust
simulator to provide a stateless interface suitable for value precomputation
and runtime PBRS evaluation.

Uses save_ninja_state/restore_ninja_state and inject_ninja_for_value_iteration
for 5D state injection (position, velocity, jump_duration).

Key design principles:
- Uses native Rust physics for performance
- Exact behavior match with game physics
- Provides tile_dic and entity_dic for level inspection and goal extraction
- Exposes contact-state queries for surface-graph PBRS
"""

from typing import Dict, Tuple, Optional, List, Any
import logging

logger = logging.getLogger(__name__)

# Entity type constants (match nclone-native types.rs)
_ENTITY_EXIT_DOOR = 3
_ENTITY_EXIT_SWITCH = 4


def _build_entity_dic_from_rust(entities) -> dict:
    """Build entity_dic from Rust EntitySnapshot list for position validity.
    Matches Python entity_dic layout: entity_dic[1], entity_dic[21] for toggle mines.
    """
    entity_dic = {1: [], 3: [], 4: [], 6: [], 7: [], 21: []}

    class _EntityProxy:
        """Minimal proxy for position_validity overlap check."""

        def __init__(self, x: float, y: float, state: int):
            self.xpos = x
            self.ypos = y
            self.state = state

    for e in entities:
        t = e.entity_type
        if t == 1:  # ToggleMine (untoggled)
            entity_dic[1].append(_EntityProxy(e.x, e.y, e.state))
        elif t == 21:  # ToggleMineToggled
            entity_dic[21].append(_EntityProxy(e.x, e.y, e.state))

    return entity_dic


class PhysicsStepWrapper:
    """Stateless physics step interface for value iteration using native Rust simulator.

    Wraps nclone_native.Simulator to provide stateless queries. Given any state
    and action, returns the resulting next state without modifying the actual
    simulator state.

    Uses save_ninja_state/restore_ninja_state and inject_ninja_for_value_iteration
    for 5D state injection (position, velocity, jump_duration).

    Attributes:
        sim: Reference to nclone_native.Simulator
        frames_per_step: Number of physics frames per step (default 5)
        tile_dic: {(col, row): tile_type} for position validity / compatibility
        entity_dic: {entity_type: [entities]} for position validity / compatibility
    """

    def __init__(
        self,
        simulator=None,
        map_data: Optional[List[int]] = None,
        map_path: Optional[str] = None,
        frames_per_step: int = 5,
    ):
        """Initialize physics step wrapper.

        Args:
            simulator: nclone_native.Simulator instance (optional)
            map_data: Raw map bytes as list of ints (optional)
            map_path: Path to .map file (optional)
            frames_per_step: Number of physics frames per step (default 5)

        Exactly one of simulator, map_data, or map_path must be provided.
        """
        from nclone_native import Simulator as NativeSimulator

        # Value function precomputation always uses native Rust physics (no Python fallback)
        if simulator is not None:
            self._sim = simulator
        elif map_data is not None:
            self._sim = NativeSimulator.load(map_data)
        elif map_path is not None:
            with open(map_path, "rb") as f:
                map_data = [int(b) for b in f.read()]
            self._sim = NativeSimulator.load(map_data)
        else:
            raise ValueError("Must provide one of: simulator, map_data, or map_path")

        self.sim = self._sim  # Alias for compatibility
        self.frames_per_step = frames_per_step

        # Verify we're using native Rust (has inject_ninja_for_value_iteration, get_ninja_velocity)
        if not hasattr(self._sim, "inject_ninja_for_value_iteration") or not hasattr(
            self._sim, "get_ninja_velocity"
        ):
            raise RuntimeError(
                "PhysicsStepWrapper requires nclone_native Rust simulator "
                "(must have inject_ninja_for_value_iteration, get_ninja_velocity)"
            )

        # Build tile_dic and entity_dic for position validity / goal extraction
        self.tile_dic = dict(self._sim.get_tile_dict())
        state = self._sim.get_state()
        self.entity_dic = _build_entity_dic_from_rust(state.entities)

    def get_goal_position(self, goal_type: str) -> Tuple[float, float]:
        """Extract goal position from level entities.

        Args:
            goal_type: "switch" or "exit"

        Returns:
            Goal position (x, y) in pixels
        """
        state = self._sim.get_state()
        for e in state.entities:
            if goal_type == "switch" and e.entity_type == _ENTITY_EXIT_SWITCH:
                return (e.x, e.y)
            if goal_type == "exit" and e.entity_type == _ENTITY_EXIT_DOOR:
                return (e.x, e.y)
        raise ValueError(f"Goal {goal_type} not found in level")

    def step(
        self,
        state: Tuple[float, float, float, float, int],
        action: Tuple[int, int],
    ) -> Tuple[Tuple[float, float, float, float, int], bool]:
        """Execute multiple physics frames from given state with constant action.

        Args:
            state: (x, y, vx, vy, jump_duration) in pixels/frames
            action: (h_input, jump_input)

        Returns:
            Tuple of (next_state, is_death)
        """
        snapshot = self._sim.save_ninja_state()

        x, y, vx, vy, jump_dur = state
        self._sim.inject_ninja_for_value_iteration(x, y, vx, vy, jump_dur)

        h_input, jump_input = action
        is_death = False

        try:
            for _ in range(self.frames_per_step):
                self._sim.tick_physics_only(h_input, jump_input)
                if self._sim.get_ninja_state_code() in (6, 7):
                    is_death = True
                    break
        except Exception as e:
            self._sim.restore_ninja_state(snapshot)
            raise RuntimeError(f"Physics step failed: {e}") from e

        next_state = (
            self._sim.get_ninja_pos()[0],
            self._sim.get_ninja_pos()[1],
            self._sim.get_ninja_velocity()[0],
            self._sim.get_ninja_velocity()[1],
            self._sim.get_ninja_jump_duration(),
        )

        self._sim.restore_ninja_state(snapshot)
        return next_state, is_death

    def is_state_valid(self, state: Tuple[float, float, float, float, int]) -> bool:
        """Check if a state is within valid bounds."""
        from ..constants.physics_constants import (
            MAX_JUMP_DURATION,
            MAX_HOR_SPEED,
            TERMINAL_VELOCITY,
        )

        x, y, vx, vy, jump_dur = state

        if x < 0 or x > 1056 or y < 0 or y > 600:
            return False
        if abs(vx) > MAX_HOR_SPEED * 1.5 or abs(vy) > TERMINAL_VELOCITY * 1.5:
            return False
        if jump_dur < 0 or jump_dur > MAX_JUMP_DURATION:
            return False

        return True

    # ------------------------------------------------------------------
    # Contact-state queries for surface-graph PBRS
    # ------------------------------------------------------------------
    def get_state_code(self) -> int:
        """Return the ninja's current state code.

        Common codes: 0=grounded, 1=jumping, 2=falling,
        5=wall_sliding, 6=dead, 7=awaiting_death.
        """
        return self._sim.get_ninja_state_code()

    def get_contact_info(self) -> Dict[str, Any]:
        """Read contact information from the NinjaSnapshot.

        Returns a dict with keys:
            airborn (bool), walled (bool), wall_normal (float),
            floor_count (int), wall_count (int),
            floor_normal_x (float), floor_normal_y (float),
            state_code (int).
        """
        state = self._sim.get_state()
        n = state.ninja
        return {
            "airborn": n.airborn,
            "walled": n.walled,
            "wall_normal": n.wall_normal,
            "floor_count": n.floor_count,
            "wall_count": n.wall_count,
            "floor_normal_x": n.floor_normalized_x,
            "floor_normal_y": n.floor_normalized_y,
            "state_code": n.state,
        }

    @staticmethod
    def derive_contact_state(state_code: int) -> int:
        """Map a ninja state code to a PBRS contact-state constant.

        Python ninja states: 0=Immobile, 1=Running, 2=Ground sliding (all floor),
        3=Jumping, 4=Falling (airborne), 5=Wall sliding (wall).
        Native sim uses same codes.

        Returns:
            0 = CONTACT_FLOOR, 1 = CONTACT_WALL, 2 = CONTACT_AIRBORNE.
        """
        if state_code in (0, 1, 2):  # Immobile, Running, Ground sliding
            return 0
        if state_code == 5:  # wall sliding
            return 1
        return 2  # jumping / falling / dead / other
