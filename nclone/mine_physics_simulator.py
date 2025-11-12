"""
Fast-forward physics simulator for mine death prediction.

This module provides lightweight physics simulation that reuses ninja physics
methods to accurately predict collisions with toggled mines over multiple frames.
"""

import math
from typing import List, Tuple

from .constants import (
    TOGGLE_MINE_RADII,
    MINE_DEATH_LOOKAHEAD_FRAMES,
    MINE_COLLISION_THRESHOLD,
)


class MinePhysicsSimulator:
    """
    Lightweight physics simulator for mine collision prediction.

    Reuses core ninja physics (integrate(), think()) to accurately simulate
    movement over multiple frames and detect collisions with toggled mines.
    """

    def __init__(self, sim):
        """
        Initialize simulator with simulation reference.

        Args:
            sim: Simulation object containing ninja and entities
        """
        self.sim = sim
        self.ninja = sim.ninja

    def simulate_action_sequence(
        self,
        initial_state: dict,
        action: int,
        mine_positions: List[Tuple[float, float]],
        num_frames: int = MINE_DEATH_LOOKAHEAD_FRAMES,
    ) -> bool:
        """
        Simulate action and check for mine collisions over multiple frames.

        Args:
            initial_state: Dictionary with ninja state (position, velocity, buffers, etc.)
            action: Action to simulate (0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP, 4=JUMP+LEFT, 5=JUMP+RIGHT)
            mine_positions: List of (x, y) positions of toggled mines
            num_frames: Number of frames to simulate

        Returns:
            True if collision detected with any mine, False otherwise
        """
        # Save original ninja state
        saved_state = self._save_ninja_state()

        # Restore to initial state
        self._restore_ninja_state(initial_state)

        # Convert action to inputs
        action_inputs = [
            (0, 0),  # 0: NOOP
            (-1, 0),  # 1: LEFT
            (1, 0),  # 2: RIGHT
            (0, 1),  # 3: JUMP
            (-1, 1),  # 4: JUMP+LEFT
            (1, 1),  # 5: JUMP+RIGHT
        ]
        hor_input, jump_input = action_inputs[action]

        # Simulate frames
        collision_detected = False
        for frame_idx in range(num_frames):
            # Set inputs for this frame
            self.ninja.hor_input = hor_input
            self.ninja.jump_input = jump_input

            # Run think() to process inputs and update state
            self.ninja.think()

            # Run integrate() to update position based on velocity
            self.ninja.integrate()

            # Check for collision at this frame
            if self._check_mine_collision(
                self.ninja.xpos, self.ninja.ypos, mine_positions
            ):
                collision_detected = True
                break

        # Restore original state
        self._restore_ninja_state(saved_state)

        return collision_detected

    def _check_mine_collision(
        self, x: float, y: float, mine_positions: List[Tuple[float, float]]
    ) -> bool:
        """
        Check if position collides with any toggled mine.

        Args:
            x: X position to check
            y: Y position to check
            mine_positions: List of (x, y) mine positions

        Returns:
            True if collision detected, False otherwise
        """
        toggled_mine_radius = TOGGLE_MINE_RADII[0]  # State 0 = toggled/deadly
        collision_threshold = (
            MINE_COLLISION_THRESHOLD  # NINJA_RADIUS + toggled_mine_radius
        )

        for mine_x, mine_y in mine_positions:
            dx = x - mine_x
            dy = y - mine_y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < collision_threshold:
                return True

        return False

    def _save_ninja_state(self) -> dict:
        """
        Save complete ninja state to dictionary.

        Returns:
            Dictionary containing all relevant state variables
        """
        return {
            "hor_input": self.ninja.hor_input,
            "jump_input": self.ninja.jump_input,
            "xpos": self.ninja.xpos,
            "ypos": self.ninja.ypos,
            "xpos_old": self.ninja.xpos_old,
            "ypos_old": self.ninja.ypos_old,
            "xspeed": self.ninja.xspeed,
            "yspeed": self.ninja.yspeed,
            "state": self.ninja.state,
            "airborn": self.ninja.airborn,
            "airborn_old": self.ninja.airborn_old,
            "walled": self.ninja.walled,
            "applied_gravity": self.ninja.applied_gravity,
            "applied_drag": self.ninja.applied_drag,
            "applied_friction": self.ninja.applied_friction,
            "jump_duration": self.ninja.jump_duration,
            "jump_buffer": self.ninja.jump_buffer,
            "floor_buffer": self.ninja.floor_buffer,
            "wall_buffer": self.ninja.wall_buffer,
            "launch_pad_buffer": self.ninja.launch_pad_buffer,
            "jump_input_old": self.ninja.jump_input_old,
            "floor_normalized_x": self.ninja.floor_normalized_x,
            "floor_normalized_y": self.ninja.floor_normalized_y,
            "ceiling_normalized_x": self.ninja.ceiling_normalized_x,
            "ceiling_normalized_y": self.ninja.ceiling_normalized_y,
        }

    def _restore_ninja_state(self, state: dict):
        """
        Restore ninja state from dictionary.

        Args:
            state: Dictionary containing state variables
        """
        self.ninja.hor_input = state["hor_input"]
        self.ninja.jump_input = state["jump_input"]
        self.ninja.xpos = state["xpos"]
        self.ninja.ypos = state["ypos"]
        self.ninja.xpos_old = state["xpos_old"]
        self.ninja.ypos_old = state["ypos_old"]
        self.ninja.xspeed = state["xspeed"]
        self.ninja.yspeed = state["yspeed"]
        self.ninja.state = state["state"]
        self.ninja.airborn = state["airborn"]
        self.ninja.airborn_old = state["airborn_old"]
        self.ninja.walled = state["walled"]
        self.ninja.applied_gravity = state["applied_gravity"]
        self.ninja.applied_drag = state["applied_drag"]
        self.ninja.applied_friction = state["applied_friction"]
        self.ninja.jump_duration = state["jump_duration"]
        self.ninja.jump_buffer = state["jump_buffer"]
        self.ninja.floor_buffer = state["floor_buffer"]
        self.ninja.wall_buffer = state["wall_buffer"]
        self.ninja.launch_pad_buffer = state["launch_pad_buffer"]
        self.ninja.jump_input_old = state["jump_input_old"]
        self.ninja.floor_normalized_x = state["floor_normalized_x"]
        self.ninja.floor_normalized_y = state["floor_normalized_y"]
        self.ninja.ceiling_normalized_x = state["ceiling_normalized_x"]
        self.ninja.ceiling_normalized_y = state["ceiling_normalized_y"]

    def create_ninja_state_dict(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        airborn: bool,
        jump_buffer_active: bool,
        floor_buffer_active: bool,
        wall_buffer_active: bool,
        state_category: int,
    ) -> dict:
        """
        Create ninja state dictionary from discretized parameters.

        This converts discretized state back to continuous state for simulation.

        Args:
            x: X position
            y: Y position
            vx: X velocity
            vy: Y velocity
            airborn: Airborne flag
            jump_buffer_active: Jump buffer active flag
            floor_buffer_active: Floor buffer active flag
            wall_buffer_active: Wall buffer active flag
            state_category: State category (0=ground, 1=air, 2=wall, 3=special)

        Returns:
            Dictionary with ninja state ready for simulation
        """
        # Map state category to actual state value
        if state_category == 0:  # Ground
            state = 1  # Running (most common ground state)
        elif state_category == 1:  # Air
            state = 4  # Falling (most common air state)
        elif state_category == 2:  # Wall
            state = 5  # Wall sliding
        else:  # Special
            state = 0  # Immobile

        # Convert buffer flags to buffer values
        # Active buffers should be in range [0, 5), use 2 as middle value
        jump_buffer = 2 if jump_buffer_active else -1
        floor_buffer = 2 if floor_buffer_active else -1
        wall_buffer = 2 if wall_buffer_active else -1

        # Create state dictionary with reasonable defaults
        return {
            "hor_input": 0,
            "jump_input": 0,
            "xpos": x,
            "ypos": y,
            "xpos_old": x,
            "ypos_old": y,
            "xspeed": vx,
            "yspeed": vy,
            "state": state,
            "airborn": airborn,
            "airborn_old": airborn,
            "walled": state_category == 2,  # Walled if wall state
            "applied_gravity": 0.06666666666666665,  # GRAVITY_FALL
            "applied_drag": 0.9933221725495059,  # DRAG_REGULAR
            "applied_friction": 0.9459290248857720,  # FRICTION_GROUND
            "jump_duration": 0,
            "jump_buffer": jump_buffer,
            "floor_buffer": floor_buffer,
            "wall_buffer": wall_buffer,
            "launch_pad_buffer": -1,
            "jump_input_old": 0,
            "floor_normalized_x": 0,
            "floor_normalized_y": -1,
            "ceiling_normalized_x": 0,
            "ceiling_normalized_y": 1,
        }
