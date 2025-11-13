"""
Fast-forward physics simulator for terminal velocity death prediction.

This module provides lightweight physics simulation that reuses ninja physics
methods to accurately predict terminal impact deaths (collision with floor/ceiling
at lethal speeds) over multiple frames.
"""

from typing import Dict

from .constants import (
    MAX_SURVIVABLE_IMPACT,
    TERMINAL_IMPACT_SIMULATION_FRAMES,
)


class TerminalVelocitySimulator:
    """
    Lightweight physics simulator for terminal impact prediction.

    Reuses core ninja physics (integrate(), think(), collide_vs_tiles(),
    collide_vs_objects(), post_collision()) to accurately simulate movement
    and detect terminal impact deaths.
    """

    def __init__(self, sim):
        """
        Initialize simulator with simulation reference.

        Args:
            sim: Simulation object containing ninja and entities
        """
        self.sim = sim
        self.ninja = sim.ninja

    def simulate_for_terminal_impact(
        self,
        action: int,
        max_frames: int = TERMINAL_IMPACT_SIMULATION_FRAMES,
    ) -> bool:
        """
        Simulate action and check for terminal impact death over multiple frames.

        Uses the exact impact calculation from ninja.py lines 383-389 (floor)
        and 408-421 (ceiling) to detect lethal impacts.

        Args:
            action: Action to simulate (0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP, 4=JUMP+LEFT, 5=JUMP+RIGHT)
            max_frames: Maximum number of frames to simulate forward

        Returns:
            True if terminal impact death detected, False otherwise
        """
        # Validate ninja state before simulation
        # Skip if ninja is already dead (state 6/7) or celebrating (state 8)
        if self.ninja.state in (6, 7, 8):
            return False  # Invalid state for simulation
        
        # Save original ninja state
        saved_state = self._save_ninja_state()

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

        # Simulate frames until impact or timeout
        impact_detected = False
        for frame_idx in range(max_frames):
            # Set inputs for this frame
            self.ninja.hor_input = hor_input
            self.ninja.jump_input = jump_input

            # Run think() to process inputs and update state
            self.ninja.think()

            # Run integrate() to update position based on velocity
            self.ninja.integrate()

            # Run collision detection to update floor/ceiling normals
            # Skip entity collision for terminal velocity (only tiles affect impact)
            self.ninja.pre_collision()
            self.ninja.collide_vs_tiles()
            # self.ninja.collide_vs_objects()  # SKIPPED - terminal impact only depends on tiles
            self.ninja.post_collision(skip_entities=True)

            # Check for terminal impact using exact ninja.py calculation
            if self._check_terminal_impact():
                impact_detected = True
                break

            # Early exit if ninja already dead (shouldn't happen, but safe)
            if self.ninja.has_died():
                impact_detected = True
                break

        # Restore original state
        self._restore_ninja_state(saved_state)

        return impact_detected

    def _check_terminal_impact(self) -> bool:
        """
        Check if ninja will die from terminal impact this frame.

        Uses exact impact calculation from ninja.py:
        - Floor impact (lines 383-389)
        - Ceiling impact (lines 408-421)

        Returns:
            True if terminal impact death will occur, False otherwise
        """
        ninja = self.ninja

        # Check floor impact (from ninja.py lines 383-389)
        if ninja.floor_count > 0 and ninja.airborn_old:
            # Calculate impact velocity using exact formula
            impact_vel = -(
                ninja.floor_normalized_x * ninja.xspeed_old
                + ninja.floor_normalized_y * ninja.yspeed_old
            )

            # Check against survivable threshold (varies with surface normal)
            # Exact formula: impact_vel > MAX_SURVIVABLE_IMPACT - 4/3 * abs(floor_normalized_y)
            threshold = MAX_SURVIVABLE_IMPACT - (4 / 3) * abs(ninja.floor_normalized_y)

            if impact_vel > threshold:
                return True  # Deadly floor impact

        # Check ceiling impact (from ninja.py lines 408-421)
        if ninja.ceiling_count > 0:
            # Calculate impact velocity using exact formula
            impact_vel = -(
                ninja.ceiling_normalized_x * ninja.xspeed_old
                + ninja.ceiling_normalized_y * ninja.yspeed_old
            )

            # Check against survivable threshold (varies with surface normal)
            # Exact formula: impact_vel > MAX_SURVIVABLE_IMPACT - 4/3 * abs(ceiling_normalized_y)
            threshold = MAX_SURVIVABLE_IMPACT - (4 / 3) * abs(
                ninja.ceiling_normalized_y
            )

            if impact_vel > threshold:
                return True  # Deadly ceiling impact

        return False  # No terminal impact

    def _save_ninja_state(self) -> Dict:
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
            "xspeed_old": self.ninja.xspeed_old,
            "yspeed_old": self.ninja.yspeed_old,
            "state": self.ninja.state,
            "airborn": self.ninja.airborn,
            "airborn_old": self.ninja.airborn_old,
            "walled": self.ninja.walled,
            "wall_normal": self.ninja.wall_normal,
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
            "floor_count": self.ninja.floor_count,
            "ceiling_count": self.ninja.ceiling_count,
            "last_wall_jump_frame": self.ninja.last_wall_jump_frame,
            "second_last_wall_jump_frame": self.ninja.second_last_wall_jump_frame,
            "third_last_wall_jump_frame": self.ninja.third_last_wall_jump_frame,
        }

    def _restore_ninja_state(self, state: Dict):
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
        self.ninja.xspeed_old = state["xspeed_old"]
        self.ninja.yspeed_old = state["yspeed_old"]
        self.ninja.state = state["state"]
        self.ninja.airborn = state["airborn"]
        self.ninja.airborn_old = state["airborn_old"]
        self.ninja.walled = state["walled"]
        self.ninja.wall_normal = state["wall_normal"]
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
        self.ninja.floor_count = state["floor_count"]
        self.ninja.ceiling_count = state["ceiling_count"]
        self.ninja.last_wall_jump_frame = state["last_wall_jump_frame"]
        self.ninja.second_last_wall_jump_frame = state["second_last_wall_jump_frame"]
        self.ninja.third_last_wall_jump_frame = state["third_last_wall_jump_frame"]
