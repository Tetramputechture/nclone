"""Physics step wrapper for stateless value iteration.

This module provides PhysicsStepWrapper, which wraps the existing
Ninja and Simulator physics code to provide a stateless interface
suitable for value iteration. The wrapper uses save_state() and
restore_state() to enable querying physics outcomes from arbitrary
states without side effects.

Key design principles:
- NO physics code duplication - reuses ninja.py entirely
- Exact behavior match - uses identical collision, gravity, etc.
- Easy maintenance - physics fixes automatically propagate
"""

from typing import Tuple, Optional
from ..nsim import Simulator
from ..constants.physics_constants import (
    GRAVITY_FALL,
    GRAVITY_JUMP,
    MAX_JUMP_DURATION,
)

import logging

logger = logging.getLogger(__name__)


class PhysicsStepWrapper:
    """Stateless physics step interface for value iteration.
    
    This class wraps the existing Simulator and Ninja physics code to
    provide a stateless query interface. Given any state and action,
    it returns the resulting next state without modifying the actual
    simulator state.
    
    Uses Ninja.save_state() / restore_state() and Simulator.tick_physics_only()
    to execute physics from arbitrary states without side effects.
    
    IMPORTANT: By default, simulates multiple physics frames per "step" to ensure
    meaningful state transitions. With grid resolution of 12px and max speed ~3px/frame,
    we need ~4 frames for the ninja to cross a cell boundary.
    
    Attributes:
        sim: Reference to the Simulator instance
        frames_per_step: Number of physics frames to simulate per step (default 5)
    """
    
    def __init__(self, simulator: Simulator, frames_per_step: int = 5):
        """Initialize physics step wrapper.
        
        Args:
            simulator: Simulator instance with loaded level
            frames_per_step: Number of physics frames per step (default 5).
                            With 12px grid and ~3px/frame max speed, need ~4 frames
                            to cross a cell boundary. 5 provides some margin.
        """
        self.sim = simulator
        self.frames_per_step = frames_per_step
        
        # Validate simulator has a ninja
        if self.sim.ninja is None:
            raise ValueError("Simulator must have a ninja (call sim.reset() first)")
    
    def step(self, state: Tuple[float, float, float, float, int], 
             action: Tuple[int, int]) -> Tuple[Tuple[float, float, float, float, int], bool]:
        """Execute multiple physics frames from given state with constant action.
        
        This method simulates `frames_per_step` physics frames to ensure meaningful
        state transitions. With grid resolution of 12px and max speed ~3px/frame,
        we need multiple frames for the ninja to cross cell boundaries.
        
        This method:
        1. Saves current ninja state
        2. Injects requested state into ninja
        3. Executes physics for `frames_per_step` frames
        4. Extracts result state
        5. Restores original ninja state
        
        Args:
            state: (x, y, vx, vy, jump_duration) in pixels/frames
                x, y: Position in pixels
                vx, vy: Velocity in pixels/frame
                jump_duration: Frames since jump started (0 if not jumping)
            action: (h_input, jump_input)
                h_input: Horizontal input in {-1, 0, 1}
                jump_input: Jump input in {0, 1}
        
        Returns:
            Tuple of (next_state, is_death):
                next_state: Same format as input state
                is_death: True if ninja died during any frame, False otherwise
        """
        # Save current state
        snapshot = self.sim.ninja.save_state()
        
        # Inject requested state into ninja
        x, y, vx, vy, jump_dur = state
        self._inject_state(x, y, vx, vy, jump_dur)
        
        # Execute multiple physics frames with same action
        h_input, jump_input = action
        is_death = False
        
        try:
            for frame in range(self.frames_per_step):
                self.sim.tick_physics_only(h_input, jump_input)
                
                # Check for death after each frame
                if self.sim.ninja.state in (6, 7):
                    is_death = True
                    break
        except Exception as e:
            # If physics fails, restore and re-raise
            self.sim.ninja.restore_state(snapshot)
            raise RuntimeError(f"Physics step failed: {e}") from e
        
        # Extract result
        next_state = (
            self.sim.ninja.xpos,
            self.sim.ninja.ypos,
            self.sim.ninja.xspeed,
            self.sim.ninja.yspeed,
            self.sim.ninja.jump_duration,
        )
        
        # Restore original state
        self.sim.ninja.restore_state(snapshot)
        
        return next_state, is_death
    
    def _inject_state(self, x: float, y: float, vx: float, vy: float, 
                      jump_dur: int):
        """Inject state into ninja for physics query.
        
        This method sets the ninja's position, velocity, and jump state
        to match the requested query state. It also sets derived state
        (airborn, applied_gravity, etc.) to be consistent with the
        core state.
        
        IMPORTANT: We default to airborn=True because we cannot know from
        the 5D state alone whether the ninja is on the ground. The physics
        tick will correctly determine ground contact via collide_vs_tiles().
        Setting airborn=False incorrectly would skip airborne physics logic
        and could cause wrong behavior at jump apex or during horizontal flight.
        
        Args:
            x, y: Position in pixels
            vx, vy: Velocity in pixels/frame
            jump_dur: Frames since jump started (0 if not jumping)
        """
        ninja = self.sim.ninja
        
        # Set core state
        ninja.xpos = x
        ninja.ypos = y
        ninja.xspeed = vx
        ninja.yspeed = vy
        ninja.jump_duration = jump_dur
        
        # Set previous position/velocity (affects collision detection)
        # For query purposes, assume previous state was slightly behind current
        ninja.xpos_old = x - vx
        ninja.ypos_old = y - vy
        ninja.xspeed_old = vx
        ninja.yspeed_old = vy
        
        # Set derived state based on jump_duration
        # States: 0=Immobile, 1=Running, 2=Ground sliding, 3=Jumping, 4=Falling, 5=Wall sliding
        if jump_dur > 0 and jump_dur < MAX_JUMP_DURATION:
            # Currently in active jump (reduced gravity)
            ninja.state = 3  # Jumping state
            ninja.applied_gravity = GRAVITY_JUMP
            ninja.airborn = True
        else:
            # Not actively jumping - use Falling state as safe default
            # The physics tick will detect ground contact and adjust state
            # We use state=4 (Falling) because:
            # - It's the correct state if ninja is airborne
            # - If ninja is actually grounded, collide_vs_tiles() will set
            #   floor_count > 0 and think() will transition to correct state
            ninja.state = 4  # Falling state (safe default)
            ninja.applied_gravity = GRAVITY_FALL
            ninja.airborn = True  # Let physics determine actual ground contact
        
        # Reset collision state (will be recomputed by collide_vs_tiles)
        ninja.floor_count = 0
        ninja.wall_count = 0
        ninja.ceiling_count = 0
        ninja.walled = False
        ninja.wall_normal = 0
        
        # Reset buffers (not relevant for stateless queries)
        ninja.jump_buffer = -1
        ninja.floor_buffer = -1
        ninja.wall_buffer = -1
        ninja.launch_pad_buffer = -1
        
        # Reset crush state
        ninja.is_crushable = False
        ninja.x_crush = 0
        ninja.y_crush = 0
        ninja.crush_len = 0
        
        # Reset death tracking
        ninja.terminal_impact = False
        ninja.death_cause = None
    
    def is_state_valid(self, state: Tuple[float, float, float, float, int]) -> bool:
        """Check if a state is within valid bounds.
        
        This method checks if position and velocity are within reasonable
        bounds for the level. Used to filter out obviously invalid states
        during value iteration.
        
        Args:
            state: (x, y, vx, vy, jump_duration)
        
        Returns:
            True if state is within valid bounds, False otherwise
        """
        x, y, vx, vy, jump_dur = state
        
        # Position bounds (level dimensions)
        if x < 0 or x > 1056 or y < 0 or y > 600:
            return False
        
        # Velocity bounds (physics limits)
        from ..constants.physics_constants import MAX_HOR_SPEED, TERMINAL_VELOCITY
        if abs(vx) > MAX_HOR_SPEED * 1.5 or abs(vy) > TERMINAL_VELOCITY * 1.5:
            return False
        
        # Jump duration bounds
        if jump_dur < 0 or jump_dur > MAX_JUMP_DURATION:
            return False
        
        return True
