"""
Physics trajectory validation for N++ ninja movements.

This module provides comprehensive physics validation for all movement types,
ensuring that proposed paths respect N++ physics constraints and are actually
executable by the ninja.
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

from nclone.constants.physics_constants import (
    MAX_HOR_SPEED,
    JUMP_FLOOR_Y,
    JUMP_WALL_REGULAR_X_MULTIPLIER,
    JUMP_WALL_REGULAR_Y,
    JUMP_WALL_SLIDE_X_MULTIPLIER,
    JUMP_WALL_SLIDE_Y,
    GRAVITY_FALL,
    GRAVITY_JUMP,
    MAX_JUMP_DURATION,
    MAX_SURVIVABLE_IMPACT,
    NINJA_RADIUS,
    TILE_PIXEL_SIZE,
)


@dataclass
class TrajectoryResult:
    """Result of trajectory validation."""
    is_valid: bool
    required_velocity: Tuple[float, float]
    flight_time: float
    landing_velocity: Tuple[float, float]
    max_height: float
    energy_cost: float
    risk_factor: float
    failure_reason: Optional[str] = None


class PhysicsTrajectoryValidator:
    """Validates movement trajectories using N++ physics."""
    
    def __init__(self):
        self.frame_rate = 60.0  # N++ runs at 60 FPS
        
    def validate_jump_trajectory(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        ninja_state: Optional[Dict[str, Any]] = None,
        level_data: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryResult:
        """
        Validate a jump trajectory using N++ physics.
        
        Calculates required initial velocity, simulates flight path,
        and validates landing conditions.
        """
        x0, y0 = start_pos
        x1, y1 = end_pos
        
        dx = x1 - x0
        dy = y1 - y0
        
        # Determine jump type based on ninja state and geometry
        jump_type = self._determine_jump_type(start_pos, end_pos, ninja_state, level_data)
        
        # Get initial velocity based on jump type
        initial_velocity = self._calculate_initial_velocity(jump_type, dx, dy, ninja_state)
        
        if initial_velocity is None:
            return TrajectoryResult(
                is_valid=False,
                required_velocity=(0, 0),
                flight_time=0,
                landing_velocity=(0, 0),
                max_height=0,
                energy_cost=0,
                risk_factor=1.0,
                failure_reason="Cannot determine initial velocity"
            )
        
        vx0, vy0 = initial_velocity
        
        # Validate velocity is within ninja capabilities
        if abs(vx0) > MAX_HOR_SPEED:
            return TrajectoryResult(
                is_valid=False,
                required_velocity=initial_velocity,
                flight_time=0,
                landing_velocity=(0, 0),
                max_height=0,
                energy_cost=0,
                risk_factor=1.0,
                failure_reason=f"Required horizontal velocity {abs(vx0):.1f} exceeds max {MAX_HOR_SPEED}"
            )
        
        # Simulate trajectory
        trajectory_data = self._simulate_trajectory(
            start_pos, initial_velocity, end_pos, level_data
        )
        
        if not trajectory_data["reaches_target"]:
            return TrajectoryResult(
                is_valid=False,
                required_velocity=initial_velocity,
                flight_time=trajectory_data["flight_time"],
                landing_velocity=trajectory_data["landing_velocity"],
                max_height=trajectory_data["max_height"],
                energy_cost=0,
                risk_factor=1.0,
                failure_reason="Trajectory does not reach target"
            )
        
        # Check landing impact
        landing_vx, landing_vy = trajectory_data["landing_velocity"]
        impact_speed = math.sqrt(landing_vx**2 + landing_vy**2)
        
        if impact_speed > MAX_SURVIVABLE_IMPACT:
            return TrajectoryResult(
                is_valid=False,
                required_velocity=initial_velocity,
                flight_time=trajectory_data["flight_time"],
                landing_velocity=trajectory_data["landing_velocity"],
                max_height=trajectory_data["max_height"],
                energy_cost=0,
                risk_factor=1.0,
                failure_reason=f"Landing impact {impact_speed:.1f} exceeds survivable limit {MAX_SURVIVABLE_IMPACT}"
            )
        
        # Check flight time
        if trajectory_data["flight_time"] > MAX_JUMP_DURATION:
            return TrajectoryResult(
                is_valid=False,
                required_velocity=initial_velocity,
                flight_time=trajectory_data["flight_time"],
                landing_velocity=trajectory_data["landing_velocity"],
                max_height=trajectory_data["max_height"],
                energy_cost=0,
                risk_factor=1.0,
                failure_reason=f"Flight time {trajectory_data['flight_time']:.1f} exceeds max duration {MAX_JUMP_DURATION}"
            )
        
        # Calculate energy cost and risk
        energy_cost = self._calculate_energy_cost(jump_type, dx, dy, trajectory_data["flight_time"])
        risk_factor = self._calculate_risk_factor(trajectory_data, level_data)
        
        return TrajectoryResult(
            is_valid=True,
            required_velocity=initial_velocity,
            flight_time=trajectory_data["flight_time"],
            landing_velocity=trajectory_data["landing_velocity"],
            max_height=trajectory_data["max_height"],
            energy_cost=energy_cost,
            risk_factor=risk_factor
        )
    
    def validate_walk_movement(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        level_data: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryResult:
        """
        Validate a walking movement.
        
        Checks that surfaces are connected and walkable.
        """
        x0, y0 = start_pos
        x1, y1 = end_pos
        
        dx = x1 - x0
        dy = y1 - y0
        distance = math.sqrt(dx**2 + dy**2)
        
        # Check if path is walkable
        if level_data and not self._is_path_walkable(start_pos, end_pos, level_data):
            return TrajectoryResult(
                is_valid=False,
                required_velocity=(0, 0),
                flight_time=0,
                landing_velocity=(0, 0),
                max_height=0,
                energy_cost=0,
                risk_factor=1.0,
                failure_reason="Path is not walkable - obstacles or gaps present"
            )
        
        # Calculate walking time and energy
        walking_time = distance / MAX_HOR_SPEED  # Simplified - actual walking involves acceleration
        energy_cost = distance * 0.1  # Walking is low energy
        
        return TrajectoryResult(
            is_valid=True,
            required_velocity=(dx / walking_time if walking_time > 0 else 0, 0),
            flight_time=walking_time,
            landing_velocity=(0, 0),
            max_height=max(y0, y1),
            energy_cost=energy_cost,
            risk_factor=0.1  # Walking is low risk
        )
    
    def validate_fall_movement(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        level_data: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryResult:
        """
        Validate a falling movement.
        
        Simulates gravity-based descent.
        """
        x0, y0 = start_pos
        x1, y1 = end_pos
        
        dx = x1 - x0
        dy = y1 - y0
        
        if dy <= 0:
            return TrajectoryResult(
                is_valid=False,
                required_velocity=(0, 0),
                flight_time=0,
                landing_velocity=(0, 0),
                max_height=0,
                energy_cost=0,
                risk_factor=1.0,
                failure_reason="Fall movement requires downward motion"
            )
        
        # Calculate fall trajectory
        # For pure fall: y = y0 + v0*t + 0.5*g*t^2
        # If starting with zero vertical velocity: t = sqrt(2*dy/g)
        fall_time = math.sqrt(2 * dy / GRAVITY_FALL)
        
        # Horizontal velocity needed to reach target
        required_vx = dx / fall_time if fall_time > 0 else 0
        
        if abs(required_vx) > MAX_HOR_SPEED:
            return TrajectoryResult(
                is_valid=False,
                required_velocity=(required_vx, 0),
                flight_time=fall_time,
                landing_velocity=(required_vx, GRAVITY_FALL * fall_time),
                max_height=y0,
                energy_cost=0,
                risk_factor=1.0,
                failure_reason=f"Required horizontal velocity {abs(required_vx):.1f} exceeds max {MAX_HOR_SPEED}"
            )
        
        # Landing velocity
        landing_vy = GRAVITY_FALL * fall_time
        landing_velocity = (required_vx, landing_vy)
        impact_speed = math.sqrt(required_vx**2 + landing_vy**2)
        
        if impact_speed > MAX_SURVIVABLE_IMPACT:
            return TrajectoryResult(
                is_valid=False,
                required_velocity=(required_vx, 0),
                flight_time=fall_time,
                landing_velocity=landing_velocity,
                max_height=y0,
                energy_cost=0,
                risk_factor=1.0,
                failure_reason=f"Landing impact {impact_speed:.1f} exceeds survivable limit {MAX_SURVIVABLE_IMPACT}"
            )
        
        # Check for obstacles in fall path
        if level_data and not self._is_fall_path_clear(start_pos, end_pos, level_data):
            return TrajectoryResult(
                is_valid=False,
                required_velocity=(required_vx, 0),
                flight_time=fall_time,
                landing_velocity=landing_velocity,
                max_height=y0,
                energy_cost=0,
                risk_factor=1.0,
                failure_reason="Fall path blocked by obstacles"
            )
        
        risk_factor = min(impact_speed / MAX_SURVIVABLE_IMPACT, 1.0)
        
        return TrajectoryResult(
            is_valid=True,
            required_velocity=(required_vx, 0),
            flight_time=fall_time,
            landing_velocity=landing_velocity,
            max_height=y0,
            energy_cost=0,  # Falling is free
            risk_factor=risk_factor
        )
    
    def _determine_jump_type(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        ninja_state: Optional[Dict[str, Any]],
        level_data: Optional[Dict[str, Any]],
    ) -> str:
        """Determine the type of jump required."""
        if ninja_state and ninja_state.get("wall_contact"):
            if ninja_state.get("state") == 5:  # Wall sliding
                return "wall_slide_jump"
            else:
                return "wall_jump"
        else:
            return "floor_jump"
    
    def _calculate_initial_velocity(
        self,
        jump_type: str,
        dx: float,
        dy: float,
        ninja_state: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[float, float]]:
        """Calculate required initial velocity for jump type."""
        if jump_type == "floor_jump":
            # Floor jump: fixed vertical velocity, calculate horizontal
            vy0 = JUMP_FLOOR_Y  # Negative (upward)
            
            # Calculate time to reach target height
            # y = y0 + vy0*t + 0.5*g*t^2
            # Solve for t when y = y0 + dy
            discriminant = vy0**2 - 2 * GRAVITY_JUMP * dy
            if discriminant < 0:
                return None  # Cannot reach target height
            
            t = (-vy0 + math.sqrt(discriminant)) / GRAVITY_JUMP
            if t <= 0:
                return None
            
            vx0 = dx / t
            return (vx0, vy0)
        
        elif jump_type == "wall_jump":
            # Wall jump: use wall jump constants
            wall_normal = ninja_state.get("wall_normal", 1) if ninja_state else 1
            vx0 = JUMP_WALL_REGULAR_X_MULTIPLIER * wall_normal
            vy0 = JUMP_WALL_REGULAR_Y
            return (vx0, vy0)
        
        elif jump_type == "wall_slide_jump":
            # Wall slide jump: different constants
            wall_normal = ninja_state.get("wall_normal", 1) if ninja_state else 1
            vx0 = JUMP_WALL_SLIDE_X_MULTIPLIER * wall_normal
            vy0 = JUMP_WALL_SLIDE_Y
            return (vx0, vy0)
        
        return None
    
    def _simulate_trajectory(
        self,
        start_pos: Tuple[float, float],
        initial_velocity: Tuple[float, float],
        target_pos: Tuple[float, float],
        level_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Simulate complete trajectory with collision detection."""
        x0, y0 = start_pos
        vx0, vy0 = initial_velocity
        target_x, target_y = target_pos
        
        dt = 1.0 / self.frame_rate  # Frame time
        max_frames = int(MAX_JUMP_DURATION)
        
        x, y = x0, y0
        vx, vy = vx0, vy0
        max_height = y0
        
        for frame in range(max_frames):
            # Update position
            x += vx * dt
            y += vy * dt
            
            # Update velocity (gravity)
            vy += GRAVITY_JUMP * dt
            
            # Track maximum height
            max_height = min(max_height, y)  # Y increases downward
            
            # Check if reached target (within tolerance)
            if (abs(x - target_x) < NINJA_RADIUS and 
                abs(y - target_y) < NINJA_RADIUS):
                return {
                    "reaches_target": True,
                    "flight_time": frame * dt,
                    "landing_velocity": (vx, vy),
                    "max_height": max_height,
                    "collision": False
                }
            
            # Check for collision with level geometry
            if level_data and self._check_collision(x, y, level_data):
                return {
                    "reaches_target": False,
                    "flight_time": frame * dt,
                    "landing_velocity": (vx, vy),
                    "max_height": max_height,
                    "collision": True
                }
        
        # Trajectory timed out
        return {
            "reaches_target": False,
            "flight_time": max_frames * dt,
            "landing_velocity": (vx, vy),
            "max_height": max_height,
            "collision": False
        }
    
    def _is_path_walkable(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        level_data: Dict[str, Any],
    ) -> bool:
        """Check if a path can be walked (no gaps or obstacles)."""
        # Sample points along the path
        x0, y0 = start_pos
        x1, y1 = end_pos
        
        steps = max(1, int(abs(x1 - x0) // (TILE_PIXEL_SIZE // 2)))
        
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = x0 + t * (x1 - x0)
            y = max(y0, y1) + NINJA_RADIUS  # Check ground level
            
            if not self._has_ground_support(x, y, level_data):
                return False
        
        return True
    
    def _is_fall_path_clear(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        level_data: Dict[str, Any],
    ) -> bool:
        """Check if fall path is clear of obstacles."""
        # Simple implementation - check for obstacles in fall trajectory
        x0, y0 = start_pos
        x1, y1 = end_pos
        
        # Sample points along fall path
        steps = max(1, int(abs(y1 - y0) // (TILE_PIXEL_SIZE // 2)))
        
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            
            if self._check_collision(x, y, level_data):
                return False
        
        return True
    
    def _check_collision(self, x: float, y: float, level_data: Dict[str, Any]) -> bool:
        """Check if position collides with level geometry."""
        if not hasattr(level_data, 'tiles'):
            return False
        
        # Convert to tile coordinates
        tile_x = int(x // TILE_PIXEL_SIZE)
        tile_y = int(y // TILE_PIXEL_SIZE)
        
        # Check bounds
        tiles = level_data.tiles
        if 0 <= tile_x < tiles.shape[1] and 0 <= tile_y < tiles.shape[0]:
            tile_type = tiles[tile_y, tile_x]
            # Simplified collision - tile type 0 is empty, others may be solid
            return tile_type != 0
        
        return True  # Out of bounds is collision
    
    def _has_ground_support(self, x: float, y: float, level_data: Dict[str, Any]) -> bool:
        """Check if position has ground support for walking."""
        # Check tile below position
        ground_y = y + NINJA_RADIUS
        tile_x = int(x // TILE_PIXEL_SIZE)
        tile_y = int(ground_y // TILE_PIXEL_SIZE)
        
        if not hasattr(level_data, 'tiles'):
            return True  # Assume walkable if no tile data
        
        tiles = level_data.tiles
        if 0 <= tile_x < tiles.shape[1] and 0 <= tile_y < tiles.shape[0]:
            tile_type = tiles[tile_y, tile_x]
            # Simplified - non-zero tiles provide ground support
            return tile_type != 0
        
        return False
    
    def _calculate_energy_cost(
        self, jump_type: str, dx: float, dy: float, flight_time: float
    ) -> float:
        """Calculate energy cost for movement."""
        base_cost = 10.0  # Base jump energy
        distance_factor = math.sqrt(dx**2 + dy**2) / 100.0
        time_factor = flight_time / 10.0
        
        if jump_type == "wall_jump":
            base_cost *= 1.2  # Wall jumps are more demanding
        elif jump_type == "wall_slide_jump":
            base_cost *= 1.5  # Wall slide jumps are most demanding
        
        return base_cost + distance_factor + time_factor
    
    def _calculate_risk_factor(
        self, trajectory_data: Dict[str, Any], level_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate risk factor for movement."""
        risk = 0.0
        
        # Landing impact risk
        landing_vx, landing_vy = trajectory_data["landing_velocity"]
        impact_speed = math.sqrt(landing_vx**2 + landing_vy**2)
        risk += impact_speed / MAX_SURVIVABLE_IMPACT * 0.5
        
        # Flight time risk (longer flights are riskier)
        risk += trajectory_data["flight_time"] / MAX_JUMP_DURATION * 0.3
        
        # Height risk (higher jumps are riskier)
        height_diff = abs(trajectory_data["max_height"])
        risk += height_diff / 200.0 * 0.2  # Normalize to reasonable height
        
        return min(risk, 1.0)