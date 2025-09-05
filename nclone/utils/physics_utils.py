"""
Centralized physics utilities for N++ simulation.
Contains reusable physics calculations to avoid code duplication.
"""

import math
from typing import Tuple, List, Dict, Any
from enum import Enum

from ..constants.physics_constants import *
from ..constants.entity_types import EntityType


class BounceBlockState(Enum):
    """Bounce block state enumeration for data tracking."""
    NEUTRAL = 0      # Default state, minimal interaction
    COMPRESSING = 1  # Being compressed, moderate interaction
    COMPRESSED = 2   # Fully compressed, high interaction potential
    EXTENDING = 3    # Extending back, maximum interaction potential


def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return math.sqrt(dx * dx + dy * dy)


def calculate_trajectory_time(
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    initial_velocity: Tuple[float, float],
    gravity: float = GRAVITY_FALL
) -> float:
    """Calculate time of flight for a trajectory."""
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    vx, vy = initial_velocity
    
    if abs(vx) < MIN_HORIZONTAL_VELOCITY:
        # Vertical movement only
        if abs(vy) < MIN_HORIZONTAL_VELOCITY:
            return DEFAULT_MINIMUM_TIME
        
        # Use kinematic equation: dy = vy*t + 0.5*g*t^2
        discriminant = vy*vy + 2*gravity*dy
        if discriminant < 0:
            return DEFAULT_MINIMUM_TIME
        
        return (-vy + math.sqrt(discriminant)) / gravity
    
    # Horizontal movement dominates
    return abs(dx / vx)


def calculate_bounce_block_boost_multiplier(
    bounce_blocks: List[Dict[str, Any]],
    ninja_pos: Tuple[float, float]
) -> float:
    """
    Calculate boost multiplier based on nearby bounce blocks.
    
    Args:
        bounce_blocks: List of bounce block entities
        ninja_pos: Current ninja position
        
    Returns:
        Boost multiplier (1.0 = no boost, >1.0 = boost)
    """
    if not bounce_blocks:
        return BOUNCE_BLOCK_BOOST_NEUTRAL
    
    total_boost = 0.0
    active_blocks = 0
    
    for block in bounce_blocks:
        block_pos = (block.get('x', 0.0), block.get('y', 0.0))
        distance = calculate_distance(ninja_pos, block_pos)
        
        # Only consider blocks within interaction radius
        if distance <= BOUNCE_BLOCK_INTERACTION_RADIUS:
            compression = block.get('compression_amount', 0.0)
            state = block.get('bounce_state', BounceBlockState.NEUTRAL)
            
            # Calculate boost based on state
            if state == BounceBlockState.EXTENDING:
                boost = BOUNCE_BLOCK_BOOST_MIN + (BOUNCE_BLOCK_BOOST_MAX - BOUNCE_BLOCK_BOOST_MIN) * compression
            elif state == BounceBlockState.COMPRESSED:
                boost = BOUNCE_BLOCK_BOOST_MIN + (BOUNCE_BLOCK_BOOST_MAX - BOUNCE_BLOCK_BOOST_MIN) * compression * 0.8
            elif state == BounceBlockState.COMPRESSING:
                boost = BOUNCE_BLOCK_BOOST_MIN + (BOUNCE_BLOCK_BOOST_MAX - BOUNCE_BLOCK_BOOST_MIN) * compression * 0.5
            else:
                boost = BOUNCE_BLOCK_BOOST_NEUTRAL
            
            # Apply distance falloff
            distance_factor = max(0.0, 1.0 - distance / BOUNCE_BLOCK_INTERACTION_RADIUS)
            total_boost += boost * distance_factor
            active_blocks += 1
    
    if active_blocks == 0:
        return BOUNCE_BLOCK_BOOST_NEUTRAL
    
    # Average boost with diminishing returns for multiple blocks
    avg_boost = total_boost / active_blocks
    diminishing_factor = 1.0 / (1.0 + (active_blocks - 1) * 0.2)
    
    return min(avg_boost * diminishing_factor, BOUNCE_BLOCK_BOOST_MAX)


def calculate_compression_amount(
    current_pos: Tuple[float, float],
    original_pos: Tuple[float, float]
) -> float:
    """
    Calculate compression amount based on displacement from original position.
    
    Args:
        current_pos: Current position of bounce block
        original_pos: Original position of bounce block
        
    Returns:
        Compression amount (0.0 = no compression, 1.0 = maximum compression)
    """
    displacement = calculate_distance(current_pos, original_pos)
    
    # Normalize displacement to compression amount
    # Assuming maximum displacement of BOUNCE_BLOCK_SIZE for full compression
    compression = min(displacement / BOUNCE_BLOCK_SIZE, BOUNCE_BLOCK_MAX_COMPRESSION)
    
    return max(compression, 0.0)


def calculate_stored_energy(compression_amount: float) -> float:
    """
    Calculate stored energy in a compressed bounce block.
    
    Args:
        compression_amount: Amount of compression (0.0 to 1.0)
        
    Returns:
        Stored energy value
    """
    # E = 0.5 * k * x^2 (spring potential energy)
    displacement = compression_amount * BOUNCE_BLOCK_SIZE
    return 0.5 * BOUNCE_BLOCK_SPRING_CONSTANT * displacement * displacement


def determine_bounce_block_state(
    compression_amount: float,
    previous_compression: float,
    ninja_contact: bool
) -> BounceBlockState:
    """
    Determine bounce block state based on compression and contact.
    This is for data tracking only, not behavior modification.
    
    Args:
        compression_amount: Current compression amount
        previous_compression: Previous frame compression amount
        ninja_contact: Whether ninja is in contact with block
        
    Returns:
        Current bounce block state
    """
    if compression_amount < BOUNCE_BLOCK_MIN_COMPRESSION:
        return BounceBlockState.NEUTRAL
    
    # Determine if compressing or extending based on change
    compression_delta = compression_amount - previous_compression
    
    if ninja_contact and compression_delta > 0:
        return BounceBlockState.COMPRESSING
    elif compression_amount >= BOUNCE_BLOCK_COMPRESSION_THRESHOLD:
        return BounceBlockState.COMPRESSED
    elif compression_delta < 0:
        return BounceBlockState.EXTENDING
    else:
        return BounceBlockState.NEUTRAL


def calculate_clearance_directions(
    block_pos: Tuple[float, float],
    level_entities: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate clearance in each direction from a bounce block.
    
    Args:
        block_pos: Position of the bounce block
        level_entities: All entities in the level
        
    Returns:
        Dictionary with clearance distances in each direction
    """
    clearances = {
        'up': MAX_JUMP_DISTANCE,
        'down': MAX_FALL_DISTANCE,
        'left': MAX_JUMP_DISTANCE,
        'right': MAX_JUMP_DISTANCE
    }
    
    block_x, block_y = block_pos
    
    # Check for obstacles in each direction
    for entity in level_entities:
        # Handle both dictionary and object entities
        entity_type = entity.get('type') if isinstance(entity, dict) else getattr(entity, 'type', None)
        if entity_type == EntityType.BOUNCE_BLOCK:
            continue  # Skip other bounce blocks for clearance calculation
        
        entity_x = entity.get('x', 0.0) if isinstance(entity, dict) else getattr(entity, 'x', 0.0)
        entity_y = entity.get('y', 0.0) if isinstance(entity, dict) else getattr(entity, 'y', 0.0)
        
        # Calculate distances in each direction
        # Check horizontal clearance (left/right)
        if entity_x > block_x:  # Right
            distance = entity_x - block_x - BOUNCE_BLOCK_SIZE / 2
            clearances['right'] = min(clearances['right'], distance)
        elif entity_x < block_x:  # Left
            distance = block_x - entity_x - BOUNCE_BLOCK_SIZE / 2
            clearances['left'] = min(clearances['left'], distance)

        # Check vertical clearance (up/down)
        if entity_y > block_y:  # Down
            distance = entity_y - block_y - BOUNCE_BLOCK_SIZE / 2
            clearances['down'] = min(clearances['down'], distance)
        elif entity_y < block_y:  # Up
            distance = block_y - entity_y - BOUNCE_BLOCK_SIZE / 2
            clearances['up'] = min(clearances['up'], distance)
    
    # Ensure non-negative clearances
    for direction in clearances:
        clearances[direction] = max(0.0, clearances[direction])
    
    return clearances


def generate_trajectory_points(
    start_pos: Tuple[float, float],
    initial_velocity: Tuple[float, float],
    gravity: float,
    time_of_flight: float,
    num_points: int = None
) -> List[Tuple[float, float]]:
    """
    Generate trajectory points for visualization and analysis.
    
    Args:
        start_pos: Starting position
        initial_velocity: Initial velocity (vx, vy)
        gravity: Gravity acceleration
        time_of_flight: Total time of flight
        num_points: Number of points to generate (auto-calculated if None)
        
    Returns:
        List of trajectory points
    """
    if num_points is None:
        num_points = max(10, int(time_of_flight / TRAJECTORY_POINT_INTERVAL))
    
    points = []
    dt = time_of_flight / num_points
    
    x0, y0 = start_pos
    vx, vy = initial_velocity
    
    for i in range(num_points + 1):
        t = i * dt
        x = x0 + vx * t
        y = y0 + vy * t + 0.5 * gravity * t * t
        points.append((x, y))
    
    return points


def calculate_trajectory_with_bounce_blocks(
    start_pos: Tuple[float, float],
    velocity: Tuple[float, float],
    bounce_blocks: List[dict],
    time_steps: int = 60
) -> List[Tuple[float, float]]:
    """Calculate trajectory considering bounce block interactions."""
    trajectory = []
    pos_x, pos_y = start_pos
    vel_x, vel_y = velocity
    
    for step in range(time_steps):
        # Apply gravity
        vel_y += GRAVITY_FALL
        
        # Check for bounce block interactions
        for block in bounce_blocks:
            block_x = block.get('x', 0.0)
            block_y = block.get('y', 0.0)
            
            # Simple distance check for interaction
            distance = ((pos_x - block_x) ** 2 + (pos_y - block_y) ** 2) ** 0.5
            if distance < BOUNCE_BLOCK_INTERACTION_RADIUS:
                # Apply bounce block boost
                boost = calculate_bounce_block_boost_multiplier([block])
                vel_y *= boost
                vel_x *= boost * 0.8  # Horizontal boost is reduced
        
        # Update position
        pos_x += vel_x
        pos_y += vel_y
        
        trajectory.append((pos_x, pos_y))
        
        # Apply drag
        vel_x *= 0.99
        vel_y *= 0.99
    
    return trajectory


def calculate_enhanced_bounce_boost(
    ninja_pos: Tuple[float, float],
    ninja_velocity: Tuple[float, float],
    bounce_blocks: List[Dict[str, Any]],
    frame_count: int = 0,
    previous_frame_data: Dict[str, Any] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Enhanced bounce block boost calculation with repeated boost mechanics.
    
    This implements the special mechanic where ninja can repeatedly jump off
    compressing bounce blocks for accumulated boost.
    
    Args:
        ninja_pos: Current ninja position
        ninja_velocity: Current ninja velocity
        bounce_blocks: List of bounce block entities
        frame_count: Current frame count for timing
        previous_frame_data: Data from previous frame for repeated boost detection
        
    Returns:
        Tuple of (boost_multiplier, detailed_boost_info)
    """
    if not bounce_blocks:
        return BOUNCE_BLOCK_BOOST_NEUTRAL, {}
    
    boost_info = {
        'repeated_boost_available': False,
        'compression_direction': None,
        'force_accumulation': 0.0,
        'active_block_count': 0,
        'boost_chain_length': 0,
        'multi_block_penalty': 1.0
    }
    
    total_boost = 0.0
    active_blocks = 0
    total_force_x = 0.0
    total_force_y = 0.0
    
    # Track blocks that are in extending state for repeated boost
    extending_blocks = []
    
    for block in bounce_blocks:
        block_x = block.get('x', 0.0)
        block_y = block.get('y', 0.0)
        distance = calculate_distance(ninja_pos, (block_x, block_y))
        
        if distance <= BOUNCE_BLOCK_INTERACTION_RADIUS:
            compression = block.get('compression_amount', 0.0)
            state = block.get('bounce_state', BounceBlockState.NEUTRAL)
            previous_compression = block.get('previous_compression', 0.0)
            
            # Calculate base boost
            if state == BounceBlockState.EXTENDING:
                boost = BOUNCE_BLOCK_BOOST_MIN + (BOUNCE_BLOCK_BOOST_MAX - BOUNCE_BLOCK_BOOST_MIN) * compression
                extending_blocks.append(block)
                
                # Check for repeated boost conditions
                if (compression > previous_compression and 
                    abs(ninja_velocity[1]) > MIN_HORIZONTAL_VELOCITY):  # Ninja is moving
                    boost_info['repeated_boost_available'] = True
                    
            elif state == BounceBlockState.COMPRESSED:
                boost = BOUNCE_BLOCK_BOOST_MIN + (BOUNCE_BLOCK_BOOST_MAX - BOUNCE_BLOCK_BOOST_MIN) * compression * 0.8
            elif state == BounceBlockState.COMPRESSING:
                boost = BOUNCE_BLOCK_BOOST_MIN + (BOUNCE_BLOCK_BOOST_MAX - BOUNCE_BLOCK_BOOST_MIN) * compression * 0.6
            else:
                boost = BOUNCE_BLOCK_BOOST_NEUTRAL
            
            # Calculate force vector for multi-block interactions
            distance_factor = max(0.0, 1.0 - distance / BOUNCE_BLOCK_INTERACTION_RADIUS)
            force_magnitude = boost * distance_factor * compression
            
            if distance > 0:
                force_x = (ninja_pos[0] - block_x) / distance * force_magnitude
                force_y = (ninja_pos[1] - block_y) / distance * force_magnitude
                total_force_x += force_x
                total_force_y += force_y
            
            total_boost += boost * distance_factor
            active_blocks += 1
    
    if active_blocks == 0:
        return BOUNCE_BLOCK_BOOST_NEUTRAL, boost_info
    
    # Multi-block force accumulation penalty
    # When multiple blocks are compressed, they create opposing forces
    if active_blocks > 1:
        # Each additional block reduces effectiveness
        penalty_factor = 1.0 / (1.0 + 0.4 * (active_blocks - 1))
        boost_info['multi_block_penalty'] = penalty_factor
        total_boost *= penalty_factor
    
    # Calculate repeated boost multiplier
    repeated_multiplier = 1.0
    if boost_info['repeated_boost_available'] and extending_blocks:
        # Repeated boost effectiveness increases with consecutive frames
        if previous_frame_data and previous_frame_data.get('repeated_boost_active', False):
            chain_length = previous_frame_data.get('boost_chain_length', 0) + 1
            boost_info['boost_chain_length'] = chain_length
            # Each consecutive frame adds 10% boost, up to 100% maximum
            repeated_multiplier = 1.0 + min(1.0, chain_length * 0.1)
        else:
            boost_info['boost_chain_length'] = 1
            repeated_multiplier = 1.1  # 10% boost for first repeated jump
    
    # Final boost calculation
    avg_boost = total_boost / active_blocks
    final_boost = min(avg_boost * repeated_multiplier, BOUNCE_BLOCK_BOOST_MAX)
    
    # Store force accumulation info
    boost_info['force_accumulation'] = (total_force_x ** 2 + total_force_y ** 2) ** 0.5
    boost_info['active_block_count'] = active_blocks
    
    return final_boost, boost_info


def calculate_bounce_block_clearance_boost(
    block_pos: Tuple[float, float],
    clearance_data: Dict[str, float],
    ninja_velocity: Tuple[float, float]
) -> Tuple[float, str]:
    """
    Calculate boost direction and magnitude based on bounce block clearance.
    
    This implements the mechanic where ninja can boost in the direction
    opposite to where the bounce block has clearance.
    
    Args:
        block_pos: Position of the bounce block
        clearance_data: Clearance distances in each direction
        ninja_velocity: Current ninja velocity
        
    Returns:
        Tuple of (boost_multiplier, boost_direction)
    """
    # Find direction with maximum clearance
    max_clearance = 0.0
    best_direction = None
    
    for direction, clearance in clearance_data.items():
        if clearance > max_clearance and clearance >= BOUNCE_BLOCK_SIZE:  # At least 1 tile clearance
            max_clearance = clearance
            best_direction = direction
    
    if not best_direction or max_clearance < BOUNCE_BLOCK_SIZE:
        return BOUNCE_BLOCK_BOOST_NEUTRAL, 'none'
    
    # Calculate boost based on clearance amount
    # More clearance = more potential boost
    clearance_factor = min(1.0, max_clearance / (BOUNCE_BLOCK_SIZE * 3))  # Normalize to 3 tiles max
    boost_multiplier = BOUNCE_BLOCK_BOOST_MIN + (BOUNCE_BLOCK_BOOST_MAX - BOUNCE_BLOCK_BOOST_MIN) * clearance_factor
    
    # Boost direction is opposite to clearance direction
    boost_direction_map = {
        'up': 'down',
        'down': 'up', 
        'left': 'right',
        'right': 'left'
    }
    
    boost_direction = boost_direction_map.get(best_direction, 'up')
    
    return boost_multiplier, boost_direction