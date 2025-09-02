"""
Centralized collision detection utilities for N++ simulation.
Contains reusable collision detection algorithms to avoid code duplication.
"""

import math
from typing import Tuple, List, Dict, Any, Optional

from ..constants.physics_constants import *
from ..constants.entity_types import EntityType


def point_in_rectangle(
    point: Tuple[float, float],
    rect_pos: Tuple[float, float],
    rect_size: Tuple[float, float]
) -> bool:
    """
    Check if a point is inside a rectangle.
    
    Args:
        point: Point coordinates (x, y)
        rect_pos: Rectangle position (x, y) - top-left corner
        rect_size: Rectangle size (width, height)
        
    Returns:
        True if point is inside rectangle
    """
    px, py = point
    rx, ry = rect_pos
    rw, rh = rect_size
    
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def rectangles_overlap(
    rect1_pos: Tuple[float, float],
    rect1_size: Tuple[float, float],
    rect2_pos: Tuple[float, float],
    rect2_size: Tuple[float, float]
) -> bool:
    """
    Check if two rectangles overlap.
    
    Args:
        rect1_pos: First rectangle position (x, y)
        rect1_size: First rectangle size (width, height)
        rect2_pos: Second rectangle position (x, y)
        rect2_size: Second rectangle size (width, height)
        
    Returns:
        True if rectangles overlap
    """
    x1, y1 = rect1_pos
    w1, h1 = rect1_size
    x2, y2 = rect2_pos
    w2, h2 = rect2_size
    
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


def point_near_line_segment(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
    threshold: float
) -> bool:
    """
    Check if point is within threshold distance of line segment.
    
    Args:
        point: Point coordinates (x, y)
        line_start: Line segment start (x, y)
        line_end: Line segment end (x, y)
        threshold: Maximum distance threshold
        
    Returns:
        True if point is within threshold distance
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line start to end
    dx = x2 - x1
    dy = y2 - y1
    
    # Vector from line start to point
    fx = px - x1
    fy = py - y1
    
    # Handle degenerate case (zero-length line)
    if dx == 0 and dy == 0:
        distance = math.sqrt(fx*fx + fy*fy)
        return distance <= threshold
    
    # Project point onto line segment
    t = max(0, min(1, (fx*dx + fy*dy) / (dx*dx + dy*dy)))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance from point to closest point on line
    dist_x = px - closest_x
    dist_y = py - closest_y
    distance = math.sqrt(dist_x*dist_x + dist_y*dist_y)
    
    return distance <= threshold


def find_entities_in_radius(
    center_pos: Tuple[float, float],
    radius: float,
    entities: List[Dict[str, Any]],
    entity_type: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Find all entities within a given radius of a center point.
    
    Args:
        center_pos: Center position (x, y)
        radius: Search radius
        entities: List of entity dictionaries
        entity_type: Optional entity type filter
        
    Returns:
        List of entities within radius
    """
    cx, cy = center_pos
    found_entities = []
    
    for entity in entities:
        if entity_type is not None and entity.get('type') != entity_type:
            continue
        
        ex = entity.get('x', 0.0)
        ey = entity.get('y', 0.0)
        
        distance = math.sqrt((ex - cx)**2 + (ey - cy)**2)
        if distance <= radius:
            found_entities.append(entity)
    
    return found_entities


def find_bounce_blocks_near_trajectory(
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    entities: List[Dict[str, Any]],
    search_radius: float = BOUNCE_BLOCK_INTERACTION_RADIUS
) -> List[Dict[str, Any]]:
    """
    Find bounce blocks near a trajectory path.
    
    Args:
        start_pos: Trajectory start position
        end_pos: Trajectory end position
        entities: List of all entities
        search_radius: Search radius around trajectory
        
    Returns:
        List of bounce blocks near the trajectory
    """
    bounce_blocks = []
    
    for entity in entities:
        if entity.get('type') != EntityType.BOUNCE_BLOCK:
            continue
        
        block_pos = (entity.get('x', 0.0), entity.get('y', 0.0))
        
        if point_near_line_segment(block_pos, start_pos, end_pos, search_radius):
            bounce_blocks.append(entity)
    
    return bounce_blocks


def find_chainable_bounce_blocks(
    trajectory_blocks: List[Dict[str, Any]],
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float]
) -> List[Dict[str, Any]]:
    """
    Find bounce blocks that can be chained together along a trajectory.
    
    Args:
        trajectory_blocks: Bounce blocks along the trajectory
        start_pos: Trajectory start position
        end_pos: Trajectory end position
        
    Returns:
        List of chainable bounce blocks in order
    """
    if len(trajectory_blocks) < 2:
        return trajectory_blocks
    
    # Sort blocks by distance along trajectory
    def trajectory_distance(block):
        block_pos = (block.get('x', 0.0), block.get('y', 0.0))
        # Project block position onto trajectory line
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        if dx == 0 and dy == 0:
            return 0.0
        
        bx = block_pos[0] - start_pos[0]
        by = block_pos[1] - start_pos[1]
        t = max(0, min(1, (bx*dx + by*dy) / (dx*dx + dy*dy)))
        return t
    
    sorted_blocks = sorted(trajectory_blocks, key=trajectory_distance)
    
    # Filter blocks that are within chaining distance
    chainable = [sorted_blocks[0]]  # Always include first block
    
    for i in range(1, len(sorted_blocks)):
        prev_block = chainable[-1]
        curr_block = sorted_blocks[i]
        
        prev_pos = (prev_block.get('x', 0.0), prev_block.get('y', 0.0))
        curr_pos = (curr_block.get('x', 0.0), curr_block.get('y', 0.0))
        
        distance = math.sqrt(
            (curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2
        )
        
        if distance <= BOUNCE_BLOCK_CHAIN_DISTANCE:
            chainable.append(curr_block)
    
    return chainable


def check_ninja_bounce_block_contact(
    ninja_pos: Tuple[float, float],
    ninja_size: Tuple[float, float],
    bounce_block_pos: Tuple[float, float]
) -> bool:
    """
    Check if ninja is in contact with a bounce block.
    
    Args:
        ninja_pos: Ninja position (x, y)
        ninja_size: Ninja size (width, height)
        bounce_block_pos: Bounce block position (x, y)
        
    Returns:
        True if ninja is in contact with bounce block
    """
    bounce_block_size = (BOUNCE_BLOCK_SIZE, BOUNCE_BLOCK_SIZE)
    
    return rectangles_overlap(
        ninja_pos, ninja_size,
        bounce_block_pos, bounce_block_size
    )


def calculate_overlap_area(
    rect1_pos: Tuple[float, float],
    rect1_size: Tuple[float, float],
    rect2_pos: Tuple[float, float],
    rect2_size: Tuple[float, float]
) -> float:
    """
    Calculate the area of overlap between two rectangles.
    
    Args:
        rect1_pos: First rectangle position (x, y)
        rect1_size: First rectangle size (width, height)
        rect2_pos: Second rectangle position (x, y)
        rect2_size: Second rectangle size (width, height)
        
    Returns:
        Overlap area (0.0 if no overlap)
    """
    x1, y1 = rect1_pos
    w1, h1 = rect1_size
    x2, y2 = rect2_pos
    w2, h2 = rect2_size
    
    # Calculate overlap bounds
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)
    
    # Check if there's actual overlap
    if left >= right or top >= bottom:
        return 0.0
    
    return (right - left) * (bottom - top)


def find_platform_bounce_blocks(
    entities: List[Dict[str, Any]],
    ninja_pos: Tuple[float, float],
    search_radius: float = BOUNCE_BLOCK_INTERACTION_RADIUS * 2
) -> List[Dict[str, Any]]:
    """
    Find bounce blocks that can serve as platforms for the ninja.
    
    Args:
        entities: List of all entities
        ninja_pos: Current ninja position
        search_radius: Search radius for platform detection
        
    Returns:
        List of bounce blocks that can serve as platforms
    """
    bounce_blocks = find_entities_in_radius(
        ninja_pos, search_radius, entities, EntityType.BOUNCE_BLOCK
    )
    
    platforms = []
    _, ninja_y = ninja_pos
    
    for block in bounce_blocks:
        block_y = block.get('y', 0.0)
        
        # Check if block is below ninja (can serve as platform)
        if block_y > ninja_y:
            platforms.append(block)
    
    return platforms