"""
Level data structure for N++ graph processing.

This module provides a unified data structure for representing level information
including tiles and entities, used across both nclone and npp-rl projects.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class LevelData:
    """
    Unified data structure for level information.
    
    This class encapsulates both tile data and entity information in a clean,
    type-safe way that can be used consistently across the graph processing
    pipeline and machine learning components.
    
    Attributes:
        tiles: 2D NumPy array representing the tile layout [height, width]
        entities: List of entity dictionaries with position and state information
        level_id: Optional unique identifier for caching purposes
        metadata: Optional dictionary for additional level information
    """
    tiles: np.ndarray
    entities: List[Dict[str, Any]]
    level_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate the data structure after initialization."""
        if not isinstance(self.tiles, np.ndarray):
            raise TypeError("tiles must be a NumPy array")
        
        if len(self.tiles.shape) != 2:
            raise ValueError("tiles must be a 2D array")
        
        if not isinstance(self.entities, list):
            raise TypeError("entities must be a list")
        
        # Generate level_id if not provided
        if self.level_id is None:
            self.level_id = f"level_{id(self.tiles)}"
    
    @property
    def height(self) -> int:
        """Get the height of the level in tiles."""
        return self.tiles.shape[0]
    
    @property
    def width(self) -> int:
        """Get the width of the level in tiles."""
        return self.tiles.shape[1]
    
    @property
    def shape(self) -> tuple:
        """Get the shape of the tile array (height, width)."""
        return self.tiles.shape
    
    def get_tile(self, row: int, col: int) -> int:
        """
        Get the tile type at the specified position.
        
        Args:
            row: Row index (y-coordinate)
            col: Column index (x-coordinate)
            
        Returns:
            Tile type as integer
            
        Raises:
            IndexError: If coordinates are out of bounds
        """
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError(f"Coordinates ({row}, {col}) out of bounds for level shape {self.shape}")
        return int(self.tiles[row, col])
    
    def is_empty(self) -> bool:
        """Check if the level has no tiles or entities."""
        return self.tiles.size == 0 and len(self.entities) == 0
    
    def get_entities_by_type(self, entity_type: Union[int, str]) -> List[Dict[str, Any]]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: The entity type to filter by
            
        Returns:
            List of entity dictionaries matching the type
        """
        return [entity for entity in self.entities if entity.get('type') == entity_type]
    
    def get_entities_in_region(self, x_min: float, y_min: float, x_max: float, y_max: float) -> List[Dict[str, Any]]:
        """
        Get all entities within a rectangular region.
        
        Args:
            x_min, y_min: Minimum coordinates (inclusive)
            x_max, y_max: Maximum coordinates (inclusive)
            
        Returns:
            List of entities within the specified region
        """
        entities_in_region = []
        for entity in self.entities:
            x = entity.get('x', 0.0)
            y = entity.get('y', 0.0)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                entities_in_region.append(entity)
        return entities_in_region
    
    def copy(self) -> 'LevelData':
        """
        Create a deep copy of the level data.
        
        Returns:
            New LevelData instance with copied data
        """
        return LevelData(
            tiles=self.tiles.copy(),
            entities=[entity.copy() if isinstance(entity, dict) else entity for entity in self.entities],
            level_id=self.level_id,
            metadata=self.metadata.copy() if self.metadata else None
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LevelData':
        """
        Create LevelData from a dictionary (for backward compatibility).
        
        Args:
            data: Dictionary containing 'tiles' and optionally 'entities'
            
        Returns:
            New LevelData instance
        """
        tiles = data.get('tiles', np.array([], dtype=np.int32))
        entities = data.get('entities', [])
        level_id = data.get('level_id', None)
        
        # Handle case where tiles might be a list
        if isinstance(tiles, list):
            tiles = np.array(tiles, dtype=np.int32)
        
        return cls(
            tiles=tiles,
            entities=entities,
            level_id=level_id,
            metadata={k: v for k, v in data.items() if k not in ['tiles', 'entities', 'level_id']}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format (for backward compatibility).
        
        Returns:
            Dictionary representation of the level data
        """
        result = {
            'tiles': self.tiles,
            'entities': self.entities,
            'level_id': self.level_id
        }
        
        if self.metadata:
            result.update(self.metadata)
        
        return result
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"LevelData(shape={self.shape}, entities={len(self.entities)}, "
                f"level_id='{self.level_id}')")
    
    def __eq__(self, other) -> bool:
        """Check equality with another LevelData instance."""
        if not isinstance(other, LevelData):
            return False
        
        return (
            np.array_equal(self.tiles, other.tiles) and
            self.entities == other.entities and
            self.level_id == other.level_id
        )


def ensure_level_data(data: Union[LevelData, Dict[str, Any], np.ndarray]) -> LevelData:
    """
    Ensure the input is a LevelData object, converting if necessary.
    
    This utility function provides backward compatibility by converting
    various input formats to LevelData.
    
    Args:
        data: Input data in various formats:
            - LevelData: Returned as-is
            - Dict: Converted using from_dict()
            - np.ndarray: Treated as tiles with empty entities
            
    Returns:
        LevelData object
        
    Raises:
        TypeError: If input format is not supported
    """
    if isinstance(data, LevelData):
        return data
    elif isinstance(data, dict):
        return LevelData.from_dict(data)
    elif isinstance(data, np.ndarray):
        return LevelData(tiles=data, entities=[])
    else:
        raise TypeError(f"Unsupported level data format: {type(data)}")


def create_level_data_dict(level_data: LevelData) -> Dict[str, Any]:
    """
    Create a dictionary with separate tiles and entities for backward compatibility.
    
    This is useful when calling older APIs that expect separate parameters.
    
    Args:
        level_data: LevelData object
        
    Returns:
        Dictionary with 'tiles' and 'entities' keys
    """
    return {
        'tiles': level_data.tiles,
        'entities': level_data.entities,
        'level_id': level_data.level_id
    }
