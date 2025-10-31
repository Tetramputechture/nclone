"""
Level data structure for N++ graph processing.

This module provides a unified data structure for representing complete game state
including tiles, entities, and player information. This eliminates the need for
separate parameter passing and defensive programming patterns throughout the system.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from ..constants.entity_types import EntityType
from ..constants import TILE_PIXEL_SIZE


def extract_start_position_from_map_data(map_data: List[int]) -> Tuple[int, int]:
    """
    Extract ninja spawn position from map_data and convert to tile data coordinates.

    This function provides a centralized way to extract the ninja spawn position
    from map_data for use in LevelData. It handles the conversion from map_data
    coordinates to tile data coordinate space.

    Args:
        map_data: List of integers representing the map data in simulator format.
                 Ninja spawn is stored at indices 1231-1232 in map_data_units.

    Returns:
        Tuple[int, int]: Start position in tile data coordinate space (pixel coordinates).
                        The coordinates are offset by -24px (1 tile) from full map space
                        to account for the 1-tile solid padding around the level.

    Conversion Process:
        1. Read map_data[1231] and map_data[1232] (map_data_units)
        2. Convert to pixels: multiply by 6 (map_data_units -> pixels)
        3. Apply -24px offset to convert from full map space to tile data coordinate space
           (tile data excludes the 1-tile solid padding around the level)

    Example:
        >>> map_data = [0] * 1233
        >>> map_data[1231] = 20  # Example spawn x in map_data_units
        >>> map_data[1232] = 30  # Example spawn y in map_data_units
        >>> pos = extract_start_position_from_map_data(map_data)
        >>> # pos = (20*6 - 24, 30*6 - 24) = (96, 156)
    """
    # Map data stores spawn at indices 1231-1232 in map_data_units
    spawn_x_map_units = map_data[1231]
    spawn_y_map_units = map_data[1232]

    # Convert map_data_units to pixels (full map space)
    # Map data units are multiplied by 6 to get pixel coordinates
    spawn_x_pixels = spawn_x_map_units * 6
    spawn_y_pixels = spawn_y_map_units * 6

    # Apply negative offset to convert to tile data coordinate space
    # Tile data excludes the 1-tile solid padding, so coordinates are offset by -1 tile (-24px)
    start_position = (
        int(spawn_x_pixels - TILE_PIXEL_SIZE),
        int(spawn_y_pixels - TILE_PIXEL_SIZE),
    )

    return start_position


@dataclass
class PlayerState:
    """
    Complete player state information.

    This encapsulates all player-related data that affects game logic
    and pathfinding calculations.
    """

    position: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    on_ground: bool = True
    facing_right: bool = True
    health: int = 1
    frame: int = 0

    @property
    def x(self) -> float:
        """Get x-coordinate of player position."""
        return self.position[0]

    @property
    def y(self) -> float:
        """Get y-coordinate of player position."""
        return self.position[1]

    @property
    def tile_position(self) -> Tuple[int, int]:
        """Get player position in tile coordinates (col, row)."""
        from ..constants import TILE_PIXEL_SIZE

        return (int(self.x // TILE_PIXEL_SIZE), int(self.y // TILE_PIXEL_SIZE))


@dataclass
class LevelData:
    """
    Complete game state including level data and player state.

    This is the primary data structure that should be passed through
    the system, eliminating the need for separate parameters and
    defensive programming patterns.

    Attributes:
        tiles: 2D NumPy array representing the tile layout [height, width]
        entities: List of entity dictionaries with position and state information
        player: Current player state information (optional for backward compatibility)
        level_id: Optional unique identifier for caching purposes
        metadata: Optional dictionary for additional level information
        switch_states: Current state of all switches in the level
    """

    start_position: Tuple[int, int]
    tiles: np.ndarray
    entities: List[Dict[str, Any]]
    player: Optional[PlayerState] = None
    level_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    switch_states: Dict[str, bool] = field(default_factory=dict)

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
            raise IndexError(
                f"Coordinates ({row}, {col}) out of bounds for level shape {self.shape}"
            )
        return int(self.tiles[row, col])

    def is_empty(self) -> bool:
        """Check if the level has no tiles or entities."""
        return self.tiles.size == 0 and len(self.entities) == 0

    def get_entities_by_type(
        self, entity_type: Union[int, str]
    ) -> List[Dict[str, Any]]:
        """
        Get all entities of a specific type.

        Args:
            entity_type: The entity type to filter by (can use EntityType constants)

        Returns:
            List of entity dictionaries matching the type
        """
        return [entity for entity in self.entities if entity.get("type") == entity_type]

    def get_active_entities(self) -> List[Dict[str, Any]]:
        """Get all currently active entities."""
        return [entity for entity in self.entities if entity.get("active", True)]

    def get_exits(self) -> List[Dict[str, Any]]:
        """Get all exit entities (doors and switches)."""
        return self.get_entities_by_type(
            EntityType.EXIT_DOOR
        ) + self.get_entities_by_type(EntityType.EXIT_SWITCH)

    def get_switches(self) -> List[Dict[str, Any]]:
        """Get all switch entities."""
        return self.get_entities_by_type(EntityType.EXIT_SWITCH)

    def get_doors(self) -> List[Dict[str, Any]]:
        """Get all door entities (regular, locked, trap)."""
        return (
            self.get_entities_by_type(EntityType.REGULAR_DOOR)
            + self.get_entities_by_type(EntityType.LOCKED_DOOR)
            + self.get_entities_by_type(EntityType.TRAP_DOOR)
        )

    def get_enemies(self) -> List[Dict[str, Any]]:
        """Get all enemy entities."""
        return (
            self.get_entities_by_type(EntityType.DRONE_ZAP)
            + self.get_entities_by_type(EntityType.MINI_DRONE)
            + self.get_entities_by_type(EntityType.THWUMP)
            + self.get_entities_by_type(EntityType.SHWUMP)
            + self.get_entities_by_type(EntityType.DEATH_BALL)
        )

    def update_switch_state(self, switch_id: str, active: bool) -> None:
        """
        Update the state of a switch and related entities.

        Args:
            switch_id: Identifier for the switch
            active: New active state
        """
        self.switch_states[switch_id] = active

    def get_entities_in_region(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> List[Dict[str, Any]]:
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
            x = entity.get("x", 0.0)
            y = entity.get("y", 0.0)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                entities_in_region.append(entity)
        return entities_in_region

    def copy(self) -> "LevelData":
        """
        Create a deep copy of the level data.

        Returns:
            New LevelData instance with copied data
        """
        return LevelData(
            start_position=self.start_position,
            tiles=self.tiles.copy(),
            entities=[
                entity.copy() if isinstance(entity, dict) else entity
                for entity in self.entities
            ],
            level_id=self.level_id,
            metadata=self.metadata.copy() if self.metadata else None,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LevelData":
        """
        Create LevelData from a dictionary (for backward compatibility).

        Args:
            data: Dictionary containing 'tiles' and optionally 'entities' and 'start_position'

        Returns:
            New LevelData instance
        """
        tiles = data.get("tiles", np.array([], dtype=np.int32))
        entities = data.get("entities", [])
        level_id = data.get("level_id", None)
        start_position = data.get("start_position", None)

        # Handle case where tiles might be a list
        if isinstance(tiles, list):
            tiles = np.array(tiles, dtype=np.int32)

        # If start_position not provided, try to extract from entities
        if start_position is None:
            # Look for ninja entity with type 14 or EntityType.NINJA
            for entity in entities:
                entity_type = entity.get("type")
                if entity_type == 14:  # Ninja spawn point
                    start_position = (entity.get("x", 0), entity.get("y", 0))
                    break

            # If still not found, fallback to center of level
            if start_position is None:
                height, width = tiles.shape if tiles.size > 0 else (0, 0)
                if height > 0 and width > 0:
                    from ..constants import TILE_PIXEL_SIZE

                    center_x = (width // 2) * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                    center_y = (height // 2) * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                    start_position = (center_x, center_y)
                else:
                    start_position = (0, 0)

        return cls(
            start_position=start_position,
            tiles=tiles,
            entities=entities,
            level_id=level_id,
            metadata={
                k: v
                for k, v in data.items()
                if k not in ["tiles", "entities", "level_id", "start_position"]
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format (for backward compatibility).

        Returns:
            Dictionary representation of the level data
        """
        result = {
            "tiles": self.tiles,
            "entities": self.entities,
            "level_id": self.level_id,
            "start_position": self.start_position,
        }

        if self.metadata:
            result.update(self.metadata)

        return result

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"LevelData(shape={self.shape}, entities={len(self.entities)}, "
            f"level_id='{self.level_id}')"
        )

    def __eq__(self, other) -> bool:
        """
        Check equality with another LevelData instance.

        Two LevelData instances are considered equal if they have:
        - Same tile data
        - Same entities with same attributes
        - Same start position

        All other attributes (level_id, player state, metadata, switch_states) are
        considered emergent from these fundamental properties and don't affect equality.
        """
        if not isinstance(other, LevelData):
            return False

        return (
            np.array_equal(self.tiles, other.tiles)
            and self.entities == other.entities
            and self.start_position == other.start_position
        )


def ensure_level_data(
    data: Union[LevelData, Dict[str, Any], np.ndarray],
    player_position: Optional[Tuple[float, float]] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
) -> LevelData:
    """
    Ensure the input is a LevelData object, converting if necessary.

    This utility function provides backward compatibility by converting
    various input formats to LevelData. It can also add player state
    and entities if they're provided separately.

    Args:
        data: Input data in various formats:
            - LevelData: Returned as-is (unless player_position is provided)
            - Dict: Converted using from_dict()
            - np.ndarray: Treated as tiles with empty entities
        player_position: Optional player position to add/update
        entities: Optional entities list to add/update

    Returns:
        LevelData object with consolidated information

    Raises:
        TypeError: If input format is not supported
    """
    if isinstance(data, LevelData):
        level_data = data
    elif isinstance(data, dict):
        level_data = LevelData.from_dict(data)
    elif isinstance(data, np.ndarray):
        level_data = LevelData(
            tiles=data, entities=entities or [], start_position=(0, 0)
        )
    else:
        raise TypeError(f"Unsupported level data format: {type(data)}")

    # Update player state if provided
    if player_position is not None:
        if level_data.player is None:
            level_data.player = PlayerState(position=player_position)
        else:
            level_data.player.position = player_position

    # Update entities if provided
    if entities is not None:
        level_data.entities = entities

    return level_data


def create_level_data_with_player(
    tiles: np.ndarray,
    entities: List[Dict[str, Any]],
    player_position: Tuple[float, float],
    level_id: Optional[str] = None,
    start_position: Optional[Tuple[int, int]] = None,
    **kwargs,
) -> LevelData:
    """
    Create LevelData with complete game state information.

    This is the preferred way to create LevelData objects with all
    necessary information in one call.

    Args:
        tiles: 2D tile array
        entities: List of entity dictionaries
        player_position: Player position tuple
        level_id: Optional level identifier
        start_position: Optional start position tuple (if None, uses player_position)
        **kwargs: Additional metadata

    Returns:
        Complete LevelData object
    """
    player_state = PlayerState(position=player_position)

    # Use start_position if provided, otherwise use player_position
    if start_position is None:
        start_position = (int(player_position[0]), int(player_position[1]))

    return LevelData(
        start_position=start_position,
        tiles=tiles,
        entities=entities,
        player=player_state,
        level_id=level_id,
        metadata=kwargs,
    )


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
        "tiles": level_data.tiles,
        "entities": level_data.entities,
        "level_id": level_data.level_id,
    }
