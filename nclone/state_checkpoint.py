"""
State checkpoint for Go-Explore action replay.

Stores action sequences that can be replayed deterministically to restore
any checkpoint state. Since N++ physics is fully deterministic, replaying
the same action sequence from spawn will always produce identical results.

This approach is simpler and more robust than full state serialization,
as it leverages the game's inherent determinism.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class StateCheckpoint:
    """
    Checkpoint storing action sequence for deterministic replay.
    
    The action sequence can be replayed from spawn to exactly reproduce
    this game state due to N++ physics being fully deterministic.
    
    Attributes:
        cell: Grid cell position (24px discretization) for spatial indexing
        action_sequence: List of actions (0-5) from spawn to this state
        frame_count: Number of frames to reach this state
        distance_to_goal: Distance to current goal when checkpoint was created
        ninja_position: Exact (x, y) position in pixels for validation
        ninja_velocity: Velocity (vx, vy) at checkpoint for physics continuity info
        switch_activated: Whether exit switch has been collected
        objective_event: What triggered this checkpoint ("switch", "locked_door_N", or None)
        discovery_timestep: Training timestep when this checkpoint was discovered
        novelty: Novelty score based on visit frequency (higher = more novel)
        visit_count: Number of times this cell has been visited
        on_success_path: Whether this checkpoint is on a path that led to success
        locked_door_states: Dict mapping door IDs to their activation states
    """
    
    # Core identification
    cell: Tuple[int, int]
    
    # Action replay data
    action_sequence: List[int] = field(default_factory=list)
    frame_count: int = 0
    
    # Goal progress
    distance_to_goal: float = float("inf")
    
    # Position for validation after replay
    ninja_position: Tuple[float, float] = (0.0, 0.0)
    ninja_velocity: Tuple[float, float] = (0.0, 0.0)
    
    # Objective state
    switch_activated: bool = False
    objective_event: Optional[str] = None
    locked_door_states: Dict[str, bool] = field(default_factory=dict)
    
    # Exploration metadata
    discovery_timestep: int = 0
    novelty: float = 1.0
    visit_count: int = 1
    on_success_path: bool = False
    
    def __post_init__(self):
        """Validate checkpoint data."""
        if not isinstance(self.action_sequence, list):
            self.action_sequence = list(self.action_sequence)
        if not isinstance(self.locked_door_states, dict):
            self.locked_door_states = dict(self.locked_door_states)
    
    @property
    def replay_length(self) -> int:
        """Number of actions to replay to reach this checkpoint."""
        return len(self.action_sequence)
    
    def get_compact_representation(self) -> Dict[str, Any]:
        """
        Get a compact dictionary representation for serialization.
        
        Converts action_sequence to numpy array for efficient storage.
        """
        return {
            "cell": self.cell,
            "actions": np.array(self.action_sequence, dtype=np.uint8),
            "frame_count": self.frame_count,
            "distance": self.distance_to_goal,
            "position": self.ninja_position,
            "velocity": self.ninja_velocity,
            "switch": self.switch_activated,
            "event": self.objective_event,
            "doors": self.locked_door_states,
            "timestep": self.discovery_timestep,
            "novelty": self.novelty,
            "visits": self.visit_count,
            "success_path": self.on_success_path,
        }
    
    @classmethod
    def from_compact_representation(cls, data: Dict[str, Any]) -> "StateCheckpoint":
        """
        Create checkpoint from compact dictionary representation.
        
        Handles numpy array conversion back to list.
        """
        actions = data.get("actions", [])
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        
        return cls(
            cell=tuple(data["cell"]),
            action_sequence=actions,
            frame_count=data.get("frame_count", len(actions)),
            distance_to_goal=data.get("distance", float("inf")),
            ninja_position=tuple(data.get("position", (0.0, 0.0))),
            ninja_velocity=tuple(data.get("velocity", (0.0, 0.0))),
            switch_activated=data.get("switch", False),
            objective_event=data.get("event"),
            locked_door_states=data.get("doors", {}),
            discovery_timestep=data.get("timestep", 0),
            novelty=data.get("novelty", 1.0),
            visit_count=data.get("visits", 1),
            on_success_path=data.get("success_path", False),
        )
    
    def copy_with_extended_actions(self, new_actions: List[int]) -> "StateCheckpoint":
        """
        Create a new checkpoint with extended action sequence.
        
        Useful for creating child checkpoints during exploration.
        
        Args:
            new_actions: Additional actions to append
            
        Returns:
            New StateCheckpoint with extended action sequence
        """
        return StateCheckpoint(
            cell=self.cell,
            action_sequence=self.action_sequence + new_actions,
            frame_count=self.frame_count + len(new_actions),
            distance_to_goal=self.distance_to_goal,
            ninja_position=self.ninja_position,
            ninja_velocity=self.ninja_velocity,
            switch_activated=self.switch_activated,
            objective_event=self.objective_event,
            locked_door_states=dict(self.locked_door_states),
            discovery_timestep=self.discovery_timestep,
            novelty=self.novelty,
            visit_count=self.visit_count,
            on_success_path=self.on_success_path,
        )


@dataclass
class CheckpointValidationResult:
    """Result of validating a checkpoint after action replay."""
    
    success: bool
    position_error: float = 0.0
    expected_position: Tuple[float, float] = (0.0, 0.0)
    actual_position: Tuple[float, float] = (0.0, 0.0)
    error_message: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed with acceptable error."""
        # Allow small floating point differences (< 0.01 pixels)
        return self.success and self.position_error < 0.01


# Constants for checkpoint management
CHECKPOINT_GRID_SIZE = 24  # Matches tile size for spatial discretization
POSITION_VALIDATION_THRESHOLD = 0.01  # Max acceptable position error in pixels

