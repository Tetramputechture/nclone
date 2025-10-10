"""
Compact Gameplay Recorder for Human Demonstrations

Records ONLY map data and input sequence during test environment runs.
Leverages deterministic physics to regenerate observations on-demand.

Storage format mirrors N++ attract files:
- Map data: 1335 bytes (fixed)
- Input sequence: 1 byte per frame
- Total: ~1-5KB per episode (vs 500KB+ with observations)
"""

import struct
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CompactReplay:
    """Compact replay format - only inputs and map data."""
    
    episode_id: str
    map_data: bytes  # Raw map data (1335 bytes)
    input_sequence: List[int]  # Input values (0-7 per frame)
    
    # Metadata
    level_id: Optional[str]
    start_time: datetime
    end_time: datetime
    success: bool
    
    def to_binary(self) -> bytes:
        """
        Convert to binary format (similar to N++ attract files).
        
        Format:
        - Header (8 bytes):
            - Map data length (4 bytes, uint32)
            - Input sequence length (4 bytes, uint32)
        - Map data (variable, typically 1335 bytes)
        - Input sequence (variable, 1 byte per frame)
        """
        map_data_len = len(self.map_data)
        input_seq_len = len(self.input_sequence)
        
        # Pack header
        header = struct.pack("<II", map_data_len, input_seq_len)
        
        # Pack input sequence
        inputs_bytes = bytes(self.input_sequence)
        
        # Combine
        return header + self.map_data + inputs_bytes
    
    @classmethod
    def from_binary(cls, data: bytes, episode_id: str = "unknown") -> "CompactReplay":
        """Load from binary format."""
        # Unpack header
        map_data_len, input_seq_len = struct.unpack("<II", data[0:8])
        
        # Extract map data
        map_data = data[8:8+map_data_len]
        
        # Extract input sequence
        input_start = 8 + map_data_len
        input_sequence = list(data[input_start:input_start+input_seq_len])
        
        return cls(
            episode_id=episode_id,
            map_data=map_data,
            input_sequence=input_sequence,
            level_id=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=True,
        )
    
    def get_file_size(self) -> int:
        """Get file size in bytes."""
        return 8 + len(self.map_data) + len(self.input_sequence)


def map_action_to_input(action: int) -> int:
    """
    Map discrete action (0-5) to N++ input byte (0-7).
    
    Actions:
        0: NOOP
        1: LEFT
        2: RIGHT
        3: JUMP
        4: LEFT + JUMP
        5: RIGHT + JUMP
    
    Input encoding (bit flags):
        Bit 0: Jump
        Bit 1: Right
        Bit 2: Left
    """
    action_to_input_map = {
        0: 0,  # NOOP: 000
        1: 4,  # LEFT: 100
        2: 2,  # RIGHT: 010
        3: 1,  # JUMP: 001
        4: 5,  # LEFT+JUMP: 101
        5: 3,  # RIGHT+JUMP: 011
    }
    return action_to_input_map.get(action, 0)


class GameplayRecorder:
    """Compact gameplay recorder - stores only map data and input sequence."""
    
    def __init__(self, output_dir: str = "datasets/human_replays"):
        """Initialize gameplay recorder.
        
        Args:
            output_dir: Directory to save recorded replays
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Recording state
        self.is_recording = False
        self.current_map_data: Optional[bytes] = None
        self.current_input_sequence: List[int] = []
        self.current_episode_id: Optional[str] = None
        self.current_level_id: Optional[str] = None
        self.episode_start_time: Optional[datetime] = None
        
        # Statistics
        self.total_episodes_recorded = 0
        self.successful_episodes_recorded = 0
        
    def start_recording(self, map_data: bytes, map_name: str = "unknown", level_id: Optional[str] = None):
        """Start recording a new episode.
        
        Args:
            map_data: Raw map data bytes (1335 bytes)
            map_name: Name/identifier for the map
            level_id: Optional level ID from test suite
        """
        if self.is_recording:
            print("⚠️  Already recording, stopping previous episode")
            self.stop_recording(success=False)
        
        self.is_recording = True
        self.current_map_data = map_data
        self.current_input_sequence = []
        self.episode_start_time = datetime.now()
        self.current_level_id = level_id
        
        # Generate unique episode ID
        self.current_episode_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{map_name}"
        
        print(f"🔴 Recording started: {self.current_episode_id}")
        print(f"   Map size: {len(map_data)} bytes")
    
    def record_action(self, action: int):
        """Record a single action.
        
        Args:
            action: Discrete action taken (0-5)
        """
        if not self.is_recording:
            return
        
        # Convert action to input byte
        input_byte = map_action_to_input(action)
        self.current_input_sequence.append(input_byte)
    
    def stop_recording(self, success: bool, save: bool = True) -> Optional[str]:
        """Stop recording and optionally save the replay.
        
        Args:
            success: Whether the episode ended in success (level completed)
            save: Whether to save the replay to disk
        
        Returns:
            Path to saved replay file, or None if not saved
        """
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        # Update statistics
        self.total_episodes_recorded += 1
        if success:
            self.successful_episodes_recorded += 1
        
        # Save replay if successful and save flag is True
        saved_path = None
        if save and success and self.current_map_data is not None:
            saved_path = self._save_replay()
            
            duration = (datetime.now() - self.episode_start_time).total_seconds()
            file_size = 8 + len(self.current_map_data) + len(self.current_input_sequence)
            
            print(f"✅ Replay saved: {saved_path}")
            print(f"   - Frames: {len(self.current_input_sequence)}")
            print(f"   - Duration: {duration:.1f}s")
            print(f"   - File size: {file_size} bytes ({file_size/1024:.1f} KB)")
        elif not success:
            print(f"❌ Episode failed, not saved")
        
        # Reset state
        self.current_map_data = None
        self.current_input_sequence = []
        self.current_episode_id = None
        
        return saved_path
    
    def _save_replay(self) -> str:
        """Save replay to disk in compact binary format.
        
        Returns:
            Path to saved replay file
        """
        if self.current_map_data is None or self.current_episode_id is None:
            raise ValueError("Cannot save replay: missing data")
        
        # Create compact replay
        replay = CompactReplay(
            episode_id=self.current_episode_id,
            map_data=self.current_map_data,
            input_sequence=self.current_input_sequence,
            level_id=self.current_level_id,
            start_time=self.episode_start_time,
            end_time=datetime.now(),
            success=True,
        )
        
        # Save binary replay file
        replay_path = self.output_dir / f"{self.current_episode_id}.replay"
        with open(replay_path, "wb") as f:
            f.write(replay.to_binary())
        
        return str(replay_path)
    
    def print_statistics(self):
        """Print recording statistics."""
        print("\n" + "=" * 60)
        print("COMPACT REPLAY RECORDING STATISTICS")
        print("=" * 60)
        print(f"Total episodes recorded: {self.total_episodes_recorded}")
        print(f"Successful episodes: {self.successful_episodes_recorded}")
        if self.total_episodes_recorded > 0:
            success_rate = self.successful_episodes_recorded / self.total_episodes_recorded * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        # List replay files and total storage
        replay_files = list(self.output_dir.glob("*.replay"))
        if replay_files:
            total_size = sum(f.stat().st_size for f in replay_files)
            print(f"Replay files: {len(replay_files)}")
            print(f"Total storage: {total_size / 1024:.1f} KB")
            print(f"Average file size: {total_size / len(replay_files) / 1024:.1f} KB")
        print("=" * 60)
