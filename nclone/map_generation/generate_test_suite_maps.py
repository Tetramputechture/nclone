"""
Generate comprehensive test suite maps for NPP-RL evaluation (Task 3.3).

This script creates a deterministic dataset of 250 N++ levels across 5 complexity categories:
- 50 simple levels (single switch, direct path)
- 100 medium levels (1-3 switches, simple dependencies, may require jumps)
- 50 complex levels (4+ switches, complex dependencies, multi-chamber)
- 30 mine-heavy levels (significant mine obstacles)
- 20 exploration levels (hidden switches, extensive exploration required)

All maps are generated deterministically using fixed seeds to ensure reproducibility.
The dataset can be used as a baseline for training and evaluating NPP-RL agents.

Usage:
    python -m nclone.map_generation.generate_test_suite_maps --output_dir /path/to/dataset
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from .map import Map
from .map_single_chamber import SingleChamberGenerator
from .map_jump_required import MapJumpRequired
from .map_maze import MazeGenerator
from .map_multi_chamber import MultiChamberGenerator
from .map_mine_maze import MapMineMaze
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


class TestSuiteGenerator:
    """Generator for comprehensive NPP-RL test suite."""
    
    # Base seeds for each category to ensure deterministic generation
    SIMPLE_BASE_SEED = 1000
    MEDIUM_BASE_SEED = 2000
    COMPLEX_BASE_SEED = 3000
    MINE_HEAVY_BASE_SEED = 4000
    EXPLORATION_BASE_SEED = 5000
    
    def __init__(self, output_dir: str):
        """Initialize the test suite generator.
        
        Args:
            output_dir: Directory where the test suite will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track generated levels
        self.levels: Dict[str, List[Dict[str, Any]]] = {
            'simple': [],
            'medium': [],
            'complex': [],
            'mine_heavy': [],
            'exploration': []
        }
    
    def _add_locked_door_to_corridor(self, map_gen: Map, corridor_x: int, corridor_y: int, 
                                      switch_x: int, switch_y: int, orientation: int = 4) -> None:
        """Add a locked door blocking a corridor with its switch placed elsewhere.
        
        Args:
            map_gen: Map instance to add door to
            corridor_x, corridor_y: Grid position where door blocks passage
            switch_x, switch_y: Grid position of the switch that opens this door
            orientation: Door orientation (0 or 4 for vertical, others for horizontal)
        """
        map_gen.add_entity(6, corridor_x, corridor_y, orientation=orientation, mode=2,
                          switch_x=switch_x, switch_y=switch_y)
    
    def _find_empty_tiles_in_region(self, map_gen: Map, x1: int, y1: int, 
                                     x2: int, y2: int) -> List[Tuple[int, int]]:
        """Find all empty (walkable) tiles in a rectangular region.
        
        Returns:
            List of (x, y) coordinates of empty tiles
        """
        empty_tiles = []
        for y in range(y1, min(y2 + 1, 25)):
            for x in range(x1, min(x2 + 1, 44)):
                idx = x + y * 43
                if 0 <= idx < len(map_gen.tile_data) and map_gen.tile_data[idx] == 0:
                    empty_tiles.append((x, y))
        return empty_tiles
    
    def generate_simple_levels(self, count: int = 50) -> None:
        """Generate simple levels: single switch, direct path to exit.
        
        These levels test basic navigation and switch activation.
        Progression: flat surfaces -> small vertical deviations -> locked doors -> require jump
        
        Args:
            count: Number of simple levels to generate
        """
        print(f"Generating {count} simple levels...")
        
        for i in range(count):
            seed = self.SIMPLE_BASE_SEED + i
            
            # First 15: very simple flat levels (minimal chamber with exit switch only)
            if i < 15:
                map_gen = self._create_minimal_simple_level(seed, i)
            # Next 10: single chamber with vertical deviations
            elif i < 25:
                map_gen = self._create_single_chamber_level(seed, with_deviation=True)
            # Next 15: simple locked door (introduces type 6 doors)
            elif i < 40:
                map_gen = self._create_simple_locked_door_level(seed, i - 25)
            # Last 10: require a jump
            else:
                map_gen = self._create_simple_jump_level(seed)
            
            level_data = {
                'level_id': f'simple_{i:03d}',
                'seed': seed,
                'category': 'simple',
                'map_data': map_gen.map_data(),
                'metadata': {
                    'description': self._get_simple_description(i),
                    'difficulty_tier': 1 if i < 15 else (2 if i < 25 else (3 if i < 40 else 4))
                }
            }
            self.levels['simple'].append(level_data)
        
        print(f"✓ Generated {count} simple levels")
    
    def _create_minimal_simple_level(self, seed: int, index: int) -> Map:
        """Create a minimal simple level (1-3 tiles high, 3-12 tiles wide)."""
        map_gen = Map(seed=seed)
        rng = map_gen.rng
        
        # Very small dimensions for simplest levels
        width = 3 + (index % 10)  # 3-12 tiles wide
        height = 1 + (index % 3)  # 1-3 tiles high
        
        # Center the chamber
        start_x = (MAP_TILE_WIDTH - width) // 2
        start_y = (MAP_TILE_HEIGHT - height) // 2
        
        # Fill everything with walls first
        for y in range(MAP_TILE_HEIGHT):
            for x in range(MAP_TILE_WIDTH):
                map_gen.set_tile(x, y, 1)
        
        # Create empty chamber
        for y in range(start_y, start_y + height):
            for x in range(start_x, start_x + width):
                map_gen.set_tile(x, y, 0)
        
        # Place ninja on one side
        ninja_x = start_x
        ninja_y = start_y + height - 1
        
        # Place switch in middle
        switch_x = start_x + width // 2
        switch_y = start_y + height - 1
        
        # Place door on other side
        door_x = start_x + width - 1
        door_y = start_y + height - 1
        
        map_gen.set_ninja_spawn(ninja_x, ninja_y, orientation=1)
        map_gen.add_entity(3, door_x, door_y, 0, 0, switch_x, switch_y)
        
        return map_gen
    
    def _create_single_chamber_level(self, seed: int, with_deviation: bool = False) -> Map:
        """Create a single chamber level with optional vertical deviation."""
        map_gen = SingleChamberGenerator(seed=seed)
        
        # Override some constants for controlled difficulty
        if not with_deviation:
            map_gen.GLOBAL_MAX_UP_DEVIATION = 0
            map_gen.GLOBAL_MAX_DOWN_DEVIATION = 0
        else:
            map_gen.GLOBAL_MAX_UP_DEVIATION = 3
            map_gen.GLOBAL_MAX_DOWN_DEVIATION = 1
        
        # Generate smaller chambers for simple levels
        map_gen.MIN_WIDTH = 6
        map_gen.MAX_WIDTH = 15
        map_gen.MIN_HEIGHT = 4
        map_gen.MAX_HEIGHT = 8
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def _create_simple_jump_level(self, seed: int) -> Map:
        """Create a simple level that requires a jump."""
        map_gen = MapJumpRequired(seed=seed)
        
        # Make it simple: small pit, easy to navigate
        # Need enough height for pit + switch placement (minimum 8)
        map_gen.MIN_WIDTH = 12
        map_gen.MAX_WIDTH = 20
        map_gen.MIN_HEIGHT = 8  # Increased to ensure switch placement works
        map_gen.MAX_HEIGHT = 10
        map_gen.MIN_PIT_WIDTH = 2
        map_gen.MAX_PIT_WIDTH = 3
        map_gen.MAX_MINES_PER_PLATFORM = 2  # Very few mines
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def _create_simple_locked_door_level(self, seed: int, index: int) -> Map:
        """Create a simple level with one locked door blocking progress.
        
        Layout: Ninja -> Switch -> Locked Door -> Exit
        """
        map_gen = Map(seed=seed)
        rng = map_gen.rng
        
        # Create a simple corridor with locked door
        width = 16 + (index % 8)
        height = 3 + (index % 2)
        
        # Center the corridor
        start_x = (MAP_TILE_WIDTH - width) // 2
        start_y = (MAP_TILE_HEIGHT - height) // 2
        
        # Fill everything with walls
        for y in range(MAP_TILE_HEIGHT):
            for x in range(MAP_TILE_WIDTH):
                map_gen.set_tile(x, y, 1)
        
        # Create corridor
        for y in range(start_y, start_y + height):
            for x in range(start_x, start_x + width):
                map_gen.set_tile(x, y, 0)
        
        # Place ninja at start
        ninja_x = start_x
        ninja_y = start_y + height - 1
        
        # Place switch in first third
        switch_x = start_x + width // 4
        switch_y = start_y + height - 1
        
        # Place locked door in middle (blocks passage)
        door_x = start_x + width // 2
        door_y = start_y + height - 1
        
        # Place exit at end
        exit_x = start_x + width - 2
        exit_y = start_y + height - 1
        exit_switch_x = start_x + 3 * width // 4
        exit_switch_y = start_y + height - 1
        
        map_gen.set_ninja_spawn(ninja_x, ninja_y, orientation=1)
        
        # Add locked door (type 6) - must collect switch before passing
        self._add_locked_door_to_corridor(map_gen, door_x, door_y, switch_x, switch_y, orientation=4)
        
        # Add exit door with its switch (type 3 + 4)
        map_gen.add_entity(3, exit_x, exit_y, 0, 0, exit_switch_x, exit_switch_y)
        
        return map_gen
    
    def _get_simple_description(self, index: int) -> str:
        """Get description for simple level based on index."""
        if index < 15:
            return "Minimal chamber: ninja -> exit switch -> door (no locked doors)"
        elif index < 25:
            return "Single chamber with vertical deviations, exit switch only"
        elif index < 40:
            return "Corridor with locked door: must collect switch to pass, then reach exit"
        else:
            return "Simple jump required to reach switch or exit"
    
    def generate_medium_levels(self, count: int = 100) -> None:
        """Generate medium levels: 1-3 switches, simple dependencies.
        
        These levels test navigation with multiple objectives and basic planning.
        Mix of: small mazes, multi-switch chambers, jump-required with switches
        
        Args:
            count: Number of medium levels to generate
        """
        print(f"Generating {count} medium levels...")
        
        for i in range(count):
            seed = self.MEDIUM_BASE_SEED + i
            
            # Mix different types of medium levels
            level_type = i % 4
            
            if level_type == 0:
                # Small maze
                map_gen = self._create_small_maze(seed)
                desc = "Small maze with 1-2 switches"
            elif level_type == 1:
                # Multi-chamber (2 chambers)
                map_gen = self._create_medium_multi_chamber(seed, num_chambers=2)
                desc = "2-chamber level with switch dependencies"
            elif level_type == 2:
                # Jump required with mines
                map_gen = self._create_medium_jump_level(seed)
                desc = "Jump required with moderate mine obstacles"
            else:
                # Larger single chamber with obstacles
                map_gen = self._create_medium_chamber_with_obstacles(seed)
                desc = "Medium chamber with 2-3 switches and obstacles"
            
            level_data = {
                'level_id': f'medium_{i:03d}',
                'seed': seed,
                'category': 'medium',
                'map_data': map_gen.map_data(),
                'metadata': {
                    'description': desc,
                    'difficulty_tier': (i // 25) + 1  # 4 tiers
                }
            }
            self.levels['medium'].append(level_data)
        
        print(f"✓ Generated {count} medium levels")
    
    def _create_small_maze(self, seed: int) -> Map:
        """Create a small maze level."""
        map_gen = MazeGenerator(seed=seed)
        
        # Small maze dimensions
        map_gen.MIN_WIDTH = 8
        map_gen.MAX_WIDTH = 16
        map_gen.MIN_HEIGHT = 6
        map_gen.MAX_HEIGHT = 10
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def _create_medium_multi_chamber(self, seed: int, num_chambers: int = 2) -> Map:
        """Create a multi-chamber level with specified number of chambers."""
        map_gen = MultiChamberGenerator(seed=seed)
        
        # Control chamber count
        map_gen.MIN_CHAMBERS = num_chambers
        map_gen.MAX_CHAMBERS = num_chambers
        
        # Medium-sized chambers
        map_gen.MIN_CHAMBER_WIDTH = 5
        map_gen.MAX_CHAMBER_WIDTH = 10
        map_gen.MIN_CHAMBER_HEIGHT = 4
        map_gen.MAX_CHAMBER_HEIGHT = 7
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def _create_medium_jump_level(self, seed: int) -> Map:
        """Create a medium difficulty jump level with mines."""
        map_gen = MapJumpRequired(seed=seed)
        
        # Medium difficulty
        map_gen.MIN_WIDTH = 16
        map_gen.MAX_WIDTH = 30
        map_gen.MIN_HEIGHT = 8
        map_gen.MAX_HEIGHT = 14
        map_gen.MIN_PIT_WIDTH = 3
        map_gen.MAX_PIT_WIDTH = 5
        map_gen.MAX_MINES_PER_PLATFORM = 4
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def _create_medium_chamber_with_obstacles(self, seed: int) -> Map:
        """Create a chamber with some mine obstacles."""
        map_gen = MapMineMaze(seed=seed)
        
        # Medium chamber with moderate mines
        map_gen.MIN_WIDTH = 12
        map_gen.MAX_WIDTH = 25
        map_gen.MIN_HEIGHT = 4
        map_gen.MAX_HEIGHT = 7
        
        # Fewer mines than mine-heavy levels
        map_gen.MIN_SKIP_COLUMNS = 3
        map_gen.MAX_SKIP_COLUMNS = 6
        map_gen.MIN_MINES_PER_COLUMN = 1
        map_gen.MAX_MINES_PER_COLUMN = 4
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def generate_complex_levels(self, count: int = 50) -> None:
        """Generate complex levels: 4+ switches, complex dependencies.
        
        These levels test advanced planning and multi-step problem solving.
        Multi-chamber with many switches, large mazes, complex jump sequences.
        
        Args:
            count: Number of complex levels to generate
        """
        print(f"Generating {count} complex levels...")
        
        for i in range(count):
            seed = self.COMPLEX_BASE_SEED + i
            
            # Alternate between different complex level types
            level_type = i % 3
            
            if level_type == 0:
                # Large multi-chamber (3-4 chambers)
                map_gen = self._create_complex_multi_chamber(seed)
                desc = "3-4 chambers with complex switch dependencies"
            elif level_type == 1:
                # Large maze
                map_gen = self._create_large_maze(seed)
                desc = "Large maze requiring extensive navigation"
            else:
                # Complex jump sequence with multiple objectives
                map_gen = self._create_complex_jump_sequence(seed)
                desc = "Complex jump sequence with multiple switches"
            
            level_data = {
                'level_id': f'complex_{i:03d}',
                'seed': seed,
                'category': 'complex',
                'map_data': map_gen.map_data(),
                'metadata': {
                    'description': desc,
                    'difficulty_tier': (i // 17) + 1  # 3 tiers
                }
            }
            self.levels['complex'].append(level_data)
        
        print(f"✓ Generated {count} complex levels")
    
    def _create_complex_multi_chamber(self, seed: int) -> Map:
        """Create a complex multi-chamber level."""
        map_gen = MultiChamberGenerator(seed=seed)
        
        # More chambers
        map_gen.MIN_CHAMBERS = 3
        map_gen.MAX_CHAMBERS = 4
        
        # Larger chambers
        map_gen.MIN_CHAMBER_WIDTH = 6
        map_gen.MAX_CHAMBER_WIDTH = 12
        map_gen.MIN_CHAMBER_HEIGHT = 5
        map_gen.MAX_CHAMBER_HEIGHT = 8
        
        # More gold per chamber for complexity
        map_gen.MAX_GOLD_PER_CHAMBER = 2
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def _create_large_maze(self, seed: int) -> Map:
        """Create a large maze level."""
        map_gen = MazeGenerator(seed=seed)
        
        # Large maze dimensions
        map_gen.MIN_WIDTH = 16
        map_gen.MAX_WIDTH = 30
        map_gen.MIN_HEIGHT = 10
        map_gen.MAX_HEIGHT = 18
        
        # Add some gold for extra objectives
        map_gen.MAX_GOLD = 3
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def _create_complex_jump_sequence(self, seed: int) -> Map:
        """Create a complex jump level with multiple platforms."""
        # For complex jump sequences, we'll use multi-chamber with jump-like properties
        map_gen = MultiChamberGenerator(seed=seed)
        
        map_gen.MIN_CHAMBERS = 3
        map_gen.MAX_CHAMBERS = 4
        map_gen.MIN_CHAMBER_WIDTH = 5
        map_gen.MAX_CHAMBER_WIDTH = 10
        map_gen.MIN_CHAMBER_HEIGHT = 4
        map_gen.MAX_CHAMBER_HEIGHT = 7
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def generate_mine_heavy_levels(self, count: int = 30) -> None:
        """Generate mine-heavy levels: significant mine obstacles.
        
        These levels test hazard avoidance and precise navigation.
        Various configurations with high mine density.
        
        Args:
            count: Number of mine-heavy levels to generate
        """
        print(f"Generating {count} mine-heavy levels...")
        
        for i in range(count):
            seed = self.MINE_HEAVY_BASE_SEED + i
            
            # Alternate between mine maze and jump with heavy mines
            if i % 2 == 0:
                map_gen = self._create_heavy_mine_maze(seed)
                desc = "Mine-heavy maze requiring careful navigation"
            else:
                map_gen = self._create_heavy_mine_jump(seed)
                desc = "Jump level with heavy mine obstacles"
            
            level_data = {
                'level_id': f'mine_heavy_{i:03d}',
                'seed': seed,
                'category': 'mine_heavy',
                'map_data': map_gen.map_data(),
                'metadata': {
                    'description': desc,
                    'difficulty_tier': (i // 10) + 1  # 3 tiers
                }
            }
            self.levels['mine_heavy'].append(level_data)
        
        print(f"✓ Generated {count} mine-heavy levels")
    
    def _create_heavy_mine_maze(self, seed: int) -> Map:
        """Create a mine maze with heavy mine density."""
        map_gen = MapMineMaze(seed=seed)
        
        # Large chamber
        map_gen.MIN_WIDTH = 15
        map_gen.MAX_WIDTH = 35
        map_gen.MIN_HEIGHT = 5
        map_gen.MAX_HEIGHT = 10
        
        # High mine density
        map_gen.MIN_SKIP_COLUMNS = 2
        map_gen.MAX_SKIP_COLUMNS = 3
        map_gen.MIN_MINES_PER_COLUMN = 3
        map_gen.MAX_MINES_PER_COLUMN = 10
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def _create_heavy_mine_jump(self, seed: int) -> Map:
        """Create a jump level with heavy mines."""
        map_gen = MapJumpRequired(seed=seed)
        
        # Large jump level
        map_gen.MIN_WIDTH = 20
        map_gen.MAX_WIDTH = 40
        map_gen.MIN_HEIGHT = 10
        map_gen.MAX_HEIGHT = 16
        map_gen.MIN_PIT_WIDTH = 4
        map_gen.MAX_PIT_WIDTH = 6
        map_gen.MAX_MINES_PER_PLATFORM = 5
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def generate_exploration_levels(self, count: int = 20) -> None:
        """Generate exploration levels: hidden switches, extensive exploration.
        
        These levels test exploration strategies and discovery.
        Large mazes, multi-chamber with distant objectives.
        
        Args:
            count: Number of exploration levels to generate
        """
        print(f"Generating {count} exploration levels...")
        
        for i in range(count):
            seed = self.EXPLORATION_BASE_SEED + i
            
            # Alternate between large mazes and sprawling multi-chamber
            if i % 2 == 0:
                map_gen = self._create_exploration_maze(seed)
                desc = "Large maze requiring extensive exploration"
            else:
                map_gen = self._create_exploration_multi_chamber(seed)
                desc = "Sprawling multi-chamber with distant objectives"
            
            level_data = {
                'level_id': f'exploration_{i:03d}',
                'seed': seed,
                'category': 'exploration',
                'map_data': map_gen.map_data(),
                'metadata': {
                    'description': desc,
                    'difficulty_tier': (i // 7) + 1  # 3 tiers
                }
            }
            self.levels['exploration'].append(level_data)
        
        print(f"✓ Generated {count} exploration levels")
    
    def _create_exploration_maze(self, seed: int) -> Map:
        """Create a large maze for exploration."""
        map_gen = MazeGenerator(seed=seed)
        
        # Maximum maze dimensions for exploration
        map_gen.MIN_WIDTH = 20
        map_gen.MAX_WIDTH = 35
        map_gen.MIN_HEIGHT = 12
        map_gen.MAX_HEIGHT = 20
        
        # Add gold for exploration incentives
        map_gen.MAX_GOLD = 5
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def _create_exploration_multi_chamber(self, seed: int) -> Map:
        """Create a sprawling multi-chamber level."""
        map_gen = MultiChamberGenerator(seed=seed)
        
        # Maximum chambers
        map_gen.MIN_CHAMBERS = 4
        map_gen.MAX_CHAMBERS = 4
        
        # Larger chambers spread out
        map_gen.MIN_CHAMBER_WIDTH = 6
        map_gen.MAX_CHAMBER_WIDTH = 14
        map_gen.MIN_CHAMBER_HEIGHT = 5
        map_gen.MAX_CHAMBER_HEIGHT = 9
        
        # Long corridors for exploration feel
        map_gen.MIN_CORRIDOR_LENGTH = 3
        map_gen.MAX_CORRIDOR_LENGTH = 8
        
        map_gen.generate(seed=seed)
        return map_gen
    
    def save_dataset(self) -> None:
        """Save the generated test suite to disk."""
        print("\nSaving test suite dataset...")
        
        # Save each category separately
        for category, levels in self.levels.items():
            if not levels:
                continue
            
            category_dir = self.output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each level as a separate file
            for level in levels:
                level_file = category_dir / f"{level['level_id']}.pkl"
                with open(level_file, 'wb') as f:
                    pickle.dump(level, f)
            
            print(f"  ✓ Saved {len(levels)} {category} levels to {category_dir}")
        
        # Save metadata summary
        metadata = {
            'total_levels': sum(len(levels) for levels in self.levels.values()),
            'categories': {
                category: {
                    'count': len(levels),
                    'level_ids': [level['level_id'] for level in levels]
                }
                for category, levels in self.levels.items()
            },
            'generation_info': {
                'script_version': '1.0',
                'description': 'NPP-RL Task 3.3 comprehensive test suite',
                'deterministic': True
            }
        }
        
        metadata_file = self.output_dir / 'test_suite_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved metadata to {metadata_file}")
        print(f"\n✓ Test suite generation complete!")
        print(f"  Total levels: {metadata['total_levels']}")
        print(f"  Output directory: {self.output_dir}")
    
    def generate_all(self) -> None:
        """Generate all test suite levels."""
        print("=" * 70)
        print("NPP-RL Test Suite Generation (Task 3.3)")
        print("=" * 70)
        print()
        
        self.generate_simple_levels(50)
        self.generate_medium_levels(100)
        self.generate_complex_levels(50)
        self.generate_mine_heavy_levels(30)
        self.generate_exploration_levels(20)
        
        self.save_dataset()
        
        print()
        print("=" * 70)


def main():
    """Main entry point for test suite generation."""
    parser = argparse.ArgumentParser(
        description='Generate NPP-RL test suite maps (Task 3.3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full test suite (250 levels)
  python -m nclone.map_generation.generate_test_suite_maps --output_dir ./test_suite
  
  # Generate to custom location
  python -m nclone.map_generation.generate_test_suite_maps --output_dir /workspace/npp-rl/datasets/test_suite
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./test_suite',
        help='Output directory for generated test suite (default: ./test_suite)'
    )
    
    args = parser.parse_args()
    
    # Generate the test suite
    generator = TestSuiteGenerator(args.output_dir)
    generator.generate_all()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
