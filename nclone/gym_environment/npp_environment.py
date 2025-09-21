"""
Consolidated N++ environment class.

This environment provides a Gym interface for the N++ game, allowing reinforcement
learning agents to learn to play levels. We use a headless version of the game
to speed up training.
"""

import gymnasium
from gymnasium.spaces import box, discrete, Dict as SpacesDict
import random
import numpy as np
import os
import uuid
from typing import Tuple, Optional, Dict, Any

# Core nclone imports
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from ..constants.entity_types import EntityType
from ..constants.physics_constants import NINJA_RADIUS
from ..nplay_headless import NPlayHeadless

# Graph and level data imports
from ..graph.hierarchical_builder import HierarchicalGraphBuilder
from ..graph.level_data import LevelData
from ..graph.common import GraphData

# Reachability system imports (optional)
from ..graph.reachability.tiered_system import TieredReachabilitySystem
from ..graph.reachability.feature_extractor import (
    ReachabilityFeatureExtractor,
    PerformanceMode,
)


# Entity classes
from ..entity_classes.entity_exit_switch import EntityExitSwitch
from ..entity_classes.entity_exit import EntityExit
from ..entity_classes.entity_door_regular import EntityDoorRegular
from ..entity_classes.entity_door_locked import EntityDoorLocked
from ..entity_classes.entity_door_trap import EntityDoorTrap
from ..entity_classes.entity_one_way_platform import EntityOneWayPlatform

from .constants import (
    GAME_STATE_CHANNELS,
    TEMPORAL_FRAMES,
    PLAYER_FRAME_WIDTH,
    PLAYER_FRAME_HEIGHT,
    MAX_TIME_IN_FRAMES,
    RENDERED_VIEW_WIDTH,
    RENDERED_VIEW_HEIGHT,
)
from .reward_calculation.main_reward_calculator import (
    RewardCalculator,
)
from .observation_processor import ObservationProcessor
from .truncation_checker import TruncationChecker


class NppEnvironment(gymnasium.Env):
    """
    Consolidated N++ environment class.

    This environment provides a Gym interface for the N++ game, allowing reinforcement
    learning agents to learn to play levels. We use a headless version of the game
    to speed up training.

    Features:
    - Multiple observation profiles (minimal/rich)
    - Potential-based reward shaping (PBRS)
    - Frame stacking support
    - Graph-based planning and visualization
    - Episode truncation based on progress
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    RANDOM_MAP_CHANCE = 0.5

    def __init__(
        self,
        render_mode: str = "rgb_array",
        enable_animation: bool = False,
        enable_logging: bool = False,
        enable_debug_overlay: bool = False,
        enable_short_episode_truncation: bool = False,
        seed: Optional[int] = None,
        eval_mode: bool = False,
        enable_pbrs: bool = True,
        pbrs_weights: Optional[dict] = None,
        pbrs_gamma: float = 0.99,
        custom_map_path: Optional[str] = None,
    ):
        """
        Initialize the N++ environment.

        Args:
            render_mode: Rendering mode ("human" or "rgb_array")
            enable_animation: Enable animation in rendering
            enable_logging: Enable debug logging
            enable_debug_overlay: Enable debug overlay visualization
            enable_short_episode_truncation: Enable episode truncation on lack of progress
            seed: Random seed for reproducibility
            eval_mode: Use evaluation maps instead of training maps
            enable_pbrs: Enable potential-based reward shaping
            pbrs_weights: PBRS component weights dictionary
            pbrs_gamma: PBRS discount factor
            custom_map_path: Path to custom map file
        """
        super().__init__()

        # Store configuration
        self.render_mode = render_mode
        self.enable_animation = enable_animation
        self.enable_logging = enable_logging
        self._enable_debug_overlay = enable_debug_overlay
        self.custom_map_path = custom_map_path
        self.eval_mode = eval_mode

        # Initialize core game interface
        self.nplay_headless = NPlayHeadless(
            render_mode=render_mode,
            enable_animation=enable_animation,
            enable_logging=enable_logging,
            enable_debug_overlay=enable_debug_overlay,
            seed=seed,
        )

        # Initialize action space (6 actions: NOOP, Left, Right, Jump, Jump+Left, Jump+Right)
        self.action_space = discrete.Discrete(6)

        # Initialize RNG
        self.rng = random.Random(seed)

        # Track reward for the current episode
        self.current_ep_reward = 0

        # Initialize observation processor
        self.observation_processor = ObservationProcessor()

        # Initialize reward calculator with PBRS configuration
        self.reward_calculator = RewardCalculator(
            enable_pbrs=enable_pbrs, pbrs_weights=pbrs_weights, pbrs_gamma=pbrs_gamma
        )

        # Initialize truncation checker
        self.truncation_checker = TruncationChecker(
            self, enable_short_episode_truncation=enable_short_episode_truncation
        )

        # Initialize reachability system
        self._reachability_system = None
        self._reachability_extractor = None
        self._reachability_cache = {}
        self._reachability_cache_ttl = 0.1  # 100ms cache TTL
        self._last_reachability_time = 0

        self._reachability_system = TieredReachabilitySystem()
        self._reachability_extractor = ReachabilityFeatureExtractor()

        # Store all configuration flags for logging and debugging
        self.config_flags = {
            "render_mode": render_mode,
            "enable_animation": enable_animation,
            "enable_logging": enable_logging,
            "enable_debug_overlay": enable_debug_overlay,
            "enable_short_episode_truncation": enable_short_episode_truncation,
            "eval_mode": eval_mode,
            "enable_pbrs": enable_pbrs,
            "pbrs_weights": pbrs_weights,
            "pbrs_gamma": pbrs_gamma,
        }

        # Build observation space
        obs_spaces = {
            # Player-centered frame
            "player_frame": box.Box(
                low=0,
                high=255,
                shape=(
                    PLAYER_FRAME_HEIGHT,
                    PLAYER_FRAME_WIDTH,
                    TEMPORAL_FRAMES,
                ),
                dtype=np.uint8,
            ),
            # Global view frame
            "global_view": box.Box(
                low=0,
                high=255,
                shape=(RENDERED_VIEW_HEIGHT, RENDERED_VIEW_WIDTH, 1),
                dtype=np.uint8,
            ),
            # Game state features
            "game_state": box.Box(
                low=-1,
                high=1,
                shape=(GAME_STATE_CHANNELS,),
            ),
            # Reachability features
            "reachability_features": box.Box(
                low=-1.0, high=1.0, shape=(64,), dtype=np.float32
            ),
        }

        self.observation_space = SpacesDict(obs_spaces)

        # Graph debug visualization state
        self._graph_debug_enabled: bool = False
        self._graph_builder: Optional[HierarchicalGraphBuilder] = None
        self._graph_debug_cache: Optional[GraphData] = None
        self._exploration_debug_enabled: bool = False
        self._grid_debug_enabled: bool = False
        self._reachability_debug_enabled: bool = False
        self._reachability_state = None
        self._reachability_subgoals = []
        self._reachability_frontiers = []

        self.mirror_map = False
        self.random_map_type = None

        # Load the initial map
        self._load_initial_map()

    def _load_initial_map(self):
        """Load the first map based on configuration."""
        if self.eval_mode:
            self.current_map_name = f"eval_map_{uuid.uuid4()}"
            self.random_map_type = self.rng.choice(["JUMP_REQUIRED", "MAZE"])
            self.nplay_headless.load_random_map(self.random_map_type)
        else:
            # Load random map for training
            self.current_map_name = f"random_map_{uuid.uuid4()}"
            self.random_map_type = self.rng.choice(
                [
                    "SIMPLE_HORIZONTAL_NO_BACKTRACK",
                    "JUMP_REQUIRED",
                    "MAZE",
                ]
            )
            self.nplay_headless.load_random_map(self.random_map_type)

    def _actions_to_execute(self, action: int) -> Tuple[int, int]:
        """
        Execute the specified action using the game controller.

        Args:
            action: Action to execute (0-5)

        Returns:
            Tuple of (horizontal_input, jump_input)

        The action mapping is:
        0: NOOP - No action
        1: Left - Press 'A' key
        2: Right - Press 'D' key
        3: Jump - Press Space key
        4: Jump + Left - Press Space + 'A' keys
        5: Jump + Right - Press Space + 'D' keys
        """
        hoz_input = 0
        jump_input = 0

        if action == 0:  # NOOP
            pass
        elif action == 1:  # Left
            hoz_input = -1
        elif action == 2:  # Right
            hoz_input = 1
        elif action == 3:  # Jump
            jump_input = 1
        elif action == 4:  # Jump + Left
            jump_input = 1
            hoz_input = -1
        elif action == 5:  # Jump + Right
            jump_input = 1
            hoz_input = 1

        return hoz_input, jump_input

    def step(self, action: int):
        """Execute one environment step with enhanced episode info."""
        # Get previous observation
        prev_obs = self._get_observation()

        # Execute action
        action_hoz, action_jump = self._actions_to_execute(action)
        self.nplay_headless.tick(action_hoz, action_jump)

        # Get current observation
        curr_obs = self._get_observation()
        terminated, truncated, player_won = self._check_termination()

        # Calculate reward
        reward = self._calculate_reward(curr_obs, prev_obs)
        self.current_ep_reward += reward

        # Process observation for training
        processed_obs = self._process_observation(curr_obs)

        # Build episode info
        info = {"is_success": player_won}

        # Add configuration flags to episode info
        info.update(
            {
                "config_flags": self.config_flags.copy(),
                "pbrs_enabled": self.config_flags["enable_pbrs"],
            }
        )

        # Add PBRS component rewards if available
        if hasattr(self.reward_calculator, "last_pbrs_components"):
            info["pbrs_components"] = self.reward_calculator.last_pbrs_components.copy()

        return processed_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment with planning components and visualization."""
        # Handle reinitialization after unpickling
        if hasattr(self, "_needs_reinit") and self._needs_reinit:
            # Reinitialize components that may have been affected by pickling
            self.observation_processor.reset()
            self.reward_calculator.reset()
            self._needs_reinit = False

        # Reset observation processor
        self.observation_processor.reset()

        # Reset reward calculator
        self.reward_calculator.reset()

        # Reset truncation checker
        self.truncation_checker.reset()

        # Reset episode reward
        self.current_ep_reward = 0

        # Reset level and load map
        self.nplay_headless.reset()
        self._load_map()

        # Get initial observation and process it
        initial_obs = self._get_observation()
        processed_obs = self._process_observation(initial_obs)

        return (processed_obs, {})

    def render(self):
        """Render the environment."""
        return self.nplay_headless.render(self._debug_info())

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation from the game state."""
        # Calculate time remaining feature
        time_remaining = (
            MAX_TIME_IN_FRAMES - self.nplay_headless.sim.frame
        ) / MAX_TIME_IN_FRAMES

        ninja_state = self.nplay_headless.get_ninja_state()
        entity_states = self.nplay_headless.get_entity_states()
        game_state = np.concatenate([ninja_state, entity_states])

        # Get entity states for PBRS hazard detection
        entity_states_raw = self.nplay_headless.get_entity_states()

        obs = {
            "screen": self.render(),
            "game_state": game_state,
            "player_dead": self.nplay_headless.ninja_has_died(),
            "player_won": self.nplay_headless.ninja_has_won(),
            "player_x": self.nplay_headless.ninja_position()[0],
            "player_y": self.nplay_headless.ninja_position()[1],
            "switch_activated": self.nplay_headless.exit_switch_activated(),
            "switch_x": self.nplay_headless.exit_switch_position()[0],
            "switch_y": self.nplay_headless.exit_switch_position()[1],
            "exit_door_x": self.nplay_headless.exit_door_position()[0],
            "exit_door_y": self.nplay_headless.exit_door_position()[1],
            "time_remaining": time_remaining,
            "sim_frame": self.nplay_headless.sim.frame,
            "gold_collected": self.nplay_headless.get_gold_collected(),
            "doors_opened": self.nplay_headless.get_doors_opened(),
            "total_gold_available": self.nplay_headless.get_total_gold_available(),
            "entity_states": entity_states_raw,  # For PBRS hazard detection
            "reachability_features": self._get_reachability_features(),
        }

        return obs

    def _load_map(self):
        """Load the map specified by custom_map_path or follow default logic."""
        # If a custom map path is provided, use that instead of default behavior
        if self.custom_map_path:
            # Extract map name from path for display purposes
            map_name = os.path.basename(self.custom_map_path)
            if not map_name:  # Handle trailing slash case
                map_name = os.path.basename(os.path.dirname(self.custom_map_path))
            self.current_map_name = map_name
            self.random_map_type = None
            self.nplay_headless.load_map(self.custom_map_path)
            return

        # If we are in eval mode, load evaluation maps
        if self.eval_mode:
            # Eval mode will load a random JUMP_REQUIRED or MAZE map
            self.random_map_type = self.rng.choice(["JUMP_REQUIRED", "MAZE"])
            self.current_map_name = f"eval_map_{uuid.uuid4()}"
            self.nplay_headless.load_random_map(self.random_map_type)
            return

        # Load the test map 'doortest' for training
        # TODO: This is hardcoded for testing, should be made configurable
        self.current_map_name = "doortest"
        self.nplay_headless.load_map("nclone/test_maps/doortest")

    def _check_termination(self) -> Tuple[bool, bool, bool]:
        """
        Check if the episode should be terminated.

        Returns:
            Tuple containing:
            - terminated: True if episode should be terminated, False otherwise
            - truncated: True if episode should be truncated, False otherwise
            - player_won: True if player won, False otherwise
        """
        player_won = self.nplay_headless.ninja_has_won()
        player_dead = self.nplay_headless.ninja_has_died()
        terminated = player_won or player_dead

        # If player won, output current map name and total reward
        if player_won:
            map_to_display = (
                self.current_map_name
                if self.random_map_type is None
                else f"Random {self.random_map_type}"
            )
            print(
                f"\n---\nPlayer won on map: {map_to_display} on frame {self.nplay_headless.sim.frame}\n---\n"
            )

        # Check truncation using our truncation checker
        ninja_x, ninja_y = self.nplay_headless.ninja_position()
        should_truncate, reason = self.truncation_checker.update(ninja_x, ninja_y)
        if should_truncate and self.enable_logging:
            print(f"Episode terminated due to time: {reason}")

        # We also terminate if the truncation state is reached, that way we can
        # learn from the episode, since our time remaining is in our observation
        return terminated or should_truncate, False, player_won

    def _calculate_reward(self, curr_obs, prev_obs):
        """Calculate the reward for the environment."""
        return self.reward_calculator.calculate_reward(curr_obs, prev_obs)

    def _process_observation(self, obs):
        """Process the observation from the environment."""
        return self.observation_processor.process_observation(obs)

    def _reset_reward_calculator(self):
        """Reset the reward calculator."""
        self.reward_calculator.reset()

    def _debug_info(self) -> Optional[Dict[str, Any]]:
        """Returns a dictionary containing debug information to be displayed on the screen."""
        info: Dict[str, Any] = {}

        # Add graph visualization payload if enabled (independent of general debug overlay)
        if self._graph_debug_enabled:
            if self._graph_builder is None:
                self._graph_builder = HierarchicalGraphBuilder()
            graph_data = self._maybe_build_graph_debug()
            if graph_data is not None:
                info["graph"] = {
                    "data": graph_data,
                }

        # Add reachability visualization payload if enabled (independent of general debug overlay)
        if self._reachability_debug_enabled and self._reachability_state:
            info["reachability"] = {
                "state": self._reachability_state,
                "subgoals": self._reachability_subgoals,
                "frontiers": self._reachability_frontiers,
            }

        # Add other debug info only if general debug overlay is enabled
        if self._enable_debug_overlay:
            # Basic environment info
            ninja_x, ninja_y = self.nplay_headless.ninja_position()

            env_info = {
                "frame": self.nplay_headless.sim.frame,
                "current_ep_reward": self.current_ep_reward,
                "current_map_name": self.current_map_name,
                "ninja_position": self.nplay_headless.ninja_position(),
                "ninja_velocity": self.nplay_headless.ninja_velocity(),
            }
            info.update(env_info)

            # Add exploration debug info if enabled
            if self._exploration_debug_enabled:
                cell_x, cell_y = (
                    self.reward_calculator.exploration_calculator._get_cell_coords(
                        ninja_x, ninja_y
                    )
                )
                area_4x4_x = cell_x // 4
                area_4x4_y = cell_y // 4
                area_8x8_x = cell_x // 8
                area_8x8_y = cell_y // 8
                area_16x16_x = cell_x // 16
                area_16x16_y = cell_y // 16

                exploration_info = {
                    "current_cell": (cell_x, cell_y),
                    "current_4x4_area": (area_4x4_x, area_4x4_y),
                    "current_8x8_area": (area_8x8_x, area_8x8_y),
                    "current_16x16_area": (area_16x16_x, area_16x16_y),
                    "visited_cells": self.reward_calculator.exploration_calculator.visited_cells,
                    "visited_4x4": self.reward_calculator.exploration_calculator.visited_4x4,
                    "visited_8x8": self.reward_calculator.exploration_calculator.visited_8x8,
                    "visited_16x16": self.reward_calculator.exploration_calculator.visited_16x16,
                    "visited_cells_count": np.sum(
                        self.reward_calculator.exploration_calculator.visited_cells
                    ),
                    "visited_4x4_count": np.sum(
                        self.reward_calculator.exploration_calculator.visited_4x4
                    ),
                    "visited_8x8_count": np.sum(
                        self.reward_calculator.exploration_calculator.visited_8x8
                    ),
                    "visited_16x16_count": np.sum(
                        self.reward_calculator.exploration_calculator.visited_16x16
                    ),
                }
                info["exploration"] = exploration_info

            # Add grid outline debug info if enabled
            if self._grid_debug_enabled:
                info["grid_outline"] = True

        return info if info else None  # Return None if no debug info is to be shown

    # Graph debug visualization methods
    def set_graph_debug_enabled(self, enabled: bool):
        """Enable/disable graph debug overlay visualization."""
        self._graph_debug_enabled = bool(enabled)
        # Invalidate cache so next render rebuilds with current state
        self._graph_debug_cache = None

    def set_exploration_debug_enabled(self, enabled: bool):
        """Enable/disable exploration debug overlay visualization."""
        self._exploration_debug_enabled = bool(enabled)

    def set_grid_debug_enabled(self, enabled: bool):
        """Enable/disable grid outline debug overlay visualization."""
        self._grid_debug_enabled = bool(enabled)

    def set_reachability_debug_enabled(self, enabled: bool):
        """Enable/disable reachability analysis debug overlay visualization."""
        self._reachability_debug_enabled = bool(enabled)

    def set_reachability_data(self, reachability_state, subgoals=None, frontiers=None):
        """Set reachability analysis data for visualization."""
        self._reachability_state = reachability_state
        self._reachability_subgoals = subgoals or []
        self._reachability_frontiers = frontiers or []

    def _extract_graph_entities(self) -> list:
        """
        Extract entities for graph construction.

        This is the centralized entity extraction logic used by both
        graph observations and debug visualization to ensure consistency.

        Entity Structure:
        - Switch entities (locked/trap doors): positioned at switch locations
        - Door segment entities: positioned at door geometry centers
        - Regular doors: positioned at door centers (proximity activated)
        - Exit doors/switches: positioned at their respective locations
        - One-way platforms: positioned at platform centers

        This ensures functional edges connect switches to door segments correctly.

        Returns:
            List of entity dictionaries with type, position, and state
        """
        entities = []

        # Add ninja as a special entity node
        ninja_pos = self.nplay_headless.ninja_position()
        entities.append(
            {
                "type": EntityType.NINJA,
                "radius": NINJA_RADIUS,
                "x": ninja_pos[0],
                "y": ninja_pos[1],
                "active": True,
                "state": 0.0,
                "entity_id": "ninja",
            }
        )

        # Exit doors and switches using direct entity relationships
        try:
            # Get exit entities from entity_dic key 3 (contains both EntityExit and EntityExitSwitch)
            if hasattr(self.nplay_headless.sim, "entity_dic"):
                exit_entities = self.nplay_headless.sim.entity_dic.get(3, [])

                # Find exit switch and door pairs
                exit_switches = [
                    e for e in exit_entities if type(e).__name__ == "EntityExitSwitch"
                ]
                exit_doors = [
                    e for e in exit_entities if type(e).__name__ == "EntityExit"
                ]

                # Create switch-door pairs with matching entity IDs
                for i, switch in enumerate(exit_switches):
                    switch_id = f"exit_pair_{i}"

                    # Add exit switch entity
                    entities.append(
                        {
                            "type": EntityType.EXIT_SWITCH,
                            "radius": EntityExitSwitch.RADIUS,
                            "x": switch.xpos,
                            "y": switch.ypos,
                            "active": getattr(switch, "active", True),
                            "state": 1.0 if getattr(switch, "active", True) else 0.0,
                            "entity_id": switch_id,
                            "entity_ref": switch,
                        }
                    )

                    # Find corresponding door (usually there's one door per switch)
                    if i < len(exit_doors):
                        door = exit_doors[i]
                        entities.append(
                            {
                                "type": EntityType.EXIT_DOOR,
                                "radius": EntityExit.RADIUS,
                                "x": door.xpos,
                                "y": door.ypos,
                                "active": getattr(door, "active", True),
                                "state": 1.0 if getattr(door, "active", True) else 0.0,
                                "entity_id": f"exit_door_{i}",
                                "switch_entity_id": switch_id,  # Link to switch
                                "entity_ref": door,
                            }
                        )

            # Regular doors (proximity activated)
            for d in self.nplay_headless.regular_doors():
                segment = getattr(d, "segment", None)
                if segment:
                    door_x = (segment.x1 + segment.x2) * 0.5
                    door_y = (segment.y1 + segment.y2) * 0.5
                else:
                    door_x, door_y = d.xpos, d.ypos

                entities.append(
                    {
                        "type": EntityType.REGULAR_DOOR,
                        "radius": EntityDoorRegular.RADIUS,
                        "x": door_x,  # Regular doors use door center as entity position
                        "y": door_y,
                        "active": getattr(d, "active", True),
                        "closed": getattr(d, "closed", True),
                        "state": 0.0,
                    }
                )

            # Locked doors using direct entity access
            for i, locked_door in enumerate(self.nplay_headless.locked_doors()):
                segment = getattr(locked_door, "segment", None)
                if segment:
                    door_x = (segment.x1 + segment.x2) * 0.5
                    door_y = (segment.y1 + segment.y2) * 0.5
                else:
                    door_x, door_y = 0.0, 0.0

                entity_id = f"locked_{i}"

                # Add locked door switch part
                entities.append(
                    {
                        "type": EntityType.LOCKED_DOOR,
                        "x": locked_door.xpos,  # Switch position (where entity is positioned)
                        "y": locked_door.ypos,  # Switch position
                        "door_x": door_x,  # Door segment position
                        "door_y": door_y,  # Door segment position
                        "active": getattr(locked_door, "active", True),
                        "closed": getattr(locked_door, "closed", True),
                        "radius": EntityDoorRegular.RADIUS,
                        "state": 0.0 if getattr(locked_door, "active", True) else 1.0,
                        "entity_id": entity_id,
                        "is_door_part": False,  # This is the switch part
                        "entity_ref": locked_door,
                    }
                )

                # Add locked door door part (at door segment position)
                entities.append(
                    {
                        "type": EntityType.LOCKED_DOOR,
                        "radius": EntityDoorLocked.RADIUS,
                        "x": door_x,  # Door segment position
                        "y": door_y,  # Door segment position
                        "door_x": door_x,  # Door segment position
                        "door_y": door_y,  # Door segment position
                        "active": getattr(locked_door, "active", True),
                        "closed": getattr(locked_door, "closed", True),
                        "state": 0.0 if getattr(locked_door, "active", True) else 1.0,
                        "entity_id": entity_id,
                        "is_door_part": True,  # This is the door part
                        "entity_ref": locked_door,
                    }
                )

            # Trap doors using direct entity access
            for i, trap_door in enumerate(self.nplay_headless.trap_doors()):
                segment = getattr(trap_door, "segment", None)
                if segment:
                    door_x = (segment.x1 + segment.x2) * 0.5
                    door_y = (segment.y1 + segment.y2) * 0.5
                else:
                    door_x, door_y = 0.0, 0.0

                entity_id = f"trap_{i}"

                # Add trap door switch part
                entities.append(
                    {
                        "type": EntityType.TRAP_DOOR,
                        "radius": EntityDoorTrap.RADIUS,
                        "x": trap_door.xpos,  # Switch position (where entity is positioned)
                        "y": trap_door.ypos,  # Switch position
                        "door_x": door_x,  # Door segment position
                        "door_y": door_y,  # Door segment position
                        "active": getattr(trap_door, "active", True),
                        "closed": getattr(
                            trap_door, "closed", False
                        ),  # Trap doors start open
                        "state": 0.0 if getattr(trap_door, "active", True) else 1.0,
                        "entity_id": entity_id,
                        "is_door_part": False,  # This is the switch part
                        "entity_ref": trap_door,
                    }
                )

                # Add trap door door part (at door segment position)
                entities.append(
                    {
                        "type": EntityType.TRAP_DOOR,
                        "radius": EntityDoorTrap.RADIUS,
                        "x": door_x,  # Door segment position
                        "y": door_y,  # Door segment position
                        "door_x": door_x,  # Door segment position
                        "door_y": door_y,  # Door segment position
                        "active": getattr(trap_door, "active", True),
                        "closed": getattr(
                            trap_door, "closed", False
                        ),  # Trap doors start open
                        "state": 0.0 if getattr(trap_door, "active", True) else 1.0,
                        "entity_id": entity_id,
                        "is_door_part": True,  # This is the door part
                        "entity_ref": trap_door,
                    }
                )

            # One-way platforms
            if hasattr(self.nplay_headless.sim, "entity_dic"):
                one_ways = self.nplay_headless.sim.entity_dic.get(11, [])
                for ow in one_ways:
                    entities.append(
                        {
                            "type": EntityType.ONE_WAY,
                            "radius": EntityOneWayPlatform.SEMI_SIDE,
                            "x": getattr(ow, "xpos", 0.0),
                            "y": getattr(ow, "ypos", 0.0),
                            "orientation": getattr(ow, "orientation", 0),
                            "active": getattr(ow, "active", True),
                            "state": 0.0,
                        }
                    )
        except Exception as e:
            print("Error extracting graph entities: ", e)

        return entities

    def _get_reachability_features(self) -> Optional[np.ndarray]:
        """
        Extract compact reachability features from current game state.

        Returns:
            64-dimensional feature vector or None if reachability is disabled
        """
        import time

        current_time = time.time()

        # Check cache
        cache_key = (
            self.nplay_headless.ninja_position(),
            self.nplay_headless.sim.frame,
        )

        if (
            cache_key in self._reachability_cache
            and current_time - self._last_reachability_time
            < self._reachability_cache_ttl
        ):
            return self._reachability_cache[cache_key]

        # Extract current game state
        ninja_pos = self.nplay_headless.ninja_position()
        level_data = self.level_data()
        entities = self.entities()

        # Extract reachability features
        features = self._reachability_extractor.extract_features(
            ninja_position=ninja_pos,
            level_data=level_data,
            entities=entities,
            performance_target=PerformanceMode.TIER_1,
        )

        # Cache the result
        self._reachability_cache[cache_key] = features
        self._last_reachability_time = current_time

        # Limit cache size
        if len(self._reachability_cache) > 100:
            # Remove oldest entries
            oldest_keys = list(self._reachability_cache.keys())[:-50]
            for key in oldest_keys:
                del self._reachability_cache[key]

        return features

    def _extract_level_data(self) -> LevelData:
        """
        Extract level structure data for graph construction.

        Returns:
            LevelData object containing tiles and entities
        """
        # Build level tiles as a compact 2D array of inner playable area [23 x 42]
        tile_dic = self.nplay_headless.get_tile_data()
        tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
        # Simulator tiles include a 1-tile border; map inner (1..42, 1..23) -> (0..41, 0..22)
        for (x, y), tile_id in tile_dic.items():
            inner_x = x - 1
            inner_y = y - 1
            if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
                tiles[inner_y, inner_x] = int(tile_id)

        # Extract entities
        entities = self._extract_graph_entities()

        return LevelData(
            tiles=tiles,
            entities=entities,
            level_id=f"level_{getattr(self.nplay_headless.sim, 'frame', 0)}",
        )

    @property
    def level_data(self) -> LevelData:
        """
        Get current level data for external access.

        Returns:
            LevelData object containing tiles and entities
        """
        return self._extract_level_data()

    @property
    def entities(self) -> list:
        """
        Get current entities for external access.

        Returns:
            List of entity dictionaries
        """
        return self._extract_graph_entities()

    def _maybe_build_graph_debug(self) -> Optional[GraphData]:
        """Build GraphData for the current state, with dynamic caching that considers door states."""
        # Enhanced cache that considers door states and ninja position
        sim_frame = getattr(self.nplay_headless.sim, "frame", None)
        cached_frame = getattr(self, "_graph_debug_cached_frame", None)

        # Get current door states for cache invalidation
        current_door_states = self._get_door_states_signature()
        cached_door_states = getattr(self, "_graph_debug_cached_door_states", None)

        # Get ninja position for cache invalidation (sub-cell level precision)
        ninja_pos = self.nplay_headless.ninja_position()
        ninja_sub_cell = (
            int(ninja_pos[1] // 12),
            int(ninja_pos[0] // 12),
        )  # (sub_row, sub_col)
        cached_ninja_sub_cell = getattr(
            self, "_graph_debug_cached_ninja_sub_cell", None
        )

        # Check if cache is still valid
        cache_valid = (
            self._graph_debug_cache is not None
            and sim_frame == cached_frame
            and current_door_states == cached_door_states
            and ninja_sub_cell == cached_ninja_sub_cell
        )

        if cache_valid:
            return self._graph_debug_cache

        # Use centralized extraction logic
        level_data = self._extract_level_data()

        hierarchical_data = self._graph_builder.build_graph(level_data, ninja_pos)

        # Extract sub-cell graph for debug visualization (maintains backward compatibility)
        graph = hierarchical_data.sub_cell_graph

        # Update cache with all relevant state
        self._graph_debug_cache = graph
        setattr(self, "_graph_debug_cached_frame", sim_frame)
        setattr(self, "_graph_debug_cached_door_states", current_door_states)
        setattr(self, "_graph_debug_cached_ninja_sub_cell", ninja_sub_cell)

        return graph

    def _get_door_states_signature(self) -> Tuple:
        """
        Get a signature of current door states for cache invalidation.

        Returns:
            Tuple representing current door states
        """
        try:
            # Extract door-related entities and their states
            entities = self._extract_graph_entities()
            door_states = []

            for entity in entities:
                entity_type = entity.get("type", "")

                # Check for door entities
                if (
                    isinstance(entity_type, int) and entity_type in {3, 5, 6, 8}
                ) or any(
                    door_type in str(entity_type).lower()
                    for door_type in ["door", "switch"]
                ):
                    # Include position and state for doors/switches
                    state_tuple = (
                        entity.get("type", ""),
                        entity.get("x", 0),
                        entity.get("y", 0),
                        entity.get("active", True),
                        entity.get("closed", False),
                    )
                    door_states.append(state_tuple)

            return tuple(sorted(door_states))

        except Exception:
            # If door state extraction fails, use frame number as fallback
            return (getattr(self.nplay_headless.sim, "frame", 0),)

    def __getstate__(self):
        """Custom pickle method to handle non-picklable pygame objects."""
        state = self.__dict__.copy()

        # Remove the entire nplay_headless object as it contains pygame objects
        # It will be recreated when needed after unpickling
        if "nplay_headless" in state:
            # Store initialization parameters instead
            state["_nplay_headless_params"] = {
                "render_mode": getattr(self.nplay_headless, "render_mode", "rgb_array"),
                "enable_animation": getattr(
                    self.nplay_headless, "enable_animation", False
                ),
                "enable_logging": getattr(self.nplay_headless, "enable_logging", False),
                "enable_debug_overlay": getattr(
                    self.nplay_headless, "enable_debug_overlay", False
                ),
                "seed": getattr(self.nplay_headless, "seed", None),
            }
            del state["nplay_headless"]

        return state

    def __setstate__(self, state):
        """Custom unpickle method to restore the environment."""
        self.__dict__.update(state)

        # Recreate nplay_headless if it was removed during pickling
        if not hasattr(self, "nplay_headless") and hasattr(
            self, "_nplay_headless_params"
        ):
            from ..nplay_headless import NPlayHeadless

            # Recreate nplay_headless with stored parameters
            params = self._nplay_headless_params
            self.nplay_headless = NPlayHeadless(
                render_mode=params["render_mode"],
                enable_animation=params["enable_animation"],
                enable_logging=params["enable_logging"],
                enable_debug_overlay=params["enable_debug_overlay"],
                seed=params["seed"],
            )
            # Clean up temporary params
            delattr(self, "_nplay_headless_params")

            # Mark that we need to reinitialize on next reset
            self._needs_reinit = True
