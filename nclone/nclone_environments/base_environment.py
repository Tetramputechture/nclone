"""Base environment class that handles logic for loading maps, resetting the game, and rendering."""

import gymnasium
from gymnasium.spaces import discrete
import random
from typing import Tuple, Optional, Dict, Any
import numpy as np

from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT, TILE_PIXEL_SIZE
from ..graph.graph_builder import GraphBuilder, GraphData

from ..nplay_headless import NPlayHeadless

class BaseEnvironment(gymnasium.Env):
    """Base environment class that handles logic for loading maps, resetting the game, and rendering."""
    metadata = {'render.modes': ['human', 'rgb_array']}

    MAP_DATA_PATH = None

    def __init__(self,
                 render_mode: str = 'rgb_array',
                 enable_animation: bool = False,
                 enable_logging: bool = False,
                 enable_debug_overlay: bool = False,
                 seed: Optional[int] = None):
        """Initialize the environment."""
        super().__init__()

        self.render_mode = render_mode
        self.enable_animation = enable_animation
        self.enable_logging = enable_logging
        self._enable_debug_overlay = enable_debug_overlay
        self.nplay_headless = NPlayHeadless(
            render_mode=render_mode,
            enable_animation=enable_animation,
            enable_logging=enable_logging,
            enable_debug_overlay=enable_debug_overlay,
            seed=seed)

        # Initialize action space
        self.action_space = discrete.Discrete(6)

        # Initialize RNG
        self.rng = random.Random(seed)

        # Track reward for the current episode
        self.current_ep_reward = 0
        
        # Placeholder for pathfinding data
        self.pathfinding_data: Optional[Dict[str, Any]] = None

        # Graph debug visualization state
        self._graph_debug_enabled: bool = False
        self._graph_builder: Optional[GraphBuilder] = None
        self._graph_debug_cache: Optional[GraphData] = None
        self._exploration_debug_enabled: bool = False
        self._grid_debug_enabled: bool = False

    def _actions_to_execute(self, action: int) -> Tuple[int, int]:
        """Execute the specified action using the game controller.

        Args:
            action (int): Action to execute (0-5)

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

        # Execute the new action
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
        """Execute one environment step with planning and visualization."""
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

        ep_info = {'is_success': player_won}

        return processed_obs, reward, terminated, truncated, ep_info

    def reset(self, seed=None, options=None):
        """Reset the environment with planning components and visualization."""
        # Reset observation processor
        self._reset_observation_processor()

        # Reset reward calculator
        self._reset_reward_calculator()

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

    def _debug_info(self) -> Optional[Dict[str, Any]]:
        """Returns a dictionary containing debug information to be displayed on the screen."""
        if not self._enable_debug_overlay:
            return None
        
        info: Dict[str, Any] = {}
        if self.pathfinding_data:
            info['pathfinding'] = self.pathfinding_data

        # Add graph visualization payload if enabled
        if self._graph_debug_enabled:
            if self._graph_builder is None:
                self._graph_builder = GraphBuilder()
            graph_data = self._maybe_build_graph_debug()
            if graph_data is not None:
                info['graph'] = {
                    'data': graph_data,
                    'cell_size': TILE_PIXEL_SIZE,
                    'grid_width': MAP_TILE_WIDTH,
                    'grid_height': MAP_TILE_HEIGHT,
                }

        # Add grid outline debug info if enabled
        if self._grid_debug_enabled:
            info['grid_outline'] = {
                'enabled': True,
                'width': MAP_TILE_WIDTH,
                'height': MAP_TILE_HEIGHT,
                'cell_size': TILE_PIXEL_SIZE,
            }
        
        # Allow subclasses to add more debug info
        # For example, agent-specific state or exploration data
        # Example: info['agent_state'] = {'foo': 'bar'}
        
        return info if info else None # Return None if no debug info is to be shown

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
        
        # Exit switch
        try:
            sw_x, sw_y = self.nplay_headless.exit_switch_position()
            entities.append({
                'type': 'exit_switch',
                'x': sw_x,
                'y': sw_y,
                'active': self.nplay_headless.exit_switch_activated(),
                'state': 1.0 if self.nplay_headless.exit_switch_activated() else 0.0,
            })
            
            # Exit door
            door_x, door_y = self.nplay_headless.exit_door_position()
            entities.append({
                'type': 'exit_door',
                'x': door_x,
                'y': door_y,
                'active': True,
                'state': 0.0,
            })
            
            # Regular doors (proximity activated)
            for d in self.nplay_headless.regular_doors():
                segment = getattr(d, 'segment', None)
                if segment:
                    door_x = (segment.x1 + segment.x2) * 0.5
                    door_y = (segment.y1 + segment.y2) * 0.5
                else:
                    door_x, door_y = d.xpos, d.ypos
                    
                entities.append({
                    'type': 5,
                    'x': door_x,  # Regular doors use door center as entity position
                    'y': door_y,
                    'active': getattr(d, 'active', True),
                    'closed': getattr(d, 'closed', True),
                    'state': 0.0
                })

            # Locked doors (have switch positions)
            for d in self.nplay_headless.locked_doors():
                segment = getattr(d, 'segment', None)
                if segment:
                    door_x = (segment.x1 + segment.x2) * 0.5
                    door_y = (segment.y1 + segment.y2) * 0.5
                else:
                    door_x, door_y = 0.0, 0.0
                
                # Add switch entity at switch location
                entities.append({
                    'type': 6,
                    'x': d.xpos,  # Switch position
                    'y': d.ypos,  # Switch position
                    'active': getattr(d, 'active', True),
                    'sw_xpos': getattr(d, 'sw_xpos', None),
                    'sw_ypos': getattr(d, 'sw_ypos', None),
                    'door_x': door_x,
                    'door_y': door_y,
                    'radius': getattr(d, 'RADIUS', 5.0),
                    'closed': getattr(d, 'closed', True),
                    'state': 0.0
                })
                
                # Add separate door segment entity at door location
                entities.append({
                    'type': 'door_segment_locked',
                    'x': door_x,
                    'y': door_y,
                    'active': getattr(d, 'active', True),
                    'closed': getattr(d, 'closed', True),
                    'parent_switch_x': d.xpos,
                    'parent_switch_y': d.ypos,
                    'state': 0.0
                })

            # Trap doors (close when switch is hit)
            for d in self.nplay_headless.trap_doors():
                segment = getattr(d, 'segment', None)
                if segment:
                    door_x = (segment.x1 + segment.x2) * 0.5
                    door_y = (segment.y1 + segment.y2) * 0.5
                else:
                    door_x, door_y = 0.0, 0.0
                
                # Add switch entity at switch location
                entities.append({
                    'type': 8,
                    'x': d.xpos,  # Switch position
                    'y': d.ypos,  # Switch position
                    'active': getattr(d, 'active', True),
                    'sw_xpos': getattr(d, 'sw_xpos', None),
                    'sw_ypos': getattr(d, 'sw_ypos', None),
                    'door_x': door_x,
                    'door_y': door_y,
                    'radius': getattr(d, 'RADIUS', 5.0),
                    'closed': getattr(d, 'closed', False),
                    'state': 0.0
                })
                
                # Add separate door segment entity at door location
                entities.append({
                    'type': 'door_segment_trap',
                    'x': door_x,
                    'y': door_y,
                    'active': getattr(d, 'active', True),
                    'closed': getattr(d, 'closed', False),
                    'parent_switch_x': d.xpos,
                    'parent_switch_y': d.ypos,
                    'state': 0.0
                })

            # One-way platforms
            if hasattr(self.nplay_headless.sim, 'entity_dic'):
                one_ways = self.nplay_headless.sim.entity_dic.get(11, [])
                for ow in one_ways:
                    entities.append({
                        'type': 11,
                        'x': getattr(ow, 'xpos', 0.0),
                        'y': getattr(ow, 'ypos', 0.0),
                        'orientation': getattr(ow, 'orientation', 0),
                        'active': getattr(ow, 'active', True),
                        'state': 0.0
                    })
        except Exception as e:
            print('Error extracting graph entities: ', e)
            
        return entities

    def _extract_level_data(self) -> Dict[str, Any]:
        """
        Extract level structure data for graph construction.
        
        Returns:
            Dictionary containing level tile and structure information
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

        return {
            'tiles': tiles,
        }

    def _maybe_build_graph_debug(self) -> Optional[GraphData]:
        """Build GraphData for the current state, with dynamic caching that considers door states."""
        # Enhanced cache that considers door states and ninja position
        sim_frame = getattr(self.nplay_headless.sim, 'frame', None)
        cached_frame = getattr(self, '_graph_debug_cached_frame', None)
        
        # Get current door states for cache invalidation
        current_door_states = self._get_door_states_signature()
        cached_door_states = getattr(self, '_graph_debug_cached_door_states', None)
        
        # Get ninja position for cache invalidation (sub-cell level precision)
        ninja_pos = self.nplay_headless.ninja_position()
        ninja_sub_cell = (int(ninja_pos[1] // 12), int(ninja_pos[0] // 12))  # (sub_row, sub_col)
        cached_ninja_sub_cell = getattr(self, '_graph_debug_cached_ninja_sub_cell', None)
        
        # Check if cache is still valid
        cache_valid = (
            self._graph_debug_cache is not None and
            sim_frame == cached_frame and
            current_door_states == cached_door_states and
            ninja_sub_cell == cached_ninja_sub_cell
        )
        
        if cache_valid:
            return self._graph_debug_cache

        # Use centralized extraction logic
        level_data = self._extract_level_data()
        entities = self._extract_graph_entities()
        
        graph = self._graph_builder.build_graph(level_data, ninja_pos, entities)
        
        # Update cache with all relevant state
        self._graph_debug_cache = graph
        setattr(self, '_graph_debug_cached_frame', sim_frame)
        setattr(self, '_graph_debug_cached_door_states', current_door_states)
        setattr(self, '_graph_debug_cached_ninja_sub_cell', ninja_sub_cell)
        
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
                entity_type = entity.get('type', '')
                
                # Check for door entities
                if (isinstance(entity_type, int) and entity_type in {3, 5, 6, 8}) or \
                   any(door_type in str(entity_type).lower() for door_type in ['door', 'switch']):
                    # Include position and state for doors/switches
                    state_tuple = (
                        entity.get('type', ''),
                        entity.get('x', 0),
                        entity.get('y', 0),
                        entity.get('active', True),
                        entity.get('closed', False)
                    )
                    door_states.append(state_tuple)
            
            return tuple(sorted(door_states))
            
        except Exception:
            # If door state extraction fails, use frame number as fallback
            return (getattr(self.nplay_headless.sim, 'frame', 0),)

    def set_pathfinding_data(self, data: Optional[Dict[str, Any]]):
        """Allows setting pathfinding data to be used by the visualizer via _debug_info."""
        self.pathfinding_data = data

    def _load_map(self):
        """Loads the map."""
        raise NotImplementedError

    def _get_observation(self):
        """Gets the observation from the environment. Unique per each environment, must be implemented in child class."""
        raise NotImplementedError

    def _check_termination(self):
        """Checks if the environment has terminated. Unique per each environment, must be implemented in child class."""
        raise NotImplementedError

    def _calculate_reward(self, curr_obs, prev_obs):
        """Calculates the reward for the environment. Unique per each environment, must be implemented in child class."""
        raise NotImplementedError

    def _process_observation(self, obs):
        """Processes the observation from the environment. Unique per each environment, must be implemented in child class."""
        raise NotImplementedError

    def _reset_observation_processor(self):
        """Resets the observation processor. Unique per each environment, must be implemented in child class."""
        raise NotImplementedError

    def _reset_reward_calculator(self):
        """Resets the reward calculator. Unique per each environment, must be implemented in child class."""
        raise NotImplementedError
