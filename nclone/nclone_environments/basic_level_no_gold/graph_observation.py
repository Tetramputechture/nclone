"""
Graph observation module for BasicLevelNoGold environment.

This module extends the environment to provide graph-based structural observations
alongside the existing visual and symbolic observations.
"""

import numpy as np
from typing import Dict, Any, Tuple
from gymnasium.spaces import Box, Dict as SpacesDict

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import GraphData, N_MAX_NODES, E_MAX_EDGES


class GraphObservationMixin:
    """
    Mixin class to add graph observation capabilities to environments.
    
    This mixin can be added to existing environment classes to provide
    graph-based structural observations without modifying the core environment.
    """
    
    def __init__(self, *args, use_graph_obs: bool = False, **kwargs):
        """
        Initialize graph observation capabilities.
        
        Args:
            use_graph_obs: Whether to include graph observations
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)
        
        self.use_graph_obs = use_graph_obs
        self.graph_builder = HierarchicalGraphBuilder() if use_graph_obs else None
        
        # Cache for graph data to avoid recomputation
        self._graph_cache = None
        self._last_graph_state = None
        
        # Extend observation space if using graph observations
        if self.use_graph_obs:
            self._extend_observation_space()
    
    def _extend_observation_space(self):
        """Extend the observation space to include graph observations."""
        if not hasattr(self, 'observation_space') or not isinstance(self.observation_space, SpacesDict):
            raise ValueError("Graph observations require Dict observation space")
        
        # Add graph observation spaces
        graph_spaces = {
            'graph_node_feats': Box(
                low=-1.0,
                high=1.0,
                shape=(N_MAX_NODES, self.graph_builder.node_feature_dim),
                dtype=np.float32
            ),
            'graph_edge_index': Box(
                low=0,
                high=N_MAX_NODES - 1,
                shape=(2, E_MAX_EDGES),
                dtype=np.int32
            ),
            'graph_edge_feats': Box(
                low=-1.0,
                high=1.0,
                shape=(E_MAX_EDGES, self.graph_builder.edge_feature_dim),
                dtype=np.float32
            ),
            'graph_node_mask': Box(
                low=0.0,
                high=1.0,
                shape=(N_MAX_NODES,),
                dtype=np.float32
            ),
            'graph_edge_mask': Box(
                low=0.0,
                high=1.0,
                shape=(E_MAX_EDGES,),
                dtype=np.float32
            )
        }
        
        # Create new observation space with graph components
        new_spaces = dict(self.observation_space.spaces)
        new_spaces.update(graph_spaces)
        self.observation_space = SpacesDict(new_spaces)
    
    def _get_graph_observation(self) -> Dict[str, np.ndarray]:
        """
        Get graph-based observation of the current environment state.
        
        Returns:
            Dictionary containing graph observation components
        """
        if not self.use_graph_obs:
            return {}
        
        # Check if we need to recompute graph (state changed)
        current_state = self._get_graph_state_signature()
        if self._graph_cache is None or current_state != self._last_graph_state:
            self._graph_cache = self._build_current_graph()
            self._last_graph_state = current_state
        
        return {
            'graph_node_feats': self._graph_cache.node_features,
            'graph_edge_index': self._graph_cache.edge_index,
            'graph_edge_feats': self._graph_cache.edge_features,
            'graph_node_mask': self._graph_cache.node_mask,
            'graph_edge_mask': self._graph_cache.edge_mask
        }
    
    def _get_graph_state_signature(self) -> Tuple:
        """
        Get a signature of the current state for graph caching.
        
        Returns:
            Tuple representing the current state for caching purposes
        """
        # Include ninja position and simulator frame for caching
        ninja_pos = self.nplay_headless.ninja_position() if hasattr(self, 'nplay_headless') else (0, 0)
        sim_frame = getattr(self.nplay_headless.sim, 'frame', None) if hasattr(self, 'nplay_headless') else None
        
        # Simple signature based on frame and ninja position
        # The base environment's entity extraction logic handles the complexity
        return (ninja_pos, sim_frame)
    
    def _build_current_graph(self) -> GraphData:
        """
        Build graph representation of the current environment state.
        
        Returns:
            GraphData containing the current level structure
        """
        # Get current game state
        ninja_pos = self.nplay_headless.ninja_position() if hasattr(self, 'nplay_headless') else (0, 0)
        
        # Extract level data (mock implementation - would use actual level data)
        level_data = self._extract_level_data()
        
        # Extract entity data
        entities = self._extract_entity_data()
        
        # Build graph
        return self.graph_builder.build_graph(level_data, ninja_pos, entities)
    
    def _extract_level_data(self) -> Dict[str, Any]:
        """
        Extract level structure data for graph construction.
        
        Returns:
            Dictionary containing level tile and structure information
        """
        # Delegate to base environment's centralized extraction logic
        if hasattr(super(), '_extract_level_data'):
            return super()._extract_level_data()
        
        # Fallback implementation for environments that don't inherit from BaseEnvironment
        return {
            'tiles': np.zeros((23, 42), dtype=np.int32),  # Mock tile grid
        }
    
    def _extract_entity_data(self) -> list:
        """
        Extract entity data for graph construction.
        
        Returns:
            List of entity dictionaries with type, position, and state
        """
        # Delegate to base environment's centralized extraction logic
        if hasattr(super(), '_extract_graph_entities'):
            return super()._extract_graph_entities()
        
        # Fallback implementation for environments that don't inherit from BaseEnvironment
        entities = []
        
        if hasattr(self, 'nplay_headless'):
            # Basic entity extraction (minimal fallback)
            try:
                switch_pos = self.nplay_headless.exit_switch_position()
                entities.append({
                    'type': 'exit_switch',
                    'x': switch_pos[0],
                    'y': switch_pos[1],
                    'active': self.nplay_headless.exit_switch_activated(),
                    'state': 1.0 if self.nplay_headless.exit_switch_activated() else 0.0
                })
            except Exception:
                pass
                
            try:
                door_pos = self.nplay_headless.exit_door_position()
                entities.append({
                    'type': 'exit_door',
                    'x': door_pos[0],
                    'y': door_pos[1],
                    'active': True,
                    'state': 0.0
                })
            except Exception:
                pass
        
        return entities
    
    def reset(self, **kwargs):
        """Reset environment and clear graph cache."""
        result = super().reset(**kwargs)
        
        # Clear graph cache on reset
        self._graph_cache = None
        self._last_graph_state = None
        
        return result


def add_graph_observation_to_env(env_class):
    """
    Decorator to add graph observation capabilities to an environment class.
    
    Args:
        env_class: Environment class to extend
        
    Returns:
        Extended environment class with graph observation capabilities
    """
    class GraphObservationEnv(GraphObservationMixin, env_class):
        """Environment with graph observation capabilities."""
        
        def _get_observation(self):
            """Get observation including graph data if enabled."""
            # Get base observation
            base_obs = super()._get_observation()
            
            # Add graph observation if enabled
            if self.use_graph_obs:
                graph_obs = self._get_graph_observation()
                
                # Merge observations
                if isinstance(base_obs, dict):
                    obs = base_obs.copy()
                    obs.update(graph_obs)
                    return obs
                else:
                    # If base observation is not a dict, we need to restructure
                    obs = {'base_obs': base_obs}
                    obs.update(graph_obs)
                    return obs
            
            return base_obs
    
    return GraphObservationEnv


# Example usage: Create enhanced BasicLevelNoGold with graph observations
def create_graph_enhanced_env(**kwargs):
    """
    Create BasicLevelNoGold environment with graph observation capabilities.
    
    Args:
        **kwargs: Environment configuration including use_graph_obs
        
    Returns:
        Enhanced environment instance
    """
    from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
    
    # Create enhanced environment class
    GraphEnhancedEnv = add_graph_observation_to_env(BasicLevelNoGold)
    
    # Return instance
    return GraphEnhancedEnv(**kwargs)