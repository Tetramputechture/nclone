"""
Simple visualization components for the enhanced debug overlay.

This module provides basic visualization functionality to support the debug overlay.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""
    
    # Display settings
    show_nodes: bool = True
    show_edges: bool = True
    show_labels: bool = True
    
    # Color settings
    node_color: str = '#4CAF50'
    edge_color: str = '#2196F3'
    text_color: str = '#000000'
    
    # Size settings
    node_size: int = 10
    edge_width: int = 2
    font_size: int = 8
    
    # Layout settings
    width: int = 800
    height: int = 600
    margin: int = 50


class GraphVisualizer:
    """
    Simple graph visualizer for debug overlay.
    
    This is a basic implementation to support the enhanced debug overlay functionality.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """
        Initialize graph visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.cached_layout = {}
        
    def render_graph(
        self, 
        graph_data, 
        highlight_nodes: Optional[List[int]] = None,
        highlight_edges: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, Any]:
        """
        Render graph visualization data.
        
        Args:
            graph_data: Graph data structure
            highlight_nodes: Nodes to highlight
            highlight_edges: Edges to highlight
            
        Returns:
            Dictionary containing visualization data
        """
        try:
            # Extract basic graph information
            nodes = self._extract_nodes(graph_data)
            edges = self._extract_edges(graph_data)
            
            # Create visualization data
            viz_data = {
                'nodes': nodes,
                'edges': edges,
                'config': self.config,
                'highlights': {
                    'nodes': highlight_nodes or [],
                    'edges': highlight_edges or []
                }
            }
            
            return viz_data
            
        except Exception as e:
            # Return minimal visualization data on error
            return {
                'nodes': [],
                'edges': [],
                'config': self.config,
                'highlights': {'nodes': [], 'edges': []},
                'error': str(e)
            }
    
    def _extract_nodes(self, graph_data) -> List[Dict[str, Any]]:
        """Extract node information from graph data."""
        nodes = []
        
        try:
            if hasattr(graph_data, 'nodes'):
                for i, node in enumerate(graph_data.nodes):
                    node_info = {
                        'id': i,
                        'position': getattr(node, 'position', (0, 0)),
                        'type': getattr(node, 'node_type', 'unknown'),
                        'label': f"Node {i}"
                    }
                    nodes.append(node_info)
            elif isinstance(graph_data, dict) and 'nodes' in graph_data:
                for i, node in enumerate(graph_data['nodes']):
                    node_info = {
                        'id': i,
                        'position': node.get('position', (0, 0)),
                        'type': node.get('type', 'unknown'),
                        'label': f"Node {i}"
                    }
                    nodes.append(node_info)
        except Exception:
            # Return empty list on error
            pass
            
        return nodes
    
    def _extract_edges(self, graph_data) -> List[Dict[str, Any]]:
        """Extract edge information from graph data."""
        edges = []
        
        try:
            if hasattr(graph_data, 'edges'):
                for i, edge in enumerate(graph_data.edges):
                    edge_info = {
                        'id': i,
                        'source': getattr(edge, 'source', 0),
                        'target': getattr(edge, 'target', 0),
                        'type': getattr(edge, 'edge_type', 'unknown'),
                        'weight': getattr(edge, 'weight', 1.0)
                    }
                    edges.append(edge_info)
            elif isinstance(graph_data, dict) and 'edges' in graph_data:
                for i, edge in enumerate(graph_data['edges']):
                    edge_info = {
                        'id': i,
                        'source': edge.get('source', 0),
                        'target': edge.get('target', 0),
                        'type': edge.get('type', 'unknown'),
                        'weight': edge.get('weight', 1.0)
                    }
                    edges.append(edge_info)
        except Exception:
            # Return empty list on error
            pass
            
        return edges
    
    def update_config(self, **kwargs):
        """Update visualization configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def clear_cache(self):
        """Clear cached layout data."""
        self.cached_layout = {}