#!/usr/bin/env python3
"""Interactive visualization tool for precomputed value functions.

This script provides multiple visualization modes to validate and analyze
precomputed value functions V*(s) for PBRS.

Usage:
    # 2D heatmap with interactive sliders
    python -m nclone.value_function.visualize_value_function \\
        --value-function value_functions/006_exit.npz \\
        --mode 2d

    # 3D surface plot
    python -m nclone.value_function.visualize_value_function \\
        --value-function value_functions/006_exit.npz \\
        --mode 3d

    # Interactive plotly visualization (recommended)
    python -m nclone.value_function.visualize_value_function \\
        --value-function value_functions/006_exit.npz \\
        --mode interactive

    # Gradient field visualization
    python -m nclone.value_function.visualize_value_function \\
        --value-function value_functions/006_exit.npz \\
        --mode gradient

    # Multi-view dashboard
    python -m nclone.value_function.visualize_value_function \\
        --value-function value_functions/006_exit.npz \\
        --mode dashboard

    # Compare discrete VI vs SL-VI value functions
    python -m nclone.value_function.visualize_value_function \\
        --value-function value_functions/006_exit.npz \\
        --compare-with value_functions/006_exit_sl.npz \\
        --mode compare
"""

import argparse
import sys
import os
from typing import Tuple, Optional, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nclone.value_function import load_value_function, StateDiscretizer


# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive mode will be limited.")


def render_tiles_to_axis(ax, tile_dic: Dict[Tuple[int, int], int], 
                        tile_size: float = 24.0, tile_color: str = "#404040",
                        alpha: float = 0.8) -> None:
    """Render level tiles to matplotlib axis using exact geometry from tile_definitions.py.
    
    Args:
        ax: Matplotlib axis to render on
        tile_dic: Dictionary mapping (x, y) grid coordinates to tile type values
        tile_size: Size of each tile in pixels (default 24)
        tile_color: Color for tiles (default dark gray)
        alpha: Transparency level
    """
    from ..tile_definitions import TILE_SEGMENT_DIAG_MAP, TILE_SEGMENT_CIRCULAR_MAP
    
    for (x, y), tile_type in tile_dic.items():
        # Skip empty tiles and glitched tiles
        if tile_type == 0 or tile_type > 33:
            continue
        
        base_x = x * tile_size
        base_y = y * tile_size
        
        # Type 1: Full solid tile
        if tile_type == 1:
            rect = mpatches.Rectangle((base_x, base_y), tile_size, tile_size,
                                      facecolor=tile_color, edgecolor=None, alpha=alpha)
            ax.add_patch(rect)
            rect.set_zorder(0)
        
        # Types 2-5: Half tiles
        elif tile_type == 2:  # Top half
            rect = mpatches.Rectangle((base_x, base_y), tile_size, tile_size / 2,
                                      facecolor=tile_color, edgecolor=None, alpha=alpha)
            ax.add_patch(rect)
            rect.set_zorder(0)
        elif tile_type == 3:  # Right half
            rect = mpatches.Rectangle((base_x + tile_size / 2, base_y), tile_size / 2, tile_size,
                                      facecolor=tile_color, edgecolor=None, alpha=alpha)
            ax.add_patch(rect)
            rect.set_zorder(0)
        elif tile_type == 4:  # Bottom half
            rect = mpatches.Rectangle((base_x, base_y + tile_size / 2), tile_size, tile_size / 2,
                                      facecolor=tile_color, edgecolor=None, alpha=alpha)
            ax.add_patch(rect)
            rect.set_zorder(0)
        elif tile_type == 5:  # Left half
            rect = mpatches.Rectangle((base_x, base_y), tile_size / 2, tile_size,
                                      facecolor=tile_color, edgecolor=None, alpha=alpha)
            ax.add_patch(rect)
            rect.set_zorder(0)
        
        # Types 6-9: 45-degree diagonal slopes (triangles)
        elif tile_type in [6, 7, 8, 9]:
            (p1x, p1y), (p2x, p2y) = TILE_SEGMENT_DIAG_MAP[tile_type]
            # Construct triangle vertices based on diagonal and tile corners
            vertices = _get_slope_polygon_vertices(tile_type, base_x, base_y, tile_size, p1x, p1y, p2x, p2y)
            polygon = mpatches.Polygon(vertices, closed=True, facecolor=tile_color,
                                      edgecolor=None, alpha=alpha)
            ax.add_patch(polygon)
            polygon.set_zorder(0)
        
        # Types 10-13: Quarter circles (convex corners)
        elif tile_type in [10, 11, 12, 13]:
            center, quadrant, is_convex = TILE_SEGMENT_CIRCULAR_MAP[tile_type]
            cx = base_x + center[0]
            cy = base_y + center[1]
            # Determine angle range based on quadrant
            theta1, theta2 = _get_circular_angle_range(quadrant)
            wedge = mpatches.Wedge((cx, cy), tile_size, theta1, theta2,
                                  facecolor=tile_color, edgecolor=None, alpha=alpha)
            ax.add_patch(wedge)
            wedge.set_zorder(0)
        
        # Types 14-17: Quarter pipes (concave corners)
        elif tile_type in [14, 15, 16, 17]:
            center, quadrant, is_convex = TILE_SEGMENT_CIRCULAR_MAP[tile_type]
            cx = base_x + center[0]
            cy = base_y + center[1]
            # For concave pipes, render the tile minus the circular arc
            # Use a polygon that traces the tile boundary minus the arc
            vertices = _get_concave_pipe_vertices(base_x, base_y, tile_size, cx, cy, tile_size, quadrant)
            polygon = mpatches.Polygon(vertices, closed=True, facecolor=tile_color,
                                      edgecolor=None, alpha=alpha)
            ax.add_patch(polygon)
            polygon.set_zorder(0)
        
        # Types 18-33: Mild and steep slopes (trapezoids/complex polygons)
        elif tile_type >= 18 and tile_type <= 33:
            (p1x, p1y), (p2x, p2y) = TILE_SEGMENT_DIAG_MAP[tile_type]
            # These are more complex slopes with diagonal + orthogonal edges
            vertices = _get_slope_polygon_vertices(tile_type, base_x, base_y, tile_size, p1x, p1y, p2x, p2y)
            polygon = mpatches.Polygon(vertices, closed=True, facecolor=tile_color,
                                      edgecolor=None, alpha=alpha)
            ax.add_patch(polygon)
            polygon.set_zorder(0)


def _get_slope_polygon_vertices(tile_type: int, base_x: float, base_y: float, 
                                tile_size: float, p1x: float, p1y: float, 
                                p2x: float, p2y: float) -> List[Tuple[float, float]]:
    """Get polygon vertices for slope tiles using diagonal endpoints."""
    # Convert tile-relative coords to world coords
    p1_world = (base_x + p1x, base_y + p1y)
    p2_world = (base_x + p2x, base_y + p2y)
    
    # Types 6-9: 45-degree slopes (simple triangles)
    if tile_type in [6, 7, 8, 9]:
        # Determine third corner based on tile type
        if tile_type == 6:  # Top-left to bottom-right
            return [p1_world, p2_world, (base_x, base_y)]
        elif tile_type == 7:  # Top-right to bottom-left
            return [p1_world, p2_world, (base_x + tile_size, base_y)]
        elif tile_type == 8:  # Bottom-left to top-right
            return [p1_world, p2_world, (base_x + tile_size, base_y + tile_size)]
        elif tile_type == 9:  # Bottom-right to top-left
            return [p1_world, p2_world, (base_x, base_y + tile_size)]
    
    # Types 18-33: Mild and steep slopes (trapezoids)
    # Need to determine which corners are part of the solid region
    corners = _get_slope_corners_for_type(tile_type, base_x, base_y, tile_size)
    # Construct polygon from diagonal + corners
    return [p1_world, p2_world] + corners


def _get_slope_corners_for_type(tile_type: int, base_x: float, base_y: float, 
                                 tile_size: float) -> List[Tuple[float, float]]:
    """Get additional corner vertices for complex slope tiles."""
    # This is a simplification; for exact rendering, we'd need to analyze TILE_SEGMENT_ORTHO_MAP
    # For now, use heuristics based on tile type ranges
    if tile_type in [18, 19, 22, 23, 26, 27, 30, 31]:  # Slopes going up/raised
        return [(base_x + tile_size, base_y + tile_size), (base_x, base_y + tile_size)]
    else:  # Slopes going down (20, 21, 24, 25, 28, 29, 32, 33)
        return [(base_x, base_y), (base_x + tile_size, base_y)]


def _get_circular_angle_range(quadrant: Tuple[int, int]) -> Tuple[float, float]:
    """Get angle range in degrees for circular arc based on quadrant direction."""
    qx, qy = quadrant
    if qx == 1 and qy == 1:  # Bottom-right quadrant
        return 0, 90
    elif qx == -1 and qy == 1:  # Bottom-left quadrant
        return 90, 180
    elif qx == -1 and qy == -1:  # Top-left quadrant
        return 180, 270
    elif qx == 1 and qy == -1:  # Top-right quadrant
        return 270, 360
    return 0, 90  # Default


def _get_concave_pipe_vertices(base_x: float, base_y: float, tile_size: float,
                               cx: float, cy: float, radius: float,
                               quadrant: Tuple[int, int]) -> List[Tuple[float, float]]:
    """Get vertices for concave pipe (quarter pipe) tiles."""
    qx, qy = quadrant
    theta1, theta2 = _get_circular_angle_range(quadrant)
    
    # Generate arc points
    angles = np.linspace(np.radians(theta1), np.radians(theta2), 20)
    arc_x = cx + radius * np.cos(angles)
    arc_y = cy + radius * np.sin(angles)
    arc_points = list(zip(arc_x, arc_y))
    
    # Tile corners (all 4 corners)
    corners = [
        (base_x, base_y),
        (base_x + tile_size, base_y),
        (base_x + tile_size, base_y + tile_size),
        (base_x, base_y + tile_size),
    ]
    
    # For concave pipe, we want the solid region outside the arc
    # Determine which corners to include based on quadrant
    # Start from arc, trace tile boundary, return to arc
    # This is complex; for simplicity, use arc + opposite corners
    if qx == -1 and qy == -1:  # Top-left pipe (hollow in top-left)
        return arc_points + [corners[1], corners[2], corners[3]]
    elif qx == 1 and qy == -1:  # Top-right pipe (hollow in top-right)
        return arc_points + [corners[2], corners[3], corners[0]]
    elif qx == 1 and qy == 1:  # Bottom-right pipe (hollow in bottom-right)
        return arc_points + [corners[3], corners[0], corners[1]]
    elif qx == -1 and qy == 1:  # Bottom-left pipe (hollow in bottom-left)
        return arc_points + [corners[0], corners[1], corners[2]]
    
    return corners  # Fallback


def render_toggle_mines_to_axis(ax, toggle_mines: List[Dict], 
                                state_colors: Optional[Dict[int, str]] = None) -> None:
    """Render toggle mine entities to matplotlib axis.
    
    Args:
        ax: Matplotlib axis to render on
        toggle_mines: List of toggle mine dicts with 'pos', 'state', 'type' keys
        state_colors: Optional mapping from state to color (default: red=toggled, green=untoggled, yellow=toggling)
    """
    from ..constants.physics_constants import TOGGLE_MINE_RADII
    
    if state_colors is None:
        state_colors = {
            0: '#FF4444',  # Toggled (deadly) - red
            1: '#44FF44',  # Untoggled (safe) - green
            2: '#FFFF44',  # Toggling - yellow
        }
    
    for mine in toggle_mines:
        x, y = mine['pos']
        state = mine['state']
        radius = TOGGLE_MINE_RADII.get(state, 4.0)
        color = state_colors.get(state, '#888888')
        
        circle = mpatches.Circle((x, y), radius, 
                                facecolor=color, edgecolor='black', 
                                linewidth=1, alpha=0.8)
        ax.add_patch(circle)
        circle.set_zorder(5)  # Render on top of tiles


class ValueFunctionVisualizer:
    """Interactive visualizer for precomputed value functions."""
    
    def __init__(self, value_function_path: str, map_path: Optional[str] = None):
        """Load value function and prepare for visualization.
        
        Args:
            value_function_path: Path to .npz file with value function
            map_path: Optional path to .map file for tile rendering
        """
        print(f"Loading value function from: {value_function_path}")
        
        # Load value function
        self.V, self.discretizer, self.goal_pos = load_value_function(value_function_path)
        self.config = self.discretizer.config
        
        # Optionally load level for tile rendering
        self.tile_dic = None
        self.entities = None
        if map_path and os.path.exists(map_path):
            print(f"Loading level geometry from: {map_path}")
            try:
                from nclone.nplay_headless import NPlayHeadless
                headless = NPlayHeadless(render_mode="grayscale_array", 
                                         enable_animation=False,
                                         enable_logging=False,
                                         enable_rendering=False)
                headless.load_map(map_path)
                headless.reset()
                self.tile_dic = headless.sim.tile_dic
                
                # Extract toggle mines (entity types 1 and 21)
                toggle_mines = []
                for entity_type in [1, 21]:
                    if entity_type in headless.sim.entity_dic:
                        for mine in headless.sim.entity_dic[entity_type]:
                            toggle_mines.append({
                                'pos': (mine.xpos, mine.ypos),
                                'state': mine.state,
                                'type': entity_type,
                            })
                
                self.entities = {
                    'switch_pos': headless.exit_switch_position(),
                    'exit_pos': headless.exit_door_position(),
                    'toggle_mines': toggle_mines,
                }
                print(f"  Loaded {len(self.tile_dic)} tiles")
                print(f"  Switch position: {self.entities['switch_pos']}")
                print(f"  Exit position: {self.entities['exit_pos']}")
                print(f"  Toggle mines: {len(toggle_mines)}")
            except Exception as e:
                print(f"  Warning: Could not load level: {e}")
        
        print(f"Value function shape: {self.V.shape}")
        print(f"Goal position: {self.goal_pos}")
        print(f"Grid bounds: x=[{self.config.x_min}, {self.config.x_max}], "
              f"y=[{self.config.y_min}, {self.config.y_max}]")
        
        # Compute statistics
        self.num_reachable = np.sum(np.isfinite(self.V))
        self.total_states = np.prod(self.V.shape)
        self.finite_values = self.V[np.isfinite(self.V)]
        
        if len(self.finite_values) > 0:
            self.v_min = np.min(self.finite_values)
            self.v_max = np.max(self.finite_values)
            self.v_mean = np.mean(self.finite_values)
        else:
            self.v_min = self.v_max = self.v_mean = 0.0
        
        print(f"\nStatistics:")
        print(f"  Reachable states: {self.num_reachable:,} / {self.total_states:,} "
              f"({100.0 * self.num_reachable / self.total_states:.2f}%)")
        print(f"  Value range: [{self.v_min:.2f}, {self.v_max:.2f}]")
        print(f"  Mean value: {self.v_mean:.2f}")
        
        if len(self.finite_values) > 0:
            percentiles = np.percentile(self.finite_values, [10, 25, 50, 75, 90, 99])
            print(f"  Percentiles: p10={percentiles[0]:.1f}, p25={percentiles[1]:.1f}, "
                  f"p50={percentiles[2]:.1f}, p75={percentiles[3]:.1f}, "
                  f"p90={percentiles[4]:.1f}, p99={percentiles[5]:.1f}")
    
    def get_2d_slice(self, vx: float = 0.0, vy: float = 0.0, jump_frames: int = 0) -> np.ndarray:
        """Extract 2D position slice at specific velocity and jump state.
        
        Args:
            vx: X velocity (px/frame)
            vy: Y velocity (px/frame)
            jump_frames: Jump duration remaining (frames)
        
        Returns:
            2D array [ny, nx] with values at (x, y) positions
        """
        # Convert to indices
        vx_idx = int(np.clip((vx - self.config.vx_min) / self.config.dvx, 0, self.config.nvx - 1))
        vy_idx = int(np.clip((vy - self.config.vy_min) / self.config.dvy, 0, self.config.nvy - 1))
        jump_idx = int(np.clip(jump_frames / self.config.djump, 0, self.config.njump - 1))
        
        # Extract slice: V[ix, iy, vx_idx, vy_idx, jump_idx]
        slice_2d = self.V[:, :, vx_idx, vy_idx, jump_idx]
        
        return slice_2d
    
    def query_value(self, x: float, y: float, vx: float = 0.0, 
                    vy: float = 0.0, jump_frames: int = 0) -> float:
        """Query the value function at a specific continuous state.
        
        Args:
            x, y: Position in pixels
            vx, vy: Velocity in pixels/frame
            jump_frames: Jump duration remaining (frames)
        
        Returns:
            V*(x, y, vx, vy, jump_frames) - the cost-to-go in frames
        """
        ix = int(np.clip((x - self.config.x_min) / self.config.dx, 0, self.config.nx - 1))
        iy = int(np.clip((y - self.config.y_min) / self.config.dy, 0, self.config.ny - 1))
        ivx = int(np.clip((vx - self.config.vx_min) / self.config.dvx, 0, self.config.nvx - 1))
        ivy = int(np.clip((vy - self.config.vy_min) / self.config.dvy, 0, self.config.nvy - 1))
        ijump = int(np.clip(jump_frames / self.config.djump, 0, self.config.njump - 1))
        return float(self.V[ix, iy, ivx, ivy, ijump])
    
    def _get_slice_color_params(self, slice_2d: np.ndarray):
        """Compute color normalization parameters for a 2D slice.
        
        Uses log normalization to handle the wide dynamic range of value
        functions (0 at goal to thousands of frames at distant positions).
        
        Returns:
            (norm, vmin_finite, vmax_finite) tuple for matplotlib
        """
        finite_vals = slice_2d[np.isfinite(slice_2d)]
        if len(finite_vals) == 0:
            return None, 0, 1
        
        vmin_f = max(float(np.min(finite_vals)), 1.0)  # Log scale needs > 0
        vmax_f = float(np.max(finite_vals))
        
        if vmax_f <= vmin_f:
            vmax_f = vmin_f + 1.0
        
        # Use log normalization for wide dynamic range
        norm = mcolors.LogNorm(vmin=vmin_f, vmax=vmax_f)
        return norm, vmin_f, vmax_f
    
    def visualize_2d_interactive(self):
        """Interactive 2D heatmap with sliders for velocity and jump state.
        
        Features:
        - Log-scale colormap to visualize wide dynamic range (0 to thousands of frames)
        - Click on any position to query V* at that point
        - Sliders for vx, vy, jump_frames
        - Tile geometry and toggle mine overlay
        """
        print("\n=== Interactive 2D Heatmap ===")
        print("Use sliders to explore different velocity/jump states")
        print("Click on the heatmap to query V* at that position")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.92)
        
        # Render tiles first (background)
        if self.tile_dic is not None:
            render_tiles_to_axis(ax, self.tile_dic, tile_size=24.0, 
                               tile_color="#303030", alpha=0.9)
        
        # Render toggle mines
        if self.entities and 'toggle_mines' in self.entities:
            render_toggle_mines_to_axis(ax, self.entities['toggle_mines'])
        
        # Initial state
        vx_init = 0.0
        vy_init = 0.0
        jump_init = 0
        
        # Get initial slice and transpose for correct orientation
        slice_2d = self.get_2d_slice(vx_init, vy_init, jump_init)
        
        # Create extent for imshow
        extent = [self.config.x_min, self.config.x_max, 
                  self.config.y_max, self.config.y_min]  # Note: y is inverted
        
        # Replace 0s with small positive value for log scale, mask inf
        slice_display = slice_2d.copy()
        slice_display[slice_display == 0] = 0.5  # Goal states visible as min color
        slice_masked = np.ma.masked_where(np.isinf(slice_display), slice_display)
        
        # Compute log-scale normalization for this slice
        norm, vmin_f, vmax_f = self._get_slice_color_params(slice_display)
        
        # Print slice statistics
        finite_in_slice = slice_2d[np.isfinite(slice_2d)]
        print(f"\nSlice stats (vx={vx_init:.1f}, vy={vy_init:.1f}, jump={jump_init}):")
        if len(finite_in_slice) > 0:
            print(f"  Finite values: {len(finite_in_slice)} / {slice_2d.size}")
            print(f"  Range: [{np.min(finite_in_slice):.1f}, {np.max(finite_in_slice):.1f}] frames")
            print(f"  Median: {np.median(finite_in_slice):.1f} frames")
        else:
            print(f"  No finite values in this slice!")
        
        # Plot heatmap with LOG scale
        im = ax.imshow(slice_masked.T, extent=extent, cmap='viridis', 
                       norm=norm,
                       interpolation='nearest', aspect='equal', alpha=0.7,
                       origin='upper')
        
        # Mark goal position from value function
        ax.plot(self.goal_pos[0], self.goal_pos[1], 'r*', markersize=20, 
                label=f'VF Goal ({self.goal_pos[0]:.0f}, {self.goal_pos[1]:.0f})')
        
        # Mark actual switch/exit positions if loaded
        if self.entities:
            if self.entities.get('switch_pos'):
                sx, sy = self.entities['switch_pos']
                ax.plot(sx, sy, 'g^', markersize=15, 
                       label=f'Switch ({sx:.0f}, {sy:.0f})')
            if self.entities.get('exit_pos'):
                ex, ey = self.entities['exit_pos']
                ax.plot(ex, ey, 'bs', markersize=12, 
                       label=f'Exit ({ex:.0f}, {ey:.0f})')
        
        ax.set_xlabel('X Position (px)')
        ax.set_ylabel('Y Position (px)')
        ax.set_title(f'Value Function V*(x, y | vx={vx_init:.1f}, vy={vy_init:.1f}, jump={jump_init})')
        ax.legend(loc='upper right')
        ax.set_xlim(self.config.x_min, self.config.x_max)
        ax.set_ylim(self.config.y_max, self.config.y_min)  # Inverted Y
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cost-to-Go (frames, log scale)')
        
        # Query marker (updated on click)
        query_marker, = ax.plot([], [], 'wo', markersize=12, markeredgecolor='red', 
                                markeredgewidth=2, zorder=10)
        
        # Text annotation for queried value
        query_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=10,
                            verticalalignment='bottom', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                            zorder=10)
        
        # Store state for callbacks
        state = {'vx': vx_init, 'vy': vy_init, 'jump': jump_init}
        
        # Create sliders
        ax_vx = plt.axes([0.1, 0.22, 0.8, 0.03])
        ax_vy = plt.axes([0.1, 0.17, 0.8, 0.03])
        ax_jump = plt.axes([0.1, 0.12, 0.8, 0.03])
        
        slider_vx = Slider(ax_vx, 'Velocity X', self.config.vx_min, self.config.vx_max, 
                           valinit=vx_init, valstep=self.config.dvx)
        slider_vy = Slider(ax_vy, 'Velocity Y', self.config.vy_min, self.config.vy_max,
                           valinit=vy_init, valstep=self.config.dvy)
        slider_jump = Slider(ax_jump, 'Jump Frames', 0, 
                            (self.config.njump - 1) * self.config.djump,
                            valinit=jump_init, valstep=self.config.djump)
        
        def update(val):
            """Update plot when sliders change."""
            vx = slider_vx.val
            vy = slider_vy.val
            jump = int(slider_jump.val)
            state['vx'] = vx
            state['vy'] = vy
            state['jump'] = jump
            
            # Get new slice and prepare for display
            slice_2d = self.get_2d_slice(vx, vy, jump)
            slice_display = slice_2d.copy()
            slice_display[slice_display == 0] = 0.5
            slice_masked = np.ma.masked_where(np.isinf(slice_display), slice_display)
            
            # Recompute color normalization for this slice
            norm, vmin_f, vmax_f = self._get_slice_color_params(slice_display)
            if norm is not None:
                im.set_norm(norm)
            
            # Update image (transposed for correct orientation)
            im.set_data(slice_masked.T)
            ax.set_title(f'Value Function V*(x, y | vx={vx:.1f}, vy={vy:.1f}, jump={jump})')
            
            # Print slice stats
            finite_in_slice = slice_2d[np.isfinite(slice_2d)]
            if len(finite_in_slice) > 0:
                print(f"Slice (vx={vx:.1f}, vy={vy:.1f}, jump={jump}): "
                      f"{len(finite_in_slice)} finite values, "
                      f"range=[{np.min(finite_in_slice):.1f}, {np.max(finite_in_slice):.1f}]")
            
            fig.canvas.draw_idle()
        
        slider_vx.on_changed(update)
        slider_vy.on_changed(update)
        slider_jump.on_changed(update)
        
        def on_click(event):
            """Handle mouse click to query value at position."""
            if event.inaxes != ax or event.xdata is None:
                return
            
            x_click = event.xdata
            y_click = event.ydata
            vx = state['vx']
            vy = state['vy']
            jump = state['jump']
            
            # Query value
            val = self.query_value(x_click, y_click, vx, vy, jump)
            
            # Update marker
            query_marker.set_data([x_click], [y_click])
            
            # Format value string
            if np.isinf(val):
                val_str = "inf (unreachable)"
            else:
                val_str = f"{val:.1f} frames"
            
            info = (f"V*({x_click:.0f}, {y_click:.0f} | vx={vx:.1f}, vy={vy:.1f}, j={jump}) = {val_str}")
            query_text.set_text(info)
            print(f"  Query: {info}")
            
            fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        # Add reset button
        ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
        button_reset = Button(ax_reset, 'Reset', hovercolor='0.975')
        
        def reset(event):
            slider_vx.reset()
            slider_vy.reset()
            slider_jump.reset()
        
        button_reset.on_clicked(reset)
        
        plt.show()
    
    def visualize_3d_surface(self, vx: float = 0.0, vy: float = 0.0, jump_frames: int = 0):
        """3D surface plot of value function."""
        print(f"\n=== 3D Surface Plot ===")
        print(f"State: vx={vx:.1f}, vy={vy:.1f}, jump={jump_frames}")
        
        # Get 2D slice
        slice_2d = self.get_2d_slice(vx, vy, jump_frames)
        
        # Create meshgrid
        x = np.linspace(self.config.x_min, self.config.x_max, self.config.nx)
        y = np.linspace(self.config.y_min, self.config.y_max, self.config.ny)
        X, Y = np.meshgrid(x, y)
        
        # Replace inf with max value for visualization
        Z = slice_2d.copy()
        Z[np.isinf(Z)] = self.v_max * 1.5
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                              vmin=self.v_min, vmax=self.v_max,
                              alpha=0.8, edgecolor='none')
        
        # Mark goal in 3D
        ax.plot([self.goal_pos[0]], [self.goal_pos[1]], [0], 
                'r*', markersize=20, label='Goal')
        
        ax.set_xlabel('X Position (px)')
        ax.set_ylabel('Y Position (px)')
        ax.set_zlabel('Cost-to-Go (frames)')
        ax.set_title(f'Value Function Surface (vx={vx:.1f}, vy={vy:.1f}, jump={jump_frames})')
        
        # Colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.show()
    
    def visualize_gradient_field(self, vx: float = 0.0, vy: float = 0.0, 
                                 jump_frames: int = 0, stride: int = 10):
        """Visualize gradient field (potential gradient) as arrow field."""
        print(f"\n=== Gradient Field ===")
        print(f"State: vx={vx:.1f}, vy={vy:.1f}, jump={jump_frames}")
        
        # Get 2D slice
        slice_2d = self.get_2d_slice(vx, vy, jump_frames)
        
        # Compute gradient
        grad_y, grad_x = np.gradient(slice_2d, self.config.dy, self.config.dx)
        
        # Create meshgrid (subsample for clarity)
        x = np.linspace(self.config.x_min, self.config.x_max, self.config.nx)
        y = np.linspace(self.config.y_min, self.config.y_max, self.config.ny)
        X, Y = np.meshgrid(x, y)
        
        # Subsample
        X_sub = X[::stride, ::stride]
        Y_sub = Y[::stride, ::stride]
        grad_x_sub = -grad_x[::stride, ::stride]  # Negative because we want direction toward goal
        grad_y_sub = -grad_y[::stride, ::stride]
        slice_sub = slice_2d[::stride, ::stride]
        
        # Only show gradients for reachable states
        mask = np.isfinite(slice_sub)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Background heatmap
        extent = [self.config.x_min, self.config.x_max, 
                  self.config.y_max, self.config.y_min]
        im = ax.imshow(slice_2d, extent=extent, cmap='viridis',
                      vmin=self.v_min, vmax=self.v_max,
                      interpolation='bilinear', aspect='equal', alpha=0.6)
        
        # Quiver plot (arrows)
        ax.quiver(X_sub[mask], Y_sub[mask], 
                 grad_x_sub[mask], grad_y_sub[mask],
                 color='red', alpha=0.7, scale=100, width=0.003)
        
        # Mark goal
        ax.plot(self.goal_pos[0], self.goal_pos[1], 'r*', markersize=20,
               label=f'Goal ({self.goal_pos[0]:.0f}, {self.goal_pos[1]:.0f})')
        
        ax.set_xlabel('X Position (px)')
        ax.set_ylabel('Y Position (px)')
        ax.set_title(f'Potential Gradient Field (vx={vx:.1f}, vy={vy:.1f}, jump={jump_frames})')
        ax.legend()
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cost-to-Go (frames)')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_interactive_plotly(self):
        """Interactive plotly visualization with multiple views."""
        if not PLOTLY_AVAILABLE:
            print("ERROR: plotly not available. Install with: pip install plotly")
            return
        
        print("\n=== Interactive Plotly Visualization ===")
        print("This will open in your web browser")
        
        # Get initial slice
        slice_2d = self.get_2d_slice(0.0, 0.0, 0)
        
        # Create meshgrid
        x = np.linspace(self.config.x_min, self.config.x_max, self.config.nx)
        y = np.linspace(self.config.y_min, self.config.y_max, self.config.ny)
        
        # Replace inf with NaN for better visualization
        Z = slice_2d.copy()
        Z[np.isinf(Z)] = np.nan
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('2D Heatmap', '3D Surface'),
            specs=[[{'type': 'heatmap'}, {'type': 'surface'}]]
        )
        
        # 2D Heatmap
        fig.add_trace(
            go.Heatmap(
                z=Z,
                x=x,
                y=y,
                colorscale='Viridis',
                zmin=self.v_min,
                zmax=self.v_max,
                colorbar=dict(title='Cost-to-Go', x=0.46),
            ),
            row=1, col=1
        )
        
        # Add goal marker
        fig.add_trace(
            go.Scatter(
                x=[self.goal_pos[0]],
                y=[self.goal_pos[1]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Goal',
            ),
            row=1, col=1
        )
        
        # 3D Surface
        X, Y = np.meshgrid(x, y)
        fig.add_trace(
            go.Surface(
                z=Z,
                x=x,
                y=y,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Cost-to-Go', x=1.0),
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Value Function Visualization (vx=0.0, vy=0.0, jump=0)',
            height=600,
            showlegend=True,
        )
        
        fig.update_xaxes(title_text='X Position (px)', row=1, col=1)
        fig.update_yaxes(title_text='Y Position (px)', row=1, col=1)
        
        fig.show()
    
    def visualize_dashboard(self):
        """Multi-view dashboard with statistics."""
        print("\n=== Multi-View Dashboard ===")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main heatmap (static state)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        # Render tiles first (background)
        if self.tile_dic is not None:
            render_tiles_to_axis(ax1, self.tile_dic, tile_size=24.0, 
                               tile_color="#303030", alpha=0.9)
        
        # Render toggle mines
        if self.entities and 'toggle_mines' in self.entities:
            render_toggle_mines_to_axis(ax1, self.entities['toggle_mines'])
        
        slice_2d = self.get_2d_slice(0.0, 0.0, 0)
        slice_display = slice_2d.copy()
        slice_display[slice_display == 0] = 0.5
        slice_masked = np.ma.masked_where(np.isinf(slice_display), slice_display)
        extent = [self.config.x_min, self.config.x_max,
                  self.config.y_max, self.config.y_min]
        norm, vmin_f, vmax_f = self._get_slice_color_params(slice_display)
        im1 = ax1.imshow(slice_masked.T, extent=extent, cmap='viridis',
                        norm=norm,
                        interpolation='nearest', aspect='equal', alpha=0.7,
                        origin='upper')
        ax1.plot(self.goal_pos[0], self.goal_pos[1], 'r*', markersize=20, 
                label=f'VF Goal ({self.goal_pos[0]:.0f}, {self.goal_pos[1]:.0f})')
        
        # Mark actual positions if loaded
        if self.entities:
            if self.entities.get('switch_pos'):
                sx, sy = self.entities['switch_pos']
                ax1.plot(sx, sy, 'g^', markersize=15, label=f'Switch ({sx:.0f}, {sy:.0f})')
            if self.entities.get('exit_pos'):
                ex, ey = self.entities['exit_pos']
                ax1.plot(ex, ey, 'bs', markersize=12, label=f'Exit ({ex:.0f}, {ey:.0f})')
        
        ax1.set_xlabel('X Position (px)')
        ax1.set_ylabel('Y Position (px)')
        ax1.set_title('Value Function (vx=0, vy=0, jump=0)')
        ax1.set_xlim(self.config.x_min, self.config.x_max)
        ax1.set_ylim(self.config.y_max, self.config.y_min)
        ax1.legend(loc='upper right')
        plt.colorbar(im1, ax=ax1, label='Cost-to-Go (frames)')
        
        # 2. Value distribution histogram
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(self.finite_values, bins=50, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Cost-to-Go (frames)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Value Distribution')
        ax2.axvline(self.v_mean, color='red', linestyle='--', label=f'Mean={self.v_mean:.1f}')
        ax2.legend()
        
        # 3. Reachability by velocity
        ax3 = fig.add_subplot(gs[1, 2])
        vx_reachability = []
        vx_values = np.linspace(self.config.vx_min, self.config.vx_max, min(20, self.config.nvx))
        for vx in vx_values:
            slice_2d = self.get_2d_slice(vx, 0.0, 0)
            reach_pct = 100.0 * np.sum(np.isfinite(slice_2d)) / slice_2d.size
            vx_reachability.append(reach_pct)
        ax3.plot(vx_values, vx_reachability, 'o-', color='green')
        ax3.set_xlabel('X Velocity (px/frame)')
        ax3.set_ylabel('Reachable States (%)')
        ax3.set_title('Reachability vs X Velocity')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cross-section at goal Y (horizontal line across X)
        ax4 = fig.add_subplot(gs[2, 0])
        goal_y_idx = int((self.goal_pos[1] - self.config.y_min) / self.config.dy)
        goal_y_idx = np.clip(goal_y_idx, 0, self.config.ny - 1)
        slice_2d = self.get_2d_slice(0.0, 0.0, 0)
        x_values = np.linspace(self.config.x_min, self.config.x_max, self.config.nx)
        # slice_2d has shape (nx, ny), so to get all X at fixed Y: [:, y_idx]
        cross_section = slice_2d[:, goal_y_idx]
        ax4.plot(x_values, cross_section, 'b-')
        ax4.axvline(self.goal_pos[0], color='red', linestyle='--', label='Goal X')
        ax4.set_xlabel('X Position (px)')
        ax4.set_ylabel('Cost-to-Go (frames)')
        ax4.set_title(f'Cross-section at Goal Y={self.goal_pos[1]:.0f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Cross-section at goal X (vertical line across Y)
        ax5 = fig.add_subplot(gs[2, 1])
        goal_x_idx = int((self.goal_pos[0] - self.config.x_min) / self.config.dx)
        goal_x_idx = np.clip(goal_x_idx, 0, self.config.nx - 1)
        y_values = np.linspace(self.config.y_min, self.config.y_max, self.config.ny)
        # slice_2d has shape (nx, ny), so to get all Y at fixed X: [x_idx, :]
        cross_section = slice_2d[goal_x_idx, :]
        ax5.plot(y_values, cross_section, 'b-')
        ax5.axvline(self.goal_pos[1], color='red', linestyle='--', label='Goal Y')
        ax5.set_xlabel('Y Position (px)')
        ax5.set_ylabel('Cost-to-Go (frames)')
        ax5.set_title(f'Cross-section at Goal X={self.goal_pos[0]:.0f}')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistics text
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        stats_text = f"""
        STATISTICS
        {'='*30}
        
        Grid Shape: {self.V.shape}
        Total States: {self.total_states:,}
        Reachable: {self.num_reachable:,} ({100.0 * self.num_reachable / self.total_states:.2f}%)
        
        Value Range:
          Min: {self.v_min:.2f} frames
          Max: {self.v_max:.2f} frames
          Mean: {self.v_mean:.2f} frames
          Std: {np.std(self.finite_values):.2f} frames
        
        Goal Position:
          X: {self.goal_pos[0]:.1f} px
          Y: {self.goal_pos[1]:.1f} px
        """
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace')
        
        plt.suptitle('Value Function Dashboard', fontsize=16, fontweight='bold')
        plt.show()


def compare_value_functions(path_a: str, path_b: str, 
                            map_path: Optional[str] = None,
                            vx: float = 0.0, vy: float = 0.0,
                            jump_frames: int = 0,
                            label_a: str = "Discrete VI",
                            label_b: str = "SL-VI") -> None:
    """Compare two value functions side-by-side with difference analysis.
    
    This is the primary verification tool for SL-VI: it shows that
    continuous-space transitions produce smoother, more informative 
    gradients within coarse grid cells compared to discrete VI.
    
    Args:
        path_a: Path to first value function .npz (typically discrete)
        path_b: Path to second value function .npz (typically SL-VI)
        map_path: Optional .map file for tile overlay
        vx, vy: Velocity slice to visualize
        jump_frames: Jump state to visualize
        label_a, label_b: Labels for the two value functions
    """
    print(f"\n{'='*60}")
    print(f"  Value Function Comparison: {label_a} vs {label_b}")
    print(f"{'='*60}")
    
    # Load both value functions
    V_a, disc_a, goal_a = load_value_function(path_a)
    V_b, disc_b, goal_b = load_value_function(path_b)
    
    print(f"\n  {label_a}: shape={V_a.shape}, goal={goal_a}")
    print(f"  {label_b}: shape={V_b.shape}, goal={goal_b}")
    
    if V_a.shape != V_b.shape:
        print("ERROR: Value functions have different shapes, cannot compare.")
        return
    
    config = disc_a.config
    
    # Extract 2D slices at the requested velocity/jump state
    vx_idx = int(np.clip((vx - config.vx_min) / config.dvx, 0, config.nvx - 1))
    vy_idx = int(np.clip((vy - config.vy_min) / config.dvy, 0, config.nvy - 1))
    jump_idx = int(np.clip(jump_frames / config.djump, 0, config.njump - 1))
    
    slice_a = V_a[:, :, vx_idx, vy_idx, jump_idx]
    slice_b = V_b[:, :, vx_idx, vy_idx, jump_idx]
    
    # --- Statistics ---
    both_finite = np.isfinite(slice_a) & np.isfinite(slice_b)
    only_a_finite = np.isfinite(slice_a) & ~np.isfinite(slice_b)
    only_b_finite = ~np.isfinite(slice_a) & np.isfinite(slice_b)
    
    print(f"\n  Slice at vx={vx:.1f}, vy={vy:.1f}, jump={jump_frames}:")
    print(f"    Finite in both:      {np.sum(both_finite):,}")
    print(f"    Finite only in {label_a}: {np.sum(only_a_finite):,}")
    print(f"    Finite only in {label_b}: {np.sum(only_b_finite):,}")
    
    if np.sum(both_finite) > 0:
        diff = np.full_like(slice_a, np.nan)
        diff[both_finite] = slice_b[both_finite] - slice_a[both_finite]
        abs_diff = np.abs(diff[both_finite])
        
        print(f"\n    Value difference ({label_b} - {label_a}) where both finite:")
        print(f"      Mean absolute diff: {np.mean(abs_diff):.3f} frames")
        print(f"      Median absolute diff: {np.median(abs_diff):.3f} frames")
        print(f"      Max absolute diff: {np.max(abs_diff):.3f} frames")
        print(f"      Std of diff: {np.std(diff[both_finite]):.3f} frames")
        pct_changed = 100.0 * np.sum(abs_diff > 0.01) / np.sum(both_finite)
        print(f"      States with |diff| > 0.01: {pct_changed:.1f}%")
    
    # --- Within-cell gradient analysis (key SL-VI metric) ---
    # For each pair of neighboring grid points, check if V changes.
    # In discrete VI with coarse grids, many neighbors have identical V.
    # SL-VI should show more variation.
    print(f"\n    Within-cell gradient analysis (the key SL-VI metric):")
    for label, slc in [(label_a, slice_a), (label_b, slice_b)]:
        finite_mask = np.isfinite(slc)
        # X-direction finite differences
        dx_diff = np.abs(slc[1:, :] - slc[:-1, :])
        dx_mask = finite_mask[1:, :] & finite_mask[:-1, :]
        # Y-direction finite differences
        dy_diff = np.abs(slc[:, 1:] - slc[:, :-1])
        dy_mask = finite_mask[:, 1:] & finite_mask[:, :-1]
        
        if np.sum(dx_mask) > 0 and np.sum(dy_mask) > 0:
            dx_vals = dx_diff[dx_mask]
            dy_vals = dy_diff[dy_mask]
            all_diffs = np.concatenate([dx_vals, dy_vals])
            nonzero_pct = 100.0 * np.sum(all_diffs > 1e-6) / len(all_diffs)
            print(f"      {label}:")
            print(f"        Neighbor pairs with non-zero gradient: {nonzero_pct:.1f}%")
            print(f"        Mean neighbor |dV|: {np.mean(all_diffs):.4f}")
            print(f"        Median neighbor |dV|: {np.median(all_diffs):.4f}")
            print(f"        Max neighbor |dV|: {np.max(all_diffs):.4f}")
    
    # --- Visualization ---
    # Load tiles for overlay
    tile_dic = None
    if map_path and os.path.exists(map_path):
        try:
            from nclone.nplay_headless import NPlayHeadless
            headless = NPlayHeadless(render_mode="grayscale_array",
                                     enable_animation=False,
                                     enable_logging=False,
                                     enable_rendering=False)
            headless.load_map(map_path)
            headless.reset()
            tile_dic = headless.sim.tile_dic
        except Exception as e:
            print(f"  Warning: Could not load level tiles: {e}")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Value Function Comparison (vx={vx:.1f}, vy={vy:.1f}, jump={jump_frames})',
                 fontsize=14, fontweight='bold')
    
    extent = [config.x_min, config.x_max, config.y_max, config.y_min]
    
    # Shared color normalization across both V* heatmaps
    all_finite = np.concatenate([
        slice_a[np.isfinite(slice_a)], 
        slice_b[np.isfinite(slice_b)]
    ])
    if len(all_finite) > 0:
        shared_vmin = max(float(np.min(all_finite)), 0.5)
        shared_vmax = float(np.max(all_finite))
        if shared_vmax <= shared_vmin:
            shared_vmax = shared_vmin + 1.0
        shared_norm = mcolors.LogNorm(vmin=shared_vmin, vmax=shared_vmax)
    else:
        shared_norm = None
    
    # Helper: prepare slice for display
    def prep(slc):
        d = slc.copy()
        d[d == 0] = 0.5
        return np.ma.masked_where(np.isinf(d), d)
    
    # Panel 1: V_a (Discrete)
    ax1 = axes[0, 0]
    if tile_dic is not None:
        render_tiles_to_axis(ax1, tile_dic, tile_size=24.0, tile_color="#303030", alpha=0.9)
    im1 = ax1.imshow(prep(slice_a).T, extent=extent, cmap='viridis',
                     norm=shared_norm, interpolation='nearest', 
                     aspect='equal', alpha=0.7, origin='upper')
    ax1.plot(goal_a[0], goal_a[1], 'r*', markersize=15)
    ax1.set_title(label_a)
    ax1.set_xlabel('X (px)')
    ax1.set_ylabel('Y (px)')
    ax1.set_xlim(config.x_min, config.x_max)
    ax1.set_ylim(config.y_max, config.y_min)
    plt.colorbar(im1, ax=ax1, label='Cost-to-Go (frames)')
    
    # Panel 2: V_b (SL-VI)
    ax2 = axes[0, 1]
    if tile_dic is not None:
        render_tiles_to_axis(ax2, tile_dic, tile_size=24.0, tile_color="#303030", alpha=0.9)
    im2 = ax2.imshow(prep(slice_b).T, extent=extent, cmap='viridis',
                     norm=shared_norm, interpolation='nearest',
                     aspect='equal', alpha=0.7, origin='upper')
    ax2.plot(goal_b[0], goal_b[1], 'r*', markersize=15)
    ax2.set_title(label_b)
    ax2.set_xlabel('X (px)')
    ax2.set_ylabel('Y (px)')
    ax2.set_xlim(config.x_min, config.x_max)
    ax2.set_ylim(config.y_max, config.y_min)
    plt.colorbar(im2, ax=ax2, label='Cost-to-Go (frames)')
    
    # Panel 3: Difference map (B - A)
    ax3 = axes[1, 0]
    if tile_dic is not None:
        render_tiles_to_axis(ax3, tile_dic, tile_size=24.0, tile_color="#303030", alpha=0.4)
    diff_map = np.full_like(slice_a, np.nan)
    diff_map[both_finite] = slice_b[both_finite] - slice_a[both_finite]
    diff_masked = np.ma.masked_where(np.isnan(diff_map), diff_map)
    if np.sum(both_finite) > 0:
        abs_max = max(abs(np.nanmin(diff_map)), abs(np.nanmax(diff_map)), 0.1)
        im3 = ax3.imshow(diff_masked.T, extent=extent, cmap='RdBu_r',
                         vmin=-abs_max, vmax=abs_max,
                         interpolation='nearest', aspect='equal', alpha=0.8,
                         origin='upper')
        plt.colorbar(im3, ax=ax3, label='Diff (frames)')
    ax3.plot(goal_a[0], goal_a[1], 'r*', markersize=15)
    ax3.set_title(f'Difference ({label_b} - {label_a})')
    ax3.set_xlabel('X (px)')
    ax3.set_ylabel('Y (px)')
    ax3.set_xlim(config.x_min, config.x_max)
    ax3.set_ylim(config.y_max, config.y_min)
    
    # Panel 4: Local gradient magnitude comparison
    ax4 = axes[1, 1]
    
    # Compute gradient magnitude for both
    def gradient_magnitude(slc):
        gy, gx = np.gradient(slc, config.dy, config.dx)
        return np.sqrt(gx**2 + gy**2)
    
    # Replace inf with nan for gradient computation
    sa_clean = slice_a.copy().astype(np.float64)
    sb_clean = slice_b.copy().astype(np.float64)
    sa_clean[np.isinf(sa_clean)] = np.nan
    sb_clean[np.isinf(sb_clean)] = np.nan
    
    grad_a = gradient_magnitude(sa_clean)
    grad_b = gradient_magnitude(sb_clean)
    
    grad_finite_a = grad_a[np.isfinite(grad_a) & np.isfinite(slice_a)]
    grad_finite_b = grad_b[np.isfinite(grad_b) & np.isfinite(slice_b)]
    
    if len(grad_finite_a) > 0 and len(grad_finite_b) > 0:
        bins = np.linspace(0, min(np.percentile(grad_finite_a, 99),
                                  np.percentile(grad_finite_b, 99)), 50)
        ax4.hist(grad_finite_a, bins=bins, alpha=0.6, label=label_a, color='steelblue')
        ax4.hist(grad_finite_b, bins=bins, alpha=0.6, label=label_b, color='coral')
        ax4.set_xlabel('|grad V| (frames/px)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Gradient Magnitude Distribution')
        ax4.legend()
        ax4.axvline(np.median(grad_finite_a), color='steelblue', linestyle='--', alpha=0.7)
        ax4.axvline(np.median(grad_finite_b), color='coral', linestyle='--', alpha=0.7)
        
        # Print gradient stats
        print(f"\n    Gradient magnitude statistics:")
        print(f"      {label_a}: median={np.median(grad_finite_a):.4f}, "
              f"mean={np.mean(grad_finite_a):.4f}")
        print(f"      {label_b}: median={np.median(grad_finite_b):.4f}, "
              f"mean={np.mean(grad_finite_b):.4f}")
    
    plt.tight_layout()
    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize precomputed value functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else None
    )
    
    parser.add_argument(
        "--value-function",
        required=True,
        help="Path to value function .npz file"
    )
    parser.add_argument(
        "--map-path",
        type=str,
        default=None,
        help="Optional path to .map file for rendering level geometry"
    )
    parser.add_argument(
        "--mode",
        choices=['2d', '3d', 'gradient', 'interactive', 'dashboard', 'query', 'compare'],
        default='2d',
        help="Visualization mode (default: 2d). 'compare' shows side-by-side diff."
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        default=None,
        help="Path to second .npz file for --mode compare (e.g. SL-VI output)"
    )
    parser.add_argument(
        "--label-a",
        type=str,
        default="Discrete VI",
        help="Label for --value-function in compare mode (default: 'Discrete VI')"
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default="SL-VI",
        help="Label for --compare-with in compare mode (default: 'SL-VI')"
    )
    parser.add_argument(
        "--pos",
        type=float,
        nargs=2,
        metavar=('X', 'Y'),
        default=None,
        help="Position (x, y) in pixels to query V* at (used with --mode query)"
    )
    parser.add_argument(
        "--vx",
        type=float,
        default=0.0,
        help="X velocity for slice (px/frame, default: 0.0)"
    )
    parser.add_argument(
        "--vy",
        type=float,
        default=0.0,
        help="Y velocity for slice (px/frame, default: 0.0)"
    )
    parser.add_argument(
        "--jump",
        type=int,
        default=0,
        help="Jump frames remaining (default: 0)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Arrow stride for gradient mode (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Check file exists
    if not os.path.exists(args.value_function):
        print(f"ERROR: File not found: {args.value_function}")
        return 1
    
    # Compare mode is handled separately (doesn't use the class)
    if args.mode == 'compare':
        if not args.compare_with:
            print("ERROR: --compare-with is required for --mode compare")
            return 1
        if not os.path.exists(args.compare_with):
            print(f"ERROR: File not found: {args.compare_with}")
            return 1
        compare_value_functions(
            path_a=args.value_function,
            path_b=args.compare_with,
            map_path=args.map_path,
            vx=args.vx, vy=args.vy, jump_frames=args.jump,
            label_a=args.label_a, label_b=args.label_b,
        )
        return 0
    
    # Create visualizer with optional map
    viz = ValueFunctionVisualizer(args.value_function, map_path=args.map_path)
    
    # Run visualization
    if args.mode == 'query':
        if args.pos is None:
            # If no position specified, query the goal position
            px, py = viz.goal_pos[0], viz.goal_pos[1]
            print(f"No --pos specified, querying goal position: ({px:.0f}, {py:.0f})")
        else:
            px, py = args.pos[0], args.pos[1]
        
        val = viz.query_value(px, py, args.vx, args.vy, args.jump)
        if np.isinf(val):
            val_str = "inf (unreachable)"
        else:
            val_str = f"{val:.1f} frames"
        print(f"\nV*({px:.0f}, {py:.0f} | vx={args.vx:.1f}, vy={args.vy:.1f}, jump={args.jump}) = {val_str}")
        
        # Also print values at neighboring velocities for context
        print(f"\nNeighboring velocity values at ({px:.0f}, {py:.0f}, jump={args.jump}):")
        print(f"  {'vx':>6} {'vy':>6} {'V*':>12}")
        print(f"  {'----':>6} {'----':>6} {'----':>12}")
        for dvx in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for dvy in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                vx_q = args.vx + dvx
                vy_q = args.vy + dvy
                v = viz.query_value(px, py, vx_q, vy_q, args.jump)
                if np.isinf(v):
                    print(f"  {vx_q:>6.1f} {vy_q:>6.1f} {'inf':>12}")
                else:
                    print(f"  {vx_q:>6.1f} {vy_q:>6.1f} {v:>12.1f}")
    elif args.mode == '2d':
        viz.visualize_2d_interactive()
    elif args.mode == '3d':
        viz.visualize_3d_surface(args.vx, args.vy, args.jump)
    elif args.mode == 'gradient':
        viz.visualize_gradient_field(args.vx, args.vy, args.jump, args.stride)
    elif args.mode == 'interactive':
        viz.visualize_interactive_plotly()
    elif args.mode == 'dashboard':
        viz.visualize_dashboard()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
