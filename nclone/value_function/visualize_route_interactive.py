"""Interactive pygame validator for the route-based PBRS.

Loads a precomputed ``*_surface_pbrs.npz`` and an optional ``.map`` level file,
then opens a resizable pygame window where you can:

- Move a cursor around the level (arrow keys or mouse click).
- The cursor snaps to the nearest reachable floor surface node.
- The optimal route from that node is computed and drawn.
- Toggle overlays to inspect every graph arc, the cost-to-go heatmap, the
  planned route arcs, and the 10-dim arc-target observation features.

Usage::

    python -m nclone.value_function.visualize_route_interactive \\
        --data value_functions/basics_surface_pbrs.npz \\
        --map-path test-single-level/basics

Key bindings
------------
Arrow keys / mouse-left : move cursor
H                       : toggle d_star heatmap on surface nodes
E                       : toggle all graph arcs (edges)
R                       : toggle route arcs from cursor
F                       : toggle arc-target features panel
T                       : toggle tile rendering
Space                   : switch phase (switch-goal ↔ exit-goal)
Escape / Q              : quit
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TILE_PX = 24  # pixels per tile in the game's physics space
_DEFAULT_SCALE = 1.5  # initial zoom factor
_FPS = 60
_CURSOR_STEP = 6  # pixels per arrow-key press
_COL_CURSOR = (255, 255, 255, 255)
_COL_SNAP = (255, 255, 0, 200)  # snapped surface node ring
_COL_BG = (30, 30, 30)
_COL_TILE = (80, 90, 100)
_COL_TILE_BORDER = (50, 60, 70)
_COL_SWITCH = (255, 220, 50, 230)
_COL_EXIT = (50, 255, 50, 230)


# ---------------------------------------------------------------------------
# RouteVisualizer
# ---------------------------------------------------------------------------


class RouteVisualizer:
    """Interactive pygame window for route PBRS validation."""

    def __init__(
        self,
        npz_path: str,
        map_path: Optional[str] = None,
        scale: float = _DEFAULT_SCALE,
        initial_phase: str = "switch",
    ) -> None:
        import pygame

        self._scale = scale
        self._phase = initial_phase  # "switch" or "exit"

        # --- Load RoutePBRS ---
        self._pbrs, self._graph = _load_route_pbrs(npz_path)

        # --- Precompute overlay caches ---
        self._surface_pts_xy, self._d_star_switch_pt, self._d_star_exit_pt = (
            _build_surface_point_arrays(self._graph, self._pbrs)
        )
        self._edge_data = _build_edge_data(self._graph)

        # --- Goal positions from graph meta ---
        # loaded from the graph's d_star arrays (fallback: centre of level)
        self._switch_pos: Optional[Tuple[float, float]] = None
        self._exit_pos: Optional[Tuple[float, float]] = None

        # --- Load tile_dic (optional) ---
        self._tile_dic: Dict = {}
        if map_path is not None:
            try:
                from .physics_step_wrapper import PhysicsStepWrapper

                psw = PhysicsStepWrapper(map_path=map_path)
                self._tile_dic = dict(psw.tile_dic)
                # Extract goal positions from entity_dic
                try:
                    self._switch_pos = psw.get_goal_position("switch")
                    self._exit_pos = psw.get_goal_position("exit")
                except (ValueError, Exception):
                    pass
            except Exception as exc:
                print(f"Warning: could not load map tiles: {exc}")

        # Level bounds in game-pixel space (44 × 25 tiles including border)
        self._level_w_px = 44 * _TILE_PX
        self._level_h_px = 25 * _TILE_PX

        screen_w = int(self._level_w_px * scale)
        screen_h = int(self._level_h_px * scale)

        pygame.init()
        pygame.font.init()
        self._screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption(
            "Route PBRS Inspector — H:heatmap  E:arcs  R:route  F:features  T:tiles  Space:phase  Q:quit"
        )
        self._clock = pygame.time.Clock()

        # Cursor position in game-pixel space
        self._cx = self._level_w_px / 2.0
        self._cy = self._level_h_px / 2.0

        # Snapped surface point
        self._snapped_node: Optional[int] = None
        self._snapped_sp_xy: Optional[Tuple[float, float]] = None
        self._current_route: List[Dict] = []
        self._current_d_star: float = float("inf")
        self._current_potential: float = 0.0
        self._current_features: np.ndarray = np.zeros(10, np.float32)

        # Airborne-cursor state
        self._is_airborne: bool = False
        self._ballistic_pts: List[Tuple[float, float]] = []

        # Overlay toggles
        self._show_tiles = bool(self._tile_dic)
        self._show_heatmap = True
        self._show_arcs = False
        self._show_route = True
        self._show_features = True

        self._update_route()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        import pygame

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_h:
                        self._show_heatmap = not self._show_heatmap
                    elif event.key == pygame.K_e:
                        self._show_arcs = not self._show_arcs
                    elif event.key == pygame.K_r:
                        self._show_route = not self._show_route
                    elif event.key == pygame.K_f:
                        self._show_features = not self._show_features
                    elif event.key == pygame.K_t:
                        self._show_tiles = not self._show_tiles
                    elif event.key == pygame.K_SPACE:
                        self._phase = "exit" if self._phase == "switch" else "switch"
                        self._update_route()
                    elif event.key == pygame.K_LEFT:
                        self._cx = max(0.0, self._cx - _CURSOR_STEP)
                        self._update_route()
                    elif event.key == pygame.K_RIGHT:
                        self._cx = min(float(self._level_w_px), self._cx + _CURSOR_STEP)
                        self._update_route()
                    elif event.key == pygame.K_UP:
                        self._cy = max(0.0, self._cy - _CURSOR_STEP)
                        self._update_route()
                    elif event.key == pygame.K_DOWN:
                        self._cy = min(float(self._level_h_px), self._cy + _CURSOR_STEP)
                        self._update_route()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mx, my = event.pos
                        self._cx = mx / self._scale
                        self._cy = my / self._scale
                        self._update_route()

            self._draw()
            self._clock.tick(_FPS)

        import pygame

        pygame.quit()

    # ------------------------------------------------------------------
    # Route update
    # ------------------------------------------------------------------

    def _update_route(self) -> None:
        from .surface_graph import node_id, node_id_to_parts
        from .route_extraction import (
            find_nearest_start_node,
            extract_and_simplify_route,
        )

        d_star = (
            self._pbrs._d_star_exit
            if self._phase == "exit"
            else self._pbrs._d_star_switch
        )
        pred = (
            self._pbrs._pred_exit if self._phase == "exit" else self._pbrs._pred_switch
        )
        best_edge = (
            self._pbrs._best_edge_exit
            if self._phase == "exit"
            else self._pbrs._best_edge_switch
        )

        start = find_nearest_start_node(
            self._graph, self._cx, self._cy, 0.0, 0.0, None, d_star
        )
        self._snapped_node = start

        if start is not None:
            pt_idx, _ = node_id_to_parts(start, self._graph.max_vel_bins)
            sp = self._graph.surface_points[pt_idx]
            self._snapped_sp_xy = (sp.x, sp.y)
            self._current_d_star = (
                float(d_star[start]) if np.isfinite(d_star[start]) else float("inf")
            )
        else:
            self._snapped_sp_xy = None
            self._current_d_star = float("inf")

        # Detect airborne: cursor is far from its nearest floor node
        if self._snapped_sp_xy is not None:
            dx_s = self._cx - self._snapped_sp_xy[0]
            dy_s = self._cy - self._snapped_sp_xy[1]
            snap_dist = math.sqrt(dx_s * dx_s + dy_s * dy_s)
            self._is_airborne = snap_dist > _TILE_PX
        else:
            snap_dist = float("inf")
            self._is_airborne = True

        # Simulate ballistic fall from cursor (vy starts at 0, pure gravity)
        self._ballistic_pts = []
        if self._is_airborne:
            try:
                from ..constants.physics_constants import GRAVITY_FALL

                bx, by = self._cx, self._cy
                bvy = 0.0
                for frame in range(90):
                    bvy += GRAVITY_FALL
                    bx += 0.0  # vx = 0
                    by += bvy
                    if by > self._level_h_px:
                        break
                    # Sample every 3 frames for display
                    if frame % 3 == 0:
                        self._ballistic_pts.append((bx, by))
                    # Stop if close to a surface point
                    if self._snapped_sp_xy is not None:
                        ddx = bx - self._snapped_sp_xy[0]
                        ddy = by - self._snapped_sp_xy[1]
                        if math.sqrt(ddx * ddx + ddy * ddy) < 12.0:
                            break
            except Exception:
                pass

        if start is not None and np.isfinite(d_star[start]):
            arc_targets = extract_and_simplify_route(
                self._graph, d_star, pred, best_edge, start
            )
        else:
            arc_targets = []

        d_max = (
            self._pbrs._d_max_exit
            if self._phase == "exit"
            else self._pbrs._d_max_switch
        )
        phi = (
            -self._current_d_star / max(d_max, 1.0)
            if math.isfinite(self._current_d_star)
            else -1.0
        )
        self._current_potential = float(phi)

        self._current_route = [
            {
                "route_idx": i,
                "launch_x": arc.launch_x,
                "launch_y": arc.launch_y,
                "landing_x": arc.landing_x,
                "landing_y": arc.landing_y,
                "jump_hold": arc.jump_hold_frames,
                "launch_vx": arc.launch_vx,
                "h_input": arc.h_input_during,
                "cumulative_cost": arc.cumulative_cost,
            }
            for i, arc in enumerate(arc_targets)
        ]

        # Arc target features: only available if route is non-empty
        if arc_targets:
            arc = arc_targets[0]
            from ..constants.physics_constants import MAX_HOR_SPEED, MAX_JUMP_DURATION

            dx_l = np.clip((arc.launch_x - self._cx) / (_TILE_PX * 4.0), -1.0, 1.0)
            dy_l = np.clip((arc.launch_y - self._cy) / (_TILE_PX * 4.0), -1.0, 1.0)
            vx_err = np.clip(arc.launch_vx / MAX_HOR_SPEED, -1.0, 1.0)
            dx_p = np.clip((arc.landing_x - self._cx) / (_TILE_PX * 8.0), -1.0, 1.0)
            dy_p = np.clip((arc.landing_y - self._cy) / (_TILE_PX * 8.0), -1.0, 1.0)
            jh = np.clip(arc.jump_hold_frames / max(MAX_JUMP_DURATION, 1), 0.0, 1.0)
            spd = np.clip(abs(arc.launch_vx) / MAX_HOR_SPEED, 0.0, 1.0)
            total_cost = arc_targets[0].cumulative_cost if arc_targets else d_max
            progress = np.clip(
                1.0 - arc.cumulative_cost / max(total_cost, 1.0), 0.0, 1.0
            )
            self._current_features = np.array(
                [dx_l, dy_l, vx_err, dx_p, dy_p, jh, spd, progress, 0.0, 0.0],
                dtype=np.float32,
            )
        else:
            self._current_features = np.zeros(10, np.float32)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        import pygame
        from .route_overlay_draw import (
            draw_surface_heatmap,
            draw_all_arcs,
            draw_route_arcs,
            draw_arc_features_panel,
            draw_info_panel,
            draw_edge_legend,
            _LEGEND_ENTRY_COUNT,
        )

        self._screen.fill(_COL_BG)
        s = self._scale
        x_off, y_off = 0, 0

        # -- Tiles --
        if self._show_tiles and self._tile_dic:
            _draw_tiles(self._screen, self._tile_dic, s)

        # -- Surface heatmap --
        if self._show_heatmap:
            d_star_pts = (
                self._d_star_exit_pt
                if self._phase == "exit"
                else self._d_star_switch_pt
            )
            overlay = pygame.Surface(self._screen.get_size(), pygame.SRCALPHA)
            draw_surface_heatmap(
                overlay, self._surface_pts_xy, d_star_pts, s, x_off, y_off
            )
            self._screen.blit(overlay, (0, 0))

        # -- All arcs --
        if self._show_arcs:
            overlay = pygame.Surface(self._screen.get_size(), pygame.SRCALPHA)
            draw_all_arcs(overlay, self._edge_data, s, x_off, y_off)
            self._screen.blit(overlay, (0, 0))

        # -- Route --
        if self._show_route and self._current_route:
            overlay = pygame.Surface(self._screen.get_size(), pygame.SRCALPHA)
            draw_route_arcs(overlay, self._current_route, 0, s, x_off, y_off)
            self._screen.blit(overlay, (0, 0))

        # -- Goal markers --
        _draw_goal_markers(
            self._screen, self._switch_pos, self._exit_pos, self._phase, s
        )

        # -- Airborne indicator --
        if self._is_airborne:
            from .route_overlay_draw import draw_airborne_indicator

            overlay = pygame.Surface(self._screen.get_size(), pygame.SRCALPHA)
            draw_airborne_indicator(
                overlay,
                (self._cx, self._cy),
                self._snapped_sp_xy,
                self._ballistic_pts,
                s,
                x_off,
                y_off,
            )
            self._screen.blit(overlay, (0, 0))

        # -- Cursor --
        _draw_cursor(self._screen, self._cx, self._cy, self._snapped_sp_xy, s)

        # -- Panels – all anchored to bottom of screen --
        _PAD, _ROW_H = 8, 16
        MARGIN = 10
        scr_w = self._screen.get_width()
        scr_h = self._screen.get_height()

        if self._show_features:
            feat_h = _PAD * 2 + 18 + _ROW_H * 10
            feat_y = scr_h - feat_h - MARGIN
            draw_arc_features_panel(
                self._screen,
                self._current_features,
                scr_w - 202,
                feat_y,
            )

        info = {
            "phase": self._phase,
            "potential": self._current_potential,
            "d_star": self._current_d_star,
            "arc_idx": 0,
            "num_arcs": len(self._current_route),
            "replans": 0,
            "cx": self._cx,
            "cy": self._cy,
        }
        info_rows = 6
        info_h = _PAD * 2 + 18 + _ROW_H * info_rows
        info_y = scr_h - info_h - MARGIN
        draw_info_panel(self._screen, info, MARGIN, info_y)

        # Edge legend – centred, shown whenever arcs or route are visible
        if self._show_arcs or self._show_route:
            legend_h = _PAD * 2 + 18 + _ROW_H * _LEGEND_ENTRY_COUNT
            legend_w = 215
            legend_x = (scr_w - legend_w) // 2
            legend_y = scr_h - legend_h - MARGIN
            draw_edge_legend(self._screen, legend_x, legend_y)

        pygame.display.flip()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_tiles(screen: "pygame.Surface", tile_dic: Dict, scale: float) -> None:
    """Render solid tiles as filled rectangles."""
    import pygame

    cell = max(1, int(_TILE_PX * scale))
    for (col, row), tile_type in tile_dic.items():
        if tile_type == 0:
            continue
        sx = int(col * _TILE_PX * scale)
        sy = int(row * _TILE_PX * scale)
        pygame.draw.rect(screen, _COL_TILE, (sx, sy, cell, cell))
        pygame.draw.rect(screen, _COL_TILE_BORDER, (sx, sy, cell, cell), 1)


def _draw_goal_markers(
    screen: "pygame.Surface",
    switch_pos: Optional[Tuple[float, float]],
    exit_pos: Optional[Tuple[float, float]],
    phase: str,
    scale: float,
) -> None:
    import pygame

    r = max(8, int(10 * scale))
    if switch_pos:
        sx, sy = int(switch_pos[0] * scale), int(switch_pos[1] * scale)
        col = (255, 255, 50) if phase == "switch" else (180, 180, 50)
        pygame.draw.circle(screen, col, (sx, sy), r, 2)
    if exit_pos:
        sx, sy = int(exit_pos[0] * scale), int(exit_pos[1] * scale)
        col = (50, 255, 50) if phase == "exit" else (50, 150, 50)
        pygame.draw.circle(screen, col, (sx, sy), r, 2)


def _draw_cursor(
    screen: "pygame.Surface",
    cx: float,
    cy: float,
    snap_xy: Optional[Tuple[float, float]],
    scale: float,
) -> None:
    import pygame

    sx, sy = int(cx * scale), int(cy * scale)
    pygame.draw.circle(screen, _COL_CURSOR, (sx, sy), max(4, int(5 * scale)), 2)
    pygame.draw.line(screen, _COL_CURSOR, (sx - 8, sy), (sx + 8, sy), 1)
    pygame.draw.line(screen, _COL_CURSOR, (sx, sy - 8), (sx, sy + 8), 1)
    if snap_xy:
        spx, spy = int(snap_xy[0] * scale), int(snap_xy[1] * scale)
        pygame.draw.circle(screen, _COL_SNAP, (spx, spy), max(5, int(6 * scale)), 2)


# ---------------------------------------------------------------------------
# Data-prep helpers
# ---------------------------------------------------------------------------


def _load_route_pbrs(npz_path: str):
    """Load RoutePBRS and the underlying SurfaceGraph from an npz file."""
    import os
    from .route_pbrs import create_route_pbrs
    from .surface_graph import build_graph
    from .surface_extraction import SurfacePoint

    value_dir = str(os.path.dirname(os.path.abspath(npz_path)))
    level_id = os.path.basename(npz_path).replace("_surface_pbrs.npz", "")
    pbrs = create_route_pbrs(level_id, value_dir)

    if pbrs is None:
        raise ValueError(
            f"Could not load RoutePBRS from {npz_path}. "
            "Ensure the file contains route-tree arrays "
            "(predecessor_switch/exit, best_edge_idx_switch/exit). "
            "Run precompute_cli.py to regenerate."
        )
    return pbrs, pbrs._graph


def _build_surface_point_arrays(graph, pbrs):
    """Return (xy_array, d_star_switch_per_point, d_star_exit_per_point)."""
    from .surface_graph import node_id, num_velocity_bins

    n = graph.n_points
    xy = np.array([[sp.x, sp.y] for sp in graph.surface_points], dtype=np.float32)

    def _min_dstar(d_star_arr):
        vals = np.full(n, np.inf, dtype=np.float32)
        for i, sp in enumerate(graph.surface_points):
            n_bins = num_velocity_bins(sp.surface_type)
            for vb in range(n_bins):
                nid = node_id(i, vb, graph.max_vel_bins)
                if nid < len(d_star_arr):
                    v = float(d_star_arr[nid])
                    if v < vals[i]:
                        vals[i] = v
        return vals

    sw_pts = _min_dstar(pbrs._d_star_switch)
    ex_pts = _min_dstar(pbrs._d_star_exit)
    return xy, sw_pts, ex_pts


def _build_edge_data(graph) -> Dict:
    """Thin wrapper around route_overlay_draw.build_edge_data."""
    from .route_overlay_draw import build_edge_data

    return build_edge_data(graph)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive route PBRS inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", required=True, help="Path to *_surface_pbrs.npz")
    parser.add_argument(
        "--map-path", default=None, help="Optional .map level file for tile rendering"
    )
    parser.add_argument(
        "--scale", type=float, default=_DEFAULT_SCALE, help="Zoom factor (default 1.5)"
    )
    parser.add_argument(
        "--phase",
        choices=["switch", "exit"],
        default="switch",
        help="Initial goal phase (default: switch)",
    )
    args = parser.parse_args()

    viz = RouteVisualizer(
        npz_path=args.data,
        map_path=args.map_path,
        scale=args.scale,
        initial_phase=args.phase,
    )
    viz.run()


if __name__ == "__main__":
    main()
