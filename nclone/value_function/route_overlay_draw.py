"""Pure pygame drawing utilities for route PBRS visualisation.

All functions are stateless: they accept a ``pygame.Surface`` plus pre-extracted
data dicts and coordinate-mapping parameters, draw onto the surface, and return.

Coordinate helpers
------------------
``to_screen(x, y)`` maps game-pixel coordinates to screen-pixel coordinates:

    screen_x = int(x * adjust) + x_off
    screen_y = int(y * adjust) + y_off

Surface points and ninja positions are in full-map pixel space (including the
1-tile border) and require no additional offset.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
    import pygame.font
except ImportError as exc:
    raise ImportError("pygame is required for route overlay drawing") from exc

# Colour palette
_COL_WALK_ARC = (60, 220, 255, 160)  # cyan  – run/slide arcs (all-edges view)
_COL_JUMP_ARC = (255, 160, 40, 160)  # amber – jump arcs (all-edges view)
_COL_ROUTE_JUMP = (255, 80, 80, 230)  # red    – route jump segment
_COL_ROUTE_CURRENT = (255, 255, 255, 255)  # white  – currently-active arc
_COL_LANDING_DOT = (100, 220, 255, 220)  # cyan dot at landing
_COL_CURSOR = (255, 255, 255, 255)
_COL_PANEL_BG = (0, 0, 0, 210)
_COL_PANEL_BORDER = (120, 180, 255, 255)
_COL_AIRBORNE_DASH = (220, 220, 255, 200)  # dashed snap line when airborne
_COL_BALLISTIC = (255, 255, 255, 170)  # ballistic fall trajectory dots

_MAX_EDGES_DRAW = 4000  # cap for "draw all arcs"
_MAX_NODES_DRAW = 3000  # cap for surface heatmap

# Legend entry count – keep in sync with draw_edge_legend entries list
_LEGEND_ENTRY_COUNT = 12

# Feature names matching RoutePBRS.get_arc_target_features layout
_FEATURE_NAMES = [
    "dx_launch",
    "dy_launch",
    "vx_error",
    "dx_land",
    "dy_land",
    "jump_hold",
    "req_speed",
    "progress",
    "phase",
    "spare",
]


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def make_to_screen(
    adjust: float, x_off: int, y_off: int
) -> Callable[[float, float], Tuple[int, int]]:
    """Return a closure that converts game coords to screen coords."""

    def _to_screen(x: float, y: float) -> Tuple[int, int]:
        return (int(x * adjust) + x_off, int(y * adjust) + y_off)

    return _to_screen


# ---------------------------------------------------------------------------
# Bezier helpers
# ---------------------------------------------------------------------------


def _bezier_quad(
    p0: Tuple[int, int],
    pc: Tuple[int, int],
    p2: Tuple[int, int],
    n: int = 12,
) -> List[Tuple[int, int]]:
    """Evaluate a quadratic bezier at ``n+1`` equidistant t values.

    Returns a list of ``n+1`` (x, y) integer screen-pixel points.
    """
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2 * t * mt * pc[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * t * mt * pc[1] + t * t * p2[1]
        pts.append((int(x), int(y)))
    return pts


def _jump_ctrl_pt(
    sx1: int, sy1: int, sx2: int, sy2: int, jump_hold: int, adjust: float
) -> Tuple[int, int]:
    """Compute the bezier control point that arcs a jump above the chord.

    The apex is elevated above the lower of the two endpoints, with height
    proportional to ``jump_hold`` (capped to keep it on-screen).
    """
    ctrl_x = (sx1 + sx2) // 2
    apex_h = max(6, min(jump_hold * 3, 120)) * adjust
    ctrl_y = int(min(sy1, sy2) - apex_h)
    return (ctrl_x, ctrl_y)


# ---------------------------------------------------------------------------
# Surface-node heatmap
# ---------------------------------------------------------------------------


def draw_surface_heatmap(
    surface: "pygame.Surface",
    surface_points_xy: np.ndarray,  # shape (N, 2) float32
    d_star_per_point: np.ndarray,  # shape (N,) float32, min-over-bins
    adjust: float,
    x_off: int,
    y_off: int,
) -> None:
    """Scatter surface nodes coloured from green (near goal) to red (far).

    Args:
        surface_points_xy: Array of (x, y) positions in game-pixel coords.
        d_star_per_point: Cost-to-go for each surface point (min across vel bins).
    """
    to_screen = make_to_screen(adjust, x_off, y_off)
    n = min(len(surface_points_xy), _MAX_NODES_DRAW)
    finite_mask = np.isfinite(d_star_per_point[:n])
    finite_vals = d_star_per_point[:n][finite_mask]
    d_max = float(np.max(finite_vals)) if len(finite_vals) > 0 else 1.0
    if d_max < 1.0:
        d_max = 1.0

    cell = max(4, int(7 * adjust))
    for i in range(n):
        x, y = float(surface_points_xy[i, 0]), float(surface_points_xy[i, 1])
        cost = float(d_star_per_point[i])
        if not math.isfinite(cost):
            colour = (60, 60, 60, 120)
        else:
            t = min(1.0, cost / d_max)
            r = int(255 * t)
            g = int(255 * (1.0 - t))
            colour = (r, g, 50, 140)
        sx, sy = to_screen(x, y)
        pygame.draw.rect(surface, colour, (sx - cell // 2, sy - cell // 2, cell, cell))


# ---------------------------------------------------------------------------
# All arcs (graph edges)  –  jump arcs use 8-point bezier
# ---------------------------------------------------------------------------


def draw_all_arcs(
    surface: "pygame.Surface",
    edge_data: Dict,  # {"src_xy": (N,2), "dst_xy": (N,2), "jump_hold": (N,)}
    adjust: float,
    x_off: int,
    y_off: int,
) -> None:
    """Draw every graph edge.

    - Walk/slide arcs (jump_hold == 0): thin cyan straight lines.
    - Jump arcs (jump_hold > 0): 8-point quadratic bezier in amber to show the
      parabolic trajectory above the chord.

    Capped at ``_MAX_EDGES_DRAW`` edges for performance.
    """
    to_screen = make_to_screen(adjust, x_off, y_off)
    src_xy: np.ndarray = edge_data["src_xy"]
    dst_xy: np.ndarray = edge_data["dst_xy"]
    jump_hold: np.ndarray = edge_data["jump_hold"]
    n = min(len(src_xy), _MAX_EDGES_DRAW)

    for i in range(n):
        sx1, sy1 = to_screen(float(src_xy[i, 0]), float(src_xy[i, 1]))
        sx2, sy2 = to_screen(float(dst_xy[i, 0]), float(dst_xy[i, 1]))
        jh = int(jump_hold[i])

        if jh > 0:
            # Parabolic bezier for jump arcs
            pc = _jump_ctrl_pt(sx1, sy1, sx2, sy2, jh, adjust)
            pts = _bezier_quad((sx1, sy1), pc, (sx2, sy2), n=8)
            if len(pts) >= 2:
                pygame.draw.lines(surface, _COL_JUMP_ARC, False, pts, 1)
        else:
            pygame.draw.line(surface, _COL_WALK_ARC, (sx1, sy1), (sx2, sy2), 1)


# ---------------------------------------------------------------------------
# Route arcs  –  bezier jumps + dashed walk chevrons
# ---------------------------------------------------------------------------


def _walk_arc_colour(launch_vx: float) -> Tuple[int, int, int, int]:
    """Interpolate steel-blue → bright-cyan based on normalised speed."""
    try:
        from ..constants.physics_constants import MAX_HOR_SPEED

        max_spd = float(MAX_HOR_SPEED)
    except Exception:
        max_spd = 3.5
    t = min(1.0, abs(launch_vx) / max_spd)
    # steel-blue (70,130,180) → bright-cyan (60,220,255)
    r = int(70 + (60 - 70) * t)
    g = int(130 + (220 - 130) * t)
    b = int(180 + (255 - 180) * t)
    return (r, g, b, 220)


def _draw_dashed_line(
    surface: "pygame.Surface",
    colour: Tuple,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    dash: int = 8,
    gap: int = 4,
    width: int = 2,
) -> None:
    """Draw a dashed line from p1 to p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1.0:
        return
    ux, uy = dx / length, dy / length
    drawn = 0.0
    drawing = True
    while drawn < length:
        seg_len = dash if drawing else gap
        seg_end = min(drawn + seg_len, length)
        if drawing:
            x1 = int(p1[0] + ux * drawn)
            y1 = int(p1[1] + uy * drawn)
            x2 = int(p1[0] + ux * seg_end)
            y2 = int(p1[1] + uy * seg_end)
            pygame.draw.line(surface, colour, (x1, y1), (x2, y2), width)
        drawn += seg_len
        drawing = not drawing


def _draw_chevron(
    surface: "pygame.Surface",
    colour: Tuple,
    mx: int,
    my: int,
    ux: float,
    uy: float,
    size: int,
    width: int = 2,
) -> None:
    """Draw a single >> chevron arrow at (mx, my) pointing in (ux, uy)."""
    perp_x, perp_y = -uy, ux  # perpendicular unit vector
    half = size // 2
    # Front tip
    tx, ty = int(mx + ux * half), int(my + uy * half)
    # Back left / right wings
    bx, by = int(mx - ux * half), int(my - uy * half)
    lx = int(bx + perp_x * half)
    ly = int(by + perp_y * half)
    rx = int(bx - perp_x * half)
    ry = int(by - perp_y * half)
    pygame.draw.line(surface, colour, (lx, ly), (tx, ty), width)
    pygame.draw.line(surface, colour, (rx, ry), (tx, ty), width)


def draw_route_arcs(
    surface: "pygame.Surface",
    route: List[Dict],  # list of arc dicts; see _update_route for keys
    arc_idx: int,
    adjust: float,
    x_off: int,
    y_off: int,
) -> None:
    """Draw the planned route.

    Jump arcs: 16-point quadratic bezier parabola (red/white) with arrowhead
               at the last bezier segment tangent.
    Walk arcs: dashed speed-coloured line with a directional chevron at midpoint
               and a small landing dot.
    The currently-active arc is drawn in white and is thicker.
    """
    to_screen = make_to_screen(adjust, x_off, y_off)
    arrow_size = max(5, int(7 * adjust))
    dot_r = max(2, int(3 * adjust))
    chevron_size = max(8, int(10 * adjust))

    for seg in route:
        i = seg["route_idx"]
        lx, ly = float(seg["launch_x"]), float(seg["launch_y"])
        px, py = float(seg["landing_x"]), float(seg["landing_y"])
        jh = int(seg.get("jump_hold", 0))
        launch_vx = float(seg.get("launch_vx", 0.0))
        h_input = int(seg.get("h_input", 0))

        is_current = i == arc_idx
        width = max(3, int(4 * adjust)) if is_current else max(2, int(3 * adjust))

        sx1, sy1 = to_screen(lx, ly)
        sx2, sy2 = to_screen(px, py)

        if is_current:
            colour = _COL_ROUTE_CURRENT
        elif jh > 0:
            colour = _COL_ROUTE_JUMP
        else:
            colour = _walk_arc_colour(launch_vx)

        if jh > 0:
            # --- Jump arc: parabolic bezier ---
            pc = _jump_ctrl_pt(sx1, sy1, sx2, sy2, jh, adjust)
            pts = _bezier_quad((sx1, sy1), pc, (sx2, sy2), n=16)
            if len(pts) >= 2:
                pygame.draw.lines(surface, colour, False, pts, width)

            # Arrowhead tangent from last bezier segment
            if len(pts) >= 2:
                ex, ey = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
                elen = math.sqrt(ex * ex + ey * ey)
                if elen > 0.5:
                    eux, euy = ex / elen, ey / elen
                    ax1 = int(sx2 - arrow_size * (eux + euy * 0.5))
                    ay1 = int(sy2 - arrow_size * (euy - eux * 0.5))
                    ax2 = int(sx2 - arrow_size * (eux - euy * 0.5))
                    ay2 = int(sy2 - arrow_size * (euy + eux * 0.5))
                    pygame.draw.polygon(
                        surface, colour, [(sx2, sy2), (ax1, ay1), (ax2, ay2)]
                    )

            # Landing dot
            pygame.draw.circle(surface, _COL_LANDING_DOT, (sx2, sy2), dot_r)

        else:
            # --- Walk arc: dashed line + directional chevron ---
            _draw_dashed_line(
                surface,
                colour,
                (sx1, sy1),
                (sx2, sy2),
                dash=max(6, int(8 * adjust)),
                gap=max(3, int(4 * adjust)),
                width=width,
            )

            # Chevron at midpoint pointing in direction of travel
            dx, dy = sx2 - sx1, sy2 - sy1
            dlen = math.sqrt(dx * dx + dy * dy)
            if dlen > 1.0:
                # Use h_input sign if available, else infer from geometry
                direction = h_input if h_input != 0 else (1 if dx >= 0 else -1)
                ux = (dx / dlen) * direction if direction >= 0 else -(dx / dlen)
                uy = (dy / dlen) * direction if direction >= 0 else -(dy / dlen)
                # Ensure chevron points from launch toward landing
                dot_prod = ux * dx + uy * dy
                if dot_prod < 0:
                    ux, uy = -ux, -uy
                mx = (sx1 + sx2) // 2
                my = (sy1 + sy2) // 2
                _draw_chevron(surface, colour, mx, my, ux, uy, chevron_size, width)

            # Small landing dot (no launch dot – dashed line makes start obvious)
            pygame.draw.circle(surface, colour[:3] + (200,), (sx2, sy2), dot_r)


# ---------------------------------------------------------------------------
# Airborne indicator
# ---------------------------------------------------------------------------


def draw_airborne_indicator(
    surface: "pygame.Surface",
    cursor_xy: Tuple[float, float],
    snap_xy: Optional[Tuple[float, float]],
    ballistic_pts: Optional[List[Tuple[float, float]]],
    adjust: float,
    x_off: int,
    y_off: int,
) -> None:
    """Visualise an airborne agent/cursor position.

    Draws:
    - A white dashed line from the cursor to the snapped floor node.
    - A series of fading dots along the predicted ballistic fall trajectory.
    - An ``"airborne"`` label near the cursor.
    """
    to_screen = make_to_screen(adjust, x_off, y_off)
    cx, cy = to_screen(cursor_xy[0], cursor_xy[1])

    # Dashed snap line
    if snap_xy is not None:
        sx, sy = to_screen(snap_xy[0], snap_xy[1])
        _draw_dashed_line(
            surface, _COL_AIRBORNE_DASH, (cx, cy), (sx, sy), dash=5, gap=5, width=1
        )
        # Small ring at snap point
        pygame.draw.circle(
            surface, _COL_AIRBORNE_DASH, (sx, sy), max(5, int(6 * adjust)), 2
        )

    # Ballistic fall trajectory dots (fading)
    if ballistic_pts:
        n = len(ballistic_pts)
        for k, (bx, by) in enumerate(ballistic_pts):
            alpha = int(180 - 120 * k / max(n - 1, 1))  # 180 → 60
            scx, scy = to_screen(bx, by)
            dot_col = (255, 255, 255, alpha)
            pygame.draw.circle(surface, dot_col, (scx, scy), max(2, int(3 * adjust)))

    # "airborne" label
    try:
        font = pygame.font.Font(None, 15)
    except pygame.error:
        font = pygame.font.SysFont("monospace", 11)
    label = font.render("airborne", True, (200, 200, 255))
    surface.blit(label, (cx + 8, cy - 18))


# ---------------------------------------------------------------------------
# Edge-colour legend
# ---------------------------------------------------------------------------


def draw_edge_legend(
    surface: "pygame.Surface",
    panel_x: int,
    panel_y: int,
) -> None:
    """Draw a compact legend explaining every colour used in the route overlay.

    12 rows; height = 8*2 + 18 + 16*_LEGEND_ENTRY_COUNT = 226 px.
    Width = 215 px.
    """
    PAD = 8
    ROW_H = 16
    SWATCH_W = 22
    GAP = 5
    TEXT_X = panel_x + PAD + SWATCH_W + GAP
    panel_w = 215
    panel_h = PAD * 2 + 18 + ROW_H * _LEGEND_ENTRY_COUNT

    try:
        title_font = pygame.font.Font(None, 17)
        sub_font = pygame.font.Font(None, 14)
        row_font = pygame.font.Font(None, 15)
    except pygame.error:
        title_font = pygame.font.SysFont("monospace", 13, bold=True)
        sub_font = pygame.font.SysFont("monospace", 10)
        row_font = pygame.font.SysFont("monospace", 11)

    pygame.draw.rect(
        surface, _COL_PANEL_BG, (panel_x, panel_y, panel_w, panel_h), border_radius=4
    )
    pygame.draw.rect(
        surface,
        _COL_PANEL_BORDER,
        (panel_x, panel_y, panel_w, panel_h),
        1,
        border_radius=4,
    )

    title_surf = title_font.render("Edge Legend", True, (200, 220, 255))
    surface.blit(title_surf, (panel_x + PAD, panel_y + 4))

    # colour references
    CYAN = _COL_WALK_ARC[:3]
    AMBER = _COL_JUMP_ARC[:3]
    WHITE = (255, 255, 255)
    RED = _COL_ROUTE_JUMP[:3]
    BLUE = (80, 100, 255)
    AIR = _COL_AIRBORNE_DASH[:3]
    BALL = _COL_BALLISTIC[:3]

    # Each entry: ("sub", label) | ("line", col, label) | ("bezier", col, label) |
    #             ("dash", col, label) | ("dots", col, label) | ("grad", label)
    entries: List[Tuple] = [
        ("sub", "HEATMAP"),
        ("grad", "d_star cost-to-go"),
        ("sub", "ALL ARCS  [E]"),
        ("line", CYAN, "walk / slide arc"),
        ("bezier", AMBER, "jump arc"),
        ("sub", "ROUTE  [O]"),
        ("line", WHITE, "current arc"),
        ("bezier", RED, "jump segment"),
        ("dash", BLUE, "walk arc (speed)"),
        ("sub", "AIRBORNE"),
        ("dash", AIR, "snap-to-surface"),
        ("dots", BALL, "ballistic fall"),
    ]

    sx0 = panel_x + PAD
    y = panel_y + PAD + 18
    for entry in entries:
        kind = entry[0]
        mid_y = y + ROW_H // 2

        if kind == "sub":
            lbl = sub_font.render(entry[1], True, (150, 200, 255))
            surface.blit(lbl, (panel_x + PAD, y + 2))

        elif kind == "grad":
            for di in range(SWATCH_W):
                t = di / max(SWATCH_W - 1, 1)
                gc = (int(255 * t), int(255 * (1.0 - t)), 0)
                pygame.draw.line(
                    surface, gc, (sx0 + di, mid_y - 2), (sx0 + di, mid_y + 2)
                )
            txt = row_font.render(entry[1], True, (200, 200, 200))
            surface.blit(txt, (TEXT_X, y))

        elif kind == "line":
            col = entry[1]
            pygame.draw.line(surface, col, (sx0, mid_y), (sx0 + SWATCH_W, mid_y), 2)
            # arrowhead
            ex = sx0 + SWATCH_W
            pygame.draw.polygon(
                surface, col, [(ex, mid_y), (ex - 5, mid_y - 3), (ex - 5, mid_y + 3)]
            )
            txt = row_font.render(entry[2], True, col)
            surface.blit(txt, (TEXT_X, y))

        elif kind == "bezier":
            col = entry[1]
            pts = []
            for i in range(9):
                t = i / 8.0
                mt = 1.0 - t
                bx = (
                    mt * mt * sx0
                    + 2 * t * mt * (sx0 + SWATCH_W // 2)
                    + t * t * (sx0 + SWATCH_W)
                )
                by = mt * mt * mid_y + 2 * t * mt * (y + 2) + t * t * mid_y
                pts.append((int(bx), int(by)))
            if len(pts) >= 2:
                pygame.draw.lines(surface, col, False, pts, 2)
            txt = row_font.render(entry[2], True, col)
            surface.blit(txt, (TEXT_X, y))

        elif kind == "dash":
            col = entry[1]
            px = sx0
            draw_dash = True
            while px < sx0 + SWATCH_W:
                end = min(px + 4, sx0 + SWATCH_W)
                if draw_dash:
                    pygame.draw.line(surface, col, (px, mid_y), (end, mid_y), 2)
                px += 7
                draw_dash = not draw_dash
            txt = row_font.render(entry[2], True, col)
            surface.blit(txt, (TEXT_X, y))

        elif kind == "dots":
            col = entry[1]
            for di in range(4):
                pygame.draw.circle(surface, col, (sx0 + 3 + di * 6, mid_y), 2)
            txt = row_font.render(entry[2], True, col)
            surface.blit(txt, (TEXT_X, y))

        y += ROW_H


# ---------------------------------------------------------------------------
# Arc target features panel
# ---------------------------------------------------------------------------


def draw_arc_features_panel(
    surface: "pygame.Surface",
    features: np.ndarray,  # length-10 float32
    panel_x: int,
    panel_y: int,
) -> None:
    """Draw a compact bar-chart panel showing the 10 arc target features."""
    BAR_W = 100
    ROW_H = 16
    LABEL_W = 75
    PAD = 8
    panel_w = LABEL_W + BAR_W + PAD * 3
    panel_h = PAD * 2 + 18 + ROW_H * 10

    try:
        title_font = pygame.font.Font(None, 17)
        row_font = pygame.font.Font(None, 15)
    except pygame.error:
        title_font = pygame.font.SysFont("monospace", 13, bold=True)
        row_font = pygame.font.SysFont("monospace", 11)

    pygame.draw.rect(
        surface, _COL_PANEL_BG, (panel_x, panel_y, panel_w, panel_h), border_radius=4
    )
    pygame.draw.rect(
        surface,
        _COL_PANEL_BORDER,
        (panel_x, panel_y, panel_w, panel_h),
        1,
        border_radius=4,
    )

    title_surf = title_font.render("Arc Features", True, (200, 220, 255))
    surface.blit(title_surf, (panel_x + PAD, panel_y + 4))

    y = panel_y + PAD + 18
    for name, val in zip(_FEATURE_NAMES, features):
        label_surf = row_font.render(f"{name}", True, (200, 200, 200))
        surface.blit(label_surf, (panel_x + PAD, y))

        # Normalised bar: clip [-1, 1] → [0, 1]
        t = float(np.clip((val + 1.0) / 2.0, 0.0, 1.0))
        bar_filled = int(BAR_W * t)
        bar_x = panel_x + PAD + LABEL_W
        pygame.draw.rect(surface, (40, 40, 40, 200), (bar_x, y + 1, BAR_W, ROW_H - 4))
        bar_col = (80, 200, 80) if val >= 0.0 else (200, 80, 80)
        if bar_filled > 0:
            pygame.draw.rect(surface, bar_col, (bar_x, y + 1, bar_filled, ROW_H - 4))

        val_surf = row_font.render(f"{val:.2f}", True, (240, 240, 240))
        surface.blit(val_surf, (bar_x + BAR_W - val_surf.get_width() - 2, y))

        y += ROW_H


# ---------------------------------------------------------------------------
# Info panel
# ---------------------------------------------------------------------------


def draw_info_panel(
    surface: "pygame.Surface",
    info: Dict,
    panel_x: int,
    panel_y: int,
    reward_components: Optional[Dict] = None,
) -> None:
    """Draw a text panel with d_star, potential, phase, arc index, and optional reward breakdown.

    When ``reward_components`` (from ``PrimitiveRewardCalculator.last_components``) is
    provided, 5 extra rows are appended showing the per-frame reward decomposition.
    """
    try:
        title_font = pygame.font.Font(None, 17)
        row_font = pygame.font.Font(None, 15)
    except pygame.error:
        title_font = pygame.font.SysFont("monospace", 13, bold=True)
        row_font = pygame.font.SysFont("monospace", 11)

    lines: List[Tuple[str, Tuple[int, int, int]]] = [
        (f"Phase: {info.get('phase', '?')}", (180, 180, 255)),
        (f"Potential \u03a6: {info.get('potential', 0.0):.4f}", (100, 255, 100)),
        (f"d_star: {info.get('d_star', float('inf')):.1f} fr", (255, 220, 100)),
        (
            f"Arc {info.get('arc_idx', 0) + 1}/{info.get('num_arcs', 0)}",
            (255, 180, 100),
        ),
        (f"Replans: {info.get('replans', 0)}", (200, 200, 200)),
        (
            f"Cursor ({info.get('cx', 0.0):.0f}, {info.get('cy', 0.0):.0f})",
            (160, 160, 160),
        ),
    ]

    if reward_components:
        rc = reward_components
        alpha = float(rc.get("alpha", 1.0))
        pbrs_shaped = alpha * float(rc.get("pbrs_reward", 0.0))
        scaled_r = float(rc.get("scaled_reward", 0.0))
        sparse_r = float(rc.get("sparse_reward", 0.0))
        time_p = float(rc.get("time_penalty", 0.0))
        lines += [
            ("\u2500\u2500 Reward (this frame) \u2500", (100, 120, 180)),
            (f"r frame: {scaled_r:+.5f}", (255, 210, 80)),
            (f"PBRS Ft: {pbrs_shaped:+.5f}", (80, 210, 255)),
            (f"Sparse:  {sparse_r:+.5f}", (160, 255, 160)),
            (f"Time:    {time_p:+.5f}", (200, 140, 140)),
        ]

    PAD = 8
    ROW_H = 16
    panel_w = 215
    panel_h = PAD * 2 + 18 + ROW_H * len(lines)

    pygame.draw.rect(
        surface, _COL_PANEL_BG, (panel_x, panel_y, panel_w, panel_h), border_radius=4
    )
    pygame.draw.rect(
        surface,
        _COL_PANEL_BORDER,
        (panel_x, panel_y, panel_w, panel_h),
        1,
        border_radius=4,
    )

    title_surf = title_font.render("Route PBRS", True, (200, 220, 255))
    surface.blit(title_surf, (panel_x + PAD, panel_y + 4))

    y = panel_y + PAD + 18
    for text, colour in lines:
        surf = row_font.render(text, True, colour)
        surface.blit(surf, (panel_x + PAD, y))
        y += ROW_H


# ---------------------------------------------------------------------------
# Data-preparation helper (also used by DebugMixin cache builder)
# ---------------------------------------------------------------------------


def build_edge_data(graph) -> Dict:
    """Extract edge src/dst positions and jump_hold arrays for drawing.

    Args:
        graph: SurfaceGraph with populated edges_src, edges_dst, edges_jump_hold.

    Returns:
        Dict with keys ``src_xy`` (N×2 float32), ``dst_xy`` (N×2 float32),
        ``jump_hold`` (N int8).
    """
    from .surface_graph import node_id_to_parts

    n_edges = len(graph.edges_src)
    src_xy = np.zeros((n_edges, 2), dtype=np.float32)
    dst_xy = np.zeros((n_edges, 2), dtype=np.float32)
    jump_hold = np.zeros(n_edges, dtype=np.int8)

    for i in range(n_edges):
        si, _ = node_id_to_parts(int(graph.edges_src[i]), graph.max_vel_bins)
        di, _ = node_id_to_parts(int(graph.edges_dst[i]), graph.max_vel_bins)
        src_xy[i] = [graph.surface_points[si].x, graph.surface_points[si].y]
        dst_xy[i] = [graph.surface_points[di].x, graph.surface_points[di].y]
        if graph.edges_jump_hold is not None:
            jump_hold[i] = graph.edges_jump_hold[i]

    return {"src_xy": src_xy, "dst_xy": dst_xy, "jump_hold": jump_hold}


# ---------------------------------------------------------------------------
# Convenience wrapper used by DebugOverlayRenderer
# ---------------------------------------------------------------------------


def draw_route_overlay_pygame(
    surface: "pygame.Surface",
    data: Dict,
    adjust: float,
    x_off: int,
    y_off: int,
) -> None:
    """Dispatch to all sub-drawers from a single overlay data dict.

    Expected keys in ``data``:
        show_heatmap (bool), show_arcs (bool), show_route (bool)
        surface_points_xy (ndarray N×2), d_star_per_point (ndarray N)
        edge_data (dict with src_xy, dst_xy, jump_hold)
        route (list of arc dicts), arc_idx (int)
        features (ndarray 10), info (dict)
        screen_w (int), screen_h (int)
        is_airborne (bool, optional)
        cursor_xy (tuple, optional), snap_xy (tuple, optional)
        ballistic_pts (list of (x,y), optional)
        reward_components (dict, optional) – PrimitiveRewardCalculator.last_components
    """
    screen_w = data.get("screen_w", surface.get_width())
    screen_h = data.get("screen_h", surface.get_height())

    if data.get("show_heatmap") and data.get("surface_points_xy") is not None:
        draw_surface_heatmap(
            surface,
            data["surface_points_xy"],
            data["d_star_per_point"],
            adjust,
            x_off,
            y_off,
        )

    # if data.get("show_arcs") and data.get("edge_data") is not None:
    #     draw_all_arcs(surface, data["edge_data"], adjust, x_off, y_off)

    if data.get("show_route") and data.get("route"):
        draw_route_arcs(
            surface, data["route"], data.get("arc_idx", 0), adjust, x_off, y_off
        )

    if data.get("is_airborne"):
        draw_airborne_indicator(
            surface,
            data.get("cursor_xy", (0.0, 0.0)),
            data.get("snap_xy"),
            data.get("ballistic_pts"),
            adjust,
            x_off,
            y_off,
        )

    # ---------------------------------------------------------------------------
    # Panels – all anchored to the bottom of the screen
    # ---------------------------------------------------------------------------
    _PAD, _ROW_H = 8, 16
    MARGIN = 10

    if data.get("features") is not None:
        feat_h = _PAD * 2 + 18 + _ROW_H * 10
        feat_x = screen_w - 200
        feat_y = screen_h - feat_h - MARGIN
        draw_arc_features_panel(surface, data["features"], feat_x, feat_y)

    reward_components = data.get("reward_components")
    if data.get("info") is not None:
        n_info_rows = 6 + (5 if reward_components else 0)
        info_h = _PAD * 2 + 18 + _ROW_H * n_info_rows
        info_y = screen_h - info_h - MARGIN
        draw_info_panel(
            surface, data["info"], MARGIN, info_y, reward_components=reward_components
        )

    # Edge legend – centred between the two side panels
    legend_h = _PAD * 2 + 18 + _ROW_H * _LEGEND_ENTRY_COUNT
    legend_w = 215
    legend_x = (screen_w - legend_w) // 2
    legend_y = screen_h - legend_h - MARGIN
    draw_edge_legend(surface, legend_x, legend_y)
