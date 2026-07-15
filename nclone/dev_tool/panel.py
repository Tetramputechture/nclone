"""Draw the dev panel onto the (already widened) display surface + hit-test.

The renderer (``NSimRenderer`` with ``panel_width > 0``) left-aligns the game
frame and skips its own flip; :meth:`DevPanel.draw` paints the reserved right
strip and the caller (the app) performs the single ``pygame.display.flip()``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pygame

from . import layout
from .controls import FIELD, HEADER, STATUS, TOGGLE, Row

# Colours (RGB).
_BG = (24, 26, 32)
_HEADER = (150, 170, 210)
_TEXT = (210, 214, 220)
_MUTED = (130, 134, 142)
_BTN = (48, 52, 62)
_BTN_HOVER = (66, 72, 86)
_ON = (60, 150, 90)
_ON_TEXT = (235, 245, 238)
_FIELD_FOCUS = (90, 110, 160)
_KEYCAP = (96, 100, 110)
_SOLVABLE = (90, 200, 120)
_UNSOLVABLE = (220, 100, 100)


class DevPanel:
    """Stateless-ish renderer for a per-frame list of :class:`Row` objects."""

    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont("monospace", 13)
        self.header_font = pygame.font.SysFont("monospace", 13, bold=True)
        # Filled by draw(); consumed by control_at().
        self._hit: List[Tuple[pygame.Rect, Row]] = []
        self.hover: Optional[Row] = None

    # -- input -------------------------------------------------------------

    def control_at(self, pos) -> Optional[Row]:
        for rect, row in self._hit:
            if rect.collidepoint(pos):
                return row
        return None

    def set_hover(self, pos) -> None:
        self.hover = self.control_at(pos) if pos is not None else None

    # -- drawing -----------------------------------------------------------

    def draw(self, screen, rows: List[Row], focused_field: Optional[str], field_buffer: str) -> None:
        screen_w, screen_h = screen.get_size()
        px, py, pw, ph = layout.panel_rect(screen_w, screen_h)
        pygame.draw.rect(screen, _BG, (px, py, pw, ph))
        cx, cw = layout.content_x(screen_w)

        self._hit = []
        y = layout.PAD
        for row in rows:
            if row.kind == HEADER:
                self._text(screen, row.label.upper(), cx, y + 3, self.header_font, _HEADER)
                y += layout.HEADER_H
            elif row.kind == STATUS:
                color = _TEXT
                if row.label.startswith("status: "):
                    color = _solvability_color(row.label)
                self._text(screen, row.label, cx, y, self.font, color)
                y += layout.LINE_H
            else:
                rect = pygame.Rect(cx, y, cw, layout.ROW_H)
                self._draw_button(screen, row, rect, focused_field, field_buffer)
                self._hit.append((rect, row))
                y += layout.ROW_H + layout.GAP

    def _draw_button(self, screen, row: Row, rect: pygame.Rect, focused_field, field_buffer) -> None:
        on = bool(row.get_state and row.get_state())
        focused = row.kind == FIELD and row.field_name == focused_field
        if on:
            bg = _ON
        elif focused:
            bg = _FIELD_FOCUS
        elif row is self.hover:
            bg = _BTN_HOVER
        else:
            bg = _BTN
        pygame.draw.rect(screen, bg, rect, border_radius=4)

        text_color = _ON_TEXT if on else _TEXT
        label = row.label
        if row.kind == FIELD:
            value = field_buffer if focused else (row.get_value() if row.get_value else "")
            caret = "_" if focused else ""
            label = f"{row.label}: {value}{caret}"
        self._text(screen, label, rect.x + 8, rect.y + 4, self.font, text_color)

        if row.kind == TOGGLE:
            tag = "ON" if on else "off"
            self._text_right(screen, tag, rect.right - 8, rect.y + 4, self.font, text_color)
        elif row.key_label:
            self._keycap(screen, row.key_label, rect)

    def _keycap(self, screen, label: str, rect: pygame.Rect) -> None:
        surf = self.font.render(label, True, _TEXT)
        w = surf.get_width() + 8
        kx = rect.right - w - 4
        cap = pygame.Rect(kx, rect.y + 3, w, rect.height - 6)
        pygame.draw.rect(screen, _KEYCAP, cap, border_radius=3)
        screen.blit(surf, (kx + 4, rect.y + 4))

    def _text(self, screen, text, x, y, font, color) -> None:
        screen.blit(font.render(text, True, color), (x, y))

    def _text_right(self, screen, text, right, y, font, color) -> None:
        surf = font.render(text, True, color)
        screen.blit(surf, (right - surf.get_width(), y))


def _solvability_color(status_line: str):
    if "UNSOLVABLE" in status_line or "ERROR" in status_line:
        return _UNSOLVABLE
    if "SOLVABLE" in status_line:
        return _SOLVABLE
    return _TEXT
