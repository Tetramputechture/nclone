"""Draw the dev panel onto the (already widened) display surface + hit-test.

The renderer (``NSimRenderer`` with ``panel_width > 0``) left-aligns the game
frame and skips its own flip; :meth:`DevPanel.draw` paints the reserved right
strip and the caller (the app) performs the single ``pygame.display.flip()``.

The row list is taller than the window for some map sources, so the panel is
scrollable: :meth:`DevPanel.scroll_by` moves the content, drawing is clipped to
the strip, and only rows that are actually visible are registered for hit
testing (a click can never activate a control scrolled out of view).
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
_SCROLL_TRACK = (34, 37, 45)
_SCROLL_THUMB = (86, 92, 108)


class DevPanel:
    """Stateless-ish renderer for a per-frame list of :class:`Row` objects."""

    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont("monospace", 13)
        self.header_font = pygame.font.SysFont("monospace", 13, bold=True)
        # Filled by draw(); consumed by control_at().
        self._hit: List[Tuple[pygame.Rect, Row]] = []
        self.hover: Optional[Row] = None
        # Scroll offset (px of content hidden above the strip) + the clamp for
        # it, recomputed from the live row list on every draw().
        self.scroll = 0
        self._max_scroll = 0
        self._panel_rect = pygame.Rect(0, 0, 0, 0)

    # -- input -------------------------------------------------------------

    def control_at(self, pos) -> Optional[Row]:
        for rect, row in self._hit:
            if rect.collidepoint(pos):
                return row
        return None

    def set_hover(self, pos) -> None:
        self.hover = self.control_at(pos) if pos is not None else None

    def contains(self, pos) -> bool:
        """True if ``pos`` is inside the panel strip (as of the last draw)."""
        return pos is not None and self._panel_rect.collidepoint(pos)

    def scroll_by(self, delta_px: int) -> None:
        """Scroll the content by ``delta_px`` (positive = further down)."""
        self.scroll = max(0, min(self.scroll + int(delta_px), self._max_scroll))

    # -- drawing -----------------------------------------------------------

    def draw(self, screen, rows: List[Row], focused_field: Optional[str], field_buffer: str) -> None:
        screen_w, screen_h = screen.get_size()
        px, py, pw, ph = layout.panel_rect(screen_w, screen_h)
        self._panel_rect = pygame.Rect(px, py, pw, ph)
        pygame.draw.rect(screen, _BG, self._panel_rect)
        cx, cw = layout.content_x(screen_w)

        # Clamp against the current content: the row list changes with the map
        # source, so a scroll that was valid last frame may not be now.
        self._max_scroll = layout.max_scroll(rows, ph)
        self.scroll = max(0, min(self.scroll, self._max_scroll))

        prev_clip = screen.get_clip()
        screen.set_clip(self._panel_rect)
        self._hit = []
        y = py + layout.PAD - self.scroll
        for row in rows:
            h = layout.row_height(row)
            if y + h < py or y > py + ph:  # scrolled out of view
                y += h
                continue
            if row.kind == HEADER:
                self._text(screen, row.label.upper(), cx, y + 2, self.header_font, _HEADER)
            elif row.kind == STATUS:
                color = _TEXT
                if row.label.startswith("status: "):
                    color = _solvability_color(row.label)
                self._text(screen, row.label, cx, y, self.font, color)
            else:
                rect = pygame.Rect(cx, y, cw, layout.ROW_H)
                self._draw_button(screen, row, rect, focused_field, field_buffer)
                self._hit.append((rect, row))
            y += h
        screen.set_clip(prev_clip)
        self._draw_scrollbar(screen, layout.content_height(rows))

    def _draw_scrollbar(self, screen, content_h: int) -> None:
        if self._max_scroll <= 0:
            return
        panel = self._panel_rect
        x = panel.right - layout.SCROLLBAR_W - 2
        track = pygame.Rect(x, panel.y + layout.PAD, layout.SCROLLBAR_W,
                            panel.height - 2 * layout.PAD)
        pygame.draw.rect(screen, _SCROLL_TRACK, track, border_radius=2)
        thumb_h = max(24, int(track.height * panel.height / max(1, content_h)))
        travel = track.height - thumb_h
        thumb_y = track.y + int(travel * self.scroll / self._max_scroll)
        pygame.draw.rect(screen, _SCROLL_THUMB,
                         (track.x, thumb_y, track.width, thumb_h), border_radius=2)

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
        ty = self._text_y(rect)
        self._text(screen, label, rect.x + 6, ty, self.font, text_color)

        right = rect.right - 6
        if row.kind == TOGGLE:
            tag = "ON" if on else "off"
            self._text_right(screen, tag, right, ty, self.font, text_color)
            # Reserve the widest tag ("off") so the keycap does not jitter when
            # the toggle flips.
            right -= self.font.size("off")[0] + 6
        if row.key_label:
            self._keycap(screen, row.key_label, rect, right)

    def _text_y(self, rect: pygame.Rect) -> int:
        """Baseline-ish y that vertically centres one text line in ``rect``."""
        return rect.y + max(0, (rect.height - self.font.get_height()) // 2)

    def _keycap(self, screen, label: str, rect: pygame.Rect, right: int) -> None:
        surf = self.font.render(label, True, _TEXT)
        w = surf.get_width() + 8
        kx = right - w
        cap = pygame.Rect(kx, rect.y + 2, w, rect.height - 4)
        pygame.draw.rect(screen, _KEYCAP, cap, border_radius=3)
        screen.blit(surf, (kx + 4, self._text_y(rect)))

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
