"""Input sources that drive the ninja: one per frame -> (hor_input, jump_input).

hor_input is in {-1, 0, 1} and jump_input in {0, 1}, exactly what ``sim.tick``
takes. The abstraction lets the render loop swap live keyboard control for
scripted or recorded playback without changing the loop itself.
"""

from __future__ import annotations


class InputSource:
    """Base class. ``poll`` returns the input for the current frame."""

    name = "input"

    def poll(self) -> tuple[int, int]:
        return 0, 0

    def exhausted(self) -> bool:
        """True when a finite source (for example a replay) has no more inputs."""
        return False

    def reset(self) -> None:
        pass

    def __len__(self) -> int:
        """Number of frames this source will produce, or 0 if it is unbounded."""
        return 0


class KeyboardSource(InputSource):
    """Live keyboard control via pygame key state.

    Left/Right (or A/D) drive horizontal movement; Z, Space, Up or W jump.
    """

    name = "keyboard"

    def poll(self) -> tuple[int, int]:
        import pygame

        keys = pygame.key.get_pressed()
        hor = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            hor -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            hor += 1
        jump = 1 if (
            keys[pygame.K_z] or keys[pygame.K_SPACE]
            or keys[pygame.K_UP] or keys[pygame.K_w]
        ) else 0
        return hor, jump


class HoldSource(InputSource):
    """Emit one fixed (hor, jump) every frame, a scripted drive for recording."""

    name = "hold"

    def __init__(self, hor: int, jump: int):
        self._v = (hor, jump)

    def poll(self) -> tuple[int, int]:
        return self._v


class PatternSource(InputSource):
    """Hold a horizontal direction and pulse jump periodically (a bhop pattern).

    A constant hold only clears trivial levels; a run plus periodic jump solves
    many geometry-rich ones. ``jump_period`` and ``jump_len`` are in frames: jump
    is held for the first ``jump_len`` frames of every ``jump_period`` cycle.
    """

    name = "pattern"

    def __init__(self, hor: int, jump_period: int = 0, jump_len: int = 3):
        self._hor = hor
        self._period = jump_period
        self._len = jump_len
        self._f = 0

    def poll(self) -> tuple[int, int]:
        jump = 1 if (self._period > 0 and (self._f % self._period) < self._len) else 0
        self._f += 1
        return self._hor, jump

    def reset(self) -> None:
        self._f = 0


class ReplaySource(InputSource):
    """Play back a fixed sequence of (hor, jump) inputs.

    Once the sequence runs out the source keeps returning neutral input and
    reports itself :meth:`exhausted`, so the caller can hold a tail.
    """

    name = "replay"

    def __init__(self, inputs):
        self._inputs = list(inputs)
        self._pos = 0

    def poll(self) -> tuple[int, int]:
        if self._pos >= len(self._inputs):
            return 0, 0
        v = self._inputs[self._pos]
        self._pos += 1
        return v

    def exhausted(self) -> bool:
        return self._pos >= len(self._inputs)

    def reset(self) -> None:
        self._pos = 0

    def __len__(self) -> int:
        return len(self._inputs)
