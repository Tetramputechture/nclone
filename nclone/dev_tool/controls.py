"""Declarative control rows shared by the panel (mouse) and the keyboard.

A :class:`Row` is one line in the dev panel. Interactive rows (toggle / action /
field) carry an ``on_activate`` callback and an optional ``key`` hotkey, so a
panel click and a key press dispatch through the exact same callback. Header /
status rows are non-interactive labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

# Row kinds.
HEADER = "header"
STATUS = "status"
TOGGLE = "toggle"
ACTION = "action"
FIELD = "field"


@dataclass
class Row:
    kind: str
    label: str = ""
    key_label: str = ""
    key: Optional[int] = None
    get_state: Optional[Callable[[], bool]] = None      # toggle: on/off
    on_activate: Optional[Callable[[], None]] = None     # toggle/action/field
    get_value: Optional[Callable[[], str]] = None        # field display value
    field_name: Optional[str] = None                     # field focus key

    @property
    def interactive(self) -> bool:
        return self.kind in (TOGGLE, ACTION, FIELD)


def dispatch_key(rows: List[Row], key: int) -> bool:
    """Activate the first interactive row bound to ``key``. Returns True if hit."""
    for row in rows:
        if row.interactive and row.key == key and row.on_activate is not None:
            row.on_activate()
            return True
    return False
