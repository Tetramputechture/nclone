"""Dataclasses and Protocols for the optimal-route planner.

Each planner produces an :class:`OptimalRoute` from a :class:`LevelGeometry`
(see :mod:`nclone.planning.optimal_route.geometry`). The output includes both
the analytically derived optimal frame count and a concrete per-frame action
sequence so that the result can be replayed and verified in the nclone
simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple


ActionFrame = Tuple[int, bool]
"""A single-frame action: ``(hor_input, jump_input)``.

``hor_input`` is in ``{-1, 0, 1}`` matching the simulator's
:meth:`Simulator.tick` contract; ``jump_input`` is a bool (maps to the ``0/1``
integer the simulator expects).
"""


@dataclass(frozen=True)
class OptimalRoute:
    """Result of planning an optimal route through a trivial map.

    Attributes:
        map_type: Topology classification, e.g. ``"wall_jump_shaft"`` or
            ``"flat_run_corridor"``.
        strategy: Name of the chosen strategy
            (e.g. ``"direct_run"``, ``"wall_sling"``, ``"wj_ping_pong"``).
        optimal_frames: Analytical frame count for the chosen strategy. This
            is the number of simulator ticks between level start and the
            frame the ninja enters the celebration / "won" state.
        action_sequence: Per-frame ``(hor_input, jump_input)`` list that
            realises the chosen strategy.
        verified_in_sim: ``True`` when the action sequence was successfully
            replayed in the sim and the ninja reached the "won" state.
            ``False`` means the analytical plan did not complete the level
            in the simulator (bug; tests should flag this).
        verified_frames: Actual number of frames the replayed sequence took
            to complete the level. ``None`` if verification was skipped.
        notes: Free-form metadata. Planners populate this with geometry
            summaries, candidate strategy timings, and decision margins.
    """

    map_type: str
    strategy: str
    optimal_frames: int
    action_sequence: List[ActionFrame]
    verified_in_sim: bool = False
    verified_frames: Optional[int] = None
    notes: Dict[str, Any] = field(default_factory=dict)


class OptimalRoutePlanner(Protocol):
    """Common interface for all topology-specific planners."""

    def supports(self, geom: "LevelGeometry") -> bool:  # noqa: F821
        """Return ``True`` iff this planner can handle ``geom``."""

    def plan(self, geom: "LevelGeometry") -> OptimalRoute:  # noqa: F821
        """Compute the optimal route for ``geom``.

        Must only be called when :meth:`supports` returned ``True``.
        """
