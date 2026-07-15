"""Forward distance curriculum: grow spawn->switch->exit distance at the frontier.

The agent's success rate drops steeply as the objective distance grows. The fix is
a FORWARD auto-curriculum (not reverse / not start-near-goal): sample levels by
SFL learnability ``p(1-p)`` — which peaks at the agent's current frontier — and,
among equally-learnable levels, prefer the ones with the LONGEST route. As the
agent masters short levels (``p -> 1`` so learnability -> 0) they drop out and the
frontier — and the preferred distance — moves outward on its own. No demos, no
reverse curriculum.

This reuses the existing distance signal (``free_space_difficulty.route_cost``,
the same proxy the CME archive bins on) and emits weights keyed by ``content_id``,
exactly what the CME parent sampler / npp-rl PLR consume. See
docs/LONG_HORIZON_DISTANCE_HANDOFF.md (Workstream C).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .free_space_difficulty import route_cost


def objective_distance(map_data: Sequence[int]) -> float:
    """Spawn->switch->exit route cost (the canonical objective distance).

    Uses the cheap free-space grid cost (``route_cost``); ``inf`` when no route
    exists. This is the same difficulty proxy the CME archive bands on, so the
    curriculum and the archive share one distance definition.
    """
    return route_cost(list(map_data))


def learnability(p: float) -> float:
    """SFL learnability ``p(1-p)`` in [0, 0.25]; peaks at the frontier (p=0.5)."""
    p = min(max(float(p), 0.0), 1.0)
    return p * (1.0 - p)


@dataclass
class LevelStat:
    """One candidate level for the sampler."""

    content_id: int
    distance: float                 # objective_distance (route cost); inf = no route
    solve_rate: Optional[float] = None  # agent p; None = unseen


def frontier_weights(
    levels: Sequence[LevelStat],
    *,
    distance_pref: float = 0.5,
    unseen_p: float = 0.5,
    normalize: bool = True,
) -> Dict[int, float]:
    """Forward-curriculum sampling weights keyed by ``content_id``.

    ``weight = learnability(p) * (1 + distance_pref * d_norm)`` where ``d_norm`` is
    the level's distance normalized to [0, 1] over the finite-distance candidates.
    The learnability factor keeps sampling at the frontier; the distance factor
    biases — among equally learnable levels — toward longer routes, so the frontier
    drifts outward as the agent improves.

    Args:
        distance_pref: strength of the longer-is-preferred bias (0 = pure
            learnability; larger = push distance harder). Kept modest by default so
            it nudges rather than overriding learnability (never jump to unlearnable
            far levels).
        unseen_p: pseudo solve-rate for levels with ``solve_rate is None`` so unseen
            levels still get explored (PLR staleness intuition). Default 0.5 =
            maximum learnability, i.e. treat unseen as frontier candidates.
        normalize: scale weights to sum to 1 (skipped if the total is 0).

    Levels with infinite/NaN distance get weight 0 (no route -> not trainable here).
    """
    finite = [lv.distance for lv in levels if lv.distance < float("inf")]
    d_max = max(finite) if finite else 1.0
    d_max = d_max if d_max > 0 else 1.0

    weights: Dict[int, float] = {}
    for lv in levels:
        if not (lv.distance < float("inf")):
            weights[lv.content_id] = 0.0
            continue
        p = unseen_p if lv.solve_rate is None else lv.solve_rate
        d_norm = lv.distance / d_max
        weights[lv.content_id] = learnability(p) * (1.0 + distance_pref * d_norm)

    if normalize:
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total
    return weights


def frontier_distance(
    levels: Sequence[LevelStat], *, mastered_p: float = 0.8
) -> float:
    """Diagnostic: the longest distance the agent has 'mastered' (``p >= mastered_p``).

    Track this over training — it should climb as the curriculum pushes the frontier
    outward. Returns 0.0 if nothing is mastered yet.
    """
    mastered = [
        lv.distance
        for lv in levels
        if lv.distance < float("inf")
        and lv.solve_rate is not None
        and lv.solve_rate >= mastered_p
    ]
    return max(mastered) if mastered else 0.0
