"""Time-MARGINALIZED threat cost: what a moving hazard does to a static potential.

The problem
-----------
``Phi(x, y, phase)`` is static.  It cannot express "safe at t=10, lethal at
t=40".  Adding a time axis is a 4D blow-up, and it is also unnecessary: PBRS is
policy-invariant for ANY potential (Ng, Harada & Russell 1999), so a time-blind
``Phi`` cannot make the optimal policy wrong.  It can only be slower to learn
from.  The job is therefore not to make ``Phi`` correct in time, but to make it
*point at corridors that are usually safe*.

The statistic
-------------
Per cell, the fraction of a rolled-out window during which that cell is lethal
to the CERTAIN tier — hazards whose motion does not depend on the ninja.  That
is a stationary scalar in ``[0, 1]``, and it enters the flood as a soft cost
multiplier exactly like the static mine density already does
(``_MINE_SOFT_K``).  No time dimension appears anywhere.

Why the marginal, and not a worst case or a quantile
----------------------------------------------------
Measured over official levels carrying real movers, comparing the statistic
computed on the first half of the window against the whole window:

    statistic          half-window max drift
    occupancy (marginal)   p50 0.028   max 0.195
    min over window        p50 0.003   max 1.604
    10th percentile        p50 0.025   max 1.412

The worst case and the quantile are each determined by a *single frame*, so they
are step functions of the window length: usually they do not move at all, and
when they do they jump by the full range of the SDF.  A statistic whose value is
a function of how long you happened to roll is not a property of the level.  The
marginal drifts smoothly and is bounded by construction, which is what makes it
safe to bake into a precompute.

Why a soft cost and not a hard mask
-----------------------------------
On the same levels, the fraction of cells that are lethal for the WHOLE window
is at most 1.5%, while the fraction lethal for *some* of it reaches 21%.  So a
mask thresholded near 1.0 cuts nothing, and a mask thresholded low enough to
matter erases a fifth of the grid — which is precisely the failure that already
cost a spawn once, at a coarser resolution than this grid
(``free_space_distance``'s 20px mine dilation vs a survivable 21.63px standoff).
Moving hazards make that worse, not better: the threat grid is 12px, twice the
flood's cell, so a mask here dilates further still.

Lethality and routability are different questions and this module answers only
the second one.  Nothing here is a safety system; mine and hazard avoidance come
from the observation channels and from dying.

Why the telescoping warning does not apply here
-----------------------------------------------
A one-sided hazard toll on this project once destroyed PBRS telescoping and
taxed winning routes by up to 117% of ``R_EXIT``.  That term was added to the
REWARD — it was not of the form ``gamma*Phi(s') - Phi(s)``, which is exactly why
it did not telescope.  This cost lives INSIDE ``Phi``, in the flood's edge
weights, so the shaping term is still a potential difference and still
telescopes to zero over any cycle by construction.  The static mine cost has
lived there since the beginning without that failure mode.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .free_space_distance import _CELL, _W, _H

#: Emitted grid of ``Simulator.threat_field`` — 12px cells over the playfield.
_T_W, _T_H = 84, 46
_T_CELL = 12.0
_T_ORIGIN_PX = 24.0

#: Tier 0 of the threat field: hazards with an exact forecast, i.e. those whose
#: motion is independent of the ninja.  Tier 1 (REACTIVE) is deliberately
#: excluded — a reactive threat follows the agent, so "the fraction of time it
#: occupies a cell" is a statement about the agent's own policy, not about the
#: level, and marginalising it would bake one rollout's behaviour into a
#: precompute.  Reactive threats belong in the observation, where they already
#: are, as the worst-case envelope.
_TIER_CERTAIN = 0
_N_TIERS = 2

#: Frames rolled forward to estimate the marginal.  600 is ~2x the drift
#: measurement's convergence window and costs ~10ms per level, which is three
#: orders of magnitude under the precompute budget.
DEFAULT_WINDOW_FRAMES = 600

#: Multiplier strength: a permanently-lethal cell costs ``1 + K`` times a clear
#: one.  Deliberately far below ``_MINE_SOFT_K`` (8.0): a mine is lethal on
#: every frame and a mover is not, so the same weight would price an
#: intermittent hazard like a permanent one.
THREAT_SOFT_K = 3.0

_ENV_FLAG = "NCLONE_THREAT_OCCUPANCY"


def threat_occupancy_enabled(default: bool = False) -> bool:
    """Opt-in, because it changes every edge weight the flood relaxes."""
    v = os.environ.get(_ENV_FLAG)
    if v is None:
        return default
    return v not in ("0", "false", "False", "")


def _flood_cell_to_threat_cell():
    """Index maps from the 6px flood grid to the 12px threat grid.

    The two grids do not share an origin: the flood grid starts at sim pixel 0
    and so includes the 24px border, while the threat grid starts at the
    playfield edge.  Getting this wrong shifts the whole hazard field by two
    tiles, which reads as "the cost model is miscalibrated" rather than as an
    off-by-four, so it is computed once here and reused.
    """
    cx = (np.arange(_W) * _CELL + _CELL / 2.0 - _T_ORIGIN_PX) / _T_CELL
    cy = (np.arange(_H) * _CELL + _CELL / 2.0 - _T_ORIGIN_PX) / _T_CELL
    return (np.clip(np.floor(cx), 0, _T_W - 1).astype(np.int64),
            np.clip(np.floor(cy), 0, _T_H - 1).astype(np.int64))


def marginal_occupancy(map_data, frames: int = DEFAULT_WINDOW_FRAMES,
                       ) -> Optional[np.ndarray]:
    """``(_W, _H)`` float32 in ``[0, 1]``: fraction of the window a cell is lethal.

    Rolled with a null input.  The ninja is not steered and may die; the roll
    stops there, because past a death the simulator is no longer advancing the
    level the agent will play.  Cells outside the playfield read 0.

    Returns ``None`` if the level has no certain-tier threat at all, which lets
    the caller skip the whole mechanism on the ~majority of levels rather than
    multiply every edge by 1.0.
    """
    try:
        import nclone_native
        sim = nclone_native.Simulator.load([int(v) for v in map_data])
    except Exception:
        return None

    acc = np.zeros((_T_W, _T_H), np.float64)
    n = 0
    try:
        for _ in range(int(frames)):
            sim.tick(0, False)
            raw = np.frombuffer(sim.threat_field([]), dtype="<f4")
            cert = raw.reshape(_N_TIERS, _T_W, _T_H)[_TIER_CERTAIN]
            # sdf < 0 is inside the contact-death boundary; the SDF saturates to
            # exactly 1.0 beyond the influence radius, so this is a clean test.
            acc += (cert < 0.0)
            n += 1
            if sim.has_died() or sim.has_won():
                break
    except Exception:
        if n == 0:
            return None
    if n == 0:
        return None
    occ_t = (acc / n).astype(np.float32)
    if not occ_t.any():
        return None
    ix, iy = _flood_cell_to_threat_cell()
    return np.ascontiguousarray(occ_t[ix[:, None], iy[None, :]])


def step_multiplier(occ: Optional[np.ndarray], k: float = THREAT_SOFT_K):
    """``1 + k * occ``, the factor a flood step into each cell is scaled by.

    Separated from the flood so the weighting can be swept without touching the
    relaxation, and so a caller can inspect the field it is about to pay for.
    """
    if occ is None:
        return None
    return (1.0 + float(k) * occ).astype(np.float64)
