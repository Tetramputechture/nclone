"""`_compute_ego_crop_route_fine`'s vectorized steepest descent is bit-exact.

That function was the single hottest thing inside `env.step()` — 6.2% of self
time for 0.9% of the observation's bytes — because its 8-neighbour scan was a
Python loop allocating six ``(S, S)`` float64 temporaries per iteration. It is
now one stacked ``argmin``.

A speed refactor of an observation channel is exactly the kind of change that
can shift training silently, so this pins the output against a literal
transcription of the replaced loop rather than against hand-written
expectations. Two subtleties it exists to catch:

* Tie-breaking. The loop took the FIRST strict minimum, so the neighbour
  ordering is load-bearing: reordering the offsets changes the output on 60/60
  random tied fields.
* ``inf - inf = nan``. The loop's ``score < best`` is False for nan; ``argmin``
  would SELECT it. Measured, this one is DEFENSIVE rather than load-bearing --
  nan arises only where ``center`` is infinite, which ``live`` already masks to
  zero, so removing the guard changes 0/60 fields today. It is pinned anyway so
  the two forms do not depend on that coupling holding.
"""

import math

import numpy as np
import pytest

from nclone.gym_environment.npp_environment import (
    _EGO_ROUTE_NEIGHBOUR_OFFSETS,
    _EGO_ROUTE_NEIGHBOUR_STEPS,
    _EGO_ROUTE_UNIT_DX,
    _EGO_ROUTE_UNIT_DY,
)


def reference(win, size):
    """Literal transcription of the scalar loop this replaced."""
    center = win[1:-1, 1:-1]
    best = np.full((size, size), np.inf)
    bdx = np.zeros((size, size))
    bdy = np.zeros((size, size))
    with np.errstate(invalid="ignore"):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = win[1 + dx:1 + dx + size, 1 + dy:1 + dy + size]
                step = math.hypot(dx, dy)
                score = (nb - center) / step
                take = score < best
                best = np.where(take, score, best)
                bdx = np.where(take, dx / step, bdx)
                bdy = np.where(take, dy / step, bdy)
    live = np.isfinite(center) & (best < -1e-9)
    return (
        np.where(live, bdx, 0.0).astype(np.float32),
        np.where(live, bdy, 0.0).astype(np.float32),
    )


def vectorized(win, size):
    """The shipped body, lifted out of the method for direct comparison."""
    center = win[1:-1, 1:-1]
    offs = _EGO_ROUTE_NEIGHBOUR_OFFSETS
    steps = _EGO_ROUTE_NEIGHBOUR_STEPS
    with np.errstate(invalid="ignore"):
        nb = np.stack(
            [win[1 + dx:1 + dx + size, 1 + dy:1 + dy + size] for dx, dy in offs]
        )
        score = (nb - center) / steps
    score = np.where(np.isnan(score), np.inf, score)
    k = np.argmin(score, axis=0)
    best = np.take_along_axis(score, k[None], axis=0)[0]
    live = np.isfinite(center) & (best < -1e-9)
    return (
        np.where(live, _EGO_ROUTE_UNIT_DX[k], 0.0).astype(np.float32),
        np.where(live, _EGO_ROUTE_UNIT_DY[k], 0.0).astype(np.float32),
    )


def check(win, size):
    rdx, rdy = reference(win, size)
    vdx, vdy = vectorized(win, size)
    assert np.array_equal(rdx, vdx), "dx channel differs"
    assert np.array_equal(rdy, vdy), "dy channel differs"


SIZES = [5, 21]


@pytest.mark.parametrize("size", SIZES)
def test_smooth_field(size):
    """A plain distance ramp: every cell has a strict descent direction."""
    n = size + 2
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    check((xx + 2.0 * yy).astype(np.float64), size)


@pytest.mark.parametrize("size", SIZES)
def test_all_infinite_field(size):
    """Every difference is inf - inf = nan. The loop selects nothing; so must we."""
    win = np.full((size + 2, size + 2), np.inf)
    dx, dy = vectorized(win, size)
    assert not dx.any() and not dy.any()
    check(win, size)


@pytest.mark.parametrize("size", SIZES)
def test_mixed_finite_and_infinite(size):
    """The real case: unreachable cells scattered through a live field."""
    rng = np.random.default_rng(0)
    n = size + 2
    win = rng.uniform(0, 50, (n, n))
    win[rng.random((n, n)) < 0.35] = np.inf
    check(win, size)


@pytest.mark.parametrize("size", SIZES)
def test_flat_field_ties_everywhere(size):
    """All neighbours tie. Pins the first-wins tie-break AND the offset order."""
    check(np.full((size + 2, size + 2), 7.0), size)


@pytest.mark.parametrize("size", SIZES)
def test_plateau_with_one_escape(size):
    """A single descending neighbour in an otherwise flat plateau."""
    n = size + 2
    win = np.full((n, n), 5.0)
    win[0, 0] = -100.0
    check(win, size)


def test_random_fields_including_ties_and_infinities():
    """Many draws over a coarse value grid, so exact ties are common."""
    size = 21
    n = size + 2
    rng = np.random.default_rng(7)
    for _ in range(60):
        win = rng.integers(0, 4, (n, n)).astype(np.float64)
        win[rng.random((n, n)) < 0.25] = np.inf
        check(win, size)


def test_offsets_match_the_original_loop_order():
    """The order is load-bearing for tie-breaking; pin it explicitly."""
    expected = tuple(
        (dx, dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if not (dx == 0 and dy == 0)
    )
    assert _EGO_ROUTE_NEIGHBOUR_OFFSETS == expected
    assert _EGO_ROUTE_NEIGHBOUR_STEPS.shape == (8, 1, 1)
    for i, (dx, dy) in enumerate(expected):
        h = math.hypot(dx, dy)
        assert _EGO_ROUTE_NEIGHBOUR_STEPS[i, 0, 0] == h
        assert _EGO_ROUTE_UNIT_DX[i] == dx / h
        assert _EGO_ROUTE_UNIT_DY[i] == dy / h
