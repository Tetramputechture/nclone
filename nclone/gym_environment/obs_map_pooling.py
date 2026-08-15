"""Block-pooling for the policy-facing global map observations.

The env computes ``hazard_map`` and ``fall_lethality_map`` on a 6 px grid
(168x92) because the ego crops and the PBRS terms slice them at that fidelity.
The POLICY does not need that resolution: their encoders pool to a small fixed
readout (``AdaptiveAvgPool2d``) regardless of input size, so the full-res grids
buy the network nothing while costing 121 KB per map per env in every
payload-scaling operation of the training loop — the GPU transfer, the rollout
buffer write, the VecNormalize copy and the SIL slice all scale with observation
bytes. Emitting them at 12 px (84x46) cuts each to 30 KB.

Pooling must be CONSERVATIVE with respect to danger, never averaging a hazard
away: a block is summarized by its worst sub-cell, so a coarse cell that reads
"safe" is safe throughout. Which reduction that means depends on the channel's
sign convention, hence the explicit per-channel ``modes``:

* hazard SDF (``clip((d - 14) / 24, -1, 1)``) — lower is closer to a mine -> min
* fall survivable margin (``[-1, 1]``)        — lower is more lethal    -> min
* fall drop distance (``[0, 1]``)             — higher is a longer drop -> max

This mirrors the treatment ``cost_to_go_map`` already gets (min-pool the cost
channels, max-pool the reachability mask).

REFLECTION. Pooling commutes exactly with the horizontal mirror --
``M(pool(x)) == pool(M(x))`` -- which the mirror-augmentation tiers depend on:
they SYNTHESISE the mirrored observation rather than letting the sim produce it,
so a non-commuting reduction would silently desynchronise the synthesised map
from the real one. It holds only because the block size divides the width, so a
flip maps whole blocks onto whole blocks (block ``b`` <-> block ``W/ds-1-b``)
with no straddle. A downsample factor that does NOT divide the width would break
this without breaking any shape assertion, so it is pinned by a test.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def pool_channels(
    full: np.ndarray, downsample: int, modes: Sequence[str]
) -> np.ndarray:
    """Block-pool a ``(C, W, H)`` map by ``downsample`` with per-channel modes.

    Args:
        full: ``(C, W, H)`` float32 field on the fine grid.
        downsample: block edge in fine cells. ``<= 1`` returns ``full`` unchanged.
        modes: one of ``"min"`` / ``"max"`` per channel, giving the reduction
            that preserves the channel's worst case.

    Returns:
        ``(C, W // downsample, H // downsample)`` float32. Any trailing cells
        that do not fill a whole block are dropped, matching the cost-map
        downsampler; the map dimensions are chosen to divide evenly.

    Raises:
        ValueError: if ``modes`` does not describe every channel, or names a
            reduction other than min/max.
    """
    if downsample <= 1:
        return full

    if full.ndim != 3:
        raise ValueError(f"expected a (C, W, H) map; got shape {full.shape}")
    n_ch = full.shape[0]
    if len(modes) != n_ch:
        raise ValueError(
            f"modes must give one reduction per channel: {n_ch} channels, "
            f"{len(modes)} modes"
        )

    for ch, mode in enumerate(modes):
        if mode not in ("min", "max"):
            raise ValueError(
                f"channel {ch}: unknown pooling mode {mode!r} (want 'min'/'max')"
            )

    w2 = full.shape[1] // downsample
    h2 = full.shape[2] // downsample
    sub = full[:, : w2 * downsample, : h2 * downsample]

    # Fold the ``downsample**2`` sub-cell offsets together with strided views
    # rather than reducing a ``(.., ds, .., ds)`` reshape over its two block axes.
    # Both give bitwise-identical results (min/max are associative and exact),
    # but numpy's reduction over a length-2 trailing axis runs one tiny reduction
    # per OUTPUT cell -- ~100 ns each, so ~320 us for the 2-channel mine map,
    # which made pooling the single largest line item in the observation. The
    # fold does ``ds**2 - 1`` full-array ufunc passes instead: ~19 us.
    #
    # Channels are grouped by reduction so the fold runs at most twice no matter
    # how many channels there are; the fancy index copies the group (~3 us) and
    # gives the accumulator its own contiguous buffer to write into.
    pooled = np.empty((n_ch, w2, h2), dtype=np.float32)
    for reducer, want in ((np.minimum, "min"), (np.maximum, "max")):
        chans = [ch for ch, mode in enumerate(modes) if mode == want]
        if not chans:
            continue
        group = sub[chans]
        acc = group[:, 0::downsample, 0::downsample].copy()
        for i in range(downsample):
            for j in range(downsample):
                if i or j:
                    reducer(acc, group[:, i::downsample, j::downsample], out=acc)
        pooled[chans] = acc
    return pooled
