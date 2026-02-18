"""Two-phase PBRS for levels with switch and exit goals.

This module provides the ``TwoPhaseValuePBRS`` class and
``create_two_phase_pbrs`` factory.  With the surface-graph approach both
phases (switch and exit d* arrays) live in a single ``.npz`` file, so
``TwoPhaseValuePBRS`` is a thin wrapper around ``SurfaceGraphPBRS``.

The ``create_two_phase_pbrs`` factory is the entry point used by
``base_environment._try_load_value_function_pbrs``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .precomputed_pbrs import SurfaceGraphPBRS

logger = logging.getLogger(__name__)


class TwoPhaseValuePBRS:
    """Two-phase PBRS using a precomputed surface graph.

    Wraps :class:`SurfaceGraphPBRS` which already handles phase selection
    via the ``switch_activated`` key in the state dictionary.

    This class exists for backward compatibility with the environment
    integration that expects a ``TwoPhaseValuePBRS`` type.
    """

    def __init__(self, data_path: str) -> None:
        """Load precomputed surface graph PBRS data.

        Args:
            data_path: Path to ``{level}_surface_pbrs.npz``.
        """
        self._inner = SurfaceGraphPBRS(data_path)

    def get_potential(self, state: Dict[str, Any]) -> float:
        """Get PBRS potential for the current game state.

        Dispatches to switch or exit phase based on
        ``state["switch_activated"]``.

        Args:
            state: Game state dict (same keys as
                   :meth:`SurfaceGraphPBRS.get_potential`).

        Returns:
            Potential value (approx. range [-1, C_PHASE]).
        """
        return self._inner.get_potential(state)

    def get_gradient(
        self, state: Dict[str, Any], epsilon: float = 6.0,
    ) -> Tuple[float, float]:
        """Estimate potential gradient at state."""
        return self._inner.get_gradient(state, epsilon)

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics for logging."""
        return self._inner.get_statistics()


def create_two_phase_pbrs(
    level_id: str,
    value_function_dir: str = "value_functions",
    **kwargs,
) -> TwoPhaseValuePBRS:
    """Factory function to create two-phase PBRS from a level ID.

    Looks for ``{level_id}_surface_pbrs.npz`` in *value_function_dir*.
    Falls back to the legacy naming ``{level_id}_switch.npz`` /
    ``{level_id}_exit.npz`` pattern -- if those files exist the caller
    should use the old code path (not this function).

    Args:
        level_id: Level identifier (e.g. ``"006"``, ``"rampjump"``).
        value_function_dir: Directory containing precomputed files.
        **kwargs: Ignored (for backward compatibility with old call-sites
                  that passed ``interpolation_method``/``blend_mode``).

    Returns:
        ``TwoPhaseValuePBRS`` instance.

    Raises:
        FileNotFoundError: If the precomputed file does not exist.
    """
    data_path = os.path.join(
        value_function_dir, f"{level_id}_surface_pbrs.npz")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Surface graph PBRS file not found: {data_path}")

    return TwoPhaseValuePBRS(data_path)


def create_neural_pbrs(
    checkpoint_path: str,
    tile_grid: Optional[Any] = None,
    switch_pos: Optional[Tuple[float, float]] = None,
    exit_pos: Optional[Tuple[float, float]] = None,
    mine_positions: Optional[List[Tuple[float, float, int]]] = None,
    headless: Optional[Any] = None,
    device: str = "cpu",
) -> "NeuralValuePBRS":
    """Factory function to create neural IQL V(s) PBRS adapter.

    Creates a ``NeuralValuePBRS`` instance from a frozen IQL V-network
    checkpoint and optionally initializes it with level data.

    Args:
        checkpoint_path: Path to exported V-network file (.pt).
        tile_grid: Optional (42, 23) tile grid for immediate level setup.
        switch_pos: Optional (x, y) switch position for immediate level setup.
        exit_pos: Optional (x, y) exit position for immediate level setup.
        mine_positions: Optional list of (x, y, entity_type) for mines.
        headless: Optional reference to headless simulator.
        device: Torch device for inference.

    Returns:
        ``NeuralValuePBRS`` instance.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"IQL V-network checkpoint not found: {checkpoint_path}"
        )

    from .neural_value_pbrs import NeuralValuePBRS

    adapter = NeuralValuePBRS(checkpoint_path, device=device)

    # Set level data if provided
    if tile_grid is not None and switch_pos is not None and exit_pos is not None:
        adapter.set_level(
            tile_grid=tile_grid,
            switch_pos=switch_pos,
            exit_pos=exit_pos,
            mine_positions=mine_positions or [],
            headless=headless,
        )

    return adapter
