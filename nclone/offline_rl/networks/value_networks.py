"""IQL value networks with UVFA conditioning.

Architecture:
    Agent state MLP: (35D) -> 256 -> 256
    Tile CNN: (18ch, 42, 23) -> 128
    Combiner: concat(256, 128) = 384 -> 512 -> 512
    Heads: V(s)->1, Q(s,a)->6, pi(a|s)->6

The agent state MLP processes 11D ninja physics + 24D nearest-8-mine features.
The tile CNN processes the static level geometry and goal overlays.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .tile_encoder import TileEncoder, LEVEL_REPR_DIM

AGENT_STATE_DIM = 11
MINE_FEATURE_DIM = 24
TOTAL_AGENT_DIM = AGENT_STATE_DIM + MINE_FEATURE_DIM  # 35
NUM_ACTIONS = 6
HIDDEN_DIM = 512


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2) -> nn.Sequential:
    """Build an MLP with LayerNorm and ReLU activations."""
    layers = []
    current_dim = in_dim
    for _ in range(n_layers):
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ])
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


class AgentStateEncoder(nn.Module):
    """MLP encoder for agent state (11D physics + 24D mine features)."""

    def __init__(self, in_dim: int = TOTAL_AGENT_DIM, out_dim: int = 256):
        super().__init__()
        self.mlp = _build_mlp(in_dim, 256, out_dim, n_layers=2)

    def forward(self, state: torch.Tensor, mine_features: torch.Tensor) -> torch.Tensor:
        """Encode agent state.

        Args:
            state: (batch, 11) ninja physics state.
            mine_features: (batch, 24) nearest-8-mine features.

        Returns:
            (batch, out_dim) agent representation.
        """
        x = torch.cat([state, mine_features], dim=-1)
        return self.mlp(x)


class IQLNetworks(nn.Module):
    """Combined V, Q, and policy networks for IQL.

    Supports both UVFA mode (with tile encoder) and single-level mode
    (tile encoder disabled, agent state only).
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        use_tile_encoder: bool = True,
        agent_out_dim: int = 256,
        level_repr_dim: int = LEVEL_REPR_DIM,
    ):
        """Initialize IQL networks.

        Args:
            hidden_dim: Hidden dimension for combiner and heads.
            use_tile_encoder: If True, use CNN tile encoder (UVFA mode).
            agent_out_dim: Output dimension of agent state encoder.
            level_repr_dim: Output dimension of tile encoder.
        """
        super().__init__()
        self.use_tile_encoder = use_tile_encoder
        self.hidden_dim = hidden_dim

        self.agent_encoder = AgentStateEncoder(out_dim=agent_out_dim)

        if use_tile_encoder:
            self.tile_encoder = TileEncoder(out_dim=level_repr_dim)
            combiner_in = agent_out_dim + level_repr_dim
        else:
            self.tile_encoder = None
            combiner_in = agent_out_dim

        # Shared combiner trunk
        self.combiner = nn.Sequential(
            nn.Linear(combiner_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Value head: V(s) -> scalar
        self.v_head = nn.Linear(hidden_dim, 1)

        # Q-value head: Q(s, a) -> one value per action
        self.q_head = nn.Linear(hidden_dim, NUM_ACTIONS)

        # Policy head: logits for action distribution
        self.policy_head = nn.Linear(hidden_dim, NUM_ACTIONS)

    def _encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Shared encoding: agent state + (optional) level encoding.

        Args:
            batch: Dictionary with keys 'state', 'mine_features', and optionally
                   'tile_grid', 'switch_overlay', 'exit_overlay'.

        Returns:
            (batch_size, hidden_dim) combined representation.
        """
        agent_repr = self.agent_encoder(batch["state"], batch["mine_features"])

        if self.use_tile_encoder and self.tile_encoder is not None:
            level_repr = self.tile_encoder(
                batch["tile_grid"],
                batch["switch_overlay"],
                batch["exit_overlay"],
            )
            combined = torch.cat([agent_repr, level_repr], dim=-1)
        else:
            combined = agent_repr

        return self.combiner(combined)

    def value(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute V(s).

        Returns:
            (batch_size, 1) state values.
        """
        h = self._encode(batch)
        return self.v_head(h)

    def q_values(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute Q(s, a) for all actions.

        Returns:
            (batch_size, 6) Q-values per action.
        """
        h = self._encode(batch)
        return self.q_head(h)

    def policy_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute policy logits for all actions.

        Returns:
            (batch_size, 6) action logits.
        """
        h = self._encode(batch)
        return self.policy_head(h)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all outputs in a single forward pass.

        Returns:
            Tuple of (V, Q, policy_logits) tensors.
        """
        h = self._encode(batch)
        return self.v_head(h), self.q_head(h), self.policy_head(h)


class IQLTargetNetworks(nn.Module):
    """Target networks for IQL (V-target and Q-target).

    Separate module to manage Polyak-averaged target parameters.
    """

    def __init__(self, source: IQLNetworks):
        """Create target networks by copying source network structure.

        Args:
            source: The online IQLNetworks to copy architecture from.
        """
        super().__init__()
        self.v_net = IQLNetworks(
            hidden_dim=source.hidden_dim,
            use_tile_encoder=source.use_tile_encoder,
        )
        # Copy initial weights
        self.v_net.load_state_dict(source.state_dict())

    @torch.no_grad()
    def soft_update(self, source: IQLNetworks, tau: float = 0.005):
        """Polyak-average update from source network.

        Args:
            source: Online network to update from.
            tau: Interpolation factor (0 = no update, 1 = hard copy).
        """
        for tp, sp in zip(self.v_net.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)

    def value(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute target V(s)."""
        return self.v_net.value(batch)

    def q_values(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute target Q(s, a)."""
        return self.v_net.q_values(batch)
