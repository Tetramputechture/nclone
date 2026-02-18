"""Implicit Q-Learning (IQL) for discrete action spaces.

Implements IQL (Kostrikov et al., ICLR 2022) with:
    - Expectile regression for V-network
    - Standard Bellman loss for Q-network
    - Advantage-weighted regression (AWR) for policy extraction
    - Polyak-averaged target networks

Reference: https://arxiv.org/abs/2110.06169
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks.value_networks import IQLNetworks, IQLTargetNetworks, NUM_ACTIONS

logger = logging.getLogger(__name__)


@dataclass
class IQLConfig:
    """Hyperparameters for IQL training."""

    expectile_tau: float = 0.9
    discount: float = 0.99
    beta: float = 3.0  # AWR temperature
    target_update_tau: float = 0.005
    learning_rate: float = 3e-4
    batch_size: int = 1024
    hidden_dim: int = 512
    use_tile_encoder: bool = True
    max_grad_norm: float = 1.0
    clip_advantage: float = 100.0  # Max exp(beta * A) to prevent overflow
    use_amp: bool = False  # Enable mixed-precision training


@dataclass
class IQLMetrics:
    """Training metrics from a single IQL update step."""

    v_loss: float = 0.0
    q_loss: float = 0.0
    policy_loss: float = 0.0
    mean_v: float = 0.0
    mean_q: float = 0.0
    mean_advantage: float = 0.0
    mean_reward: float = 0.0


class IQLTrainer:
    """IQL trainer for discrete action spaces.

    Trains V(s), Q(s,a), and pi(a|s) from an offline dataset of
    expert demonstrations.
    """

    def __init__(self, config: IQLConfig, device: str = "cpu"):
        """Initialize IQL trainer.

        Args:
            config: IQL hyperparameters.
            device: Torch device ('cpu' or 'cuda').
        """
        self.config = config
        self.device = torch.device(device)

        # Online networks
        self.networks = IQLNetworks(
            hidden_dim=config.hidden_dim,
            use_tile_encoder=config.use_tile_encoder,
        ).to(self.device)

        # Target networks (Polyak-averaged copy)
        self.target_networks = IQLTargetNetworks(self.networks).to(self.device)

        # Optimizers (separate for V, Q, policy for stability)
        self.v_optimizer = torch.optim.Adam(
            self._v_parameters(), lr=config.learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            self._q_parameters(), lr=config.learning_rate
        )
        self.policy_optimizer = torch.optim.Adam(
            self._policy_parameters(), lr=config.learning_rate
        )

        # Mixed-precision training
        self._use_amp = config.use_amp and self.device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        self._step = 0

    def _v_parameters(self):
        """Parameters for V-network (shared encoder + V head)."""
        params = list(self.networks.agent_encoder.parameters())
        params += list(self.networks.combiner.parameters())
        params += list(self.networks.v_head.parameters())
        if self.networks.tile_encoder is not None:
            params += list(self.networks.tile_encoder.parameters())
        return params

    def _q_parameters(self):
        """Parameters for Q-network head only."""
        return list(self.networks.q_head.parameters())

    def _policy_parameters(self):
        """Parameters for policy head only."""
        return list(self.networks.policy_head.parameters())

    def _expectile_loss(
        self, diff: torch.Tensor, tau: float
    ) -> torch.Tensor:
        """Asymmetric squared loss for expectile regression.

        L(u) = |tau - 1(u < 0)| * u^2

        Args:
            diff: Q_target - V (can be positive or negative).
            tau: Expectile parameter (0.5 = standard MSE, >0.5 = upper expectile).

        Returns:
            Scalar loss value.
        """
        weight = torch.where(diff > 0, tau, 1 - tau)
        return (weight * diff.pow(2)).mean()

    def _build_next_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build a batch dict for next-state evaluation.

        Swaps 'state' -> 'next_state' and 'mine_features' -> 'next_mine_features'
        while keeping level data unchanged.
        """
        next_batch = {
            "state": batch["next_state"],
            "mine_features": batch["next_mine_features"],
        }
        if "tile_grid" in batch:
            next_batch["tile_grid"] = batch["tile_grid"]
            next_batch["switch_overlay"] = batch["switch_overlay"]
            next_batch["exit_overlay"] = batch["exit_overlay"]
        return next_batch

    def update(self, batch: Dict[str, torch.Tensor]) -> IQLMetrics:
        """Perform one IQL update step on a batch of transitions.

        Supports mixed-precision training when ``config.use_amp`` is True.

        Args:
            batch: Dictionary with keys:
                state (B, 11), next_state (B, 11), action (B,), reward (B,),
                done (B,), mine_features (B, 24), next_mine_features (B, 24),
                tile_grid (B, 42, 23), switch_overlay (B, 42, 23),
                exit_overlay (B, 42, 23)

        Returns:
            IQLMetrics with loss values and diagnostics.
        """
        # Move batch to device if needed
        batch = {k: v.to(self.device) for k, v in batch.items()}

        actions = batch["action"]  # (B,)
        rewards = batch["reward"]  # (B,)
        dones = batch["done"]  # (B,)

        amp_ctx = torch.amp.autocast("cuda", enabled=self._use_amp)

        # === Step 1: Q-network loss ===
        with amp_ctx:
            q_all = self.networks.q_values(batch)  # (B, 6)
            q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

            next_batch = self._build_next_batch(batch)
            with torch.no_grad():
                v_next = self.target_networks.value(next_batch).squeeze(1)
                q_target = rewards + self.config.discount * v_next * (1 - dones)

            q_loss = F.mse_loss(q_taken, q_target)

        self.q_optimizer.zero_grad()
        self._scaler.scale(q_loss).backward()
        self._scaler.unscale_(self.q_optimizer)
        torch.nn.utils.clip_grad_norm_(self._q_parameters(), self.config.max_grad_norm)
        self._scaler.step(self.q_optimizer)

        # === Step 2: V-network loss (expectile regression) ===
        with amp_ctx:
            v_pred = self.networks.value(batch).squeeze(1)  # (B,)

            with torch.no_grad():
                q_target_for_v = self.target_networks.q_values(batch)
                q_taken_target = q_target_for_v.gather(1, actions.unsqueeze(1)).squeeze(1)

            diff = q_taken_target - v_pred
            v_loss = self._expectile_loss(diff, self.config.expectile_tau)

        self.v_optimizer.zero_grad()
        self._scaler.scale(v_loss).backward()
        self._scaler.unscale_(self.v_optimizer)
        torch.nn.utils.clip_grad_norm_(self._v_parameters(), self.config.max_grad_norm)
        self._scaler.step(self.v_optimizer)

        # === Step 3: Policy loss (advantage-weighted regression) ===
        with amp_ctx:
            with torch.no_grad():
                v_for_adv = self.networks.value(batch).squeeze(1)
                q_for_adv = self.networks.q_values(batch)
                q_taken_for_adv = q_for_adv.gather(1, actions.unsqueeze(1)).squeeze(1)
                advantage = q_taken_for_adv - v_for_adv
                weights = torch.exp(self.config.beta * advantage).clamp(
                    max=self.config.clip_advantage
                )

            logits = self.networks.policy_logits(batch)  # (B, 6)
            log_probs = F.log_softmax(logits, dim=-1)
            log_prob_taken = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            policy_loss = -(weights * log_prob_taken).mean()

        self.policy_optimizer.zero_grad()
        self._scaler.scale(policy_loss).backward()
        self._scaler.unscale_(self.policy_optimizer)
        torch.nn.utils.clip_grad_norm_(
            self._policy_parameters(), self.config.max_grad_norm
        )
        self._scaler.step(self.policy_optimizer)

        # Single scaler update per iteration (covers all three optimizer steps)
        self._scaler.update()

        # === Step 4: Update target networks ===
        self.target_networks.soft_update(self.networks, self.config.target_update_tau)

        self._step += 1

        return IQLMetrics(
            v_loss=v_loss.item(),
            q_loss=q_loss.item(),
            policy_loss=policy_loss.item(),
            mean_v=v_pred.mean().item(),
            mean_q=q_taken.mean().item(),
            mean_advantage=advantage.mean().item(),
            mean_reward=rewards.mean().item(),
        )

    @property
    def step(self) -> int:
        return self._step

    def state_dict(self) -> Dict:
        """Get full trainer state for checkpointing."""
        sd = {
            "networks": self.networks.state_dict(),
            "target_networks": self.target_networks.v_net.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "step": self._step,
            "config": {
                "expectile_tau": self.config.expectile_tau,
                "discount": self.config.discount,
                "beta": self.config.beta,
                "target_update_tau": self.config.target_update_tau,
                "learning_rate": self.config.learning_rate,
                "hidden_dim": self.config.hidden_dim,
                "use_tile_encoder": self.config.use_tile_encoder,
                "use_amp": self.config.use_amp,
            },
        }
        if self._use_amp:
            sd["scaler"] = self._scaler.state_dict()
        return sd

    def load_state_dict(self, state: Dict):
        """Restore trainer state from checkpoint."""
        self.networks.load_state_dict(state["networks"])
        self.target_networks.v_net.load_state_dict(state["target_networks"])
        self.v_optimizer.load_state_dict(state["v_optimizer"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self._step = state["step"]
        if self._use_amp and "scaler" in state:
            self._scaler.load_state_dict(state["scaler"])
