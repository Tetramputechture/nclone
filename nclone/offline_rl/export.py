"""Export frozen V-network from IQL checkpoint for PBRS use.

Extracts the V-network components (agent encoder + tile encoder + combiner + V-head)
into a standalone TorchScript or state_dict file for efficient inference
during FiGAR-PPO training.

Usage:
    python -m nclone.offline_rl.export \
        --checkpoint iql_checkpoints/iql_best.pt \
        --output v_network.pt \
        [--torchscript]
"""

import argparse
import logging
import os
from typing import Dict

import torch

from .algorithms.iql import IQLConfig
from .networks.value_networks import IQLNetworks

logger = logging.getLogger(__name__)


def export_v_network(
    checkpoint_path: str,
    output_path: str,
    use_torchscript: bool = False,
) -> str:
    """Export the V-network from an IQL checkpoint.

    The exported file contains:
        - V-network state_dict (agent encoder + tile encoder + combiner + V head)
        - Metadata (state format, normalization constants, training config)

    Args:
        checkpoint_path: Path to IQL checkpoint (.pt file).
        output_path: Path for the exported V-network file.
        use_torchscript: If True, export as TorchScript (not yet supported with
            dict inputs; falls back to state_dict).

    Returns:
        Path to the exported file.
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reconstruct config from checkpoint
    ckpt_config = ckpt.get("config", {})
    config = IQLConfig(
        hidden_dim=ckpt_config.get("hidden_dim", 512),
        use_tile_encoder=ckpt_config.get("use_tile_encoder", True),
    )

    # Build network and load weights
    networks = IQLNetworks(
        hidden_dim=config.hidden_dim,
        use_tile_encoder=config.use_tile_encoder,
    )
    networks.load_state_dict(ckpt["networks"])
    networks.eval()

    # Extract V-network components
    v_state = {
        "agent_encoder": networks.agent_encoder.state_dict(),
        "combiner": networks.combiner.state_dict(),
        "v_head": networks.v_head.state_dict(),
    }
    if networks.tile_encoder is not None:
        v_state["tile_encoder"] = networks.tile_encoder.state_dict()

    export_data = {
        "v_network": v_state,
        "config": {
            "hidden_dim": config.hidden_dim,
            "use_tile_encoder": config.use_tile_encoder,
            "discount": ckpt_config.get("discount", 0.99),
            "expectile_tau": ckpt_config.get("expectile_tau", 0.9),
        },
        "training_step": ckpt.get("step", 0),
        "agent_state_dim": 11,
        "mine_feature_dim": 24,
        "num_actions": 6,
    }

    torch.save(export_data, output_path)

    size_mb = os.path.getsize(output_path) / 1e6
    logger.info(
        "Exported V-network: %s (%.1f MB, step %d, tile_encoder=%s)",
        output_path,
        size_mb,
        export_data["training_step"],
        config.use_tile_encoder,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export IQL V-network for PBRS")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="IQL checkpoint path"
    )
    parser.add_argument(
        "--output", type=str, default="v_network.pt", help="Output file path"
    )
    parser.add_argument(
        "--torchscript",
        action="store_true",
        help="Export as TorchScript (experimental)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    export_v_network(args.checkpoint, args.output, use_torchscript=args.torchscript)


if __name__ == "__main__":
    main()
