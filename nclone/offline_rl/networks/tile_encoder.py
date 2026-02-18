"""Tile grid CNN encoder with entity overlay channels.

Encodes the 42x23 tile grid + switch/exit overlays into a fixed-size
level representation vector. All 18 input channels are static per-level,
so the CNN output can be computed once and cached across transitions.
"""

import torch
import torch.nn as nn

NUM_TILE_TYPES = 38
TILE_EMBED_DIM = 16
GRID_W = 42
GRID_H = 23
# 16 tile embed + 1 switch overlay + 1 exit overlay = 18 channels
CNN_IN_CHANNELS = TILE_EMBED_DIM + 2
LEVEL_REPR_DIM = 128


class TileEncoder(nn.Module):
    """CNN encoder for level geometry and goal overlays.

    Input channels (all static per-level):
        - 16 channels: tile type embeddings via nn.Embedding(38, 16)
        - 1 channel: switch overlay (1.0 at exit switch cell)
        - 1 channel: exit overlay (1.0 at exit door cell)

    Output: (batch, 128) level representation vector.
    """

    def __init__(self, embed_dim: int = TILE_EMBED_DIM, out_dim: int = LEVEL_REPR_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.tile_embedding = nn.Embedding(NUM_TILE_TYPES, embed_dim)

        in_ch = embed_dim + 2  # +2 for switch/exit overlays
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.LayerNorm([32, GRID_W, GRID_H]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LayerNorm([64, GRID_W, GRID_H]),
            nn.ReLU(),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.LayerNorm([out_dim, GRID_W, GRID_H]),
            nn.ReLU(),
        )

        # Global average pooling is done in forward()

    def forward(
        self,
        tile_grid: torch.Tensor,
        switch_overlay: torch.Tensor,
        exit_overlay: torch.Tensor,
    ) -> torch.Tensor:
        """Encode level geometry and goal positions.

        Args:
            tile_grid: (batch, 42, 23) int64 tensor of tile type IDs (0-37).
            switch_overlay: (batch, 42, 23) float32 with 1.0 at switch cell.
            exit_overlay: (batch, 42, 23) float32 with 1.0 at exit cell.

        Returns:
            (batch, out_dim) level representation vector.
        """
        # Tile embedding: (batch, 42, 23) -> (batch, 42, 23, embed_dim)
        tile_embed = self.tile_embedding(tile_grid)
        # Permute to channels-first: (batch, embed_dim, 42, 23)
        tile_embed = tile_embed.permute(0, 3, 1, 2)

        # Stack overlays: (batch, 1, 42, 23) each
        sw = switch_overlay.unsqueeze(1)
        ex = exit_overlay.unsqueeze(1)

        # Concatenate all channels: (batch, 18, 42, 23)
        x = torch.cat([tile_embed, sw, ex], dim=1)

        # CNN
        x = self.conv(x)  # (batch, out_dim, 42, 23)

        # Global average pooling
        x = x.mean(dim=(2, 3))  # (batch, out_dim)

        return x
