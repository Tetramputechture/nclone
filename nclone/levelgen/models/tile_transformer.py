"""Phase 1: tile-stage transformer over the 42x23 grid.

Two backbones behind one config (``mode``):
  * ``maskgit`` (primary): bidirectional masked-token prediction with parallel
    iterative (cosine-schedule) decoding -- better global 2D coherence, fast sampling.
  * ``gpt`` (ablation baseline): causal raster-order next-tile prediction.

Conditioning (source / difficulty band / subset flag / mechanic multi-hot) is reduced
to a single global vector added to every token embedding -- a simplification of the
plan's "prepended condition tokens" that composes cleanly with the fixed 2D grid -- and
trained with classifier-free-guidance dropout so fidelity/control trades off at sample
time. Tile values 0..37 occupy ids 0..37; id 38 is the MASK token (MaskGIT only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokenizer import DIFFICULTY_BANDS, MECHANIC_TAGS, SOURCES

GRID_H, GRID_W = 23, 42
NUM_TILES = GRID_H * GRID_W
N_TILE_VALUES = 38  # tile ids 0..37
MASK_ID = N_TILE_VALUES  # 38


@dataclass
class TileConfig:
    mode: str = "maskgit"  # "maskgit" | "gpt"
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    cfg_dropout: float = 0.12  # prob of dropping the condition during training
    vocab: int = N_TILE_VALUES + 1  # +MASK


class _Block(nn.Module):
    def __init__(self, cfg: TileConfig, causal: bool):
        super().__init__()
        self.causal = causal
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout)
        )

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TileTransformer(nn.Module):
    def __init__(self, cfg: TileConfig):
        super().__init__()
        self.cfg = cfg
        self.causal = cfg.mode == "gpt"
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.row_emb = nn.Embedding(GRID_H, cfg.d_model)
        self.col_emb = nn.Embedding(GRID_W, cfg.d_model)
        # conditioning
        self.src_emb = nn.Embedding(len(SOURCES), cfg.d_model)
        self.diff_emb = nn.Embedding(len(DIFFICULTY_BANDS), cfg.d_model)
        self.subset_emb = nn.Embedding(2, cfg.d_model)
        self.mech_proj = nn.Linear(len(MECHANIC_TAGS), cfg.d_model)
        self.blocks = nn.ModuleList([_Block(cfg, self.causal) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        rows = torch.arange(NUM_TILES) // GRID_W
        cols = torch.arange(NUM_TILES) % GRID_W
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _cond(self, cond: Optional[dict], B: int, device, drop: bool):
        if cond is None:
            return torch.zeros(B, self.cfg.d_model, device=device)
        c = (
            self.src_emb(cond["source"].to(device))
            + self.diff_emb(cond["difficulty"].to(device))
            + self.subset_emb(cond["subset"].to(device))
            + self.mech_proj(cond["mech"].to(device))
        )
        if drop and self.training and self.cfg.cfg_dropout > 0:
            keep = (torch.rand(B, 1, device=device) > self.cfg.cfg_dropout).float()
            c = c * keep
        return c

    def _embed(self, ids, cond, drop):
        B, L = ids.shape
        x = self.tok_emb(ids)
        x = x + self.row_emb(self.rows[:L]) + self.col_emb(self.cols[:L])
        x = x + self._cond(cond, B, ids.device, drop).unsqueeze(1)
        return x

    def forward(self, ids, cond=None, drop=True):
        x = self._embed(ids, cond, drop)
        attn_mask = None
        if self.causal:
            L = ids.shape[1]
            attn_mask = torch.triu(torch.full((L, L), float("-inf"), device=ids.device), 1)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.head(self.ln_f(x))

    # ---------------------------------------------------------------- losses
    def loss(self, tiles, cond=None):
        """Training loss for the configured mode. ``tiles``: (B, 966) long in 0..37."""
        if self.cfg.mode == "gpt":
            logits = self.forward(tiles, cond)
            # predict token t from < t (shift)
            return F.cross_entropy(logits[:, :-1].reshape(-1, self.cfg.vocab), tiles[:, 1:].reshape(-1))
        # maskgit
        B, L = tiles.shape
        ratio = torch.rand(B, 1, device=tiles.device)
        ratio = torch.cos(0.5 * math.pi * (1 - ratio))  # cosine masking schedule
        mask = torch.rand(B, L, device=tiles.device) < ratio
        mask[:, 0] |= ~mask.any(dim=1)  # ensure at least one masked
        inp = torch.where(mask, torch.full_like(tiles, MASK_ID), tiles)
        logits = self.forward(inp, cond)
        loss = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab), tiles.reshape(-1), reduction="none"
        )
        loss = (loss * mask.reshape(-1).float()).sum() / mask.sum().clamp_min(1)
        return loss

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, n, cond=None, steps=12, temperature=1.0, cfg_scale=1.0, device=None, generator=None):
        device = device or next(self.parameters()).device
        if self.cfg.mode == "gpt":
            return self._sample_gpt(n, cond, temperature, device, generator)
        return self._sample_maskgit(n, cond, steps, temperature, cfg_scale, device, generator)

    def _apply_cfg(self, ids, cond, cfg_scale):
        if cond is None or cfg_scale == 1.0:
            return self.forward(ids, cond, drop=False)
        cond_logits = self.forward(ids, cond, drop=False)
        uncond_logits = self.forward(ids, None, drop=False)
        return uncond_logits + cfg_scale * (cond_logits - uncond_logits)

    @torch.no_grad()
    def _sample_maskgit(self, n, cond, steps, temperature, cfg_scale, device, generator):
        ids = torch.full((n, NUM_TILES), MASK_ID, dtype=torch.long, device=device)
        for step in range(steps):
            logits = self._apply_cfg(ids, cond, cfg_scale)[:, :, :N_TILE_VALUES]  # forbid MASK output
            probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
            pred = torch.multinomial(probs.reshape(-1, N_TILE_VALUES), 1, generator=generator).reshape(n, NUM_TILES)
            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            is_mask = ids == MASK_ID
            conf = torch.where(is_mask, conf, torch.full_like(conf, -1.0))
            # cosine schedule: how many to keep masked after this step
            r = (step + 1) / steps
            keep_masked = int(math.floor(NUM_TILES * math.cos(0.5 * math.pi * r)))
            ids = torch.where(is_mask, pred, ids)
            if keep_masked > 0:
                # re-mask the lowest-confidence freshly-filled positions
                thresh = torch.topk(conf, keep_masked, dim=-1, largest=False).values[:, -1:]
                remask = (conf <= thresh) & is_mask
                ids = torch.where(remask, torch.full_like(ids, MASK_ID), ids)
        # any leftover masks -> argmax fill
        leftover = ids == MASK_ID
        if leftover.any():
            logits = self._apply_cfg(ids, cond, cfg_scale)[:, :, :N_TILE_VALUES]
            ids = torch.where(leftover, logits.argmax(-1), ids)
        return ids

    @torch.no_grad()
    def _sample_gpt(self, n, cond, temperature, device, generator):
        # grow the context one tile at a time; read the logits at the last position.
        out = torch.empty((n, 0), dtype=torch.long, device=device)
        for pos in range(NUM_TILES):
            ctx = torch.cat([out, torch.full((n, 1), MASK_ID, dtype=torch.long, device=device)], dim=1)
            logits = self.forward(ctx, cond, drop=False)[:, -1, :N_TILE_VALUES]
            probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            out = torch.cat([out, nxt.unsqueeze(1)], dim=1)
        return out


__all__ = ["TileTransformer", "TileConfig", "GRID_H", "GRID_W", "NUM_TILES", "MASK_ID"]
