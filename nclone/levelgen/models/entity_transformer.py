"""Phase 2: entity-stage transformer (tile-conditioned, constrained decode).

Autoregressive decoder over the flat entity token stream (shared
:class:`~nclone.levelgen.tokenizer.Vocab`), cross-attending to a tile-grid encoder so
placement respects geometry (mine on a ledge lip, gold over a gap). Trained on the full
entity vocab to learn natural priors; **constrained decoding** enforces our subset at
sample time.

Constrained decode guarantees subset validity + exact counts *by construction*: the
multiset of entity types (1 exit door, 1 exit switch, k mines, m gold) is fixed up front
and the model only chooses each entity's position/orientation/mode (its byte fields, with
logits masked to the legal value block). So the assembled level is always subset-valid and
correctly-counted; the lattice gate then decides solvability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokenizer import Entity, Vocab
from .tile_transformer import GRID_H, GRID_W, NUM_TILES, N_TILE_VALUES

# plausible interior placement window in map_data units (gx*4+offset, gy*4+offset)
_X_MIN, _X_MAX = 8, 168
_Y_MIN, _Y_MAX = 8, 92
SUBSET_TYPES = (1, 2, 3, 4, 21)


@dataclass
class EntityConfig:
    d_model: int = 256
    n_layers: int = 6
    n_enc_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    vocab: int = Vocab.SIZE


class _SelfBlock(nn.Module):
    def __init__(self, cfg: EntityConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class _CrossBlock(nn.Module):
    def __init__(self, cfg: EntityConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(cfg.d_model)
        self.cross = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))

    def forward(self, x, mem, causal_mask):
        h = self.ln1(x)
        a, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + a
        q = self.ln_q(x)
        c, _ = self.cross(q, mem, mem, need_weights=False)
        x = x + c
        return x + self.mlp(self.ln2(x))


class TileEncoder(nn.Module):
    """Small bidirectional encoder producing (B, 966, d) tile memory.

    Can be initialized from a trained tile transformer's token/pos embeddings and
    optionally frozen (the plan's "frozen tile-grid embeddings").
    """

    def __init__(self, cfg: EntityConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(N_TILE_VALUES + 1, cfg.d_model)
        self.row_emb = nn.Embedding(GRID_H, cfg.d_model)
        self.col_emb = nn.Embedding(GRID_W, cfg.d_model)
        self.blocks = nn.ModuleList([_SelfBlock(cfg) for _ in range(cfg.n_enc_layers)])
        rows = torch.arange(NUM_TILES) // GRID_W
        cols = torch.arange(NUM_TILES) % GRID_W
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)

    def forward(self, tile_ids):
        x = self.tok_emb(tile_ids) + self.row_emb(self.rows) + self.col_emb(self.cols)
        for b in self.blocks:
            x = b(x)
        return x


class EntityTransformer(nn.Module):
    def __init__(self, cfg: EntityConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TileEncoder(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos_emb = nn.Embedding(2048, cfg.d_model)
        self.blocks = nn.ModuleList([_CrossBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, tile_ids, ent_tokens):
        mem = self.encoder(tile_ids)
        B, L = ent_tokens.shape
        pos = torch.arange(L, device=ent_tokens.device)
        x = self.tok_emb(ent_tokens) + self.pos_emb(pos)
        causal = torch.triu(torch.full((L, L), float("-inf"), device=ent_tokens.device), 1)
        for b in self.blocks:
            x = b(x, mem, causal)
        return self.head(self.ln_f(x))

    def loss(self, tile_ids, ent_tokens, ent_mask):
        logits = self.forward(tile_ids, ent_tokens)
        tgt = ent_tokens[:, 1:]
        logits = logits[:, :-1]
        m = ent_mask[:, 1:].reshape(-1)
        ce = F.cross_entropy(logits.reshape(-1, self.cfg.vocab), tgt.reshape(-1), reduction="none")
        return (ce * m.float()).sum() / m.sum().clamp_min(1)

    # ------------------------------------------------------- constrained decode
    @torch.no_grad()
    def _sample_byte(self, tile_ids, prefix, lo, hi, temperature, generator):
        """Sample one byte token in [lo, hi] given the current prefix."""
        logits = self.forward(tile_ids, prefix)[:, -1, :]  # (B, vocab)
        block = logits[:, Vocab.BYTE_BASE + lo : Vocab.BYTE_BASE + hi + 1]
        probs = F.softmax(block / max(temperature, 1e-5), dim=-1)
        idx = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        return (lo + idx).tolist()

    @torch.no_grad()
    def constrained_generate(
        self,
        tile_ids: "torch.Tensor",
        ninja: tuple = (86, 42, 0),
        n_mines: int = 4,
        n_gold: int = 0,
        temperature: float = 1.0,
        device=None,
        generator=None,
    ) -> List[List[Entity]]:
        """Sample subset-valid, exactly-counted entity sets for each map in the batch.

        Returns a list (len B) of entity lists. Always: 1 exit door, 1 exit switch,
        ``n_mines`` mines, ``n_gold`` gold -- positions sampled from the model.
        """
        device = device or next(self.parameters()).device
        tile_ids = tile_ids.to(device)
        B = tile_ids.shape[0]
        # fixed type multiset (placement order: switch, door, then mines, gold)
        type_plan = [4, 3] + [1] * n_mines + [2] * n_gold

        # start each sequence with the ninja prefix + ENT_SEP
        seqs = [
            [Vocab.NINJA, Vocab.byte(ninja[0]), Vocab.byte(ninja[1]), Vocab.byte(ninja[2]), Vocab.ENT_SEP]
            for _ in range(B)
        ]
        ents: List[List[Entity]] = [[] for _ in range(B)]

        def prefix_tensor():
            L = max(len(s) for s in seqs)
            t = torch.full((B, L), Vocab.PAD, dtype=torch.long, device=device)
            for j, s in enumerate(seqs):
                t[j, : len(s)] = torch.tensor(s, device=device)
            return t

        for etype in type_plan:
            for s in seqs:
                s.append(Vocab.ENT)
                s.append(Vocab.etype(etype))
            xs = self._sample_byte(tile_ids, prefix_tensor(), _X_MIN, _X_MAX, temperature, generator)
            for j, s in enumerate(seqs):
                s.append(Vocab.byte(int(xs[j])))
            ys = self._sample_byte(tile_ids, prefix_tensor(), _Y_MIN, _Y_MAX, temperature, generator)
            for j, s in enumerate(seqs):
                s.append(Vocab.byte(int(ys[j])))
            # orient + mode: 0 for our subset entities (deterministic, valid)
            for j, s in enumerate(seqs):
                s.append(Vocab.byte(0))
                s.append(Vocab.byte(0))
                ents[j].append(Entity(etype, int(xs[j]), int(ys[j]), 0, 0))
        return ents


__all__ = ["EntityTransformer", "EntityConfig", "TileEncoder", "SUBSET_TYPES"]
