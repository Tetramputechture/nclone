"""Phase 2: train the entity-stage transformer (GPU spend #2).

Autoregressive over the entity token stream, cross-attending to the tile encoder.
``--tile-ckpt`` initializes the tile encoder from a trained tile transformer's
token/pos embeddings (the plan's "frozen tile-grid embeddings"); ``--freeze-encoder``
keeps them fixed.

Example::

    python -m scripts.levelgen.train_entity --data datasets/levelgen/tiles_subset.h5 \\
        --tile-ckpt ckpt/tile.pt --freeze-encoder --epochs 20 --out ckpt/entity.pt
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from nclone.levelgen.dataset import EntityDataset, collate_entity
from nclone.levelgen.models.entity_transformer import EntityConfig, EntityTransformer

logger = logging.getLogger("levelgen.train_entity")


def save_ckpt(path, model: EntityTransformer):
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load_ckpt(path, device="cpu") -> EntityTransformer:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = EntityTransformer(EntityConfig(**blob["config"])).to(device)
    model.load_state_dict(blob["state_dict"])
    return model


def _init_encoder_from_tile(model: EntityTransformer, tile_ckpt: str, device: str):
    blob = torch.load(tile_ckpt, map_location=device, weights_only=False)
    sd = blob["state_dict"]
    # copy matching token/row/col embeddings if dimensions agree
    for name in ("tok_emb", "row_emb", "col_emb"):
        w = sd.get(f"{name}.weight")
        tgt = getattr(model.encoder, name).weight
        if w is not None and w.shape == tgt.shape:
            tgt.data.copy_(w.to(device))
    logger.info("Initialized tile encoder embeddings from %s", tile_ckpt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the entity transformer")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-ckpt", default=None, help="init encoder from tile checkpoint")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ds = EntityDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_entity)

    cfg = EntityConfig(d_model=args.d_model, n_layers=args.n_layers, d_ff=4 * args.d_model)
    model = EntityTransformer(cfg).to(args.device)
    if args.tile_ckpt:
        _init_encoder_from_tile(model, args.tile_ckpt, args.device)
    if args.freeze_encoder:
        model.freeze_encoder()
    logger.info("Entity model: %.2fM params, %d levels", sum(p.numel() for p in model.parameters()) / 1e6, len(ds))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    step = 0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        nb = 0
        for b in dl:
            loss = model.loss(b["tiles"].to(args.device), b["ent_tokens"].to(args.device), b["ent_mask"].to(args.device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        logger.info("epoch %d: loss=%.4f", ep, tot / max(nb, 1))
        if args.max_steps and step >= args.max_steps:
            break
    save_ckpt(args.out, model)
    logger.info("Saved entity checkpoint -> %s", args.out)


if __name__ == "__main__":
    main()
