"""Phase 1: train the tile-stage transformer (GPU spend #1).

Two-tier per the plan: ``--pretrain`` on the full-geometry HDF5 (masked objective,
entities ignored) then fine-tune on the subset-only HDF5 via ``--init-from``.

Example::

    # pretrain on full-geometry grids
    python -m scripts.levelgen.train_tile --data datasets/levelgen/tiles_full.h5 \\
        --mode maskgit --epochs 30 --out ckpt/tile_pretrain.pt
    # fine-tune on subset-only
    python -m scripts.levelgen.train_tile --data datasets/levelgen/tiles_subset.h5 \\
        --init-from ckpt/tile_pretrain.pt --epochs 20 --out ckpt/tile.pt
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from nclone.levelgen.dataset import TileDataset
from nclone.levelgen.models.tile_transformer import TileConfig, TileTransformer

logger = logging.getLogger("levelgen.train_tile")


def save_ckpt(path, model: TileTransformer):
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load_ckpt(path, device="cpu", override=None) -> TileTransformer:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = TileConfig(**blob["config"])
    if override:
        for k, v in override.items():
            setattr(cfg, k, v)
    model = TileTransformer(cfg).to(device)
    model.load_state_dict(blob["state_dict"])
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the tile transformer")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="maskgit", choices=["maskgit", "gpt"])
    ap.add_argument("--init-from", default=None, help="checkpoint to warm-start (fine-tune)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ds = TileDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    if args.init_from:
        model = load_ckpt(args.init_from, args.device, override={"mode": args.mode})
        logger.info("Warm-started from %s", args.init_from)
    else:
        cfg = TileConfig(mode=args.mode, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                         d_ff=4 * args.d_model)
        model = TileTransformer(cfg).to(args.device)
    logger.info("Tile model: %.2fM params, %d levels", sum(p.numel() for p in model.parameters()) / 1e6, len(ds))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step = 0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        nb = 0
        for b in dl:
            loss = model.loss(b["tiles"].to(args.device), b["cond"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    logger.info("Saved tile checkpoint -> %s", args.out)


if __name__ == "__main__":
    main()
