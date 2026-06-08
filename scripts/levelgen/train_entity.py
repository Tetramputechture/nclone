"""Phase 2: train the entity-stage transformer (GPU spend #2).

Autoregressive over the entity token stream, cross-attending to the tile encoder.
``--tile-ckpt`` warm-starts the tile encoder's *embeddings* from a trained tile
transformer (only when ``--d-model`` matches the tile stage; the encoder's transformer
blocks always train from scratch). ``--freeze-encoder`` exists but is discouraged: it
would pin those from-scratch blocks -- leave the encoder trainable.

Example::

    python -m scripts.levelgen.train_entity --data datasets/levelgen/tiles_subset.h5 \\
        --tile-ckpt ckpt/tile.pt --epochs 20 --out ckpt/entity.pt
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nclone.levelgen.dataset import EntityDataset, collate_entity
from nclone.levelgen.models.entity_transformer import EntityConfig, EntityTransformer
from nclone.levelgen.train_utils import (
    EMA,
    Throughput,
    best_ckpt_path,
    cosine_warmup_lambda,
    make_writer,
    save_run_config,
    split_dataset,
)

logger = logging.getLogger("levelgen.train_entity")


@torch.no_grad()
def _validate(model, dl, device, autocast) -> float:
    """Mean per-sample masked loss over the holdout loader."""
    model.eval()
    tot = 0.0
    n = 0
    for b in dl:
        bs = b["tiles"].shape[0]
        with autocast():
            loss = model.loss(
                b["tiles"].to(device), b["ent_tokens"].to(device), b["ent_mask"].to(device)
            )
        tot += loss.item() * bs
        n += bs
    return tot / max(n, 1)


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
    # Copy matching token/row/col embeddings if dimensions agree. Only the embeddings
    # transfer (the encoder transformer blocks are always trained from scratch), so a
    # d_model mismatch between the tile and entity stages makes this a complete no-op.
    copied = []
    for name in ("tok_emb", "row_emb", "col_emb"):
        w = sd.get(f"{name}.weight")
        tgt = getattr(model.encoder, name).weight
        if w is not None and w.shape == tgt.shape:
            tgt.data.copy_(w.to(device))
            copied.append(name)
    if copied:
        logger.info("Warm-started tile encoder embeddings %s from %s", copied, tile_ckpt)
    else:
        tile_dm = sd["tok_emb.weight"].shape[1] if "tok_emb.weight" in sd else "?"
        logger.warning(
            "--tile-ckpt %s transferred NOTHING (tile d_model=%s != entity d_model=%d); "
            "the encoder is trained from scratch. Match --d-model to reuse the embeddings.",
            tile_ckpt, tile_dm, model.cfg.d_model,
        )


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
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (cuda only); ~halves VRAM")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (0 = main process)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--tb-logdir", default=None, help="TensorBoard logdir root (per-run subdir created)")
    ap.add_argument("--run-name", default=None, help="TensorBoard run subdir name (default: ckpt stem)")
    ap.add_argument("--val-frac", type=float, default=0.05, help="held-out fraction for val/loss (0 disables)")
    ap.add_argument("--split-seed", type=int, default=0, help="seed for the deterministic train/val split")
    ap.add_argument("--log-every", type=int, default=20, help="step interval for scalar logging")
    ap.add_argument("--compile", action="store_true", help="torch.compile the forward pass (cuda; Hopper win)")
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"])
    ap.add_argument("--warmup-frac", type=float, default=0.03, help="warmup fraction of total steps (cosine)")
    ap.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay for sampling weights (0=off)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    full_ds = EntityDataset(args.data)
    train_ds, val_ds = split_dataset(full_ds, args.val_frac, seed=args.split_seed)
    dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_entity,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_entity,
            num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
            persistent_workers=args.num_workers > 0,
        )

    cfg = EntityConfig(d_model=args.d_model, n_layers=args.n_layers, d_ff=4 * args.d_model)
    model = EntityTransformer(cfg).to(args.device)
    if args.tile_ckpt:
        _init_encoder_from_tile(model, args.tile_ckpt, args.device)
    if args.freeze_encoder:
        # NOTE: the encoder's transformer blocks are never transferred from the tile
        # checkpoint -- only its embeddings can be. Freezing therefore pins randomly
        # initialized blocks and starves the entity stage of tile geometry. Leave it off
        # unless a fully-pretrained encoder is wired in.
        logger.warning(
            "--freeze-encoder freezes the (mostly from-scratch) tile encoder; "
            "the entity stage will cross-attend to fixed random tile features."
        )
        model.freeze_encoder()
    logger.info("Entity model: %.2fM params, %d levels", sum(p.numel() for p in model.parameters()) / 1e6, len(full_ds))

    run_name = args.run_name or Path(args.out).stem
    writer = make_writer(args.tb_logdir, run_name)
    save_run_config(args.out, args, model, extra={"run_name": run_name, "n_levels": len(full_ds)})

    if args.compile and args.device.startswith("cuda"):
        model.forward = torch.compile(model.forward)
        logger.info("torch.compile enabled on forward")
    elif args.compile:
        logger.warning("--compile ignored: requires a cuda device (got %s)", args.device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(len(dl), 1)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = None
    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, cosine_warmup_lambda(total_steps, int(total_steps * args.warmup_frac)))
        logger.info("LR cosine schedule: %d total steps, %d warmup", total_steps,
                    int(total_steps * args.warmup_frac))
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema:
        logger.info("EMA enabled (decay=%.4f); val + checkpoints use EMA weights", args.ema_decay)

    use_amp = args.amp and args.device.startswith("cuda")
    autocast = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if use_amp
        else contextlib.nullcontext()
    )
    if args.amp and not use_amp:
        logger.warning("--amp ignored: requires a cuda device (got %s)", args.device)

    def _save(path):
        if ema:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            save_ckpt(path, model)
            model.load_state_dict(backup)
        else:
            save_ckpt(path, model)

    def _val():
        if ema:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            v = _validate(model, val_dl, args.device, autocast)
            model.load_state_dict(backup)
            return v
        return _validate(model, val_dl, args.device, autocast)

    thr = Throughput()
    best_val = float("inf")
    step = 0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        nb = 0
        for b in dl:
            bs = b["tiles"].shape[0]
            with autocast():
                loss = model.loss(b["tiles"].to(args.device), b["ent_tokens"].to(args.device), b["ent_mask"].to(args.device))
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item()
            nb += 1
            step += 1
            thr.add(bs)
            if writer is not None and step % args.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                writer.add_scalar("train/grad_norm", float(grad_norm), step)
                writer.add_scalar("train/samples_per_s", thr.rate(), step)
            if args.max_steps and step >= args.max_steps:
                break
        train_loss = tot / max(nb, 1)
        msg = "epoch %d: loss=%.4f" % (ep, train_loss)
        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_loss, ep)
        if val_dl is not None:
            val_loss = _val()
            msg += " val=%.4f" % val_loss
            if writer is not None:
                writer.add_scalar("epoch/val_loss", val_loss, ep)
            if val_loss < best_val:
                best_val = val_loss
                _save(best_ckpt_path(args.out))
                msg += " (best)"
        logger.info("%s", msg)
        if args.max_steps and step >= args.max_steps:
            break
    _save(args.out)
    logger.info("Saved entity checkpoint -> %s", args.out)
    if val_dl is not None and best_val < float("inf"):
        logger.info("Best val=%.4f -> %s", best_val, best_ckpt_path(args.out))
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
