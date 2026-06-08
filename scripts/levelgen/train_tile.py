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
import contextlib
import logging
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nclone.levelgen.dataset import TileDataset
from nclone.levelgen.models.tile_transformer import TileConfig, TileTransformer
from nclone.levelgen.tokenizer import DIFFICULTY_BANDS, MECHANIC_TAGS, SOURCES
from nclone.levelgen.train_utils import (
    EMA,
    Throughput,
    best_ckpt_path,
    cosine_warmup_lambda,
    make_writer,
    pairwise_tile_diversity,
    save_run_config,
    split_dataset,
    tiles_to_rgb,
)

logger = logging.getLogger("levelgen.train_tile")


@torch.no_grad()
def _validate(model, dl, device, autocast, loss_kw) -> float:
    """Mean per-sample masked loss over the holdout loader."""
    model.eval()
    tot = 0.0
    n = 0
    for b in dl:
        bs = b["tiles"].shape[0]
        with autocast():
            loss = model.loss(b["tiles"].to(device), b["cond"], **loss_kw)
        tot += loss.item() * bs
        n += bs
    return tot / max(n, 1)


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
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (cuda only); ~halves VRAM")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (0 = main process)")
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke)")
    ap.add_argument("--tb-logdir", default=None, help="TensorBoard logdir root (per-run subdir created)")
    ap.add_argument("--run-name", default=None, help="TensorBoard run subdir name (default: ckpt stem)")
    ap.add_argument("--val-frac", type=float, default=0.05, help="held-out fraction for val/loss (0 disables)")
    ap.add_argument("--split-seed", type=int, default=0, help="seed for the deterministic train/val split")
    ap.add_argument("--log-every", type=int, default=20, help="step interval for scalar logging")
    # --- token-imbalance controls (Stage B: corpus is ~84% empty tile 0) ---
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help="focal-loss gamma (0=plain CE); ~2.0 down-weights easy empty tiles")
    ap.add_argument("--empty-weight", type=float, default=1.0,
                    help="CE class weight for the empty tile (0); <1 down-weights it (e.g. 0.3)")
    ap.add_argument("--mask-nonempty-bias", type=float, default=1.0,
                    help="balanced masking: mask non-empty tiles this many x more often (e.g. 2.0)")
    # --- efficiency + EMA ---
    ap.add_argument("--compile", action="store_true", help="torch.compile the forward pass (cuda; Hopper win)")
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"],
                    help="cosine = linear warmup then cosine decay")
    ap.add_argument("--warmup-frac", type=float, default=0.03, help="warmup fraction of total steps (cosine)")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="EMA decay for sampling weights (0=off; ~0.999 typical)")
    # --- periodic generation viz to TensorBoard ---
    ap.add_argument("--sample-every", type=int, default=0,
                    help="every N epochs, sample + log grids/fill/diversity to TB (0=off)")
    ap.add_argument("--sample-n", type=int, default=8, help="grids to sample per viz")
    ap.add_argument("--sample-cfg", type=float, default=2.0, help="CFG scale for viz samples")
    ap.add_argument("--sample-gumbel", type=float, default=0.0, help="Gumbel temp for viz samples")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True  # Hopper/Ampere matmul speedup
    torch.backends.cudnn.allow_tf32 = True
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    full_ds = TileDataset(args.data)
    train_ds, val_ds = split_dataset(full_ds, args.val_frac, seed=args.split_seed)
    dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
            persistent_workers=args.num_workers > 0,
        )

    if args.init_from:
        model = load_ckpt(args.init_from, args.device, override={"mode": args.mode})
        logger.info("Warm-started from %s", args.init_from)
    else:
        cfg = TileConfig(mode=args.mode, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                         d_ff=4 * args.d_model)
        model = TileTransformer(cfg).to(args.device)
    logger.info("Tile model: %.2fM params, %d levels", sum(p.numel() for p in model.parameters()) / 1e6, len(full_ds))

    run_name = args.run_name or Path(args.out).stem
    writer = make_writer(args.tb_logdir, run_name)
    save_run_config(args.out, args, model, extra={"run_name": run_name, "n_levels": len(full_ds)})

    # token-imbalance loss controls (default = plain masked CE)
    class_weight = None
    if args.empty_weight != 1.0:
        class_weight = torch.ones(model.cfg.vocab, device=args.device)
        class_weight[0] = args.empty_weight  # tile 0 == empty
    loss_kw = dict(focal_gamma=args.focal_gamma, class_weight=class_weight,
                   mask_nonempty_bias=args.mask_nonempty_bias)
    if args.focal_gamma or args.empty_weight != 1.0 or args.mask_nonempty_bias != 1.0:
        logger.info("Imbalance loss: focal_gamma=%.2f empty_weight=%.2f mask_nonempty_bias=%.2f",
                    args.focal_gamma, args.empty_weight, args.mask_nonempty_bias)

    if args.compile and args.device.startswith("cuda"):
        model.forward = torch.compile(model.forward)  # keeps state_dict clean (EMA-safe)
        logger.info("torch.compile enabled on forward")
    elif args.compile:
        logger.warning("--compile ignored: requires a cuda device (got %s)", args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
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

    def _save(path):  # save EMA weights when enabled (that's what we sample from)
        if ema:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            save_ckpt(path, model)
            model.load_state_dict(backup)
        else:
            save_ckpt(path, model)

    def _val():  # evaluate on EMA weights when enabled, so best-by-val tracks them
        if ema:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            v = _validate(model, val_dl, args.device, autocast, loss_kw)
            model.load_state_dict(backup)
            return v
        return _validate(model, val_dl, args.device, autocast, loss_kw)

    def _sample_cond(n):
        cond = {
            "source": torch.full((n,), SOURCES.index("generated"), dtype=torch.long),
            "difficulty": torch.full((n,), DIFFICULTY_BANDS.index("d2"), dtype=torch.long),
            "subset": torch.ones(n, dtype=torch.long),
            "mech": torch.zeros(n, len(MECHANIC_TAGS)),
        }
        cond["mech"][:, MECHANIC_TAGS.index("mine")] = 1.0
        return cond

    @torch.no_grad()
    def _log_samples(ep):
        """Sample (from EMA weights), log grid montage + fill + diversity to TB."""
        if writer is None or args.sample_every <= 0:
            return
        backup = None
        try:
            if ema:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
            model.eval()
            gsmpl = torch.Generator(device=args.device).manual_seed(1234)  # fixed -> comparable
            tiles = model.sample(args.sample_n, cond=_sample_cond(args.sample_n), steps=12,
                                 temperature=1.0, cfg_scale=args.sample_cfg, device=args.device,
                                 generator=gsmpl, gumbel_temp=args.sample_gumbel)
            t_np = tiles.cpu().numpy()
            writer.add_scalar("sample/fill", float((t_np != 0).mean()), ep)
            writer.add_scalar("sample/diversity", pairwise_tile_diversity(t_np), ep)
            writer.add_images("sample/grids", tiles_to_rgb(t_np), ep, dataformats="NCHW")
        except Exception as e:
            logger.warning("sample logging failed: %s", e)
        finally:
            if backup is not None:
                model.load_state_dict(backup)
            model.train()

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
                loss = model.loss(b["tiles"].to(args.device), b["cond"], **loss_kw)
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        if args.sample_every and (ep % args.sample_every == 0 or ep == args.epochs - 1):
            _log_samples(ep)
        if args.max_steps and step >= args.max_steps:
            break
    _save(args.out)
    logger.info("Saved tile checkpoint -> %s", args.out)
    if val_dl is not None and best_val < float("inf"):
        logger.info("Best val=%.4f -> %s", best_val, best_ckpt_path(args.out))
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
