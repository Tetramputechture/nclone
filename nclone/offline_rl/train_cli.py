"""Training CLI for IQL offline value learning.

Usage:
    python -m nclone.offline_rl.train_cli \
        --dataset dataset.h5 \
        --output-dir checkpoints/ \
        --epochs 100 \
        --device cuda \
        --amp \
        --prefetch
"""

import argparse
import logging
import os
import time

import torch

from .algorithms.iql import IQLConfig, IQLTrainer
from .dataset.replay_dataset import OfflineRLDataset, PrefetchingBatchIterator

logger = logging.getLogger(__name__)


def train(
    dataset_path: str,
    output_dir: str,
    config: IQLConfig,
    epochs: int = 100,
    steps_per_epoch: int = 1000,
    save_every: int = 10,
    device: str = "cpu",
    log_every: int = 100,
    prefetch: bool = False,
    resume_from: str = "",
):
    """Run IQL training loop.

    Args:
        dataset_path: Path to HDF5 dataset.
        output_dir: Directory for checkpoints and logs.
        config: IQL hyperparameters.
        epochs: Number of training epochs.
        steps_per_epoch: Gradient steps per epoch.
        save_every: Save checkpoint every N epochs.
        device: Torch device.
        log_every: Log metrics every N steps.
        prefetch: If True, use background-thread batch prefetching for GPU.
        resume_from: Path to checkpoint to resume training from.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Optional TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = os.path.join(output_dir, "tb_logs")
        writer = SummaryWriter(tb_dir)
        logger.info("TensorBoard logs: %s", tb_dir)
    except ImportError:
        logger.info("TensorBoard not available, logging to console only")

    # Load dataset (keep on CPU when prefetching; prefetcher handles GPU transfer)
    ds_device = "cpu" if prefetch else device
    dataset = OfflineRLDataset(dataset_path, device=ds_device)
    logger.info(
        "Dataset: %d transitions, %d levels", dataset.n_transitions, dataset.n_levels
    )

    # Initialize trainer
    trainer = IQLTrainer(config, device=device)

    # Resume from checkpoint if specified
    start_epoch = 1
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        trainer.load_state_dict(ckpt)
        start_epoch = ckpt.get("step", 0) // steps_per_epoch + 1
        logger.info("Resumed at step %d (epoch %d)", trainer.step, start_epoch)

    total_params = sum(p.numel() for p in trainer.networks.parameters())
    logger.info("Network parameters: %d (%.1fM)", total_params, total_params / 1e6)
    logger.info("AMP: %s | Prefetch: %s", config.use_amp, prefetch)

    # Set up batch iterator
    prefetcher = None
    if prefetch and device != "cpu":
        prefetcher = PrefetchingBatchIterator(
            dataset,
            batch_size=config.batch_size,
            device=device,
            prefetch_count=2,
        )
        logger.info("Prefetching batch iterator active (2 batches ahead)")

    # Training loop
    global_step = trainer.step
    best_v_loss = float("inf")
    t0 = time.time()

    progress_interval = max(1, steps_per_epoch // 10)  # Log ~10x per epoch

    for epoch in range(start_epoch, epochs + 1):
        logger.info("Epoch %d/%d starting...", epoch, epochs)
        epoch_v_loss = 0.0
        epoch_q_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_mean_v = 0.0

        for step in range(steps_per_epoch):
            if step == 0:
                logger.info("  Fetching first batch...")
            if prefetcher is not None:
                batch = next(prefetcher)
            else:
                batch = dataset.sample_batch(config.batch_size)
            if step == 0:
                logger.info("  First batch received, running update...")
            step_t0 = time.time()
            metrics = trainer.update(batch)
            if step == 0:
                logger.info("  First step complete (%.2fs)", time.time() - step_t0)

            epoch_v_loss += metrics.v_loss
            epoch_q_loss += metrics.q_loss
            epoch_policy_loss += metrics.policy_loss
            epoch_mean_v += metrics.mean_v
            global_step += 1

            # Periodic progress log during epoch (log at 1, 10, 50, 100, then every 100)
            progress_every = min(100, progress_interval)
            should_log = (
                step + 1 == 1
                or step + 1 == 10
                or step + 1 == 20
                or step + 1 == 50
                or (step + 1) % progress_every == 0
                or step == steps_per_epoch - 1
            )
            if should_log:
                pct = 100 * (step + 1) / steps_per_epoch
                logger.info(
                    "  Epoch %d: %d/%d (%.0f%%) | V=%.4f Q=%.4f Pi=%.4f | mean_V=%.2f",
                    epoch,
                    step + 1,
                    steps_per_epoch,
                    pct,
                    metrics.v_loss,
                    metrics.q_loss,
                    metrics.policy_loss,
                    metrics.mean_v,
                )

            if writer and global_step % log_every == 0:
                writer.add_scalar("loss/v_loss", metrics.v_loss, global_step)
                writer.add_scalar("loss/q_loss", metrics.q_loss, global_step)
                writer.add_scalar("loss/policy_loss", metrics.policy_loss, global_step)
                writer.add_scalar("value/mean_v", metrics.mean_v, global_step)
                writer.add_scalar("value/mean_q", metrics.mean_q, global_step)
                writer.add_scalar(
                    "value/mean_advantage", metrics.mean_advantage, global_step
                )
                writer.add_scalar("data/mean_reward", metrics.mean_reward, global_step)

        # Epoch averages
        n = steps_per_epoch
        avg_v = epoch_v_loss / n
        avg_q = epoch_q_loss / n
        avg_pi = epoch_policy_loss / n
        avg_mean_v = epoch_mean_v / n
        elapsed = time.time() - t0
        steps_per_sec = (epoch - start_epoch + 1) * steps_per_epoch / elapsed

        logger.info(
            "Epoch %d/%d | V=%.4f Q=%.4f Pi=%.4f | mean_V=%.2f | %.0f steps/s | %.0fs",
            epoch,
            epochs,
            avg_v,
            avg_q,
            avg_pi,
            avg_mean_v,
            steps_per_sec,
            elapsed,
        )

        if writer:
            writer.add_scalar("epoch/v_loss", avg_v, epoch)
            writer.add_scalar("epoch/q_loss", avg_q, epoch)
            writer.add_scalar("epoch/policy_loss", avg_pi, epoch)
            writer.add_scalar("epoch/mean_v", avg_mean_v, epoch)
            writer.add_scalar("perf/steps_per_sec", steps_per_sec, epoch)

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(output_dir, f"iql_epoch_{epoch:04d}.pt")
            torch.save(trainer.state_dict(), ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

        # Track best model
        if avg_v < best_v_loss:
            best_v_loss = avg_v
            best_path = os.path.join(output_dir, "iql_best.pt")
            torch.save(trainer.state_dict(), best_path)

    # Save final model
    final_path = os.path.join(output_dir, "iql_final.pt")
    torch.save(trainer.state_dict(), final_path)
    logger.info("Training complete. Final model: %s", final_path)

    if prefetcher is not None:
        prefetcher.stop()

    if writer:
        writer.close()

    dataset.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train IQL on offline demonstration dataset"
    )
    parser.add_argument("--dataset", type=str, required=True, help="HDF5 dataset path")
    parser.add_argument(
        "--output-dir", type=str, default="iql_checkpoints", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--steps-per-epoch", type=int, default=1000, help="Gradient steps per epoch"
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--expectile-tau", type=float, default=0.9, help="IQL expectile parameter"
    )
    parser.add_argument("--beta", type=float, default=3.0, help="AWR temperature")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--hidden-dim", type=int, default=512, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--no-tile-encoder",
        action="store_true",
        help="Disable tile encoder (single-level mode)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed-precision training (AMP) for ~2x GPU speedup",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Enable background-thread batch prefetching for GPU training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    config = IQLConfig(
        expectile_tau=args.expectile_tau,
        discount=args.discount,
        beta=args.beta,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        use_tile_encoder=not args.no_tile_encoder,
        use_amp=args.amp,
    )

    logger.info("IQL Config: %s", config)
    logger.info("Device: %s", args.device)

    train(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        config=config,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        save_every=args.save_every,
        device=args.device,
        prefetch=args.prefetch,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
