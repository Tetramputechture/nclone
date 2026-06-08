"""Shared training instrumentation for the level-generator stages.

Centralizes TensorBoard setup, run-config/hparam persistence, git provenance,
deterministic train/val splitting, and best-checkpoint tracking so
``train_tile.py`` and ``train_entity.py`` stay thin and log identically.

The two training loops still live in their own scripts (their loss-call shapes
differ), but everything cross-cutting -- the ``SummaryWriter``, the holdout
split, the run-config sidecar, the best/val bookkeeping -- comes from here.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
from torch.utils.data import random_split

logger = logging.getLogger("levelgen.train_utils")


def git_sha(default: str = "unknown") -> str:
    """Best-effort current commit SHA (for run provenance)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return default


def split_dataset(ds, val_frac: float, seed: int = 0):
    """Deterministic train/val split.

    Returns ``(train_ds, val_ds)``; ``val_ds`` is ``None`` when ``val_frac<=0``
    or the dataset is too small to spare a holdout sample. Always keeps at least
    one training sample.
    """
    n = len(ds)
    if val_frac <= 0 or n <= 1:
        return ds, None
    n_val = int(round(n * val_frac))
    n_val = min(max(n_val, 1), n - 1)
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    logger.info("Train/val split: %d train / %d val (val_frac=%.3f, seed=%d)",
                n_train, n_val, val_frac, seed)
    return train_ds, val_ds


def make_writer(tb_logdir, run_name: str):
    """Create a ``SummaryWriter`` under ``<tb_logdir>/<run_name>`` (or ``None``)."""
    if not tb_logdir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # tensorboard not installed
        logger.warning("TensorBoard unavailable (%s); scalar logging disabled. "
                       "pip install tensorboard to enable.", exc)
        return None
    path = Path(tb_logdir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    logger.info("TensorBoard logdir: %s", path)
    return SummaryWriter(log_dir=str(path))


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def save_run_config(out_path, args, model, extra: dict | None = None) -> dict:
    """Persist hparams + git SHA + model size as a ``<ckpt>.run_config.json`` sidecar."""
    cfg = {
        "args": {k: _jsonable(v) for k, v in vars(args).items()},
        "git_sha": git_sha(),
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "model_config": asdict(model.cfg) if is_dataclass(model.cfg) else None,
    }
    if extra:
        cfg.update(extra)
    p = Path(out_path).with_suffix(".run_config.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2, default=str))
    logger.info("Wrote run config -> %s", p)
    return cfg


class Throughput:
    """Smoothed samples/sec over a logging window (perf_counter based)."""

    def __init__(self):
        self._t = time.perf_counter()
        self._samples = 0

    def add(self, n: int) -> None:
        self._samples += n

    def rate(self) -> float:
        now = time.perf_counter()
        dt = now - self._t
        rate = self._samples / dt if dt > 0 else 0.0
        self._t = now
        self._samples = 0
        return rate


def best_ckpt_path(out_path) -> Path:
    """Sibling path for the best-by-val checkpoint (``foo.pt`` -> ``foo.best.pt``)."""
    p = Path(out_path)
    return p.with_suffix(".best" + p.suffix)


def cosine_warmup_lambda(total_steps: int, warmup_steps: int, floor: float = 0.0):
    """LR multiplier: linear warmup to 1.0 then cosine decay to ``floor``.

    Use with ``torch.optim.lr_scheduler.LambdaLR`` and ``.step()`` per optimizer step.
    """
    import math

    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    def f(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        prog = min(max(prog, 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return f


class EMA:
    """Exponential moving average of a model's float tensors (params + buffers).

    Sampling from an EMA copy is a cheap, standard generative-quality bump. Built
    from ``model.state_dict()`` so it works as long as the model isn't wrapped
    (compile via ``model.forward = torch.compile(model.forward)`` keeps it clean).
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow.get(k)
            if s is not None:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model) -> None:
        sd = model.state_dict()
        for k, s in self.shadow.items():
            sd[k].copy_(s)
        model.load_state_dict(sd)
