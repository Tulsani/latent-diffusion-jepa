"""
training/trainer.py
-------------------
DDP-aware training loop for LD-JEPA on Slurm with spot instance preemption handling.

Key features:
  - Automatic checkpoint save on preemption (SIGUSR1 / SIGTERM)
  - Resume from checkpoint on requeue (#SBATCH --requeue)
  - Mixed precision (bf16) training
  - Periodic linear probe evaluation (frozen encoder) during training
  - Rich console + file logging (no W&B dependency — pure file-based for HPC)
  - EMA momentum cosine annealing
"""

import os
import sys
import math
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
import torch.cuda.amp as amp
import yaml
from types import SimpleNamespace

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.ld_jepa import LDJEPA, update_ema, get_ema_momentum
from data.dataset import build_dataloaders
from training.losses import check_representation_collapse
from evaluation.linear_probe import run_linear_probe_eval


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> SimpleNamespace:
    """Load YAML config and convert to nested SimpleNamespace for dot-access."""
    with open(path) as f:
        d = yaml.safe_load(f)

    def to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
        return obj

    return to_ns(d)


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, rank: int) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ld_jepa")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (rank 0 only)
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # File handler (all ranks, separate files)
    fh = logging.FileHandler(log_dir / f"train_rank{rank}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    ckpt_dir: str,
    epoch: int,
    step: int,
    model: LDJEPA,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    metrics: Dict,
    rank: int,
    filename: Optional[str] = None,
):
    """Save full training state to disk. Only rank 0 writes."""
    if rank != 0:
        return

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fname = filename or f"ckpt_epoch{epoch:04d}_step{step:07d}.pt"
    path  = ckpt_dir / fname

    # Unwrap DDP if needed
    enc_state  = model.module.context_encoder.state_dict() \
                 if isinstance(model, DDP) else model.context_encoder.state_dict()
    tgt_state  = model.module.target_encoder.state_dict() \
                 if isinstance(model, DDP) else model.target_encoder.state_dict()
    pred_state = model.module.diffusion_pred.state_dict() \
                 if isinstance(model, DDP) else model.diffusion_pred.state_dict()

    torch.save({
        "epoch":              epoch,
        "step":               step,
        "context_encoder":    enc_state,
        "target_encoder":     tgt_state,
        "diffusion_pred":     pred_state,
        "optimizer":          optimizer.state_dict(),
        "scheduler":          scheduler.state_dict() if scheduler else None,
        "scaler":             scaler.state_dict(),
        "metrics":            metrics,
    }, path)

    # Also save a "latest" symlink for easy resume
    latest = ckpt_dir / "latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(fname)

    return str(path)


def load_checkpoint(
    ckpt_path: str,
    model: LDJEPA,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    """Load checkpoint and restore all training state."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # Unwrap DDP
    raw_model = model.module if isinstance(model, DDP) else model

    raw_model.context_encoder.load_state_dict(ckpt["context_encoder"])
    raw_model.target_encoder.load_state_dict(ckpt["target_encoder"])
    raw_model.diffusion_pred.load_state_dict(ckpt["diffusion_pred"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt["scheduler"]:
        scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])

    logger.info(f"Resumed from {ckpt_path} "
                f"(epoch {ckpt['epoch']}, step {ckpt['step']})")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate schedule
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg, total_steps: int):
    """
    Cosine LR schedule with linear warmup.
    Warmup: 0 → base_lr over first warmup_steps steps
    Then:   cosine decay from base_lr → 0 over remaining steps
    """
    warmup_steps = int(cfg.training.warmup_epochs *
                       total_steps / cfg.training.epochs)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    LD-JEPA trainer with DDP, mixed precision, and Slurm spot-instance support.
    """

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg        = cfg
        self.rank       = rank
        self.world_size = world_size
        self.device     = torch.device(f"cuda:{rank}")
        self.is_main    = (rank == 0)

        # Expand env vars in paths
        self._expand_paths()

        # Logging
        self.logger = setup_logging(cfg.experiment.log_dir, rank)

        # Build model
        self.model = LDJEPA(cfg).to(self.device)
        if world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[rank],
                find_unused_parameters=False,
            )

        # Optimizer — only trainable params (context_encoder + diffusion_pred)
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        self.optimizer = torch.optim.AdamW(
            raw_model.get_trainable_params(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            betas=(0.9, 0.95),
        )

        # Data
        stats_cache = os.path.join(cfg.experiment.checkpoint_dir, "channel_stats.npz")
        self.loaders = build_dataloaders(cfg, rank, world_size, stats_cache)

        # Total steps
        self.steps_per_epoch = len(self.loaders["train"])
        self.total_steps     = cfg.training.epochs * self.steps_per_epoch

        # LR scheduler
        self.scheduler = build_scheduler(self.optimizer, cfg, self.total_steps)

        # Mixed precision scaler (bf16 on A100 → use autocast, not scaler)
        self.use_amp   = cfg.training.mixed_precision
        self.scaler    = GradScaler(enabled=False)  # bf16 doesn't need scaler

        # Preemption / checkpoint state
        self.start_epoch = 0
        self.global_step = 0
        self._preempted  = False

        # Register signal handlers for Slurm preemption
        signal.signal(signal.SIGUSR1, self._handle_preemption)
        signal.signal(signal.SIGTERM, self._handle_preemption)

        # Resume from checkpoint if exists
        ckpt_path = os.path.join(cfg.experiment.checkpoint_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            ckpt = load_checkpoint(
                ckpt_path, self.model, self.optimizer,
                self.scheduler, self.scaler, self.device, self.logger
            )
            self.start_epoch = ckpt["epoch"] + 1
            self.global_step = ckpt["step"]
            # Advance scheduler to current step
            for _ in range(self.global_step):
                self.scheduler.step()

        self.logger.info(
            f"Trainer initialized | rank={rank}/{world_size} | "
            f"steps_per_epoch={self.steps_per_epoch} | "
            f"total_steps={self.total_steps} | "
            f"start_epoch={self.start_epoch}"
        )

    def _expand_paths(self):
        """Expand $NETID and other env vars in config paths."""
        for section_name in ["experiment"]:
            section = getattr(self.cfg, section_name)
            for key in vars(section):
                val = getattr(section, key)
                if isinstance(val, str):
                    setattr(section, key, os.path.expandvars(val))

    def _handle_preemption(self, signum, frame):
        """
        Called when Slurm sends SIGUSR1 (preemption warning) or SIGTERM.
        Save checkpoint immediately and mark for graceful exit.
        """
        self.logger.warning(f"Received signal {signum} — saving checkpoint for requeue...")
        save_checkpoint(
            self.cfg.experiment.checkpoint_dir,
            epoch=self._current_epoch,
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            metrics={"preempted": True},
            rank=self.rank,
            filename="preemption_ckpt.pt",
        )
        self.logger.warning("Preemption checkpoint saved. Exiting.")
        sys.exit(0)

    def train(self):
        """Main training loop."""
        cfg = self.cfg

        for epoch in range(self.start_epoch, cfg.training.epochs):
            self._current_epoch = epoch

            # Set epoch for DistributedSampler shuffling
            if self.world_size > 1 and hasattr(self.loaders["train"].sampler, "set_epoch"):
                self.loaders["train"].sampler.set_epoch(epoch)

            train_metrics = self._train_epoch(epoch)

            # ── Periodic evaluation ───────────────────────────────────────
            if self.is_main and (epoch + 1) % cfg.evaluation.eval_every_epochs == 0:
                self._evaluate(epoch)

            # ── Epoch checkpoint ──────────────────────────────────────────
            save_checkpoint(
                cfg.experiment.checkpoint_dir,
                epoch=epoch,
                step=self.global_step,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                metrics=train_metrics,
                rank=self.rank,
            )

            if self.is_main:
                self.logger.info(
                    f"Epoch {epoch:4d}/{cfg.training.epochs} | "
                    f"loss={train_metrics.get('loss_mean', 0):.4f} | "
                    f"diff={train_metrics.get('diffusion_loss_mean', 0):.4f} | "
                    f"vicreg={train_metrics.get('vicreg_loss_mean', 0):.4f}"
                )

        self.logger.info("Training complete.")

    def _train_epoch(self, epoch: int) -> Dict:
        """Run one training epoch."""
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        self.model.train()
        raw_model.target_encoder.eval()   # Target encoder always in eval mode

        epoch_metrics = {}
        t0 = time.time()

        for batch_idx, (x, labels) in enumerate(self.loaders["train"]):
            x = x.to(self.device, non_blocking=True)   # (B, C, T, H, W)

            # Forward pass with bf16 autocast
            with amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=self.use_amp):
                loss, metrics = raw_model(
                    x,
                    step=self.global_step,
                    total_steps=self.total_steps,
                )

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.cfg.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    raw_model.get_trainable_params(),
                    self.cfg.training.grad_clip,
                )

            self.optimizer.step()
            self.scheduler.step()

            # EMA update (separate from forward pass when using DDP)
            momentum = get_ema_momentum(
                self.global_step, self.total_steps,
                ema_start=self.cfg.model.ema_momentum,
                ema_end=0.9999,
            )
            update_ema(raw_model.target_encoder, raw_model.context_encoder, momentum)

            self.global_step += 1

            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics.setdefault(k + "_mean", 0.0)
                epoch_metrics[k + "_mean"] += v

            # ── Periodic step logging ─────────────────────────────────────
            if self.is_main and batch_idx % 50 == 0:
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                self.logger.info(
                    f"  E{epoch:03d} [{batch_idx:4d}/{self.steps_per_epoch}] "
                    f"loss={metrics['loss']:.4f} "
                    f"diff={metrics['diffusion_loss']:.4f} "
                    f"z0_mse={metrics['z0_pred_mse']:.4f} "
                    f"vicreg={metrics['vicreg_loss']:.4f} "
                    f"lr={lr:.2e} "
                    f"ema={momentum:.5f} "
                    f"t={elapsed:.1f}s"
                )
                t0 = time.time()

            # ── Representation collapse check (every 500 steps) ───────────
            if self.is_main and self.global_step % 500 == 0:
                with torch.no_grad():
                    z_mean = raw_model.context_encoder.get_embedding(
                        x[:min(32, x.shape[0])]
                    )
                    collapse_metrics = check_representation_collapse(
                        z_mean.float(), prefix="repr_"
                    )
                    self.logger.info(
                        f"  Collapse check: "
                        f"eff_rank={collapse_metrics.get('repr_effective_rank', 0):.1f} "
                        f"cos_sim={collapse_metrics.get('repr_mean_cos_sim', 0):.4f} "
                        f"std_dead={collapse_metrics.get('repr_std_dead', 0):.4f}"
                    )

        # Normalize accumulated metrics
        n = len(self.loaders["train"])
        epoch_metrics = {k: v / n for k, v in epoch_metrics.items()}

        return epoch_metrics

    def _evaluate(self, epoch: int):
        """Run linear probe evaluation on validation set."""
        self.logger.info(f"Running evaluation at epoch {epoch}...")
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model

        # Extract embeddings for train and val splits
        from evaluation.linear_probe import extract_embeddings, run_linear_probe_eval
        from evaluation.knn_regression import run_knn_eval

        train_embs, train_labels = extract_embeddings(
            raw_model, self.loaders["train"], self.device, max_samples=2000
        )
        val_embs, val_labels = extract_embeddings(
            raw_model, self.loaders["val"], self.device, max_samples=1200
        )

        # Linear probe MSE
        lp_results = run_linear_probe_eval(train_embs, train_labels,
                                           val_embs, val_labels)
        # kNN MSE
        knn_results = run_knn_eval(train_embs, train_labels, val_embs, val_labels,
                                    k=self.cfg.evaluation.knn_k)

        self.logger.info(
            f"  [Eval E{epoch:03d}] "
            f"LP alpha_mse={lp_results['alpha_mse']:.4f} "
            f"LP zeta_mse={lp_results['zeta_mse']:.4f} | "
            f"kNN alpha_mse={knn_results['alpha_mse']:.4f} "
            f"kNN zeta_mse={knn_results['zeta_mse']:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DDP entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_worker(rank: int, world_size: int, cfg_path: str):
    """Worker function for each DDP process."""
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    # Reproducibility
    cfg = load_config(cfg_path)
    seed = cfg.experiment.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Build and run trainer
    trainer = Trainer(cfg, rank, world_size)
    trainer.train()

    dist.destroy_process_group()