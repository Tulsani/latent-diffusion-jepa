#!/usr/bin/env python3
"""
scripts/extract_embeddings.py
------------------------------
Extract and save frozen encoder embeddings for all splits (train/val/test).

Saves embeddings as .pt files for:
  - Offline kNN sweep / linear probe without re-running the encoder
  - UMAP / t-SNE visualization of the representation space
  - Comparing embeddings across checkpoints (e.g., epoch 10 vs epoch 100)
  - Fast hyperparameter search over kNN k or linear probe lr

Usage:
  python scripts/extract_embeddings.py \
      --config configs/temporal_mask.yaml \
      --checkpoint /scratch/$NETID/checkpoints/ldjjepa_temporal/latest.pt \
      --output_dir /scratch/$NETID/embeddings/temporal_epoch100 \
      --splits train val test \
      --batch_size 32

Output structure:
  output_dir/
      train_embeddings.pt   → {"embeddings": (8750, D), "alpha": (8750,), "zeta": (8750,)}
      val_embeddings.pt     → {"embeddings": (1200, D), "alpha": (1200,), "zeta": (1200,)}
      test_embeddings.pt    → {"embeddings": (1300, D), "alpha": (1300,), "zeta": (1300,)}
      meta.json             → checkpoint info, config path, embed_dim, timestamp
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import load_config
from models.ld_jepa import LDJEPA
from data.dataset import ActiveMatterDataset
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_split(
    model: LDJEPA,
    dataset: ActiveMatterDataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    desc: str = "",
) -> dict:
    """
    Extract embeddings for every sample in a dataset split.

    Returns:
        dict with keys:
            "embeddings": FloatTensor (N, D)
            "alpha":      FloatTensor (N,)
            "zeta":       FloatTensor (N,)
            "file_idx":   LongTensor  (N,)  — index into dataset.files[]
    """
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,          # CRITICAL: keep order for file_idx tracking
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_embs   = []
    all_alphas = []
    all_zetas  = []
    all_idx    = []

    n_total   = len(dataset)
    n_done    = 0
    t0        = time.time()

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device, non_blocking=True)     # (B, C, T, H, W)

        # Forward pass: mean-pool all T*N tokens → (B, D)
        emb = raw_model.get_embedding(x)         # (B, D)

        B = x.shape[0]
        global_start = batch_idx * batch_size

        all_embs.append(emb.cpu().float())
        all_alphas.append(labels["alpha"].float())
        all_zetas.append(labels["zeta"].float())
        all_idx.append(torch.arange(global_start, global_start + B))

        n_done += B

        # Progress logging
        if (batch_idx + 1) % 20 == 0 or n_done >= n_total:
            elapsed = time.time() - t0
            speed   = n_done / elapsed
            eta     = (n_total - n_done) / max(speed, 1e-6)
            print(f"  {desc} [{n_done:5d}/{n_total}] "
                  f"{speed:.1f} samples/s  ETA {eta:.0f}s")

    embeddings = torch.cat(all_embs,   dim=0)   # (N, D)
    alphas     = torch.cat(all_alphas, dim=0)   # (N,)
    zetas      = torch.cat(all_zetas,  dim=0)   # (N,)
    file_idx   = torch.cat(all_idx,    dim=0)   # (N,)

    elapsed = time.time() - t0
    print(f"  {desc} done: {len(embeddings)} samples in {elapsed:.1f}s  "
          f"embedding shape: {embeddings.shape}")

    # Basic health checks
    nan_frac = torch.isnan(embeddings).float().mean().item()
    if nan_frac > 0:
        print(f"  WARNING: {nan_frac*100:.1f}% NaN values in embeddings — "
              f"check checkpoint and normalization stats")

    emb_std = embeddings.std(dim=0).mean().item()
    emb_norm = embeddings.norm(dim=-1).mean().item()
    print(f"  Embedding stats: mean_norm={emb_norm:.3f}  mean_std_per_dim={emb_std:.4f}")

    return {
        "embeddings": embeddings,
        "alpha":      alphas,
        "zeta":       zetas,
        "file_idx":   file_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frozen encoder embeddings for all dataset splits"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (e.g. configs/temporal_mask.yaml)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pt checkpoint file (use 'latest.pt' for most recent)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save .pt embedding files"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to extract (default: all three)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for extraction (32 is safe on A100 40GB)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker processes"
    )
    parser.add_argument(
        "--stats_cache", type=str, default=None,
        help="Path to channel_stats.npz (auto-detected from checkpoint dir if omitted)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (e.g. 'cpu', 'cuda:0'). Auto-detected if omitted."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract_embeddings] Device: {device}")

    # ── Output directory ─────────────────────────────────────────────────────
    output_dir = Path(os.path.expandvars(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract_embeddings] Output: {output_dir}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = os.path.expandvars(args.checkpoint)
    print(f"[extract_embeddings] Loading checkpoint: {ckpt_path}")

    model = LDJEPA(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.context_encoder.load_state_dict(ckpt["context_encoder"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    # Note: diffusion_pred weights NOT needed for embedding extraction
    model.eval()

    embed_dim = cfg.model.embed_dim
    epoch     = ckpt.get("epoch", -1)
    step      = ckpt.get("step",  -1)
    print(f"[extract_embeddings] Checkpoint: epoch={epoch}, step={step}, embed_dim={embed_dim}")

    # ── Channel stats ─────────────────────────────────────────────────────────
    # Try to auto-detect stats file from checkpoint directory
    stats_cache = args.stats_cache
    if stats_cache is None:
        ckpt_dir    = Path(ckpt_path).parent
        auto_stats  = ckpt_dir / "channel_stats.npz"
        if auto_stats.exists():
            stats_cache = str(auto_stats)
            print(f"[extract_embeddings] Auto-detected stats: {stats_cache}")
        else:
            print(f"[extract_embeddings] WARNING: No channel_stats.npz found. "
                  f"Using zero-mean/unit-std normalization. "
                  f"Pass --stats_cache for correct normalization.")

    # ── Extract embeddings per split ──────────────────────────────────────────
    results_meta = {
        "checkpoint":  ckpt_path,
        "config":      args.config,
        "epoch":       epoch,
        "step":        step,
        "embed_dim":   embed_dim,
        "splits":      args.splits,
        "timestamp":   datetime.now().isoformat(),
        "files":       {},
    }

    for split in args.splits:
        print(f"\n[extract_embeddings] Extracting {split} split...")

        dataset = ActiveMatterDataset(
            data_root=os.path.expandvars(cfg.data.data_root),
            split=split,
            num_frames=cfg.data.num_frames,
            spatial_size=cfg.data.spatial_size,
            num_channels=cfg.data.num_channels,
            normalize=True,
            augment=False,          # NO augmentation during extraction
            precompute_stats=False,
            stats_cache=stats_cache,
        )

        result = extract_split(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            desc=split,
        )

        # Save to disk
        out_path = output_dir / f"{split}_embeddings.pt"
        torch.save(result, out_path)
        print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

        results_meta["files"][split] = {
            "path":       str(out_path),
            "n_samples":  len(result["embeddings"]),
            "shape":      list(result["embeddings"].shape),
        }

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(results_meta, f, indent=2)
    print(f"\n[extract_embeddings] Metadata saved: {meta_path}")

    # ── Quick downstream eval from saved embeddings ───────────────────────────
    if set(args.splits) >= {"train", "val"}:
        print("\n[extract_embeddings] Running quick kNN eval from saved embeddings...")
        from evaluation.knn_regression import run_knn_eval

        train_data = torch.load(output_dir / "train_embeddings.pt", weights_only=True)
        val_data   = torch.load(output_dir / "val_embeddings.pt",   weights_only=True)

        train_labels = {"alpha": train_data["alpha"], "zeta": train_data["zeta"]}
        val_labels   = {"alpha": val_data["alpha"],   "zeta": val_data["zeta"]}

        knn_results = run_knn_eval(
            train_data["embeddings"], train_labels,
            val_data["embeddings"],   val_labels,
            k=cfg.evaluation.knn_k,
            device=device,
        )
        results_meta["quick_knn_val"] = knn_results

        # Update meta with eval results
        with open(meta_path, "w") as f:
            json.dump(results_meta, f, indent=2)

        print(f"\n  Quick kNN val: "
              f"alpha_mse={knn_results['alpha_mse']:.4f}  "
              f"zeta_mse={knn_results['zeta_mse']:.4f}")

    print(f"\n[extract_embeddings] Done. All embeddings in: {output_dir}")
    print(f"  Load with: data = torch.load('{output_dir}/train_embeddings.pt')")
    print(f"             embs = data['embeddings']  # (N, {embed_dim})")
    print(f"             alpha = data['alpha']       # (N,)")
    print(f"             zeta  = data['zeta']        # (N,)")


if __name__ == "__main__":
    main()