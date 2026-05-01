#!/usr/bin/env python3
"""
scripts/evaluate.py
-------------------
Final evaluation script: loads a trained checkpoint and runs
linear probe + kNN regression on val and test splits.

Usage:
  python scripts/evaluate.py \
      --config configs/temporal_mask.yaml \
      --checkpoint /scratch/$NETID/checkpoints/ldjjepa_temporal/latest.pt \
      --split test \
      --output_dir /scratch/$NETID/results/

Outputs:
  - results.json: all MSE metrics
  - embeddings/: saved train/val/test embeddings (for kNN sweep, UMAP, etc.)
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.trainer import load_config
from models.ld_jepa import LDJEPA
from data.dataset import build_dataloaders
from evaluation.linear_probe import (
    extract_embeddings, run_linear_probe_eval
)
from evaluation.knn_regression import run_knn_eval, sweep_k


def parse_args():
    parser = argparse.ArgumentParser(description="LD-JEPA Evaluation")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split",      type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--k_sweep",    action="store_true",
                        help="Sweep over k values for kNN")
    parser.add_argument("--save_embs",  action="store_true",
                        help="Save embeddings to disk (for UMAP/analysis)")
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[evaluate.py] Loading checkpoint: {args.checkpoint}")
    print(f"[evaluate.py] Evaluating on: {args.split}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = LDJEPA(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.context_encoder.load_state_dict(ckpt["context_encoder"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.diffusion_pred.load_state_dict(ckpt["diffusion_pred"])
    model.eval()

    print(f"[evaluate.py] Checkpoint loaded from epoch {ckpt['epoch']}, "
          f"step {ckpt['step']}")

    # ── Build dataloaders ─────────────────────────────────────────────────────
    # Use batch_size=32 for fast extraction
    cfg.training.batch_size = 32
    loaders = build_dataloaders(cfg, rank=0, world_size=1)

    # ── Extract embeddings ────────────────────────────────────────────────────
    print("[evaluate.py] Extracting train embeddings...")
    train_embs, train_labels = extract_embeddings(
        model, loaders["train"], device, max_samples=None
    )
    print(f"  Train: {train_embs.shape}")

    print(f"[evaluate.py] Extracting {args.split} embeddings...")
    eval_embs, eval_labels = extract_embeddings(
        model, loaders[args.split], device, max_samples=None
    )
    print(f"  {args.split}: {eval_embs.shape}")

    # Also extract val for linear probe hyperparameter selection
    val_embs, val_labels = None, None
    if args.split == "test":
        print("[evaluate.py] Extracting val embeddings for probe training...")
        val_embs, val_labels = extract_embeddings(
            model, loaders["val"], device, max_samples=None
        )

    # ── Save embeddings ───────────────────────────────────────────────────────
    if args.save_embs:
        emb_dir = output_dir / "embeddings"
        emb_dir.mkdir(exist_ok=True)
        torch.save({"embeddings": train_embs, "labels": train_labels},
                   emb_dir / "train_embeddings.pt")
        torch.save({"embeddings": eval_embs, "labels": eval_labels},
                   emb_dir / f"{args.split}_embeddings.pt")
        print(f"[evaluate.py] Embeddings saved to {emb_dir}")

    # ── Linear probe ──────────────────────────────────────────────────────────
    print("[evaluate.py] Running linear probe evaluation...")

    if args.split == "test":
        lp_results = run_linear_probe_eval(
            train_embs, train_labels,
            val_embs,   val_labels,    # use val for probe training
            test_embs=eval_embs,
            test_labels=eval_labels,
            device=device,
        )
    else:
        lp_results = run_linear_probe_eval(
            train_embs, train_labels,
            eval_embs,  eval_labels,
            device=device,
        )

    # ── kNN regression ────────────────────────────────────────────────────────
    print("[evaluate.py] Running kNN regression evaluation...")

    if args.k_sweep:
        knn_sweep = sweep_k(train_embs, train_labels, eval_embs, eval_labels,
                            k_values=[5, 10, 20, 50, 100], device=device)
        best_k = knn_sweep["best_k"]
    else:
        best_k = cfg.evaluation.knn_k

    knn_results = run_knn_eval(
        train_embs, train_labels,
        eval_embs,  eval_labels,
        test_embs=(eval_embs if args.split == "test" else None),
        test_labels=(eval_labels if args.split == "test" else None),
        k=best_k,
        device=device,
    )

    # ── Compile and save results ──────────────────────────────────────────────
    all_results = {
        "checkpoint":   args.checkpoint,
        "config":       args.config,
        "split":        args.split,
        "epoch":        ckpt["epoch"],
        "linear_probe": lp_results,
        "knn":          knn_results,
    }

    if args.k_sweep:
        all_results["knn_sweep"] = knn_sweep

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  FINAL RESULTS ({args.split} split)")
    print("=" * 60)
    print(f"  Linear Probe:")
    print(f"    alpha MSE: {lp_results.get('alpha_mse', lp_results.get('test_alpha_mse', 'N/A')):.4f}")
    print(f"    zeta  MSE: {lp_results.get('zeta_mse',  lp_results.get('test_zeta_mse',  'N/A')):.4f}")
    print(f"  kNN (k={best_k}):")
    print(f"    alpha MSE: {knn_results.get('alpha_mse', knn_results.get('test_alpha_mse', 'N/A')):.4f}")
    print(f"    zeta  MSE: {knn_results.get('zeta_mse',  knn_results.get('test_zeta_mse',  'N/A')):.4f}")
    print("=" * 60)
    print(f"\n  Full results saved to: {results_path}")


if __name__ == "__main__":
    main()