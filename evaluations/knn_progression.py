"""
evaluation/knn_regression.py
-----------------------------
kNN regression evaluation on frozen encoder embeddings.

Protocol:
  - Use frozen encoder embeddings (same as linear probe)
  - Fit kNN regressor (k=20, cosine distance) on training embeddings
  - Predict continuous alpha and zeta values on val/test
  - Report MSE on z-score normalized labels

We use a pure PyTorch implementation (no sklearn) for GPU-accelerated
nearest-neighbor search — critical given N_train=8750, D=384.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger("ld_jepa")


def cosine_knn_regression(
    train_embs:   torch.Tensor,       # (N_train, D)
    train_labels: torch.Tensor,       # (N_train, 2) — [alpha, zeta], normalized
    query_embs:   torch.Tensor,       # (N_query, D)
    k:            int = 20,
    batch_size:   int = 256,
    device:       torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    GPU-accelerated cosine kNN regression.

    For each query embedding, finds k nearest neighbors in the training set
    (by cosine similarity) and returns the mean of their labels.

    Uses chunked computation to avoid OOM on large N_train.

    Returns:
        predictions: (N_query, 2) predicted [alpha, zeta] values
    """
    # L2-normalize for cosine similarity via dot product
    train_norm  = F.normalize(train_embs, dim=-1).to(device)   # (N_train, D)
    train_y     = train_labels.to(device)                       # (N_train, 2)
    query_norm  = F.normalize(query_embs, dim=-1).to(device)   # (N_query, D)

    N_query = query_norm.shape[0]
    preds   = torch.zeros(N_query, 2, device=device)

    for start in range(0, N_query, batch_size):
        end    = min(start + batch_size, N_query)
        q_chunk = query_norm[start:end]                # (chunk, D)

        # Cosine similarity: (chunk, N_train)
        sim = q_chunk @ train_norm.T

        # Top-k nearest neighbors
        topk_sim, topk_idx = sim.topk(k, dim=-1, largest=True, sorted=False)
        # topk_idx: (chunk, k)

        # Distance-weighted average (weight by similarity)
        weights = F.softmax(topk_sim, dim=-1)               # (chunk, k)
        nn_labels = train_y[topk_idx]                        # (chunk, k, 2)
        pred_chunk = (weights.unsqueeze(-1) * nn_labels).sum(dim=1)  # (chunk, 2)

        preds[start:end] = pred_chunk

    return preds.cpu()


def run_knn_eval(
    train_embs:   torch.Tensor,
    train_labels: Dict[str, torch.Tensor],
    val_embs:     torch.Tensor,
    val_labels:   Dict[str, torch.Tensor],
    test_embs:    Optional[torch.Tensor] = None,
    test_labels:  Optional[Dict[str, torch.Tensor]] = None,
    k:            int = 20,
    device:       torch.device = torch.device("cpu"),
) -> Dict:
    """
    Full kNN regression evaluation pipeline.

    Labels are z-score normalized using training set statistics
    (same normalization as linear probe for fair comparison).

    Returns dict of MSE results.
    """
    from evaluation.linear_probe import zscore_normalize_labels

    # Z-score normalize (same as linear probe)
    norm_train, norm_val, norm_test, label_stats = zscore_normalize_labels(
        train_labels, val_labels, test_labels
    )

    # Stack to (N, 2)
    train_y = torch.stack([norm_train["alpha"], norm_train["zeta"]], dim=1)
    val_y   = torch.stack([norm_val["alpha"],   norm_val["zeta"]],   dim=1)

    # Run kNN
    val_pred = cosine_knn_regression(
        train_embs, train_y, val_embs, k=k, device=device
    )

    results = {
        "alpha_mse": F.mse_loss(val_pred[:, 0], val_y[:, 0]).item(),
        "zeta_mse":  F.mse_loss(val_pred[:, 1], val_y[:, 1]).item(),
        "total_mse": F.mse_loss(val_pred, val_y).item(),
        "k": k,
    }

    logger.info(
        f"  [kNN k={k}] "
        f"alpha_mse={results['alpha_mse']:.4f} "
        f"zeta_mse={results['zeta_mse']:.4f} "
        f"total={results['total_mse']:.4f}"
    )

    # Test set
    if test_embs is not None and norm_test is not None:
        test_y    = torch.stack([norm_test["alpha"], norm_test["zeta"]], dim=1)
        test_pred = cosine_knn_regression(
            train_embs, train_y, test_embs, k=k, device=device
        )
        results["test_alpha_mse"] = F.mse_loss(test_pred[:, 0], test_y[:, 0]).item()
        results["test_zeta_mse"]  = F.mse_loss(test_pred[:, 1], test_y[:, 1]).item()
        results["test_total_mse"] = F.mse_loss(test_pred, test_y).item()

        logger.info(
            f"  [kNN TEST k={k}] "
            f"alpha_mse={results['test_alpha_mse']:.4f} "
            f"zeta_mse={results['test_zeta_mse']:.4f}"
        )

    return results


def sweep_k(
    train_embs:   torch.Tensor,
    train_labels: Dict[str, torch.Tensor],
    val_embs:     torch.Tensor,
    val_labels:   Dict[str, torch.Tensor],
    k_values:     list = [5, 10, 20, 50, 100],
    device:       torch.device = torch.device("cpu"),
) -> Dict:
    """
    Sweep over k values and return all results.
    Useful for ablation: which k is optimal for this representation?
    """
    from evaluation.linear_probe import zscore_normalize_labels

    norm_train, norm_val, _, _ = zscore_normalize_labels(
        train_labels, val_labels
    )
    train_y = torch.stack([norm_train["alpha"], norm_train["zeta"]], dim=1)
    val_y   = torch.stack([norm_val["alpha"],   norm_val["zeta"]],   dim=1)

    all_results = {}
    for k in k_values:
        val_pred = cosine_knn_regression(
            train_embs, train_y, val_embs, k=k, device=device
        )
        total_mse = F.mse_loss(val_pred, val_y).item()
        all_results[k] = total_mse
        logger.info(f"  kNN k={k:3d}: total_mse={total_mse:.4f}")

    best_k   = min(all_results, key=all_results.get)
    logger.info(f"  Best k={best_k} (mse={all_results[best_k]:.4f})")

    return {"k_sweep": all_results, "best_k": best_k}