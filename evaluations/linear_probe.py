"""
evaluation/linear_probe.py
--------------------------
Linear probing evaluation for LD-JEPA representations.

Protocol (strictly per project rules):
  1. Freeze context_encoder completely (no gradient updates)
  2. Extract (B, D) embeddings for ALL training samples
  3. Fit a SINGLE linear layer (no MLP, no attention) via MSE regression
  4. Evaluate on validation / test splits
  5. Report MSE for alpha and zeta separately (z-score normalized)

Z-score normalization of labels:
  - Compute mean/std of alpha and zeta on training split
  - Normalize train + val + test labels
  - Report MSE on normalized labels (as required by project spec)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger("ld_jepa")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model,
    loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Extract frozen encoder embeddings for all samples in loader.

    Returns:
        embeddings: (N, D) tensor
        labels:     dict with 'alpha' (N,) and 'zeta' (N,) tensors
    """
    model.eval()
    # Get raw model (unwrap DDP if needed)
    raw_model = model.module if hasattr(model, "module") else model

    all_embs   = []
    all_alpha  = []
    all_zeta   = []
    n_collected = 0

    for x, labels in loader:
        if max_samples and n_collected >= max_samples:
            break

        x = x.to(device, non_blocking=True)   # (B, C, T, H, W)

        # Extract (B, D) embedding — no masking, mean pool all tokens
        emb = raw_model.get_embedding(x)       # (B, D)

        all_embs.append(emb.cpu().float())
        all_alpha.append(labels["alpha"])
        all_zeta.append(labels["zeta"])
        n_collected += x.shape[0]

    embeddings = torch.cat(all_embs,  dim=0)   # (N, D)
    alphas     = torch.cat(all_alpha, dim=0)   # (N,)
    zetas      = torch.cat(all_zeta,  dim=0)   # (N,)

    if max_samples:
        embeddings = embeddings[:max_samples]
        alphas     = alphas[:max_samples]
        zetas      = zetas[:max_samples]

    return embeddings, {"alpha": alphas, "zeta": zetas}


# ─────────────────────────────────────────────────────────────────────────────
# Label normalization
# ─────────────────────────────────────────────────────────────────────────────

def zscore_normalize_labels(
    train_labels: Dict[str, torch.Tensor],
    val_labels:   Dict[str, torch.Tensor],
    test_labels:  Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple:
    """
    Z-score normalize alpha and zeta labels using training set statistics.

    Returns normalized versions of all label dicts, plus the normalization stats.
    """
    stats = {}
    norm_train = {}
    norm_val   = {}
    norm_test  = {} if test_labels else None

    for key in ["alpha", "zeta"]:
        mean = train_labels[key].mean()
        std  = train_labels[key].std().clamp(min=1e-6)

        stats[key] = {"mean": mean.item(), "std": std.item()}

        norm_train[key] = (train_labels[key] - mean) / std
        norm_val[key]   = (val_labels[key]   - mean) / std
        if test_labels:
            norm_test[key]  = (test_labels[key]  - mean) / std

    return norm_train, norm_val, norm_test, stats


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """
    Single linear layer for regression.
    Predicts both alpha and zeta simultaneously (2 outputs).
    """
    def __init__(self, in_dim: int, out_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_probe(
    train_embs:   torch.Tensor,    # (N_train, D)
    train_labels: Dict[str, torch.Tensor],   # normalized
    val_embs:     torch.Tensor,    # (N_val, D)
    val_labels:   Dict[str, torch.Tensor],   # normalized
    num_epochs:   int = 100,
    lr:           float = 1e-3,
    batch_size:   int = 256,
    device:       torch.device = torch.device("cpu"),
    l2_reg:       float = 1e-4,
) -> Tuple[LinearProbe, Dict]:
    """
    Train a linear probe via MSE regression on frozen embeddings.

    Trains both alpha and zeta prediction with a single linear layer
    (2 output dimensions).

    Returns trained probe and final validation metrics.
    """
    D = train_embs.shape[1]

    # Stack labels into (N, 2) tensor: [alpha, zeta]
    train_y = torch.stack([train_labels["alpha"], train_labels["zeta"]], dim=1)
    val_y   = torch.stack([val_labels["alpha"],   val_labels["zeta"]],   dim=1)

    # DataLoaders
    train_ds = TensorDataset(train_embs, train_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=False, num_workers=0)

    probe     = LinearProbe(D, out_dim=2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_mse = float("inf")
    best_state   = None

    for epoch in range(num_epochs):
        probe.train()
        for emb_b, y_b in train_dl:
            emb_b = emb_b.to(device)
            y_b   = y_b.to(device)
            pred  = probe(emb_b)
            loss  = F.mse_loss(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            probe.eval()
            with torch.no_grad():
                val_pred = probe(val_embs.to(device))
                val_mse_alpha = F.mse_loss(val_pred[:, 0], val_y[:, 0].to(device)).item()
                val_mse_zeta  = F.mse_loss(val_pred[:, 1], val_y[:, 1].to(device)).item()
                val_mse_total = (val_mse_alpha + val_mse_zeta) / 2

            if val_mse_total < best_val_mse:
                best_val_mse = val_mse_total
                best_state   = {k: v.clone() for k, v in probe.state_dict().items()}

    # Restore best
    if best_state:
        probe.load_state_dict(best_state)

    probe.eval()
    with torch.no_grad():
        val_pred      = probe(val_embs.to(device))
        val_y_dev     = val_y.to(device)
        final_metrics = {
            "alpha_mse":  F.mse_loss(val_pred[:, 0], val_y_dev[:, 0]).item(),
            "zeta_mse":   F.mse_loss(val_pred[:, 1], val_y_dev[:, 1]).item(),
            "total_mse":  F.mse_loss(val_pred, val_y_dev).item(),
        }

    logger.info(
        f"  [LinearProbe] alpha_mse={final_metrics['alpha_mse']:.4f} "
        f"zeta_mse={final_metrics['zeta_mse']:.4f} "
        f"total={final_metrics['total_mse']:.4f}"
    )

    return probe, final_metrics


def run_linear_probe_eval(
    train_embs:   torch.Tensor,
    train_labels: Dict[str, torch.Tensor],
    val_embs:     torch.Tensor,
    val_labels:   Dict[str, torch.Tensor],
    test_embs:    Optional[torch.Tensor] = None,
    test_labels:  Optional[Dict[str, torch.Tensor]] = None,
    device:       torch.device = torch.device("cpu"),
) -> Dict:
    """
    Full linear probe evaluation pipeline:
      1. Z-score normalize labels
      2. Train linear probe on train embeddings
      3. Evaluate on val (and optionally test)

    Returns dict of MSE results.
    """
    # Z-score normalize
    norm_train, norm_val, norm_test, label_stats = zscore_normalize_labels(
        train_labels, val_labels, test_labels
    )

    logger.info(
        f"  Label stats: "
        f"alpha mean={label_stats['alpha']['mean']:.4f} "
        f"std={label_stats['alpha']['std']:.4f} | "
        f"zeta mean={label_stats['zeta']['mean']:.4f} "
        f"std={label_stats['zeta']['std']:.4f}"
    )

    # Train linear probe
    probe, val_metrics = train_linear_probe(
        train_embs, norm_train,
        val_embs,   norm_val,
        device=device,
    )

    results = {"val_" + k: v for k, v in val_metrics.items()}
    results.update(val_metrics)   # also without prefix for backward compat

    # Test set evaluation
    if test_embs is not None and norm_test is not None:
        test_y = torch.stack([norm_test["alpha"], norm_test["zeta"]], dim=1)
        probe.eval()
        with torch.no_grad():
            test_pred = probe(test_embs.to(device))
            test_y_d  = test_y.to(device)
            results["test_alpha_mse"]  = F.mse_loss(test_pred[:, 0], test_y_d[:, 0]).item()
            results["test_zeta_mse"]   = F.mse_loss(test_pred[:, 1], test_y_d[:, 1]).item()
            results["test_total_mse"]  = F.mse_loss(test_pred, test_y_d).item()

        logger.info(
            f"  [LinearProbe TEST] "
            f"alpha_mse={results['test_alpha_mse']:.4f} "
            f"zeta_mse={results['test_zeta_mse']:.4f}"
        )

    return results