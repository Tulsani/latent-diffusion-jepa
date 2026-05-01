"""
training/losses.py
------------------
Loss functions and representation health diagnostics.

Includes:
  - vicreg_loss: variance-covariance regularization to prevent collapse
  - representation diagnostics: eigenspectrum rank, mean/std checks
    (important for ablation analysis in report)
"""

import torch
import torch.nn.functional as F
from typing import Dict


def check_representation_collapse(z: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    """
    Diagnose representation collapse.

    A collapsed representation has:
      - Very low effective rank (most variance in 1-2 dimensions)
      - Very small std across batch dimension
      - High cosine similarity between samples

    Returns a dict of diagnostic metrics to log to W&B / Slurm logs.

    Args:
        z: (B, D) embedding matrix
        prefix: optional prefix for metric names

    Reference: Hua et al. 2021 - On Feature Decorrelation in Self-Supervised Learning
    """
    B, D = z.shape
    metrics = {}

    with torch.no_grad():
        z_centered = z - z.mean(dim=0, keepdim=True)

        # ── Std per dimension ─────────────────────────────────────────────
        std_per_dim = z_centered.std(dim=0)                    # (D,)
        metrics[f"{prefix}std_mean"]   = std_per_dim.mean().item()
        metrics[f"{prefix}std_min"]    = std_per_dim.min().item()
        metrics[f"{prefix}std_dead"]   = (std_per_dim < 0.01).float().mean().item()
        # "dead" dimensions: std < 0.01 → those dims are collapsed

        # ── Effective rank via normalized singular values ──────────────────
        # eff_rank = exp(entropy of normalized singular values)
        # = D for perfectly uniform, ~1 for fully collapsed
        if B >= D:
            # Full SVD feasible
            try:
                _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
                S = S[:min(B, D)]
            except Exception:
                S = None
        else:
            # B < D: compute SVD of (B,D) directly
            try:
                _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
            except Exception:
                S = None

        if S is not None and S.sum() > 0:
            S_norm = S / (S.sum() + 1e-8)
            entropy = -(S_norm * (S_norm + 1e-10).log()).sum()
            eff_rank = entropy.exp().item()
            metrics[f"{prefix}effective_rank"] = eff_rank
            metrics[f"{prefix}top1_sv_frac"]   = (S[0] / (S.sum() + 1e-8)).item()
            # top1_sv_frac ≈ 1.0 means all variance in one direction → collapsed

        # ── Mean pairwise cosine similarity (small sample for speed) ──────
        if B >= 8:
            z_norm = F.normalize(z_centered[:32], dim=-1)   # at most 32 samples
            sim_matrix = z_norm @ z_norm.T                  # (32, 32)
            # Exclude diagonal
            mask = ~torch.eye(z_norm.shape[0], dtype=torch.bool, device=z.device)
            mean_cos_sim = sim_matrix[mask].mean().item()
            metrics[f"{prefix}mean_cos_sim"] = mean_cos_sim
            # mean_cos_sim ≈ 1.0 → all representations are the same → collapsed
            # mean_cos_sim ≈ 0.0 → good diversity

    return metrics