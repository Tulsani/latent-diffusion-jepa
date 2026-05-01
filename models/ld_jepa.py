"""
models/ld_jepa.py
-----------------
Full LD-JEPA (Latent Diffusion Joint-Embedding Predictive Architecture) model.

Components:
  1. context_encoder  — processes visible/context tokens, updated by gradient
  2. target_encoder   — EMA copy of context_encoder, processes all tokens
                        (no gradient, updated by exponential moving average)
  3. diffusion_pred   — latent diffusion predictor conditioned on context_encoder output
  4. mask_sampler     — either temporal or spatiotemporal (from config)

Training forward pass:
  1. Sample masks (context tokens, target tokens)
  2. Context encoder: encode visible context tokens → z_context
  3. Target encoder (EMA, no_grad): encode ALL tokens → z_all → select z_target tokens
  4. Diffusion predictor: compute_loss(z_context, z_target) → L_diffusion
  5. Optional VICReg loss on z_context → L_vicreg (prevents collapse)
  6. Total loss: L_diffusion + λ_vicreg * L_vicreg

Representation extraction (for linear probe / kNN):
  - Use ONLY context_encoder.get_embedding(x) → (B, D)
  - Diffusion predictor is NOT used — zero inference overhead at eval time
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from models.encoder import PhysicsEncoder
from models.diffusion import LatentDiffusionPredictor
from models.mlp_predictor import MLPPredictor
from data.masking import build_mask_sampler, TemporalMaskSampler, SpatiotemporalMaskSampler


# ─────────────────────────────────────────────────────────────────────────────
# VICReg auxiliary loss (collapse prevention)
# ─────────────────────────────────────────────────────────────────────────────

def vicreg_loss(
    z: torch.Tensor,        # (B, D) embeddings
    lam: float = 25.0,      # invariance weight (not used here — no two views)
    mu: float = 25.0,       # variance weight
    nu: float = 1.0,        # covariance weight
) -> torch.Tensor:
    """
    VICReg regularization terms on a single embedding batch.

    We apply variance + covariance terms on z_context embeddings
    (mean-pooled over tokens) to prevent representation collapse.

    Reference: Bardes et al. 2022 - VICReg
    """
    B, D = z.shape
    z = z - z.mean(dim=0, keepdim=True)   # center

    # Variance term: push std of each dimension toward 1
    std  = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std))

    # Covariance term: push off-diagonal covariance toward 0
    cov = (z.T @ z) / (B - 1)   # (D, D)
    diag_mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
    cov_loss  = cov[diag_mask].pow(2).sum() / D

    return mu * var_loss + nu * cov_loss


# ─────────────────────────────────────────────────────────────────────────────
# EMA update
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, momentum: float):
    """
    Update EMA target encoder:
      θ_ema ← momentum * θ_ema + (1 - momentum) * θ

    Called every training step. momentum is cosine-annealed from
    ema_start (0.996) to ema_end (0.9999) over training.
    """
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(momentum).add_(p.data, alpha=1.0 - momentum)


def get_ema_momentum(
    step: int,
    total_steps: int,
    ema_start: float = 0.996,
    ema_end: float = 0.9999,
) -> float:
    """Cosine anneal EMA momentum from ema_start to ema_end."""
    return ema_end - (ema_end - ema_start) * (
        math.cos(math.pi * step / total_steps) + 1
    ) / 2


import math


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class LDJEPA(nn.Module):
    """
    Latent Diffusion JEPA.

    This is the full model combining all components.
    In DDP training, only context_encoder and diffusion_pred have gradients.
    target_encoder is updated via EMA.
    """

    def __init__(self, cfg):
        super().__init__()

        # ── Context encoder (trained via backprop) ──────────────────────────
        self.context_encoder = PhysicsEncoder(
            in_channels=cfg.data.num_channels,
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.encoder_depth,
            num_heads=cfg.model.encoder_heads,
            mlp_ratio=cfg.model.encoder_mlp_ratio,
            dropout=cfg.model.dropout,
            num_frames=cfg.data.num_frames,
            spatial_size=cfg.data.spatial_size,
        )

        # ── Target encoder (EMA, no gradients) ─────────────────────────────
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # ── Predictor: diffusion (proposed) or MLP (ablation baseline) ─────
        predictor_type = getattr(cfg.model, "predictor_type", "diffusion")
        self.predictor_type = predictor_type

        if predictor_type == "diffusion":
            self.diffusion_pred = LatentDiffusionPredictor(
                embed_dim=cfg.model.embed_dim,
                predictor_depth=cfg.model.predictor_depth,
                num_heads=cfg.model.predictor_heads,
                mlp_ratio=4.0,
                diffusion_steps=cfg.model.diffusion_steps,
                diffusion_schedule=cfg.model.diffusion_schedule,
                ddim_steps=cfg.model.ddim_steps,
            )
        elif predictor_type == "mlp":
            self.diffusion_pred = MLPPredictor(
                embed_dim=cfg.model.embed_dim,
                predictor_depth=cfg.model.predictor_depth,
                num_heads=cfg.model.predictor_heads,
                mlp_ratio=4.0,
            )
        else:
            raise ValueError(
                f"Unknown predictor_type: '{predictor_type}'. "
                f"Choose from ['diffusion', 'mlp']"
            )

        # ── Mask sampler ─────────────────────────────────────────────────────
        self.mask_sampler = build_mask_sampler(cfg)

        # ── Config ──────────────────────────────────────────────────────────
        self.cfg = cfg
        self.use_vicreg = cfg.training.use_vicreg_aux
        self.vicreg_weight = cfg.training.vicreg_aux_weight

        self._print_total_params()

    def _print_total_params(self):
        """Print and verify total parameter count (must be < 100M)."""
        enc_params   = sum(p.numel() for p in self.context_encoder.parameters())
        pred_params  = sum(p.numel() for p in self.diffusion_pred.parameters())
        # target_encoder shares architecture with context_encoder (EMA copy, no grad)
        total_params = enc_params + pred_params  # EMA encoder not counted (no grad update)

        print(f"\n{'='*55}")
        print(f"  LD-JEPA Parameter Count:")
        print(f"    Context encoder:     {enc_params:>12,} ({enc_params/1e6:.2f}M)")
        print(f"    Diffusion predictor: {pred_params:>12,} ({pred_params/1e6:.2f}M)")
        print(f"    Total (trainable):   {total_params:>12,} ({total_params/1e6:.2f}M)")
        print(f"    Limit:               {'< 100M':>12}")
        assert total_params < 100_000_000, \
            f"Model exceeds 100M parameter limit! Got {total_params/1e6:.1f}M"
        print(f"  ✓ Parameter count within limit")
        print(f"{'='*55}\n")

    def forward(
        self,
        x: torch.Tensor,            # (B, C, T, H, W)
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full training forward pass.

        Returns:
            total_loss: scalar tensor (backprop through this)
            metrics:    dict of loggable scalars
        """
        B = x.shape[0]
        device = x.device

        # ── 1. Sample masks ──────────────────────────────────────────────────
        context_mask, target_mask = self.mask_sampler.sample(B, device)
        # context_mask: (B, T*N) True = context token
        # target_mask:  (B, T*N) True = target token

        # ── 2. Context encoder (with gradient) ──────────────────────────────
        # Only processes context (unmasked) tokens
        z_context = self.context_encoder(x, context_mask=context_mask)
        # z_context: (B, M_context, D)

        # ── 3. Target encoder (EMA, no gradient) ────────────────────────────
        with torch.no_grad():
            z_all = self.target_encoder(x, context_mask=None)
            # z_all: (B, T*N, D) — all token embeddings from EMA encoder

            # Extract target token embeddings
            # target_mask[0]: (T*N,) — same mask for all batch items
            target_indices = target_mask[0].nonzero(as_tuple=True)[0]
            z_target = z_all[:, target_indices, :]   # (B, M_target, D)

        # ── 4. Diffusion predictor loss ──────────────────────────────────────
        diff_loss, diff_metrics = self.diffusion_pred.compute_loss(z_context, z_target)

        # ── 5. VICReg collapse prevention (optional) ─────────────────────────
        vicreg = torch.tensor(0.0, device=device)
        if self.use_vicreg and z_context.shape[0] > 1:
            # Requires B > 1 for meaningful variance computation
            # Apply on mean-pooled context embeddings
            z_mean = z_context.mean(dim=1)   # (B, D)
            vicreg = vicreg_loss(
                z_mean,
                mu=self.cfg.training.vicreg_mu,
                nu=self.cfg.training.vicreg_nu,
            )

        # ── 6. Total loss ────────────────────────────────────────────────────
        total_loss = diff_loss + self.vicreg_weight * vicreg

        # ── 7. EMA update (called here for single-GPU; DDP trainer calls separately) ──
        if step is not None and total_steps is not None:
            momentum = get_ema_momentum(
                step, total_steps,
                ema_start=self.cfg.model.ema_momentum,
                ema_end=0.9999,
            )
            update_ema(self.target_encoder, self.context_encoder, momentum)

        metrics = {
            "loss":             total_loss.item(),
            "diffusion_loss":   diff_loss.item(),
            "vicreg_loss":      vicreg.item(),
            "context_tokens":   z_context.shape[1],
            "target_tokens":    z_target.shape[1],
            **diff_metrics,
        }

        return total_loss, metrics

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract (B, D) representation for linear probing / kNN.

        Uses only the frozen context_encoder in eval mode.
        No masking, no diffusion — pure encoder forward pass.

        Args:
            x: (B, C, T, H, W)
        Returns:
            emb: (B, D) mean-pooled over all T*N tokens
        """
        self.context_encoder.eval()
        emb = self.context_encoder.get_embedding(x)
        return emb

    def get_trainable_params(self):
        """Return only trainable parameters (context_encoder + diffusion_pred)."""
        return list(self.context_encoder.parameters()) \
             + list(self.diffusion_pred.parameters())