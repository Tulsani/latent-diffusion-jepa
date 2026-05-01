"""
models/mlp_predictor.py
-----------------------
Standard JEPA MLP/Transformer predictor — the ablation baseline.

This is a DROP-IN replacement for LatentDiffusionPredictor.
Uses the EXACT same encoder, masking, and training loop.
The only difference: instead of a DDPM denoiser, we use a small
Transformer that directly regresses z_target from z_context.

This ablation directly answers:
  "Does the diffusion predictor help over a standard deterministic predictor?"

Architecture: context tokens → cross-attention Transformer → z_target prediction
Loss: L2 (MSE) in latent space (standard JEPA objective)

To use: set model.predictor_type: "mlp" in config instead of "diffusion"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class PredictorBlock(nn.Module):
    """
    One Transformer block for the deterministic predictor.
    Target token queries attend to context token keys/values.
    """

    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()

        # Self-attention on target tokens
        self.norm1    = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Cross-attention to context tokens
        self.norm2      = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # MLP
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, M_target, D)
        context: (B, M_context, D)
        """
        # Self-attention
        res = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, need_weights=False)
        x = res + x

        # Cross-attention to context
        res = x
        x = self.norm2(x)
        x, _ = self.cross_attn(x, context, context, need_weights=False)
        x = res + x

        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class MLPPredictor(nn.Module):
    """
    Deterministic Transformer predictor (the JEPA baseline).

    Directly predicts z_target embeddings from z_context.
    Loss: L2 MSE in latent space.

    This is intentionally lightweight (~same param count as diffusion predictor)
    for a fair ablation comparison.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        predictor_depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable mask token: a single learned vector broadcast over all
        # target positions, giving the predictor a "slot" to fill
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Predictor Transformer blocks
        self.blocks = nn.ModuleList([
            PredictorBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(predictor_depth)
        ])

        self.norm     = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Zero-init output: start from predicting zero → stable early training
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"[MLPPredictor] Parameters: {total:,} ({total/1e6:.2f}M)")

    def forward(
        self,
        z_context: torch.Tensor,   # (B, M_context, D)
        num_target_tokens: int,    # how many target tokens to predict
    ) -> torch.Tensor:
        """
        Predict z_target from z_context.

        Returns: z_target_pred (B, num_target_tokens, D)
        """
        B = z_context.shape[0]
        D = self.embed_dim

        # Initialize target slots with learned mask token
        x = self.mask_token.expand(B, num_target_tokens, D)  # (B, M_t, D)

        # Process through predictor blocks
        for block in self.blocks:
            x = block(x, z_context)

        x = self.norm(x)
        z_pred = self.out_proj(x)   # (B, M_target, D)
        return z_pred

    def compute_loss(
        self,
        z_context: torch.Tensor,   # (B, M_context, D)
        z_target: torch.Tensor,    # (B, M_target, D) from EMA encoder
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute L2 prediction loss in latent space.

        This is the standard JEPA objective:
          L = ||z_pred - z_target||^2

        Returns:
            loss:    scalar tensor
            metrics: dict of loggable values
        """
        M_target = z_target.shape[1]
        z_pred   = self.forward(z_context, M_target)

        loss = F.mse_loss(z_pred, z_target)

        # Cosine similarity between predicted and target (good diagnostic)
        with torch.no_grad():
            z_pred_n  = F.normalize(z_pred.reshape(-1, self.embed_dim), dim=-1)
            z_tgt_n   = F.normalize(z_target.reshape(-1, self.embed_dim), dim=-1)
            cos_sim   = (z_pred_n * z_tgt_n).sum(dim=-1).mean().item()

        metrics = {
            "mlp_pred_loss":    loss.item(),
            "pred_cos_sim":     cos_sim,   # should rise during training
            "z0_pred_mse":      loss.item(),  # alias for uniform logging w/ diffusion
            "diffusion_loss":   loss.item(),  # alias — trainer uses this key
            "mean_t":           0.0,          # not applicable, kept for uniform logging
        }

        return loss, metrics