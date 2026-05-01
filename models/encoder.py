"""
models/encoder.py
-----------------
ViT-style encoder adapted for spatiotemporal physical simulation data.

Key design choices:
  1. 11-channel input projected via a learned linear to embed_dim
     (analogous to 3-channel RGB → embed_dim in standard ViT)
  2. Factored (divided) space-time attention:
       - Spatial attention:  each frame attends within its own 196 patches
       - Temporal attention: each patch location attends across 16 frames
     This reduces memory from O((T*N)^2) → O(T*N^2 + N*T^2)
     For T=16, N=196: 3136^2=9.8M → 16*196^2 + 196*16^2 ≈ 664K (15× reduction)
  3. Sinusoidal 3D positional embeddings (t, h, w)
  4. EMA target encoder is an exact copy — see ld_jepa.py

Parameter budget (ViT-Small adapted):
  embed_dim=384, depth=6, heads=6 → ~2.5M params (well within 100M limit)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """
    Convert spatiotemporal fields to patch embeddings.

    Input:  (B, C, T, H, W)  — C=11 channels, T=16 frames, H=W=224
    Output: (B, T*num_patches, embed_dim)

    Each frame is independently patchified (2D patches), then
    all frame tokens are concatenated along the sequence dimension.
    """

    def __init__(
        self,
        in_channels: int = 11,
        patch_size: int = 16,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim

        # Conv2d with kernel=stride=patch_size is equivalent to patch extraction + linear
        # Operates independently on each frame (treat T as batch dim)
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        returns: (B, T*N, D) where N = (H//patch_size)*(W//patch_size)
        """
        B, C, T, H, W = x.shape

        # Process all frames simultaneously: merge B and T into batch dim
        x = x.permute(0, 2, 1, 3, 4)          # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)          # (B*T, C, H, W)
        x = self.proj(x)                        # (B*T, D, H//P, W//P)

        _, D, Hp, Wp = x.shape
        x = x.flatten(2)                        # (B*T, D, N)
        x = x.transpose(1, 2)                   # (B*T, N, D)
        x = x.reshape(B, T * Hp * Wp, D)       # (B, T*N, D)

        return x


class SinCos3DPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal 3D positional embeddings for (t, h, w) token positions.

    Each token at position (t, h, w) gets a D-dim positional embedding
    computed as concatenation of 1D sin/cos embeddings for each axis,
    with D//3 dims each (with any remainder added to t).

    No learnable parameters → no overfitting on position, generalizes to
    different numbers of frames or spatial sizes.
    """

    def __init__(
        self,
        embed_dim: int,
        num_frames: int = 16,
        num_patches_h: int = 14,
        num_patches_w: int = 14,
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.num_frames   = num_frames
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w

        pe = self._build_embedding()
        self.register_buffer("pe", pe, persistent=False)

    def _build_embedding(self) -> torch.Tensor:
        T  = self.num_frames
        Hp = self.num_patches_h
        Wp = self.num_patches_w
        D  = self.embed_dim

        d_t = D // 3 + D % 3   # temporal gets any remainder
        d_h = D // 3
        d_w = D // 3

        def sincos1d(length, d):
            pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)  # (L, 1)
            div = torch.exp(
                torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d)
            )  # (d//2,)
            pe  = torch.zeros(length, d)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[:d // 2])
            return pe  # (L, d)

        pe_t = sincos1d(T,  d_t)   # (T,  d_t)
        pe_h = sincos1d(Hp, d_h)   # (Hp, d_h)
        pe_w = sincos1d(Wp, d_w)   # (Wp, d_w)

        # Expand to full (T, Hp, Wp, D) grid
        pe_t = pe_t[:, None, None, :].expand(T, Hp, Wp, d_t)
        pe_h = pe_h[None, :, None, :].expand(T, Hp, Wp, d_h)
        pe_w = pe_w[None, None, :, :].expand(T, Hp, Wp, d_w)

        pe = torch.cat([pe_t, pe_h, pe_w], dim=-1)  # (T, Hp, Wp, D)
        pe = pe.reshape(T * Hp * Wp, D)              # (T*N, D)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T*N, D) → x + pe"""
        return x + self.pe.unsqueeze(0)


class Attention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 6, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)     # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Use Flash Attention when available (PyTorch 2.0+)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True,
                                             enable_mem_efficient=True):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FactoredSpaceTimeBlock(nn.Module):
    """
    Factored (divided) space-time attention block.

    Runs two attention operations sequentially:
      1. Spatial attention:  each frame's N=196 patches attend to each other
      2. Temporal attention: each patch location's T=16 tokens attend to each other

    Shared MLP after both attention ops.

    This is ~15× more memory-efficient than full 3D attention for our token counts.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_frames: int = 16,
        num_spatial: int = 196,
    ):
        super().__init__()
        self.num_frames  = num_frames
        self.num_spatial = num_spatial

        # Spatial attention (within each frame)
        self.norm_s  = nn.LayerNorm(dim)
        self.attn_s  = Attention(dim, num_heads, dropout)

        # Temporal attention (across frames for each patch location)
        self.norm_t  = nn.LayerNorm(dim)
        self.attn_t  = Attention(dim, num_heads, dropout)

        # MLP
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp      = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T*N, D) where T=num_frames, N=num_spatial
        """
        B, TN, D = x.shape
        T = self.num_frames
        N = self.num_spatial

        # ── Spatial attention: attend within each frame ───────────────────────
        # Reshape to (B*T, N, D) so each frame is an independent sequence
        xs = x.reshape(B * T, N, D)
        xs = xs + self.attn_s(self.norm_s(xs))
        x  = xs.reshape(B, T * N, D)

        # ── Temporal attention: attend across frames per patch ────────────────
        # Reshape to (B*N, T, D) so each patch location is an independent sequence
        xt = x.reshape(B, T, N, D)
        xt = xt.permute(0, 2, 1, 3)    # (B, N, T, D)
        xt = xt.reshape(B * N, T, D)
        xt = xt + self.attn_t(self.norm_t(xt))
        xt = xt.reshape(B, N, T, D)
        xt = xt.permute(0, 2, 1, 3)    # (B, T, N, D)
        x  = xt.reshape(B, T * N, D)

        # ── MLP ──────────────────────────────────────────────────────────────
        x = x + self.mlp(self.norm_mlp(x))

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full encoder
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsEncoder(nn.Module):
    """
    ViT-Small style encoder for spatiotemporal physical fields.

    Encoder only processes CONTEXT tokens (unmasked).
    The EMA copy of this encoder processes ALL tokens → target embeddings.

    Architecture:
      PatchEmbed → SinCos3DPosEmbed → N × FactoredSpaceTimeBlock → LayerNorm

    At inference/evaluation, we mean-pool all tokens → (B, D) embedding
    used for linear probing and kNN regression.
    """

    def __init__(
        self,
        in_channels: int = 11,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_frames: int = 16,
        spatial_size: int = 224,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.patch_size  = patch_size
        self.num_frames  = num_frames
        num_patches_h    = spatial_size // patch_size    # 14
        num_patches_w    = spatial_size // patch_size    # 14
        self.num_spatial = num_patches_h * num_patches_w # 196

        # Patch embedding
        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)

        # 3D positional embedding (fixed sinusoidal)
        self.pos_embed = SinCos3DPositionalEmbedding(
            embed_dim, num_frames, num_patches_h, num_patches_w
        )

        # Transformer blocks with factored space-time attention
        self.blocks = nn.ModuleList([
            FactoredSpaceTimeBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                num_frames=num_frames,
                num_spatial=self.num_spatial,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()
        self._print_param_count()

    def _init_weights(self):
        """Initialize weights following ViT conventions."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"[PhysicsEncoder] Parameters: {total:,} ({total/1e6:.2f}M)")

    def forward(
        self,
        x: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            (B, C, T, H, W) — input fields
            context_mask: (B, T*N) BoolTensor — True = token is context (keep)
                          If None, use all tokens (target encoder mode)

        Returns:
            tokens: (B, M, D) where M = number of kept tokens
                    M = T*N if context_mask is None (target encoder)
                    M = sum(context_mask[0]) if context_mask is given
        """
        B = x.shape[0]

        # Patchify and embed
        tokens = self.patch_embed(x)       # (B, T*N, D)
        tokens = self.pos_embed(tokens)    # (B, T*N, D)

        # Apply context masking: keep only visible tokens for context encoder
        if context_mask is not None:
            # context_mask: (B, T*N) — assume same mask for all items
            # We process each item; for efficiency assume same mask in batch
            # (if not, we'd need padding — keep it simple with same mask)
            # Use the mask from first item (same for all in batch during JEPA)
            mask_1d = context_mask[0]            # (T*N,)
            tokens  = tokens[:, mask_1d, :]      # (B, M_context, D)

            # Adjust positional embedding dimensions for factored attention
            # Since we've removed tokens, factored attention won't align to T/N anymore
            # → use full attention on context tokens (M_context is small enough)
            tokens = self._full_attention_forward(tokens)
        else:
            # Target encoder: process all T*N tokens with factored attention
            tokens = self._factored_attention_forward(tokens, B)

        tokens = self.norm(tokens)
        return tokens

    def _factored_attention_forward(self, tokens: torch.Tensor, B: int) -> torch.Tensor:
        """Full factored space-time attention for complete token sequences."""
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def _full_attention_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Fall back to full self-attention when tokens don't align to T×N grid
        (i.e., after masking breaks the regular structure).
        Uses each block's spatial attention path only (treats all as one sequence).
        """
        B, M, D = tokens.shape
        for block in self.blocks:
            # Use spatial attention on entire sequence (treat M patches as "one frame")
            tokens = tokens + block.attn_s(block.norm_s(tokens))
            tokens = tokens + block.mlp(block.norm_mlp(tokens))
        return tokens

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get a single (B, D) embedding for linear probing / kNN.
        Processes all tokens (no masking), then mean-pools.

        Args:
            x: (B, C, T, H, W)
        Returns:
            emb: (B, D)
        """
        tokens = self.forward(x, context_mask=None)   # (B, T*N, D)
        emb    = tokens.mean(dim=1)                    # (B, D)
        return emb