"""
models/diffusion.py
-------------------
Latent Diffusion Predictor — the core novel component of LD-JEPA.

Architecture:
  - DDPM-style denoising in the latent embedding space
  - The noisy latent z_t (target tokens corrupted with Gaussian noise at step t)
    is denoised conditioned on z_c (context encoder output)
  - Cross-attention conditioning: target tokens attend to context tokens
  - Cosine noise schedule (better than linear for small T)
  - DDIM sampling for fast inference (10 steps instead of 100)

Why diffusion in latent space (not pixel space)?
  - Latent space is low-dimensional → fast DDPM convergence
  - The denoising target is z_target from EMA encoder (smooth, well-behaved)
  - Multimodal future distributions are captured naturally (phase transition!)
  - At eval time, we only need the context encoder → no diffusion overhead

Training objective:
  L = E_{t,ε} [ ||ε - ε_θ(z_t, t, z_c)||^2 ]
  where z_t = √ᾱ_t * z_target + √(1-ᾱ_t) * ε, ε ~ N(0,I)

  This is equivalent to predicting the clean target (x0-prediction variant used
  for better gradient flow: predict z_target directly, then recover ε).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Noise schedule
# ─────────────────────────────────────────────────────────────────────────────

class CosineNoiseSchedule(nn.Module):
    """
    Cosine noise schedule from Nichol & Dhariwal 2021 (Improved DDPM).

    ᾱ_t = cos²(π/2 · (t/T + s) / (1 + s))  / cos²(π/2 · s / (1 + s))

    Advantages over linear schedule:
      - Very little noise added at early steps → preserves structure
      - Smooth decay → stable gradients throughout training
      - Better for latent-space diffusion where signals are already compressed
    """

    def __init__(self, num_steps: int = 100, s: float = 0.008):
        super().__init__()
        self.num_steps = num_steps

        # Precompute schedule tables
        t = torch.arange(num_steps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(
            ((t / num_steps) + s) / (1 + s) * math.pi * 0.5
        ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(max=0.999).float()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).float()
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).float()

        # Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8)
        ).clamp(min=1e-20).float()

        # Register all as buffers (moved to device with model)
        self.register_buffer("betas",                 betas)
        self.register_buffer("alphas_cumprod",        alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev",   alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod",   alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("posterior_variance",    posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                             posterior_variance.clamp(min=1e-20).log())
        self.register_buffer("posterior_mean_coef1",
                             betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod + 1e-8))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod + 1e-8))

    def q_sample(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: q(z_t | z_0) = N(√ᾱ_t·z_0, (1-ᾱ_t)·I)

        Args:
            z0:    (B, M, D) clean target latents
            t:     (B,) integer timesteps in [0, T-1]
            noise: (B, M, D) optional pre-sampled noise

        Returns:
            z_t:   (B, M, D) noisy latents
            noise: (B, M, D) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(z0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        z_t = sqrt_alpha * z0 + sqrt_one_minus * noise
        return z_t, noise

    def predict_z0_from_noise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Recover z0 prediction from predicted noise."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return (z_t - sqrt_one_minus * noise_pred) / (sqrt_alpha + 1e-8)

    @torch.no_grad()
    def ddim_sample(
        self,
        model_fn,            # callable: (z_t, t, z_c) → noise_pred
        z_c: torch.Tensor,   # context conditioning
        shape: tuple,        # shape of target latents
        num_steps: int = 10, # DDIM steps (<<T for fast inference)
        eta: float = 0.0,    # eta=0 = deterministic DDIM
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        DDIM sampling for fast inference.

        Uses a subset of `num_steps` uniformly-spaced timesteps from [0, T].
        With eta=0, sampling is fully deterministic (no stochasticity).

        Returns: z0 prediction of shape `shape`
        """
        if device is None:
            device = z_c.device

        # Sample timestep subsequence
        times = torch.linspace(self.num_steps - 1, 0, num_steps, dtype=torch.long)

        z = torch.randn(shape, device=device)

        for i, t_val in enumerate(times):
            t_batch = torch.full((shape[0],), t_val, dtype=torch.long, device=device)

            # Predict noise
            noise_pred = model_fn(z, t_batch, z_c)

            # DDIM update
            alpha_t      = self.alphas_cumprod[t_val]
            alpha_t_prev = self.alphas_cumprod[times[i + 1]] if i + 1 < len(times) \
                           else torch.tensor(1.0, device=device)

            z0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)
            z0_pred = z0_pred.clamp(-10, 10)

            sigma = eta * ((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)).sqrt()
            direction = (1 - alpha_t_prev - sigma ** 2).clamp(min=0).sqrt() * noise_pred

            noise_component = sigma * torch.randn_like(z) if eta > 0 else 0
            z = alpha_t_prev.sqrt() * z0_pred + direction + noise_component

        return z   # final z0 prediction


# ─────────────────────────────────────────────────────────────────────────────
# Denoising network (the ε_θ model)
# ─────────────────────────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding + 2-layer MLP → adaLN shift/scale or
    added to token sequence (we use the latter for simplicity).
    """

    def __init__(self, dim: int, max_steps: int = 1000):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_steps) * torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        self.register_buffer("freqs", freqs)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer timesteps → (B, D) embedding"""
        t = t.float()
        args  = t[:, None] * self.freqs[None]   # (B, half)
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)   # (B, D)
        return self.mlp(emb)


class CrossAttention(nn.Module):
    """
    Cross-attention: target tokens (query) attend to context tokens (key/value).
    Used to condition the denoising network on context embeddings.
    """

    def __init__(self, dim: int, num_heads: int = 6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q    = nn.Linear(dim, dim, bias=False)
        self.kv   = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, M_target, D)  — noisy target tokens (queries)
        context: (B, M_context, D) — context encoder output (keys/values)
        returns: (B, M_target, D)
        """
        B, M, D = x.shape
        H = self.num_heads

        q  = self.q(x).reshape(B, M, H, D // H).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, -1, 2, H, D // H).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True,
                                             enable_mem_efficient=True):
            out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(B, M, D)
        return self.proj(out)


class DenoisingBlock(nn.Module):
    """
    One block of the denoising transformer:
      1. Self-attention on noisy target tokens + timestep embedding
      2. Cross-attention to context tokens (conditioning)
      3. MLP
    """

    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        # Self-attention
        self.norm1   = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Cross-attention (condition on context)
        self.norm2   = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads)

        # MLP
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        # Timestep conditioning via additive shift (simple but effective)
        self.t_proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,       # (B, M_target, D) noisy target tokens
        t_emb: torch.Tensor,   # (B, D) timestep embedding
        context: torch.Tensor, # (B, M_context, D) context encoder output
    ) -> torch.Tensor:
        # Inject timestep embedding via additive shift
        x = x + self.t_proj(t_emb).unsqueeze(1)

        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, need_weights=False)
        x = residual + x

        # Cross-attention to context
        x = x + self.cross_attn(self.norm2(x), context)

        # MLP
        x = x + self.mlp(self.norm3(x))

        return x


class LatentDiffusionPredictor(nn.Module):
    """
    The core novel component: a DDPM denoising network in latent space.

    This replaces the standard JEPA MLP/Transformer predictor.
    Instead of directly predicting z_target, we learn to denoise
    corrupted target latents conditioned on context latents.

    At training:
      1. Get z_target from EMA encoder (target encoder output on target tokens)
      2. Sample random timestep t ~ Uniform[0, T-1]
      3. Corrupt: z_t = q_sample(z_target, t)
      4. Predict: ε_θ(z_t, t, z_context)
      5. Loss: MSE(ε_pred, ε_true)

    At inference for representation extraction (linear probe / kNN):
      - We use ONLY the frozen context encoder → no diffusion overhead
      - Diffusion predictor is NOT used during evaluation

    At inference for visualization/analysis:
      - Run DDIM sampling to generate predicted z_target
      - Decode with a simple linear → can visualize predicted fields
    """

    def __init__(
        self,
        embed_dim: int = 384,
        predictor_depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        diffusion_steps: int = 100,
        diffusion_schedule: str = "cosine",
        ddim_steps: int = 10,
    ):
        super().__init__()
        self.embed_dim      = embed_dim
        self.diffusion_steps = diffusion_steps
        self.ddim_steps     = ddim_steps

        # Noise schedule
        if diffusion_schedule == "cosine":
            self.schedule = CosineNoiseSchedule(num_steps=diffusion_steps)
        else:
            raise ValueError(f"Unknown schedule: {diffusion_schedule}")

        # Timestep embedding
        self.t_embed = TimestepEmbedding(embed_dim, max_steps=diffusion_steps)

        # Denoising transformer blocks
        self.blocks = nn.ModuleList([
            DenoisingBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(predictor_depth)
        ])

        self.norm_out = nn.LayerNorm(embed_dim)

        # Output projection: predict noise ε (same dim as input)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self._init_weights()
        self._print_param_count()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-init output projection → start from identity denoising
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"[LatentDiffusionPredictor] Parameters: {total:,} ({total/1e6:.2f}M)")

    def forward(
        self,
        z_t: torch.Tensor,       # (B, M_target, D) noisy target latents
        t: torch.Tensor,          # (B,) integer timesteps
        z_context: torch.Tensor,  # (B, M_context, D) context encoder output
    ) -> torch.Tensor:
        """
        Predict noise ε given noisy latents, timestep, and context.

        Returns: noise_pred (B, M_target, D)
        """
        # Timestep embedding
        t_emb = self.t_embed(t)   # (B, D)

        # Process through denoising blocks
        x = z_t
        for block in self.blocks:
            x = block(x, t_emb, z_context)

        x = self.norm_out(x)
        noise_pred = self.out_proj(x)   # (B, M_target, D)

        return noise_pred

    def compute_loss(
        self,
        z_context: torch.Tensor,  # (B, M_context, D) from context encoder
        z_target: torch.Tensor,   # (B, M_target, D) from EMA target encoder
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute DDPM training loss.

        Randomly samples timesteps and noise, corrupts target latents,
        predicts noise, and returns MSE loss.

        Returns:
            loss: scalar tensor
            metrics: dict of loggable values
        """
        B, M, D = z_target.shape
        device   = z_target.device

        # Sample random timesteps uniformly
        t = torch.randint(0, self.diffusion_steps, (B,), device=device)

        # Sample noise and corrupt target latents
        noise = torch.randn_like(z_target)
        z_t, _ = self.schedule.q_sample(z_target, t, noise)

        # Predict noise
        noise_pred = self.forward(z_t, t, z_context)

        # Simple MSE loss on noise prediction (standard DDPM)
        loss = F.mse_loss(noise_pred, noise)

        # Also compute x0-prediction MSE for monitoring (not used for backprop)
        with torch.no_grad():
            z0_pred = self.schedule.predict_z0_from_noise(z_t, t, noise_pred)
            z0_mse  = F.mse_loss(z0_pred, z_target)

        metrics = {
            "diffusion_loss":  loss.item(),
            "z0_pred_mse":     z0_mse.item(),
            "mean_t":          t.float().mean().item(),
        }

        return loss, metrics

    @torch.no_grad()
    def predict(
        self,
        z_context: torch.Tensor,   # (B, M_context, D)
        target_shape: tuple,        # (B, M_target, D)
        use_ddim: bool = True,
    ) -> torch.Tensor:
        """
        Generate predicted target latents via DDIM sampling.
        Used for visualization and analysis, NOT for linear probe evaluation.

        Returns: z_target_pred (B, M_target, D)
        """
        if use_ddim:
            def model_fn(z_t, t, z_c):
                return self.forward(z_t, t, z_c)

            return self.schedule.ddim_sample(
                model_fn=model_fn,
                z_c=z_context,
                shape=target_shape,
                num_steps=self.ddim_steps,
                eta=0.0,
                device=z_context.device,
            )
        else:
            # Full DDPM reverse process (slow, for analysis)
            z = torch.randn(target_shape, device=z_context.device)
            for t_val in reversed(range(self.diffusion_steps)):
                t_batch = torch.full((target_shape[0],), t_val,
                                     dtype=torch.long, device=z_context.device)
                noise_pred = self.forward(z, t_batch, z_context)

                alpha_t    = self.schedule.alphas_cumprod[t_val]
                alpha_prev = self.schedule.alphas_cumprod_prev[t_val]
                beta_t     = self.schedule.betas[t_val]

                z0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)
                z0_pred = z0_pred.clamp(-10, 10)

                mean = (
                    self.schedule.posterior_mean_coef1[t_val] * z0_pred
                    + self.schedule.posterior_mean_coef2[t_val] * z
                )
                if t_val > 0:
                    z = mean + self.schedule.posterior_variance[t_val].sqrt() \
                        * torch.randn_like(z)
                else:
                    z = mean

            return z