"""
data/masking.py
---------------
Masking strategies for the LD-JEPA pretext task.

Two strategies:
  1. TemporalMaskSampler    — context = first K frames, target = last T-K frames
  2. SpatiotemporalMaskSampler — VideoMAE-style tube masking across space+time

Both return boolean masks over the (T, num_patches_h, num_patches_w) token grid,
where True = masked (target), False = visible (context).

Token grid for 224x224, patch_size=16:
  num_patches_h = num_patches_w = 224 // 16 = 14
  total spatial patches = 196
  total tokens = 16 frames × 196 = 3,136
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class MaskSampler:
    """Base class for mask samplers."""

    def __init__(self, num_frames: int, patch_size: int, spatial_size: int):
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.num_patches_h = spatial_size // patch_size   # 14
        self.num_patches_w = spatial_size // patch_size   # 14
        self.num_spatial   = self.num_patches_h * self.num_patches_w  # 196
        self.total_tokens  = num_frames * self.num_spatial             # 3136

    def sample(self, batch_size: int, device: torch.device):
        """
        Returns:
            context_mask: BoolTensor (B, T*num_spatial) — True = keep as context
            target_mask:  BoolTensor (B, T*num_spatial) — True = predict (target)
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# 1. Temporal masking
# ─────────────────────────────────────────────────────────────────────────────

class TemporalMaskSampler(MaskSampler):
    """
    Context = all patches in first `num_context_frames` frames.
    Target  = all patches in remaining `num_target_frames` frames.

    This is the "causal forecasting" setup:
      - The encoder sees the first half of the trajectory
      - The diffusion predictor predicts the second half's latents
      - Mirrors physical forecasting (predict future from past)
    """

    def __init__(
        self,
        num_frames: int,
        patch_size: int,
        spatial_size: int,
        num_context_frames: int = 8,    # how many frames to use as context
    ):
        super().__init__(num_frames, patch_size, spatial_size)
        self.num_context_frames = num_context_frames
        self.num_target_frames  = num_frames - num_context_frames

        assert 1 <= num_context_frames < num_frames, \
            f"num_context_frames must be in [1, {num_frames-1}], got {num_context_frames}"

        # Precompute the masks (same for every sample in the batch, broadcast)
        # Token layout: [t0_patch0, t0_patch1, ..., t0_patch195,
        #                t1_patch0, ..., t15_patch195]
        # i.e. tokens are ordered frame-major
        context_mask = torch.zeros(num_frames * self.num_spatial, dtype=torch.bool)
        target_mask  = torch.zeros(num_frames * self.num_spatial, dtype=torch.bool)

        for t in range(num_frames):
            start = t * self.num_spatial
            end   = start + self.num_spatial
            if t < num_context_frames:
                context_mask[start:end] = True
            else:
                target_mask[start:end] = True

        # Store as buffers — expand to batch dim in sample()
        self._context = context_mask  # (T*N,)
        self._target  = target_mask   # (T*N,)

    def sample(self, batch_size: int, device: torch.device):
        context = self._context.unsqueeze(0).expand(batch_size, -1).to(device)
        target  = self._target.unsqueeze(0).expand(batch_size, -1).to(device)
        return context, target

    def get_context_frame_indices(self) -> List[int]:
        return list(range(self.num_context_frames))

    def get_target_frame_indices(self) -> List[int]:
        return list(range(self.num_context_frames, self.num_frames))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spatiotemporal tube masking (VideoMAE / JEPA-style)
# ─────────────────────────────────────────────────────────────────────────────

class SpatiotemporalMaskSampler(MaskSampler):
    """
    Spatiotemporal tube masking:
      - A "tube" is the same spatial patch location masked across
        `tube_length` consecutive frames.
      - Tubes are randomly sampled until `mask_ratio` of total tokens are masked.
      - Prevents trivial "copy from neighbor frame" shortcuts.

    Context = unmasked tokens (can optionally apply light masking here too).
    Target  = masked tokens.

    Additionally supports JEPA-style block targets: instead of random tubes,
    sample `num_target_blocks` contiguous spatiotemporal blocks.
    """

    def __init__(
        self,
        num_frames: int,
        patch_size: int,
        spatial_size: int,
        mask_ratio: float = 0.75,
        tube_length: int = 2,
        num_target_blocks: int = 4,           # JEPA multi-block targets
        target_scale: Tuple[float, float] = (0.15, 0.2),   # fraction of tokens
        aspect_ratio: Tuple[float, float] = (0.75, 1.5),   # spatial aspect ratio
        context_mask_ratio: float = 0.15,     # optionally mask some context tokens too
        use_block_targets: bool = True,        # True=JEPA blocks, False=random tubes
    ):
        super().__init__(num_frames, patch_size, spatial_size)
        self.mask_ratio         = mask_ratio
        self.tube_length        = tube_length
        self.num_target_blocks  = num_target_blocks
        self.target_scale       = target_scale
        self.aspect_ratio       = aspect_ratio
        self.context_mask_ratio = context_mask_ratio
        self.use_block_targets  = use_block_targets

    def _sample_block(
        self,
        occupied: np.ndarray,   # (T, H_p, W_p) bool — already-targeted tokens
    ) -> Optional[np.ndarray]:
        """
        Sample one spatiotemporal block target.
        Returns a bool mask (T, H_p, W_p) or None if sampling fails.
        """
        H, W = self.num_patches_h, self.num_patches_w

        for _ in range(100):   # max attempts
            # Sample block size
            scale = np.random.uniform(*self.target_scale)
            aspect = np.random.uniform(*self.aspect_ratio)

            # Spatial extents
            num_spatial_target = int(scale * H * W)
            block_h = int(round(np.sqrt(num_spatial_target * aspect)))
            block_w = int(round(np.sqrt(num_spatial_target / aspect)))
            block_h = max(1, min(block_h, H))
            block_w = max(1, min(block_w, W))

            # Temporal extent: sample tube_length consecutive frames
            t_len = self.tube_length
            if self.num_frames - t_len < 0:
                t_len = self.num_frames
            t_start = np.random.randint(0, max(1, self.num_frames - t_len + 1))

            # Spatial anchor
            h_start = np.random.randint(0, max(1, H - block_h + 1))
            w_start = np.random.randint(0, max(1, W - block_w + 1))

            # Build block mask
            block = np.zeros((self.num_frames, H, W), dtype=bool)
            block[t_start:t_start + t_len,
                  h_start:h_start + block_h,
                  w_start:w_start + block_w] = True

            # Accept if no overlap with already-occupied
            if not (block & occupied).any():
                return block

        return None  # failed to place non-overlapping block

    def sample(self, batch_size: int, device: torch.device):
        """
        Sample masks independently for each item in the batch.

        Returns:
            context_mask: (B, T*N) BoolTensor — True = visible context
            target_mask:  (B, T*N) BoolTensor — True = target to predict
        """
        H, W = self.num_patches_h, self.num_patches_w
        N    = self.num_spatial
        T    = self.num_frames

        all_context = []
        all_target  = []

        for _ in range(batch_size):
            # ── Sample target tokens ──────────────────────────────────────────
            if self.use_block_targets:
                # JEPA-style: sample contiguous spatiotemporal blocks
                target_3d = np.zeros((T, H, W), dtype=bool)
                for _ in range(self.num_target_blocks):
                    block = self._sample_block(occupied=target_3d)
                    if block is not None:
                        target_3d |= block

                # Ensure mask_ratio approximately met; if too few, randomly add tubes
                current_ratio = target_3d.mean()
                if current_ratio < self.mask_ratio * 0.5:
                    # Fallback: randomly add individual tubes
                    target_3d = self._add_random_tubes(target_3d)

            else:
                # Pure random tube masking
                target_3d = np.zeros((T, H, W), dtype=bool)
                target_3d = self._add_random_tubes(target_3d)

            # ── Context = complement of target ────────────────────────────────
            context_3d = ~target_3d

            # Optionally lightly mask some context tokens (harder task)
            if self.context_mask_ratio > 0:
                context_positions = np.argwhere(context_3d)
                n_context = len(context_positions)
                n_extra_mask = int(n_context * self.context_mask_ratio)
                if n_extra_mask > 0:
                    drop_idx = np.random.choice(n_context, n_extra_mask, replace=False)
                    for di in drop_idx:
                        t, h, w = context_positions[di]
                        context_3d[t, h, w] = False

            # ── Flatten to (T*N,) ─────────────────────────────────────────────
            # Order: token[t, h, w] = t * (H*W) + h * W + w
            target_flat  = torch.from_numpy(target_3d.reshape(T * N))
            context_flat = torch.from_numpy(context_3d.reshape(T * N))

            all_target.append(target_flat)
            all_context.append(context_flat)

        target_mask  = torch.stack(all_target,  dim=0).to(device)   # (B, T*N)
        context_mask = torch.stack(all_context, dim=0).to(device)   # (B, T*N)

        return context_mask, target_mask

    def _add_random_tubes(self, target_3d: np.ndarray) -> np.ndarray:
        """
        Randomly add spatiotemporal tubes until mask_ratio is reached.
        A tube = one spatial patch location × tube_length consecutive frames.
        """
        H, W = self.num_patches_h, self.num_patches_w
        T    = self.num_frames
        total = T * H * W
        target_count = int(total * self.mask_ratio)

        spatial_positions = [(h, w) for h in range(H) for w in range(W)]
        np.random.shuffle(spatial_positions)

        for h, w in spatial_positions:
            if target_3d.sum() >= target_count:
                break
            t_starts = list(range(0, T, self.tube_length))
            np.random.shuffle(t_starts)
            for t_start in t_starts:
                t_end = min(t_start + self.tube_length, T)
                target_3d[t_start:t_end, h, w] = True
                if target_3d.sum() >= target_count:
                    break

        return target_3d


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_mask_sampler(cfg) -> MaskSampler:
    """Build the appropriate mask sampler from config."""
    strategy = cfg.masking.strategy

    if strategy == "temporal":
        return TemporalMaskSampler(
            num_frames=cfg.data.num_frames,
            patch_size=cfg.model.patch_size,
            spatial_size=cfg.data.spatial_size,
            num_context_frames=cfg.masking.num_context_frames,
        )
    elif strategy == "spatiotemporal":
        return SpatiotemporalMaskSampler(
            num_frames=cfg.data.num_frames,
            patch_size=cfg.model.patch_size,
            spatial_size=cfg.data.spatial_size,
            mask_ratio=cfg.masking.mask_ratio,
            tube_length=cfg.masking.tube_length,
            num_target_blocks=cfg.masking.num_target_blocks,
            target_scale=cfg.masking.target_block_scale,
            aspect_ratio=cfg.masking.aspect_ratio,
            context_mask_ratio=cfg.masking.context_mask_ratio,
            use_block_targets=True,
        )
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}. "
                         f"Choose from ['temporal', 'spatiotemporal']")