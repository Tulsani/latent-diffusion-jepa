"""
data/dataset.py
---------------
PyTorch Dataset for the active_matter subset of The Well.

The dataset is stored as HDF5 files on disk after downloading via:
    huggingface-cli download polymathic-ai/active_matter --local-dir <data_root>

Each HDF5 file contains one simulation trajectory:
    - 'fields': float32 array of shape (T, H, W, C) = (81, 256, 256, 11)
      (raw) or after preprocessing (16, 224, 224, 11)
    - 'alpha': scalar float — active dipole strength
    - 'zeta':  scalar float — steric alignment

We load 16-frame windows of 224x224 and return them as (C, T, H, W) tensors
to match the (channels, time, height, width) convention used by video models.

Labels alpha and zeta are returned for evaluation only — never used in training.
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Optional, Tuple, Dict


class ActiveMatterDataset(Dataset):
    """
    Dataset for active_matter physical simulations.

    Returns:
        frames: Tensor of shape (C, T, H, W) — normalized physical fields
        labels: dict with keys 'alpha' and 'zeta' (raw float values)
        meta:   dict with filename and parameter info
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",                   # "train", "val", "test"
        num_frames: int = 16,
        spatial_size: int = 224,
        num_channels: int = 11,
        channel_indices: Optional[list] = None,  # None = use all channels
        normalize: bool = True,
        augment: bool = True,                    # spatial augmentations for train
        precompute_stats: bool = False,          # if True, compute channel stats on init
        stats_cache: Optional[str] = None,       # path to cached mean/std .npz
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        self.num_channels = num_channels
        self.normalize = normalize
        self.augment = augment and (split == "train")

        # Channel selection
        self.channel_indices = channel_indices if channel_indices is not None \
            else list(range(11))
        self.out_channels = len(self.channel_indices)

        # Find all HDF5 files for this split
        split_dir = os.path.join(data_root, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.h5")))
        if len(self.files) == 0:
            # Try flat structure
            self.files = sorted(glob.glob(os.path.join(data_root, f"{split}_*.h5")))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No HDF5 files found in {data_root}/{split}. "
                f"Expected structure: {data_root}/train/*.h5"
            )

        print(f"[ActiveMatterDataset] {split}: found {len(self.files)} trajectories")

        # Load or compute per-channel normalization statistics
        self.channel_mean = None
        self.channel_std = None
        if normalize:
            self._load_or_compute_stats(stats_cache, precompute_stats)

    def _load_or_compute_stats(self, stats_cache: Optional[str], precompute: bool):
        """Load cached channel stats or compute from a subset of training data."""
        if stats_cache and os.path.exists(stats_cache):
            data = np.load(stats_cache)
            self.channel_mean = torch.tensor(data["mean"], dtype=torch.float32)
            self.channel_std  = torch.tensor(data["std"],  dtype=torch.float32)
            print(f"[ActiveMatterDataset] Loaded channel stats from {stats_cache}")
            return

        if not precompute:
            # Use fixed reasonable defaults — will be overridden by proper stats
            # These are placeholders; replace with actual computed stats
            self.channel_mean = torch.zeros(self.out_channels)
            self.channel_std  = torch.ones(self.out_channels)
            print("[ActiveMatterDataset] WARNING: Using zero-mean/unit-std normalization. "
                  "Run scripts/compute_stats.py for proper per-channel statistics.")
            return

        # Compute stats from first 500 training samples
        print("[ActiveMatterDataset] Computing channel statistics (subset of training data)...")
        running_sum  = np.zeros(11, dtype=np.float64)
        running_sq   = np.zeros(11, dtype=np.float64)
        running_count = 0

        sample_files = self.files[:min(500, len(self.files))]
        for f in sample_files:
            with h5py.File(f, "r") as hf:
                fields = hf["fields"][:]  # (T, H, W, C)
                fields = fields.reshape(-1, 11).astype(np.float64)
                running_sum   += fields.sum(axis=0)
                running_sq    += (fields ** 2).sum(axis=0)
                running_count += fields.shape[0]

        mean = running_sum / running_count
        std  = np.sqrt(running_sq / running_count - mean ** 2)
        std  = np.clip(std, 1e-6, None)

        # Select channels
        mean = mean[self.channel_indices]
        std  = std[self.channel_indices]

        self.channel_mean = torch.tensor(mean, dtype=torch.float32)
        self.channel_std  = torch.tensor(std,  dtype=torch.float32)

        if stats_cache:
            np.savez(stats_cache, mean=mean, std=std)
            print(f"[ActiveMatterDataset] Saved channel stats to {stats_cache}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        filepath = self.files[idx]

        with h5py.File(filepath, "r") as hf:
            # fields: (T, H, W, C) — expect (16, 224, 224, 11) after preprocessing
            # If raw (81, 256, 256, 11), we subsample temporally and crop spatially
            fields = hf["fields"][:]  # numpy float32

            # Physical parameter labels — used ONLY for evaluation, never training
            alpha = float(hf["alpha"][()])
            zeta  = float(hf["zeta"][()])

        fields = self._preprocess(fields)  # → (C, T, H, W)

        if self.augment:
            fields = self._augment(fields)

        if self.normalize:
            # channel_mean/std shape: (C,) → broadcast over (C, T, H, W)
            mean = self.channel_mean[:, None, None, None]
            std  = self.channel_std[:, None, None, None]
            fields = (fields - mean) / std

        labels = {"alpha": torch.tensor(alpha, dtype=torch.float32),
                  "zeta":  torch.tensor(zeta,  dtype=torch.float32)}

        return fields, labels

    def _preprocess(self, fields: np.ndarray) -> torch.Tensor:
        """
        Convert raw fields array to (C, T, H, W) tensor.

        Handles two cases:
          - Already preprocessed: (16, 224, 224, 11)
          - Raw simulation:       (81, 256, 256, 11) → subsample + center crop
        """
        T, H, W, C = fields.shape

        # --- Temporal subsampling ---
        if T > self.num_frames:
            # Uniformly sample `num_frames` indices from T frames
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
            fields = fields[indices]  # (num_frames, H, W, C)

        # --- Spatial crop/resize ---
        if H != self.spatial_size or W != self.spatial_size:
            # Center crop to spatial_size x spatial_size
            h_start = (H - self.spatial_size) // 2
            w_start = (W - self.spatial_size) // 2
            fields = fields[
                :,
                h_start:h_start + self.spatial_size,
                w_start:w_start + self.spatial_size,
                :
            ]

        # --- Channel selection ---
        fields = fields[:, :, :, self.channel_indices]  # (T, H, W, out_C)

        # --- Convert to tensor and rearrange to (C, T, H, W) ---
        fields = torch.from_numpy(fields.astype(np.float32))
        fields = fields.permute(3, 0, 1, 2)  # (C, T, H, W)

        return fields

    def _augment(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Spatial augmentations consistent across time dimension.
        fields: (C, T, H, W)

        Active matter is statistically isotropic → random flips are valid.
        We do NOT do random crops (already 224x224) or color jitter (physical fields).
        """
        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            fields = torch.flip(fields, dims=[3])  # flip W

        # Random vertical flip
        if torch.rand(1).item() < 0.5:
            fields = torch.flip(fields, dims=[2])  # flip H

        # Random 90-degree rotation (preserves grid structure)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            fields = torch.rot90(fields, k=k, dims=[2, 3])

        return fields


def build_dataloaders(
    cfg,
    rank: int = 0,
    world_size: int = 1,
    stats_cache: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders with DistributedSampler for DDP training.

    Args:
        cfg: config object (from yaml)
        rank: process rank for DDP
        world_size: total number of processes
        stats_cache: path to cached channel statistics

    Returns:
        dict with keys "train", "val", "test"
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        is_train = (split == "train")

        dataset = ActiveMatterDataset(
            data_root=cfg.data.data_root,
            split=split,
            num_frames=cfg.data.num_frames,
            spatial_size=cfg.data.spatial_size,
            num_channels=cfg.data.num_channels,
            normalize=True,
            augment=is_train,
            precompute_stats=(split == "train"),
            stats_cache=stats_cache,
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,
            drop_last=is_train,
        ) if world_size > 1 else None

        loader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            sampler=sampler,
            shuffle=(is_train and sampler is None),
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=is_train,
            persistent_workers=(cfg.data.num_workers > 0),
        )

        loaders[split] = loader

    return loaders