#!/usr/bin/env python3
"""
scripts/compute_stats.py
------------------------
Compute per-channel mean and std over the ENTIRE training set.

MUST be run before the first training run. The output .npz file is
referenced by the dataset loader for z-score normalization of all 11
physical channels.

Why this matters:
  The 11 channels are physically heterogeneous:
    - Concentration: scalar field, O(1) values
    - Velocity (vx, vy): signed, O(0.1) values
    - Orientation tensor (Qxx, Qxy, Qyx, Qyy): [-0.5, 0.5] range
    - Strain-rate tensor (exx, exy, eyx, eyy): can be large near vortices
  Without per-channel normalization, the loss is dominated by channels
  with large magnitude, and the encoder ignores the rest.

Usage:
  # Run once before training — takes ~5-10 min on CPU, ~2 min with parallel workers
  python scripts/compute_stats.py \
      --data_root /scratch/$NETID/data/active_matter \
      --output /scratch/$NETID/checkpoints/channel_stats.npz \
      --num_workers 8

Output:
  channel_stats.npz with keys:
    - mean: (11,) float32 array
    - std:  (11,) float32 array
    - channel_names: list of 11 channel name strings
"""

import os
import sys
import glob
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import h5py


# ─────────────────────────────────────────────────────────────────────────────
# Channel names for interpretability
# ─────────────────────────────────────────────────────────────────────────────

CHANNEL_NAMES = [
    "concentration",           # 0:  scalar concentration field φ
    "velocity_x",              # 1:  fluid velocity vx
    "velocity_y",              # 2:  fluid velocity vy
    "orientation_Qxx",         # 3:  nematic order tensor Q_xx
    "orientation_Qxy",         # 4:  nematic order tensor Q_xy
    "orientation_Qyx",         # 5:  nematic order tensor Q_yx
    "orientation_Qyy",         # 6:  nematic order tensor Q_yy
    "strain_rate_exx",         # 7:  strain rate tensor e_xx
    "strain_rate_exy",         # 8:  strain rate tensor e_xy
    "strain_rate_eyx",         # 9:  strain rate tensor e_yx
    "strain_rate_eyy",         # 10: strain rate tensor e_yy
]


def process_file(filepath: str):
    """
    Worker function: compute sum and sum-of-squares for one HDF5 file.
    Returns (sum_C, sumsq_C, count) where C=11.
    """
    try:
        with h5py.File(filepath, "r") as hf:
            fields = hf["fields"][:]                    # (T, H, W, C)
            fields = fields.reshape(-1, 11).astype(np.float64)  # (T*H*W, 11)
        return (
            fields.sum(axis=0),       # (11,)
            (fields ** 2).sum(axis=0), # (11,)
            fields.shape[0],           # T*H*W count
        )
    except Exception as e:
        print(f"WARNING: Failed to process {filepath}: {e}")
        return None


def compute_stats(
    data_root: str,
    output_path: str,
    num_workers: int = 8,
    split: str = "train",       # only use training split
    max_files: int = None,      # None = use all files
) -> dict:
    """
    Compute per-channel statistics over all training HDF5 files.

    Uses parallel processing for speed.
    """
    # Find all training files
    split_dir = os.path.join(data_root, split)
    files = sorted(glob.glob(os.path.join(split_dir, "*.h5")))

    if len(files) == 0:
        # Try flat structure
        files = sorted(glob.glob(os.path.join(data_root, f"{split}_*.h5")))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No HDF5 files found in {data_root}/{split}.\n"
            f"Expected: {data_root}/train/*.h5\n"
            f"Check your data_root and that the dataset is downloaded."
        )

    if max_files:
        files = files[:max_files]

    print(f"Computing stats over {len(files)} {split} files...")
    print(f"Using {num_workers} parallel workers")

    running_sum   = np.zeros(11, dtype=np.float64)
    running_sumsq = np.zeros(11, dtype=np.float64)
    running_count = 0
    n_failed      = 0

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is None:
                n_failed += 1
                continue
            s, sq, n = result
            running_sum   += s
            running_sumsq += sq
            running_count += n

            if (i + 1) % 500 == 0 or (i + 1) == len(files):
                elapsed = time.time() - t0
                print(f"  [{i+1:5d}/{len(files)}] "
                      f"processed={i+1-n_failed} failed={n_failed} "
                      f"elapsed={elapsed:.1f}s")

    if running_count == 0:
        raise RuntimeError("No data successfully processed. Check your data_root.")

    # Compute mean and std
    mean = running_sum   / running_count
    var  = running_sumsq / running_count - mean ** 2
    std  = np.sqrt(np.clip(var, 1e-12, None))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — processed {running_count:,} pixels×frames×channels")
    print(f"Failed files: {n_failed}/{len(files)}")

    # ── Print stats table ─────────────────────────────────────────────────────
    print(f"\n{'Channel':<25} {'Mean':>10} {'Std':>10} {'Min std OK':>12}")
    print("-" * 60)
    for i, name in enumerate(CHANNEL_NAMES):
        ok = "✓" if std[i] > 0.01 else "⚠ LOW"
        print(f"  {name:<23} {mean[i]:>10.6f} {std[i]:>10.6f} {ok:>12}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        channel_names=CHANNEL_NAMES,
        num_files=len(files) - n_failed,
        num_pixels=running_count,
    )
    print(f"\nSaved to: {output_path}")

    return {"mean": mean, "std": std}


def verify_stats(stats_path: str):
    """Load and print a saved stats file for verification."""
    data = np.load(stats_path, allow_pickle=True)
    print(f"\nVerifying {stats_path}")
    print(f"  num_files:  {data.get('num_files', 'N/A')}")
    print(f"  num_pixels: {data.get('num_pixels', 'N/A')}")
    print(f"\n{'Channel':<25} {'Mean':>10} {'Std':>10}")
    print("-" * 48)
    names = data.get("channel_names", [f"ch{i}" for i in range(11)])
    for i, name in enumerate(names):
        print(f"  {str(name):<23} {data['mean'][i]:>10.6f} {data['std'][i]:>10.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-channel normalization statistics for active_matter dataset"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root directory containing train/ val/ test/ subdirectories"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for .npz stats file (e.g. /scratch/$NETID/checkpoints/channel_stats.npz)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of parallel workers for reading HDF5 files"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Which split to compute stats on (always use 'train')"
    )
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Limit to first N files (for quick testing — omit for full dataset)"
    )
    parser.add_argument(
        "--verify", type=str, default=None,
        help="Path to existing .npz to verify instead of computing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.verify:
        verify_stats(args.verify)
    else:
        # Expand env vars
        data_root   = os.path.expandvars(args.data_root)
        output_path = os.path.expandvars(args.output)

        print(f"data_root : {data_root}")
        print(f"output    : {output_path}")
        print(f"split     : {args.split}")
        print(f"workers   : {args.num_workers}")
        if args.max_files:
            print(f"max_files : {args.max_files} (TEST MODE)")
        print()

        compute_stats(
            data_root=data_root,
            output_path=output_path,
            num_workers=args.num_workers,
            split=args.split,
            max_files=args.max_files,
        )

        # Auto-verify the output
        verify_stats(output_path)