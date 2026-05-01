#!/usr/bin/env python3
"""
scripts/train.py
----------------
Entry point for LD-JEPA training.

Single GPU:
  python scripts/train.py --config configs/temporal_mask.yaml

Multi GPU (torchrun):
  torchrun --nproc_per_node=2 scripts/train.py --config configs/spatiotemporal_mask.yaml

Slurm (see slurm/ directory for batch scripts):
  sbatch slurm/train_a100.sh
"""

import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.trainer import run_worker, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="LD-JEPA Training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank (set automatically by torchrun)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Detect DDP environment (torchrun sets these env vars)
    if "LOCAL_RANK" in os.environ:
        # torchrun / Slurm + srun mode
        rank       = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        run_worker(rank, world_size, args.config)

    elif "SLURM_PROCID" in os.environ:
        # Direct Slurm srun mode (alternative)
        rank       = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank))

        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"]       = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Set master address for multi-node (single-node: localhost)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        run_worker(rank, world_size, args.config)

    else:
        # Single GPU fallback
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

        cfg = load_config(args.config)
        print(f"[train.py] Single GPU mode | config: {args.config}")

        # Init process group for rank 0 / world_size 1
        import torch.distributed as dist
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=1,
            rank=0,
        )
        torch.cuda.set_device(0)

        from training.trainer import Trainer
        import torch
        torch.manual_seed(cfg.experiment.seed)
        torch.cuda.manual_seed(cfg.experiment.seed)

        trainer = Trainer(cfg, rank=0, world_size=1)
        trainer.train()

        dist.destroy_process_group()


if __name__ == "__main__":
    main()