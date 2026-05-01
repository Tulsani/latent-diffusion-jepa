#!/bin/bash
#SBATCH --job-name=ldjjepa_temporal
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/logs/ldjjepa_temporal_%j.out
#SBATCH --error=/scratch/%u/logs/ldjjepa_temporal_%j.err
#SBATCH --requeue
# ^^^ CRITICAL: --requeue automatically resubmits this job if the spot
# instance is preempted. The training script saves a checkpoint on SIGUSR1
# and resumes from latest.pt on restart.

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

export NETID=$USER
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1

# Create output directories
mkdir -p /scratch/$NETID/logs
mkdir -p /scratch/$NETID/checkpoints/ldjjepa_temporal

# ─────────────────────────────────────────────────────────────────────────────
# Singularity + Conda environment
# Update OVERLAY_PATH and SIF_PATH to match your setup
# ─────────────────────────────────────────────────────────────────────────────
OVERLAY_PATH=/scratch/$NETID/overlay-ldjjepa.ext3
SIF_PATH=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu20.04.sif

singularity exec \
    --nv \
    --overlay $OVERLAY_PATH:ro \
    $SIF_PATH \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate ldjjepa
        echo 'Python: $(which python)'
        echo 'PyTorch: $(python -c \"import torch; print(torch.__version__)\")'
        echo 'CUDA available: $(python -c \"import torch; print(torch.cuda.is_available())\")'
        echo 'GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)'

        cd /scratch/$NETID/ld_jepa

        python scripts/train.py \
            --config configs/temporal_mask.yaml \
            2>&1 | tee /scratch/$NETID/logs/ldjjepa_temporal_${SLURM_JOB_ID}.log
    "

echo "End: $(date)"
echo "Exit code: $?"