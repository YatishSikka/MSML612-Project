#!/bin/bash
#SBATCH --job-name=story_viz
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail

SCRATCH="/home/batman/scratch.msml612pcs3"

module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cuda/12.3.0/gcc

source "${SCRATCH}/venv/bin/activate"

export HF_HOME="${SCRATCH}/hf_cache"
export TORCH_HOME="${SCRATCH}/torch_cache"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}:${PWD}/third_party/StoryDiffusion:${PYTHONPATH:-}"

python training/train_identity_adapter.py \
    --config configs/train_config.yaml \
    --precision fp16 \
    --batch_size 2 \
    --gradient_checkpointing \
    --lora_rank 8
