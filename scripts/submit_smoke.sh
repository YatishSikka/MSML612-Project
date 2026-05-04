#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4

set -euo pipefail

SCRATCH="/home/batman/scratch.msml612pcs3"

module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cuda/12.3.0/gcc

source "${SCRATCH}/venv/bin/activate"

export HF_HOME="${SCRATCH}/hf_cache"
export TORCH_HOME="${SCRATCH}/torch_cache"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python scripts/smoke_test.py --config configs/train_config.yaml --precision fp16
