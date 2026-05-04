#!/bin/bash
#SBATCH --job-name=story_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail

SCRATCH="/home/batman/scratch.msml612pcs3"

module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cuda/12.3.0/gcc

source "${SCRATCH}/venv/bin/activate"

export HF_HOME="${SCRATCH}/hf_cache"
export TORCH_HOME="${SCRATCH}/torch_cache"
export PYTHONPATH="${PWD}:${PWD}/third_party/StoryDiffusion:${PYTHONPATH:-}"

CKPT="${CKPT:-outputs/checkpoints/latest.pt}"
OUT="${OUT:-outputs/eval/run_${SLURM_JOB_ID}}"

python inference/generate_stories.py \
    --config configs/eval_config.yaml \
    --ckpt "${CKPT}" \
    --out "${OUT}/gen"

python evaluation/run_all_metrics.py \
    --config configs/eval_config.yaml \
    --gen_dir "${OUT}/gen" \
    --real_dir data/pororo/real_test \
    --gt_json data/pororo/test_characters.json \
    --out "${OUT}/results.json"
