#!/bin/bash
# Zaratan env setup — uses module python + venv (no conda available).
set -euo pipefail

SCRATCH="/home/batman/scratch.msml612pcs3"
VENV_DIR="${SCRATCH}/venv"

module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cuda/12.3.0/gcc

python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

# Clone third-party repos into scratch.
mkdir -p "${SCRATCH}/third_party"
ln -sfn "${SCRATCH}/third_party" third_party

[ -d third_party/StoryDiffusion ] || \
  git clone https://github.com/HVision-NKU/StoryDiffusion third_party/StoryDiffusion
[ -d third_party/IP-Adapter ] || \
  git clone https://github.com/tencent-ailab/IP-Adapter third_party/IP-Adapter
[ -d third_party/ARLDM ] || \
  git clone https://github.com/xichenpan/ARLDM third_party/ARLDM
[ -d third_party/VLCStoryGan ] || \
  git clone https://github.com/adymaharana/VLCStoryGan third_party/VLCStoryGan

# Data + outputs + checkpoints on scratch too.
mkdir -p "${SCRATCH}/data/pororo/raw" "${SCRATCH}/data/flintstones/raw"
mkdir -p "${SCRATCH}/outputs/checkpoints" "${SCRATCH}/checkpoints" "${SCRATCH}/logs"
ln -sfn "${SCRATCH}/data" data
ln -sfn "${SCRATCH}/outputs" outputs
ln -sfn "${SCRATCH}/checkpoints" checkpoints
ln -sfn "${SCRATCH}/logs" logs

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
echo "All large files go to: ${SCRATCH}"
