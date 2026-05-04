#!/bin/bash
# PororoSV + FlintstonesSV download helpers.
set -euo pipefail

SCRATCH="/home/batman/scratch.msml612pcs3"
DATA_DIR="${SCRATCH}/data"
mkdir -p "${DATA_DIR}/pororo/raw" "${DATA_DIR}/flintstones/raw"

module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
source "${SCRATCH}/venv/bin/activate"

if ! command -v gdown >/dev/null; then
  pip install gdown
fi

echo "==> PororoSV"
echo "See: https://github.com/xichenpan/ARLDM#data-preparation"
echo "     https://github.com/adymaharana/storydalle"
echo "Expected files under ${DATA_DIR}/pororo/raw/"

# Example (IDs change — verify upstream):
# gdown --id <PORORO_GDRIVE_ID> -O "${DATA_DIR}/pororo/raw/pororo.zip"
# unzip -q "${DATA_DIR}/pororo/raw/pororo.zip" -d "${DATA_DIR}/pororo/raw/"

echo "==> FlintstonesSV"
echo "See AR-LDM / StoryDALL-E repos for current links."

echo "Once raw data is in place, convert to HDF5:"
echo "  python data/preprocessing/prepare_pororo.py \\"
echo "      --data_dir ${DATA_DIR}/pororo/raw --save_path ${DATA_DIR}/pororo/pororo.h5"
