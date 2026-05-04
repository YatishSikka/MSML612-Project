#!/bin/bash
# Download SD 1.5, IP-Adapter weights, and eval checkpoints.
# All cached to scratch via HF_HOME / TORCH_HOME (set in .bashrc).
set -euo pipefail

SCRATCH="/home/batman/scratch.msml612pcs3"

module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cuda/12.3.0/gcc
source "${SCRATCH}/venv/bin/activate"

CKPT_DIR="${SCRATCH}/checkpoints"
mkdir -p "${CKPT_DIR}"

python - <<'PY'
from diffusers import StableDiffusionPipeline
import torch
StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None,
)
print("SD 1.5 cached.")
PY

python - <<'PY'
from huggingface_hub import hf_hub_download
for fname in ["ip-adapter_sd15.bin", "ip-adapter-plus_sd15.bin",
              "ip-adapter-plus-face_sd15.bin"]:
    try:
        p = hf_hub_download("h94/IP-Adapter", f"models/{fname}")
        print("Cached:", p)
    except Exception as e:
        print(f"Skipped {fname}: {e}")
PY

python - <<'PY'
from transformers import CLIPVisionModelWithProjection
CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
print("CLIP ViT-H cached.")
PY

python - <<'PY'
import torch
torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
print("DINOv2 cached.")
PY

echo "TODO: download VLCStoryGan's fine-tuned Inception-v3 into ${CKPT_DIR}/vlcstorygan_pororo_inception.pth"
echo "  See third_party/VLCStoryGan README for the link."
