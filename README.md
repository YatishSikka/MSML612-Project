# Identity-Preserving Story Visualization via Diffusion Models with Decoupled Identity Conditioning

**MSML612 Final Project (Group-9)**
Yatish Sikka, Pratham Dabas, Boram Lee, Negin Nazemzadeh, Ritvik Chaturvedi

## Overview

Story visualization generates a sequence of images from text captions that form a coherent narrative. The key challenge is maintaining consistent character identity across all frames. We combine two complementary mechanisms:

1. **Consistent Self-Attention (CSA)** from StoryDiffusion (NeurIPS 2024) -- shares self-attention keys/values across frames for cross-frame consistency
2. **Identity Adapter** (our contribution) -- an IP-Adapter-style decoupled cross-attention module that injects character identity features from a reference image

We use Anything-v5 (anime-finetuned SD 1.5) as the base model and train only the lightweight identity adapter (~22M params, ~2.5% of the model) while keeping the 860M-parameter base model frozen.

## Key Results (PororoSV)

- **IPS (Identity Preservation Score) increases monotonically** with adapter scale (0.760 to 0.780), confirming the adapter injects character identity
- **Removing CSA drops CLIP similarity by 5% and DINOv2 by 15%**, proving CSA drives cross-frame consistency
- **The two components are independent** -- adapter scale does not affect cross-frame metrics, and CSA removal does not reduce IPS

## Project Structure

```
project/
├── configs/
│   ├── train_config.yaml          # Training configuration
│   └── eval_config.yaml           # Evaluation configuration
├── data/
│   └── preprocessing/
│       └── story_dataset.py       # PororoSV dataset loader
├── models/
│   ├── identity_encoder.py        # CLIP/DINOv2/InsightFace encoders
│   ├── identity_injection.py      # Decoupled cross-attention module
│   ├── consistent_self_attention.py  # CSA implementation
│   └── story_pipeline.py          # End-to-end generation pipeline
├── training/
│   └── train_identity_adapter.py  # Adapter training loop
├── evaluation/
│   ├── compute_fid.py             # FID metric
│   ├── compute_character_f1.py    # Character F1 (VLCStoryGan classifier)
│   ├── compute_frame_accuracy.py  # Frame Accuracy
│   ├── compute_clip_similarity.py # Cross-frame CLIP similarity
│   ├── compute_dino_similarity.py # Cross-frame DINOv2 similarity
│   ├── compute_ips.py             # Identity Preservation Score
│   └── run_all_metrics.py         # Run all metrics
├── inference/
│   └── generate_stories.py        # Story generation script
├── scripts/
│   ├── generate_gt_json.py        # GT character labels + real frames
│   └── run_phase4.sh              # Full evaluation orchestration
├── final_report.tex               # Final report
└── ppt.md                         # Presentation outline
```

## Setup

### Requirements
- Python 3.10
- PyTorch with CUDA support
- Key packages: diffusers, transformers, accelerate, peft, pytorch-fid, tqdm

### Environment
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate peft pytorch-fid tqdm pillow pyyaml
```

### Pre-download Models
Models must be cached before running on compute nodes without internet:
```bash
python -c "
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor, AutoModel
StableDiffusionPipeline.from_pretrained('stablediffusionapi/anything-v5')
CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
AutoModel.from_pretrained('facebook/dinov2-large')
"
```

## Training

```bash
python training/train_identity_adapter.py \
    --config configs/train_config.yaml \
    --precision fp16
```

Training configuration: AdamW (lr=1e-5), cosine schedule, 30K steps, batch size 8, fp16. Only the identity adapter is trained; the base model, VAE, and encoders remain frozen.

## Inference

```bash
python inference/generate_stories.py \
    --config configs/eval_config.yaml \
    --out outputs/eval/generated \
    --ckpt outputs/checkpoints/step_30000.pt \
    --skip_lora \
    --identity_scales 0.5 \
    --num_stories 100
```

Flags:
- `--identity_scales`: adapter strength (0.0 = no adapter, 1.0 = max)
- `--skip_lora`: skip LoRA weight merge (use when no LoRA was trained)
- `--num_stories`: number of test stories to generate
- `--jpeg`: save as JPEG instead of PNG (saves disk space)

## Evaluation

```bash
python evaluation/run_all_metrics.py \
    --config configs/eval_config.yaml \
    --gen_dir outputs/eval/generated \
    --real_dir data/pororo/real_test \
    --gt_json data/pororo/test_characters.json \
    --out outputs/eval/results.json
```

## References

1. Zhou et al., "StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation," NeurIPS 2024
2. Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models," arXiv 2023
3. Pan et al., "Synthesizing Coherent Story with Auto-Regressive Latent Diffusion Models," WACV 2024
4. Maharana & Bansal, "Integrating Visuospatial, Linguistic and Commonsense Structure into Story Visualization," EMNLP 2021
