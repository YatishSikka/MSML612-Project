# Project Context: Identity-Preserving Story Visualization

## Overview

Build a story visualization system that generates sequences of images from text captions where characters maintain consistent visual identity across all frames. We augment StoryDiffusion (NeurIPS 2024) with an identity conditioning module based on IP-Adapter's decoupled cross-attention to inject character appearance features into the diffusion process.

---

## Architecture Summary

```
Input: [Caption_1, Caption_2, ..., Caption_5] + [Reference_Image_per_character]
                          |
                          v
              ┌─────────────────────┐
              │  Text Encoder (CLIP)│
              └────────┬────────────┘
                       │ text embeddings
                       v
              ┌─────────────────────┐
              │   U-Net (SD 1.5)    │
              │                     │
              │  ┌───────────────┐  │
              │  │ Text Cross-Attn│  │  ← existing (frozen)
              │  └───────────────┘  │
              │  ┌───────────────┐  │
              │  │ Consistent    │  │  ← from StoryDiffusion (shared self-attn across frames)
              │  │ Self-Attention│  │
              │  └───────────────┘  │
              │  ┌───────────────┐  │
              │  │ Identity      │  │  ← NEW: our addition (decoupled cross-attn for identity)
              │  │ Cross-Attn    │  │
              │  └───────────────┘  │
              └────────┬────────────┘
                       │
                       v
              ┌─────────────────────┐
              │  VAE Decoder        │
              └────────┬────────────┘
                       │
                       v
              [Generated_Image_1, ..., Generated_Image_5]
```

### Identity Encoder (what extracts character features)

We experiment with three encoders. Priority order for implementation:

1. **IP-Adapter CLIP encoder** — most mature, easiest to integrate, pretrained weights available
2. **DINOv2** — stronger semantic features, may work better for cartoon characters
3. **InsightFace** — face-specific, likely less useful for cartoon datasets but worth testing

### Identity Injection (how features enter the U-Net)

- Add a **parallel cross-attention layer** at each cross-attention block in the U-Net
- This layer attends to identity features SEPARATELY from text features (decoupled design from IP-Adapter)
- Scale parameter `identity_scale` (default 0.5) controls strength of identity conditioning
- For multi-character: use spatial masks so different image regions attend to different character embeddings

### Consistent Self-Attention (from StoryDiffusion, keep as-is)

- Shares self-attention keys/values across all frames in a batch
- This is StoryDiffusion's core mechanism — we keep it and ADD identity conditioning on top
- Requires minimum 3 text prompts, recommended 5-6

---

## Code Repositories to Use

### StoryDiffusion (PRIMARY — our base)
- **Repo:** https://github.com/HVision-NKU/StoryDiffusion
- **Branch:** main
- **Key files:**
  - `Comic_Generation.ipynb` — main inference notebook
  - The Consistent Self-Attention module (look for where self-attention K/V are shared across batch)
- **Setup:**
  ```bash
  conda create --name storydiffusion python=3.10
  conda activate storydiffusion
  pip install -r requirements.txt
  ```
- **GPU requirement:** tested on 24GB (A10), has low-memory version for ~20GB
- **Compatible with:** SD 1.5 and SDXL backbones

### AR-LDM (SECONDARY — baseline for comparison)
- **Repo:** https://github.com/xichenpan/ARLDM
- **Key files:**
  - `data_script/pororo_hdf5.py` — PororoSV preprocessing
  - `data_script/flintstones_hdf5.py` — FlintstonesSV preprocessing
  - Config in `config.yaml`
- **Setup:**
  ```bash
  conda create -n arldm python=3.8
  conda activate arldm
  conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
  pip install -r requirements.txt
  ```
- **Data conversion:** Both datasets should be converted to HDF5 for faster I/O:
  ```bash
  python data_script/pororo_hdf5.py --data_dir /path/to/pororo --save_path /path/to/hdf5
  python data_script/flintstones_hdf5.py --data_dir /path/to/flintstones --save_path /path/to/hdf5
  ```

### IP-Adapter (IDENTITY MODULE — what we adapt)
- **Repo:** https://github.com/tencent-ailab/IP-Adapter
- **Models on HuggingFace:** https://huggingface.co/h94/IP-Adapter
- **Key design to replicate:**
  - Decoupled cross-attention: separate cross-attn layers for text and image features
  - Image encoder: CLIP ViT-H/14 (or ViT-bigG, but H is sufficient and smaller)
  - Adapter weights: ~22M parameters
  - Works with SD 1.5 (`ip-adapter_sd15.bin`) and SDXL (`ip-adapter_sdxl.bin`)
- **Integration with diffusers:**
  ```python
  from diffusers import StableDiffusionPipeline
  pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
  pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
  pipeline.set_ip_adapter_scale(0.5)  # identity_scale
  ```
- **For face-specific variant:** `ip-adapter-face_sd15.bin` or `ip-adapter-plus-face_sd15.bin`

### Evaluation Code
- **Repo:** https://github.com/adymaharana/VLCStoryGan
- Contains the standard evaluation scripts for:
  - FID computation
  - Character F1 score (uses fine-tuned Inception-v3 on PororoSV characters)
  - Frame Accuracy
- The fine-tuned Inception-v3 classifier checkpoint is included in this repo

### StoryBench (additional evaluation)
- **Repo:** https://github.com/google/storybench
- NeurIPS 2023 benchmark with more comprehensive metrics
- Good for FID computation code

---

## Datasets

### PororoSV (PRIMARY)
- **Download:** Links in AR-LDM repo README and StoryDALL-E repo (https://github.com/adymaharana/storydalle)
- **Size:** ~10K training stories, ~2.3K val, ~2.2K test
- **Format:** 5 consecutive frames per story, each with a text caption
- **Characters:** 9 recurring cartoon characters (Pororo, Crong, Loopy, Poby, Eddy, Petty, Harry, Tongtong, Rody)
- **Resolution:** 64x64 in original StoryGAN, but modern methods resize to 256x256 or 512x512
- **Preprocessing:** Convert to HDF5 using AR-LDM's scripts for faster loading

### FlintstonesSV (SECONDARY)
- **Download:** Links in AR-LDM and StoryDALL-E repos
- **Format:** Same structure as PororoSV but with Flintstones characters
- **Use:** Cross-dataset generalization evaluation

---

## Compute Constraints (IMPORTANT)

We are running on **UMD Zaratan HPC with NVIDIA V100 GPUs**.

### V100 Constraints:
- **VRAM:** 16GB or 32GB depending on variant
- **NO bf16 support** — must use fp16 everywhere
- **Must explicitly set** `torch_dtype=torch.float16` in all model loading
- **Training configs must use** fp16 mixed precision, NOT bf16

### Memory Strategy:
- **Use SD 1.5** backbone (NOT SDXL) — fits in 16-32GB with fp16
- **Use LoRA** for any fine-tuning (rank 4-16, reduces trainable params by 99%+)
- **Use gradient checkpointing** during training
- **IP-Adapter adds ~22M params** — negligible overhead
- **Batch size:** likely 1-2 for full training, 4-8 with LoRA
- For inference: `pipeline.enable_model_cpu_offload()` if memory is tight

### Example LoRA config:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.05,
)
```

---

## Evaluation Metrics

### Standard metrics (use VLCStoryGan evaluation code):

1. **FID (Fréchet Inception Distance)**
   - Lower is better
   - Measures image quality/realism
   - Computed using Inception-v3 features between generated and real images

2. **Character F1 Score**
   - Higher is better
   - Fine-tuned Inception-v3 classifier predicts which characters appear in each frame
   - F1 between predicted characters and ground-truth characters from caption
   - Classifier checkpoint is in the VLCStoryGan repo

3. **Frame Accuracy (Exact Match)**
   - Higher is better
   - Percentage of frames where ALL characters from the caption are correctly present
   - Stricter than Character F1

### Our proposed metrics (implement from scratch):

4. **Cross-Frame CLIP Similarity**
   ```python
   # Pseudocode
   from transformers import CLIPModel, CLIPProcessor
   
   model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
   
   # For each character in a story:
   #   1. Crop the character region from each generated frame
   #   2. Extract CLIP image embeddings for each crop
   #   3. Compute pairwise cosine similarity across frames
   #   4. Average = cross-frame consistency score for that character
   ```

5. **Cross-Frame DINOv2 Similarity**
   ```python
   # Same as above but using DINOv2 features
   import torch
   model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
   ```

### Ablation dimensions to evaluate:
- **Identity encoder type:** IP-Adapter CLIP vs DINOv2 vs InsightFace
- **Identity scale:** 0.0 (baseline), 0.3, 0.5, 0.7, 1.0
- **Sequence length:** 3, 5, 7, 10 frames — measure consistency degradation
- **Character count:** 1 vs 2 vs 3+ characters per scene
- **With/without Consistent Self-Attention:** isolate contribution of each component

---

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── configs/
│   ├── train_config.yaml
│   └── eval_config.yaml
├── data/
│   ├── pororo/                    # PororoSV dataset
│   ├── flintstones/               # FlintstonesSV dataset
│   └── preprocessing/
│       ├── prepare_pororo.py      # Download + HDF5 conversion
│       └── prepare_flintstones.py
├── models/
│   ├── identity_encoder.py        # Identity feature extraction (CLIP/DINOv2/InsightFace)
│   ├── identity_injection.py      # Decoupled cross-attention module
│   ├── story_pipeline.py          # Modified StoryDiffusion pipeline with identity conditioning
│   └── baselines/
│       ├── storydiffusion.py      # Vanilla StoryDiffusion wrapper
│       └── arldm.py               # AR-LDM wrapper
├── training/
│   ├── train_identity_adapter.py  # Train the identity cross-attention adapter
│   ├── train_lora.py              # LoRA fine-tuning script
│   └── losses.py                  # Training losses (diffusion loss + identity consistency loss)
├── evaluation/
│   ├── compute_fid.py
│   ├── compute_character_f1.py    # Uses fine-tuned Inception-v3
│   ├── compute_frame_accuracy.py
│   ├── compute_clip_similarity.py # Cross-frame CLIP consistency
│   ├── compute_dino_similarity.py # Cross-frame DINOv2 consistency
│   ├── run_all_metrics.py         # Run all metrics and produce results table
│   └── ablation_runner.py         # Automated ablation experiments
├── inference/
│   ├── generate_stories.py        # Generate story image sequences
│   └── visualize_results.py       # Side-by-side comparison plots
├── scripts/
│   ├── setup_environment.sh       # Conda env setup
│   ├── download_models.sh         # Download SD 1.5, IP-Adapter weights, eval checkpoints
│   ├── download_datasets.sh       # Download PororoSV + FlintstonesSV
│   └── submit_zaratan.sh          # SLURM job submission script for Zaratan
└── notebooks/
    ├── 01_baseline_demo.ipynb     # StoryDiffusion baseline demo
    ├── 02_identity_module_demo.ipynb  # Identity conditioning demo
    └── 03_results_analysis.ipynb  # Results visualization and analysis
```

---

## Implementation Progress

### Phase 1: Environment + Baseline — DONE
- ~~Set up environment~~ — venv on Zaratan (no conda), Python 3.10, pip install
- ~~Download PororoSV dataset~~ — PNGs + numpy index arrays, no HDF5 conversion needed
- ~~Baseline generation~~ — vanilla SD 1.5 generates photorealistic images (no Pororo knowledge)

### Phase 2: Identity Module — DONE
- ~~`identity_encoder.py`~~ — CLIP ViT-H/14, DINOv2 ViT-L, InsightFace ArcFace
- ~~`identity_injection.py`~~ — DecoupledCrossAttnProcessor on attn2 blocks
- ~~`consistent_self_attention.py`~~ — reimplemented without global variables
- ~~`story_pipeline.py`~~ — end-to-end pipeline with `use_consistent_self_attn` flag
- ~~Smoke test~~ — validated shapes, gradients, VRAM (20.3 GB for 3 frames at 512x512)

### Phase 3: Training — DONE
- ~~Training loop~~ — `train_identity_adapter.py` with Accelerate, cosine LR, checkpointing
- ~~First training run~~ — 26,000 steps completed with lr=1e-4; **adapter diverged** (stripe artifacts at identity_scale >= 0.3, adapter inert at step 2k/scale 0.1). Root cause: lr too high + no domain adaptation.
- ~~Wire in LoRA~~ — rank 8 on `[to_k, to_q, to_v, to_out.0]`, applied via peft `get_peft_model` before re-enabling adapter params. LoRA state saved under `"unet_lora"` key in checkpoints.
- ~~`--resume` flag~~ — implemented 2026-04-29. Loads adapters/identity_proj/unet_lora; restores optimizer+scheduler if present.
- ~~Warning flood fix~~ — `_IdentityKwargFilter` wrapper in `identity_injection.py`.
- ~~File logging~~ — `setup_logger()` in training script writes to `logs/train_<run|resume>_<timestamp>.log`.
- ~~**Second training run COMPLETE**~~ — finished step 30,000 on 2026-04-30 01:53. Final checkpoint: `outputs/checkpoints/step_30000.pt`. LR reached 0, no divergence. Speed: ~3.45 s/step.
- TODO: Encoder ablation (DINOv2, InsightFace) — in Phase 4 ablation sweep
- TODO: Identity scale ablation — in Phase 4 ablation sweep

### Phase 4: Full Evaluation — IN PROGRESS
- ~~`story_pipeline.py` CSA guard~~ — `generate()` now guards `self.csa_state is not None` so CSA ablation (use_consistent_self_attn=False) doesn't crash.
- ~~`story_pipeline.load_checkpoint()`~~ — loads adapters + identity_proj + merges LoRA via `_apply_lora()` (direct weight merge, no peft at inference). Formula: `W += (alpha/r) * B @ A` per layer, using peft key format `base_model.model.<path>.lora_A.default.weight`.
- ~~`inference/generate_stories.py`~~ — fixed to call `pipeline.load_checkpoint()` (previously skipped `unet_lora`); passes `use_consistent_self_attn` from config.
- ~~`configs/eval_config.yaml`~~ — checkpoint updated to `step_30000.pt`.
- ~~`scripts/generate_gt_json.py`~~ — extracts GT character labels from PororoSV test captions (regex on 9 character names); copies/resizes GT frames to `data/pororo/real_test/` for FID. Outputs `data/pororo/test_characters.json`.
- ~~`scripts/run_phase4.sh`~~ — full orchestration script (kept for reference; steps run manually instead).
- ~~Step 1~~ — GT json + real_test frames done: `data/pororo/test_characters.json` (2208 stories), `data/pororo/real_test/` (11040 files).
- ~~Step 2~~ — `gen_scale05` DONE: 2208 stories, scale=0.5, CSA on. Output: `outputs/eval/gen_scale05/`.
- ~~Step 3~~ — Metrics on `gen_scale05` DONE. Results: `outputs/eval/results_scale05.json`
- ~~VLCStoryGan classifier~~ — downloaded to `checkpoints/vlcstorygan_pororo_inception.pth` (raw OrderedDict, keys like `Conv2d_1a_3x3.conv.weight`). `load_classifier()` handles it correctly.
- ~~`evaluation/compute_dino_similarity.py`~~ — switched from `torch.hub` to HuggingFace `facebook/dinov2-large` (avoids xformers/scipy/OpenBLAS issues on Zaratan). Uses `outputs.last_hidden_state[:, 0]` for CLS token.
- ~~`evaluation/compute_ips.py`~~ — NEW: Identity Preservation Score. CLIP cosine similarity between reference image (GT frame 0) and each generated frame. Averaged across frames and stories. Added to `run_all_metrics.py` and `eval_config.yaml`.
- **`--num_stories` flag is `--num_stories` (double dash)** — easy typo, caused gen to fail silently.
- ~~Step 4~~ — `gen_scale00` DONE (2208 stories). Metrics done: `outputs/eval/results_scale00.json`.
- ~~Step 5~~ — scale ablations (0.3, 0.7, 1.0) DONE. Results in `outputs/eval/results_scale{03,07,10}.json`.
- Step 6 IN PROGRESS — CSA ablation (`use_consistent_self_attn=False`) generating 500 stories on second GPU srun.
- IPS metric IN PROGRESS — running on all 5 gen dirs (scale00/03/05/07/10) on first GPU srun.
- TODO: IPS results → update table
- TODO: CSA ablation metrics → update table
- TODO: Encoder ablation (DINOv2, InsightFace)
- TODO: FlintstonesSV generalization

**Results so far (2026-05-03):**

| Condition | FID↓ | Char F1↑ | Frame Acc↑ | CLIP↑ | DINOv2↑ |
|-----------|------|----------|------------|-------|---------|
| scale=0.5 (main) | 310.23 | 0.2557 | 0.0375 | 0.9132 | 0.7232 |
| scale=0.0 (baseline) | 295.82 | 0.2572 | 0.0387 | 0.9106 | 0.7275 |
| scale=0.3 | 306.33 | 0.307 | 0.026 | 0.9120 | 0.7202 |
| scale=0.7 | 313.64 | 0.304 | 0.020 | 0.9142 | 0.7208 |
| scale=1.0 | 311.24 | 0.303 | 0.022 | 0.9136 | 0.7189 |
| CSA off | — | — | — | — | — |

Note: High FID is expected — SD 1.5 generates photorealistic images while PororoSV is cartoon. Key comparison is cross-frame CLIP/DINOv2 consistency and IPS across ablations. IPS column pending — running now. CSA ablation pending.

**Remaining eval commands:**

```bash
# After gen_scale00 finishes (~11040 files in outputs/eval/gen_scale00/):
python evaluation/run_all_metrics.py \
    --config configs/eval_config.yaml \
    --gen_dir outputs/eval/gen_scale00 \
    --real_dir data/pororo/real_test \
    --gt_json data/pororo/test_characters.json \
    --out outputs/eval/results_scale00.json

# Scale ablations (500 stories each, can run in parallel second srun):
for scale in 0.3 0.7 1.0; do
    tag="scale$(echo $scale | tr -d '.')"
    python inference/generate_stories.py \
        --config configs/eval_config.yaml \
        --out "outputs/eval/gen_${tag}" \
        --ckpt outputs/checkpoints/step_30000.pt \
        --identity_scales "$scale" \
        --num_stories 500
    python evaluation/run_all_metrics.py \
        --config configs/eval_config.yaml \
        --gen_dir "outputs/eval/gen_${tag}" \
        --real_dir data/pororo/real_test \
        --gt_json data/pororo/test_characters.json \
        --out "outputs/eval/results_${tag}.json"
done

# CSA ablation (CSA off, 500 stories):
python - <<'PYEOF'
import yaml
with open("configs/eval_config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("storydiffusion", {})["use_consistent_self_attn"] = False
with open("configs/eval_config_nocsa.yaml", "w") as f:
    yaml.dump(cfg, f)
PYEOF
python inference/generate_stories.py \
    --config configs/eval_config_nocsa.yaml \
    --out outputs/eval/gen_nocsa \
    --ckpt outputs/checkpoints/step_30000.pt \
    --num_stories 500
python evaluation/run_all_metrics.py \
    --config configs/eval_config.yaml \
    --gen_dir outputs/eval/gen_nocsa \
    --real_dir data/pororo/real_test \
    --gt_json data/pororo/test_characters.json \
    --out outputs/eval/results_nocsa.json
rm -f configs/eval_config_nocsa.yaml
```

---

## Key Implementation Details

### Modifying StoryDiffusion's Self-Attention

StoryDiffusion's Consistent Self-Attention works by sharing K/V across all frames in a batch during self-attention. The key code to find and understand:

```python
# In StoryDiffusion, during self-attention:
# Instead of each frame having its own K, V:
#   K_shared = concat([K_frame1, K_frame2, ..., K_frameN])
#   V_shared = concat([V_frame1, V_frame2, ..., V_frameN])
# Each frame's Q attends to ALL frames' K/V
```

We ADD our identity cross-attention AFTER this, not replacing it.

### Adding Identity Cross-Attention (IP-Adapter style)

```python
# Pseudocode for the identity injection module
class IdentityCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.to_k_identity = nn.Linear(identity_dim, dim)  # project identity features to K
        self.to_v_identity = nn.Linear(identity_dim, dim)  # project identity features to V
        # Q comes from the existing hidden states (shared with text cross-attn)
    
    def forward(self, hidden_states, identity_features, scale=0.5):
        # hidden_states: [B, seq_len, dim] from U-Net
        # identity_features: [B, num_tokens, identity_dim] from identity encoder
        q = self.to_q(hidden_states)  # reuse existing Q projection
        k = self.to_k_identity(identity_features)
        v = self.to_v_identity(identity_features)
        attn_output = scaled_dot_product_attention(q, k, v)
        return hidden_states + scale * attn_output  # residual connection with scale
```

### Character Cropping for Evaluation

For cross-frame similarity metrics, you need character crops. Options:
- Use the fine-tuned Inception-v3 classifier from VLCStoryGan to locate characters
- For PororoSV: characters are often centered and distinct enough for simple detection
- Alternative: use a pretrained object detector (YOLO or similar) fine-tuned on PororoSV characters

### Interactive Session on Zaratan

```bash
# Start tmux + get a GPU node
tmux new -s train
srun --partition=gpu --gres=gpu:1 --mem=32G --time=24:00:00 --cpus-per-task=4 --pty bash

# Setup environment (no module load needed — venv has Python 3.10)
SCRATCH="/home/batman/scratch.msml612pcs3"
source "${SCRATCH}/venv/bin/activate"
export HF_HOME="${SCRATCH}/hf_cache"
export TORCH_HOME="${SCRATCH}/torch_cache"
export PYTHONPATH="/home/batman/msml612/project:${PYTHONPATH:-}"
cd /home/batman/msml612/project

# Training is COMPLETE — step_30000.pt is the final checkpoint.
# Resume command kept for reference (not needed unless retraining):
# python training/train_identity_adapter.py \
#     --config configs/train_config.yaml \
#     --precision fp16 \
#     --resume outputs/checkpoints/step_N.pt

# Phase 4: full eval pipeline (run once, ~6–8h total on V100)
bash scripts/run_phase4.sh

# Or run individual pieces:

# Step 1: build GT json + copy real test frames for FID
python scripts/generate_gt_json.py \
    --data_dir /home/batman/scratch.msml612pcs3/data/pororo/raw/pororo_png \
    --gt_json_out data/pororo/test_characters.json \
    --real_dir_out data/pororo/real_test

# Step 2: generate with trained model
python inference/generate_stories.py \
    --config configs/eval_config.yaml \
    --out outputs/eval/gen_scale05 \
    --ckpt outputs/checkpoints/step_30000.pt

# Step 3: run all metrics
python evaluation/run_all_metrics.py \
    --config configs/eval_config.yaml \
    --gen_dir outputs/eval/gen_scale05 \
    --real_dir data/pororo/real_test \
    --gt_json data/pororo/test_characters.json \
    --out outputs/eval/results_scale05.json

# Download VLCStoryGan classifier (needed for char F1 + frame accuracy):
# gdown "https://drive.google.com/uc?id=1xK6JOgQn_INQ3mBrA338BC2KoeM0TagR" \
#     -O checkpoints/vlcstorygan_pororo_inception.pth
```

---

## Reference Papers

| Paper | Venue | Key Idea | Repo |
|-------|-------|----------|------|
| StoryDiffusion (Zhou et al.) | NeurIPS 2024 | Consistent Self-Attention across frames | github.com/HVision-NKU/StoryDiffusion |
| AR-LDM (Pan et al.) | WACV 2024 | Autoregressive LDM conditioned on history | github.com/xichenpan/ARLDM |
| Make-A-Story (Rahman et al.) | CVPR 2023 | Visual memory module for context | github.com/ubc-vision/Make-A-Story |
| IP-Adapter (Ye et al.) | arXiv 2023 | Decoupled cross-attn for image prompts | github.com/tencent-ailab/IP-Adapter |
| TemporalStory (Chen et al.) | arXiv 2024 | Spatial-Temporal attention, StoryFlow | — |
| StoryDALL-E (Maharana et al.) | ECCV 2022 | Pretrained T2I adaptation for stories | github.com/adymaharana/storydalle |
| InstantID (Wang et al.) | arXiv 2024 | Zero-shot identity preservation | github.com/instantX-research/InstantID |
| VLCStoryGan (Maharana & Bansal) | EMNLP 2021 | Evaluation code + metrics | github.com/adymaharana/VLCStoryGan |
| StoryBench (Bugliarello et al.) | NeurIPS 2023 | Comprehensive benchmark | github.com/google/storybench |

---

## Common Pitfalls to Avoid

1. **bf16 on V100:** Will silently produce garbage or NaN. Always use `torch.float16`.
2. **StoryDiffusion needs 3+ prompts:** The Consistent Self-Attention module requires at least 3 captions. Single-frame generation won't use it.
3. **IP-Adapter square images:** IP-Adapter's CLIP encoder center-crops inputs. For non-square character crops, resize to 224x224 first rather than letting it crop.
4. **PororoSV download links:** Google Drive links in the repos can go stale. Download early and keep a backup.
5. **Evaluation classifier:** The Inception-v3 classifier for Character F1 is specific to PororoSV's 9 characters. For FlintstonesSV, a separate classifier needs to be trained or obtained.
6. **Mixed precision training:** Trainable adapter params MUST be fp32 for Accelerate's grad scaler. Frozen UNet/VAE/text-encoder stay fp16. In `IdentityCrossAttention.forward()`, cast inputs to `self.to_k_identity.weight.dtype` (not query dtype) to avoid autocast mismatches.
7. **LoRA target modules:** For SD 1.5 U-Net, target the cross-attention layers: `["to_k", "to_q", "to_v", "to_out.0"]`. Don't apply LoRA to self-attention if using Consistent Self-Attention (could interfere).
8. **CSA is inference-only.** Consistent Self-Attention's two-pass write/read scheme causes shape mismatches during training forward passes. Use `use_consistent_self_attn=False` in StoryPipeline for training.
9. **Adapter learning rate:** lr=1e-4 causes divergence. Use lr=1e-5 (matches original IP-Adapter paper).
10. **DataLoader collation:** PororoSV dataset returns PIL images — need custom `story_collate` function (in `train_identity_adapter.py`).
11. **Module load on Zaratan:** `python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2` does NOT exist. Just activate the venv directly.
12. **Gradient checkpointing:** Has no effect with our custom attention processors — they bypass diffusers' checkpointing mechanism. Use 3 frames instead of 5 to fit in VRAM.
13. **peft LoRA + custom adapters ordering:** `get_peft_model` calls `mark_only_lora_as_trainable` which freezes ALL non-LoRA params, including our identity adapter params. Fix: apply LoRA first, then re-enable adapter params. The re-enable in `pipeline.trainable_parameters()` must come AFTER `get_peft_model`.
14. **LoRA checkpoint saving:** Save only LoRA delta weights, not the full U-Net state dict. Filter `{k: v for k, v in unet_sd.items() if "lora_" in k}`. Full state dict would be ~3.4GB per checkpoint.
15. **`identity_tokens` kwarg floods attn1 processors:** When `cross_attention_kwargs={"identity_tokens": ...}` is passed to the UNet, diffusers forwards it to ALL processors including self-attention (attn1) blocks. Fix: `_IdentityKwargFilter` wrapper in `identity_injection.py` wraps all non-attn2 processors at attach time. Already implemented — don't remove it.
16. **Zaratan srun walltime:** Interactive GPU sessions max out at 24h. At ~4.2s/step, a 30k-step run needs ~35h total across 2 sessions. Always resume with `--resume outputs/checkpoints/step_N.pt` using the highest-numbered checkpoint available.
17. **`--gradient_checkpointing` is a no-op:** Has no effect with custom attention processors (they bypass diffusers' checkpointing mechanism). Don't bother passing the flag.
18. **LoRA loading at inference — don't use peft:** At inference, apply LoRA delta weights directly via `_apply_lora()` in `story_pipeline.py`. Wrapping with `get_peft_model` then `merge_and_unload()` has edge cases with custom attn processors. The direct formula `W += (alpha/r) * B @ A` is safer. Peft key format: `base_model.model.<original_param_path>.lora_A.default.weight`.
19. **VLCStoryGan classifier checkpoint:** Download from StoryViz repo (Google Drive id `1xK6JOgQn_INQ3mBrA338BC2KoeM0TagR`), not VLCStoryGan repo. `load_classifier()` handles both raw state dict and `{"state_dict": ...}` wrapping. If keys look wrong, inspect with `torch.load(ckpt).keys()`.
20. **Cross-frame similarity uses whole frames:** No character detector available for PororoSV — `collect_whole_frame_crops()` uses full frames. Results measure overall frame consistency, not per-character identity. Note this limitation in the report.
21. **Zaratan compute nodes have no internet:** Compute nodes block all outbound connections. Login nodes DO have internet. Pre-download all models on the login node before starting the srun session: CLIP (`openai/clip-vit-large-patch14`), DINOv2 (`facebook/dinov2-large`), FID Inception (`pytorch_fid.inception.InceptionV3`), and VLCStoryGan classifier (`gdown`). Set `HF_HOME` and `TORCH_HOME` to scratch so the compute node finds the cache.
22. **DINOv2 via torch.hub fails on Zaratan:** `torch.hub.load("facebookresearch/dinov2", ...)` pulls in xformers → scipy → OpenBLAS, which hits the login node's thread limit (RLIMIT_NPROC 256). Use HuggingFace instead: `AutoModel.from_pretrained("facebook/dinov2-large")`. Output is `BaseModelOutputWithPooling` — use `.last_hidden_state[:, 0]` for the CLS token. Already fixed in `evaluation/compute_dino_similarity.py`.
23. **OpenBLAS thread warnings on login node:** Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` before running any scipy/numpy-heavy code on the login node to suppress thread init errors.
24. **generate_stories.py flag is `--identity_scales` (plural):** Not `--identity_scale`. Pass a single float value: `--identity_scales 0.0`.
