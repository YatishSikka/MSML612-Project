# Project Summary: Identity-Preserving Story Visualization

## Goal
Generate 5-frame image sequences from text captions where characters maintain consistent visual identity across all frames.

## Approach
Take **StoryDiffusion** (NeurIPS 2024) as the base — it already provides Consistent Self-Attention shared across frames — and bolt on an **IP-Adapter-style decoupled cross-attention** module that injects character identity features into SD 1.5's U-Net.

- **Base model:** Stable Diffusion 1.5 (not SDXL, due to V100 memory)
- **Cross-frame consistency:** StoryDiffusion's Consistent Self-Attention (kept as-is)
- **Identity conditioning:** New parallel cross-attention layer per cross-attn block, attending to identity features separately from text. Controlled by `identity_scale` (default 0.5).
- **Identity encoder (ablation):**
  1. IP-Adapter CLIP (priority — most mature, pretrained weights)
  2. DINOv2 (stronger semantic features for cartoons)
  3. InsightFace (face-specific, likely weak on cartoons)

## Compute Constraints
- **UMD Zaratan HPC, NVIDIA V100 (16/32GB)**
- **NO bf16 support** — must use `torch.float16` everywhere
- LoRA (rank 4-16) for any fine-tuning
- Gradient checkpointing required
- Batch size 1-2 full / 4-8 with LoRA

## Datasets
- **PororoSV (primary):** 9 cartoon characters, 5-frame stories (~10K train / 2.3K val / 2.2K test). Convert to HDF5 via AR-LDM scripts.
- **FlintstonesSV (secondary):** Cross-dataset generalization eval.

## Evaluation Metrics
**Standard (via VLCStoryGan code):**
- FID (image quality, lower = better)
- Character F1 (fine-tuned Inception-v3 classifier on PororoSV characters)
- Frame Accuracy (exact match — all caption characters present)

**Proposed (implement from scratch):**
- Cross-Frame CLIP Similarity (per-character crops, pairwise cosine across frames)
- Cross-Frame DINOv2 Similarity (same, with DINOv2 features)

## Ablation Dimensions
- Identity encoder type (CLIP / DINOv2 / InsightFace)
- Identity scale (0.0, 0.3, 0.5, 0.7, 1.0)
- Sequence length (3, 5, 7, 10 frames — measure consistency degradation)
- Character count (1 / 2 / 3+ per scene)
- With/without Consistent Self-Attention (isolate component contributions)

## Repos Involved
| Role | Repo |
|------|------|
| Base (primary) | github.com/HVision-NKU/StoryDiffusion |
| Identity module | github.com/tencent-ailab/IP-Adapter |
| Secondary baseline | github.com/xichenpan/ARLDM |
| Eval code + Inception-v3 classifier | github.com/adymaharana/VLCStoryGan |
| Extra eval | github.com/google/storybench |

## Implementation Phases
1. **Environment + Baseline:** Set up env, run StoryDiffusion on PororoSV, set up eval pipeline, baseline numbers, AR-LDM secondary.
2. **Identity Module (core contribution):** `identity_encoder.py`, `identity_injection.py`, modify U-Net forward, train identity adapter only with frozen U-Net, qualitative results.
3. **Training + Experiments:** Training loop, optional LoRA on U-Net cross-attn, optional identity consistency loss, encoder + scale ablations.
4. **Full Evaluation:** All metrics × all configs on PororoSV test, sequence length / multi-character ablations, FlintstonesSV generalization, failure cases, final tables.

## Key Pitfalls
- bf16 on V100 silently produces NaN/garbage — always fp16
- StoryDiffusion needs ≥3 prompts (Consistent Self-Attn requirement)
- IP-Adapter CLIP center-crops — resize character crops to 224×224 first
- Keep identity adapter projections in fp32 under `autocast` for stability, cast outputs to fp16
- Don't apply LoRA to self-attention — interferes with Consistent Self-Attn
- PororoSV Google Drive links go stale; download early and back up
- VLCStoryGan's Inception-v3 classifier is PororoSV-specific; need separate one for FlintstonesSV

## Reference
Full architecture, file layout, code snippets, and SLURM template: `CLAUDE_CODE_CONTEXT.md`
