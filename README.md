# Identity-Preserving Story Visualization

MSML612 course project. Augments StoryDiffusion (NeurIPS 2024) with an IP-Adapter-style decoupled cross-attention module for character identity conditioning.

See `CLAUDE_CODE_CONTEXT.md` for full design and `PROJECT_SUMMARY.md` for a short summary.

## Quickstart (Zaratan)

```bash
bash scripts/setup_environment.sh
bash scripts/download_models.sh
bash scripts/download_datasets.sh
sbatch scripts/submit_zaratan.sh
```

## Layout
- `configs/` — YAML training and eval configs
- `data/` — PororoSV, FlintstonesSV, preprocessing scripts
- `models/` — identity encoder, injection module, pipeline, baselines
- `training/` — training loops and losses
- `evaluation/` — metrics (FID, Char F1, Frame Acc, CLIP/DINOv2 similarity)
- `inference/` — generation and visualization
- `scripts/` — env setup, downloads, SLURM
- `notebooks/` — demos and analysis
