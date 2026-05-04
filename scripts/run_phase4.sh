#!/usr/bin/env bash
# Phase 4: inference + quantitative evaluation + ablation sweep.
# Run inside a tmux+srun session on Zaratan with the venv activated.
#
# Usage:
#   bash scripts/run_phase4.sh
#
# Prerequisites:
#   - outputs/checkpoints/step_30000.pt exists
#   - venv activated, PYTHONPATH set, HF_HOME set
#   - (optional) checkpoints/vlcstorygan_pororo_inception.pth for char F1/frame acc

set -euo pipefail
PROJ="/home/batman/msml612/project"
cd "$PROJ"

export PYTHONPATH="${PROJ}:${PYTHONPATH:-}"
export HF_HOME="/home/batman/scratch.msml612pcs3/hf_cache"
export TORCH_HOME="/home/batman/scratch.msml612pcs3/torch_cache"

# ── 1. Ground-truth JSON + real test frames for FID ─────────────────────────
echo "=== Step 1: generate GT json + real test frames ==="
python scripts/generate_gt_json.py \
    --data_dir /home/batman/scratch.msml612pcs3/data/pororo/raw/pororo_png \
    --split test \
    --frames_per_story 5 \
    --gt_json_out data/pororo/test_characters.json \
    --real_dir_out data/pororo/real_test \
    --image_resolution 512

# ── 2. Generate with trained model (identity_scale=0.5, CSA on) ─────────────
echo "=== Step 2: generate — trained model (scale=0.5) ==="
python inference/generate_stories.py \
    --config configs/eval_config.yaml \
    --out outputs/eval/gen_scale05 \
    --ckpt outputs/checkpoints/step_30000.pt

# ── 3. Baseline: no adapter (identity_scale=0.0) ────────────────────────────
echo "=== Step 3: generate — baseline (scale=0.0) ==="
python inference/generate_stories.py \
    --config configs/eval_config.yaml \
    --out outputs/eval/gen_scale00 \
    --ckpt outputs/checkpoints/step_30000.pt \
    --identity_scales 0.0

# ── 4. Run all metrics on the trained model ──────────────────────────────────
echo "=== Step 4: evaluate trained model ==="
python evaluation/run_all_metrics.py \
    --config configs/eval_config.yaml \
    --gen_dir outputs/eval/gen_scale05 \
    --real_dir data/pororo/real_test \
    --gt_json data/pororo/test_characters.json \
    --out outputs/eval/results_scale05.json

# ── 5. Run all metrics on the baseline ──────────────────────────────────────
echo "=== Step 5: evaluate baseline ==="
python evaluation/run_all_metrics.py \
    --config configs/eval_config.yaml \
    --gen_dir outputs/eval/gen_scale00 \
    --real_dir data/pororo/real_test \
    --gt_json data/pororo/test_characters.json \
    --out outputs/eval/results_scale00.json

# ── 6. Identity scale ablation (0.3, 0.5, 0.7, 1.0 — 0.0 already done) ─────
echo "=== Step 6: identity scale ablation ==="
for scale in 0.3 0.7 1.0; do
    tag="scale$(echo $scale | tr -d '.')"
    echo "  scale=$scale -> outputs/eval/gen_${tag}"
    python inference/generate_stories.py \
        --config configs/eval_config.yaml \
        --out "outputs/eval/gen_${tag}" \
        --ckpt outputs/checkpoints/step_30000.pt \
        --identity_scales "$scale"
    python evaluation/run_all_metrics.py \
        --config configs/eval_config.yaml \
        --gen_dir "outputs/eval/gen_${tag}" \
        --real_dir data/pororo/real_test \
        --gt_json data/pororo/test_characters.json \
        --out "outputs/eval/results_${tag}.json"
done

# ── 7. CSA ablation (CSA off — vanilla SD1.5 + identity adapter) ─────────────
echo "=== Step 7: CSA ablation (CSA disabled) ==="
# Temporarily patch config to disable CSA, then restore
python - <<'PYEOF'
import yaml, copy
with open("configs/eval_config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("storydiffusion", {})["use_consistent_self_attn"] = False
with open("configs/eval_config_nocsa.yaml", "w") as f:
    yaml.dump(cfg, f)
PYEOF
python inference/generate_stories.py \
    --config configs/eval_config_nocsa.yaml \
    --out outputs/eval/gen_nocsa \
    --ckpt outputs/checkpoints/step_30000.pt
python evaluation/run_all_metrics.py \
    --config configs/eval_config.yaml \
    --gen_dir outputs/eval/gen_nocsa \
    --real_dir data/pororo/real_test \
    --gt_json data/pororo/test_characters.json \
    --out outputs/eval/results_nocsa.json
rm -f configs/eval_config_nocsa.yaml

echo ""
echo "=== Phase 4 complete ==="
echo "Results:"
for f in outputs/eval/results_*.json; do
    echo "  $f"
    python -c "import json,sys; d=json.load(open('$f')); [print(f'    {k}: {v:.4f}') for k,v in d.items() if isinstance(v,(int,float))]"
done
