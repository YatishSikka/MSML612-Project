"""Frame Accuracy (exact match): fraction of frames where ALL gt characters are predicted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from evaluation.compute_character_f1 import (
    PORORO_CHARACTERS, classify_frame, load_classifier,
)


def compute_frame_accuracy(gen_dir: str, gt_json: str, ckpt: str,
                           device: str = "cuda") -> float:
    model = load_classifier(ckpt, device=device)
    with open(gt_json) as f:
        gt_map = json.load(f)

    hits, total = 0, 0
    for story_id, frames in gt_map.items():
        for frame_idx, gt_chars in frames.items():
            path = Path(gen_dir) / f"{story_id}_{frame_idx}.png"
            if not path.exists():
                continue
            preds = classify_frame(model, Image.open(path), device)
            pred_chars = {PORORO_CHARACTERS[i] for i, v in enumerate(preds) if v}
            total += 1
            if set(gt_chars) == pred_chars:
                hits += 1
    return hits / max(total, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    print(f"Frame Accuracy: {compute_frame_accuracy(args.gen, args.gt, args.ckpt):.4f}")
