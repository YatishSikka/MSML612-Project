"""Cross-frame CLIP similarity for identity consistency.

For each story:
  1. Crop character region(s) from each generated frame (via classifier or detector)
  2. Extract CLIP image embeddings per crop
  3. Average pairwise cosine similarity across frames = per-character consistency
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image


def load_clip(model_name: str = "openai/clip-vit-large-patch14",
              device: str = "cuda"):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


@torch.no_grad()
def embed_crops(model, processor, crops: List[Image.Image], device: str):
    inputs = processor(images=crops, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    return F.normalize(feats, dim=-1)


def pairwise_consistency(feats: torch.Tensor) -> float:
    if feats.shape[0] < 2:
        return float("nan")
    sim = feats @ feats.T
    n = feats.shape[0]
    mask = 1.0 - torch.eye(n, device=feats.device)
    return ((sim * mask).sum() / (n * (n - 1))).item()


def compute_cross_frame_clip(story_crops: Dict[str, List[Image.Image]],
                             device: str = "cuda") -> float:
    """story_crops: {story_id: [crop_frame_0, crop_frame_1, ...]} for one character."""
    model, processor = load_clip(device=device)
    scores = []
    for story_id, crops in story_crops.items():
        if len(crops) < 2:
            continue
        feats = embed_crops(model, processor, crops, device)
        scores.append(pairwise_consistency(feats))
    return sum(scores) / max(len(scores), 1)


def collect_whole_frame_crops(gen_dir: str, frames_per_story: int = 5):
    """Fallback: use whole frames when no character detector is available."""
    out: Dict[str, List[Image.Image]] = {}
    for p in sorted(Path(gen_dir).glob("*.png")):
        story_id = "_".join(p.stem.split("_")[:-1])
        out.setdefault(story_id, []).append(Image.open(p))
    return {k: v for k, v in out.items() if len(v) == frames_per_story}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--frames_per_story", type=int, default=5)
    args = ap.parse_args()
    crops = collect_whole_frame_crops(args.gen, args.frames_per_story)
    print(f"Cross-frame CLIP similarity: {compute_cross_frame_clip(crops):.4f}")
