"""Cross-frame DINOv2 similarity. Same shape as CLIP version, different backbone."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from evaluation.compute_clip_similarity import collect_whole_frame_crops, pairwise_consistency


def load_dino(variant: str = "facebook/dinov2-large", device: str = "cuda"):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(variant).eval().to(device)
    tfm = T.Compose([
        T.Resize(224), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tfm


@torch.no_grad()
def embed_crops(model, tfm, crops: List[Image.Image], device: str):
    batch = torch.stack([tfm(c.convert("RGB")) for c in crops]).to(device)
    outputs = model(pixel_values=batch)
    feats = outputs.last_hidden_state[:, 0]  # CLS token
    return F.normalize(feats, dim=-1)


def compute_cross_frame_dino(story_crops: Dict[str, List[Image.Image]],
                             device: str = "cuda") -> float:
    model, tfm = load_dino(device=device)
    scores = []
    for _, crops in story_crops.items():
        if len(crops) < 2:
            continue
        feats = embed_crops(model, tfm, crops, device)
        scores.append(pairwise_consistency(feats))
    return sum(scores) / max(len(scores), 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--frames_per_story", type=int, default=5)
    args = ap.parse_args()
    crops = collect_whole_frame_crops(args.gen, args.frames_per_story)
    print(f"Cross-frame DINOv2 similarity: {compute_cross_frame_dino(crops):.4f}")
