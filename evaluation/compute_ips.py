"""Identity Preservation Score (IPS).

For each story, computes cosine similarity between the CLIP embedding of the
reference image (ground-truth frame 0) and each generated frame, then averages
across frames and stories.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def load_clip(device: str = "cuda"):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor


@torch.no_grad()
def clip_embed(model, processor, images: list[Image.Image], device: str) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    return F.normalize(feats, dim=-1)


def compute_ips(gen_dir: str, data_dir: str, split: str = "test",
                image_resolution: int = 512, frames_per_story: int = 5,
                num_stories: int | None = None, device: str = "cuda") -> float:
    from data.preprocessing.story_dataset import PororoDataset

    ds = PororoDataset(data_dir=data_dir, split=split,
                       image_resolution=image_resolution,
                       frames_per_story=frames_per_story)

    model, processor = load_clip(device=device)
    gen_path = Path(gen_dir)
    n = num_stories or len(ds)
    scores = []

    for idx in tqdm(range(n), desc="IPS"):
        story = ds[idx]
        ref_img = story["ref_images_pil"]

        frame_imgs = []
        for i in range(frames_per_story):
            p = gen_path / f"{idx:06d}_{i}.png"
            if not p.exists():
                break
            frame_imgs.append(Image.open(p).convert("RGB"))

        if not frame_imgs:
            continue

        ref_feat = clip_embed(model, processor, [ref_img], device)        # [1, D]
        frame_feats = clip_embed(model, processor, frame_imgs, device)     # [F, D]
        sims = (frame_feats @ ref_feat.T).squeeze(-1)                      # [F]
        scores.append(sims.mean().item())

    return sum(scores) / max(len(scores), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--image_resolution", type=int, default=512)
    ap.add_argument("--frames_per_story", type=int, default=5)
    ap.add_argument("--num_stories", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    score = compute_ips(
        gen_dir=args.gen_dir,
        data_dir=args.data_dir,
        split=args.split,
        image_resolution=args.image_resolution,
        frames_per_story=args.frames_per_story,
        num_stories=args.num_stories,
        device=args.device,
    )
    print(f"IPS: {score:.4f}")


if __name__ == "__main__":
    main()
