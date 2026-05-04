"""Generate story image sequences from captions + reference images.

Reads stories from an HDF5 test split and writes PNGs named
`{story_id}_{frame_idx}.png` into `--out`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from data.preprocessing.story_dataset import PororoDataset
from models.story_pipeline import StoryInput, StoryPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--identity_encoders", default=None)   # override for ablations
    ap.add_argument("--identity_scales", type=float, default=None)
    ap.add_argument("--sequence_lengths", type=int, default=None)
    ap.add_argument("--num_stories", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.identity_encoders:
        cfg["model"]["identity_encoder"] = args.identity_encoders
    if args.identity_scales is not None:
        cfg["model"]["identity_scale"] = args.identity_scales

    frames_per_story = args.sequence_lengths or cfg["eval"]["frames_per_story"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if cfg["model"]["torch_dtype"] == "float16" else torch.float32

    use_csa = cfg.get("storydiffusion", {}).get("use_consistent_self_attn", True)
    pipeline = StoryPipeline(
        base_model="runwayml/stable-diffusion-v1-5",
        identity_encoder_name=cfg["model"]["identity_encoder"],
        identity_dim=1024, num_identity_tokens=4,
        identity_scale=cfg["model"]["identity_scale"],
        torch_dtype=dtype, device=device,
        use_consistent_self_attn=use_csa,
    )
    if args.ckpt:
        lora_alpha = cfg.get("lora", {}).get("alpha", 32.0)
        pipeline.load_checkpoint(args.ckpt, lora_alpha=lora_alpha)

    ds_cfg = cfg["eval"]
    data_dir = (cfg["data"]["pororo_png_dir"] if ds_cfg["dataset"] == "pororo"
                else cfg["data"]["flintstones_png_dir"])
    ds = PororoDataset(data_dir=data_dir, split=ds_cfg["split"],
                       image_resolution=ds_cfg["image_resolution"],
                       frames_per_story=frames_per_story)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    n = args.num_stories or len(ds)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])

    for idx in tqdm(range(n)):
        story = ds[idx]
        images = pipeline.generate(
            StoryInput(
                captions=story["captions"],
                reference_images=[story["ref_images_pil"]],
            ),
            generator=generator,
            height=ds_cfg["image_resolution"], width=ds_cfg["image_resolution"],
        )
        for i, img in enumerate(images):
            img.save(out_dir / f"{idx:06d}_{i}.png")


if __name__ == "__main__":
    main()
