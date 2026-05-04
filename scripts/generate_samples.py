"""Generate sample story frames from a trained identity adapter checkpoint.

Usage:
    python scripts/generate_samples.py \
        --config configs/train_config.yaml \
        --checkpoint outputs/checkpoints/step_26000.pt \
        --output_dir outputs/samples \
        --num_stories 5

    # Baseline (no adapter):
    python scripts/generate_samples.py \
        --config configs/train_config.yaml \
        --no_adapter \
        --output_dir outputs/samples_baseline \
        --num_stories 5
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import torch
import yaml
from PIL import Image

warnings.filterwarnings("ignore", message=".*cross_attention_kwargs.*not expected.*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--no_adapter", action="store_true", help="Baseline: no identity adapter")
    ap.add_argument("--output_dir", default="outputs/samples")
    ap.add_argument("--num_stories", type=int, default=5)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--identity_scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    if not args.no_adapter and args.checkpoint is None:
        ap.error("--checkpoint is required unless --no_adapter is set")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load dataset ---
    print("Loading dataset...")
    from data.preprocessing.story_dataset import PororoDataset
    ds = PororoDataset(
        data_dir=cfg["data"]["pororo_png_dir"],
        split=args.split,
        image_resolution=512,
        frames_per_story=5,
    )
    print(f"Dataset: {len(ds)} stories ({args.split} split)")

    # --- Load SD 1.5 ---
    print("Loading SD 1.5...")
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model"]["base_model"], torch_dtype=dtype, safety_checker=None,
    ).to(device)
    pipe.enable_vae_slicing()

    id_encoder = None
    adapters = None
    use_adapter = not args.no_adapter

    if use_adapter:
        print("Loading identity encoder...")
        from models.identity_encoder import build_identity_encoder
        id_encoder = build_identity_encoder(
            cfg["model"]["identity_encoder"],
            output_dim=cfg["model"]["identity_dim"],
            num_tokens=cfg["model"]["num_identity_tokens"],
        ).to(device=device, dtype=dtype)

        print("Attaching identity adapters...")
        from models.identity_injection import attach_identity_adapters
        adapters = attach_identity_adapters(
            pipe.unet,
            identity_dim=cfg["model"]["identity_dim"],
            num_tokens=cfg["model"]["num_identity_tokens"],
            scale=cfg["model"]["identity_scale"],
        )

        # Load checkpoint THEN cast to fp16
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        adapters.load_state_dict(ckpt["adapters"])
        id_encoder.proj.load_state_dict(ckpt["identity_proj"])
        step = ckpt.get("step", "unknown")
        print(f"Loaded checkpoint from step {step}")
        del ckpt

        # Cast to fp16 AFTER loading fp32 weights
        adapters = adapters.to(device=device, dtype=dtype)
        id_encoder = id_encoder.to(device=device, dtype=dtype)

        if args.identity_scale is not None:
            from models.identity_injection import set_identity_scale
            set_identity_scale(adapters, args.identity_scale)
            print(f"Identity scale set to {args.identity_scale}")

    # --- Generate ---
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for story_idx in range(min(args.num_stories, len(ds))):
        print(f"\n--- Story {story_idx + 1}/{args.num_stories} ---")
        sample = ds[story_idx]
        captions = sample["captions"]
        ref_pil = sample["ref_images_pil"]
        gt_frames = sample["frames"]

        for i, cap in enumerate(captions):
            print(f"  Frame {i}: {cap[:80]}...")

        story_dir = os.path.join(args.output_dir, f"story_{story_idx:03d}")
        os.makedirs(story_dir, exist_ok=True)

        # Save reference and ground truth
        ref_pil.save(os.path.join(story_dir, "reference.png"))
        from torchvision import transforms
        for i in range(gt_frames.shape[0]):
            gt_img = gt_frames[i] * 0.5 + 0.5  # denormalize from [-1,1] to [0,1]
            gt_img = gt_img.clamp(0, 1)
            gt_pil = transforms.ToPILImage()(gt_img)
            gt_pil.save(os.path.join(story_dir, f"gt_frame_{i}.png"))

        # Encode identity (if using adapter)
        cross_attn_kwargs = None
        if use_adapter:
            with torch.no_grad():
                id_tokens = id_encoder([ref_pil]).to(device=device, dtype=dtype)
                id_tok = id_tokens.expand(1, -1, -1)
            cross_attn_kwargs = {"identity_tokens": id_tok}

        # Generate frames
        t0 = time.time()
        for i, caption in enumerate(captions):
            with torch.no_grad():
                result = pipe(
                    prompt=caption,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=512, width=512,
                    generator=generator,
                    cross_attention_kwargs=cross_attn_kwargs,
                )
            img = result.images[0]
            img.save(os.path.join(story_dir, f"gen_frame_{i}.png"))
            print(f"  Generated frame {i}")

        # Comparison grid
        gt_imgs = []
        gen_imgs = []
        for i in range(len(captions)):
            gt_imgs.append(Image.open(os.path.join(story_dir, f"gt_frame_{i}.png")))
            gen_imgs.append(Image.open(os.path.join(story_dir, f"gen_frame_{i}.png")))

        n = len(captions)
        grid_w = n * 512
        grid_h = 2 * 512 + 40
        grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
        for i in range(n):
            grid.paste(gt_imgs[i].resize((512, 512)), (i * 512, 0))
            grid.paste(gen_imgs[i].resize((512, 512)), (i * 512, 512 + 40))
        grid.save(os.path.join(story_dir, "comparison.png"))

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

    print(f"\nAll samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
