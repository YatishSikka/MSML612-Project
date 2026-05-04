"""Smoke test: load 1 PororoSV sample, run 1 training step, verify shapes.

Run on Zaratan via: sbatch scripts/submit_smoke.sh
Or interactively:   salloc --partition=gpu --gres=gpu:1 --mem=32G --time=00:30:00
                    python scripts/smoke_test.py --config configs/train_config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.precision == "fp16" and device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Step 1: Dataset ---
    print("\n=== Step 1: Load PororoSV dataset ===")
    t0 = time.time()
    from data.preprocessing.story_dataset import PororoDataset
    frames_per_story = cfg["storydiffusion"].get("frames_per_story", 5)
    ds = PororoDataset(
        data_dir=cfg["data"]["pororo_png_dir"],
        split="train",
        image_resolution=cfg["data"]["image_resolution"],
        frames_per_story=frames_per_story,
    )
    print(f"Dataset size: {len(ds)} stories")

    sample = ds[0]
    frames = sample["frames"]
    captions = sample["captions"]
    ref_pil = sample["ref_images_pil"]
    print(f"frames shape: {frames.shape}")  # [F, 3, H, W]
    print(f"captions ({len(captions)}): {captions}")
    print(f"ref_pil type: {type(ref_pil)}, size: {ref_pil.size}")
    print(f"Dataset load: {time.time() - t0:.1f}s")

    # --- Step 2: Load SD 1.5 pipeline ---
    print("\n=== Step 2: Load SD 1.5 ===")
    t0 = time.time()
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model"]["base_model"], torch_dtype=dtype, safety_checker=None,
    ).to(device)
    pipe.enable_vae_slicing()
    print(f"SD 1.5 loaded: {time.time() - t0:.1f}s")

    # --- Step 3: Install Consistent Self-Attention ---
    print("\n=== Step 3: Install Consistent Self-Attention ===")
    from models.consistent_self_attention import install_consistent_self_attention
    csa_state = install_consistent_self_attention(
        pipe.unet,
        id_length=frames_per_story - 1,
        device=device, dtype=dtype,
    )
    print(f"CSA installed, total_count={csa_state.total_count} attn blocks patched")

    # --- Step 4: Identity encoder ---
    print("\n=== Step 4: Build identity encoder ===")
    t0 = time.time()
    from models.identity_encoder import build_identity_encoder
    id_encoder = build_identity_encoder(
        cfg["model"]["identity_encoder"],
        output_dim=cfg["model"]["identity_dim"],
        num_tokens=cfg["model"]["num_identity_tokens"],
    ).to(device=device, dtype=dtype)
    print(f"Identity encoder ({cfg['model']['identity_encoder']}): {time.time() - t0:.1f}s")

    with torch.no_grad():
        id_tokens = id_encoder([ref_pil])
    print(f"Identity tokens shape: {id_tokens.shape}")  # [1, N, dim]

    # --- Step 5: Attach identity adapters ---
    print("\n=== Step 5: Attach identity adapters ===")
    from models.identity_injection import attach_identity_adapters
    adapters = attach_identity_adapters(
        pipe.unet,
        identity_dim=cfg["model"]["identity_dim"],
        num_tokens=cfg["model"]["num_identity_tokens"],
        scale=cfg["model"]["identity_scale"],
    ).to(device=device, dtype=dtype)
    adapter_params = sum(p.numel() for p in adapters.parameters())
    proj_params = sum(p.numel() for p in id_encoder.proj.parameters())
    print(f"Adapter params: {adapter_params:,}")
    print(f"Identity proj params: {proj_params:,}")
    print(f"Total trainable: {adapter_params + proj_params:,}")

    # Enable gradient checkpointing AFTER all processor modifications.
    pipe.unet.enable_gradient_checkpointing()
    print("Gradient checkpointing: ON")

    # --- Step 6: Single training forward pass ---
    print("\n=== Step 6: Single training forward pass ===")
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    for p in adapters.parameters():
        p.requires_grad_(True)
    for p in id_encoder.proj.parameters():
        p.requires_grad_(True)

    frames_batch = frames.unsqueeze(0).to(device, dtype=dtype)  # [1, F, 3, H, W]
    b, fcount, c, h, w = frames_batch.shape
    frames_flat = frames_batch.view(b * fcount, c, h, w)

    with torch.no_grad():
        latents = pipe.vae.encode(frames_flat).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
    print(f"Latents shape: {latents.shape}")  # [F, 4, 64, 64]

    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps,
                              (latents.shape[0],), device=device, dtype=torch.long)
    noisy = pipe.scheduler.add_noise(latents, noise, timesteps)

    text_in = pipe.tokenizer(
        captions, padding="max_length", truncation=True,
        max_length=pipe.tokenizer.model_max_length, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        text_emb = pipe.text_encoder(**text_in)[0]
    print(f"Text embeddings shape: {text_emb.shape}")

    id_tok = id_tokens.repeat_interleave(fcount, dim=0)
    print(f"Identity tokens (repeated): {id_tok.shape}")

    with torch.cuda.amp.autocast(enabled=(dtype == torch.float16)):
        model_pred = pipe.unet(
            noisy, timesteps, encoder_hidden_states=text_emb,
            cross_attention_kwargs={"identity_tokens": id_tok},
        ).sample
    print(f"U-Net output shape: {model_pred.shape}")

    from training.losses import diffusion_loss
    target = noise
    loss = diffusion_loss(model_pred, target)
    print(f"Diffusion loss: {loss.item():.4f}")

    loss.backward()
    grad_norms = []
    for p in adapters.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    for p in id_encoder.proj.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    print(f"Grad norms — min: {min(grad_norms):.6f}, max: {max(grad_norms):.6f}, "
          f"mean: {sum(grad_norms)/len(grad_norms):.6f}")
    print(f"Forward+backward: {time.time() - t0:.1f}s")

    # --- Step 7: Memory usage ---
    if device == "cuda":
        print(f"\n=== VRAM ===")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Peak:      {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
