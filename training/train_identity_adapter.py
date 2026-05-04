"""Train the identity cross-attention adapter (IP-Adapter style).

Freezes the U-Net and text encoder; only the identity adapter's K/V/out
projections and the identity-encoder projection head are trained.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from pathlib import Path

logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.story_pipeline import StoryPipeline
from training.losses import diffusion_loss, identity_consistency_loss


def setup_logger(log_dir: Path, resume_step: int = 0) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = "resume" if resume_step > 0 else "run"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{tag}_{stamp}.log"

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    logger.info(f"log file: {log_path}")
    return logger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def story_collate(batch):
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    captions = [b["captions"] for b in batch]
    ref_pils = [b["ref_images_pil"] for b in batch]
    return {"frames": frames, "captions": captions, "ref_images_pil": ref_pils}


def build_dataloader(cfg):
    from data.preprocessing.story_dataset import PororoDataset
    ds = PororoDataset(
        data_dir=cfg["data"]["pororo_png_dir"] if cfg["data"]["dataset"] == "pororo"
        else cfg["data"]["flintstones_png_dir"],
        split="train",
        image_resolution=cfg["data"]["image_resolution"],
        frames_per_story=cfg["storydiffusion"]["frames_per_story"],
    )
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                      num_workers=cfg["data"]["num_workers"], shuffle=True,
                      pin_memory=True, drop_last=True, collate_fn=story_collate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--lora_rank", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lora_rank:
        cfg["lora"]["rank"] = args.lora_rank
    lora_cfg = cfg.get("lora", {})

    accelerator = Accelerator(
        mixed_precision="fp16" if args.precision == "fp16" else "no",
        gradient_accumulation_steps=cfg["training"]["grad_accum_steps"],
    )
    device = accelerator.device
    dtype = torch.float16 if args.precision == "fp16" else torch.float32

    pipeline = StoryPipeline(
        base_model=cfg["model"]["base_model"],
        identity_encoder_name=cfg["model"]["identity_encoder"],
        identity_dim=cfg["model"]["identity_dim"],
        num_identity_tokens=cfg["model"]["num_identity_tokens"],
        identity_scale=cfg["model"]["identity_scale"],
        torch_dtype=dtype,
        device=str(device),
        use_consistent_self_attn=False,
    )

    # Freeze everything except adapters + identity encoder projection.
    pipeline.pipe.unet.requires_grad_(False)
    pipeline.pipe.text_encoder.requires_grad_(False)
    pipeline.pipe.vae.requires_grad_(False)

    # Apply LoRA BEFORE re-enabling adapter params: peft's mark_only_lora_as_trainable
    # freezes everything, then we re-enable our adapter params on top.
    if lora_cfg.get("enabled", False):
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
        )
        pipeline.pipe.unet = get_peft_model(pipeline.pipe.unet, peft_config)
        for p in pipeline.pipe.unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    # Keep trainable params in fp32 so grad scaler can unscale properly.
    # Called after LoRA setup so these re-enable wins over peft's freeze.
    for p in pipeline.trainable_parameters():
        p.data = p.data.float()
        p.requires_grad_(True)

    if args.gradient_checkpointing:
        pipeline.pipe.unet.enable_gradient_checkpointing()

    trainable = list(pipeline.trainable_parameters())
    if lora_cfg.get("enabled", False):
        trainable += [p for p in pipeline.pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["training"]["warmup_steps"],
        num_training_steps=cfg["training"]["max_train_steps"],
    )

    loader = build_dataloader(cfg)
    loader, optimizer, scheduler = accelerator.prepare(loader, optimizer, scheduler)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["paths"].get("log_dir", str(ckpt_dir.parent / "logs")))

    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        accelerator.unwrap_model(pipeline.adapters).load_state_dict(ckpt["adapters"])
        pipeline.identity_encoder.proj.load_state_dict(ckpt["identity_proj"])
        if lora_cfg.get("enabled", False) and "unet_lora" in ckpt:
            unet = accelerator.unwrap_model(pipeline.pipe.unet)
            unet_sd = unet.state_dict()
            unet_sd.update(ckpt["unet_lora"])
            unet.load_state_dict(unet_sd)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            accelerator.print(f"[resume] no optimizer state in checkpoint — optimizer reset")
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # fast-forward the LR schedule to match the resumed step
            for _ in range(ckpt["step"]):
                scheduler.step()
            accelerator.print(f"[resume] no scheduler state in checkpoint — fast-forwarded to step {ckpt['step']}")
        global_step = ckpt["step"]
        accelerator.print(f"[resume] loaded checkpoint: {args.resume}  (step {global_step})")

    logger = None
    if accelerator.is_main_process:
        logger = setup_logger(log_dir, resume_step=global_step)
        trainable_count = sum(p.numel() for p in trainable if p.requires_grad)
        logger.info(f"config={args.config} | precision={args.precision} | "
                    f"resume={args.resume or 'none'} | start_step={global_step}")
        logger.info(f"trainable_params={trainable_count:,} | "
                    f"max_steps={cfg['training']['max_train_steps']} | "
                    f"lr={cfg['training']['learning_rate']} | "
                    f"grad_accum={cfg['training']['grad_accum_steps']}")
        if torch.cuda.is_available():
            logger.info(f"gpu={torch.cuda.get_device_name(0)} | "
                        f"vram_total_gb={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")

    pipe = pipeline.pipe
    noise_scheduler = pipe.scheduler
    identity_weight = cfg["training"]["identity_consistency_loss_weight"]

    progress = tqdm(total=cfg["training"]["max_train_steps"], initial=global_step,
                    disable=not accelerator.is_main_process)
    try:
        while global_step < cfg["training"]["max_train_steps"]:
            for batch in loader:
                with accelerator.accumulate(pipeline.adapters):
                    frames = batch["frames"].to(device, dtype=dtype)
                    b, fcount, c, h, w = frames.shape
                    frames_flat = frames.view(b * fcount, c, h, w)

                    latents = pipe.vae.encode(frames_flat).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],), device=device, dtype=torch.long,
                    )
                    noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                    flat_captions = [c for story in batch["captions"] for c in story]
                    text_in = pipe.tokenizer(
                        flat_captions, padding="max_length", truncation=True,
                        max_length=pipe.tokenizer.model_max_length, return_tensors="pt",
                    ).to(device)
                    with torch.no_grad():
                        text_emb = pipe.text_encoder(**text_in)[0]

                    id_tokens = pipeline.identity_encoder(
                        [batch["ref_images_pil"][i] for i in range(b)]
                    )
                    id_tokens = id_tokens.repeat_interleave(fcount, dim=0)

                    model_pred = pipe.unet(
                        noisy, timesteps, encoder_hidden_states=text_emb,
                        cross_attention_kwargs={"identity_tokens": id_tokens},
                    ).sample

                    target = noise if noise_scheduler.config.prediction_type == "epsilon" else \
                        noise_scheduler.get_velocity(latents, noise, timesteps)

                    loss = diffusion_loss(model_pred, target)

                    if identity_weight > 0:
                        pass  # TODO: identity consistency loss (needs detector)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable, cfg["training"]["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    progress.update(1)
                    if global_step % cfg["training"]["log_every"] == 0 and accelerator.is_main_process:
                        loss_val = float(loss.detach())
                        lr_val = scheduler.get_last_lr()[0]
                        vram_gb = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
                        progress.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_val:.2e}")
                        logger.info(
                            f"step={global_step} | loss={loss_val:.4f} | "
                            f"lr={lr_val:.3e} | vram_gb={vram_gb:.1f}"
                        )
                    if global_step % cfg["training"]["save_every"] == 0 and accelerator.is_main_process:
                        save_dict = {
                            "adapters": accelerator.unwrap_model(pipeline.adapters).state_dict(),
                            "identity_proj": pipeline.identity_encoder.proj.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": global_step,
                        }
                        if lora_cfg.get("enabled", False):
                            unet_sd = accelerator.unwrap_model(pipeline.pipe.unet).state_dict()
                            save_dict["unet_lora"] = {k: v for k, v in unet_sd.items() if "lora_" in k}
                        ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                        torch.save(save_dict, ckpt_path)
                        logger.info(f"checkpoint saved | step={global_step} | path={ckpt_path}")
                    if global_step >= cfg["training"]["max_train_steps"]:
                        break
    except Exception:
        import traceback
        if accelerator.is_main_process and logger:
            logger.error(f"CRASH at step={global_step}\n{traceback.format_exc()}")
        raise
    finally:
        if accelerator.is_main_process and logger:
            logger.info(f"training ended | final_step={global_step}")


if __name__ == "__main__":
    main()
