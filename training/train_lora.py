"""Optional: LoRA fine-tune U-Net cross-attention alongside the identity adapter."""

from __future__ import annotations

import argparse

import torch
import yaml
from peft import LoraConfig, get_peft_model


def attach_lora(unet, rank: int, alpha: int, dropout: float, targets):
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=targets, bias="none",
    )
    return get_peft_model(unet, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # NOTE: Delegates to train_identity_adapter.py after wrapping U-Net in LoRA.
    # Kept as a separate entrypoint so the full-tuning vs LoRA-only distinction is explicit.
    raise NotImplementedError(
        "Import train_identity_adapter.main and wrap pipeline.pipe.unet with attach_lora "
        "before trainable_parameters() is called. Targets: "
        f"{cfg['lora']['target_modules']}"
    )


if __name__ == "__main__":
    main()
