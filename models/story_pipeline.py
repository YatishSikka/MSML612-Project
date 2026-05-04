"""Story generation pipeline: StoryDiffusion + identity cross-attention.

Two-pass generation following StoryDiffusion:
  1. Write pass: generate id_length reference frames as a batch (banks self-attn states).
  2. Read pass: generate each remaining frame one at a time (attends to banked states).

Our identity cross-attention is layered on top via DecoupledCrossAttnProcessor on attn2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from PIL import Image

from models.consistent_self_attention import install_consistent_self_attention
from models.identity_encoder import build_identity_encoder
from models.identity_injection import attach_identity_adapters, set_identity_scale


@dataclass
class StoryInput:
    captions: Sequence[str]
    reference_images: Sequence[Image.Image]  # one per character
    character_masks: Optional[Sequence[torch.Tensor]] = None


class StoryPipeline:
    """Inference pipeline: SD 1.5 + Consistent Self-Attention + identity adapter."""

    def __init__(self, base_model: str, identity_encoder_name: str,
                 identity_dim: int, num_identity_tokens: int,
                 identity_scale: float = 0.5,
                 id_length: int = 4,
                 sa32: float = 0.5, sa64: float = 0.5,
                 torch_dtype: torch.dtype = torch.float16,
                 device: str = "cuda",
                 use_consistent_self_attn: bool = True):
        from diffusers import StableDiffusionPipeline
        self.device = device
        self.dtype = torch_dtype
        self.id_length = id_length

        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model, torch_dtype=torch_dtype, safety_checker=None,
        ).to(device)
        self.pipe.enable_vae_slicing()

        # Consistent Self-Attention (StoryDiffusion's core mechanism).
        # Skipped during training — CSA is for inference-time consistency.
        self.csa_state = None
        if use_consistent_self_attn:
            self.csa_state = install_consistent_self_attention(
                self.pipe.unet, id_length=id_length,
                sa32=sa32, sa64=sa64, device=device, dtype=torch_dtype,
            )

        # Identity encoder + decoupled cross-attention adapters.
        self.identity_encoder = build_identity_encoder(
            identity_encoder_name, output_dim=identity_dim,
            num_tokens=num_identity_tokens,
        ).to(device=device, dtype=torch_dtype)

        self.adapters = attach_identity_adapters(
            self.pipe.unet, identity_dim=identity_dim,
            num_tokens=num_identity_tokens, scale=identity_scale,
        ).to(device=device, dtype=torch_dtype)

    def set_identity_scale(self, scale: float):
        set_identity_scale(self.adapters, scale)

    @staticmethod
    def _apply_lora(unet, lora_sd: dict, alpha: float = 32.0):
        """Merge LoRA delta weights directly into UNet base weights (no peft at inference)."""
        params = dict(unet.named_parameters())
        for key, val in lora_sd.items():
            if ".lora_A." not in key:
                continue
            b_key = key.replace("lora_A", "lora_B")
            if b_key not in lora_sd:
                continue
            # peft keys: "base_model.model.<path>.lora_A.default.weight"
            base_key = key.removeprefix("base_model.model.")
            base_key = base_key[: base_key.index(".lora_A")] + ".weight"
            if base_key not in params:
                continue
            A = val.float()                      # [r, in_features]
            B = lora_sd[b_key].float()           # [out_features, r]
            r = A.shape[0]
            delta = (alpha / r) * (B @ A)
            params[base_key].data.add_(delta.to(params[base_key].dtype))

    def load_checkpoint(self, ckpt_path: str, lora_alpha: float = 32.0):
        """Load adapter, identity projection, and LoRA weights from a checkpoint."""
        state = torch.load(ckpt_path, map_location=self.device)
        self.adapters.load_state_dict(state["adapters"])
        self.identity_encoder.proj.load_state_dict(state["identity_proj"])
        if state.get("unet_lora"):
            self._apply_lora(self.pipe.unet, state["unet_lora"], alpha=lora_alpha)

    @torch.no_grad()
    def generate(self, story: StoryInput, num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 height: int = 512, width: int = 512,
                 generator: Optional[torch.Generator] = None) -> List[Image.Image]:
        assert len(story.captions) >= 3, "StoryDiffusion requires >=3 captions"

        identity_tokens = self.identity_encoder(list(story.reference_images))
        id_tok = identity_tokens.view(1, -1, identity_tokens.shape[-1]).to(
            device=self.device, dtype=self.dtype)

        id_prompts = list(story.captions[:self.id_length])
        real_prompts = list(story.captions[self.id_length:])

        # --- Write pass: generate reference frames ---
        if self.csa_state is not None:
            self.csa_state.reset(write=True)
        id_tokens_batch = id_tok.expand(len(id_prompts), -1, -1)
        id_images = self.pipe(
            prompt=id_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height, width=width,
            generator=generator,
            cross_attention_kwargs={"identity_tokens": id_tokens_batch},
        ).images

        # --- Read pass: generate remaining frames one at a time ---
        real_images = []
        if self.csa_state is not None:
            self.csa_state.write = False
        for prompt in real_prompts:
            if self.csa_state is not None:
                self.csa_state.cur_step = 0
                self.csa_state.attn_count = 0
            single_id_tok = id_tok.expand(1, -1, -1)
            img = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height, width=width,
                generator=generator,
                cross_attention_kwargs={"identity_tokens": single_id_tok},
            ).images[0]
            real_images.append(img)

        return id_images + real_images

    def trainable_parameters(self):
        yield from self.adapters.parameters()
        yield from self.identity_encoder.proj.parameters()
