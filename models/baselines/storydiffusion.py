"""Vanilla StoryDiffusion wrapper (no identity conditioning) for baseline runs."""

from __future__ import annotations

from typing import List, Sequence

import torch
from PIL import Image

from models.consistent_self_attention import install_consistent_self_attention


class VanillaStoryDiffusion:
    """StoryDiffusion baseline: Consistent Self-Attention only, no identity adapter."""

    def __init__(self, base_model: str = "runwayml/stable-diffusion-v1-5",
                 id_length: int = 4,
                 sa32: float = 0.5, sa64: float = 0.5,
                 torch_dtype: torch.dtype = torch.float16,
                 device: str = "cuda"):
        from diffusers import StableDiffusionPipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model, torch_dtype=torch_dtype, safety_checker=None,
        ).to(device)
        self.pipe.enable_vae_slicing()
        self.device = device
        self.id_length = id_length

        self.csa_state = install_consistent_self_attention(
            self.pipe.unet, id_length=id_length,
            sa32=sa32, sa64=sa64, device=device, dtype=torch_dtype,
        )

    @torch.no_grad()
    def generate(self, captions: Sequence[str], num_inference_steps: int = 50,
                 guidance_scale: float = 7.5, height: int = 512, width: int = 512,
                 generator=None) -> List[Image.Image]:
        assert len(captions) >= 3

        id_prompts = list(captions[:self.id_length])
        real_prompts = list(captions[self.id_length:])

        # Write pass.
        self.csa_state.reset(write=True)
        id_images = self.pipe(
            prompt=id_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height, width=width,
            generator=generator,
        ).images

        # Read pass.
        self.csa_state.write = False
        real_images = []
        for prompt in real_prompts:
            self.csa_state.cur_step = 0
            self.csa_state.attn_count = 0
            img = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height, width=width,
                generator=generator,
            ).images[0]
            real_images.append(img)

        return id_images + real_images
