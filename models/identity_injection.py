"""Decoupled cross-attention for identity conditioning (IP-Adapter style).

Attaches a parallel cross-attention layer at each U-Net cross-attention block. Q
comes from the existing hidden states, K/V from the identity feature tokens.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityCrossAttention(nn.Module):
    def __init__(self, query_dim: int, identity_dim: int, num_heads: int = 8,
                 head_dim: Optional[int] = None, scale: float = 0.5):
        super().__init__()
        inner_dim = query_dim if head_dim is None else num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scale_attn = self.head_dim ** -0.5
        self.identity_scale = scale

        # Reuse the parent block's Q externally; here we own K/V and output.
        self.to_k_identity = nn.Linear(identity_dim, inner_dim, bias=False)
        self.to_v_identity = nn.Linear(identity_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

        # Keep projections fp32 for stability under autocast; cast outputs to match input.
        nn.init.zeros_(self.to_out.weight)  # start as identity pass-through

    def _split(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, identity_tokens: torch.Tensor) -> torch.Tensor:
        """query: [B, seq, query_dim]; identity_tokens: [B, N, identity_dim]."""
        in_dtype = query.dtype
        w_dtype = self.to_k_identity.weight.dtype
        identity_tokens = identity_tokens.to(dtype=w_dtype)
        k = self.to_k_identity(identity_tokens)
        v = self.to_v_identity(identity_tokens)
        q = query.to(dtype=w_dtype)

        q, k, v = self._split(q), self._split(k), self._split(v)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        out = self.to_out(out)
        return (self.identity_scale * out).to(in_dtype)


class DecoupledCrossAttnProcessor(nn.Module):
    """diffusers AttnProcessor that adds identity cross-attn alongside text.

    Intended to replace the existing cross-attn processor on U-Net cross-attn blocks.
    Identity tokens are passed via `cross_attention_kwargs["identity_tokens"]`.
    """

    def __init__(self, hidden_size: int, cross_attention_dim: int,
                 identity_dim: int, num_tokens: int = 4, scale: float = 0.5):
        super().__init__()
        self.identity_attn = IdentityCrossAttention(
            query_dim=hidden_size,
            identity_dim=identity_dim,
            num_heads=hidden_size // 64 if hidden_size % 64 == 0 else 8,
            scale=scale,
        )
        self.num_identity_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim

    def set_scale(self, scale: float):
        self.identity_attn.identity_scale = scale

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, identity_tokens=None, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape
        q = attn.to_q(hidden_states)
        ctx = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = attn.to_k(ctx)
        v = attn.to_v(ctx)

        # Standard text cross-attention.
        def _reshape(x):
            return x.view(batch_size, -1, attn.heads, attn.to_q.out_features // attn.heads).transpose(1, 2)
        qh, kh, vh = _reshape(q), _reshape(k), _reshape(v)
        text_out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attention_mask)
        text_out = text_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        text_out = attn.to_out[0](text_out)
        text_out = attn.to_out[1](text_out)

        # Identity branch (decoupled — separate K/V).
        if identity_tokens is not None:
            # CFG doubles the UNet batch (uncond + cond); repeat identity tokens to match.
            if identity_tokens.shape[0] != batch_size:
                repeat = batch_size // identity_tokens.shape[0]
                identity_tokens = identity_tokens.repeat(repeat, 1, 1)
            id_out = self.identity_attn(hidden_states, identity_tokens)
            text_out = text_out + id_out

        if input_ndim == 4:
            text_out = text_out.transpose(1, 2).view(b, c, h, w)
        if attn.residual_connection:
            text_out = text_out + residual
        return text_out / attn.rescale_output_factor


class _IdentityKwargFilter:
    """Wraps an attn1 processor to silently absorb the identity_tokens kwarg.

    Without this, diffusers' default AttnProcessor2_0 warns on every step when
    cross_attention_kwargs={"identity_tokens": ...} is forwarded to all blocks.
    """
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, attn, hidden_states, *, identity_tokens=None, **kwargs):
        return self.inner(attn, hidden_states, **kwargs)


def attach_identity_adapters(unet, identity_dim: int, num_tokens: int,
                             scale: float = 0.5) -> nn.ModuleList:
    """Swap cross-attention processors on `unet` with decoupled variants.

    Returns the list of new processor modules (trainable params live here).
    """
    new_procs = {}
    adapters = nn.ModuleList()
    for name, module in unet.attn_processors.items():
        if "attn2" not in name:  # attn2 = cross-attention in diffusers U-Net
            new_procs[name] = _IdentityKwargFilter(module)
            continue
        # Figure out hidden_size + cross_attention_dim from module path.
        block = unet
        for part in name.split(".")[:-1]:
            block = getattr(block, part) if not part.isdigit() else block[int(part)]
        hidden_size = block.to_q.out_features
        cross_dim = block.to_k.in_features
        proc = DecoupledCrossAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_dim,
            identity_dim=identity_dim,
            num_tokens=num_tokens,
            scale=scale,
        )
        new_procs[name] = proc
        adapters.append(proc)
    unet.set_attn_processor(new_procs)
    return adapters


def set_identity_scale(adapters: nn.ModuleList, scale: float):
    for proc in adapters:
        if isinstance(proc, DecoupledCrossAttnProcessor):
            proc.set_scale(scale)
