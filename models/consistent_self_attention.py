"""Consistent Self-Attention from StoryDiffusion.

Reimplemented without global variables. The processor shares self-attention
K/V across frames so characters stay visually consistent.

Two-pass generation:
  1. Write pass: generate id_length reference frames as a batch, bank hidden states.
  2. Read pass: generate each remaining frame. The processor concatenates banked
     ID states with the current frame, so Q attends to all ID frames' K/V.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConsistentAttnState:
    """Shared mutable state for all ConsistentSelfAttnProcessors in one U-Net."""
    write: bool = True
    cur_step: int = 0
    attn_count: int = 0
    total_count: int = 0
    id_length: int = 4
    total_length: int = 5  # id_length + 1
    sa16: float = 0.5
    sa32: float = 0.5
    sa64: float = 0.5
    mask256: Optional[torch.Tensor] = None
    mask1024: Optional[torch.Tensor] = None
    mask4096: Optional[torch.Tensor] = None

    def reset(self, write: bool = True):
        self.write = write
        self.cur_step = 0
        self.attn_count = 0


def cal_attn_mask(total_length, id_length, sa16, sa32, sa64, device="cuda", dtype=torch.float16):
    bool_matrix256 = torch.rand((1, total_length * 256), device=device, dtype=dtype) < sa16
    bool_matrix1024 = torch.rand((1, total_length * 1024), device=device, dtype=dtype) < sa32
    bool_matrix4096 = torch.rand((1, total_length * 4096), device=device, dtype=dtype) < sa64
    bool_matrix256 = bool_matrix256.repeat(total_length, 1)
    bool_matrix1024 = bool_matrix1024.repeat(total_length, 1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length, 1)
    for i in range(total_length):
        bool_matrix256[i:i+1, id_length*256:] = False
        bool_matrix1024[i:i+1, id_length*1024:] = False
        bool_matrix4096[i:i+1, id_length*4096:] = False
        bool_matrix256[i:i+1, i*256:(i+1)*256] = True
        bool_matrix1024[i:i+1, i*1024:(i+1)*1024] = True
        bool_matrix4096[i:i+1, i*4096:(i+1)*4096] = True
    mask256 = bool_matrix256.unsqueeze(1).repeat(1, 256, 1).reshape(-1, total_length * 256)
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1, 1024, 1).reshape(-1, total_length * 1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1, 4096, 1).reshape(-1, total_length * 4096)
    return mask256, mask1024, mask4096


class ConsistentSelfAttnProcessor(nn.Module):
    """Replaces self-attention (attn1) processors on U-Net up_blocks.

    During write pass: banks hidden states per step.
    During read pass: concatenates banked states so current frame can attend to ID frames.
    """

    def __init__(self, state: ConsistentAttnState, device="cuda", dtype=torch.float16):
        super().__init__()
        self.state = state
        self.device = device
        self.dtype = dtype
        self.id_bank: Dict[int, List[torch.Tensor]] = {}

    def _call_standard(self, attn, hidden_states, encoder_hidden_states=None,
                       attention_mask=None, temb=None):
        """Standard self-attention (no cross-frame sharing)."""
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def _call_cross_frame(self, attn, hidden_states, encoder_hidden_states=None,
                          attention_mask=None, temb=None):
        """Cross-frame attention: all frames' tokens concatenated, Q attends to all K/V."""
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)

        total_batch_size, nums_token, channel_dim = hidden_states.shape
        img_nums = total_batch_size // 2  # classifier-free guidance doubles batch
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel_dim).reshape(
            -1, img_nums * nums_token, channel_dim)

        batch_size, sequence_length, _ = hidden_states.shape
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Reshape back from [batch_size=2, img_nums*nums_token, dim] to
        # [total_batch_size, nums_token, dim] so the residual add in
        # BasicTransformerBlock sees the expected shape.
        hidden_states = hidden_states.reshape(total_batch_size, nums_token, -1)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        s = self.state

        if s.write:
            self.id_bank[s.cur_step] = [
                hidden_states[:s.id_length].clone(),
                hidden_states[s.id_length:].clone(),
            ]
        else:
            if s.cur_step in self.id_bank:
                encoder_hidden_states = torch.cat((
                    self.id_bank[s.cur_step][0].to(self.device),
                    hidden_states[:1],
                    self.id_bank[s.cur_step][1].to(self.device),
                    hidden_states[1:],
                ))

        if s.cur_step < 5:
            hidden_states = self._call_standard(attn, hidden_states, None,
                                                attention_mask, temb)
        else:
            rand_threshold = 0.3 if s.cur_step < 20 else 0.1
            if random.random() > rand_threshold:
                seq = hidden_states.shape[1]
                if seq == 32 * 32:
                    mask = s.mask1024
                elif seq == 16 * 16:
                    mask = s.mask256
                else:
                    mask = s.mask4096
                n_id = mask.shape[0] // s.total_length * s.id_length
                if s.write:
                    # Q = K = id_length frames → slice rows AND cols
                    attention_mask = mask[:n_id, :n_id]
                else:
                    # Q = 1 frame, K = total_length frames → slice rows only
                    attention_mask = mask[n_id:]
                hidden_states = self._call_cross_frame(attn, hidden_states, encoder_hidden_states,
                                                       attention_mask, temb)
            else:
                hidden_states = self._call_standard(attn, hidden_states, None,
                                                    attention_mask, temb)

        s.attn_count += 1
        if s.attn_count == s.total_count:
            s.attn_count = 0
            s.cur_step += 1
            s.mask256, s.mask1024, s.mask4096 = cal_attn_mask(
                s.total_length, s.id_length, s.sa16, s.sa32, s.sa64,
                device=self.device, dtype=self.dtype)

        return hidden_states


class DefaultAttnProcessor(nn.Module):
    """Pass-through processor for non-up_block self-attention (keeps standard behavior)."""
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def install_consistent_self_attention(unet, id_length: int = 4,
                                      sa16: float = 0.5, sa32: float = 0.5,
                                      sa64: float = 0.5,
                                      device: str = "cuda",
                                      dtype: torch.dtype = torch.float16) -> ConsistentAttnState:
    """Replace self-attention processors on U-Net up_blocks with ConsistentSelfAttnProcessor.

    Returns the shared state object for controlling write/read passes.
    Only up_block self-attention (attn1) gets the consistent processor —
    this matches StoryDiffusion's original implementation.
    """
    state = ConsistentAttnState(
        id_length=id_length,
        total_length=id_length + 1,
        sa16=sa16, sa32=sa32, sa64=sa64,
    )

    attn_procs = {}
    total = 0
    for name in unet.attn_processors.keys():
        is_self_attn = name.endswith("attn1.processor")
        if is_self_attn and name.startswith("up_blocks"):
            attn_procs[name] = ConsistentSelfAttnProcessor(
                state=state, device=device, dtype=dtype)
            total += 1
        else:
            attn_procs[name] = DefaultAttnProcessor()

    state.total_count = total
    unet.set_attn_processor(attn_procs)
    return state
