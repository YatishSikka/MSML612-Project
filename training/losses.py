"""Training losses: diffusion MSE + identity consistency auxiliary."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def diffusion_loss(model_pred: torch.Tensor, target: torch.Tensor,
                   loss_type: str = "mse") -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    if loss_type == "huber":
        return F.smooth_l1_loss(model_pred.float(), target.float(), reduction="mean")
    raise ValueError(loss_type)


def identity_consistency_loss(frame_features: torch.Tensor) -> torch.Tensor:
    """Average pairwise cosine distance between identity features across frames.

    frame_features: [B, F, D] where F = num frames per story.
    Lower = more consistent identity across frames.
    """
    b, f, d = frame_features.shape
    feats = F.normalize(frame_features, dim=-1)
    sim = torch.matmul(feats, feats.transpose(1, 2))  # [B, F, F]
    mask = 1.0 - torch.eye(f, device=feats.device).unsqueeze(0)
    off_diag = (sim * mask).sum(dim=(1, 2)) / (f * (f - 1))
    return (1.0 - off_diag).mean()
