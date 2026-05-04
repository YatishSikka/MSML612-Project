"""Character F1 via VLCStoryGan's fine-tuned Inception-v3 classifier.

Place the checkpoint at checkpoints/vlcstorygan_pororo_inception.pth. The
classifier predicts which of the 9 PororoSV characters are present in a frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


PORORO_CHARACTERS = ["pororo", "crong", "loopy", "poby", "eddy",
                     "petty", "harry", "tongtong", "rody"]


def load_classifier(ckpt: str, num_classes: int = 9, device: str = "cuda"):
    from torchvision.models import inception_v3
    model = inception_v3(weights=None, num_classes=num_classes, aux_logits=True)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state if "state_dict" not in state else state["state_dict"])
    model.eval().to(device)
    return model


def classify_frame(model, img: Image.Image, device: str, threshold: float = 0.5):
    import torchvision.transforms as T
    tfm = T.Compose([
        T.Resize(299), T.CenterCrop(299), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.sigmoid(logits)[0].cpu()
    return (probs > threshold).tolist()


def f1(gt: set, pred: set) -> float:
    if not gt and not pred:
        return 1.0
    tp = len(gt & pred)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gt)
    return 2 * p * r / (p + r)


def compute_character_f1(gen_dir: str, gt_json: str, ckpt: str,
                         device: str = "cuda") -> float:
    """gt_json: {"story_id": {"frame_idx": ["char1", "char2"]}}."""
    model = load_classifier(ckpt, device=device)
    with open(gt_json) as f:
        gt_map = json.load(f)

    scores = []
    for story_id, frames in gt_map.items():
        for frame_idx, gt_chars in frames.items():
            path = Path(gen_dir) / f"{story_id}_{frame_idx}.png"
            if not path.exists():
                continue
            preds = classify_frame(model, Image.open(path), device)
            pred_chars = {PORORO_CHARACTERS[i] for i, v in enumerate(preds) if v}
            scores.append(f1(set(gt_chars), pred_chars))
    return sum(scores) / max(len(scores), 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    print(f"Character F1: {compute_character_f1(args.gen, args.gt, args.ckpt):.4f}")
