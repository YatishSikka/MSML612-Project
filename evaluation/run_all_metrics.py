"""Run all metrics and produce a results table (CSV + JSON)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from evaluation.compute_character_f1 import compute_character_f1
from evaluation.compute_clip_similarity import (
    collect_whole_frame_crops, compute_cross_frame_clip,
)
from evaluation.compute_dino_similarity import compute_cross_frame_dino
from evaluation.compute_fid import compute_fid
from evaluation.compute_frame_accuracy import compute_frame_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--gt_json", required=True)
    ap.add_argument("--out", default="outputs/eval/results.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running metrics on device: {device}")

    results = {"gen_dir": args.gen_dir}
    metrics = cfg["metrics"]

    if metrics.get("fid"):
        results["fid"] = compute_fid(args.real_dir, args.gen_dir, device=device)
    if metrics.get("character_f1"):
        results["character_f1"] = compute_character_f1(
            args.gen_dir, args.gt_json, cfg["classifiers"]["pororo_inception"],
            device=device,
        )
    if metrics.get("frame_accuracy"):
        results["frame_accuracy"] = compute_frame_accuracy(
            args.gen_dir, args.gt_json, cfg["classifiers"]["pororo_inception"],
            device=device,
        )
    if metrics.get("cross_frame_clip") or metrics.get("cross_frame_dino"):
        crops = collect_whole_frame_crops(args.gen_dir, cfg["eval"]["frames_per_story"])
        if metrics.get("cross_frame_clip"):
            results["cross_frame_clip"] = compute_cross_frame_clip(crops, device=device)
        if metrics.get("cross_frame_dino"):
            results["cross_frame_dino"] = compute_cross_frame_dino(crops, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
