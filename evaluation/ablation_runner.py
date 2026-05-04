"""Automated ablation runner: iterate over encoder / scale / seq length / char count."""

from __future__ import annotations

import argparse
import json
import subprocess
from itertools import product
from pathlib import Path

import yaml


def run_one(cfg_path: str, overrides: dict, out_root: Path) -> Path:
    """Generate + evaluate one configuration. Returns the results JSON path."""
    tag = "_".join(f"{k}={v}" for k, v in overrides.items())
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate
    gen_cmd = ["python", "inference/generate_stories.py",
               "--config", cfg_path, "--out", str(out_dir / "gen")]
    for k, v in overrides.items():
        gen_cmd += [f"--{k}", str(v)]
    subprocess.run(gen_cmd, check=True)

    # 2. Evaluate
    results_json = out_dir / "results.json"
    eval_cmd = ["python", "evaluation/run_all_metrics.py",
                "--config", cfg_path,
                "--gen_dir", str(out_dir / "gen"),
                "--real_dir", "data/pororo/real_test",
                "--gt_json", "data/pororo/test_characters.json",
                "--out", str(results_json)]
    subprocess.run(eval_cmd, check=True)
    return results_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_root", default="outputs/ablations")
    ap.add_argument("--dims", nargs="+",
                    default=["identity_encoders", "identity_scales"])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    axes = {d: cfg["ablations"][d] for d in args.dims}
    keys = list(axes.keys())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for combo in product(*[axes[k] for k in keys]):
        overrides = dict(zip(keys, combo))
        rj = run_one(args.config, overrides, out_root)
        with open(rj) as f:
            row = json.load(f)
        row.update(overrides)
        summary.append(row)

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
