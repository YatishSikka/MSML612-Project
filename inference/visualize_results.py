"""Side-by-side comparison plots: baseline vs ours for the same stories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def plot_comparison(story_id: str, frames_per_story: int, dirs: dict, out: Path):
    fig, axes = plt.subplots(len(dirs), frames_per_story,
                             figsize=(3 * frames_per_story, 3 * len(dirs)))
    if len(dirs) == 1:
        axes = [axes]
    for row, (label, d) in enumerate(dirs.items()):
        for i in range(frames_per_story):
            p = Path(d) / f"{story_id}_{i}.png"
            if p.exists():
                axes[row][i].imshow(Image.open(p))
            axes[row][i].axis("off")
            if i == 0:
                axes[row][i].set_title(label, loc="left")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="label=path pairs, e.g. baseline=outputs/baseline ours=outputs/ours")
    ap.add_argument("--story_ids", nargs="+", required=True)
    ap.add_argument("--frames_per_story", type=int, default=5)
    ap.add_argument("--out_dir", default="outputs/figures")
    args = ap.parse_args()

    parsed = dict(pair.split("=", 1) for pair in args.dirs)
    for sid in args.story_ids:
        plot_comparison(sid, args.frames_per_story, parsed,
                        Path(args.out_dir) / f"{sid}.png")


if __name__ == "__main__":
    main()
