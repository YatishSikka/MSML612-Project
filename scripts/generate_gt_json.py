"""Build ground-truth character JSON and copy real test frames for FID.

Outputs:
  data/pororo/test_characters.json  -- {story_id: {frame_idx: [char, ...]}}
  data/pororo/real_test/            -- flat dir of all GT test PNGs (for FID)

Character labels are extracted by searching each caption for the 9 known
PororoSV character names (case-insensitive).  Frames with no match get [].
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


PORORO_CHARACTERS = [
    "pororo", "crong", "loopy", "poby", "eddy",
    "petty", "harry", "tongtong", "rody",
]
_PATTERN = re.compile(
    r"\b(" + "|".join(PORORO_CHARACTERS) + r")\b", re.IGNORECASE
)

SPLIT_MAP = {"train": 0, "val": 1, "test": 2}


def extract_characters(caption: str) -> list[str]:
    return list(dict.fromkeys(m.lower() for m in _PATTERN.findall(caption)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="/home/batman/scratch.msml612pcs3/data/pororo/raw/pororo_png")
    ap.add_argument("--split", default="test")
    ap.add_argument("--frames_per_story", type=int, default=5)
    ap.add_argument("--gt_json_out", default="data/pororo/test_characters.json")
    ap.add_argument("--real_dir_out", default="data/pororo/real_test")
    ap.add_argument("--image_resolution", type=int, default=512)
    ap.add_argument("--num_stories", type=int, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    imgs_list = np.load(data_dir / "img_cache4.npy", encoding="latin1")
    followings = np.load(data_dir / "following_cache4.npy")
    descriptions = np.load(
        data_dir / "descriptions.npy", allow_pickle=True, encoding="latin1"
    ).item()
    all_ids = np.load(
        data_dir / "train_seen_unseen_ids.npy", allow_pickle=True
    )
    ids = np.sort(all_ids[SPLIT_MAP[args.split]])

    real_dir = Path(args.real_dir_out)
    real_dir.mkdir(parents=True, exist_ok=True)
    gt_json_path = Path(args.gt_json_out)
    gt_json_path.parent.mkdir(parents=True, exist_ok=True)

    n = args.num_stories or len(ids)
    gt_map: dict = {}

    try:
        from tqdm import tqdm
        _range = tqdm(range(n), desc="stories", unit="story")
    except ImportError:
        _range = range(n)

    for story_idx in _range:
        item = ids[story_idx]
        story_id = f"{story_idx:06d}"

        # Collect frame paths
        first = str(imgs_list[item])
        if first.startswith("b'") or first.startswith('b"'):
            first = first[2:-1]
        paths = [first]
        for j in range(4):
            p = str(followings[item][j])
            if p.startswith("b'") or p.startswith('b"'):
                p = p[2:-1]
            paths.append(p)
        paths = paths[: args.frames_per_story]

        gt_map[story_id] = {}
        for frame_idx, rel_path in enumerate(paths):
            full_path = data_dir / rel_path
            key = rel_path.replace(".png", "")
            desc_list = descriptions.get(key, [""])
            caption = desc_list[0].strip() if desc_list else ""
            chars = extract_characters(caption)
            gt_map[story_id][str(frame_idx)] = chars

            # Copy/resize to real_test dir for FID
            dst = real_dir / f"{story_id}_{frame_idx}.png"
            if not dst.exists():
                img = Image.open(full_path).convert("RGB").resize(
                    (args.image_resolution, args.image_resolution)
                )
                img.save(dst)

    with open(gt_json_path, "w") as f:
        json.dump(gt_map, f, indent=2)

    print(f"Wrote {len(gt_map)} stories to {gt_json_path}")
    print(f"Copied real frames to {real_dir}/ ({len(list(real_dir.glob('*.png')))} files)")


if __name__ == "__main__":
    main()
