"""PororoSV dataset loader — reads directly from PNG files + numpy index arrays.

No HDF5 conversion needed. Uses the same index files as AR-LDM:
  - img_cache4.npy: first frame path per story (N,)
  - following_cache4.npy: next 4 frame paths per story (N, 4)
  - descriptions.npy: dict mapping "Season/Episode/frame_id" -> [caption]
  - train_seen_unseen_ids.npy: [train_ids, val_ids, test_ids]
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

SPLIT_MAP = {"train": 0, "val": 1, "test": 2}


class PororoDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train",
                 image_resolution: int = 512, frames_per_story: int = 5):
        self.data_dir = data_dir
        self.frames_per_story = frames_per_story

        self.imgs_list = np.load(
            os.path.join(data_dir, "img_cache4.npy"), encoding="latin1")
        self.followings = np.load(
            os.path.join(data_dir, "following_cache4.npy"))
        self.descriptions = np.load(
            os.path.join(data_dir, "descriptions.npy"),
            allow_pickle=True, encoding="latin1").item()

        all_ids = np.load(
            os.path.join(data_dir, "train_seen_unseen_ids.npy"),
            allow_pickle=True)
        self.ids = np.sort(all_ids[SPLIT_MAP[split]])

        self.tfm = T.Compose([
            T.Resize((image_resolution, image_resolution)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.tfm_pil = T.Resize((image_resolution, image_resolution))

    def __len__(self):
        return len(self.ids)

    def _get_paths(self, item_idx: int) -> List[str]:
        first = str(self.imgs_list[item_idx])
        if first.startswith("b'") or first.startswith('b"'):
            first = first[2:-1]
        rest = []
        for j in range(4):
            p = str(self.followings[item_idx][j])
            if p.startswith("b'") or p.startswith('b"'):
                p = p[2:-1]
            rest.append(p)
        return [first] + rest

    def _path_to_key(self, path: str) -> str:
        return path.replace(".png", "")

    def __getitem__(self, idx) -> Dict:
        item = self.ids[idx]
        paths = self._get_paths(item)
        paths = paths[:self.frames_per_story]

        frames = []
        captions = []
        ref_pil = None
        for i, rel_path in enumerate(paths):
            full_path = os.path.join(self.data_dir, rel_path)
            img = Image.open(full_path).convert("RGB")
            if i == 0:
                ref_pil = self.tfm_pil(img.copy())
            frames.append(self.tfm(img))

            key = self._path_to_key(rel_path)
            desc_list = self.descriptions.get(key, [""])
            captions.append(desc_list[0].strip())

        return {
            "frames": torch.stack(frames, dim=0),
            "captions": captions,
            "ref_images_pil": ref_pil,
        }
