"""Clip dataset and augmentation policies.

A sample is a tensor of shape (T, 3, S, S): T=8 frames, one per segment.
- Evaluation: centre candidate (j=1) of every segment, no randomness.
- Training, temporal jitter on (sport-aware policy): j ~ U{0,1,2} per segment.

Spatial and colour transforms are drawn ONCE per clip and applied to all T
frames (torchvision.transforms.v2 on a tv_tensors.Video does this).

Policies (aug.policy):
  none        : resize only
  generic     : RandomResizedCrop + HorizontalFlip + ColorJitter (as in the
                original VideoBadminton benchmark's augmentation types)
  sport_aware : RandomResizedCrop + ColorJitter + temporal jitter, NO flip

Spatial mode (data.spatial):
  squash      : resize the full 4:3 frame to SxS (keeps the whole court, so the
                far player near a side line is never cropped away)
  center_crop : resize short side to S, then centre crop SxS
"""
import os

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_jpeg, read_file
from torchvision.transforms import v2

from src.utils import read_csv

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transform(cfg, train):
    size = cfg["train"]["resolution"]
    aug = cfg["aug"]
    spatial = cfg["data"]["spatial"]
    t = []
    if train and aug["policy"] in ("generic", "sport_aware"):
        t.append(v2.RandomResizedCrop((size, size), scale=tuple(aug["rrc_scale"]),
                                      ratio=tuple(aug["rrc_ratio"]), antialias=True))
        if aug["policy"] == "generic":
            t.append(v2.RandomHorizontalFlip(p=aug["flip_p"]))
        b, c, s, h = aug["color_jitter"]
        t.append(v2.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h))
    elif spatial == "squash":
        t.append(v2.Resize((size, size), antialias=True))
    else:
        t += [v2.Resize(size, antialias=True), v2.CenterCrop(size)]
    t += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
    return v2.Compose(t)


class ClipDataset(Dataset):
    def __init__(self, cfg, split, train):
        d = cfg["data"]
        root = d.get("root", ".")
        self.frames_root = os.path.join(root, d["frames_root"])
        index = {r["clip_id"]: r for r in read_csv(os.path.join(root, d["index_csv"]))}
        split_rows = read_csv(os.path.join(root, d["split_csv"]))
        self.items = [(r["clip_id"], int(index[r["clip_id"]]["label_idx"]))
                      for r in split_rows if r["split"] == split]
        if cfg.get("smoke"):
            self.items = self.items[: 4 * cfg["train"]["batch_size"]]
        self.T = d["num_frames"]
        self.n_cand = d["n_cand"]
        self.jitter = train and cfg["aug"]["policy"] == "sport_aware"
        self.transform = build_transform(cfg, train)

    def __len__(self):
        return len(self.items)

    def labels(self):
        return [y for _, y in self.items]

    def clip_ids(self):
        return [c for c, _ in self.items]

    def __getitem__(self, i):
        clip_id, label = self.items[i]
        if self.jitter:
            js = torch.randint(0, self.n_cand, (self.T,)).tolist()
        else:
            js = [self.n_cand // 2] * self.T
        frames = [decode_jpeg(read_file(os.path.join(self.frames_root, clip_id, f"f{k}_{j}.jpg")))
                  for k, j in enumerate(js)]
        video = tv_tensors.Video(torch.stack(frames))  # T,3,H,W uint8
        video = self.transform(video)
        return video.as_subclass(torch.Tensor), label
