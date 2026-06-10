import os, platform
import numpy as np
import torch
from torch.utils.data import (Dataset, DataLoader, ConcatDataset,
                               WeightedRandomSampler)
from config import CFG
from utils import get_sampler_generator

# Windows does not support fork-based multiprocessing in DataLoader
_NUM_WORKERS = 0 if platform.system() == "Windows" else 4


class FeatureDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, global_offset: int = 0):
        feat_path  = os.path.join(CFG.feature_dir, f"{dataset_name}_{split}_feat.npy")
        label_path = os.path.join(CFG.feature_dir, f"{dataset_name}_{split}_label.npy")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Features not found: {feat_path}\nRun: python extract_features.py")
        self.feats  = np.load(feat_path)
        self.labels = np.load(label_path)
        self.offset = global_offset
        self.ds_id  = CFG.datasets.index(dataset_name)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.feats[idx])           # [1536] always on disk
        if CFG.feature_mode == "cls":
            feat = feat[:CFG.backbone_dim]                 # [768]  CLS only
        # "concat": use full [1536] as-is
        label = int(self.labels[idx]) + self.offset
        return feat, label, self.ds_id


def get_loaders(split: str):
    if split == "train":
        datasets, sample_weights = [], []
        for ds_name in CFG.datasets:
            offset = CFG.dataset_offsets[ds_name]
            ds     = FeatureDataset(ds_name, "train", global_offset=offset)
            datasets.append(ds)
            w = 1.0 / len(ds)
            sample_weights.extend([w] * len(ds))
        combined    = ConcatDataset(datasets)
        num_samples = CFG.samples_per_dataset * len(CFG.datasets)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=num_samples,
            replacement=True,
            generator=get_sampler_generator(CFG.seed),  # seeded!
        )
        return DataLoader(combined, batch_size=CFG.batch_size, sampler=sampler,
                          num_workers=_NUM_WORKERS, pin_memory=True, drop_last=True)
    else:
        loaders = {}
        for ds_name in CFG.datasets:
            ds = FeatureDataset(ds_name, split, global_offset=0)
            loaders[ds_name] = DataLoader(ds, batch_size=256, shuffle=False,
                                          num_workers=_NUM_WORKERS, pin_memory=True)
        return loaders
