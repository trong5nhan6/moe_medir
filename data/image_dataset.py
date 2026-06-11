"""
Raw-image MedMNIST DataLoader for end-to-end fine-tuning.

Mirrors data/dataset.py (FeatureDataset / get_loaders) but loads images
instead of pre-extracted .npy features. Used by train_finetune.py.
"""
import platform
import medmnist
from medmnist import INFO
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from config import CFG
from utils import get_sampler_generator

_NUM_WORKERS = 0 if platform.system() == "Windows" else 4

NORM_STATS = {
    "clip_vitb32":   ((0.48145466, 0.4578275,  0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
    "biomedclip":    ((0.48145466, 0.4578275,  0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
    "dinov2_vitb14": ((0.485, 0.456, 0.406),
                      (0.229, 0.224, 0.225)),
    "convnext_base": ((0.485, 0.456, 0.406),
                      (0.229, 0.224, 0.225)),
}


def _make_transform(n_channels: int, backbone: str, augment: bool) -> transforms.Compose:
    mean, std = NORM_STATS[backbone]
    ops = [transforms.Resize((CFG.image_size, CFG.image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC)]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    if n_channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


class ImageMedMNIST(Dataset):
    """Single-dataset MedMNIST image loader with global label offset."""

    def __init__(self, dataset_name: str, split: str,
                 global_offset: int = 0, augment: bool = False):
        info      = INFO[dataset_name]
        transform = _make_transform(info["n_channels"], CFG.backbone, augment)
        DataClass = getattr(medmnist, info["python_class"])
        self.ds     = DataClass(split=split, transform=transform, download=True, size=28)
        self.offset = global_offset
        self.ds_id  = CFG.datasets.index(dataset_name)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        label = int(label.squeeze()) + self.offset
        return img, label, self.ds_id


def get_image_loaders(split: str):
    """
    Returns image DataLoader(s), same interface as get_loaders() in dataset.py.

    train → single DataLoader with WeightedRandomSampler (balanced across datasets)
    val/test → dict {dataset_name: DataLoader}
    """
    if split == "train":
        datasets, weights = [], []
        for ds_name in CFG.datasets:
            offset = CFG.dataset_offsets[ds_name]
            ds     = ImageMedMNIST(ds_name, "train", global_offset=offset, augment=True)
            datasets.append(ds)
            w = 1.0 / len(ds)
            weights.extend([w] * len(ds))
        combined    = ConcatDataset(datasets)
        num_samples = CFG.samples_per_dataset * len(CFG.datasets)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float32),
            num_samples=num_samples, replacement=True,
            generator=get_sampler_generator(CFG.seed),
        )
        return DataLoader(combined, batch_size=CFG.batch_size, sampler=sampler,
                          num_workers=_NUM_WORKERS, pin_memory=True, drop_last=True)
    else:
        loaders = {}
        for ds_name in CFG.datasets:
            ds = ImageMedMNIST(ds_name, split, global_offset=0, augment=False)
            loaders[ds_name] = DataLoader(ds, batch_size=128, shuffle=False,
                                          num_workers=_NUM_WORKERS, pin_memory=True)
        return loaders
