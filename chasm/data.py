from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, IMAGE_EXTENSIONS



class AddGaussianNoise:
    def __init__(self, std: float = 0.02, p: float = 0.3):
        self.std = float(std)
        self.p = float(p)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return tensor + torch.randn_like(tensor) * self.std
        return tensor


def build_transform(train: bool, image_size: int = IMAGE_SIZE):
    if train:
        # Matches the final checkpoint training notebook: resize to 256 then random 224 crop.
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.02, p=0.3),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FolderDataset(Dataset):
    """Dataset for output of scripts/make_leakage_safe_split.py.

    Expected layout: root/{train,val,test}/<class_name>/<image>.
    """

    def __init__(self, root: str | Path, split: str, train_transform: Optional[bool] = None):
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {self.split_dir}")

        present = {p.name for p in self.split_dir.iterdir() if p.is_dir()}
        missing = [name for name in CLASS_NAMES if name not in present]
        if missing:
            raise RuntimeError(
                "Dataset class folders do not match the checkpoint class order. Missing: " + ", ".join(missing)
            )
        extras = sorted(present - set(CLASS_NAMES))
        if extras:
            raise RuntimeError("Unexpected class folders: " + ", ".join(extras))

        self.samples = []
        for label, class_name in enumerate(CLASS_NAMES):
            for path in sorted((self.split_dir / class_name).iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError(f"No supported images found in {self.split_dir}")
        if train_transform is None:
            train_transform = split == "train"
        self.transform = build_transform(bool(train_transform))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("L").convert("RGB")
            image = self.transform(image)
        return image, label, str(path)
