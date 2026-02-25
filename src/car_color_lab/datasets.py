from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from car_color_lab.types import SampleRecord

ImageTransform = Callable[[Image.Image], Tensor]


class CarColorDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        samples: list[SampleRecord],
        targets: list[int],
        indices: list[int],
        transform: ImageTransform,
    ) -> None:
        self._samples: list[SampleRecord] = samples
        self._targets: list[int] = targets
        self._indices: list[int] = indices
        self._transform: ImageTransform = transform

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        sample_idx: int = self._indices[idx]
        sample: SampleRecord = self._samples[sample_idx]
        target: int = self._targets[sample_idx]

        with Image.open(sample.path) as image:
            rgb_image: Image.Image = image.convert("RGB")
            tensor: Tensor = self._transform(rgb_image)

        target_tensor: Tensor = torch.tensor(target, dtype=torch.long)
        return tensor, target_tensor


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize: transforms.Normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    train_transform: transforms.Compose = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )

    eval_transform: transforms.Compose = transforms.Compose(
        [
            transforms.Resize(size=img_size + 32),
            transforms.CenterCrop(size=img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, eval_transform
