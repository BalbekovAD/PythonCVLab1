from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from sklearn.model_selection import train_test_split

from car_color_lab.constants import EXCLUDED_CLASSES
from car_color_lab.types import IndexedDataset, SampleRecord, SplitIndices

_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _iter_image_paths(dataset_dir: Path) -> Iterable[Path]:
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
            yield path


def parse_color_from_filename(path: Path) -> str:
    parts: list[str] = path.stem.split("$$")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return parts[3]


def build_indexed_dataset(
    dataset_dir: Path,
    min_class_count: int,
    split_ratios: tuple[float, float, float],
    seed: int,
) -> IndexedDataset:
    samples_all: list[SampleRecord] = []
    color_counts: Counter[str] = Counter()

    for path in _iter_image_paths(dataset_dir):
        color: str = parse_color_from_filename(path)
        samples_all.append(SampleRecord(path=path, color=color))
        color_counts[color] += 1

    excluded_normalized: set[str] = {color.casefold() for color in EXCLUDED_CLASSES}
    kept_colors: set[str] = {
        color
        for color, count in color_counts.items()
        if count >= min_class_count and color.casefold() not in excluded_normalized
    }

    filtered_samples: list[SampleRecord] = [
        sample for sample in samples_all if sample.color in kept_colors
    ]

    idx_to_class: list[str] = sorted(kept_colors)
    class_to_idx: dict[str, int] = {name: idx for idx, name in enumerate(idx_to_class)}

    targets: list[int] = [class_to_idx[sample.color] for sample in filtered_samples]

    all_indices: list[int] = list(range(len(filtered_samples)))
    train_ratio, val_ratio, test_ratio = split_ratios

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=targets,
    )

    temp_targets: list[int] = [targets[idx] for idx in temp_idx]
    val_share: float = val_ratio / (val_ratio + test_ratio)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_share),
        random_state=seed,
        stratify=temp_targets,
    )

    class_counts_filtered: Counter[str] = Counter(sample.color for sample in filtered_samples)

    return IndexedDataset(
        samples=filtered_samples,
        targets=targets,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        splits=SplitIndices(train=train_idx, val=val_idx, test=test_idx),
        class_counts=dict(class_counts_filtered),
    )
