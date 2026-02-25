from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from car_color_lab.constants import ModelName


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    color: str


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


@dataclass(frozen=True)
class IndexedDataset:
    samples: list[SampleRecord]
    targets: list[int]
    class_to_idx: dict[str, int]
    idx_to_class: list[str]
    splits: SplitIndices
    class_counts: dict[str, int]


@dataclass(frozen=True)
class LoaderProfile:
    batch_size: int
    num_workers: int
    samples_per_second: float


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    val_f1_macro: list[float]
    learning_rate: list[float]


@dataclass(frozen=True)
class EvalMetrics:
    model_name: ModelName
    f1_macro: float
    f1_per_class: dict[str, float]


@dataclass(frozen=True)
class CheckpointPaths:
    checkpoint: Path
    history_json: Path
    curve_plot: Path
    metrics_json: Path
    confusion_matrix_plot: Path
