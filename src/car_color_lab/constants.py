from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

ModelName = Literal["resnet18_scratch", "resnet18_pretrain", "resnet50_pretrain"]

RunMode = Literal["full", "evaluation_only"]

RUN_MODE: Final[RunMode] = "evaluation_only"

DATASET_DIR: Final[Path] = Path("dataset")

ARTIFACTS_DIR: Final[Path] = Path("artifacts")

MODELS_TO_RUN: Final[tuple[ModelName, ModelName, ModelName]] = (
    "resnet18_scratch",
    "resnet18_pretrain",
    "resnet50_pretrain"
)

MIN_CLASS_COUNT: Final[int] = 300

EXCLUDED_CLASSES: Final[set[str]] = {"Unlisted"}

SPLIT_RATIOS: Final[tuple[float, float, float]] = (0.7, 0.15, 0.15)

SEED: Final[int] = 42

IMG_SIZE: Final[int] = 224

TARGET_F1: Final[float] = 0.8

PERSISTENT_WORKERS: Final[bool] = True

MODEL_TRAIN_BATCH_SIZE: Final[dict[ModelName, int]] = {
    "resnet18_scratch": 64 * 4,#448,
    "resnet18_pretrain": 64 * 4,#448,
    "resnet50_pretrain": 64 * 4#448,
}

MODEL_EVAL_BATCH_SIZE: Final[dict[ModelName, int]] = MODEL_TRAIN_BATCH_SIZE

MODEL_TRAIN_NUM_WORKERS: Final[dict[ModelName, int]] = {
    "resnet18_scratch": 4, # 4
    "resnet18_pretrain": 4, # 4
    "resnet50_pretrain": 4 , # 2
}
MODEL_EVAL_NUM_WORKERS: Final[dict[ModelName, int]] = MODEL_TRAIN_NUM_WORKERS

PRETRAIN_FREEZE_STAGES: Final[dict[ModelName, int]] = {
    "resnet18_scratch": 0,
    "resnet18_pretrain": 2,
    "resnet50_pretrain": 5,
}


@dataclass(frozen=True)
class TrainHyperParams:
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int


MODEL_HPARAMS: Final[dict[ModelName, TrainHyperParams]] = {
    "resnet18_scratch": TrainHyperParams(
        epochs=30,#epochs=30,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=4,
    ),
    "resnet18_pretrain": TrainHyperParams(
        epochs=25,#epochs=25,
        learning_rate=3e-4,
        weight_decay=1e-4,
        patience=3,
    ),
    "resnet50_pretrain": TrainHyperParams(
        epochs=25,#epochs=25,
        learning_rate=2e-4,
        weight_decay=1e-4,
        patience=3,
    ),
}
