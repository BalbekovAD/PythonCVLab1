from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

# Available model ids in this project.
ModelName = Literal["resnet18_scratch", "resnet18_pretrain", "resnet50_pretrain"]

# Global pipeline mode.
# "full" -> prepare + train + evaluate
# "evaluation_only" -> skip train, only evaluate saved checkpoints
RunMode = Literal["full", "evaluation_only"]

# Pipeline mode used by run_lab.py.
RUN_MODE: Final[RunMode] = "full"

# Path to the dataset root folder.
DATASET_DIR: Final[Path] = Path("dataset")

# Path where all outputs (checkpoints, plots, reports) are stored.
ARTIFACTS_DIR: Final[Path] = Path("artifacts")

# Models to run in sequence.
MODELS_TO_RUN: Final[tuple[ModelName, ModelName, ModelName]] = (
    "resnet18_scratch",
    "resnet18_pretrain",
    "resnet50_pretrain"
)

# Drop classes with fewer samples than this threshold.
MIN_CLASS_COUNT: Final[int] = 300

# Classes to forcefully exclude even if they pass MIN_CLASS_COUNT.
EXCLUDED_CLASSES: Final[set[str]] = {"Unlisted"}

# Split ratios for train / val / test.
SPLIT_RATIOS: Final[tuple[float, float, float]] = (0.7, 0.15, 0.15)

# Global random seed for reproducibility.
SEED: Final[int] = 42

# Input image size (after preprocessing) for all models.
IMG_SIZE: Final[int] = 224

# Target metric for the lab.
TARGET_F1: Final[float] = 0.8

# Per-model train batch size.
# Use smaller value for larger model to avoid OOM.
MODEL_TRAIN_BATCH_SIZE: Final[dict[ModelName, int]] = {
    "resnet18_scratch": 256 + 64*3, # 256
    "resnet18_pretrain": 256 + 64*3, # 256
    "resnet50_pretrain": 128 * 2, # 128
}

# Per-model eval batch size.
MODEL_EVAL_BATCH_SIZE: Final[dict[ModelName, int]] = {
    "resnet18_scratch": 256 + 64*3, # 128
    "resnet18_pretrain": 256 + 64*3, # 128
    "resnet50_pretrain": 128 * 2, # 64
}

# Per-model train workers.
MODEL_TRAIN_NUM_WORKERS: Final[dict[ModelName, int]] = {
    "resnet18_scratch": 8, # 4
    "resnet18_pretrain": 8, # 4
    "resnet50_pretrain": 4, # 2
}

# Per-model eval workers.
MODEL_EVAL_NUM_WORKERS: Final[dict[ModelName, int]] = {
    "resnet18_scratch": 8, # 4
    "resnet18_pretrain": 8, # 4
    "resnet50_pretrain": 4, # 2
}

# Number of frozen early ResNet stages for each model.
# Stage mapping:
# 0 -> no freeze
# 1 -> conv1 + bn1 (stem)
# 2 -> stem + layer1
# 3 -> stem + layer1 + layer2
# 4 -> stem + layer1 + layer2 + layer3
#
# Based on transfer-learning guidance:
# - early layers are more general (Yosinski et al., NeurIPS 2014)
# - freezing most of the network speeds training (official PyTorch transfer tutorial)
# Default choice here is a conservative speed/quality compromise for color classification.
PRETRAIN_FREEZE_STAGES: Final[dict[ModelName, int]] = {
    "resnet18_scratch": 0,
    "resnet18_pretrain": 2,
    "resnet50_pretrain": 5,
}


@dataclass(frozen=True)
class TrainHyperParams:
    # Maximum epochs for this model.
    epochs: int
    # Initial learning rate.
    learning_rate: float
    # Weight decay for AdamW.
    weight_decay: float
    # Early stopping patience (epochs without val F1 improvement).
    patience: int


# Per-model optimization hyperparameters.
MODEL_HPARAMS: Final[dict[ModelName, TrainHyperParams]] = {
    "resnet18_scratch": TrainHyperParams(
        epochs=1,#epochs=30,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=4,
    ),
    "resnet18_pretrain": TrainHyperParams(
        epochs=1,#epochs=25,
        learning_rate=3e-4,
        weight_decay=1e-4,
        patience=3,
    ),
    "resnet50_pretrain": TrainHyperParams(
        epochs=1,#epochs=25,
        learning_rate=2e-4,
        weight_decay=1e-4,
        patience=3,
    ),
}
