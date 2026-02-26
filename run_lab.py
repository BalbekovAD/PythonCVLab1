from __future__ import annotations

import gc
import os
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, cast

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch import Tensor, nn

PROJECT_ROOT: Path = Path(__file__).resolve().parent
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from car_color_lab.constants import (
    ARTIFACTS_DIR,
    DATASET_DIR,
    IMG_SIZE,
    MIN_CLASS_COUNT,
    MODEL_EVAL_BATCH_SIZE,
    MODEL_EVAL_NUM_WORKERS,
    MODEL_HPARAMS,
    MODEL_TRAIN_BATCH_SIZE,
    MODEL_TRAIN_NUM_WORKERS,
    MODELS_TO_RUN,
    RUN_MODE,
    SEED,
    SPLIT_RATIOS,
    TARGET_F1,
)
from car_color_lab.data_index import build_indexed_dataset
from car_color_lab.datasets import CarColorDataset, build_transforms
from car_color_lab.evaluate import evaluate_model, print_eval_metrics
from car_color_lab.io_utils import ensure_dir, load_checkpoint, save_json
from car_color_lab.models.factory import build_model
from car_color_lab.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_training_curves,
)
from car_color_lab.train import create_loader, train_model
from car_color_lab.types import CheckpointPaths, IndexedDataset, TrainingHistory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return torch.device("cuda")
    return torch.device("cpu")


def checkpoint_paths(model_name: str) -> CheckpointPaths:
    checkpoints_dir: Path = ARTIFACTS_DIR / "checkpoints"
    history_dir: Path = ARTIFACTS_DIR / "history"
    plots_dir: Path = ARTIFACTS_DIR / "plots"
    reports_dir: Path = ARTIFACTS_DIR / "reports"

    return CheckpointPaths(
        checkpoint=checkpoints_dir / f"{model_name}_best.pt",
        history_json=history_dir / f"{model_name}_history.json",
        curve_plot=plots_dir / f"{model_name}_curves.png",
        metrics_json=reports_dir / f"{model_name}_metrics.json",
        confusion_matrix_plot=plots_dir / f"{model_name}_confusion_matrix.png",
    )


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total: int = sum(parameter.numel() for parameter in model.parameters())
    trainable: int = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def _save_metadata(indexed: IndexedDataset) -> None:
    payload: dict[str, Any] = {
        "class_to_idx": indexed.class_to_idx,
        "split_sizes": {
            "train": len(indexed.splits.train),
            "val": len(indexed.splits.val),
            "test": len(indexed.splits.test),
        },
        "num_classes": len(indexed.idx_to_class),
        "num_samples": len(indexed.samples),
    }
    save_json(ARTIFACTS_DIR / "metadata" / "dataset_summary.json", payload)


def _print_class_distribution(indexed: IndexedDataset) -> None:
    print("\nClass distribution (after MIN_CLASS_COUNT filter):")
    total_samples: int = 0
    for class_name, count in sorted(indexed.class_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  {class_name:>14}: {count}")
        total_samples += count
    print(f"  {'TOTAL':>14}: {total_samples}")


def run() -> None:
    pipeline_start_time: float = perf_counter()
    if RUN_MODE not in {"full", "evaluation_only"}:
        raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}")

    set_seed(SEED)

    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(ARTIFACTS_DIR / "plots")
    ensure_dir(ARTIFACTS_DIR / "history")
    ensure_dir(ARTIFACTS_DIR / "checkpoints")
    ensure_dir(ARTIFACTS_DIR / "reports")
    ensure_dir(ARTIFACTS_DIR / "metadata")

    device: torch.device = resolve_device()
    print(f"Using device: {device}")

    indexed: IndexedDataset = build_indexed_dataset(
        dataset_dir=DATASET_DIR,
        min_class_count=MIN_CLASS_COUNT,
        split_ratios=SPLIT_RATIOS,
        seed=SEED,
    )

    plot_class_distribution(
        class_counts=indexed.class_counts,
        output_path=ARTIFACTS_DIR / "plots" / "class_distribution.png",
    )
    _print_class_distribution(indexed)
    _save_metadata(indexed)

    train_transform, eval_transform = build_transforms(IMG_SIZE)

    for model_name in MODELS_TO_RUN:
        # if model_name != "resnet50_pretrain": continue
        print(f"\n===== {model_name} =====")
        paths: CheckpointPaths = checkpoint_paths(model_name)

        train_batch_size: int = MODEL_TRAIN_BATCH_SIZE[model_name]
        eval_batch_size: int = MODEL_EVAL_BATCH_SIZE[model_name]
        train_num_workers: int = MODEL_TRAIN_NUM_WORKERS[model_name]
        eval_num_workers: int = MODEL_EVAL_NUM_WORKERS[model_name]

        if RUN_MODE == "full":
            model_train: nn.Module = build_model(
                model_name=model_name,
                num_classes=len(indexed.idx_to_class),
            )
            trainable_params, total_params = _count_parameters(model_train)
            print(
                f"[{model_name}] trainable_params={trainable_params:,} / total_params={total_params:,}"
            )

            train_dataset = CarColorDataset(
                samples=indexed.samples,
                targets=indexed.targets,
                indices=indexed.splits.train,
                transform=train_transform,
            )
            val_dataset = CarColorDataset(
                samples=indexed.samples,
                targets=indexed.targets,
                indices=indexed.splits.val,
                transform=eval_transform,
            )

            train_loader = create_loader(
                dataset=train_dataset,
                batch_size=train_batch_size,
                num_workers=train_num_workers,
                shuffle=True,
            )
            val_loader = create_loader(
                dataset=val_dataset,
                batch_size=eval_batch_size,
                num_workers=eval_num_workers,
                shuffle=False,
            )

            train_targets_split: list[int] = [indexed.targets[i] for i in indexed.splits.train]

            history: TrainingHistory = train_model(
                model=model_train,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                train_targets=train_targets_split,
                num_classes=len(indexed.idx_to_class),
                hparams=MODEL_HPARAMS[model_name],
                device=device,
                class_to_idx=indexed.class_to_idx,
                checkpoint_path=paths.checkpoint,
                history_path=paths.history_json,
            )

            plot_training_curves(history=history, output_path=paths.curve_plot)

        if not paths.checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint for {model_name} not found: {paths.checkpoint}. "
                "Run full mode first."
            )

        model_eval: nn.Module = build_model(
            model_name=model_name,
            num_classes=len(indexed.idx_to_class),
        )
        checkpoint: dict[str, Any] = load_checkpoint(paths.checkpoint, device=device)
        model_state_any: Any = checkpoint.get("model_state_dict")
        if not isinstance(model_state_any, dict):
            raise ValueError(f"Checkpoint {paths.checkpoint} does not contain model_state_dict")

        model_state = cast(dict[str, Tensor], model_state_any)
        model_eval.load_state_dict(model_state)
        model_eval.to(device)

        test_dataset = CarColorDataset(
            samples=indexed.samples,
            targets=indexed.targets,
            indices=indexed.splits.test,
            transform=eval_transform,
        )
        test_loader = create_loader(
            dataset=test_dataset,
            batch_size=eval_batch_size,
            num_workers=eval_num_workers,
            shuffle=False,
        )

        test_start_time: float = perf_counter()
        eval_artifacts = evaluate_model(
            model=model_eval,
            loader=test_loader,
            device=device,
            idx_to_class=indexed.idx_to_class,
            model_name=model_name,
        )
        test_elapsed: float = perf_counter() - test_start_time
        print(f"[{model_name}] test_sec={test_elapsed:.2f}")
        print_eval_metrics(eval_artifacts.metrics)

        metrics_payload: dict[str, Any] = {
            "model_name": eval_artifacts.metrics.model_name,
            "f1_macro": eval_artifacts.metrics.f1_macro,
            "f1_per_class": eval_artifacts.metrics.f1_per_class,
            "target_f1": TARGET_F1,
            "meets_target": eval_artifacts.metrics.f1_macro >= TARGET_F1,
            "eval_batch_size": eval_batch_size,
            "eval_num_workers": eval_num_workers,
        }
        save_json(paths.metrics_json, metrics_payload)

        plot_confusion_matrix(
            y_true=eval_artifacts.y_true,
            y_pred=eval_artifacts.y_pred,
            labels=indexed.idx_to_class,
            output_path=paths.confusion_matrix_plot,
        )

        if eval_artifacts.metrics.f1_macro < TARGET_F1:
            print(
                f"[warning] {model_name} did not reach TARGET_F1={TARGET_F1:.2f}. "
                f"Current={eval_artifacts.metrics.f1_macro:.4f}"
            )

        del model_eval
        del test_loader
        if RUN_MODE == "full":
            del model_train
            del train_loader
            del val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pipeline_elapsed: float = perf_counter() - pipeline_start_time
    print(f"\nPipeline finished. total_time_sec={pipeline_elapsed:.2f}")


if __name__ == "__main__":
    run()

