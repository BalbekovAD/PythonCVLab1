from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from car_color_lab.constants import ModelName
from car_color_lab.types import EvalMetrics


@dataclass(frozen=True)
class EvalArtifacts:
    metrics: EvalMetrics
    y_true: list[int]
    y_pred: list[int]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    idx_to_class: list[str],
    model_name: ModelName,
) -> EvalArtifacts:
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    skipped_oom_batches: int = 0

    for images, targets in tqdm(loader, desc=f"Eval {model_name}", leave=False):
        try:
            images = images.to(device)
            outputs: Tensor = model(images)
            predictions: Tensor = outputs.argmax(dim=1).cpu()

            y_pred.extend(predictions.tolist())
            y_true.extend(targets.tolist())
        except torch.OutOfMemoryError:
            skipped_oom_batches += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

    if skipped_oom_batches > 0:
        print(f"[warning] eval {model_name}: skipped OOM batches = {skipped_oom_batches}")
    if len(y_true) == 0:
        raise RuntimeError("All evaluation batches were skipped due to OOM.")

    labels: list[int] = list(range(len(idx_to_class)))
    per_class_values = f1_score(y_true, y_pred, average=None, labels=labels)
    metrics = EvalMetrics(
        model_name=model_name,
        f1_macro=(float(f1_score(y_true, y_pred, average="macro", labels=labels))),
        f1_per_class={
            idx_to_class[idx]: float(per_class_values[idx]) for idx in range(len(idx_to_class))
        },
    )
    return EvalArtifacts(metrics=metrics, y_true=y_true, y_pred=y_pred)


def print_eval_metrics(metrics: EvalMetrics) -> None:
    print(f"\nModel: {metrics.model_name}")
    print(f"F1_macro: {metrics.f1_macro:.4f}")
    print("F1 per class:")
    for class_name, class_f1 in sorted(metrics.f1_per_class.items(), key=lambda item: item[0]):
        print(f"  {class_name:>14}: {class_f1:.4f}")
