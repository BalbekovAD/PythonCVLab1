from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from car_color_lab.constants import TrainHyperParams
from car_color_lab.io_utils import save_checkpoint, save_json
from car_color_lab.types import TrainingHistory


def create_loader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader[tuple[Tensor, Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def _compute_class_weights(train_targets: list[int], num_classes: int, device: torch.device) -> Tensor:
    counts = np.bincount(np.array(train_targets, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_sqrt = 1.0 / np.sqrt(counts)
    normalized = inv_sqrt / np.sum(inv_sqrt) * float(num_classes)
    return torch.tensor(normalized, dtype=torch.float32, device=device)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    criterion: nn.Module,
    scaler: Any,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss: float = 0.0
    total_samples: int = 0
    skipped_oom_batches: int = 0

    progress = tqdm(loader, desc=f"Train {epoch_idx + 1}/{total_epochs}", leave=False)
    for images, targets in progress:
        try:
            images = images.to(device)
            target_tensor: Tensor = targets.to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs: Tensor = model(images)
                loss: Tensor = cast(Tensor, criterion(outputs, target_tensor))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_now: int = images.size(0)
            running_loss += float(loss.item()) * batch_size_now
            total_samples += batch_size_now
            progress.set_postfix(loss=f"{loss.item():.4f}")
        except torch.OutOfMemoryError:
            skipped_oom_batches += 1
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

    if skipped_oom_batches > 0:
        print(
            f"[warning] train epoch {epoch_idx + 1}: skipped OOM batches = {skipped_oom_batches}"
        )
    if total_samples == 0:
        raise RuntimeError("All train batches were skipped due to OOM.")

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def _validate_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
) -> tuple[float, float]:
    model.eval()
    running_loss: float = 0.0
    total_samples: int = 0
    skipped_oom_batches: int = 0

    y_true: list[int] = []
    y_pred: list[int] = []

    progress = tqdm(loader, desc=f"Val {epoch_idx + 1}/{total_epochs}", leave=False)
    for images, targets in progress:
        try:
            images = images.to(device)
            target_tensor: Tensor = targets.to(device=device, dtype=torch.long)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs: Tensor = model(images)
                loss: Tensor = cast(Tensor, criterion(outputs, target_tensor))

            predictions: Tensor = outputs.argmax(dim=1).cpu()
            y_pred.extend(predictions.tolist())
            y_true.extend(targets.tolist())

            batch_size_now: int = images.size(0)
            running_loss += float(loss.item()) * batch_size_now
            total_samples += batch_size_now
        except torch.OutOfMemoryError:
            skipped_oom_batches += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

    if skipped_oom_batches > 0:
        print(
            f"[warning] val epoch {epoch_idx + 1}: skipped OOM batches = {skipped_oom_batches}"
        )
    if total_samples == 0:
        raise RuntimeError("All validation batches were skipped due to OOM.")

    labels = sorted(set(y_true))
    val_f1_macro: float = float(f1_score(y_true, y_pred, average="macro", labels=labels))
    val_loss: float = running_loss / max(total_samples, 1)
    return val_loss, val_f1_macro


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    train_targets: list[int],
    num_classes: int,
    hparams: TrainHyperParams,
    device: torch.device,
    class_to_idx: dict[str, int],
    checkpoint_path: Path,
    history_path: Path,
) -> TrainingHistory:
    model.to(device)

    class_weights: Tensor = _compute_class_weights(train_targets, num_classes, device)
    criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    trainable_params: list[nn.Parameter] = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check freeze configuration.")

    optimizer: Optimizer = AdamW(
        params=trainable_params,
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
    )
    scheduler: LRScheduler = CosineAnnealingLR(optimizer=optimizer, T_max=hparams.epochs)
    scaler_cls: Any = getattr(torch.amp, "GradScaler")
    scaler: Any = scaler_cls(device.type, enabled=(device.type == "cuda"))

    history = TrainingHistory(train_loss=[], val_loss=[], val_f1_macro=[], learning_rate=[])
    best_f1: float = -1.0
    epochs_without_improvement: int = 0

    for epoch in range(hparams.epochs):
        epoch_start_time: float = perf_counter()

        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch_idx=epoch,
            total_epochs=hparams.epochs,
        )
        val_loss, val_f1 = _validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch_idx=epoch,
            total_epochs=hparams.epochs,
        )

        scheduler.step()

        current_lr: float = float(optimizer.param_groups[0]["lr"])
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_f1_macro.append(val_f1)
        history.learning_rate.append(current_lr)

        epoch_seconds: float = perf_counter() - epoch_start_time

        print(
            f"[{model_name}] epoch={epoch + 1}/{hparams.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} "
            f"lr={current_lr:.2e} epoch_time_sec={epoch_seconds:.2f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                class_to_idx=class_to_idx,
                model_name=model_name,
                best_val_f1=best_f1,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= hparams.patience:
                print(f"[{model_name}] early stopping at epoch {epoch + 1}")
                break

    history_payload: dict[str, object] = asdict(history)
    history_payload["best_val_f1"] = best_f1
    save_json(history_path, payload=history_payload)

    return history
