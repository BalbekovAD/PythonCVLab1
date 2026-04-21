from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data: Any = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON payload is not an object: {path}")
    return dict(data)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_to_idx: dict[str, int],
    model_name: str,
    best_val_f1: float,
) -> None:
    ensure_dir(path.parent)
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "model_name": model_name,
        "best_val_f1": best_val_f1,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint_any: Any = torch.load(path, map_location=device)
    if not isinstance(checkpoint_any, dict):
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return dict(checkpoint_any)
