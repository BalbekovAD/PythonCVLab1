from __future__ import annotations

import gc
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.profiler import ProfilerActivity, profile

PROJECT_ROOT: Path = Path(__file__).resolve().parent
SRC_ROOT: Path = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from car_color_lab.constants import (  # noqa: E402
    ARTIFACTS_DIR,
    DATASET_DIR,
    IMG_SIZE,
    MIN_CLASS_COUNT,
    MODELS_TO_RUN,
    MODEL_TRAIN_BATCH_SIZE,
    MODEL_TRAIN_NUM_WORKERS,
    SEED,
    SPLIT_RATIOS,
)
from car_color_lab.data_index import build_indexed_dataset  # noqa: E402
from car_color_lab.datasets import CarColorDataset, build_transforms  # noqa: E402
from car_color_lab.models.factory import build_model  # noqa: E402
from car_color_lab.train import create_loader  # noqa: E402

# -------- profiling configuration --------
PROFILE_STEPS: int = 40
WARMUP_STEPS: int = 5
ROW_LIMIT: int = 20


@dataclass(frozen=True)
class SectionTimings:
    build_datasets_sec: float
    build_dataloader_sec: float
    build_model_sec: float


@dataclass(frozen=True)
class LoopStats:
    wait_mean_ms: float
    wait_p50_ms: float
    wait_p95_ms: float
    step_mean_ms: float
    step_p50_ms: float
    step_p95_ms: float
    total_steps: int


@dataclass(frozen=True)
class ModelProfileSummary:
    device: str
    model_name: str
    batch_size: int
    num_workers: int
    trainable_params: int
    total_params: int
    section_timings: SectionTimings
    loop_stats: LoopStats


@dataclass(frozen=True)
class GlobalSetupTimings:
    build_indexed_dataset_sec: float
    build_transforms_sec: float


@dataclass(frozen=True)
class FullProfileSummary:
    global_setup: GlobalSetupTimings
    models: list[ModelProfileSummary]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _quantile_ms(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.array(values, dtype=np.float64), q) * 1000.0)


def _mean_ms(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.array(values, dtype=np.float64)) * 1000.0)


def _profile_one_model(
    model_name: str,
    indexed: Any,
    train_transform: Any,
    device: torch.device,
    profile_dir: Path,
) -> ModelProfileSummary:
    t0: float = perf_counter()
    train_dataset = CarColorDataset(
        samples=indexed.samples,
        targets=indexed.targets,
        indices=indexed.splits.train,
        transform=train_transform,
    )
    t1: float = perf_counter()

    batch_size: int = MODEL_TRAIN_BATCH_SIZE[model_name]
    num_workers: int = MODEL_TRAIN_NUM_WORKERS[model_name]
    train_loader = create_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    t2: float = perf_counter()

    model = build_model(model_name, num_classes=len(indexed.idx_to_class)).to(device)
    t3: float = perf_counter()

    params = [p for p in model.parameters() if p.requires_grad]
    trainable_params: int = sum(p.numel() for p in params)
    total_params: int = sum(p.numel() for p in model.parameters())

    optimizer = AdamW(params=params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    loader_iter = iter(train_loader)

    for _ in range(WARMUP_STEPS):
        try:
            images, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            images, targets = next(loader_iter)

        images = images.to(device)
        targets = targets.to(device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs: Tensor = model(images)
            loss: Tensor = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()

    activities: list[ProfilerActivity] = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    wait_times: list[float] = []
    step_times: list[float] = []

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(PROFILE_STEPS):
            start_wait: float = perf_counter()
            try:
                images, targets = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                images, targets = next(loader_iter)
            end_wait: float = perf_counter()

            start_step: float = perf_counter()
            images = images.to(device)
            targets = targets.to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_step: float = perf_counter()

            wait_times.append(end_wait - start_wait)
            step_times.append(end_step - start_step)
            prof.step()

    section_timings = SectionTimings(
        build_datasets_sec=(t1 - t0),
        build_dataloader_sec=(t2 - t1),
        build_model_sec=(t3 - t2),
    )

    loop_stats = LoopStats(
        wait_mean_ms=_mean_ms(wait_times),
        wait_p50_ms=_quantile_ms(wait_times, 0.50),
        wait_p95_ms=_quantile_ms(wait_times, 0.95),
        step_mean_ms=_mean_ms(step_times),
        step_p50_ms=_quantile_ms(step_times, 0.50),
        step_p95_ms=_quantile_ms(step_times, 0.95),
        total_steps=PROFILE_STEPS,
    )

    key: str = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table: str = prof.key_averages().table(sort_by=key, row_limit=ROW_LIMIT)
    table_path: Path = profile_dir / f"torch_profiler_table_{model_name}.txt"
    with table_path.open("w", encoding="utf-8") as f:
        f.write(table)

    print(f"\n=== MODEL {model_name} ===")
    print(
        json.dumps(
            {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "section_timings": asdict(section_timings),
                "loop_stats": asdict(loop_stats),
                "table_path": str(table_path),
            },
            indent=2,
        )
    )

    del model
    del train_loader
    del train_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ModelProfileSummary(
        device=str(device),
        model_name=model_name,
        batch_size=batch_size,
        num_workers=num_workers,
        trainable_params=trainable_params,
        total_params=total_params,
        section_timings=section_timings,
        loop_stats=loop_stats,
    )


def main() -> None:
    _set_seed(SEED)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0: float = perf_counter()
    indexed = build_indexed_dataset(
        dataset_dir=DATASET_DIR,
        min_class_count=MIN_CLASS_COUNT,
        split_ratios=SPLIT_RATIOS,
        seed=SEED,
    )
    t1: float = perf_counter()

    train_transform, _ = build_transforms(IMG_SIZE)
    t2: float = perf_counter()

    global_setup = GlobalSetupTimings(
        build_indexed_dataset_sec=(t1 - t0),
        build_transforms_sec=(t2 - t1),
    )

    profile_dir: Path = ARTIFACTS_DIR / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    model_summaries: list[ModelProfileSummary] = []
    for model_name in MODELS_TO_RUN:
        summary = _profile_one_model(
            model_name=model_name,
            indexed=indexed,
            train_transform=train_transform,
            device=device,
            profile_dir=profile_dir,
        )
        model_summaries.append(summary)

    full_summary = FullProfileSummary(global_setup=global_setup, models=model_summaries)

    summary_path: Path = profile_dir / "profile_summary_all_models.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(full_summary), f, ensure_ascii=False, indent=2)

    print("\n=== GLOBAL SETUP ===")
    print(json.dumps(asdict(global_setup), indent=2))
    print("\nSaved summary:")
    print(str(summary_path))


if __name__ == "__main__":
    main()
