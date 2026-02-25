from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from car_color_lab.io_utils import ensure_dir
from car_color_lab.types import TrainingHistory


def plot_class_distribution(class_counts: dict[str, int], output_path: Path) -> None:
    ensure_dir(output_path.parent)

    labels: list[str] = list(class_counts.keys())
    values: list[int] = [class_counts[label] for label in labels]

    plt.figure(figsize=(14, 6))
    plt.bar(labels, values)
    plt.title("Class distribution after filtering")
    plt.xlabel("Color")
    plt.ylabel("Count")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_training_curves(history: TrainingHistory, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    epochs: list[int] = list(range(1, len(history.train_loss) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history.train_loss, label="train_loss")
    axes[0].plot(epochs, history.val_loss, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history.val_f1_macro, label="val_f1_macro", color="tab:green")
    axes[1].set_title("Validation F1 macro")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history.learning_rate, label="lr", color="tab:orange")
    axes[2].set_title("Learning rate")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(9, 9))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
