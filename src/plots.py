

"""Reusable figure-saving helpers for the phishing detection project."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

FIGURES_DIR = Path("figures")


def save_class_balance(label_series: pd.Series, figures_dir: Path = FIGURES_DIR) -> None:
    """Save a bar chart of class balance to *figures_dir/class_balance.png*."""
    figures_dir.mkdir(exist_ok=True)
    class_counts = label_series.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    class_counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title("Class Balance")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of rows")
    ax.set_xticklabels(class_counts.index.astype(str), rotation=0)
    for pos, val in enumerate(class_counts):
        ax.text(pos, val, str(val), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(figures_dir / "class_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_feature_distributions(
    df: pd.DataFrame,
    feature_columns: list[str],
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """Save bar-plot distributions for every numeric feature."""
    figures_dir.mkdir(exist_ok=True)
    n_cols = 4
    n_rows = math.ceil(len(feature_columns) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, feature_columns):
        counts = df[col].value_counts(dropna=False).sort_index()
        counts.plot(kind="bar", ax=ax, color="#54A24B")
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)

    for ax in axes[len(feature_columns):]:
        ax.set_visible(False)

    fig.suptitle("Numeric and Binary Feature Distributions", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrices(
    y_true: pd.Series,
    predictions: dict,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """Save a grid of confusion matrices for every model in *predictions*."""
    figures_dir.mkdir(exist_ok=True)
    n_models = len(predictions)
    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        matrix = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(name)

    for ax in axes[n_models:]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices on Test Set", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curves(
    y_true: pd.Series,
    scores: dict,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """Save precision-recall curves for every model in *scores*."""
    figures_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, y_score in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})")

    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline={baseline:.3f}")
    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(figures_dir / "precision_recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
