#!/usr/bin/env python3
"""
Generate only the evidence-backed reports referenced in chapter04.tex.

The experiment JSON is the single source of truth. Reports are skipped when the
required values are absent. No missing values are estimated or fabricated.

Generated PNG reports:
  - fig:training_loss
  - fig:validation_loss              (uses test_loss because validation_loss is absent)
  - fig:accuracy_architecture
  - fig:cm_<architecture>
  - fig:generalization_gap_accuracy
  - fig:learning_rate_evolution
  - fig:<architecture>_results
  - fig:architecture_comparison
  - fig:training_time_comparison
  - fig:efficiency_comparison

Printed LaTeX tables:
  - tab:overall_summary
  - tab:overall_results
  - tab:convergence_rate
  - tab:test_accuracy
  - tab:precision_results
  - tab:recall_results
  - tab:f1_results
  - tab:training_time

Not generated because the JSON does not contain the required evidence:
  - experimental workflow diagram
  - gradient norm plot
  - optimization trajectory visualization
  - class-wise precision/recall/F1 table

Usage:
    python chapter04_reports.py mnist_full_experiment_13.json \
        --output-dir chapter04_reports > chapter04_tables.tex

        python chapter04_reports.py logs/mnist_full_experiment_13.json  --output-dir chapter04_reports5 > chapter04_tables5.tex
Dependencies:
    pip install numpy matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


STRATEGY_NAMES = {
    "constant": "Constant LR",
    "momentum": "Momentum",
    "StepLR": "Step LR",
    "ExponentialLR": "Exponential LR",
    "CosineAnnealingLR": "Cosine Annealing",
    "armijo": "Armijo",
    "wolfe": "Wolfe",
}

ARCHITECTURE_NAMES = {
    "MLP": "MLP",
    "DeepMLP": "DeepMLP",
    "DeepMLPBN": "DeepMLPBN",
    "CNN": "CNN",
}

MNIST_CLASS_NAMES = [str(i) for i in range(10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chapter04_reports"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("The JSON root must be an object.")
    return data


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def strategy_name(key: str) -> str:
    return STRATEGY_NAMES.get(key, key)


def architecture_name(key: str) -> str:
    return ARCHITECTURE_NAMES.get(key, key)


def numeric_series(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        return []
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return []
        number = float(item)
        if not math.isfinite(number):
            return []
        result.append(number)
    return result


def final_value(strategy: Mapping[str, Any], metric: str) -> float | None:
    values = numeric_series(strategy.get(metric))
    return values[-1] if values else None


def total_time(strategy: Mapping[str, Any]) -> float | None:
    values = numeric_series(strategy.get("times_per_epoch"))
    return float(sum(values)) if values else None


def mean_epoch_time(strategy: Mapping[str, Any]) -> float | None:
    values = numeric_series(strategy.get("times_per_epoch"))
    return float(np.mean(values)) if values else None


def final_confusion_matrix(strategy: Mapping[str, Any]) -> np.ndarray | None:
    matrices = strategy.get("test_confusion_matrices")
    if not isinstance(matrices, list) or not matrices:
        return None
    try:
        matrix = np.asarray(matrices[-1], dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.isfinite(matrix).all():
        return None
    return matrix


def architectures(data: Mapping[str, Any]):
    for architecture_key, architecture_data in data.items():
        if not isinstance(architecture_data, Mapping):
            continue
        strategies = architecture_data.get("strategies")
        if isinstance(strategies, Mapping):
            yield architecture_key, architecture_data


def strategies(architecture_data: Mapping[str, Any]):
    raw = architecture_data.get("strategies", {})
    if not isinstance(raw, Mapping):
        return
    for strategy_key, strategy_data in raw.items():
        if isinstance(strategy_data, Mapping):
            yield strategy_key, strategy_data


def best_strategy(
    architecture_data: Mapping[str, Any],
    metric: str = "test_acc",
) -> tuple[str, Mapping[str, Any]] | None:
    candidates = []
    for key, values in strategies(architecture_data):
        score = final_value(values, metric)
        if score is not None:
            candidates.append((score, key, values))
    if not candidates:
        return None
    _, key, values = max(candidates, key=lambda item: item[0])
    return key, values


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {path}", file=sys.stderr)


def latex_escape(text: Any) -> str:
    value = str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def fmt(value: float | None, decimals: int) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def print_table(
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    alignment: str | None = None,
) -> None:
    if not rows:
        return
    if alignment is None:
        alignment = "l" * len(headers)

    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(rf"\caption{{{caption}}}")
    print(rf"\label{{{label}}}")
    print(rf"\begin{{tabular}}{{{alignment}}}")
    print(r"\hline")
    print(" & ".join(headers) + r" \\")
    print(r"\hline")
    for row in rows:
        print(" & ".join(row) + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def _generate_loss_matrix(
    data: Mapping[str, Any],
    output_dir: Path,
    dpi: int,
    metric_key: str,
    ylabel: str,
    title: str,
    output_name: str,
) -> None:
    architecture_items = list(architectures(data))

    if not architecture_items:
        return

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    plotted_any = False

    for ax_index, ax in enumerate(axes):
        if ax_index >= len(architecture_items):
            ax.axis("off")
            continue

        architecture_key, architecture_data = architecture_items[ax_index]
        architecture_plotted = False

        for strategy_key, strategy_data in strategies(architecture_data):
            values = numeric_series(strategy_data.get(metric_key))
            if not values:
                continue

            epochs = np.arange(1, len(values) + 1)
            ax.plot(
                epochs,
                values,
                label=strategy_name(strategy_key),
                linewidth=2,
            )
            architecture_plotted = True
            plotted_any = True

        if architecture_plotted:
            ax.set_title(architecture_name(architecture_key), fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.legend(fontsize=8)
        else:
            ax.axis("off")

    if not plotted_any:
        plt.close(fig)
        return

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    save(fig, output_dir / output_name, dpi)


def generate_training_loss(
    data: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    _generate_loss_matrix(
        data=data,
        output_dir=output_dir,
        dpi=dpi,
        metric_key="train_loss",
        ylabel="Training loss",
        title="Training Loss Evolution by Architecture",
        output_name="training_loss.png",
    )


def generate_validation_loss(
    data: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    # The JSON contains test_loss but no validation_loss.
    _generate_loss_matrix(
        data=data,
        output_dir=output_dir,
        dpi=dpi,
        metric_key="test_loss",
        ylabel="Test loss",
        title="Test Loss Evolution by Architecture",
        output_name="validation_loss.png",
    )


def generate_accuracy_across_architectures(
    data: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    architecture_items = list(architectures(data))
    strategy_order: list[str] = []
    for _, architecture_data in architecture_items:
        for strategy_key, _ in strategies(architecture_data):
            if strategy_key not in strategy_order:
                strategy_order.append(strategy_key)

    x = np.arange(len(architecture_items))
    width = 0.8 / max(len(strategy_order), 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = False

    for index, strategy_key in enumerate(strategy_order):
        values = []
        positions = []
        for architecture_index, (_, architecture_data) in enumerate(architecture_items):
            strategy_data = architecture_data.get("strategies", {}).get(strategy_key)
            if not isinstance(strategy_data, Mapping):
                continue
            value = final_value(strategy_data, "test_acc")
            if value is None:
                continue
            positions.append(
                architecture_index - 0.4 + width / 2 + index * width
            )
            values.append(value)
        if values:
            ax.bar(positions, values, width=width, label=strategy_name(strategy_key))
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels(
        [architecture_name(key) for key, _ in architecture_items]
    )
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title("Classification accuracy across architectures")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save(fig, output_dir / "accuracy_architecture.png", dpi)


def generate_confusion_matrices(
    data: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    for architecture_key, architecture_data in architectures(data):
        selected = best_strategy(architecture_data)
        if selected is None:
            continue
        strategy_key, strategy_data = selected
        matrix = final_confusion_matrix(strategy_data)
        if matrix is None:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        image = ax.imshow(matrix, aspect="auto")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(
            f"{architecture_name(architecture_key)} — "
            f"{strategy_name(strategy_key)} final test confusion matrix"
        )
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")

        class_names = (
            MNIST_CLASS_NAMES
            if matrix.shape[0] == 10
            else [str(i) for i in range(matrix.shape[0])]
        )
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        threshold = float(matrix.max()) / 2.0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = int(matrix[row, column])
                ax.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > threshold else "black",
                )

        save(
            fig,
            output_dir / f"cm_{slug(architecture_key)}.png",
            dpi,
        )


def generate_generalization_gap_accuracy(
    data: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    """Generate the Chapter 4 accuracy generalization-gap report.

    The JSON field ``generalization_gap_acc`` is plotted exactly as recorded.
    A combined report is written to ``generalization_gap_accuracy.png`` and
    architecture-specific reports are also produced for easier inspection.
    """
    combined_fig, combined_ax = plt.subplots(figsize=(9.5, 5.8))
    combined_plotted = False

    for architecture_key, architecture_data in architectures(data):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        architecture_plotted = False

        for strategy_key, strategy_data in strategies(architecture_data):
            values = numeric_series(strategy_data.get("generalization_gap_acc"))
            if not values:
                continue

            epochs = np.arange(1, len(values) + 1)
            label = strategy_name(strategy_key)
            ax.plot(epochs, values, label=label)
            combined_ax.plot(
                epochs,
                values,
                label=(
                    f"{architecture_name(architecture_key)} — "
                    f"{strategy_name(strategy_key)}"
                ),
            )
            architecture_plotted = True
            combined_plotted = True

        if architecture_plotted:
            ax.axhline(0.0, linewidth=1.0, linestyle="--")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training accuracy - test accuracy (percentage points)")
            ax.set_title(
                f"{architecture_name(architecture_key)} accuracy generalization gap"
            )
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
            save(
                fig,
                output_dir / f"generalization_gap_accuracy_{slug(architecture_key)}.png",
                dpi,
            )
        else:
            plt.close(fig)

    if combined_plotted:
        combined_ax.axhline(0.0, linewidth=1.0, linestyle="--")
        combined_ax.set_xlabel("Epoch")
        combined_ax.set_ylabel(
            "Training accuracy - test accuracy (percentage points)"
        )
        combined_ax.set_title("Accuracy generalization gap during training")
        combined_ax.grid(True, alpha=0.25)
        combined_ax.legend(fontsize=6.5, ncol=2)
        save(combined_fig, output_dir / "generalization_gap_accuracy.png", dpi)
    else:
        plt.close(combined_fig)


def generate_learning_rate_evolution(
    data: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    """Generate combined and architecture-specific learning-rate reports."""
    combined_fig, combined_ax = plt.subplots(figsize=(9.5, 5.8))
    combined_plotted = False

    for architecture_key, architecture_data in architectures(data):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        architecture_plotted = False

        for strategy_key, strategy_data in strategies(architecture_data):
            values = numeric_series(strategy_data.get("learning_rates"))
            if not values:
                continue

            epochs = np.arange(1, len(values) + 1)
            label = strategy_name(strategy_key)
            ax.plot(epochs, values, label=label)
            combined_ax.plot(
                epochs,
                values,
                label=(
                    f"{architecture_name(architecture_key)} — "
                    f"{strategy_name(strategy_key)}"
                ),
            )
            architecture_plotted = True
            combined_plotted = True

        if architecture_plotted:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning rate")
            ax.set_title(
                f"{architecture_name(architecture_key)} learning-rate evolution"
            )
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
            save(
                fig,
                output_dir / f"learning_rate_{slug(architecture_key)}.png",
                dpi,
            )
        else:
            plt.close(fig)

    if combined_plotted:
        combined_ax.set_xlabel("Epoch")
        combined_ax.set_ylabel("Learning rate")
        combined_ax.set_title("Learning-rate evolution during training")
        combined_ax.grid(True, alpha=0.25)
        combined_ax.legend(fontsize=6.5, ncol=2)
        save(combined_fig, output_dir / "learning_rate_evolution.png", dpi)
    else:
        plt.close(combined_fig)


def generate_architecture_results(
    data: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    for architecture_key, architecture_data in architectures(data):
        labels: list[str] = []
        accuracy: list[float] = []
        f1: list[float] = []

        for strategy_key, strategy_data in strategies(architecture_data):
            acc = final_value(strategy_data, "test_acc")
            score = final_value(strategy_data, "f1_score")
            if acc is None and score is None:
                continue
            labels.append(strategy_name(strategy_key))
            accuracy.append(np.nan if acc is None else acc / 100.0)
            f1.append(np.nan if score is None else score)

        if not labels:
            continue

        x = np.arange(len(labels))
        width = 0.38
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.bar(x - width / 2, accuracy, width=width, label="Test accuracy")
        ax.bar(x + width / 2, f1, width=width, label="F1-score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"{architecture_name(architecture_key)} performance comparison")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        ax.set_ylim(0.95, 1.0)
        save(
            fig,
            output_dir / f"{slug(architecture_key)}_results.png",
            dpi,
        )


def generate_architecture_comparison(
    data: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = False

    for architecture_key, architecture_data in architectures(data):
        labels: list[str] = []
        values: list[float] = []
        for strategy_key, strategy_data in strategies(architecture_data):
            value = final_value(strategy_data, "test_acc")
            if value is None:
                continue
            labels.append(strategy_name(strategy_key))
            values.append(value)

        if values:
            ax.plot(labels, values, marker="o", label=architecture_name(architecture_key))
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Optimization strategy")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title("Architecture and optimization strategy comparison")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, output_dir / "architecture_comparison.png", dpi)


def aggregate_strategy_values(
    data: Mapping[str, Any], extractor
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for _, architecture_data in architectures(data):
        for strategy_key, strategy_data in strategies(architecture_data):
            value = extractor(strategy_data)
            if value is None or not math.isfinite(value):
                continue
            result.setdefault(strategy_key, []).append(float(value))
    return result


def generate_training_time_comparison(
    data: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    aggregated = aggregate_strategy_values(data, total_time)
    if not aggregated:
        return

    labels = [strategy_name(key) for key in aggregated]
    values = [float(np.mean(aggregated[key])) for key in aggregated]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    positions = np.arange(len(labels))
    ax.bar(positions, values)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean total training time (s)")
    ax.set_title("Computational cost across optimization strategies")
    ax.grid(True, axis="y", alpha=0.25)
    save(fig, output_dir / "training_time_comparison.png", dpi)


def generate_efficiency_comparison(
    data: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted = False

    for architecture_key, architecture_data in architectures(data):
        for strategy_key, strategy_data in strategies(architecture_data):
            time = total_time(strategy_data)
            accuracy = final_value(strategy_data, "test_acc")
            if time is None or accuracy is None:
                continue
            ax.scatter(time, accuracy)
            ax.annotate(
                f"{architecture_name(architecture_key)}\n{strategy_name(strategy_key)}",
                (time, accuracy),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Total training time (s)")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title("Predictive performance versus computational cost")
    ax.grid(True, alpha=0.25)
    save(fig, output_dir / "efficiency_comparison.png", dpi)


def print_overall_summary(data: Mapping[str, Any]) -> None:
    rows = []
    for architecture_key, architecture_data in architectures(data):
        for strategy_key, strategy_data in strategies(architecture_data):
            acc = final_value(strategy_data, "test_acc")
            f1 = final_value(strategy_data, "f1_score")
            precision = final_value(strategy_data, "precision")
            recall = final_value(strategy_data, "recall")
            if all(value is None for value in (acc, f1, precision, recall)):
                continue
            rows.append(
                [
                    latex_escape(architecture_name(architecture_key)),
                    latex_escape(strategy_name(strategy_key)),
                    fmt(acc, 2),
                    fmt(f1, 4),
                    fmt(precision, 4),
                    fmt(recall, 4),
                ]
            )

    print_table(
        "Overall summary of final recorded experimental results.",
        "tab:overall_summary",
        ["Architecture", "Optimizer", "Test Acc (\\%)", "F1 Score", "Precision", "Recall"],
        rows,
        "llrrrr",
    )


def print_overall_results(data: Mapping[str, Any]) -> None:
    raw_rows = []
    for architecture_key, architecture_data in architectures(data):
        architecture_rows = []
        for strategy_key, strategy_data in strategies(architecture_data):
            acc = final_value(strategy_data, "test_acc")
            loss = final_value(strategy_data, "test_loss")
            precision = final_value(strategy_data, "precision")
            time = total_time(strategy_data)
            if all(value is None for value in (acc, loss, precision, time)):
                continue
            architecture_rows.append(
                {
                    "strategy": strategy_key,
                    "accuracy": acc,
                    "loss": loss,
                    "precision": precision,
                    "time": time,
                }
            )

        ranked = sorted(
            architecture_rows,
            key=lambda row: (
                -(row["accuracy"] if row["accuracy"] is not None else -math.inf),
                row["loss"] if row["loss"] is not None else math.inf,
            ),
        )
        ranks = {row["strategy"]: index + 1 for index, row in enumerate(ranked)}

        for row in architecture_rows:
            raw_rows.append(
                [
                    latex_escape(strategy_name(row["strategy"])),
                    latex_escape(architecture_name(architecture_key)),
                    fmt(row["accuracy"], 2),
                    fmt(row["loss"], 5),
                    fmt(row["precision"], 4),
                    fmt(row["time"], 2),
                    str(ranks[row["strategy"]]),
                ]
            )

    print_table(
        "Overall comparison of the recorded optimization strategies.",
        "tab:overall_results",
        ["Optimizer", "Architecture", "Accuracy (\\%)", "Loss", "Precision", "Training Time (s)", "Rank"],
        raw_rows,
        "llrrrrr",
    )


def convergence_statistics(strategy_data: Mapping[str, Any]) -> tuple[float, float, float] | None:
    loss = numeric_series(strategy_data.get("test_loss"))
    if len(loss) < 3:
        return None

    initial_reduction = loss[0] - loss[min(4, len(loss) - 1)]
    tail = np.asarray(loss[-5:], dtype=float)
    final_stability = float(np.std(tail))
    oscillation = float(np.mean(np.abs(np.diff(loss))))
    return initial_reduction, final_stability, oscillation


def print_convergence_rate(data: Mapping[str, Any]) -> None:
    aggregate: dict[str, list[tuple[float, float, float]]] = {}
    for _, architecture_data in architectures(data):
        for strategy_key, strategy_data in strategies(architecture_data):
            stats = convergence_statistics(strategy_data)
            if stats is not None:
                aggregate.setdefault(strategy_key, []).append(stats)

    rows = []
    for strategy_key, values in aggregate.items():
        matrix = np.asarray(values, dtype=float)
        rows.append(
            [
                latex_escape(strategy_name(strategy_key)),
                fmt(float(np.mean(matrix[:, 0])), 5),
                fmt(float(np.mean(matrix[:, 1])), 6),
                fmt(float(np.mean(matrix[:, 2])), 6),
            ]
        )

    print_table(
        "Data-derived convergence characteristics averaged across architectures.",
        "tab:convergence_rate",
        ["Optimizer", "Initial Loss Reduction", "Final-Tail Std.", "Mean Loss Oscillation"],
        rows,
        "lrrr",
    )


def print_metric_matrix(
    data: Mapping[str, Any],
    metric: str,
    caption: str,
    label: str,
    decimals: int,
) -> None:
    architecture_items = list(architectures(data))
    strategy_order: list[str] = []
    for _, architecture_data in architecture_items:
        for strategy_key, _ in strategies(architecture_data):
            if strategy_key not in strategy_order:
                strategy_order.append(strategy_key)

    rows = []
    for strategy_key in strategy_order:
        row = [latex_escape(strategy_name(strategy_key))]
        has_value = False
        for _, architecture_data in architecture_items:
            strategy_data = architecture_data.get("strategies", {}).get(strategy_key)
            value = (
                final_value(strategy_data, metric)
                if isinstance(strategy_data, Mapping)
                else None
            )
            if value is not None:
                has_value = True
            row.append(fmt(value, decimals))
        if has_value:
            rows.append(row)

    print_table(
        caption,
        label,
        ["Optimizer"] + [latex_escape(architecture_name(key)) for key, _ in architecture_items],
        rows,
        "l" + "r" * len(architecture_items),
    )


def print_training_time(data: Mapping[str, Any]) -> None:
    total = aggregate_strategy_values(data, total_time)
    mean_epoch = aggregate_strategy_values(data, mean_epoch_time)
    if not total and not mean_epoch:
        return

    constant_values = total.get("constant", [])
    constant_mean = float(np.mean(constant_values)) if constant_values else None

    ordered_keys = []
    for key in list(total) + list(mean_epoch):
        if key not in ordered_keys:
            ordered_keys.append(key)

    rows = []
    for key in ordered_keys:
        total_value = float(np.mean(total[key])) if key in total else None
        epoch_value = float(np.mean(mean_epoch[key])) if key in mean_epoch else None
        relative = (
            total_value / constant_mean
            if total_value is not None and constant_mean not in (None, 0.0)
            else None
        )
        rows.append(
            [
                latex_escape(strategy_name(key)),
                fmt(total_value, 2),
                fmt(epoch_value, 3),
                fmt(relative, 3),
            ]
        )

    print_table(
        "Training-time comparison averaged across the recorded architectures.",
        "tab:training_time",
        ["Optimizer", "Training Time (s)", "Average Epoch Time (s)", "Relative Cost"],
        rows,
        "lrrr",
    )


def generate_all(data: Mapping[str, Any], output_dir: Path, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_training_loss(data, output_dir, dpi)
    generate_validation_loss(data, output_dir, dpi)
    generate_accuracy_across_architectures(data, output_dir, dpi)
    generate_confusion_matrices(data, output_dir, dpi)
    generate_generalization_gap_accuracy(data, output_dir, dpi)
    generate_learning_rate_evolution(data, output_dir, dpi)
    generate_architecture_results(data, output_dir, dpi)
    generate_architecture_comparison(data, output_dir, dpi)
    generate_training_time_comparison(data, output_dir, dpi)
    generate_efficiency_comparison(data, output_dir, dpi)

    print_overall_summary(data)
    print_overall_results(data)
    print_convergence_rate(data)
    print_metric_matrix(
        data,
        "test_acc",
        "Final test accuracy of the recorded optimization strategies.",
        "tab:test_accuracy",
        2,
    )
    print_metric_matrix(
        data,
        "precision",
        "Final precision of the recorded optimization strategies.",
        "tab:precision_results",
        4,
    )
    print_metric_matrix(
        data,
        "recall",
        "Final recall of the recorded optimization strategies.",
        "tab:recall_results",
        4,
    )
    print_metric_matrix(
        data,
        "f1_score",
        "Final F1-score of the recorded optimization strategies.",
        "tab:f1_results",
        4,
    )
    print_training_time(data)


def main() -> int:
    args = parse_args()
    try:
        data = load_json(args.json_file)
        generate_all(data, args.output_dir, args.dpi)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
