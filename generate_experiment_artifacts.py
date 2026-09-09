#!/usr/bin/env python3
"""
Generate thesis-ready PNG figures and LaTeX tables from an experiment JSON file.

The JSON file is treated as the single source of truth:
- Only fields that exist in the JSON are plotted or printed.
- Missing values are omitted rather than estimated.
- Final scalar values are taken from the last element of each available metric list.
- Final confusion matrices and per-class accuracies are taken from the last epoch.

Usage:
    python generate_experiment_artifacts.py   logs/mnist_full_experiment_13.json  --output-dir chapter4_artifacts

Dependencies:
    Python 3.9+
    matplotlib
    numpy
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = [
    "Class 0",
    "Class 1",
    "Class 2",
    "Class 3",
    "Class 4",
    "Class 5",
    "Class 6",
    "Class 7",
    "Class 8",
    "Class 9",
]

STRATEGY_LABELS = {
    "constant": "Constant LR",
    "momentum": "Momentum",
    "StepLR": "StepLR",
    "ExponentialLR": "ExponentialLR",
    "CosineAnnealingLR": "Cosine Annealing",
    "armijo": "Armijo",
    "wolfe": "Wolfe",
}

SERIES_LABELS = {
    "train_loss": "Training loss",
    "test_loss": "Test loss",
    "train_acc": "Training accuracy",
    "test_acc": "Test accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
    "generalization_gap_acc": "Accuracy gap",
    "generalization_gap_loss": "Loss gap",
    "learning_rates": "Learning rate",
    "times_per_epoch": "Time per epoch",
}

TABLE_METRICS = [
    ("train_loss", "Final train loss", 5),
    ("test_loss", "Final test loss", 5),
    ("train_acc", "Final train accuracy (\\%)", 2),
    ("test_acc", "Final test accuracy (\\%)", 2),
    ("precision", "Final precision", 5),
    ("recall", "Final recall", 5),
    ("f1_score", "Final F1-score", 5),
    ("generalization_gap_acc", "Final accuracy gap", 3),
    ("generalization_gap_loss", "Final loss gap", 5),
    ("learning_rates", "Final learning rate", 8),
]

PLOT_GROUPS = [
    ("loss_curves", ["train_loss", "test_loss"], "Loss", "Loss"),
    ("accuracy_curves", ["train_acc", "test_acc"], "Accuracy", "Accuracy (\\%)"),
    (
        "classification_metrics",
        ["precision", "recall", "f1_score"],
        "Classification metrics",
        "Score",
    ),
    (
        "generalization_gap_accuracy",
        ["generalization_gap_acc"],
        "Accuracy generalization gap",
        "Gap",
    ),
    (
        "generalization_gap_loss",
        ["generalization_gap_loss"],
        "Loss generalization gap",
        "Gap",
    ),
    ("learning_rate", ["learning_rates"], "Learning-rate evolution", "Learning rate"),
    ("epoch_time", ["times_per_epoch"], "Epoch duration", "Time (s)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create PNG figures and print LaTeX tables from an experiment JSON file."
        )
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to the experiment JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chapter4_artifacts"),
        help="Directory in which PNG files are stored.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300.",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help=(
            "Optional class names for confusion matrices and per-class plots. "
            "Provide one name per class."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("The root JSON value must be an object.")
    return data


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_").lower() or "unnamed"


def escape_latex(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def strategy_label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy)


def is_numeric_sequence(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in value
    )


def finite_numeric_sequence(value: Any) -> list[float]:
    if not is_numeric_sequence(value):
        return []
    result = []
    for item in value:
        number = float(item)
        if math.isfinite(number):
            result.append(number)
    return result


def final_numeric_value(strategy_data: Mapping[str, Any], metric: str) -> float | None:
    values = finite_numeric_sequence(strategy_data.get(metric))
    return values[-1] if values else None


def mean_numeric_value(strategy_data: Mapping[str, Any], metric: str) -> float | None:
    values = finite_numeric_sequence(strategy_data.get(metric))
    return float(np.mean(values)) if values else None


def sum_numeric_value(strategy_data: Mapping[str, Any], metric: str) -> float | None:
    values = finite_numeric_sequence(strategy_data.get(metric))
    return float(np.sum(values)) if values else None


def format_number(value: float | None, decimals: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return ""
    if value != 0 and abs(value) < 10 ** (-decimals):
        return f"{value:.3e}"
    return f"{value:.{decimals}f}"


def iter_architectures(data: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for architecture, architecture_data in data.items():
        if not isinstance(architecture_data, Mapping):
            continue
        strategies = architecture_data.get("strategies")
        if isinstance(strategies, Mapping):
            yield architecture, architecture_data


def available_strategies(
    architecture_data: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    strategies = architecture_data.get("strategies", {})
    if not isinstance(strategies, Mapping):
        return []
    return [
        (name, values)
        for name, values in strategies.items()
        if isinstance(values, Mapping)
    ]


def save_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"% PNG: {output_path.as_posix()}", file=sys.stderr)


def plot_metric_group(
    architecture: str,
    architecture_data: Mapping[str, Any],
    file_stem: str,
    metrics: Sequence[str],
    title_suffix: str,
    ylabel: str,
    output_dir: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    plotted = False

    for strategy, strategy_data in available_strategies(architecture_data):
        for metric in metrics:
            values = finite_numeric_sequence(strategy_data.get(metric))
            if not values:
                continue
            epochs = np.arange(1, len(values) + 1)
            label = f"{strategy_label(strategy)} — {SERIES_LABELS.get(metric, metric)}"
            ax.plot(epochs, values, label=label)
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(f"{architecture}: {title_suffix}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_figure(
        fig,
        output_dir / f"{slugify(architecture)}_{file_stem}.png",
        dpi,
    )


def plot_final_metric_comparison(
    architecture: str,
    architecture_data: Mapping[str, Any],
    metric: str,
    output_dir: Path,
    dpi: int,
) -> None:
    labels: list[str] = []
    values: list[float] = []

    for strategy, strategy_data in available_strategies(architecture_data):
        value = final_numeric_value(strategy_data, metric)
        if value is None:
            continue
        labels.append(strategy_label(strategy))
        values.append(value)

    if not values:
        return

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    positions = np.arange(len(values))
    ax.bar(positions, values)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(
        f"{architecture}: final {SERIES_LABELS.get(metric, metric).lower()}"
    )
    ax.set_ylabel(SERIES_LABELS.get(metric, metric))
    ax.grid(True, axis="y", alpha=0.25)

    save_figure(
        fig,
        output_dir / f"{slugify(architecture)}_final_{slugify(metric)}.png",
        dpi,
    )


def get_final_matrix(strategy_data: Mapping[str, Any], key: str) -> np.ndarray | None:
    matrices = strategy_data.get(key)
    if not isinstance(matrices, list) or not matrices:
        return None
    candidate = matrices[-1]
    try:
        matrix = np.asarray(candidate, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.isfinite(matrix).all():
        return None
    return matrix


def resolve_class_names(
    requested_names: Sequence[str] | None,
    class_count: int,
) -> list[str]:
    if requested_names and len(requested_names) == class_count:
        return list(requested_names)
    if class_count == len(CLASS_NAMES):
        return CLASS_NAMES.copy()
    return [f"Class {index}" for index in range(class_count)]


def plot_confusion_matrix(
    architecture: str,
    strategy: str,
    matrix: np.ndarray,
    split_name: str,
    output_dir: Path,
    dpi: int,
    requested_class_names: Sequence[str] | None,
) -> None:
    names = resolve_class_names(requested_class_names, matrix.shape[0])
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    image = ax.imshow(matrix, interpolation="nearest", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(
        f"{architecture} — {strategy_label(strategy)}: final {split_name} confusion matrix"
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)

    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value > threshold else "black",
            )

    save_figure(
        fig,
        output_dir
        / (
            f"{slugify(architecture)}_{slugify(strategy)}_"
            f"{slugify(split_name)}_confusion_matrix.png"
        ),
        dpi,
    )


def plot_per_class_accuracy(
    architecture: str,
    strategy: str,
    strategy_data: Mapping[str, Any],
    output_dir: Path,
    dpi: int,
    requested_class_names: Sequence[str] | None,
) -> None:
    all_epochs = strategy_data.get("per_class_accuracy")
    if not isinstance(all_epochs, list) or not all_epochs:
        return

    final_values = all_epochs[-1]
    if not is_numeric_sequence(final_values):
        return

    values = np.asarray(final_values, dtype=float)
    if not np.isfinite(values).all():
        return

    names = resolve_class_names(requested_class_names, len(values))
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    positions = np.arange(len(values))
    ax.bar(positions, values)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"{architecture} — {strategy_label(strategy)}: final per-class accuracy"
    )
    ax.grid(True, axis="y", alpha=0.25)

    save_figure(
        fig,
        output_dir
        / f"{slugify(architecture)}_{slugify(strategy)}_per_class_accuracy.png",
        dpi,
    )


def latex_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str,
    label: str,
    alignment: str | None = None,
) -> str:
    if not headers or not rows:
        return ""

    if alignment is None:
        alignment = "l" + "r" * (len(headers) - 1)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def print_architecture_summary_table(
    architecture: str,
    architecture_data: Mapping[str, Any],
) -> None:
    available_metrics = [
        (metric, heading, decimals)
        for metric, heading, decimals in TABLE_METRICS
        if any(
            final_numeric_value(strategy_data, metric) is not None
            for _, strategy_data in available_strategies(architecture_data)
        )
    ]

    include_time_total = any(
        sum_numeric_value(strategy_data, "times_per_epoch") is not None
        for _, strategy_data in available_strategies(architecture_data)
    )
    include_time_mean = any(
        mean_numeric_value(strategy_data, "times_per_epoch") is not None
        for _, strategy_data in available_strategies(architecture_data)
    )

    headers = ["Strategy"] + [heading for _, heading, _ in available_metrics]
    if include_time_total:
        headers.append("Total time (s)")
    if include_time_mean:
        headers.append("Mean epoch time (s)")

    rows: list[list[str]] = []
    for strategy, strategy_data in available_strategies(architecture_data):
        row = [escape_latex(strategy_label(strategy))]
        for metric, _, decimals in available_metrics:
            row.append(format_number(final_numeric_value(strategy_data, metric), decimals))
        if include_time_total:
            row.append(format_number(sum_numeric_value(strategy_data, "times_per_epoch"), 2))
        if include_time_mean:
            row.append(format_number(mean_numeric_value(strategy_data, "times_per_epoch"), 3))
        rows.append(row)

    if not rows:
        return

    print(
        latex_table(
            headers=headers,
            rows=rows,
            caption=(
                f"Final recorded metrics for the {escape_latex(architecture)} architecture."
            ),
            label=f"tab:{slugify(architecture)}_final_metrics",
        )
    )
    print()


def print_parameter_table(data: Mapping[str, Any]) -> None:
    rows: list[list[str]] = []
    for architecture, architecture_data in iter_architectures(data):
        total_parameters = architecture_data.get("total_parameters")
        if isinstance(total_parameters, int) and not isinstance(total_parameters, bool):
            rows.append(
                [
                    escape_latex(architecture),
                    f"{total_parameters:,}",
                ]
            )

    if not rows:
        return

    print(
        latex_table(
            headers=["Architecture", "Trainable parameters"],
            rows=rows,
            caption="Recorded model parameter counts.",
            label="tab:model_parameter_counts",
            alignment="lr",
        )
    )
    print()


def print_per_class_table(
    architecture: str,
    strategy: str,
    strategy_data: Mapping[str, Any],
    requested_class_names: Sequence[str] | None,
) -> None:
    all_epochs = strategy_data.get("per_class_accuracy")
    if not isinstance(all_epochs, list) or not all_epochs:
        return
    final_values = all_epochs[-1]
    if not is_numeric_sequence(final_values):
        return

    names = resolve_class_names(requested_class_names, len(final_values))
    rows = [
        [escape_latex(name), format_number(float(value), 5)]
        for name, value in zip(names, final_values)
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if not rows:
        return

    print(
        latex_table(
            headers=["Class", "Final accuracy"],
            rows=rows,
            caption=(
                f"Final per-class accuracy for {escape_latex(architecture)} "
                f"with {escape_latex(strategy_label(strategy))}."
            ),
            label=(
                f"tab:{slugify(architecture)}_{slugify(strategy)}_per_class_accuracy"
            ),
            alignment="lr",
        )
    )
    print()


def generate_all(
    data: Mapping[str, Any],
    output_dir: Path,
    dpi: int,
    requested_class_names: Sequence[str] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print_parameter_table(data)

    for architecture, architecture_data in iter_architectures(data):
        print_architecture_summary_table(architecture, architecture_data)

        for file_stem, metrics, title_suffix, ylabel in PLOT_GROUPS:
            plot_metric_group(
                architecture=architecture,
                architecture_data=architecture_data,
                file_stem=file_stem,
                metrics=metrics,
                title_suffix=title_suffix,
                ylabel=ylabel,
                output_dir=output_dir,
                dpi=dpi,
            )

        for metric in [
            "test_acc",
            "test_loss",
            "precision",
            "recall",
            "f1_score",
        ]:
            if any(
                final_numeric_value(strategy_data, metric) is not None
                for _, strategy_data in available_strategies(architecture_data)
            ):
                plot_final_metric_comparison(
                    architecture=architecture,
                    architecture_data=architecture_data,
                    metric=metric,
                    output_dir=output_dir,
                    dpi=dpi,
                )

        for strategy, strategy_data in available_strategies(architecture_data):
            for json_key, split_name in [
                ("train_confusion_matrices", "training"),
                ("test_confusion_matrices", "test"),
            ]:
                matrix = get_final_matrix(strategy_data, json_key)
                if matrix is not None:
                    plot_confusion_matrix(
                        architecture=architecture,
                        strategy=strategy,
                        matrix=matrix,
                        split_name=split_name,
                        output_dir=output_dir,
                        dpi=dpi,
                        requested_class_names=requested_class_names,
                    )

            if isinstance(strategy_data.get("per_class_accuracy"), list):
                plot_per_class_accuracy(
                    architecture=architecture,
                    strategy=strategy,
                    strategy_data=strategy_data,
                    output_dir=output_dir,
                    dpi=dpi,
                    requested_class_names=requested_class_names,
                )
                print_per_class_table(
                    architecture=architecture,
                    strategy=strategy,
                    strategy_data=strategy_data,
                    requested_class_names=requested_class_names,
                )


def main() -> int:
    args = parse_args()
    try:
        data = load_json(args.json_file)
        generate_all(
            data=data,
            output_dir=args.output_dir,
            dpi=args.dpi,
            requested_class_names=args.class_names,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
