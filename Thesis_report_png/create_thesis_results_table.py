#!/usr/bin/env python3
"""Create a thesis-ready PNG results table and print its LaTeX code.

The script reads the experiment JSON and extracts, for every architecture and
optimizer:
    Architecture, Optimizer, Test Accuracy, Validation Accuracy,
    Training Time, and Rank.

Ranking is calculated separately within each architecture. Higher final test
accuracy is better; training time is used as a tie-breaker.

Usage:
    python create_thesis_results_table.py

Optional:
    python create_thesis_results_table.py \
        --input mnist_full_experiment_13.json \
        --output thesis_results_table.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ARCHITECTURE_ORDER = ["MLP", "DeepMLP", "DeepMLPBN", "CNN"]
OPTIMIZER_ORDER = [
    "constant",
    "momentum",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "armijo",
]

OPTIMIZER_LABELS = {
    "constant": "Constant SGD",
    "momentum": "SGD with Momentum",
    "StepLR": "StepLR",
    "ExponentialLR": "ExponentialLR",
    "CosineAnnealingLR": "Cosine Annealing",
    "armijo": "Armijo Line Search",
}

VALIDATION_KEYS = (
    "validation_acc",
    "validation_accuracy",
    "val_acc",
    "val_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PNG table and print LaTeX from MNIST experiment JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("mnist_full_experiment_13.json"),
        help="Path to the experiment JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("thesis_results_table.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution in dots per inch.",
    )
    return parser.parse_args()


def final_numeric_value(values: Any) -> float | None:
    """Return the final numeric value from a scalar or sequence."""
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        value = float(values)
        return value if math.isfinite(value) else None

    if isinstance(values, list):
        for item in reversed(values):
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                value = float(item)
                if math.isfinite(value):
                    return value
    return None


def get_validation_accuracy(strategy_data: dict[str, Any]) -> float | None:
    """Read validation accuracy when the JSON contains a validation metric."""
    for key in VALIDATION_KEYS:
        if key in strategy_data:
            return final_numeric_value(strategy_data[key])
    return None


def get_training_time(strategy_data: dict[str, Any]) -> float | None:
    """Return total training time in seconds."""
    for key in ("total_training_time", "training_time", "total_time"):
        value = final_numeric_value(strategy_data.get(key))
        if value is not None:
            return value

    epoch_times = strategy_data.get("times_per_epoch")
    if isinstance(epoch_times, list):
        valid_times = [
            float(value)
            for value in epoch_times
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ]
        return sum(valid_times) if valid_times else None

    return None


def extract_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    architecture_names = sorted(
        data.keys(),
        key=lambda name: (
            ARCHITECTURE_ORDER.index(name)
            if name in ARCHITECTURE_ORDER
            else len(ARCHITECTURE_ORDER),
            name,
        ),
    )

    for architecture in architecture_names:
        strategies = data[architecture].get("strategies", {})
        optimizer_names = sorted(
            strategies.keys(),
            key=lambda name: (
                OPTIMIZER_ORDER.index(name)
                if name in OPTIMIZER_ORDER
                else len(OPTIMIZER_ORDER),
                name,
            ),
        )

        architecture_rows: list[dict[str, Any]] = []
        for optimizer in optimizer_names:
            strategy_data = strategies[optimizer]
            test_accuracy = final_numeric_value(strategy_data.get("test_acc"))

            if test_accuracy is None:
                print(
                    f"Warning: skipping {architecture}/{optimizer}; "
                    "no final test accuracy was found."
                )
                continue

            architecture_rows.append(
                {
                    "architecture": architecture,
                    "optimizer": OPTIMIZER_LABELS.get(optimizer, optimizer),
                    "test_accuracy": test_accuracy,
                    "validation_accuracy": get_validation_accuracy(strategy_data),
                    "training_time": get_training_time(strategy_data),
                }
            )

        # Rank independently for each architecture.
        ranked = sorted(
            architecture_rows,
            key=lambda row: (
                -row["test_accuracy"],
                row["training_time"]
                if row["training_time"] is not None
                else float("inf"),
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank

        rank_lookup = {row["optimizer"]: row["rank"] for row in ranked}
        for row in architecture_rows:
            row["rank"] = rank_lookup[row["optimizer"]]
            rows.append(row)

    if not rows:
        raise ValueError("No valid architecture/optimizer result rows were found.")

    return rows


def format_accuracy(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}\\%"


def format_accuracy_png(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def format_time(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} s"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def create_png_table(rows: list[dict[str, Any]], output: Path, dpi: int) -> None:
    columns = [
        "Architecture",
        "Optimizer",
        "Test Accuracy",
        "Validation Accuracy",
        "Training Time",
        "Rank",
    ]

    cell_text = [
        [
            row["architecture"],
            row["optimizer"],
            format_accuracy_png(row["test_accuracy"]),
            format_accuracy_png(row["validation_accuracy"]),
            format_time(row["training_time"]),
            str(row["rank"]),
        ]
        for row in rows
    ]

    figure_height = max(5.0, 0.43 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(14.5, figure_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.12, 0.24, 0.16, 0.18, 0.15, 0.07],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.12)
        elif column_index in (0, 1):
            cell.set_text_props(ha="left")

    ax.set_title(
        "Final Performance and Training-Time Comparison",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def build_latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Final performance and training-time comparison of the evaluated optimization strategies. Rankings are calculated separately for each architecture using final test accuracy, with lower training time used as a tie-breaker.}",
        r"\label{tab:final_optimizer_ranking}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Architecture} & \textbf{Optimizer} & \textbf{Test Acc.} & \textbf{Validation Acc.} & \textbf{Time (s)} & \textbf{Rank} " + r"\\",
        r"\midrule",
    ]

    previous_architecture: str | None = None
    for row in rows:
        architecture = latex_escape(row["architecture"])
        optimizer = latex_escape(row["optimizer"])
        test_acc = format_accuracy(row["test_accuracy"])
        validation_acc = format_accuracy(row["validation_accuracy"])
        training_time = (
            "N/A"
            if row["training_time"] is None
            else f'{row["training_time"]:.2f}'
        )

        if previous_architecture is not None and row["architecture"] != previous_architecture:
            lines.append(r"\midrule")

        lines.append(
            f"{architecture} & {optimizer} & {test_acc} & "
            f"{validation_acc} & {training_time} & {row['rank']} \\\\"
        )
        previous_architecture = row["architecture"]

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize\textit{Note:} The experiment JSON does not contain a validation-accuracy field; therefore, validation accuracy is reported as N/A unless such a field is added to the file.",
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input.resolve()}\n"
            "Place the JSON beside this script or pass --input /path/to/file.json"
        )

    with args.input.open("r", encoding="utf-8") as file:
        data = json.load(file)

    rows = extract_rows(data)
    create_png_table(rows, args.output, args.dpi)

    print(f"PNG table saved to: {args.output.resolve()}")
    print("\nLaTeX table code:\n")
    print(build_latex_table(rows))


if __name__ == "__main__":
    main()
