#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


STRATEGY_NAMES = {
    "constant": "Constant LR",
    "momentum": "Momentum",
    "StepLR": "Step LR",
    "ExponentialLR": "Exponential LR",
    "CosineAnnealingLR": "Cosine Annealing",
    "armijo": "Armijo",
}

ARCHITECTURE_FILES = {
    "MLP": "mlp_per_class_accuracy.tex",
    "DeepMLP": "deepmlp_per_class_accuracy.tex",
    "DeepMLPBN": "deepmlpbn_per_class_accuracy.tex",
    "CNN": "cnn_per_class_accuracy.tex",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-class accuracy LaTeX tables from the experiment JSON."
        )
    )

    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to mnist_full_experiment_13.json",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables"),
        help="Directory in which LaTeX tables will be written.",
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object.")

    return data


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }

    return "".join(replacements.get(char, char) for char in text)


def strategy_name(strategy_key: str) -> str:
    return STRATEGY_NAMES.get(strategy_key, strategy_key)


def get_final_per_class_accuracy(
    strategy_data: Mapping[str, Any],
) -> list[float] | None:

    history = strategy_data.get("per_class_accuracy")

    if not isinstance(history, list) or not history:
        return None

    final_epoch = history[-1]

    if not isinstance(final_epoch, list) or not final_epoch:
        return None

    result: list[float] = []

    for value in final_epoch:

        if isinstance(value, bool):
            return None

        if not isinstance(value, (int, float)):
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        result.append(value)

    return result


def generate_table(
    architecture_name: str,
    architecture_data: Mapping[str, Any],
) -> str | None:

    strategies = architecture_data.get("strategies")

    if not isinstance(strategies, Mapping):
        return None

    rows: list[tuple[str, list[float]]] = []

    number_of_classes = None

    for strategy_key, strategy_data in strategies.items():

        if not isinstance(strategy_data, Mapping):
            continue

        values = get_final_per_class_accuracy(strategy_data)

        if values is None:
            continue

        if number_of_classes is None:
            number_of_classes = len(values)

        if len(values) != number_of_classes:
            continue

        rows.append(
            (
                strategy_name(strategy_key),
                values,
            )
        )

    if not rows or number_of_classes is None:
        return None

    architecture_label = architecture_name.lower()

    class_headers = [
        f"Class {i}"
        for i in range(number_of_classes)
    ]

    column_format = "l" + ("r" * number_of_classes)

    lines: list[str] = []

    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")

    lines.append(
        rf"\caption{{Final per-class test accuracy for the "
        rf"{latex_escape(architecture_name)} architecture.}}"
    )

    lines.append(
        rf"\label{{tab:{architecture_label}_per_class_accuracy}}"
    )

    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{column_format}}}")

    lines.append(r"\hline")

    header = (
        "Optimizer & "
        + " & ".join(class_headers)
        + r" \\"
    )

    lines.append(header)
    lines.append(r"\hline")

    for optimizer, values in rows:

        # Stored per_class_accuracy values are fractions in [0, 1].
        # Convert them to percentages for readability.
        formatted_values = [
            f"{value * 100:.2f}"
            for value in values
        ]

        row = (
            latex_escape(optimizer)
            + " & "
            + " & ".join(formatted_values)
            + r" \\"
        )

        lines.append(row)

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_all_tables(
    data: Mapping[str, Any],
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for architecture, filename in ARCHITECTURE_FILES.items():

        architecture_data = data.get(architecture)

        if not isinstance(architecture_data, Mapping):
            print(
                f"Skipping {architecture}: "
                "architecture not present in JSON."
            )
            continue

        table = generate_table(
            architecture,
            architecture_data,
        )

        if table is None:
            print(
                f"Skipping {architecture}: "
                "per_class_accuracy not available."
            )
            continue

        output_path = output_dir / filename

        output_path.write_text(
            table + "\n",
            encoding="utf-8",
        )

        print(f"Generated: {output_path}")


def main() -> None:

    args = parse_args()

    data = load_json(args.json_file)

    generate_all_tables(
        data=data,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()