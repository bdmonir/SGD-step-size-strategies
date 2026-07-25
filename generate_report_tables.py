"""
generate_report_tables.py
--------------------------
Generates PNG table reports from an MNIST step-size strategy experiment JSON.

Goal: Compare how different step size strategies influence training dynamics,
      convergence speed, and classification performance across model architectures.

Usage:
    python generate_report_tables.py --input <path/to/experiment.json> --output_dir <path/to/output/>

Tables generated:
    1. final_performance_summary.png       – Final test acc, loss, F1, precision, recall per model/strategy
    2. convergence_speed.png               – Epoch at which each model/strategy first crosses accuracy thresholds
    3. training_dynamics.png               – Train vs test accuracy gap (generalization) and avg epoch time
    4. learning_rate_schedule_summary.png  – LR range (min/max) per strategy
    5. best_strategy_per_model.png         – Ranked strategies per model across key metrics
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────
STRATEGY_COLORS = {
    "constant":         "#4C72B0",
    "momentum":         "#DD8452",
    "StepLR":           "#55A868",
    "ExponentialLR":    "#C44E52",
    "CosineAnnealingLR":"#8172B2",
    "armijo":           "#937860",
}

MODEL_COLORS = {
    "MLP":        "#4C72B0",
    "DeepMLP":    "#DD8452",
    "DeepMLPBN":  "#55A868",
    "CNN":        "#C44E52",
}

STRATEGY_LABELS = {
    "constant":          "Constant LR",
    "momentum":          "Momentum",
    "StepLR":            "Step LR",
    "ExponentialLR":     "Exponential LR",
    "CosineAnnealingLR": "Cosine Annealing",
    "armijo":            "Armijo (line search)",
}

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _save(fig: plt.Figure, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✔  Saved: {path}")
    plt.close(fig)


def _styled_table(ax, col_labels, row_labels, cell_data,
                  col_colors=None, row_colors=None,
                  highlight_cols=None, highlight_fn=None):
    """Draw a nicely styled table on *ax* (which should be axis-off)."""
    ax.axis("off")
    n_rows = len(row_labels)
    n_cols = len(col_labels)

    # Build colour matrix for cells
    cell_colors = [["#f8f8f8" if r % 2 == 0 else "#ffffff"
                    for _ in range(n_cols)] for r in range(n_rows)]

    # Optionally highlight best value in specified columns
    if highlight_cols and highlight_fn:
        for c in highlight_cols:
            col_vals = []
            for r in range(n_rows):
                try:
                    col_vals.append(float(cell_data[r][c]))
                except (ValueError, TypeError):
                    col_vals.append(None)
            best_idx = highlight_fn(col_vals)
            if best_idx is not None:
                cell_colors[best_idx][c] = "#d4edda"   # soft green

    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=row_colors or (["#e8eaf6"] * n_rows),
        colColours=col_colors or (["#c5cae9"] * n_cols),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Bold column headers
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold", color="#1a237e")
        if c == -1:  # row label column
            cell.set_text_props(fontweight="bold", color="#37474f")
        cell.set_edgecolor("#cccccc")

    return tbl


def _best_idx(vals, mode="max"):
    """Return index of best non-None value."""
    valid = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not valid:
        return None
    if mode == "max":
        return max(valid, key=lambda x: x[1])[0]
    return min(valid, key=lambda x: x[1])[0]


# ──────────────────────────────────────────────
# Table 1 – Final Performance Summary
# ──────────────────────────────────────────────

def table_final_performance(data: dict, out_dir: str):
    models = list(data.keys())
    strategies = list(data[models[0]]["strategies"].keys())
    strat_labels = [STRATEGY_LABELS[s] for s in strategies]

    metrics = [
        ("Test Acc (%)",   "test_acc",   True,  "{:.2f}"),
        ("Train Acc (%)",  "train_acc",  True,  "{:.2f}"),
        ("Test Loss",      "test_loss",  False, "{:.4f}"),
        ("F1 Score",       "f1_score",   True,  "{:.4f}"),
        ("Precision",      "precision",  True,  "{:.4f}"),
        ("Recall",         "recall",     True,  "{:.4f}"),
    ]

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1,
                             figsize=(14, 4.5 * n_models),
                             facecolor="#fafafa")
    fig.suptitle("Table 1 — Final Performance Summary (Epoch 30)\n"
                 "Comparing Step Size Strategies on MNIST",
                 fontsize=14, fontweight="bold", y=1.01, color="#1a237e")

    for ax, model_name in zip(axes, models):
        params = data[model_name]["total_parameters"]
        ax.set_title(f"{model_name}  ({params:,} parameters)",
                     fontsize=11, fontweight="bold", color="#263238", pad=8)

        col_labels = [m[0] for m in metrics]
        row_labels  = strat_labels
        cell_data   = []
        for strat in strategies:
            sd = data[model_name]["strategies"][strat]
            row = []
            for _, key, _, fmt in metrics:
                val = sd[key][-1]
                if key in ("test_acc", "train_acc"):
                    row.append(fmt.format(val))
                elif key == "f1_score":
                    row.append(fmt.format(val))
                else:
                    row.append(fmt.format(val))
            cell_data.append(row)

        # highlight best per column
        highlight_cols = list(range(len(metrics)))

        def best_fn_for_col(col_idx):
            mode = "max" if metrics[col_idx][2] else "min"
            return lambda vals: _best_idx(vals, mode)

        # We can't easily pass col_idx into a lambda in a loop,
        # so manually highlight after drawing the table
        cell_colors_matrix = [["#f8f8f8" if r % 2 == 0 else "#ffffff"
                                for _ in range(len(metrics))]
                               for r in range(len(strategies))]

        for ci, (_, _, higher_is_better, _) in enumerate(metrics):
            vals = []
            for ri, strat in enumerate(strategies):
                sd = data[model_name]["strategies"][strat]
                vals.append(sd[metrics[ci][1]][-1])
            best = _best_idx(vals, "max" if higher_is_better else "min")
            if best is not None:
                cell_colors_matrix[best][ci] = "#d4edda"

        row_colors = [["#e8eaf6"] * 1 for _ in strategies]

        tbl = ax.table(
            cellText=cell_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellColours=cell_colors_matrix,
            rowColours=["#e8eaf6"] * len(strategies),
            colColours=["#c5cae9"] * len(metrics),
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.65)
        ax.axis("off")
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold", color="#1a237e")
            if c == -1:
                cell.set_text_props(fontweight="bold", color="#37474f")
            cell.set_edgecolor("#cccccc")

    patch = mpatches.Patch(color="#d4edda", label="Best value per column")
    fig.legend(handles=[patch], loc="lower right", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 1])
    _save(fig, os.path.join(out_dir, "1_final_performance_summary.png"))


# ──────────────────────────────────────────────
# Table 2 – Convergence Speed
# ──────────────────────────────────────────────

def table_convergence_speed(data: dict, out_dir: str):
    models     = list(data.keys())
    strategies = list(data[models[0]]["strategies"].keys())
    thresholds = [95.0, 97.0, 98.0, 98.5, 99.0]

    fig, axes = plt.subplots(len(models), 1,
                             figsize=(14, 4.0 * len(models)),
                             facecolor="#fafafa")
    fig.suptitle("Table 2 — Convergence Speed\n"
                 "Epoch at Which Test Accuracy First Reaches Each Threshold",
                 fontsize=14, fontweight="bold", y=1.01, color="#1a237e")

    for ax, model_name in zip(axes, models):
        ax.set_title(model_name, fontsize=11, fontweight="bold",
                     color="#263238", pad=8)
        col_labels = [f"≥{t:.1f}%" for t in thresholds] + ["Final Test Acc (%)"]
        row_labels  = [STRATEGY_LABELS[s] for s in strategies]
        cell_data   = []
        for strat in strategies:
            sd   = data[model_name]["strategies"][strat]
            accs = sd["test_acc"]
            row  = []
            for thr in thresholds:
                epoch = next((i + 1 for i, a in enumerate(accs) if a >= thr), None)
                row.append(str(epoch) if epoch is not None else "—")
            row.append(f"{accs[-1]:.2f}")
            cell_data.append(row)

        # colour cells: lower epoch = greener (column-wise for threshold cols)
        n_rows, n_cols = len(strategies), len(col_labels)
        cell_colors = [["#f8f8f8" if r % 2 == 0 else "#ffffff"
                        for _ in range(n_cols)] for r in range(n_rows)]

        for ci in range(len(thresholds)):
            vals = []
            for ri in range(n_rows):
                try:
                    vals.append(int(cell_data[ri][ci]))
                except (ValueError, TypeError):
                    vals.append(None)
            best = _best_idx(vals, "min")
            if best is not None:
                cell_colors[best][ci] = "#d4edda"

        # best final acc
        final_vals = [float(cell_data[ri][-1]) for ri in range(n_rows)]
        best_final = _best_idx(final_vals, "max")
        if best_final is not None:
            cell_colors[best_final][-1] = "#d4edda"

        tbl = ax.table(
            cellText=cell_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellColours=cell_colors,
            rowColours=["#e8eaf6"] * n_rows,
            colColours=["#c5cae9"] * n_cols,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.65)
        ax.axis("off")
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold", color="#1a237e")
            if c == -1:
                cell.set_text_props(fontweight="bold", color="#37474f")
            cell.set_edgecolor("#cccccc")

    patch = mpatches.Patch(color="#d4edda", label="Best value per column")
    fig.legend(handles=[patch], loc="lower right", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 1])
    _save(fig, os.path.join(out_dir, "2_convergence_speed.png"))


# ──────────────────────────────────────────────
# Table 3 – Training Dynamics & Epoch Time
# ──────────────────────────────────────────────

def table_training_dynamics(data: dict, out_dir: str):
    models     = list(data.keys())
    strategies = list(data[models[0]]["strategies"].keys())

    fig, axes = plt.subplots(len(models), 1,
                             figsize=(14, 4.2 * len(models)),
                             facecolor="#fafafa")
    fig.suptitle("Table 3 — Training Dynamics\n"
                 "Generalisation Gap, Overfitting, and Epoch Timing",
                 fontsize=14, fontweight="bold", y=1.01, color="#1a237e")

    col_labels = [
        "Gen. Gap Acc\n(final, %)",
        "Gen. Gap Loss\n(final)",
        "Max Gen. Gap\nAcc (%)",
        "Avg Epoch\nTime (s)",
        "Total Train\nTime (s)",
        "Train Loss\n(final)",
        "Test Loss\n(final)",
    ]

    for ax, model_name in zip(axes, models):
        ax.set_title(model_name, fontsize=11, fontweight="bold",
                     color="#263238", pad=8)
        row_labels = [STRATEGY_LABELS[s] for s in strategies]
        cell_data  = []
        for strat in strategies:
            sd   = data[model_name]["strategies"][strat]
            times = sd["times_per_epoch"]
            row  = [
                f"{sd['generalization_gap_acc'][-1]:.4f}",
                f"{sd['generalization_gap_loss'][-1]:.4f}",
                f"{max(sd['generalization_gap_acc']):.4f}",
                f"{sum(times) / len(times):.2f}",
                f"{sum(times):.1f}",
                f"{sd['train_loss'][-1]:.4f}",
                f"{sd['test_loss'][-1]:.4f}",
            ]
            cell_data.append(row)

        n_rows, n_cols = len(strategies), len(col_labels)
        cell_colors = [["#f8f8f8" if r % 2 == 0 else "#ffffff"
                        for _ in range(n_cols)] for r in range(n_rows)]

        # lower gen-gap = better (cols 0,1,2), lower time = better (3,4), lower loss = better (5,6)
        for ci, mode in [(0,"min"),(1,"min"),(2,"min"),(3,"min"),(4,"min"),(5,"min"),(6,"min")]:
            vals = []
            for ri in range(n_rows):
                try:
                    vals.append(float(cell_data[ri][ci]))
                except ValueError:
                    vals.append(None)
            best = _best_idx(vals, mode)
            if best is not None:
                cell_colors[best][ci] = "#d4edda"

        tbl = ax.table(
            cellText=cell_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellColours=cell_colors,
            rowColours=["#e8eaf6"] * n_rows,
            colColours=["#c5cae9"] * n_cols,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.65)
        ax.axis("off")
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold", color="#1a237e")
            if c == -1:
                cell.set_text_props(fontweight="bold", color="#37474f")
            cell.set_edgecolor("#cccccc")

    patch = mpatches.Patch(color="#d4edda", label="Best (lowest) value per column")
    fig.legend(handles=[patch], loc="lower right", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 1])
    _save(fig, os.path.join(out_dir, "3_training_dynamics.png"))


# ──────────────────────────────────────────────
# Table 4 – Learning Rate Schedule Summary
# ──────────────────────────────────────────────

def table_lr_schedule(data: dict, out_dir: str):
    models     = list(data.keys())
    strategies = list(data[models[0]]["strategies"].keys())

    col_labels = [
        "LR Type",
        "Initial LR",
        "Min LR",
        "Max LR",
        "Final LR",
        "LR Range\n(max − min)",
        "Adaptive?",
    ]

    lr_types = {
        "constant":          "Fixed",
        "momentum":          "Fixed + Momentum",
        "StepLR":            "Step Decay",
        "ExponentialLR":     "Exponential Decay",
        "CosineAnnealingLR": "Cosine Annealing",
        "armijo":            "Armijo Line Search",
    }
    adaptive = {
        "constant": "No", "momentum": "No", "StepLR": "No",
        "ExponentialLR": "No", "CosineAnnealingLR": "No", "armijo": "Yes",
    }

    # Use MLP as representative (LR schedules same across models)
    rep = models[0]
    row_labels = [STRATEGY_LABELS[s] for s in strategies]
    cell_data  = []
    for strat in strategies:
        lrs = data[rep]["strategies"][strat]["learning_rates"]
        init_lr  = lrs[0]
        min_lr   = min(lrs)
        max_lr   = max(lrs)
        final_lr = lrs[-1]
        row = [
            lr_types.get(strat, strat),
            f"{init_lr:.6f}",
            f"{min_lr:.6f}",
            f"{max_lr:.6f}",
            f"{final_lr:.6f}",
            f"{max_lr - min_lr:.6f}",
            adaptive.get(strat, "—"),
        ]
        cell_data.append(row)

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#fafafa")
    fig.suptitle("Table 4 — Learning Rate Schedule Characteristics\n"
                 "(LR values recorded from MLP training; schedules are strategy-level, not model-level)",
                 fontsize=13, fontweight="bold", color="#1a237e")

    n_rows, n_cols = len(strategies), len(col_labels)
    cell_colors = [["#f8f8f8" if r % 2 == 0 else "#ffffff"
                    for _ in range(n_cols)] for r in range(n_rows)]

    # Colour adaptive column
    for ri, strat in enumerate(strategies):
        if adaptive[strat] == "Yes":
            cell_colors[ri][6] = "#fff9c4"

    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=["#e8eaf6"] * n_rows,
        colColours=["#c5cae9"] * n_cols,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.9)
    ax.axis("off")
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold", color="#1a237e")
        if c == -1:
            cell.set_text_props(fontweight="bold", color="#37474f")
        cell.set_edgecolor("#cccccc")

    patches = [
        mpatches.Patch(color="#fff9c4", label="Adaptive LR (Armijo line search)"),
    ]
    fig.legend(handles=patches, loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "4_learning_rate_schedule_summary.png"))


# ──────────────────────────────────────────────
# Table 5 – Strategy Ranking per Model
# ──────────────────────────────────────────────

def table_strategy_ranking(data: dict, out_dir: str):
    models     = list(data.keys())
    strategies = list(data[models[0]]["strategies"].keys())

    # Rank strategies per model across 4 key metrics (test_acc, f1, test_loss, gen_gap_acc)
    # Score = sum of ranks (lower = better)
    scoring_metrics = [
        ("Test Acc",       "test_acc",              "max"),
        ("F1 Score",       "f1_score",               "max"),
        ("Test Loss",      "test_loss",              "min"),
        ("Gen. Gap Acc",   "generalization_gap_acc", "min"),
        ("Avg Epoch Time", "times_per_epoch",        "min"),
    ]

    fig, axes = plt.subplots(len(models), 1,
                             figsize=(14, 4.5 * len(models)),
                             facecolor="#fafafa")
    fig.suptitle("Table 5 — Strategy Ranking per Model\n"
                 "(Rank 1 = best; composite score = sum of individual ranks across all metrics)",
                 fontsize=14, fontweight="bold", y=1.01, color="#1a237e")

    for ax, model_name in zip(axes, models):
        ax.set_title(model_name, fontsize=11, fontweight="bold",
                     color="#263238", pad=8)

        # Gather metric values
        metric_vals = {}
        for mname, mkey, mmode in scoring_metrics:
            vals = {}
            for strat in strategies:
                sd = data[model_name]["strategies"][strat]
                if mkey == "times_per_epoch":
                    v = sum(sd[mkey]) / len(sd[mkey])
                else:
                    v = sd[mkey][-1]
                vals[strat] = v
            metric_vals[mname] = (vals, mmode)

        # Rank each strategy per metric
        ranks = {strat: [] for strat in strategies}
        for mname, (vals, mmode) in metric_vals.items():
            sorted_strats = sorted(vals.keys(),
                                   key=lambda s: vals[s],
                                   reverse=(mmode == "max"))
            for rank_pos, strat in enumerate(sorted_strats, start=1):
                ranks[strat].append(rank_pos)

        # Build table: columns = metrics + composite + overall rank
        col_labels = [m[0] for m in scoring_metrics] + ["Composite\nScore", "Overall\nRank"]
        row_data   = []
        composite  = {}
        for strat in strategies:
            r_list = ranks[strat]
            comp   = sum(r_list)
            composite[strat] = comp
            row_data.append((strat, r_list, comp))

        # Sort by composite score for overall rank
        sorted_by_comp = sorted(composite.keys(), key=lambda s: composite[s])
        overall_rank   = {strat: i + 1 for i, strat in enumerate(sorted_by_comp)}

        row_labels = [STRATEGY_LABELS[s] for s in strategies]
        cell_data  = []
        for strat in strategies:
            r_list = ranks[strat]
            comp   = composite[strat]
            ov_rank = overall_rank[strat]
            cell_data.append([str(r) for r in r_list] + [str(comp), str(ov_rank)])

        n_rows, n_cols = len(strategies), len(col_labels)
        cell_colors = [["#f8f8f8" if r % 2 == 0 else "#ffffff"
                        for _ in range(n_cols)] for r in range(n_rows)]

        # colour rank-1 cells (individual metrics) green
        for ci in range(len(scoring_metrics)):
            for ri in range(n_rows):
                if cell_data[ri][ci] == "1":
                    cell_colors[ri][ci] = "#d4edda"

        # colour best composite
        comp_vals = [int(cell_data[ri][-2]) for ri in range(n_rows)]
        best_comp = _best_idx(comp_vals, "min")
        if best_comp is not None:
            cell_colors[best_comp][-2] = "#d4edda"
            cell_colors[best_comp][-1] = "#d4edda"

        tbl = ax.table(
            cellText=cell_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellColours=cell_colors,
            rowColours=["#e8eaf6"] * n_rows,
            colColours=["#c5cae9"] * n_cols,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.65)
        ax.axis("off")
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold", color="#1a237e")
            if c == -1:
                cell.set_text_props(fontweight="bold", color="#37474f")
            cell.set_edgecolor("#cccccc")

    patch = mpatches.Patch(color="#d4edda", label="Best (rank 1) per column")
    fig.legend(handles=[patch], loc="lower right", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 1])
    _save(fig, os.path.join(out_dir, "5_best_strategy_ranking.png"))


# ──────────────────────────────────────────────
# Table 6 – Cross-Model Architecture Comparison
# ──────────────────────────────────────────────

def table_cross_model(data: dict, out_dir: str):
    """Best strategy result per model, plus parameter efficiency."""
    models     = list(data.keys())
    strategies = list(data[models[0]]["strategies"].keys())

    col_labels = [
        "Parameters",
        "Best Strategy",
        "Best Test\nAcc (%)",
        "Best F1",
        "Best Test\nLoss",
        "Params /\nTest Acc",
        "Avg Epoch\nTime (s)\n@ Best Strat",
    ]

    row_labels = models
    cell_data  = []
    for model_name in models:
        params = data[model_name]["total_parameters"]
        best_strat = max(
            strategies,
            key=lambda s: data[model_name]["strategies"][s]["test_acc"][-1]
        )
        sd = data[model_name]["strategies"][best_strat]
        best_acc  = sd["test_acc"][-1]
        best_f1   = sd["f1_score"][-1]
        best_loss = sd["test_loss"][-1]
        efficiency = params / best_acc
        avg_time  = sum(sd["times_per_epoch"]) / len(sd["times_per_epoch"])
        cell_data.append([
            f"{params:,}",
            STRATEGY_LABELS[best_strat],
            f"{best_acc:.2f}",
            f"{best_f1:.4f}",
            f"{best_loss:.4f}",
            f"{efficiency:,.0f}",
            f"{avg_time:.2f}",
        ])

    n_rows, n_cols = len(models), len(col_labels)
    cell_colors = [["#f8f8f8" if r % 2 == 0 else "#ffffff"
                    for _ in range(n_cols)] for r in range(n_rows)]

    for ci, mode in [(2,"max"),(3,"max"),(4,"min"),(5,"min"),(6,"min")]:
        vals = []
        for ri in range(n_rows):
            try:
                vals.append(float(cell_data[ri][ci].replace(",","")))
            except ValueError:
                vals.append(None)
        best = _best_idx(vals, mode)
        if best is not None:
            cell_colors[best][ci] = "#d4edda"

    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor="#fafafa")
    fig.suptitle("Table 6 — Cross-Architecture Comparison\n"
                 "Best strategy result and parameter efficiency per model",
                 fontsize=13, fontweight="bold", color="#1a237e")

    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=["#e8eaf6"] * n_rows,
        colColours=["#c5cae9"] * n_cols,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)
    ax.axis("off")
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold", color="#1a237e")
        if c == -1:
            cell.set_text_props(fontweight="bold", color="#37474f")
        cell.set_edgecolor("#cccccc")

    patch = mpatches.Patch(color="#d4edda", label="Best value per column")
    fig.legend(handles=[patch], loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "6_cross_model_comparison.png"))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate PNG table reports from MNIST step-size experiment JSON."
    )
    parser.add_argument("--input",      required=True, help="Path to experiment JSON file")
    parser.add_argument("--output_dir", required=True, help="Directory to save PNG tables")
    args = parser.parse_args()

    # Load data
    print(f"\n📂 Loading: {args.input}")
    with open(args.input) as f:
        data = json.load(f)

    # Ensure output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {out_dir.resolve()}\n")

    # Generate all tables
    print("Generating tables …")
    table_final_performance(data, str(out_dir))
    table_convergence_speed(data, str(out_dir))
    table_training_dynamics(data, str(out_dir))
    table_lr_schedule(data, str(out_dir))
    table_strategy_ranking(data, str(out_dir))
    table_cross_model(data, str(out_dir))

    print(f"\n✅ All 6 tables saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
