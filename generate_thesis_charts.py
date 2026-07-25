"""
generate_thesis_charts.py
==========================
Generates publication-quality charts for a thesis on step-size strategies
in deep learning, based on MNIST experiment results.

Usage:
    python generate_thesis_charts.py --input <experiment.json> --output_dir <dir/>

Charts produced (12 total):
    01_learning_curves_test_acc.png         – Test accuracy over epochs (all models × strategies)
    02_learning_curves_train_test_loss.png  – Train vs test loss curves per model (best strategy)
    03_learning_rate_schedules.png          – LR schedule trajectories (one representative model)
    04_final_test_accuracy_bar.png          – Grouped bar: final test accuracy per model × strategy
    05_convergence_speed_heatmap.png        – Heatmap: epochs to reach 98 % threshold
    06_generalization_gap_curves.png        – Generalisation gap over epochs (all models)
    07_training_time_comparison.png         – Avg epoch time grouped bar + total time
    08_confusion_matrix_grid.png            – Final-epoch confusion matrices (CNN, all strategies)
    09_per_class_accuracy_radar.png         – Radar chart: per-class accuracy (best strategy per model)
    10_precision_recall_f1_bar.png          – Grouped bar: Precision / Recall / F1 at final epoch
    11_overfitting_scatter.png              – Scatter: test acc vs generalisation gap (all runs)
    12_parameter_efficiency.png             – Parameters vs test accuracy bubble chart
"""

import argparse
import json
import os
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── Aesthetics ────────────────────────────────────────────────────────────────

STRATEGY_ORDER = ["constant", "momentum", "StepLR", "ExponentialLR",
                  "CosineAnnealingLR", "armijo"]

STRATEGY_LABELS = {
    "constant":          "Constant LR",
    "momentum":          "Momentum SGD",
    "StepLR":            "Step LR",
    "ExponentialLR":     "Exponential LR",
    "CosineAnnealingLR": "Cosine Annealing",
    "armijo":            "Armijo (line search)",
}

STRATEGY_COLORS = {
    "constant":          "#2c7bb6",
    "momentum":          "#d7191c",
    "StepLR":            "#1a9641",
    "ExponentialLR":     "#ff7f00",
    "CosineAnnealingLR": "#984ea3",
    "armijo":            "#795548",
}

STRATEGY_DASH = {
    "constant":          (None, None),
    "momentum":          (6, 2),
    "StepLR":            (4, 1, 1, 1),
    "ExponentialLR":     (3, 1),
    "CosineAnnealingLR": (8, 2),
    "armijo":            (2, 2),
}

MODEL_COLORS = {
    "MLP":        "#4C72B0",
    "DeepMLP":    "#C44E52",
    "DeepMLPBN":  "#55A868",
    "CNN":        "#8172B2",
}

MODEL_MARKERS = {"MLP": "o", "DeepMLP": "s", "DeepMLPBN": "^", "CNN": "D"}

DIGITS = list(range(10))

# ── Style ─────────────────────────────────────────────────────────────────────

THESIS_RC = {
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.labelsize":      11,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,
    "figure.dpi":          150,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.3,
    "grid.linestyle":      "--",
    "lines.linewidth":     1.8,
    "figure.facecolor":    "white",
    "axes.facecolor":      "#fafafa",
    "savefig.bbox":        "tight",
    "savefig.dpi":         200,
    "savefig.facecolor":   "white",
}

def apply_style():
    plt.rcParams.update(THESIS_RC)

def _linestyle(strat):
    d = STRATEGY_DASH[strat]
    if d[0] is None:
        return "solid"
    return (0, d)

def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  ✔  {os.path.basename(path)}")
    plt.close(fig)


# ── Data helpers ──────────────────────────────────────────────────────────────

def epochs_arr(data, model, strat, key):
    return np.array(data[model]["strategies"][strat][key])

def final(data, model, strat, key):
    return data[model]["strategies"][strat][key][-1]

def best_strategy(data, model, metric="test_acc", mode="max"):
    strats = STRATEGY_ORDER
    vals = {s: data[model]["strategies"][s][metric][-1] for s in strats}
    fn = max if mode == "max" else min
    return fn(strats, key=lambda s: vals[s])


# ══════════════════════════════════════════════════════════════════════════════
# Chart 01 – Learning Curves: Test Accuracy
# ══════════════════════════════════════════════════════════════════════════════

def chart_learning_curves_acc(data, out_dir):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    axes = axes.flatten()
    epochs = np.arange(1, 31)

    for ax, model in zip(axes, models):
        for strat in STRATEGY_ORDER:
            acc = epochs_arr(data, model, strat, "test_acc")
            ax.plot(epochs, acc,
                    color=STRATEGY_COLORS[strat],
                    linestyle=_linestyle(strat),
                    label=STRATEGY_LABELS[strat], linewidth=1.9)

        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_xlim(1, 30)
        ymin = min(epochs_arr(data, model, s, "test_acc").min() for s in STRATEGY_ORDER)
        ax.set_ylim(max(ymin - 1, 85), 100)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    handles = [mpatches.Patch(color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s])
               for s in STRATEGY_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=True, edgecolor="#cccccc")
    fig.suptitle("Figure 1: Test Accuracy Learning Curves\nAll Models × Step-Size Strategies (MNIST, 30 Epochs)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "01_learning_curves_test_acc.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 02 – Train vs Test Loss (best strategy per model)
# ══════════════════════════════════════════════════════════════════════════════

def chart_train_test_loss(data, out_dir):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    epochs = np.arange(1, 31)

    for ax, model in zip(axes, models):
        for strat in STRATEGY_ORDER:
            tr = epochs_arr(data, model, strat, "train_loss")
            te = epochs_arr(data, model, strat, "test_loss")
            c  = STRATEGY_COLORS[strat]
            ls = _linestyle(strat)
            ax.plot(epochs, tr, color=c, linestyle=ls, alpha=0.5, linewidth=1.4)
            ax.plot(epochs, te, color=c, linestyle=ls, linewidth=2.0,
                    label=STRATEGY_LABELS[strat])

        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xlim(1, 30)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

        # Annotation: solid = test, faint = train
        ax.annotate("Solid = test loss\nFaded = train loss",
                    xy=(0.97, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))

    handles = [mpatches.Patch(color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s])
               for s in STRATEGY_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=True, edgecolor="#cccccc")
    fig.suptitle("Figure 2: Train vs Test Loss Curves\n(Solid = test, faded = train)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "02_learning_curves_train_test_loss.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 03 – Learning Rate Schedules
# ══════════════════════════════════════════════════════════════════════════════

def chart_lr_schedules(data, out_dir):
    model = "MLP"   # LR schedules are strategy-level; any model is representative
    epochs = np.arange(1, 31)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, strat in zip(axes, STRATEGY_ORDER):
        lrs = np.array(data[model]["strategies"][strat]["learning_rates"])
        ax.plot(epochs, lrs, color=STRATEGY_COLORS[strat],
                linewidth=2.2, marker="o", markersize=3)
        ax.set_title(STRATEGY_LABELS[strat], fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_xlim(1, 30)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

        # Annotate min/max
        ax.annotate(f"min: {lrs.min():.2e}", xy=(0.03, 0.07),
                    xycoords="axes fraction", fontsize=8, color="#555")
        ax.annotate(f"max: {lrs.max():.2e}", xy=(0.03, 0.15),
                    xycoords="axes fraction", fontsize=8, color="#555")

    fig.suptitle("Figure 3: Learning Rate Schedule Trajectories over 30 Epochs\n(Recorded from MLP training run)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "03_learning_rate_schedules.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 04 – Final Test Accuracy: Grouped Bar
# ══════════════════════════════════════════════════════════════════════════════

def chart_final_accuracy_bar(data, out_dir):
    models    = list(data.keys())
    n_models  = len(models)
    n_strats  = len(STRATEGY_ORDER)
    x         = np.arange(n_models)
    width     = 0.13
    offsets   = np.linspace(-(n_strats - 1) / 2, (n_strats - 1) / 2, n_strats) * width

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, strat in enumerate(STRATEGY_ORDER):
        vals = [final(data, m, strat, "test_acc") for m in models]
        bars = ax.bar(x + offsets[i], vals,
                      width=width * 0.9,
                      color=STRATEGY_COLORS[strat],
                      label=STRATEGY_LABELS[strat],
                      zorder=3, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.04,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(95, 100.2)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.set_title("Figure 4: Final Test Accuracy by Model and Step-Size Strategy (Epoch 30)",
                 fontweight="bold", pad=10)
    ax.legend(loc="lower right", ncol=2, frameon=True, edgecolor="#cccccc")
    ax.axhline(99, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(98, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.text(n_models - 0.05, 99.05, "99 %", fontsize=8, color="grey", ha="right")
    ax.text(n_models - 0.05, 98.05, "98 %", fontsize=8, color="grey", ha="right")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "04_final_test_accuracy_bar.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 05 – Convergence Speed Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def chart_convergence_heatmap(data, out_dir):
    models    = list(data.keys())
    threshold = 98.0

    matrix = []
    for strat in STRATEGY_ORDER:
        row = []
        for model in models:
            accs  = data[model]["strategies"][strat]["test_acc"]
            epoch = next((i + 1 for i, a in enumerate(accs) if a >= threshold), 31)
            row.append(epoch)
        matrix.append(row)

    matrix = np.array(matrix, dtype=float)
    # Replace 31 (never reached) with NaN for display
    never_mask = matrix >= 31

    cmap = LinearSegmentedColormap.from_list(
        "conv", ["#1a9641", "#ffffbf", "#d7191c"], N=256)
    cmap.set_bad("#dddddd")

    display_matrix = matrix.copy()
    display_matrix[never_mask] = np.nan

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(display_matrix, cmap=cmap, aspect="auto",
                   vmin=1, vmax=30)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=11)
    ax.set_yticks(range(len(STRATEGY_ORDER)))
    ax.set_yticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER], fontsize=10)
    ax.set_xlabel("Model Architecture")
    ax.set_ylabel("Step-Size Strategy")

    for i in range(len(STRATEGY_ORDER)):
        for j in range(len(models)):
            val = matrix[i, j]
            txt = "—" if val >= 31 else str(int(val))
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if (val < 8 or val >= 31) else "#222")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Epoch", fontsize=10)

    ax.set_title(f"Figure 5: Convergence Speed — Epoch First Reaching {threshold:.0f}% Test Accuracy\n"
                 "(Grey = never reached within 30 epochs)",
                 fontweight="bold", pad=10)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "05_convergence_speed_heatmap.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 06 – Generalisation Gap over Epochs
# ══════════════════════════════════════════════════════════════════════════════

def chart_generalization_gap(data, out_dir):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    axes = axes.flatten()
    epochs = np.arange(1, 31)

    for ax, model in zip(axes, models):
        for strat in STRATEGY_ORDER:
            gap = epochs_arr(data, model, strat, "generalization_gap_acc")
            ax.plot(epochs, gap,
                    color=STRATEGY_COLORS[strat],
                    linestyle=_linestyle(strat),
                    label=STRATEGY_LABELS[strat], linewidth=1.9)

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Acc − Test Acc (pp)")
        ax.set_xlim(1, 30)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    handles = [mpatches.Patch(color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s])
               for s in STRATEGY_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=True, edgecolor="#cccccc")
    fig.suptitle("Figure 6: Generalisation Gap (Train − Test Accuracy) over Training\n"
                 "Higher values indicate greater overfitting",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "06_generalization_gap_curves.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 07 – Training Time Comparison
# ══════════════════════════════════════════════════════════════════════════════

def chart_training_time(data, out_dir):
    models   = list(data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    n_models = len(models)
    n_strats = len(STRATEGY_ORDER)
    x        = np.arange(n_models)
    width    = 0.13
    offsets  = np.linspace(-(n_strats - 1) / 2, (n_strats - 1) / 2, n_strats) * width

    # Avg epoch time
    ax = axes[0]
    for i, strat in enumerate(STRATEGY_ORDER):
        vals = []
        for model in models:
            t = data[model]["strategies"][strat]["times_per_epoch"]
            vals.append(sum(t) / len(t))
        ax.bar(x + offsets[i], vals, width=width * 0.9,
               color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat],
               zorder=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Average Epoch Time (s)")
    ax.set_title("Average Epoch Time", fontweight="bold")
    ax.legend(fontsize=8, ncol=1, loc="upper left",
              frameon=True, edgecolor="#cccccc")

    # Total training time (30 epochs)
    ax2 = axes[1]
    for i, strat in enumerate(STRATEGY_ORDER):
        vals = []
        for model in models:
            t = data[model]["strategies"][strat]["times_per_epoch"]
            vals.append(sum(t) / 60)   # convert to minutes
        ax2.bar(x + offsets[i], vals, width=width * 0.9,
                color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat],
                zorder=3, edgecolor="white", linewidth=0.5)

    ax2.set_xticks(x); ax2.set_xticklabels(models)
    ax2.set_ylabel("Total Training Time (min)")
    ax2.set_title("Total Training Time (30 Epochs)", fontweight="bold")

    fig.suptitle("Figure 7: Computational Cost Comparison by Strategy and Architecture",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "07_training_time_comparison.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 08 – Confusion Matrix Grid (CNN, final epoch)
# ══════════════════════════════════════════════════════════════════════════════

def chart_confusion_matrices(data, out_dir):
    model  = "CNN"
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes   = axes.flatten()

    cmap = LinearSegmentedColormap.from_list("cm", ["#ffffff", "#1a6faf"], N=256)

    for ax, strat in zip(axes, STRATEGY_ORDER):
        cm = np.array(data[model]["strategies"][strat]["test_confusion_matrices"][-1])
        # Normalise row-wise (true label)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{cm[i, j]}",
                        ha="center", va="center",
                        fontsize=7,
                        color="white" if cm_norm[i, j] > 0.6 else "#222")
        ax.set_title(STRATEGY_LABELS[strat], fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(DIGITS, fontsize=8)
        ax.set_yticklabels(DIGITS, fontsize=8)
        ax.grid(False)

    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="Row-normalised rate")
    fig.suptitle(f"Figure 8: Confusion Matrices — {model} (Final Epoch, Test Set)\n"
                 "Values show raw counts; colour shows row-normalised accuracy",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "08_confusion_matrix_grid.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 09 – Per-Class Accuracy Radar
# ══════════════════════════════════════════════════════════════════════════════

def chart_radar_per_class(data, out_dir):
    """Radar chart: best-strategy per-class accuracy for each model."""
    models   = list(data.keys())
    N        = 10
    angles   = [n / float(N) * 2 * math.pi for n in range(N)]
    angles  += angles[:1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                             subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for ax, model in zip(axes, models):
        best_strat = best_strategy(data, model)

        # All strategies faint
        for strat in STRATEGY_ORDER:
            pca = data[model]["strategies"][strat]["per_class_accuracy"][-1]
            vals = list(pca) + [pca[0]]
            ax.plot(angles, vals, color=STRATEGY_COLORS[strat],
                    alpha=0.25, linewidth=1)

        # Best strategy bold
        pca  = data[model]["strategies"][best_strat]["per_class_accuracy"][-1]
        vals = list(pca) + [pca[0]]
        ax.plot(angles, vals, color=STRATEGY_COLORS[best_strat],
                linewidth=2.5, label=STRATEGY_LABELS[best_strat])
        ax.fill(angles, vals, color=STRATEGY_COLORS[best_strat], alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"Digit {d}" for d in DIGITS], fontsize=9)
        ax.set_ylim(0.93, 1.0)
        ax.set_yticks([0.94, 0.96, 0.98, 1.0])
        ax.set_yticklabels(["94 %", "96 %", "98 %", "100 %"], fontsize=7)
        ax.set_title(f"{model}\nBest: {STRATEGY_LABELS[best_strat]}",
                     fontweight="bold", pad=18, fontsize=10)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                  fontsize=8, frameon=True)

    # Shared faint-lines legend
    handles = [mpatches.Patch(color=STRATEGY_COLORS[s], alpha=0.4,
                               label=STRATEGY_LABELS[s])
               for s in STRATEGY_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), fontsize=9,
               frameon=True, edgecolor="#cccccc",
               title="All strategies (faded) | Bold = best strategy")
    fig.suptitle("Figure 9: Per-Class Test Accuracy — Radar Chart\n"
                 "Bold line = best-performing strategy per model",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "09_per_class_accuracy_radar.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 10 – Precision / Recall / F1 Grouped Bar
# ══════════════════════════════════════════════════════════════════════════════

def chart_precision_recall_f1(data, out_dir):
    models  = list(data.keys())
    metrics = [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1_score")]
    n_m     = len(metrics)
    x       = np.arange(len(STRATEGY_ORDER))
    width   = 0.22

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    metric_colors = ["#2c7bb6", "#1a9641", "#d7191c"]

    for ax, model in zip(axes, models):
        for mi, (mlabel, mkey) in enumerate(metrics):
            vals = [final(data, model, s, mkey) for s in STRATEGY_ORDER]
            offset = (mi - 1) * width
            bars = ax.bar(x + offset, vals, width=width * 0.9,
                          color=metric_colors[mi], label=mlabel,
                          zorder=3, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER],
                           rotation=20, ha="right", fontsize=8.5)
        ax.set_ylabel("Score")
        ax.set_ylim(0.96, 1.001)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.set_title(model, fontweight="bold")
        if ax is axes[0]:
            ax.legend(fontsize=9, loc="lower right",
                      frameon=True, edgecolor="#cccccc")

    fig.suptitle("Figure 10: Precision, Recall, and F1 Score at Final Epoch\nAll Models × Step-Size Strategies",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "10_precision_recall_f1_bar.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 11 – Overfitting Scatter: Test Acc vs Gen Gap
# ══════════════════════════════════════════════════════════════════════════════

def chart_overfitting_scatter(data, out_dir):
    models = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 7))

    for model in models:
        for strat in STRATEGY_ORDER:
            x_val = final(data, model, strat, "test_acc")
            y_val = final(data, model, strat, "generalization_gap_acc")
            ax.scatter(x_val, y_val,
                       color=STRATEGY_COLORS[strat],
                       marker=MODEL_MARKERS[model],
                       s=90, zorder=5,
                       edgecolors="white", linewidth=0.8)
            ax.annotate(f"{model[:1]}+{strat[:2]}",
                        (x_val, y_val),
                        textcoords="offset points",
                        xytext=(5, 3), fontsize=6.5, color="#444")

    # Legend: strategies (colour) + models (shape)
    strat_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=STRATEGY_COLORS[s],
                   markersize=9, label=STRATEGY_LABELS[s])
        for s in STRATEGY_ORDER
    ]
    model_handles = [
        plt.Line2D([0], [0], marker=MODEL_MARKERS[m], color="w",
                   markerfacecolor="#555", markersize=9, label=m)
        for m in models
    ]
    leg1 = ax.legend(handles=strat_handles, loc="upper left",
                     title="Strategy (colour)", fontsize=8,
                     frameon=True, edgecolor="#cccccc")
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, loc="lower right",
              title="Model (shape)", fontsize=8,
              frameon=True, edgecolor="#cccccc")

    ax.set_xlabel("Final Test Accuracy (%)")
    ax.set_ylabel("Generalisation Gap — Train − Test Acc (pp)")
    ax.set_title("Figure 11: Overfitting Profile — Test Accuracy vs Generalisation Gap\n"
                 "Bottom-right corner = best (high accuracy, low gap)",
                 fontweight="bold", pad=10)
    # Arrow annotation
    ax.annotate("← Less overfitting\n← Higher accuracy →",
                xy=(0.97, 0.03), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8, color="#555",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.8))
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "11_overfitting_scatter.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 12 – Parameter Efficiency Bubble Chart
# ══════════════════════════════════════════════════════════════════════════════

def chart_parameter_efficiency(data, out_dir):
    models = list(data.keys())
    fig, ax = plt.subplots(figsize=(11, 7))

    for model in models:
        params = data[model]["total_parameters"]
        for strat in STRATEGY_ORDER:
            acc      = final(data, model, strat, "test_acc")
            avg_time = (sum(data[model]["strategies"][strat]["times_per_epoch"]) /
                        len(data[model]["strategies"][strat]["times_per_epoch"]))
            bubble   = avg_time * 12   # size proportional to epoch time
            ax.scatter(params / 1e6, acc,
                       s=bubble,
                       color=STRATEGY_COLORS[strat],
                       marker=MODEL_MARKERS[model],
                       alpha=0.75, zorder=5,
                       edgecolors="white", linewidth=1.0)

    # Add model labels at centroid
    for model in models:
        params = data[model]["total_parameters"]
        accs   = [final(data, model, s, "test_acc") for s in STRATEGY_ORDER]
        ax.annotate(model,
                    xy=(params / 1e6, np.mean(accs)),
                    fontsize=10, fontweight="bold",
                    color=MODEL_COLORS[model],
                    xytext=(6, 4), textcoords="offset points")

    strat_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=STRATEGY_COLORS[s],
                   markersize=9, label=STRATEGY_LABELS[s])
        for s in STRATEGY_ORDER
    ]
    model_handles = [
        plt.Line2D([0], [0], marker=MODEL_MARKERS[m], color="w",
                   markerfacecolor="#555", markersize=9, label=m)
        for m in models
    ]
    size_handle = plt.scatter([], [], s=100, color="#aaa", label="Bubble ∝ avg epoch time")

    leg1 = ax.legend(handles=strat_handles, loc="lower right",
                     title="Strategy", fontsize=8,
                     frameon=True, edgecolor="#cccccc")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=model_handles + [size_handle],
                     loc="lower left", title="Model + time",
                     fontsize=8, frameon=True, edgecolor="#cccccc")

    ax.set_xlabel("Number of Parameters (Millions)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Figure 12: Parameter Efficiency — Accuracy vs Model Size\n"
                 "Bubble size ∝ average epoch training time",
                 fontweight="bold", pad=10)
    ax.set_ylim(97, 100)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "12_parameter_efficiency.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis-quality charts from MNIST step-size experiment JSON."
    )
    parser.add_argument("--input",      required=True,
                        help="Path to the experiment JSON file")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save PNG charts")
    args = parser.parse_args()

    apply_style()

    print(f"\n📂 Loading data: {args.input}")
    with open(args.input) as f:
        data = json.load(f)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output dir  : {out.resolve()}")
    print(f"\nGenerating 12 thesis charts …\n")

    chart_learning_curves_acc(data,   str(out))
    chart_train_test_loss(data,       str(out))
    chart_lr_schedules(data,          str(out))
    chart_final_accuracy_bar(data,    str(out))
    chart_convergence_heatmap(data,   str(out))
    chart_generalization_gap(data,    str(out))
    chart_training_time(data,         str(out))
    chart_confusion_matrices(data,    str(out))
    chart_radar_per_class(data,       str(out))
    chart_precision_recall_f1(data,   str(out))
    chart_overfitting_scatter(data,   str(out))
    chart_parameter_efficiency(data,  str(out))

    print(f"\n✅  All 12 charts saved to: {out.resolve()}\n")


if __name__ == "__main__":
    main()
