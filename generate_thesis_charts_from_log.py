"""
generate_thesis_charts_from_log.py
====================================
Parses the MNIST step-size experiment .log file and produces
publication-quality charts for a thesis paper.

Usage:
    python generate_thesis_charts_from_log.py \
        --input  mnist_full_experiment_13.log \
        --output_dir ./thesis_charts/

Log format (one line per epoch):
    <strategy> | Epoch <n> | Train Loss: <v> | Train Acc: <v>% |
    Test Loss: <v> | Test Acc: <v>% | F1: <v> | GradNorm: <v> |
    UpdateNorm: <v> | LR: <v> | Epoch Time: <v>s

Model / parameter blocks:
    Training model: <name>
    Total trainable parameters: <n>
    Starting strategy: <name>

Charts produced (14 total):
    01  Test accuracy learning curves          (all models × strategies)
    02  Train vs test loss curves              (all models × strategies)
    03  Learning rate schedule trajectories    (all strategies, one panel each)
    04  Final test accuracy grouped bar        (models × strategies)
    05  Convergence speed heatmap              (epochs to reach 98 %)
    06  Generalisation gap curves              (train − test acc, all models)
    07  Training time comparison               (avg + total)
    08  Gradient norm dynamics                 (GradNorm over epochs) ← log-exclusive
    09  Update norm dynamics                   (UpdateNorm over epochs) ← log-exclusive
    10  GradNorm vs UpdateNorm scatter         (all epochs, all runs) ← log-exclusive
    11  Precision / Recall / F1 bar            (final epoch, all models)
    12  Overfitting scatter                    (test acc vs gen-gap)
    13  LR vs GradNorm dual-axis               (per strategy, CNN panel) ← log-exclusive
    14  Parameter efficiency bubble            (params vs accuracy, bubble = time)
"""

import argparse
import re
import os
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── Palette & style ───────────────────────────────────────────────────────────

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
    "constant":          "solid",
    "momentum":          (6, 2),
    "StepLR":            (4, 1, 1, 1),
    "ExponentialLR":     (3, 1),
    "CosineAnnealingLR": (8, 2),
    "armijo":            (2, 2),
}

MODEL_COLORS  = {"MLP": "#4C72B0", "DeepMLP": "#C44E52",
                 "DeepMLPBN": "#55A868", "CNN": "#8172B2"}
MODEL_MARKERS = {"MLP": "o", "DeepMLP": "s", "DeepMLPBN": "^", "CNN": "D"}

THESIS_RC = {
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "lines.linewidth":    1.8,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#fafafa",
    "savefig.dpi":        200,
    "savefig.facecolor":  "white",
}

def apply_style():
    plt.rcParams.update(THESIS_RC)

def _ls(strat):
    d = STRATEGY_DASH[strat]
    return "solid" if d == "solid" else (0, d)

def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  ✔  {os.path.basename(path)}")
    plt.close(fig)

def _strat_legend(fig, strategies=None):
    if strategies is None:
        strategies = STRATEGY_ORDER
    handles = [mpatches.Patch(color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s])
               for s in strategies]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=True, edgecolor="#cccccc")

def _best_idx(vals, mode="max"):
    valid = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not valid:
        return None
    fn = max if mode == "max" else min
    return fn(valid, key=lambda x: x[1])[0]


# ══════════════════════════════════════════════════════════════════════════════
# Log parser
# ══════════════════════════════════════════════════════════════════════════════

EPOCH_RE = re.compile(
    r"^(?P<strategy>\S+) \| Epoch (?P<epoch>\d+) \| "
    r"Train Loss: (?P<train_loss>[\d.]+) \| "
    r"Train Acc: (?P<train_acc>[\d.]+)% \| "
    r"Test Loss: (?P<test_loss>[\d.]+) \| "
    r"Test Acc: (?P<test_acc>[\d.]+)% \| "
    r"F1: (?P<f1>[\d.]+) \| "
    r"GradNorm: (?P<grad_norm>[\d.]+) \| "
    r"UpdateNorm: (?P<update_norm>[\d.]+) \| "
    r"LR: (?P<lr>[\d.]+) \| "
    r"Epoch Time: (?P<epoch_time>[\d.]+)s"
)
MODEL_RE  = re.compile(r"^Training model:\s+(\S+)")
PARAM_RE  = re.compile(r"^Total trainable parameters:\s+([\d,]+)")
STRAT_RE  = re.compile(r"^Starting strategy:\s+(\S+)")


def parse_log(log_path: str) -> dict:
    """
    Returns:
        {
          model_name: {
            "params": int,
            "strategies": {
              strategy_name: {
                "train_loss": [...], "train_acc": [...],
                "test_loss":  [...], "test_acc":  [...],
                "f1":         [...], "grad_norm": [...],
                "update_norm":[...], "lr":        [...],
                "epoch_time": [...]
              }
            }
          }
        }
    """
    data = {}
    current_model   = None
    current_params  = None
    current_strategy = None

    with open(log_path) as fh:
        for raw in fh:
            line = raw.strip()

            m = MODEL_RE.match(line)
            if m:
                current_model = m.group(1)
                data[current_model] = {"params": None, "strategies": {}}
                current_strategy = None
                continue

            m = PARAM_RE.match(line)
            if m and current_model:
                data[current_model]["params"] = int(m.group(1).replace(",", ""))
                continue

            m = STRAT_RE.match(line)
            if m and current_model:
                current_strategy = m.group(1)
                data[current_model]["strategies"][current_strategy] = {
                    k: [] for k in ("train_loss","train_acc","test_loss","test_acc",
                                    "f1","grad_norm","update_norm","lr","epoch_time")
                }
                continue

            m = EPOCH_RE.match(line)
            if m and current_model and current_strategy:
                sd = data[current_model]["strategies"][current_strategy]
                sd["train_loss"].append(float(m.group("train_loss")))
                sd["train_acc"].append(float(m.group("train_acc")))
                sd["test_loss"].append(float(m.group("test_loss")))
                sd["test_acc"].append(float(m.group("test_acc")))
                sd["f1"].append(float(m.group("f1")))
                sd["grad_norm"].append(float(m.group("grad_norm")))
                sd["update_norm"].append(float(m.group("update_norm")))
                sd["lr"].append(float(m.group("lr")))
                sd["epoch_time"].append(float(m.group("epoch_time")))

    return data


# ── Convenience accessors ─────────────────────────────────────────────────────

def arr(data, model, strat, key):
    return np.array(data[model]["strategies"][strat][key])

def final(data, model, strat, key):
    return data[model]["strategies"][strat][key][-1]

def best_strat(data, model, key="test_acc", mode="max"):
    fn = max if mode == "max" else min
    return fn(STRATEGY_ORDER, key=lambda s: data[model]["strategies"][s][key][-1])


# ══════════════════════════════════════════════════════════════════════════════
# Chart 01 – Test Accuracy Learning Curves
# ══════════════════════════════════════════════════════════════════════════════

def chart01_test_acc(data, out):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    epochs = np.arange(1, 31)

    for ax, model in zip(axes, models):
        for s in STRATEGY_ORDER:
            ax.plot(epochs, arr(data, model, s, "test_acc"),
                    color=STRATEGY_COLORS[s], linestyle=_ls(s),
                    label=STRATEGY_LABELS[s], linewidth=1.9)
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy (%)")
        ax.set_xlim(1, 30)
        ymin = min(arr(data, model, s, "test_acc").min() for s in STRATEGY_ORDER)
        ax.set_ylim(max(ymin - 1, 85), 100.5)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    _strat_legend(fig)
    fig.suptitle("Figure 1 — Test Accuracy Learning Curves\nAll Models × Step-Size Strategies (MNIST, 30 Epochs)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out, "01_test_accuracy_curves.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 02 – Train vs Test Loss
# ══════════════════════════════════════════════════════════════════════════════

def chart02_loss_curves(data, out):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    epochs = np.arange(1, 31)

    for ax, model in zip(axes, models):
        for s in STRATEGY_ORDER:
            c = STRATEGY_COLORS[s]; ls = _ls(s)
            ax.plot(epochs, arr(data, model, s, "train_loss"),
                    color=c, linestyle=ls, alpha=0.4, linewidth=1.4)
            ax.plot(epochs, arr(data, model, s, "test_loss"),
                    color=c, linestyle=ls, linewidth=2.0, label=STRATEGY_LABELS[s])
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_xlim(1, 30)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.annotate("Solid = test | Faded = train",
                    xy=(0.97, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#cccccc", alpha=0.8))

    _strat_legend(fig)
    fig.suptitle("Figure 2 — Train vs Test Loss Curves\n(Solid = test loss, faded = train loss)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out, "02_train_test_loss_curves.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 03 – Learning Rate Schedule Trajectories
# ══════════════════════════════════════════════════════════════════════════════

def chart03_lr_schedules(data, out):
    # Use MLP (LR schedules are strategy-level)
    model  = list(data.keys())[0]
    epochs = np.arange(1, 31)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, s in zip(axes, STRATEGY_ORDER):
        lrs = arr(data, model, s, "lr")
        ax.plot(epochs, lrs, color=STRATEGY_COLORS[s],
                linewidth=2.2, marker="o", markersize=3)
        ax.set_title(STRATEGY_LABELS[s], fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
        ax.set_xlim(1, 30)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.annotate(f"min: {lrs.min():.2e}",
                    xy=(0.04, 0.08), xycoords="axes fraction", fontsize=8, color="#555")
        ax.annotate(f"max: {lrs.max():.2e}",
                    xy=(0.04, 0.17), xycoords="axes fraction", fontsize=8, color="#555")

    fig.suptitle("Figure 3 — Learning Rate Schedule Trajectories over 30 Epochs",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out, "03_lr_schedule_trajectories.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 04 – Final Test Accuracy Grouped Bar
# ══════════════════════════════════════════════════════════════════════════════

def chart04_final_acc_bar(data, out):
    models   = list(data.keys())
    n_strats = len(STRATEGY_ORDER)
    x        = np.arange(len(models))
    width    = 0.13
    offsets  = np.linspace(-(n_strats-1)/2, (n_strats-1)/2, n_strats) * width

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, s in enumerate(STRATEGY_ORDER):
        vals = [final(data, m, s, "test_acc") for m in models]
        bars = ax.bar(x + offsets[i], vals, width=width*0.9,
                      color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s],
                      zorder=3, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(90, 100.8)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    for ref in (98, 99):
        ax.axhline(ref, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.text(len(models) - 0.05, ref + 0.1, f"{ref} %",
                fontsize=8, color="grey", ha="right")
    ax.legend(loc="lower right", ncol=2, frameon=True, edgecolor="#cccccc")
    ax.set_title("Figure 4 — Final Test Accuracy by Model and Step-Size Strategy (Epoch 30)",
                 fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, os.path.join(out, "04_final_test_accuracy_bar.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 05 – Convergence Speed Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def chart05_convergence_heatmap(data, out):
    models    = list(data.keys())
    threshold = 98.0
    matrix    = []

    for s in STRATEGY_ORDER:
        row = []
        for model in models:
            accs  = data[model]["strategies"][s]["test_acc"]
            epoch = next((i+1 for i, a in enumerate(accs) if a >= threshold), 31)
            row.append(float(epoch))
        matrix.append(row)

    matrix     = np.array(matrix)
    never_mask = matrix >= 31
    display    = matrix.copy(); display[never_mask] = np.nan

    cmap = LinearSegmentedColormap.from_list(
        "conv", ["#1a9641", "#ffffbf", "#d7191c"], N=256)
    cmap.set_bad("#cccccc")

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(display, cmap=cmap, aspect="auto", vmin=1, vmax=30)

    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, fontsize=11)
    ax.set_yticks(range(len(STRATEGY_ORDER)))
    ax.set_yticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER], fontsize=10)
    ax.set_xlabel("Model Architecture"); ax.set_ylabel("Step-Size Strategy")

    for i in range(len(STRATEGY_ORDER)):
        for j in range(len(models)):
            v   = matrix[i, j]
            txt = "—" if v >= 31 else str(int(v))
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    fontweight="bold",
                    color="white" if (v < 8 or v >= 31) else "#222")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Epoch", fontsize=10)
    ax.set_title(f"Figure 5 — Convergence Speed: Epoch First Reaching {threshold:.0f}% Test Accuracy\n"
                 "(Grey = threshold never reached within 30 epochs)",
                 fontweight="bold", pad=10)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, os.path.join(out, "05_convergence_speed_heatmap.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 06 – Generalisation Gap Curves
# ══════════════════════════════════════════════════════════════════════════════

def chart06_gen_gap(data, out):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    epochs = np.arange(1, 31)

    for ax, model in zip(axes, models):
        for s in STRATEGY_ORDER:
            gap = arr(data, model, s, "train_acc") - arr(data, model, s, "test_acc")
            ax.plot(epochs, gap, color=STRATEGY_COLORS[s],
                    linestyle=_ls(s), linewidth=1.9, label=STRATEGY_LABELS[s])
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Train − Test Accuracy (pp)")
        ax.set_xlim(1, 30)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    _strat_legend(fig)
    fig.suptitle("Figure 6 — Generalisation Gap (Train − Test Accuracy) over Training\n"
                 "Higher values indicate greater overfitting",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out, "06_generalisation_gap_curves.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 07 – Training Time Comparison
# ══════════════════════════════════════════════════════════════════════════════

def chart07_training_time(data, out):
    models   = list(data.keys())
    n_strats = len(STRATEGY_ORDER)
    x        = np.arange(len(models))
    width    = 0.13
    offsets  = np.linspace(-(n_strats-1)/2, (n_strats-1)/2, n_strats) * width

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for panel, (ax, ylabel, divisor, title) in enumerate(zip(
        axes,
        ["Average Epoch Time (s)", "Total Training Time (min)"],
        [1, 60],
        ["Average Epoch Time", "Total Training Time (30 Epochs)"]
    )):
        for i, s in enumerate(STRATEGY_ORDER):
            vals = [np.mean(data[m]["strategies"][s]["epoch_time"]) / divisor
                    for m in models]
            ax.bar(x + offsets[i], vals, width=width*0.9,
                   color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s],
                   zorder=3, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(models)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        if panel == 0:
            ax.legend(fontsize=8, ncol=1, loc="upper left",
                      frameon=True, edgecolor="#cccccc")

    fig.suptitle("Figure 7 — Computational Cost Comparison by Strategy and Architecture",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out, "07_training_time_comparison.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 08 – Gradient Norm Dynamics  ★ log-exclusive ★
# ══════════════════════════════════════════════════════════════════════════════

def chart08_grad_norm(data, out):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    epochs = np.arange(1, 31)

    for ax, model in zip(axes, models):
        for s in STRATEGY_ORDER:
            gn = arr(data, model, s, "grad_norm")
            ax.plot(epochs, gn, color=STRATEGY_COLORS[s],
                    linestyle=_ls(s), linewidth=1.9, label=STRATEGY_LABELS[s])
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Gradient Norm")
        ax.set_xlim(1, 30)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    _strat_legend(fig)
    fig.suptitle("Figure 8 — Gradient Norm Dynamics over Training (log scale)\n"
                 "Measures the L2 norm of parameter gradients at each epoch",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out, "08_gradient_norm_dynamics.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 09 – Update Norm Dynamics  ★ log-exclusive ★
# ══════════════════════════════════════════════════════════════════════════════

def chart09_update_norm(data, out):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    epochs = np.arange(1, 31)

    for ax, model in zip(axes, models):
        for s in STRATEGY_ORDER:
            un = arr(data, model, s, "update_norm")
            # Avoid log(0) for armijo zero-epochs
            un_safe = np.where(un == 0, np.nan, un)
            ax.plot(epochs, un_safe, color=STRATEGY_COLORS[s],
                    linestyle=_ls(s), linewidth=1.9, label=STRATEGY_LABELS[s])
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Update Norm")
        ax.set_xlim(1, 30)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    _strat_legend(fig)
    fig.suptitle("Figure 9 — Parameter Update Norm Dynamics over Training (log scale)\n"
                 "Measures the effective step magnitude applied to model parameters",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out, "09_update_norm_dynamics.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 10 – GradNorm vs UpdateNorm Scatter  ★ log-exclusive ★
# ══════════════════════════════════════════════════════════════════════════════

def chart10_grad_vs_update_scatter(data, out):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for ax, model in zip(axes, models):
        for s in STRATEGY_ORDER:
            gn = arr(data, model, s, "grad_norm")
            un = arr(data, model, s, "update_norm")
            epochs_idx = np.arange(1, len(gn)+1)

            # Mask exact zeros for log scale
            mask = (gn > 0) & (un > 0)
            sc = ax.scatter(gn[mask], un[mask],
                            c=epochs_idx[mask],
                            cmap="viridis",
                            marker=MODEL_MARKERS[model],
                            alpha=0.7, s=30, zorder=4,
                            label=STRATEGY_LABELS[s])

        # One colourbar per axes
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Epoch", fontsize=8)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Gradient Norm"); ax.set_ylabel("Update Norm")
        ax.set_title(model, fontweight="bold")

    # Strategy legend using colour patches (the scatter uses epoch colour)
    handles = [mpatches.Patch(color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s])
               for s in STRATEGY_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=True, edgecolor="#cccccc",
               title="Strategy — colour encodes epoch (dark=early, bright=late)")
    fig.suptitle("Figure 10 — Gradient Norm vs Parameter Update Norm\n"
                 "(log–log; colour = epoch number, early=dark, late=bright)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out, "10_gradnorm_vs_updatenorm_scatter.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 11 – Precision / Recall / F1 Bar
# ══════════════════════════════════════════════════════════════════════════════

def chart11_prf_bar(data, out):
    models  = list(data.keys())
    # F1 is in the log; Precision & Recall are not — derive from JSON if available,
    # else show F1 only. Here we have F1 from log; show it per strategy per model.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes = axes.flatten()
    x = np.arange(len(STRATEGY_ORDER))
    width = 0.55

    for ax, model in zip(axes, models):
        vals = [final(data, model, s, "f1") for s in STRATEGY_ORDER]
        colors = [STRATEGY_COLORS[s] for s in STRATEGY_ORDER]
        bars = ax.bar(x, vals, width=width, color=colors,
                      zorder=3, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.0003, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER],
                           rotation=18, ha="right", fontsize=8.5)
        ax.set_ylabel("F1 Score")
        ax.set_ylim(min(vals) - 0.005, 1.002)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.set_title(model, fontweight="bold")

        # Mark best bar
        best = int(np.argmax(vals))
        bars[best].set_edgecolor("#333"); bars[best].set_linewidth(2)

    fig.suptitle("Figure 11 — F1 Score at Final Epoch by Model and Strategy\n"
                 "(Bold border = best per model)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out, "11_f1_score_bar.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 12 – Overfitting Scatter: Test Acc vs Gen Gap
# ══════════════════════════════════════════════════════════════════════════════

def chart12_overfitting_scatter(data, out):
    models = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 7))

    for model in models:
        for s in STRATEGY_ORDER:
            x_val = final(data, model, s, "test_acc")
            y_val = (final(data, model, s, "train_acc") -
                     final(data, model, s, "test_acc"))
            ax.scatter(x_val, y_val,
                       color=STRATEGY_COLORS[s],
                       marker=MODEL_MARKERS[model],
                       s=90, zorder=5, edgecolors="white", linewidth=0.8)
            ax.annotate(f"{model[:1]}+{s[:3]}",
                        (x_val, y_val), textcoords="offset points",
                        xytext=(5, 3), fontsize=6.5, color="#444")

    strat_h = [plt.Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=STRATEGY_COLORS[s], markersize=9,
                           label=STRATEGY_LABELS[s]) for s in STRATEGY_ORDER]
    model_h = [plt.Line2D([0],[0], marker=MODEL_MARKERS[m], color="w",
                           markerfacecolor="#555", markersize=9, label=m)
               for m in models]
    leg1 = ax.legend(handles=strat_h, loc="upper left", title="Strategy",
                     fontsize=8, frameon=True, edgecolor="#cccccc")
    ax.add_artist(leg1)
    ax.legend(handles=model_h, loc="lower right", title="Model (shape)",
              fontsize=8, frameon=True, edgecolor="#cccccc")

    ax.set_xlabel("Final Test Accuracy (%)")
    ax.set_ylabel("Generalisation Gap — Train − Test Acc (pp)")
    ax.set_title("Figure 12 — Overfitting Profile: Test Accuracy vs Generalisation Gap\n"
                 "Ideal region = bottom-right (high accuracy, low gap)",
                 fontweight="bold", pad=10)
    ax.annotate("← Lower gap is better\n Higher accuracy →",
                xy=(0.97, 0.03), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8, color="#555",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.8))
    fig.tight_layout()
    _save(fig, os.path.join(out, "12_overfitting_scatter.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 13 – LR vs GradNorm Dual-Axis  ★ log-exclusive ★
# ══════════════════════════════════════════════════════════════════════════════

def chart13_lr_vs_gradnorm(data, out):
    """Dual-axis per strategy: LR on left, GradNorm on right, using CNN."""
    model  = "CNN"
    epochs = np.arange(1, 31)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, s in zip(axes, STRATEGY_ORDER):
        lrs = arr(data, model, s, "lr")
        gn  = arr(data, model, s, "grad_norm")
        color_lr = STRATEGY_COLORS[s]
        color_gn = "#333333"

        ax.plot(epochs, lrs, color=color_lr, linewidth=2.2,
                label="Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate", color=color_lr)
        ax.tick_params(axis="y", labelcolor=color_lr)

        ax2 = ax.twinx()
        ax2.plot(epochs, gn, color=color_gn, linewidth=1.6,
                 linestyle="--", label="Grad Norm")
        ax2.set_ylabel("Gradient Norm", color=color_gn)
        ax2.tick_params(axis="y", labelcolor=color_gn)
        ax2.set_yscale("log")

        ax.set_title(STRATEGY_LABELS[s], fontweight="bold")
        ax.set_xlim(1, 30)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.grid(True, alpha=0.25, linestyle="--")
        ax2.grid(False)

        # Combined legend
        lines  = [matplotlib.lines.Line2D([0],[0], color=color_lr, lw=2, label="LR"),
                  matplotlib.lines.Line2D([0],[0], color=color_gn, lw=1.6,
                                          linestyle="--", label="GradNorm")]
        ax.legend(handles=lines, fontsize=8, loc="upper right",
                  frameon=True, edgecolor="#cccccc")

    fig.suptitle(f"Figure 13 — Learning Rate vs Gradient Norm (Dual-Axis) — {model}\n"
                 "Dashed = gradient norm (log scale, right axis); Solid = LR (left axis)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out, "13_lr_vs_gradnorm_dual_axis.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Chart 14 – Parameter Efficiency Bubble
# ══════════════════════════════════════════════════════════════════════════════

def chart14_param_efficiency(data, out):
    models = list(data.keys())
    fig, ax = plt.subplots(figsize=(11, 7))

    for model in models:
        params = data[model]["params"]
        for s in STRATEGY_ORDER:
            acc      = final(data, model, s, "test_acc")
            avg_time = np.mean(data[model]["strategies"][s]["epoch_time"])
            ax.scatter(params / 1e6, acc,
                       s=avg_time * 12,
                       color=STRATEGY_COLORS[s],
                       marker=MODEL_MARKERS[model],
                       alpha=0.75, zorder=5,
                       edgecolors="white", linewidth=1.0)

    # Model centroid labels
    for model in models:
        params = data[model]["params"]
        accs   = [final(data, model, s, "test_acc") for s in STRATEGY_ORDER]
        ax.annotate(model,
                    xy=(params / 1e6, float(np.mean(accs))),
                    fontsize=10, fontweight="bold", color=MODEL_COLORS[model],
                    xytext=(6, 4), textcoords="offset points")

    strat_h = [plt.Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=STRATEGY_COLORS[s], markersize=9,
                           label=STRATEGY_LABELS[s]) for s in STRATEGY_ORDER]
    model_h = [plt.Line2D([0],[0], marker=MODEL_MARKERS[m], color="w",
                           markerfacecolor="#555", markersize=9, label=m)
               for m in models]
    sz_h    = [plt.scatter([], [], s=100, color="#aaa", label="Bubble ∝ avg epoch time")]

    leg1 = ax.legend(handles=strat_h, loc="lower right", title="Strategy",
                     fontsize=8, frameon=True, edgecolor="#cccccc")
    ax.add_artist(leg1)
    ax.legend(handles=model_h + sz_h, loc="lower left",
              title="Model + time", fontsize=8, frameon=True, edgecolor="#cccccc")

    ax.set_xlabel("Number of Parameters (Millions)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Figure 14 — Parameter Efficiency: Accuracy vs Model Size\n"
                 "Bubble size ∝ average epoch training time",
                 fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, os.path.join(out, "14_parameter_efficiency_bubble.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis-quality charts from MNIST experiment .log file."
    )
    parser.add_argument("--input",      required=True,
                        help="Path to the .log file")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save PNG charts")
    args = parser.parse_args()

    apply_style()

    print(f"\n📂 Parsing log: {args.input}")
    data = parse_log(args.input)

    models     = list(data.keys())
    strategies = [s for m in models
                  for s in data[m]["strategies"]]
    total_runs  = sum(len(data[m]["strategies"]) for m in models)
    total_eps   = sum(len(data[m]["strategies"][s]["test_acc"])
                      for m in models
                      for s in data[m]["strategies"])

    print(f"  Models parsed   : {models}")
    print(f"  Strategies/model: {list(data[models[0]]['strategies'].keys())}")
    print(f"  Total runs      : {total_runs}")
    print(f"  Total epochs    : {total_eps}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output dir: {out.resolve()}")
    print(f"\nGenerating 14 thesis charts …\n")

    chart01_test_acc(data,               str(out))
    chart02_loss_curves(data,            str(out))
    chart03_lr_schedules(data,           str(out))
    chart04_final_acc_bar(data,          str(out))
    chart05_convergence_heatmap(data,    str(out))
    chart06_gen_gap(data,                str(out))
    chart07_training_time(data,          str(out))
    chart08_grad_norm(data,              str(out))
    chart09_update_norm(data,            str(out))
    chart10_grad_vs_update_scatter(data, str(out))
    chart11_prf_bar(data,                str(out))
    chart12_overfitting_scatter(data,    str(out))
    chart13_lr_vs_gradnorm(data,         str(out))
    chart14_param_efficiency(data,       str(out))

    print(f"\n✅  All 14 charts saved to: {out.resolve()}\n")


if __name__ == "__main__":
    main()
