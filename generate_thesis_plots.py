"""
Thesis Report Generator — MNIST Step Size Strategy Comparison
=============================================================
Generates publication-quality PNG figures (300 dpi, no subplots) comparing
four step-size strategies (constant, momentum, decay, armijo) across four
architectures (MLP, DeepMLP, DeepMLPBN, CNN) on the MNIST dataset.

Each figure is saved as a separate PNG, ready for \includegraphics{} in LaTeX.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import itertools
from config import TrainingConfig

# ── paths ────────────────────────────────────────────────────────────────────
config = TrainingConfig()
INPUT_PATH = f"logs/mnist_full_experiment_{config.experiment}.json"
OUTPUT_DIR = f"thesis_reports/{config.experiment}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
with open(INPUT_PATH) as f:
    RAW = json.load(f)

MODELS     = list(RAW.keys())                          # MLP, DeepMLP, DeepMLPBN, CNN
STRATEGIES = list(RAW[MODELS[0]]["strategies"].keys()) # constant, momentum, decay, armijo
EPOCHS     = list(range(1, len(RAW[MODELS[0]]["strategies"][STRATEGIES[0]]["train_loss"]) + 1))

MODEL_LABELS = {
    "MLP":       "MLP",
    "DeepMLP":   "Deep MLP",
    "DeepMLPBN": "Deep MLP + BN",
    "CNN":       "CNN",
}
STRAT_LABELS = {
    "constant": "Constant LR",
    "momentum": "Momentum",
    "decay":    "LR Decay",
    "armijo":   "Armijo Line Search",
}

# ── design system ─────────────────────────────────────────────────────────────
PALETTE = {
    "constant": "#2E86AB",   # steel blue
    "momentum": "#E84855",   # crimson
    "decay":    "#F4A261",   # amber
    "armijo":   "#3BB273",   # emerald
}
MARKERS = {
    "constant": "o",
    "momentum": "s",
    "decay":    "^",
    "armijo":   "D",
}
LINE_STYLES = {
    "constant": "-",
    "momentum": "--",
    "decay":    "-.",
    "armijo":   ":",
}

DPI        = 300
FIG_W      = 7.0   # inches — fits a standard thesis page column
FIG_H      = 4.5
GRID_COLOR = "#E8E8E8"
FONT_TITLE = 13
FONT_LABEL = 11
FONT_TICK  = 9
FONT_LEGEND= 9
FONT_ANNOT = 8

def apply_style(ax, title, xlabel, ylabel):
    """Apply consistent thesis-quality style to an axes object."""
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.set_facecolor("white")
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {name}")

def new_fig():
    return plt.figure(figsize=(FIG_W, FIG_H))

def get(model, strat, key):
    return RAW[model]["strategies"][strat][key]

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  TRAINING LOSS — one figure per model
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] Training loss curves …")
for model in MODELS:
    fig = new_fig()
    ax  = fig.add_subplot(111)
    for strat in STRATEGIES:
        y = get(model, strat, "train_loss")
        ax.plot(EPOCHS, y,
                color=PALETTE[strat], marker=MARKERS[strat],
                linestyle=LINE_STYLES[strat], linewidth=1.8,
                markersize=5, label=STRAT_LABELS[strat], zorder=3)
    apply_style(ax,
                title=f"Training Loss — {MODEL_LABELS[model]}",
                xlabel="Epoch",
                ylabel="Cross-Entropy Loss")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
    fig.tight_layout()
    save(fig, f"01_train_loss_{model}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TEST LOSS — one figure per model
# ═══════════════════════════════════════════════════════════════════════════════
print("[2] Test loss curves …")
for model in MODELS:
    fig = new_fig()
    ax  = fig.add_subplot(111)
    for strat in STRATEGIES:
        y = get(model, strat, "test_loss")
        ax.plot(EPOCHS, y,
                color=PALETTE[strat], marker=MARKERS[strat],
                linestyle=LINE_STYLES[strat], linewidth=1.8,
                markersize=5, label=STRAT_LABELS[strat], zorder=3)
    apply_style(ax,
                title=f"Test Loss — {MODEL_LABELS[model]}",
                xlabel="Epoch",
                ylabel="Cross-Entropy Loss")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
    fig.tight_layout()
    save(fig, f"02_test_loss_{model}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TRAINING ACCURACY — one figure per model
# ═══════════════════════════════════════════════════════════════════════════════
print("[3] Training accuracy curves …")
for model in MODELS:
    fig = new_fig()
    ax  = fig.add_subplot(111)
    for strat in STRATEGIES:
        y = get(model, strat, "train_acc")
        ax.plot(EPOCHS, y,
                color=PALETTE[strat], marker=MARKERS[strat],
                linestyle=LINE_STYLES[strat], linewidth=1.8,
                markersize=5, label=STRAT_LABELS[strat], zorder=3)
    apply_style(ax,
                title=f"Training Accuracy — {MODEL_LABELS[model]}",
                xlabel="Epoch",
                ylabel="Accuracy (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
    fig.tight_layout()
    save(fig, f"03_train_acc_{model}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TEST ACCURACY — one figure per model
# ═══════════════════════════════════════════════════════════════════════════════
print("[4] Test accuracy curves …")
for model in MODELS:
    fig = new_fig()
    ax  = fig.add_subplot(111)
    for strat in STRATEGIES:
        y = get(model, strat, "test_acc")
        ax.plot(EPOCHS, y,
                color=PALETTE[strat], marker=MARKERS[strat],
                linestyle=LINE_STYLES[strat], linewidth=1.8,
                markersize=5, label=STRAT_LABELS[strat], zorder=3)
    apply_style(ax,
                title=f"Test Accuracy — {MODEL_LABELS[model]}",
                xlabel="Epoch",
                ylabel="Accuracy (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
    fig.tight_layout()
    save(fig, f"04_test_acc_{model}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LEARNING RATE EVOLUTION — one figure per model
# ═══════════════════════════════════════════════════════════════════════════════
print("[5] Learning rate schedules …")
for model in MODELS:
    fig = new_fig()
    ax  = fig.add_subplot(111)
    for strat in STRATEGIES:
        lr = get(model, strat, "learning_rates")
        ax.plot(EPOCHS, lr,
                color=PALETTE[strat], marker=MARKERS[strat],
                linestyle=LINE_STYLES[strat], linewidth=1.8,
                markersize=5, label=STRAT_LABELS[strat], zorder=3)
    apply_style(ax,
                title=f"Learning Rate Schedule — {MODEL_LABELS[model]}",
                xlabel="Epoch",
                ylabel="Learning Rate")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_yscale("log")
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
    fig.tight_layout()
    save(fig, f"05_learning_rate_{model}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  GENERALISATION GAP (Train Acc − Test Acc) — one figure per model
# ═══════════════════════════════════════════════════════════════════════════════
print("[6] Generalisation gap …")
for model in MODELS:
    fig = new_fig()
    ax  = fig.add_subplot(111)
    for strat in STRATEGIES:
        tr = np.array(get(model, strat, "train_acc"))
        te = np.array(get(model, strat, "test_acc"))
        gap = tr - te
        ax.plot(EPOCHS, gap,
                color=PALETTE[strat], marker=MARKERS[strat],
                linestyle=LINE_STYLES[strat], linewidth=1.8,
                markersize=5, label=STRAT_LABELS[strat], zorder=3)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    apply_style(ax,
                title=f"Generalisation Gap — {MODEL_LABELS[model]}",
                xlabel="Epoch",
                ylabel="Train Acc − Test Acc (pp)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
    fig.tight_layout()
    save(fig, f"06_gen_gap_{model}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  TIME PER EPOCH — grouped bar chart (all models × all strategies)
# ═══════════════════════════════════════════════════════════════════════════════
print("[7] Time per epoch (mean) …")
fig = new_fig()
ax  = fig.add_subplot(111)

n_models = len(MODELS)
n_strats = len(STRATEGIES)
bar_w    = 0.18
x        = np.arange(n_models)

for i, strat in enumerate(STRATEGIES):
    means = [np.mean(get(m, strat, "times_per_epoch")) for m in MODELS]
    stds  = [np.std(get(m, strat,  "times_per_epoch")) for m in MODELS]
    offset = (i - n_strats / 2 + 0.5) * bar_w
    ax.bar(x + offset, means, width=bar_w, yerr=stds,
           color=PALETTE[strat], label=STRAT_LABELS[strat],
           capsize=3, error_kw={"linewidth": 0.8},
           edgecolor="white", linewidth=0.5, zorder=3)

apply_style(ax,
            title="Mean Time per Epoch by Architecture and Strategy",
            xlabel="Architecture",
            ylabel="Time per Epoch (s)")
ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
ax.set_ylim(bottom=0)
fig.tight_layout()
save(fig, "07_time_per_epoch.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CUMULATIVE TRAINING TIME — one figure per model
# ═══════════════════════════════════════════════════════════════════════════════
print("[8] Cumulative training time …")
for model in MODELS:
    fig = new_fig()
    ax  = fig.add_subplot(111)
    for strat in STRATEGIES:
        t = np.cumsum(get(model, strat, "times_per_epoch"))
        ax.plot(EPOCHS, t,
                color=PALETTE[strat], marker=MARKERS[strat],
                linestyle=LINE_STYLES[strat], linewidth=1.8,
                markersize=5, label=STRAT_LABELS[strat], zorder=3)
    apply_style(ax,
                title=f"Cumulative Training Time — {MODEL_LABELS[model]}",
                xlabel="Epoch",
                ylabel="Wall-clock Time (s)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
    fig.tight_layout()
    save(fig, f"08_cumulative_time_{model}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  FINAL TEST ACCURACY — grouped bar chart summary
# ═══════════════════════════════════════════════════════════════════════════════
print("[9] Final test accuracy summary …")
fig = new_fig()
ax  = fig.add_subplot(111)

bar_w = 0.22
x     = np.arange(n_models)
for i, strat in enumerate(STRATEGIES):
    accs   = [max(get(m, strat, "test_acc")) for m in MODELS]
    offset = (i - n_strats / 2 + 0.5) * bar_w
    bars   = ax.bar(x + offset, accs, width=bar_w,
                    color=PALETTE[strat], label=STRAT_LABELS[strat],
                    edgecolor="white", linewidth=0.5, zorder=3)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=6.5, color="#333333", fontweight="bold")

apply_style(ax,
            title="Maximum Test Accuracy after Last Epoch",
            xlabel="Architecture",
            ylabel="Test Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
ax.set_ylim(0, 105)
fig.tight_layout()
save(fig, "09_final_test_accuracy.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  TEST ACCURACY EFFICIENCY — Accuracy per Second (final acc / total time)
# ═══════════════════════════════════════════════════════════════════════════════
print("[10] Accuracy-per-second efficiency …")
fig = new_fig()
ax  = fig.add_subplot(111)

bar_w = 0.18
x     = np.arange(n_models)
for i, strat in enumerate(STRATEGIES):
    eff    = [get(m, strat, "test_acc")[-1] /
              sum(get(m, strat, "times_per_epoch")) for m in MODELS]
    offset = (i - n_strats / 2 + 0.5) * bar_w
    bars   = ax.bar(x + offset, eff, width=bar_w,
                    color=PALETTE[strat], label=STRAT_LABELS[strat],
                    edgecolor="white", linewidth=0.5, zorder=3)
    for bar, val in zip(bars, eff):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=6.5, color="#333333", fontweight="bold")

apply_style(ax,
            title="Training Efficiency: Final Test Accuracy per Second",
            xlabel="Architecture",
            ylabel="Test Accuracy (%) / Wall-clock Time (s)")
ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
ax.set_ylim(bottom=0)
fig.tight_layout()
save(fig, "10_efficiency.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  TRAIN vs TEST ACCURACY (overlay) — one figure per model per strategy
#      Shows over-fitting or under-fitting at a glance
# ═══════════════════════════════════════════════════════════════════════════════
print("[11] Train vs Test accuracy overlay (per model × strategy) …")
for model in MODELS:
    for strat in STRATEGIES:
        fig = new_fig()
        ax  = fig.add_subplot(111)
        tr  = get(model, strat, "train_acc")
        te  = get(model, strat, "test_acc")

        ax.plot(EPOCHS, tr, color=PALETTE[strat], linewidth=2.0,
                marker=MARKERS[strat], markersize=6,
                label="Train", linestyle="-", zorder=3)
        ax.plot(EPOCHS, te, color=PALETTE[strat], linewidth=2.0,
                marker=MARKERS[strat], markersize=6,
                label="Test",  linestyle="--", alpha=0.65, zorder=3)

        ax.fill_between(EPOCHS, tr, te,
                        alpha=0.10, color=PALETTE[strat], zorder=2)

        apply_style(ax,
                    title=f"Train vs Test Accuracy — {MODEL_LABELS[model]}, {STRAT_LABELS[strat]}",
                    xlabel="Epoch",
                    ylabel="Accuracy (%)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
        fig.tight_layout()
        save(fig, f"11_train_vs_test_{model}_{strat}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 12.  CONVERGENCE SPEED — Loss drop per second (loss reduction rate)
# ═══════════════════════════════════════════════════════════════════════════════
print("[12] Convergence speed (loss drop / cumulative time) …")
for model in MODELS:
    fig = new_fig()
    ax  = fig.add_subplot(111)
    for strat in STRATEGIES:
        losses = np.array(get(model, strat, "train_loss"))
        times  = np.cumsum(get(model, strat, "times_per_epoch"))
        rate   = -np.diff(losses) / np.diff(times)      # loss drop per second
        # plot at midpoints
        mid_t  = (times[:-1] + times[1:]) / 2
        ax.plot(range(1, len(rate) + 1), rate,
                color=PALETTE[strat], marker=MARKERS[strat],
                linestyle=LINE_STYLES[strat], linewidth=1.8,
                markersize=5, label=STRAT_LABELS[strat], zorder=3)

    apply_style(ax,
                title=f"Convergence Speed (Loss Drop / Wall-clock s) — {MODEL_LABELS[model]}",
                xlabel="Epoch Interval",
                ylabel="Loss Reduction per Second")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
    fig.tight_layout()
    save(fig, f"12_convergence_speed_{model}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 13.  CONFUSION MATRICES — last epoch, one per model per strategy
# ═══════════════════════════════════════════════════════════════════════════════
print("[13] Confusion matrices (last epoch) …")
CLASS_NAMES = [str(i) for i in range(10)]

# Custom blue-white colormap for a clean publication look
cmap_cm = LinearSegmentedColormap.from_list(
    "thesis_cm", ["#FFFFFF", "#2E86AB"], N=256)

for model in MODELS:
    for strat in STRATEGIES:
        cm_raw = np.array(get(model, strat, "confusion_matrices")[-1])

        # Normalise rows → recall per class
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        cm_norm  = cm_raw / row_sums.clip(min=1)

        fig  = plt.figure(figsize=(5.5, 4.8))   # square-ish for a 10×10 matrix
        ax   = fig.add_subplot(111)

        im = ax.imshow(cm_norm, cmap=cmap_cm, vmin=0, vmax=1, aspect="equal")

        # Colour bar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=FONT_TICK)
        cbar.set_label("Recall", fontsize=FONT_TICK)

        # Annotate cells
        for r, c in itertools.product(range(10), range(10)):
            val  = cm_norm[r, c]
            txt  = f"{val:.2f}"
            color = "white" if val > 0.55 else "#222222"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=5.5, color=color, fontweight="bold")

        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASS_NAMES, fontsize=FONT_TICK)
        ax.set_yticklabels(CLASS_NAMES, fontsize=FONT_TICK)
        ax.set_xlabel("Predicted Label", fontsize=FONT_LABEL)
        ax.set_ylabel("True Label", fontsize=FONT_LABEL)
        ax.set_title(
            f"Confusion Matrix (Epoch {len(EPOCHS)}) —\n"
            f"{MODEL_LABELS[model]}, {STRAT_LABELS[strat]}",
            fontsize=FONT_TITLE - 1, fontweight="bold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        save(fig, f"13_confusion_{model}_{strat}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 14.  PER-CLASS RECALL (last epoch) — horizontal bar chart per model per strategy
# ═══════════════════════════════════════════════════════════════════════════════
print("[14] Per-class recall (last epoch) …")
for model in MODELS:
    for strat in STRATEGIES:
        cm_raw   = np.array(get(model, strat, "confusion_matrices")[-1])
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        cm_norm  = cm_raw / row_sums.clip(min=1)
        recall   = np.diag(cm_norm)

        fig = plt.figure(figsize=(FIG_W, 4.2))
        ax  = fig.add_subplot(111)
        colors = [PALETTE["armijo"] if v >= 0.7 else
                  PALETTE["decay"]  if v >= 0.5 else
                  PALETTE["momentum"] if v >= 0.3 else
                  PALETTE["constant"] for v in recall]
        bars = ax.barh(CLASS_NAMES, recall * 100, color=colors,
                       edgecolor="white", linewidth=0.4, zorder=3)
        for bar, val in zip(bars, recall):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", fontsize=FONT_ANNOT,
                    color="#333333")
        ax.axvline(recall.mean() * 100, color="#333333", linewidth=1.0,
                   linestyle="--", label=f"Mean {recall.mean()*100:.1f}%")
        apply_style(ax,
                    title=f"Per-Class Recall (Epoch {len(EPOCHS)}) —\n"
                          f"{MODEL_LABELS[model]}, {STRAT_LABELS[strat]}",
                    xlabel="Recall (%)",
                    ylabel="Digit Class")
        ax.set_xlim(0, 115)
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
        ax.set_facecolor("white")
        ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
        fig.tight_layout()
        save(fig, f"14_per_class_recall_{model}_{strat}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 15.  STRATEGY COMPARISON HEATMAP — final test accuracy across all combos
# ═══════════════════════════════════════════════════════════════════════════════
print("[15] Strategy × Architecture heatmap …")
matrix = np.array(
    [[max(get(m, s, "test_acc")) for s in STRATEGIES] for m in MODELS]
)

fig = plt.figure(figsize=(6.5, 4.0))
ax  = fig.add_subplot(111)
im  = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=100, aspect="auto")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=FONT_TICK)
cbar.set_label("Test Accuracy (%)", fontsize=FONT_TICK)

for (r, c), val in np.ndenumerate(matrix):
    color = "white" if val > 75 else "#222222"
    ax.text(c, r, f"{val:.2f}%", ha="center", va="center",
            fontsize=9, fontweight="bold", color=color)

ax.set_xticks(range(len(STRATEGIES)))
ax.set_yticks(range(len(MODELS)))
ax.set_xticklabels([STRAT_LABELS[s] for s in STRATEGIES],
                   fontsize=FONT_TICK, rotation=20, ha="right")
ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=FONT_TICK)
ax.set_title("Final Test Accuracy (%) — Architecture × Strategy",
             fontsize=FONT_TITLE, fontweight="bold", pad=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save(fig, "15_heatmap_accuracy.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 16.  LOSS CONVERGENCE COMPARISON — all models overlaid, one per strategy
#       Shows which architecture converges fastest under each step-size rule
# ═══════════════════════════════════════════════════════════════════════════════
print("[16] Cross-architecture loss convergence per strategy …")
MODEL_COLORS = {
    "MLP":       "#264653",
    "DeepMLP":   "#E76F51",
    "DeepMLPBN": "#2A9D8F",
    "CNN":       "#E9C46A",
}
MODEL_MARKERS = {"MLP": "o", "DeepMLP": "s", "DeepMLPBN": "^", "CNN": "D"}

for strat in STRATEGIES:
    fig = new_fig()
    ax  = fig.add_subplot(111)
    for model in MODELS:
        y = get(model, strat, "train_loss")
        ax.plot(EPOCHS, y,
                color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
                linewidth=1.8, markersize=5,
                label=MODEL_LABELS[model], zorder=3)
    apply_style(ax,
                title=f"Training Loss by Architecture — {STRAT_LABELS[strat]}",
                xlabel="Epoch",
                ylabel="Cross-Entropy Loss")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#CCCCCC")
    fig.tight_layout()
    save(fig, f"16_arch_loss_{strat}.png")

# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n✅  All figures written to: {OUTPUT_DIR}")
print(f"    Total files generated: see above list.")
