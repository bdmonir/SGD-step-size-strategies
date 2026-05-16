"""
generate_tables.py
==================
Generates LaTeX-ready summary tables as PNG files from MNIST SGD experiment results.

Tables 1–6   : Summary / comparative analysis (original)
Tables 7–12  : Deep analysis (new)

  7  – Per-Class Accuracy from Confusion Matrix
  8  – Learning Rate Behaviour (Armijo vs Decay)
  9  – Training Loss Reduction Ratio
  10 – Late-Training Stability (last-5-epoch σ)
  11 – Peak vs Final Test Accuracy (Post-Peak Drop)
  12 – Composite Score Ranking (accuracy · stability / time)

Each table is saved as an individual PNG for direct \\includegraphics use in LaTeX.

Usage:
    python generate_tables.py
    (JSON path and output directory are hardcoded below)
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from config import TrainingConfig
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (hardcoded as requested)
# ─────────────────────────────────────────────────────────────────────────────
config = TrainingConfig()
JSON_PATH = f"logs/mnist_full_experiment_{config.experiment}.json"
OUTPUT_DIR = f"thesis_table_reports/{config.experiment}"
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_LABELS = {
    "constant": "Constant",
    "momentum": "Momentum",
    "decay":    "Decay",
    "armijo":   "Armijo LS",
}
MODEL_LABELS = {
    "MLP":       "MLP",
    "DeepMLP":   "Deep MLP",
    "DeepMLPBN": "Deep MLP + BN",
    "CNN":       "CNN",
}

# ── Colour palette ────────────────────────────────────────────────────────────
HEADER_BG  = "#1a3a5c"
HEADER_FG  = "#ffffff"
SUBHEAD_BG = "#2e6da4"
SUBHEAD_FG = "#ffffff"
ROW_ODD    = "#eef3f9"
ROW_EVEN   = "#ffffff"
BEST_COLOR = "#d4edda"   # soft green  – best value
WARN_COLOR = "#fde8e8"   # soft red    – worst value
GRID_COLOR = "#c8d6e5"
FONT_FAMILY = "DejaVu Sans"

matplotlib.rcParams.update({
    "font.family": FONT_FAMILY,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


# =============================================================================
# DATA HELPERS
# =============================================================================

def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def final(series: list) -> float:
    return series[-1]


def epoch_to_threshold(series: list, threshold: float):
    for i, v in enumerate(series):
        if v >= threshold:
            return i + 1
    return None


def per_class_acc(confusion_matrix) -> np.ndarray:
    cm = np.array(confusion_matrix)
    row_sums = cm.sum(axis=1)
    return np.where(row_sums > 0, cm.diagonal() / row_sums * 100, 0.0)


def extract_metrics(data: dict) -> dict:
    metrics = {}
    for model, mdata in data.items():
        metrics[model] = {}
        params = mdata["total_parameters"]
        for strat, sdata in mdata["strategies"].items():
            ta    = sdata["train_acc"]
            tea   = sdata["test_acc"]
            tl    = sdata["train_loss"]
            tel   = sdata["test_loss"]
            times = sdata["times_per_epoch"]
            lrs   = sdata["learning_rates"]
            cms   = sdata["confusion_matrices"]

            peak_epoch = int(np.argmax(tea))
            lr_arr     = np.array(lrs, dtype=float)

            metrics[model][strat] = {
                # basic
                "params":              params,
                "final_train_acc":     final(ta),
                "final_test_acc":      final(tea),
                "final_train_loss":    final(tl),
                "final_test_loss":     final(tel),
                "total_time_s":        sum(times),
                "avg_epoch_time_s":    float(np.mean(times)),
                "epoch_95":            epoch_to_threshold(tea, 95.0),
                "epoch_98":            epoch_to_threshold(tea, 98.0),
                "peak_test_acc":       float(max(tea)),
                "peak_epoch":          peak_epoch + 1,          # 1-based
                "post_peak_drop":      float(max(tea) - tea[-1]),
                "final_lr":            final(lrs),
                "num_epochs":          len(ta),
                # deep
                "late_std":            float(np.std(tea[-5:])),
                "loss_reduction":      tl[0] / tl[-1] if tl[-1] > 1e-12 else float("inf"),
                "lr_mean":             float(lr_arr.mean()),
                "lr_std":              float(lr_arr.std()),
                "lr_min":              float(lr_arr.min()),
                "lr_max":              float(lr_arr.max()),
                "per_class_acc_final": per_class_acc(cms[-1]),
                "raw_tea":             tea,
                "raw_tl":              tl,
                "raw_lrs":             lrs,
            }
    return metrics


# =============================================================================
# RENDERING ENGINE
# =============================================================================

def _parse_float(cell_text: str) -> float:
    s = (str(cell_text)
         .replace("m", "").replace("s", "").replace(" ", "")
         .replace(",", "").replace("x", "").replace("X", "")
         .replace("*", "").replace("—", "nan").replace("inf", "inf"))
    s = s.rstrip("x").rstrip("X")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def render_table(
    ax: plt.Axes,
    col_headers: list,
    row_groups: list,
    best_cols: list  = None,
    worst_cols: list = None,
    red_cols: list   = None,
    col_widths: list = None,
    title: str = "",
    highlight_within_group: bool = False,
) -> None:
    """
    Renders a styled table on *ax*.

    best_cols  : columns where HIGHER value is better -> green cell
    worst_cols : columns where LOWER value is better  -> green cell
    red_cols   : columns where HIGHEST value is worst -> red cell
    highlight_within_group : restrict highlight detection to within each group
    """
    ax.axis("off")

    n_cols = len(col_headers)
    n_data = sum(len(g["rows"]) for g in row_groups)
    n_rows = n_data + len(row_groups)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols
    total_w = sum(col_widths)
    col_widths = [w / total_w for w in col_widths]

    ROW_H = 1.0 / (n_rows + 1.5)

    all_rows    = []
    group_spans = []
    for g in row_groups:
        s = len(all_rows)
        all_rows.extend(g["rows"])
        group_spans.append((s, len(all_rows)))

    green_set = set()
    red_set   = set()

    def _mark(col_idx, rows_slice, offset, higher_better, target_set):
        vals = [_parse_float(rows_slice[i][col_idx]) for i in range(len(rows_slice))]
        if all(np.isnan(vals)):
            return
        best = np.nanmax(vals) if higher_better else np.nanmin(vals)
        for i, v in enumerate(vals):
            if not np.isnan(v) and abs(v - best) < 1e-9:
                target_set.add((offset + i, col_idx))

    def _process(cols, higher_better, target_set):
        if not cols:
            return
        for ci in cols:
            if highlight_within_group:
                for (s, e) in group_spans:
                    _mark(ci, all_rows[s:e], s, higher_better, target_set)
            else:
                _mark(ci, all_rows, 0, higher_better, target_set)

    _process(best_cols,  True,  green_set)
    _process(worst_cols, False, green_set)
    _process(red_cols,   True,  red_set)

    x_edges = [0.0]
    for w in col_widths:
        x_edges.append(x_edges[-1] + w)

    def add_cell(row_i, col_i, text, bg, fg="#111111",
                 fontsize=8.2, bold=False, ha="center"):
        x = x_edges[col_i]
        y = 1.0 - (row_i + 1) * ROW_H
        w = col_widths[col_i]
        h = ROW_H
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=0.4,
            edgecolor=GRID_COLOR,
            facecolor=bg,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2 if ha == "center" else x + 0.008,
            y + h / 2,
            str(text),
            color=fg,
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            ha=ha, va="center",
            transform=ax.transAxes,
            clip_on=False,
        )

    for ci, hdr in enumerate(col_headers):
        add_cell(0, ci, hdr, HEADER_BG, HEADER_FG, fontsize=8.5, bold=True)

    row_cursor   = 1
    data_row_idx = 0

    for g in row_groups:
        for ci in range(n_cols):
            add_cell(row_cursor, ci,
                     g["label"] if ci == 0 else "",
                     SUBHEAD_BG, SUBHEAD_FG,
                     fontsize=8.5, bold=True,
                     ha="left" if ci == 0 else "center")
        row_cursor += 1

        for ri, row in enumerate(g["rows"]):
            base_bg = ROW_ODD if ri % 2 == 0 else ROW_EVEN
            for ci, cell in enumerate(row):
                if (data_row_idx, ci) in green_set:
                    bg = BEST_COLOR
                elif (data_row_idx, ci) in red_set:
                    bg = WARN_COLOR
                else:
                    bg = base_bg
                add_cell(row_cursor, ci, cell, bg,
                         ha="left" if ci == 0 else "center",
                         fontsize=8.2)
            row_cursor   += 1
            data_row_idx += 1

    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold",
                     pad=6, color="#1a3a5c", loc="left")


def save_fig(fig: plt.Figure, name: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved -> {path}")
    return path


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"

def _fmt_epoch(v) -> str:
    return str(v) if v is not None else "n/a"

def _fmt_inf(v: float, fmt=".1f") -> str:
    if np.isinf(v):
        return "inf"
    return f"{v:{fmt}}"


# =============================================================================
# TABLES 1-6  (original, lightly refactored to use updated render_table)
# =============================================================================

def table_final_performance(metrics: dict, output_dir: str) -> None:
    print("Generating Table 1: Final Performance Summary ...")
    col_headers = ["Strategy", "Test Acc. (%)", "Train Acc. (%)", "Test Loss", "Train Loss"]
    col_widths  = [1.6, 1.0, 1.0, 0.85, 0.85]
    row_groups  = []
    for mk, ml in MODEL_LABELS.items():
        rows = []
        for sk, sl in STRATEGY_LABELS.items():
            m = metrics[mk][sk]
            rows.append([sl,
                         f"{m['final_test_acc']:.2f}",
                         f"{m['final_train_acc']:.2f}",
                         f"{m['final_test_loss']:.4f}",
                         f"{m['final_train_loss']:.4f}"])
        row_groups.append({"label": ml, "rows": rows})
    fig, ax = plt.subplots(figsize=(9, 8))
    render_table(ax, col_headers, row_groups,
                 best_cols=[1, 2], worst_cols=[3, 4],
                 col_widths=col_widths, highlight_within_group=True,
                 title="Table 1 - Final Performance Summary (epoch 30)\n"
                       "Green = best value per column within each model.")
    fig.tight_layout(pad=0.5)
    save_fig(fig, "table1_final_performance", output_dir)


def table_convergence_speed(metrics: dict, output_dir: str) -> None:
    print("Generating Table 2: Convergence Speed ...")
    col_headers = ["Strategy", "Epoch >= 95%", "Epoch >= 98%", "Peak Test Acc. (%)"]
    col_widths  = [1.6, 1.0, 1.0, 1.2]
    row_groups  = []
    for mk, ml in MODEL_LABELS.items():
        rows = []
        for sk, sl in STRATEGY_LABELS.items():
            m = metrics[mk][sk]
            rows.append([sl,
                         _fmt_epoch(m["epoch_95"]),
                         _fmt_epoch(m["epoch_98"]),
                         f"{m['peak_test_acc']:.2f}"])
        row_groups.append({"label": ml, "rows": rows})
    fig, ax = plt.subplots(figsize=(8, 8))
    render_table(ax, col_headers, row_groups,
                 worst_cols=[1, 2], best_cols=[3],
                 col_widths=col_widths, highlight_within_group=True,
                 title="Table 2 - Convergence Speed\n"
                       "Epoch at which test accuracy first reaches threshold. "
                       "Green = fastest convergence / highest peak.")
    fig.tight_layout(pad=0.5)
    save_fig(fig, "table2_convergence_speed", output_dir)


def table_computational_cost(metrics: dict, output_dir: str) -> None:
    print("Generating Table 3: Computational Cost ...")
    col_headers = ["Strategy", "Total Time", "Avg. Epoch Time (s)", "Parameters"]
    col_widths  = [1.6, 1.0, 1.3, 1.1]
    row_groups  = []
    for mk, ml in MODEL_LABELS.items():
        rows = []
        for sk, sl in STRATEGY_LABELS.items():
            m = metrics[mk][sk]
            rows.append([sl,
                         _fmt_time(m["total_time_s"]),
                         f"{m['avg_epoch_time_s']:.2f}",
                         f"{m['params']:,}"])
        row_groups.append({"label": ml, "rows": rows})
    fig, ax = plt.subplots(figsize=(8.5, 8))
    render_table(ax, col_headers, row_groups,
                 worst_cols=[2],
                 col_widths=col_widths, highlight_within_group=True,
                 title="Table 3 - Computational Cost\n"
                       "Green = lowest average epoch time (most efficient strategy).")
    fig.tight_layout(pad=0.5)
    save_fig(fig, "table3_computational_cost", output_dir)


def table_generalisation_gap(metrics: dict, output_dir: str) -> None:
    print("Generating Table 4: Generalisation Gap ...")
    col_headers = ["Strategy", "Acc. Gap (pp)", "Loss Gap"]
    col_widths  = [1.6, 1.1, 1.0]
    row_groups  = []
    for mk, ml in MODEL_LABELS.items():
        rows = []
        for sk, sl in STRATEGY_LABELS.items():
            m = metrics[mk][sk]
            acc_gap  = m["final_train_acc"] - m["final_test_acc"]
            loss_gap = m["final_test_loss"] - m["final_train_loss"]
            rows.append([sl, f"{acc_gap:.2f}", f"{loss_gap:.4f}"])
        row_groups.append({"label": ml, "rows": rows})
    fig, ax = plt.subplots(figsize=(6.5, 8))
    render_table(ax, col_headers, row_groups,
                 worst_cols=[1, 2],
                 col_widths=col_widths, highlight_within_group=True,
                 title="Table 4 - Generalisation Gap (Train - Test)\n"
                       "pp = percentage points. Green = smallest gap (best generalisation).")
    fig.tight_layout(pad=0.5)
    save_fig(fig, "table4_generalisation_gap", output_dir)


def table_strategy_ranking(metrics: dict, output_dir: str) -> None:
    print("Generating Table 5: Strategy Ranking ...")
    models     = list(MODEL_LABELS.keys())
    strategies = list(STRATEGY_LABELS.keys())
    col_headers = ["Strategy"] + [MODEL_LABELS[m] for m in models] + ["Mean (%)"]
    col_widths  = [1.4] + [1.0] * len(models) + [1.1]
    rows = []
    for sk in strategies:
        accs = [metrics[mk][sk]["final_test_acc"] for mk in models]
        rows.append([STRATEGY_LABELS[sk]]
                    + [f"{a:.2f}" for a in accs]
                    + [f"{np.mean(accs):.2f}"])
    rows.sort(key=lambda r: float(r[-1]), reverse=True)
    row_groups = [{"label": "All Models", "rows": rows}]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    render_table(ax, col_headers, row_groups,
                 best_cols=list(range(1, len(models) + 2)),
                 col_widths=col_widths,
                 title="Table 5 - Strategy Ranking by Final Test Accuracy (sorted by mean)\n"
                       "Green = highest test accuracy per column.")
    fig.tight_layout(pad=0.5)
    save_fig(fig, "table5_strategy_ranking", output_dir)


def table_efficiency_tradeoff(metrics: dict, output_dir: str) -> None:
    print("Generating Table 6: Efficiency vs Accuracy Trade-off ...")
    col_headers = ["Model", "Strategy", "Test Acc. (%)", "Total Time (s)", "Acc / min"]
    col_widths  = [1.4, 1.4, 1.0, 1.15, 1.0]
    rows_all = []
    for mk, ml in MODEL_LABELS.items():
        for sk, sl in STRATEGY_LABELS.items():
            m   = metrics[mk][sk]
            apm = m["final_test_acc"] / (m["total_time_s"] / 60.0)
            rows_all.append({
                "model": ml, "strat": sl,
                "test_acc": m["final_test_acc"],
                "total_time": m["total_time_s"],
                "acc_per_min": apm,
            })
    rows_all.sort(key=lambda r: r["acc_per_min"], reverse=True)
    table_rows = [[r["model"], r["strat"],
                   f"{r['test_acc']:.2f}",
                   f"{r['total_time']:.0f}",
                   f"{r['acc_per_min']:.2f}"]
                  for r in rows_all]
    row_groups = [{"label": "All Combinations", "rows": table_rows}]
    fig, ax = plt.subplots(figsize=(9, 12))
    render_table(ax, col_headers, row_groups,
                 best_cols=[2, 4], worst_cols=[3],
                 col_widths=col_widths,
                 title="Table 6 - Efficiency vs Accuracy Trade-off\n"
                       "Acc / min = test accuracy per minute of training. "
                       "Sorted by Acc / min (descending).")
    fig.tight_layout(pad=0.5)
    save_fig(fig, "table6_efficiency_tradeoff", output_dir)


# =============================================================================
# TABLE 7 - Per-Class Accuracy from Final Confusion Matrix
#
# Goal: Reveal which digit classes are hardest for each architecture/strategy.
#       Digit-level weaknesses expose model capacity limits and class complexity
#       (e.g. digit 9 vs 4, or 3 vs 8). Essential for thesis discussion of
#       where models fail regardless of aggregate accuracy.
# =============================================================================

def table_per_class_accuracy(data: dict, metrics: dict, output_dir: str) -> None:
    print("Generating Table 7: Per-Class Accuracy ...")

    col_headers = (["Strategy"]
                   + [f"Digit {d}" for d in range(10)]
                   + ["Worst Digit", "Best Digit"])
    col_widths  = [1.5] + [0.7] * 10 + [1.1, 1.1]

    row_groups = []
    for mk, ml in MODEL_LABELS.items():
        rows = []
        for sk, sl in STRATEGY_LABELS.items():
            pca     = metrics[mk][sk]["per_class_acc_final"]
            worst_d = int(np.argmin(pca))
            best_d  = int(np.argmax(pca))
            row = ([sl]
                   + [f"{v:.1f}" for v in pca]
                   + [f"D{worst_d} ({pca[worst_d]:.1f}%)",
                      f"D{best_d} ({pca[best_d]:.1f}%)"])
            rows.append(row)
        row_groups.append({"label": ml, "rows": rows})

    # Columns 1-10 are per-digit: green = best strategy for that digit,
    # red = worst strategy for that digit (within each model group)
    fig, ax = plt.subplots(figsize=(17, 8))
    render_table(ax, col_headers, row_groups,
                 best_cols=list(range(1, 11)),
                 red_cols=list(range(1, 11)),
                 col_widths=col_widths,
                 highlight_within_group=True,
                 title="Table 7 - Per-Class Accuracy (%) from Final Confusion Matrix\n"
                       "Green = highest accuracy for that digit within the model group. "
                       "Red = lowest accuracy (hardest digit / weakest strategy).")
    fig.tight_layout(pad=0.5)
    save_fig(fig, "table7_per_class_accuracy", output_dir)


# =============================================================================
# TABLE 8 - Learning Rate Behaviour
#
# Goal: Quantify how each step-size strategy evolves over 30 epochs.
#       Armijo's high std-dev and collapsing final LR contrast sharply with
#       fixed/decay schedules. This table bridges theory (LR rules) to the
#       empirical training curves observed in Tables 1-2.
# =============================================================================

def table_lr_behaviour(metrics: dict, output_dir: str) -> None:
    print("Generating Table 8: Learning Rate Behaviour ...")

    col_headers = ["Model", "Strategy", "LR @ Ep.1", "LR @ Ep.30",
                   "Min LR", "Max LR", "Std Dev", "Schedule Type"]
    col_widths  = [1.3, 1.3, 0.9, 0.9, 0.9, 0.9, 0.9, 1.3]

    def fmt_lr(v: float) -> str:
        return f"{v:.2e}" if v < 0.001 or v > 9.99 else f"{v:.4f}"

    def schedule_type(sk: str, lrs: list) -> str:
        if sk in ("constant", "momentum"):
            return "Fixed"
        if sk == "decay":
            return "Polynomial decay"
        # Armijo
        return "Adaptive (collapsed)" if lrs[-1] < 1e-6 else "Adaptive"

    row_groups = []
    for mk, ml in MODEL_LABELS.items():
        rows = []
        for sk, sl in STRATEGY_LABELS.items():
            m   = metrics[mk][sk]
            lrs = m["raw_lrs"]
            rows.append([ml, sl,
                         fmt_lr(lrs[0]),
                         fmt_lr(lrs[-1]),
                         fmt_lr(m["lr_min"]),
                         fmt_lr(m["lr_max"]),
                         f"{m['lr_std']:.4f}",
                         schedule_type(sk, lrs)])
        row_groups.append({"label": ml, "rows": rows})

    fig, ax = plt.subplots(figsize=(15, 8))
    render_table(ax, col_headers, row_groups,
                 col_widths=col_widths,
                 highlight_within_group=False,
                 title="Table 8 - Learning Rate Behaviour Across Strategies (30 Epochs)\n"
                       "High Std Dev (Armijo) indicates aggressive adaptive step-size; "
                       "Decay shows monotone decrease; Constant/Momentum are fixed.")
    fig.tight_layout(pad=0.5)
    save_fig(fig, "table8_lr_behaviour", output_dir)


# =============================================================================
# TABLE 9 - Training Loss Reduction Ratio
#
# Goal: Measure how aggressively each strategy reduces training loss from
#       epoch 1 to epoch 30. A large ratio (e.g. 21,000x for DeepMLP/momentum)
#       means the optimiser traversed a very steep loss landscape. Connects
#       loss theory to practical convergence narratives in the thesis.
# =============================================================================

def table_loss_reduction(metrics: dict, output_dir: str) -> None:
    print("Generating Table 9: Training Loss Reduction Ratio ...")

    col_headers = ["Strategy", "Loss @ Ep.1", "Loss @ Ep.30",
                   "Reduction Ratio", "Abs. Drop", "Convergence Class"]
    col_widths  = [1.4, 1.0, 1.0, 1.2, 1.0, 1.5]

    def convergence_class(ratio: float) -> str:
        if ratio > 5000:
            return "Extreme (>5000x)"
        if ratio > 500:
            return "Very strong"
        if ratio > 50:
            return "Strong"
        if ratio > 10:
            return "Moderate"
        return "Weak"

    row_groups = []
    for mk, ml in MODEL_LABELS.items():
        rows = []
        for sk, sl in STRATEGY_LABELS.items():
            m     = metrics[mk][sk]
            tl    = m["raw_tl"]
            l1    = tl[0]
            l30   = tl[-1]
            ratio = l1 / l30 if l30 > 1e-12 else float("inf")
            rows.append([sl,
                         f"{l1:.4f}",
                         f"{l30:.6f}",
                         _fmt_inf(ratio, ".1f") + "x",
                         f"{l1 - l30:.4f}",
                         convergence_class(ratio)])
        row_groups.append({"label": ml, "rows": rows})

    fig, ax = plt.subplots(figsize=(12, 8))
    render_table(ax, col_headers, row_groups,
                 best_cols=[3, 4],
                 worst_cols=[2],
                 col_widths=col_widths,
                 highlight_within_group=True,
                 title="Table 9 - Training Loss Reduction Ratio (Epoch 1 to 30)\n"
                       "Ratio = L_ep1 / L_ep30. Green = largest reduction / lowest final loss "
                       "within each model. Reveals how aggressively each strategy optimises.")
    fig.tight_layout(pad=0.5)
    save_fig(fig, "table9_loss_reduction", output_dir)


# =============================================================================
# TABLE 10 - Late-Training Stability
#
# Goal: Measure test accuracy oscillation during the final 5 epochs (26-30).
#       A high sigma means the model has not settled to a stable optimum -
#       problematic for deployment. Low sigma indicates reliable convergence.
#       This distinguishes strategies that converge cleanly from those that
#       keep bouncing even after many epochs.
# =============================================================================

def table_late_stability(metrics: dict, output_dir: str) -> None:
    print("Generating Table 10: Late-Training Stability ...")

    col_headers = ["Strategy",
                   "Ep.26", "Ep.27", "Ep.28", "Ep.29", "Ep.30",
                   "sigma (pp)", "Verdict"]
    col_widths  = [1.4, 0.75, 0.75, 0.75, 0.75, 0.75, 0.85, 1.1]

    def verdict(sigma: float) -> str:
        if sigma < 0.02:
            return "Stable"
        if sigma < 0.06:
            return "Mild oscillation"
        return "Unstable"

    row_groups = []
    for mk, ml in MODEL_LABELS.items():
        rows = []
        for sk, sl in STRATEGY_LABELS.items():
            m     = metrics[mk][sk]
            tea   = m["raw_tea"]
            last5 = tea[-5:]
            sigma = m["late_std"]
            rows.append([sl]
                        + [f"{v:.2f}" for v in last5]
                        + [f"{sigma:.4f}", verdict(sigma)])
        row_groups.append({"label": ml, "rows": rows})

    fig, ax = plt.subplots(figsize=(13, 8))
    render_table(ax, col_headers, row_groups,
                 worst_cols=[6],
                 col_widths=col_widths,
                 highlight_within_group=True,
                 title="Table 10 - Late-Training Stability (Epochs 26-30, Test Accuracy)\n"
                       "sigma = standard deviation in percentage points (pp). "
                       "Green = most stable strategy within each model. "
                       "High sigma signals the optimiser has not fully converged.")
    fig.tight_layout(pad=0.5)
    save_fig(fig, "table10_late_stability", output_dir)


# =============================================================================
# TABLE 11 - Peak vs Final Test Accuracy (Post-Peak Drop)
#
# Goal: Detect strategies that peak early and then degrade - a sign of
#       over-optimisation or destabilising LR dynamics after the loss
#       landscape flattens. The drop metric isolates strategies that are
#       good at convergence but bad at maintaining it.
# =============================================================================

def table_peak_vs_final(metrics: dict, output_dir: str) -> None:
    print("Generating Table 11: Peak vs Final Test Accuracy ...")

    col_headers = ["Strategy", "Peak Acc. (%)", "Peak Epoch",
                   "Final Acc. (%)", "Drop (pp)", "Interpretation"]
    col_widths  = [1.4, 1.0, 0.85, 1.0, 0.85, 1.8]

    def interpret(drop: float, peak_epoch: int, n_epochs: int) -> str:
        early = peak_epoch <= n_epochs // 2
        if drop < 0.02:
            return "No degradation"
        if drop < 0.08 and not early:
            return "Mild late oscillation"
        if drop < 0.08 and early:
            return "Early peak, stable"
        if drop >= 0.08 and early:
            return "Early peak, degrades"
        return "Late-stage overfit"

    row_groups = []
    for mk, ml in MODEL_LABELS.items():
        rows = []
        for sk, sl in STRATEGY_LABELS.items():
            m    = metrics[mk][sk]
            drop = m["post_peak_drop"]
            ne   = m["num_epochs"]
            rows.append([sl,
                         f"{m['peak_test_acc']:.2f}",
                         str(m["peak_epoch"]),
                         f"{m['final_test_acc']:.2f}",
                         f"{drop:.2f}",
                         interpret(drop, m["peak_epoch"], ne)])
        row_groups.append({"label": ml, "rows": rows})

    fig, ax = plt.subplots(figsize=(13, 8))
    render_table(ax, col_headers, row_groups,
                 best_cols=[1, 3],
                 worst_cols=[4],
                 col_widths=col_widths,
                 highlight_within_group=True,
                 title="Table 11 - Peak vs Final Test Accuracy (Post-Peak Drop)\n"
                       "Drop = Peak_acc - Final_acc (pp). Green = highest accuracy / smallest drop. "
                       "Reveals strategies prone to late-stage degradation or instability.")
    fig.tight_layout(pad=0.5)
    save_fig(fig, "table11_peak_vs_final", output_dir)


# =============================================================================
# TABLE 12 - Composite Score Ranking
#
# Goal: Provide a single, thesis-ready practical recommendation matrix that
#       combines accuracy, stability, and training cost into one interpretable
#       score. Gives readers a clear answer to: "Which strategy should I choose
#       for this architecture, and why?"
#
# Score formula:
#   raw = (test_acc / 100) x (1 / (1 + late_std * 100)) x (60 / avg_epoch_s)
#   - Rewards high test accuracy
#   - Penalises late-epoch oscillation (instability)
#   - Rewards fast epochs (lower compute cost)
#   Normalised to [0, 100] across all 16 model x strategy combinations.
# =============================================================================

def table_composite_score(metrics: dict, output_dir: str) -> None:
    print("Generating Table 12: Composite Score Ranking ...")

    col_headers = ["Rank", "Model", "Strategy",
                   "Test Acc. (%)", "Stability sigma", "Avg Ep. (s)",
                   "Composite Score", "Recommendation"]
    col_widths  = [0.5, 1.3, 1.3, 1.0, 1.0, 0.9, 1.3, 1.6]

    records = []
    for mk, ml in MODEL_LABELS.items():
        for sk, sl in STRATEGY_LABELS.items():
            m     = metrics[mk][sk]
            acc   = m["final_test_acc"] / 100.0
            stab  = 1.0 / (1.0 + m["late_std"] * 100)
            speed = 60.0 / m["avg_epoch_time_s"]
            records.append({
                "model": ml, "strat": sl,
                "test_acc":  m["final_test_acc"],
                "late_std":  m["late_std"],
                "avg_time":  m["avg_epoch_time_s"],
                "raw":       acc * stab * speed,
            })

    max_raw = max(r["raw"] for r in records)
    for r in records:
        r["score"] = r["raw"] / max_raw * 100.0

    records.sort(key=lambda r: r["score"], reverse=True)

    def recommendation(rank: int, score: float) -> str:
        if rank == 1:
            return "Best overall"
        if rank <= 3:
            return "Excellent"
        if score >= 70:
            return "Good"
        if score >= 50:
            return "Moderate"
        return "Trade-off heavy"

    table_rows = []
    for rank, r in enumerate(records, 1):
        table_rows.append([
            str(rank),
            r["model"], r["strat"],
            f"{r['test_acc']:.2f}",
            f"{r['late_std']:.4f}",
            f"{r['avg_time']:.2f}",
            f"{r['score']:.1f}",
            recommendation(rank, r["score"]),
        ])

    row_groups = [{"label": "All Model x Strategy Combinations (16 total)", "rows": table_rows}]

    fig, ax = plt.subplots(figsize=(15, 12))
    render_table(ax, col_headers, row_groups,
                 best_cols=[3, 6],
                 worst_cols=[4, 5],
                 col_widths=col_widths,
                 title="Table 12 - Composite Score Ranking\n"
                       "Score = f(test accuracy, late-training stability, epoch speed), "
                       "normalised to [0, 100] across all 16 combinations.\n"
                       "Formula: (acc/100) x (1/(1+sigma*100)) x (60/avg_epoch_s). "
                       "Green = best per column. Sorted by composite score (descending).")
    fig.tight_layout(pad=0.5)
    save_fig(fig, "table12_composite_ranking", output_dir)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Loading data from : {JSON_PATH}")
    data    = load_data(JSON_PATH)
    metrics = extract_metrics(data)
    print(f"Output directory  : {OUTPUT_DIR}\n")

    # Original 6 tables
    table_final_performance(metrics, OUTPUT_DIR)
    table_convergence_speed(metrics, OUTPUT_DIR)
    table_computational_cost(metrics, OUTPUT_DIR)
    table_generalisation_gap(metrics, OUTPUT_DIR)
    table_strategy_ranking(metrics, OUTPUT_DIR)
    table_efficiency_tradeoff(metrics, OUTPUT_DIR)

    # Deep analysis tables (new)
    table_per_class_accuracy(data, metrics, OUTPUT_DIR)
    table_lr_behaviour(metrics, OUTPUT_DIR)
    table_loss_reduction(metrics, OUTPUT_DIR)
    table_late_stability(metrics, OUTPUT_DIR)
    table_peak_vs_final(metrics, OUTPUT_DIR)
    table_composite_score(metrics, OUTPUT_DIR)

    print("\nDone. 12 tables saved to:", OUTPUT_DIR)
    print(
        "\nLaTeX usage example:\n"
        "  \\begin{figure}[H]\n"
        "    \\centering\n"
        "    \\includegraphics[width=\\linewidth]{tables/table7_per_class_accuracy.png}\n"
        "    \\caption{Per-class recognition accuracy derived from the final confusion matrix.}\n"
        "    \\label{fig:table7}\n"
        "  \\end{figure}"
    )


if __name__ == "__main__":
    main()
