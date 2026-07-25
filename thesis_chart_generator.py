#!/usr/bin/env python3
"""
Thesis Chart Generator for MNIST Step-Size Strategy Experiments
================================================================
Generates publication-quality figures comparing learning rate strategies
(constant, momentum, decay, armijo) across different neural architectures.

Usage:
    python thesis_chart_generator.py --input mnist_full_experiment_1.log --output ./thesis_charts/

Author: Auto-generated for thesis report
"""

import argparse
import re
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec

# =============================================================================
# THESIS-QUALITY STYLING
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# Color palette - colorblind-friendly, thesis suitable
STRATEGY_COLORS = {
    'constant': '#0173B2',   # blue
    'momentum': '#DE8F05',   # orange
    'decay':    '#029E73',   # green
    'armijo':   '#D55E00',   # red
}

STRATEGY_LINESTYLES = {
    'constant': '-',
    'momentum': '--',
    'decay':    '-.',
    'armijo':   ':',
}

STRATEGY_MARKERS = {
    'constant': 'o',
    'momentum': 's',
    'decay':    '^',
    'armijo':   'D',
}

# =============================================================================
# LOG PARSER
# =============================================================================

def parse_log(filepath):
    """Parse the experiment log file into structured data."""
    with open(filepath, 'r') as f:
        content = f.read()

    data = defaultdict(lambda: defaultdict(list))
    current_model = None
    current_strategy = None

    for line in content.split('\n'):
        if line.startswith("Training model:"):
            current_model = line.split(":")[1].strip()
        elif line.startswith("Starting strategy:"):
            current_strategy = line.split(":")[1].strip()
        elif current_model and current_strategy and "|" in line and "Epoch" in line:
            try:
                epoch = int(re.search(r'Epoch (\d+)', line).group(1))
                train_loss = float(re.search(r'Train Loss: ([\d.]+)', line).group(1))
                train_acc = float(re.search(r'Train Acc: ([\d.]+)%', line).group(1))
                test_loss = float(re.search(r'Test Loss: ([\d.]+)', line).group(1))
                test_acc = float(re.search(r'Test Acc: ([\d.]+)%', line).group(1))
                lr = float(re.search(r'LR: ([\d.]+)', line).group(1))
                time = float(re.search(r'Time: ([\d.]+)s', line).group(1))

                data[current_model][current_strategy].append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'lr': lr,
                    'time': time
                })
            except Exception:
                continue

    return {k: dict(v) for k, v in data.items()}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_epochs(strategy_data):
    return [e['epoch'] for e in strategy_data]

def get_metric(strategy_data, metric):
    return [e[metric] for e in strategy_data]

def epochs_to_threshold(strategy_data, metric, threshold):
    """Return first epoch where metric crosses threshold."""
    for e in strategy_data:
        if e[metric] >= threshold:
            return e['epoch']
    return len(strategy_data)

def total_time(strategy_data):
    return sum(e['time'] for e in strategy_data)


# =============================================================================
# FIGURE 1: Training Loss Curves (One subplot per model)
# =============================================================================

def plot_training_loss(data, output_dir):
    models = list(data.keys())
    n_models = len(models)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        for strategy in data[model]:
            epochs = get_epochs(data[model][strategy])
            values = get_metric(data[model][strategy], 'train_loss')
            ax.plot(epochs, values, 
                   color=STRATEGY_COLORS[strategy],
                   linestyle=STRATEGY_LINESTYLES[strategy],
                   marker=STRATEGY_MARKERS[strategy],
                   markevery=5,
                   label=strategy.capitalize(),
                   alpha=0.9)

        ax.set_title(f'{model}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_yscale('log')
        ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
        ax.set_xlim(1, 30)

    fig.suptitle('Training Loss Convergence Across Architectures', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_training_loss_curves.png')
    plt.close(fig)
    print("[OK] Saved fig1_training_loss_curves.png")


# =============================================================================
# FIGURE 2: Test Accuracy Curves (One subplot per model)
# =============================================================================

def plot_test_accuracy(data, output_dir):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        for strategy in data[model]:
            epochs = get_epochs(data[model][strategy])
            values = get_metric(data[model][strategy], 'test_acc')
            ax.plot(epochs, values,
                   color=STRATEGY_COLORS[strategy],
                   linestyle=STRATEGY_LINESTYLES[strategy],
                   marker=STRATEGY_MARKERS[strategy],
                   markevery=5,
                   label=strategy.capitalize(),
                   alpha=0.9)

        ax.set_title(f'{model}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy (%)')
        ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
        ax.set_xlim(1, 30)
        ax.set_ylim(90, 100)

    fig.suptitle('Test Accuracy Evolution Across Architectures', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_test_accuracy_curves.png')
    plt.close(fig)
    print("[OK] Saved fig2_test_accuracy_curves.png")


# =============================================================================
# FIGURE 3: Test Loss Curves
# =============================================================================

def plot_test_loss(data, output_dir):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        for strategy in data[model]:
            epochs = get_epochs(data[model][strategy])
            values = get_metric(data[model][strategy], 'test_loss')
            ax.plot(epochs, values,
                   color=STRATEGY_COLORS[strategy],
                   linestyle=STRATEGY_LINESTYLES[strategy],
                   marker=STRATEGY_MARKERS[strategy],
                   markevery=5,
                   label=strategy.capitalize(),
                   alpha=0.9)

        ax.set_title(f'{model}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Loss')
        ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
        ax.set_xlim(1, 30)

    fig.suptitle('Test Loss Evolution Across Architectures', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_test_loss_curves.png')
    plt.close(fig)
    print("[OK] Saved fig3_test_loss_curves.png")


# =============================================================================
# FIGURE 4: Learning Rate Evolution
# =============================================================================

def plot_learning_rate(data, output_dir):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        for strategy in data[model]:
            epochs = get_epochs(data[model][strategy])
            values = get_metric(data[model][strategy], 'lr')
            # Handle zero values for log scale
            values = [max(v, 1e-8) for v in values]
            ax.plot(epochs, values,
                   color=STRATEGY_COLORS[strategy],
                   linestyle=STRATEGY_LINESTYLES[strategy],
                   marker=STRATEGY_MARKERS[strategy],
                   markevery=5,
                   label=strategy.capitalize(),
                   alpha=0.9)

        ax.set_title(f'{model}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
        ax.set_xlim(1, 30)

    fig.suptitle('Learning Rate Schedule Dynamics', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_learning_rate_dynamics.png')
    plt.close(fig)
    print("[OK] Saved fig4_learning_rate_dynamics.png")


# =============================================================================
# FIGURE 5: Time per Epoch
# =============================================================================

def plot_time_per_epoch(data, output_dir):
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        for strategy in data[model]:
            epochs = get_epochs(data[model][strategy])
            values = get_metric(data[model][strategy], 'time')
            ax.plot(epochs, values,
                   color=STRATEGY_COLORS[strategy],
                   linestyle=STRATEGY_LINESTYLES[strategy],
                   marker=STRATEGY_MARKERS[strategy],
                   markevery=5,
                   label=strategy.capitalize(),
                   alpha=0.9)

        ax.set_title(f'{model}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time per Epoch (s)')
        ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
        ax.set_xlim(1, 30)

    fig.suptitle('Computational Cost per Epoch', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_time_per_epoch.png')
    plt.close(fig)
    print("[OK] Saved fig5_time_per_epoch.png")


# =============================================================================
# FIGURE 6: Final Performance Comparison (Grouped Bar Chart)
# =============================================================================

def plot_final_comparison(data, output_dir):
    models = list(data.keys())
    strategies = ['constant', 'momentum', 'decay', 'armijo']

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.25)

    # --- Left: Final Test Accuracy ---
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(models))
    width = 0.18

    for i, strategy in enumerate(strategies):
        vals = []
        for model in models:
            vals.append(data[model][strategy][-1]['test_acc'])
        ax1.bar(x + (i - 1.5) * width, vals, width, 
                label=strategy.capitalize(),
                color=STRATEGY_COLORS[strategy],
                edgecolor='black',
                linewidth=0.5)

    ax1.set_ylabel('Final Test Accuracy (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(95, 100)
    ax1.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    ax1.set_title('(a) Final Classification Performance', fontweight='bold')
    ax1.axhline(y=99.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # --- Right: Total Training Time ---
    ax2 = fig.add_subplot(gs[0, 1])
    for i, strategy in enumerate(strategies):
        vals = []
        for model in models:
            vals.append(total_time(data[model][strategy]))
        ax2.bar(x + (i - 1.5) * width, vals, width,
                label=strategy.capitalize(),
                color=STRATEGY_COLORS[strategy],
                edgecolor='black',
                linewidth=0.5)

    ax2.set_ylabel('Total Training Time (s)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black')
    ax2.set_title('(b) Total Computational Cost', fontweight='bold')

    fig.suptitle('Aggregate Performance and Cost Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig6_final_comparison.png')
    plt.close(fig)
    print("[OK] Saved fig6_final_comparison.png")


# =============================================================================
# FIGURE 7: Convergence Speed Analysis
# =============================================================================

def plot_convergence_speed(data, output_dir):
    """Heatmap: epochs required to reach target test accuracy thresholds."""
    models = list(data.keys())
    strategies = ['constant', 'momentum', 'decay', 'armijo']

    # Define thresholds per model based on their performance
    thresholds = {
        'MLP': 97.0,
        'DeepMLP': 97.0,
        'DeepMLPBN': 98.0,
        'CNN': 99.0,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Epochs to threshold ---
    matrix_epochs = np.zeros((len(models), len(strategies)))
    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            matrix_epochs[i, j] = epochs_to_threshold(
                data[model][strategy], 'test_acc', thresholds[model]
            )

    im1 = axes[0].imshow(matrix_epochs, cmap='YlOrRd', aspect='auto')
    axes[0].set_xticks(np.arange(len(strategies)))
    axes[0].set_xticklabels([s.capitalize() for s in strategies])
    axes[0].set_yticks(np.arange(len(models)))
    axes[0].set_yticklabels(models)
    axes[0].set_title('(a) Epochs to Reach Target Accuracy', fontweight='bold')

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(strategies)):
            text = axes[0].text(j, i, f'{int(matrix_epochs[i, j])}',
                               ha="center", va="center", color="black", fontsize=10)

    cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Epochs')

    # --- Right: Time to threshold ---
    matrix_time = np.zeros((len(models), len(strategies)))
    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            epoch = epochs_to_threshold(data[model][strategy], 'test_acc', thresholds[model])
            # Sum times up to that epoch
            t = sum(data[model][strategy][k]['time'] for k in range(epoch))
            matrix_time[i, j] = t

    im2 = axes[1].imshow(matrix_time, cmap='YlOrRd', aspect='auto')
    axes[1].set_xticks(np.arange(len(strategies)))
    axes[1].set_xticklabels([s.capitalize() for s in strategies])
    axes[1].set_yticks(np.arange(len(models)))
    axes[1].set_yticklabels(models)
    axes[1].set_title('(b) Wall-Clock Time to Reach Target (s)', fontweight='bold')

    for i in range(len(models)):
        for j in range(len(strategies)):
            text = axes[1].text(j, i, f'{matrix_time[i, j]:.1f}',
                               ha="center", va="center", color="black", fontsize=9)

    cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label('Seconds')

    fig.suptitle('Convergence Speed Analysis', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig7_convergence_speed.png')
    plt.close(fig)
    print("[OK] Saved fig7_convergence_speed.png")


# =============================================================================
# FIGURE 8: Generalization Gap (Train Acc - Test Acc)
# =============================================================================

def plot_generalization_gap(data, output_dir):
    """Analyze overfitting by plotting the generalization gap."""
    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        for strategy in data[model]:
            epochs = get_epochs(data[model][strategy])
            train_vals = get_metric(data[model][strategy], 'train_acc')
            test_vals = get_metric(data[model][strategy], 'test_acc')
            gap = [t - v for t, v in zip(train_vals, test_vals)]

            ax.plot(epochs, gap,
                   color=STRATEGY_COLORS[strategy],
                   linestyle=STRATEGY_LINESTYLES[strategy],
                   marker=STRATEGY_MARKERS[strategy],
                   markevery=5,
                   label=strategy.capitalize(),
                   alpha=0.9)

        ax.set_title(f'{model}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Generalization Gap (%)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black')
        ax.set_xlim(1, 30)

    fig.suptitle('Generalization Gap: Train Accuracy − Test Accuracy', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig8_generalization_gap.png')
    plt.close(fig)
    print("[OK] Saved fig8_generalization_gap.png")


# =============================================================================
# FIGURE 9: Summary Table as Figure
# =============================================================================

def plot_summary_table(data, output_dir):
    """Generate a summary statistics table as a figure."""
    models = list(data.keys())
    strategies = ['constant', 'momentum', 'decay', 'armijo']

    # Build table data
    rows = []
    for model in models:
        for strategy in strategies:
            d = data[model][strategy]
            rows.append([
                model,
                strategy.capitalize(),
                f"{d[-1]['test_acc']:.2f}",
                f"{d[-1]['train_loss']:.4f}",
                f"{epochs_to_threshold(d, 'test_acc', 95.0)}",
                f"{sum(e['time'] for e in d):.1f}",
                f"{np.mean([e['time'] for e in d]):.2f}",
            ])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.axis('tight')

    columns = ['Model', 'Strategy', 'Final Test Acc (%)', 'Final Train Loss', 
               'Epochs to 95% Acc', 'Total Time (s)', 'Mean Epoch Time (s)']

    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = '#D9E1F2' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_edgecolor('black')

    ax.set_title('Summary Statistics: Step-Size Strategy Comparison', 
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig9_summary_table.png')
    plt.close(fig)
    print("[OK] Saved fig9_summary_table.png")


# =============================================================================
# FIGURE 10: Multi-panel Overview (Thesis cover/overview figure)
# =============================================================================

def plot_overview(data, output_dir):
    """Create a large overview figure suitable for thesis introduction."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    models = list(data.keys())
    strategies = list(data[models[0]].keys())

    # --- Row 0: MLP and DeepMLP Test Accuracy ---
    for col, model in enumerate(models[:2]):
        ax = fig.add_subplot(gs[0, col])
        for strategy in data[model]:
            epochs = get_epochs(data[model][strategy])
            values = get_metric(data[model][strategy], 'test_acc')
            ax.plot(epochs, values,
                   color=STRATEGY_COLORS[strategy],
                   linestyle=STRATEGY_LINESTYLES[strategy],
                   label=strategy.capitalize(),
                   alpha=0.9)
        ax.set_title(f'{model}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_ylim(90, 100)
        if col == 0:
            ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')

    # --- Row 0, Col 2: Final accuracy radar-like bar for all models ---
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(models))
    width = 0.18
    for i, strategy in enumerate(strategies):
        vals = [data[m][strategy][-1]['test_acc'] for m in models]
        ax.bar(x + (i - 1.5) * width, vals, width,
               label=strategy.capitalize(), color=STRATEGY_COLORS[strategy],
               edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Final Test Acc (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(95, 100)
    ax.set_title('Final Performance', fontweight='bold')

    # --- Row 1: DeepMLPBN and CNN ---
    for col, model in enumerate(models[2:], start=0):
        ax = fig.add_subplot(gs[1, col])
        for strategy in data[model]:
            epochs = get_epochs(data[model][strategy])
            values = get_metric(data[model][strategy], 'test_acc')
            ax.plot(epochs, values,
                   color=STRATEGY_COLORS[strategy],
                   linestyle=STRATEGY_LINESTYLES[strategy],
                   label=strategy.capitalize(),
                   alpha=0.9)
        ax.set_title(f'{model}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_ylim(95, 100)
        if col == 0:
            ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')

    # --- Row 1, Col 2: Training loss convergence comparison (MLP only) ---
    ax = fig.add_subplot(gs[1, 2])
    model = 'MLP'
    for strategy in data[model]:
        epochs = get_epochs(data[model][strategy])
        values = get_metric(data[model][strategy], 'train_loss')
        ax.plot(epochs, values,
               color=STRATEGY_COLORS[strategy],
               linestyle=STRATEGY_LINESTYLES[strategy],
               label=strategy.capitalize(),
               alpha=0.9)
    ax.set_title(f'{model} Training Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_yscale('log')

    # --- Row 2: Time and LR dynamics ---
    ax_time = fig.add_subplot(gs[2, 0])
    for strategy in data['CNN']:
        epochs = get_epochs(data['CNN'][strategy])
        values = get_metric(data['CNN'][strategy], 'time')
        ax_time.plot(epochs, values,
                    color=STRATEGY_COLORS[strategy],
                    linestyle=STRATEGY_LINESTYLES[strategy],
                    label=strategy.capitalize(),
                    alpha=0.9)
    ax_time.set_title('CNN Epoch Time', fontweight='bold')
    ax_time.set_xlabel('Epoch')
    ax_time.set_ylabel('Time (s)')

    ax_lr = fig.add_subplot(gs[2, 1])
    for strategy in data['CNN']:
        epochs = get_epochs(data['CNN'][strategy])
        values = [max(v, 1e-8) for v in get_metric(data['CNN'][strategy], 'lr')]
        ax_lr.plot(epochs, values,
                  color=STRATEGY_COLORS[strategy],
                  linestyle=STRATEGY_LINESTYLES[strategy],
                  label=strategy.capitalize(),
                  alpha=0.9)
    ax_lr.set_title('CNN Learning Rate', fontweight='bold')
    ax_lr.set_xlabel('Epoch')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.set_yscale('log')

    ax_gap = fig.add_subplot(gs[2, 2])
    for strategy in data['CNN']:
        epochs = get_epochs(data['CNN'][strategy])
        train_vals = get_metric(data['CNN'][strategy], 'train_acc')
        test_vals = get_metric(data['CNN'][strategy], 'test_acc')
        gap = [t - v for t, v in zip(train_vals, test_vals)]
        ax_gap.plot(epochs, gap,
                   color=STRATEGY_COLORS[strategy],
                   linestyle=STRATEGY_LINESTYLES[strategy],
                   label=strategy.capitalize(),
                   alpha=0.9)
    ax_gap.set_title('CNN Generalization Gap', fontweight='bold')
    ax_gap.set_xlabel('Epoch')
    ax_gap.set_ylabel('Gap (%)')
    ax_gap.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    fig.suptitle('Comprehensive Overview: Step-Size Strategies on MNIST', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig10_overview.png')
    plt.close(fig)
    print("[OK] Saved fig10_overview.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate thesis-quality charts from MNIST experiment logs.'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to the experiment .log file')
    parser.add_argument('--output', '-o', type=str, default='./thesis_charts',
                        help='Directory to save PNG figures')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing log file: {input_path}")
    data = parse_log(input_path)

    if not data:
        raise ValueError("No data parsed from log file. Check format.")

    print(f"Found {len(data)} models: {list(data.keys())}")
    print(f"Strategies: {list(data[list(data.keys())[0]].keys())}")
    print(f"Output directory: {output_dir}\n")

    print("Generating figures...")
    plot_training_loss(data, output_dir)
    plot_test_accuracy(data, output_dir)
    plot_test_loss(data, output_dir)
    plot_learning_rate(data, output_dir)
    plot_time_per_epoch(data, output_dir)
    plot_final_comparison(data, output_dir)
    plot_convergence_speed(data, output_dir)
    plot_generalization_gap(data, output_dir)
    plot_summary_table(data, output_dir)
    plot_overview(data, output_dir)

    print(f"\nAll figures saved to: {output_dir.absolute()}")
    print("Done!")


if __name__ == '__main__':
    main()
