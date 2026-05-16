import json
import matplotlib.pyplot as plt
import numpy as np

from config import TrainingConfig
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (hardcoded as requested)
# ─────────────────────────────────────────────────────────────────────────────
config = TrainingConfig()
JSON_FILE_PATH = f"logs/mnist_full_experiment_{config.experiment}.json"
OUTPUT_DIR = f"thesis_table_reports2/{config.experiment}"
# ─────────────────────────────────────────────────────────────────────────────


def load_json_data(filepath):
    """Load JSON experiment data."""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_model_summary(data, model_name, strategy_name):
    """Extract key metrics for a specific model and strategy."""
    model_data = data.get(model_name, {})
    strategy_data = model_data.get("strategies", {}).get(strategy_name, {})
    
    if not strategy_data:
        return None
    
    train_acc = strategy_data.get("train_acc", [])
    test_acc = strategy_data.get("test_acc", [])
    train_loss = strategy_data.get("train_loss", [])
    test_loss = strategy_data.get("test_loss", [])
    times = strategy_data.get("times_per_epoch", [])
    lrs = strategy_data.get("learning_rates", [])
    
    # Get confusion matrix for final epoch
    conf_matrices = strategy_data.get("confusion_matrices", [])
    final_conf_matrix = conf_matrices[-1] if conf_matrices else None
    
    return {
        "model": model_name,
        "strategy": strategy_name,
        "params": model_data.get("total_parameters", "N/A"),
        "final_train_acc": train_acc[-1] if train_acc else 0,
        "final_test_acc": test_acc[-1] if test_acc else 0,
        "final_train_loss": train_loss[-1] if train_loss else 0,
        "final_test_loss": test_loss[-1] if test_loss else 0,
        "best_test_acc": max(test_acc) if test_acc else 0,
        "best_test_epoch": test_acc.index(max(test_acc)) if test_acc else 0,
        "total_time": sum(times) if times else 0,
        "avg_time_per_epoch": np.mean(times) if times else 0,
        "learning_rates": lrs,
        "final_confusion_matrix": final_conf_matrix,
        "train_acc_history": train_acc,
        "test_acc_history": test_acc,
        "train_loss_history": train_loss,
        "test_loss_history": test_loss,
    }

def generate_table_figures(data, output_dir):
    """Generate publication-ready tables for LaTeX."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    models = ["MLP", "DeepMLP", "DeepMLPBN", "CNN"]
    strategies = ["constant", "momentum", "decay", "armijo"]
    
    all_summaries = []
    
    for model in models:
        for strategy in strategies:
            summary = get_model_summary(data, model, strategy)
            if summary:
                all_summaries.append(summary)
    
    # ============================================================
    # TABLE 1: Final Performance Summary
    # ============================================================
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    ax1.axis('tight')
    ax1.axis('off')
    
    table_data = []
    headers = ['Model', 'Optimizer', 'Params', 'Final Train Acc (%)', 
               'Final Test Acc (%)', 'Best Test Acc (%)', 'Best Epoch', 
               'Total Time (s)', 'Avg Time/Epoch (s)']
    
    for s in all_summaries:
        row = [
            s['model'],
            s['strategy'].capitalize(),
            f"{s['params']:,}",
            f"{s['final_train_acc']:.2f}",
            f"{s['final_test_acc']:.2f}",
            f"{s['best_test_acc']:.2f}",
            s['best_test_epoch'],
            f"{s['total_time']:.1f}",
            f"{s['avg_time_per_epoch']:.2f}"
        ]
        table_data.append(row)
    
    table1 = ax1.table(cellText=table_data, colLabels=headers, 
                       cellLoc='center', loc='center',
                       colWidths=[0.12, 0.10, 0.12, 0.12, 0.12, 0.12, 0.08, 0.10, 0.12])
    table1.auto_set_font_size(False)
    table1.set_fontsize(8)
    table1.scale(1.2, 1.5)
    
    # Color coding for best per model-strategy combination
    for model in models:
        model_rows = [i for i, s in enumerate(all_summaries) if s['model'] == model]
        if model_rows:
            best_test_idx = max(model_rows, key=lambda i: all_summaries[i]['best_test_acc'])
            for j in range(1, len(headers)):
                table1[(best_test_idx + 1, j)].set_facecolor('#90EE90')
    
    ax1.set_title('Table 1: Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/table1_performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # TABLE 2: Learning Rate Analysis (for decay and armijo)
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.axis('tight')
    ax2.axis('off')
    
    lr_data = []
    lr_headers = ['Model', 'Strategy', 'Initial LR', 'Final LR', 
                  'LR Reduction Factor', 'LR Schedule Summary']
    
    for s in all_summaries:
        if s['strategy'] in ['Decay', 'Armijo']:
            lrs = s['learning_rates']
            if lrs and len(lrs) > 0:
                initial_lr = lrs[0]
                final_lr = lrs[-1]
                reduction = initial_lr / final_lr if final_lr > 0 else float('inf')
                
                # Count unique LR changes
                unique_lrs = len(set([round(lr, 10) for lr in lrs]))
                
                schedule_summary = f"{unique_lrs} changes" if unique_lrs < len(lrs) else "Constant (adaptive)"
                
                lr_data.append([
                    s['model'], s['strategy'], 
                    f"{initial_lr:.2e}" if initial_lr < 0.01 else f"{initial_lr:.4f}",
                    f"{final_lr:.2e}" if final_lr < 0.01 else f"{final_lr:.4f}",
                    f"{reduction:.1f}x" if reduction != float('inf') else "N/A",
                    schedule_summary
                ])
    
    if lr_data:
        table2 = ax2.table(cellText=lr_data, colLabels=lr_headers,
                           cellLoc='center', loc='center',
                           colWidths=[0.12, 0.12, 0.12, 0.12, 0.15, 0.25])
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1.2, 1.5)
        ax2.set_title('Table 2: Learning Rate Analysis for Adaptive Strategies', 
                      fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/table2_lr_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # TABLE 3: Model Size vs Best Accuracy
    # ============================================================
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.axis('tight')
    ax3.axis('off')
    
    size_acc_data = []
    size_acc_headers = ['Model', 'Parameters', 'Best Test Acc (Momentum)', 
                        'Best Test Acc (Armijo)', 'Best Overall', 'Best Strategy']
    
    for model in models:
        model_summaries = [s for s in all_summaries if s['model'] == model]
        if model_summaries:
            params = model_summaries[0]['params']
            momentum_acc = next((s['best_test_acc'] for s in model_summaries if s['strategy'] == 'Momentum'), 'N/A')
            armijo_acc = next((s['best_test_acc'] for s in model_summaries if s['strategy'] == 'Armijo'), 'N/A')
            
            best_overall = max(model_summaries, key=lambda x: x['best_test_acc'])
            best_acc = best_overall['best_test_acc']
            best_strat = best_overall['strategy']
            
            size_acc_data.append([
                model, f"{params:,}", 
                f"{momentum_acc:.2f}" if isinstance(momentum_acc, float) else momentum_acc,
                f"{armijo_acc:.2f}" if isinstance(armijo_acc, float) else armijo_acc,
                f"{best_acc:.2f}", best_strat
            ])
    
    table3 = ax3.table(cellText=size_acc_data, colLabels=size_acc_headers,
                       cellLoc='center', loc='center',
                       colWidths=[0.15, 0.20, 0.15, 0.15, 0.15, 0.15])
    table3.auto_set_font_size(False)
    table3.set_fontsize(10)
    table3.scale(1.2, 1.5)
    ax3.set_title('Table 3: Model Size vs. Best Test Accuracy', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/table3_size_vs_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # TABLE 4: Confusion Matrix Summary (Classification Errors)
    # ============================================================
    fig4, ax4 = plt.subplots(figsize=(14, 10))
    ax4.axis('tight')
    ax4.axis('off')
    
    conf_data = []
    conf_headers = ['Model', 'Strategy', 'Most Confused Pair', 'Error Count', 
                    'Total Errors', 'Per-Class Accuracy Range']
    
    for s in all_summaries:
        if s['final_confusion_matrix']:
            cm = np.array(s['final_confusion_matrix'])
            
            # Find most confused pair (off-diagonal max)
            off_diag = cm.copy()
            np.fill_diagonal(off_diag, 0)
            max_error = off_diag.max()
            error_positions = np.where(off_diag == max_error)
            most_confused = f"{error_positions[0][0]}→{error_positions[1][0]}" if len(error_positions[0]) > 0 else "None"
            
            total_errors = cm.sum() - np.trace(cm)
            per_class_acc = [cm[i, i] / cm[i, :].sum() * 100 for i in range(10)]
            acc_range = f"{min(per_class_acc):.1f}-{max(per_class_acc):.1f}%"
            
            conf_data.append([
                s['model'], s['strategy'], most_confused, 
                int(max_error), int(total_errors), acc_range
            ])
    
    table4 = ax4.table(cellText=conf_data, colLabels=conf_headers,
                       cellLoc='center', loc='center',
                       colWidths=[0.12, 0.10, 0.15, 0.10, 0.12, 0.20])
    table4.auto_set_font_size(False)
    table4.set_fontsize(9)
    table4.scale(1.2, 1.5)
    ax4.set_title('Table 4: Confusion Matrix Analysis (Final Epoch)', 
                  fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/table4_confusion_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # TABLE 5: Convergence Speed Comparison
    # ============================================================
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    ax5.axis('tight')
    ax5.axis('off')
    
    convergence_data = []
    conv_headers = ['Model', 'Strategy', 'Epochs to 95%', 'Epochs to 98%', 
                    'Epochs to 99%', 'Final Accuracy']
    
    for s in all_summaries:
        train_acc = s['train_acc_history']
        epochs_to_95 = next((i for i, acc in enumerate(train_acc) if acc >= 95), len(train_acc))
        epochs_to_98 = next((i for i, acc in enumerate(train_acc) if acc >= 98), len(train_acc))
        epochs_to_99 = next((i for i, acc in enumerate(train_acc) if acc >= 99), len(train_acc))
        
        convergence_data.append([
            s['model'], s['strategy'], epochs_to_95, epochs_to_98, 
            epochs_to_99, f"{s['final_train_acc']:.2f}%"
        ])
    
    table5 = ax5.table(cellText=convergence_data, colLabels=conv_headers,
                       cellLoc='center', loc='center',
                       colWidths=[0.12, 0.10, 0.13, 0.13, 0.13, 0.15])
    table5.auto_set_font_size(False)
    table5.set_fontsize(9)
    table5.scale(1.2, 1.5)
    ax5.set_title('Table 5: Convergence Speed Analysis', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/table5_convergence_speed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated 5 table figures in '{output_dir}/'")
    print("   - table1_performance_summary.png")
    print("   - table2_lr_analysis.png")
    print("   - table3_size_vs_accuracy.png")
    print("   - table4_confusion_analysis.png")
    print("   - table5_convergence_speed.png")

def main():
    """Main execution function."""
    print("=" * 60)
    print("MNIST Training Report Table Generator")
    print("=" * 60)
    print(f"Loading data from: {JSON_FILE_PATH}")
    
    try:
        data = load_json_data(JSON_FILE_PATH)
        print("✓ Data loaded successfully")
        
        generate_table_figures(data, OUTPUT_DIR)
        
        print("\n" + "=" * 60)
        print("✅ Report generation complete!")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"❌ Error: File '{JSON_FILE_PATH}' not found!")
        print("Please update JSON_FILE_PATH variable with the correct location.")
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()