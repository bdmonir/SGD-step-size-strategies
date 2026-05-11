
import matplotlib.pyplot as plt
import config


def plot_results(results, model_name,config):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)

    for strategy, metrics in results.items():
        plt.plot(metrics["train_loss"], label=strategy)

    plt.title(f"{model_name} - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)

    for strategy, metrics in results.items():
        plt.plot(metrics["test_acc"], label=strategy)

    plt.title(f"{model_name} - Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(f"plots/{model_name}_results_{config.experiment}.png", dpi=300)

    plt.close()
