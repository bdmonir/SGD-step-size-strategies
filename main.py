
import torch

from config import TrainingConfig
from dataset_loader import get_dataloaders
from logger import ExperimentLogger
from models import MODEL_REGISTRY
from plotting import plot_results
from trainer import train_model


def main():
    config = TrainingConfig()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = ExperimentLogger(
        experiment_name=f"mnist_full_experiment_{config.experiment}"
    )

    logger.log(f"Using device: {config.device}")

    train_loader, test_loader = get_dataloaders(config)

    strategies = [
        "constant",
        "momentum",
        "decay",
        "armijo",
    ]

    all_results = {}

    for model_name, model_class in MODEL_REGISTRY.items():

        logger.log("=" * 70)
        logger.log(f"Training model: {model_name}")
        logger.log("=" * 70)

        # Create temporary model for parameter counting
        temp_model = model_class()

        total_params = sum(
            p.numel()
            for p in temp_model.parameters()
            if p.requires_grad
        )

        logger.log(f"Total trainable parameters: {total_params:,}")

        model_results = {
            "total_parameters": total_params,
            "strategies": {}
        }

        for strategy in strategies:

            model = model_class()

            metrics = train_model(
                strategy=strategy,
                config=config,
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                logger=logger,
            )

            model_results["strategies"][strategy] = metrics

        all_results[model_name] = model_results

        plot_results(
            model_results["strategies"],
            model_name,
            config
        )

        logger.log(f"Plots generated for {model_name}")

    logger.save_results(all_results)

    logger.log("Training completed successfully.")


if __name__ == "__main__":
    main()
