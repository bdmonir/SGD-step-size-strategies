
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import sls

from evaluation import evaluate_model


def get_optimizer(strategy, model, config):
    if strategy == "constant":
        optimizer = optim.SGD(model.parameters(), lr=config.base_lr,momentum=0.0)
        scheduler = None

    elif strategy == "decay":
        optimizer = optim.SGD(model.parameters(), lr=config.base_lr,momentum=0.0)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )

    elif strategy == "momentum":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.base_lr,
            momentum=config.momentum
        )

        scheduler = None

    elif strategy == "armijo":
        optimizer = sls.Sls(model.parameters())
        scheduler = None

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return optimizer, scheduler


def train_model(
    strategy,
    config,
    model,
    train_loader,
    test_loader,
    logger,
):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = model.to(config.device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer, scheduler = get_optimizer(
        strategy,
        model,
        config
    )

    metrics = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "times_per_epoch": [],
        "learning_rates": [],
        "confusion_matrices": [],
    }

    logger.log(f"Starting strategy: {strategy}")

    for epoch in range(config.epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"{strategy} Epoch {epoch + 1}/{config.epochs}"
        )

        for images, labels in pbar:
            images = images.to(config.device)
            labels = labels.to(config.device)

            if strategy == "armijo":

                def closure():
                    optimizer.zero_grad()

                    outputs = model(images)

                    loss = loss_fn(outputs, labels)

                    return loss

                loss = optimizer.step(closure=closure)

                with torch.no_grad():
                    outputs = model(images)

                current_lr = optimizer.state.get("step_size", 0.0)

            else:
                optimizer.zero_grad()

                outputs = model(images)

                loss = loss_fn(outputs, labels)

                loss.backward()

                optimizer.step()

                current_lr = optimizer.param_groups[0]["lr"]

            running_loss += loss.item()

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / len(train_loader)

        train_acc = 100.0 * correct / total

        test_acc, test_loss, cm = evaluate_model(
            model,
            test_loader,
            loss_fn,
            config.device
        )

        epoch_time = time.time() - start_time

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["test_loss"].append(test_loss)
        metrics["test_acc"].append(test_acc)
        metrics["times_per_epoch"].append(epoch_time)
        metrics["learning_rates"].append(current_lr)
        metrics["confusion_matrices"].append(cm)

        logger.log(
            f"{strategy} | "
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

    return metrics
