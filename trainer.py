import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

import sls

from evaluation import evaluate_model


def get_optimizer(strategy, model, config):

    if strategy == "constant":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.base_lr,
            momentum=0.0
        )
        scheduler = None

    elif strategy == "momentum":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.base_lr,
            momentum=config.momentum
        )
        scheduler = None

    elif strategy == "StepLR":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.base_lr,
            momentum=config.momentum
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )

    elif strategy == "ExponentialLR":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.base_lr,
            momentum=config.momentum
        )

        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )

    elif strategy == "CosineAnnealingLR":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.base_lr,
            momentum=config.momentum
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=1e-6
        )

    elif strategy == "armijo":
        optimizer = sls.Sls(model.parameters())
        scheduler = None

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return optimizer, scheduler


def compute_gradient_norm(model):

    grads = [
        p.grad.norm()
        for p in model.parameters()
        if p.grad is not None
    ]

    if len(grads) == 0:
        return 0.0

    grad_norm = torch.norm(
        torch.stack(grads)
    ).item()

    return grad_norm


def compute_update_norm(old_params, model):

    update_norm = 0.0

    for old_p, new_p in zip(old_params, model.parameters()):

        update_norm += (
            new_p.detach() - old_p
        ).norm().item()

    return update_norm


def train_model(
    strategy,
    config,
    model,
    train_loader,
    test_loader,
    logger,
):

    # Reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = model.to(config.device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer, scheduler = get_optimizer(
        strategy,
        model,
        config
    )

    metrics = {

        # Epoch metrics
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],

        "precision": [],
        "recall": [],
        "f1_score": [],

        "generalization_gap_acc": [],
        "generalization_gap_loss": [],

        "learning_rates": [],
        "times_per_epoch": [],

        "train_confusion_matrices": [],
        "test_confusion_matrices": [],

        "per_class_accuracy": [],
       
    }

    logger.log(f"Starting strategy: {strategy}")

    total_forward_passes = 0
    total_backward_passes = 0

    reached_90 = False
    reached_95 = False

    for epoch in range(config.epochs):

        model.train()

        epoch_start_time = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"{strategy} Epoch {epoch + 1}/{config.epochs}"
        )

        for images, labels in pbar:

            batch_start_time = time.time()

            images = images.to(config.device)
            labels = labels.to(config.device)

            old_params = [
                p.detach().clone()
                for p in model.parameters()
            ]

            if strategy == "armijo":

                def closure():

                    optimizer.zero_grad()

                    with torch.enable_grad():

                        outputs = model(images)

                        loss = loss_fn(
                            outputs,
                            labels
                        )

                    return loss

                loss = optimizer.step(
                    closure=closure
                )

                loss_value = (
                    loss.item()
                    if loss is not None
                    else 0.0
                )

                current_lr = optimizer.state.get(
                    "step_size",
                    0.0
                )

                

            else:

                optimizer.zero_grad()

                outputs = model(images)

                loss = loss_fn(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

                loss_value = loss.item()

                current_lr = optimizer.param_groups[0]["lr"]

                total_forward_passes += 1
                total_backward_passes += 1

            grad_norm = compute_gradient_norm(
                model
            )

            update_norm = compute_update_norm(
                old_params,
                model
            )


            

        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = (
            time.time() - epoch_start_time
        )

        # Evaluation
        (
            train_acc,
            train_loss,
            train_cm,
            _,
            _,
        ) = evaluate_model(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=config.device
        )

        (
            test_acc,
            test_loss,
            test_cm,
            test_preds,
            test_labels,
        ) = evaluate_model(
            model=model,
            loader=test_loader,
            loss_fn=loss_fn,
            device=config.device
        )

        precision = precision_score(
            test_labels,
            test_preds,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            test_labels,
            test_preds,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            test_labels,
            test_preds,
            average="macro",
            zero_division=0
        )

        per_class_acc = (
            np.diag(test_cm)
            / np.maximum(
                test_cm.sum(axis=1),
                1
            )
        ).tolist()

       

        # Store epoch metrics
        metrics["train_loss"].append(
            train_loss
        )

        metrics["train_acc"].append(
            train_acc
        )

        metrics["test_loss"].append(
            test_loss
        )

        metrics["test_acc"].append(
            test_acc
        )

        metrics["precision"].append(
            precision
        )

        metrics["recall"].append(
            recall
        )

        metrics["f1_score"].append(
            f1
        )

        metrics["generalization_gap_acc"].append(
            train_acc - test_acc
        )

        metrics["generalization_gap_loss"].append(
            train_loss - test_loss
        )

        metrics["learning_rates"].append(
            current_lr
        )

        metrics["times_per_epoch"].append(
            epoch_time
        )

        metrics["train_confusion_matrices"].append(
            train_cm.tolist()
        )

        metrics["test_confusion_matrices"].append(
            test_cm.tolist()
        )

        metrics["per_class_accuracy"].append(
            per_class_acc
        )

        logger.log(
            f"{strategy} | "
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}% | "
            f"F1: {f1:.4f} | "
            f"GradNorm: {grad_norm:.4f} | "
            f"UpdateNorm: {update_norm:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Epoch Time: {epoch_time:.2f}s"
        )

    return metrics