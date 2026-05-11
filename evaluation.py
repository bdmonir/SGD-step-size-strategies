
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def evaluate_model(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100.0 * (all_preds == all_labels).mean()

    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, total_loss / len(loader), cm.tolist()
