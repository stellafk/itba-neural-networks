from time import perf_counter

import torch
from torch import nn


class SoftmaxRegression(nn.Linear):
    """A linear classifier returning one logit per class."""


@torch.inference_mode()
def evaluate_classifier(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, count = 0.0, 0
    predictions, targets = [], []
    for features, labels in loader:
        logits = model(features.to(device))
        loss_sum += criterion(logits, labels.to(device)).item()
        count += labels.numel()
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    predicted = torch.cat(predictions)
    actual = torch.cat(targets)
    return {
        "loss": loss_sum / count,
        "accuracy": predicted.eq(actual).float().mean().item(),
        "predictions": predicted.numpy(),
        "targets": actual.numpy(),
    }


def train_classifier(model, loader, optimizer, epochs, device, *, validation_loader=None,
                     train_evaluation_loader=None, penalty=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    history = []
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features.to(device)), labels.to(device))
            if penalty is not None:
                loss = loss + penalty(model)
            loss.backward()
            optimizer.step()
        record = {"epoch": epoch}
        for split_name, evaluation_loader in (
            ("train", train_evaluation_loader), ("val", validation_loader)
        ):
            if evaluation_loader is not None:
                metrics = evaluate_classifier(model, evaluation_loader, device)
                record[f"{split_name}_loss"] = metrics["loss"]
                record[f"{split_name}_acc"] = metrics["accuracy"]
        record["elapsed_s"] = perf_counter() - started
        history.append(record)
    return history
