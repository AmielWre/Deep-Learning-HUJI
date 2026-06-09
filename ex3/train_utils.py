"""Training, evaluation, and diagnostics for Exercise 3.

Args:
    None.

Returns:
    None.

Raises:
    RuntimeError: If model execution fails during training or evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader


class SentimentModel(Protocol):
    """Protocol for Exercise 3 classifiers.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Computes two-class logits.

        Args:
            input_ids: Token ids with shape ``N_batch x N_word``.
            attention_mask: Valid-token mask with shape ``N_batch x N_word``.

        Returns:
            Logits with shape ``N_batch x 2``.

        Raises:
            RuntimeError: If the model cannot execute.
        """
        ...


@dataclass(frozen=True)
class EpochMetrics:
    """Stores loss and accuracy for one evaluation point.

    Args:
        train_loss: Average training loss.
        train_accuracy: Training accuracy.
        test_loss: Average test loss.
        test_accuracy: Test accuracy.

    Returns:
        None.

    Raises:
        None.
    """

    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float


def move_batch_to_device(
    batch: dict[str, torch.Tensor | list[list[str]]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Moves tensor batch fields to a device.

    Args:
        batch: Batch dictionary from the dataloader.
        device: Target torch device.

    Returns:
        ``(input_ids, attention_mask, labels)`` on ``device``.

    Raises:
        KeyError: If required tensor fields are missing.
        TypeError: If required fields are not tensors.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["label"]
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("input_ids, attention_mask, and label must be tensors.")
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor | list[list[str]]]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluates a model on a dataloader.

    Args:
        model: Sentiment classifier returning two-class logits.
        dataloader: Evaluation dataloader.
        criterion: Loss function, normally ``nn.CrossEntropyLoss``.
        device: Target torch device.

    Returns:
        Tuple ``(average_loss, accuracy)``.

    Raises:
        RuntimeError: If model evaluation fails.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = move_batch_to_device(batch, device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            batch_size = labels.numel()
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total_count += batch_size
    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor | list[list[str]]]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> None:
    """Trains a model for one epoch.

    Args:
        model: Sentiment classifier returning two-class logits.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Target torch device.
        grad_clip: Maximum gradient norm.

    Returns:
        None.

    Raises:
        ValueError: If ``grad_clip`` is not positive.
        RuntimeError: If model training fails.
    """
    if grad_clip <= 0:
        raise ValueError("grad_clip must be positive.")
    model.train()
    for batch in dataloader:
        input_ids, attention_mask, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


def train_for_fixed_window(
    model: nn.Module,
    train_loader: DataLoader[dict[str, torch.Tensor | list[list[str]]]],
    test_loader: DataLoader[dict[str, torch.Tensor | list[list[str]]]],
    epochs: int,
    lr: float,
    device: torch.device,
    model_name: str,
) -> list[EpochMetrics]:
    """Trains for a fixed evaluation window and prints assignment metrics.

    Args:
        model: Sentiment classifier returning two-class logits.
        train_loader: Training dataloader.
        test_loader: Test dataloader.
        epochs: Number of epochs, fixed to 10 by default in ``main.py``.
        lr: Adam learning rate.
        device: Target torch device.
        model_name: Human-readable model name for print statements.

    Returns:
        Per-epoch metrics.

    Raises:
        ValueError: If ``epochs`` or ``lr`` are invalid.
        RuntimeError: If training fails.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if lr <= 0:
        raise ValueError("lr must be positive.")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=lr)
    metrics: list[EpochMetrics] = []
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_loss, train_accuracy = evaluate(model, train_loader, criterion, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        metrics.append(EpochMetrics(train_loss, train_accuracy, test_loss, test_accuracy))
        print(
            f"{model_name} | Epoch {epoch:02d}/{epochs} | "
            f"Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f} | "
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
        )
    return metrics


def print_subprediction_diagnostics(
    model: nn.Module,
    batch: dict[str, torch.Tensor | list[list[str]]],
    device: torch.device,
    class_names: tuple[str, str] = ("negative", "positive"),
) -> None:
    """Prints raw per-word sub-prediction scores beside tokens.

    Args:
        model: Model exposing ``subprediction_scores``.
        batch: Encoded custom-text batch.
        device: Target torch device.
        class_names: Names for the two output logit coordinates.

    Returns:
        None.

    Raises:
        AttributeError: If the model lacks ``subprediction_scores``.
        TypeError: If required batch fields have invalid types.
    """
    if not hasattr(model, "subprediction_scores"):
        raise AttributeError("model must expose subprediction_scores for diagnostics.")
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    tokens = batch["tokens"]
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor) or not isinstance(tokens, list):
        raise TypeError("batch must contain tensor input_ids, tensor attention_mask, and list tokens.")

    model.eval()
    model.to(device)
    with torch.no_grad():
        scores = model.subprediction_scores(input_ids.to(device), attention_mask.to(device))
        logits = model(input_ids.to(device), attention_mask.to(device))
        probabilities = torch.softmax(logits, dim=-1)

    scores_cpu = scores.cpu()
    for row_index, row_tokens in enumerate(tokens):
        prediction_id = int(probabilities[row_index].argmax().item())
        print(f"\nText #{row_index + 1}: predicted={class_names[prediction_id]} probs={probabilities[row_index].cpu().tolist()}")
        print("token\tnegative_logit\tpositive_logit")
        for token_index, token in enumerate(row_tokens):
            neg_score = float(scores_cpu[row_index, token_index, 0].item())
            pos_score = float(scores_cpu[row_index, token_index, 1].item())
            print(f"{token}\t{neg_score:.4f}\t{pos_score:.4f}")

