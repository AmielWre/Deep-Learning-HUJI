"""Visualization and diagnostic helpers for Exercise 3."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch

import config


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy.

    Args:
        logits: Model outputs with shape ``[batch, classes]``.
        labels: Integer class labels with shape ``[batch]``.

    Returns:
        Accuracy as a Python float in ``[0, 1]``.
    """

    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def plot_training_history(history: dict[str, list[float]], output_path: Path) -> None:
    """Save loss and accuracy curves to disk.

    Args:
        history: Metric lists keyed by ``train_loss``, ``test_loss``,
            ``train_acc``, and ``test_acc``.
        output_path: Destination PNG path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["test_loss"], label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["test_acc"], label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def print_review(
    tokens: Sequence[str],
    token_scores: torch.Tensor,
    logits: torch.Tensor,
    target: int,
    max_tokens: int = 30,
) -> None:
    """Print token-level diagnostic scores for one review.

    Args:
        tokens: Tokenized review text.
        token_scores: Raw per-token logits with shape ``[seq_len, 2]``.
        logits: Review-level logits with shape ``[2]``.
        target: Ground-truth integer label.
        max_tokens: Maximum number of tokens to display.
    """

    probabilities = torch.softmax(logits.detach().cpu(), dim=0)
    scores = token_scores.detach().cpu()
    prediction = int(probabilities.argmax().item())
    print("\nReview diagnostic")
    print(f"Target: {config.INDEX_TO_LABEL[target]} | Prediction: {config.INDEX_TO_LABEL[prediction]}")
    print(f"P(negative): {probabilities[0]:.4f} | P(positive): {probabilities[1]:.4f}")
    print("-" * 58)
    print(f"{'idx':>3}  {'token':<22} {'neg_logit':>10} {'pos_logit':>10}")
    print("-" * 58)
    for index, token in enumerate(tokens[:max_tokens]):
        print(f"{index:>3}  {token:<22.22} {scores[index, 0]:>10.4f} {scores[index, 1]:>10.4f}")
    if len(tokens) > max_tokens:
        print(f"... {len(tokens) - max_tokens} more tokens")
    print("-" * 58)
