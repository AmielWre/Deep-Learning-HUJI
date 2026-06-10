"""Unified execution hub for Exercise 3 sentiment analysis.

The file is intentionally organized as independent task blocks. Edit only the
small block inside ``main()`` to choose the experiment you want to run:
Task 1 RNN, Task 1 GRU, Task 2 isolated MLP, or Task 3/4 restricted local
self-attention. All paths, hyperparameters, diagnostics, and device selection
come from ``config.py`` so no command-line arguments are required.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import Batch, get_data_loaders
from models import ExGRU, ExLRestSelfAtten, ExMLP, ExRNN
from utils import accuracy_from_logits, plot_training_history, print_review


ModelFactory = Callable[[], nn.Module]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Seed used by Python, NumPy, and PyTorch.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    model_factory: ModelFactory,
    train_loader: DataLoader[Batch],
    test_loader: DataLoader[Batch],
) -> None:
    """Create, train, evaluate, and diagnose one model.

    Args:
        model_factory: Zero-argument callable that constructs the model. The
            factory is called after seeding so comparisons start from a
            reproducible initialization.
        train_loader: Training mini-batches.
        test_loader: Evaluation mini-batches.
    """

    set_seed(config.RANDOM_SEED)
    model = model_factory()
    train_and_evaluate(model, train_loader, test_loader)


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader[Batch],
    test_loader: DataLoader[Batch],
) -> None:
    """Run a full training job and save diagnostics.

    Args:
        model: Sentiment model to train.
        train_loader: Training mini-batches.
        test_loader: Evaluation mini-batches.
    """

    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    history: dict[str, list[float]] = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    print(f"Using model: {model.name()} on {config.DEVICE}")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f}, acc {test_acc:.4f}"
        )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = config.OUTPUT_DIR / f"{model.name()}_curves.png"
    plot_training_history(history, plot_path)
    if config.SAVE_MODELS:
        torch.save(model.state_dict(), config.OUTPUT_DIR / f"{model.name()}.pth")
    print(f"Saved curves to {plot_path}")
    print_diagnostics(model, test_loader)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Batch],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Run one train or evaluation epoch.

    Args:
        model: Sentiment model.
        loader: DataLoader of batches.
        criterion: Loss function.
        optimizer: Optimizer for training; ``None`` for evaluation.

    Returns:
        Average loss and accuracy.
    """

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0
    with torch.set_grad_enabled(is_training):
        for batch in loader:
            batch = batch.to(config.DEVICE)
            logits, _, _ = model(batch.embeddings, batch.mask)
            loss = criterion(logits, batch.labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_size = int(batch.labels.numel())
            total_loss += float(loss.item()) * batch_size
            total_correct += accuracy_from_logits(logits, batch.labels) * batch_size
            total_examples += batch_size
    return total_loss / total_examples, total_correct / total_examples


def print_diagnostics(model: nn.Module, test_loader: DataLoader[Batch]) -> None:
    """Print token-level diagnostics for configured custom reviews.

    Args:
        model: Trained model.
        test_loader: Evaluation loader containing custom diagnostic samples.
    """

    model.eval()
    printed = 0
    with torch.no_grad():
        for batch in test_loader:
            diagnostic_indices = [
                index for index, name in enumerate(batch.split_names) if name != "imdb"
            ]
            if not diagnostic_indices:
                continue
            batch = batch.to(config.DEVICE)
            logits, token_scores, _ = model(batch.embeddings, batch.mask)
            if token_scores is None:
                continue
            for index in diagnostic_indices:
                print(f"\nDiagnostic sample: {batch.split_names[index]}")
                print_review(
                    batch.tokens[index],
                    token_scores[index],
                    logits[index],
                    int(batch.labels[index].item()),
                )
                printed += 1
            if printed >= len(config.CUSTOM_DIAGNOSTIC_REVIEWS):
                return


def main() -> None:
    """Select and run assignment tasks.

    The data loaders are created once and reused across every active experiment.
    Toggle the boolean values in the configurations below to control which
    tasks execute.
    """
    train_loader, test_loader, _ = get_data_loaders()

    # Define all assignment experiments cleanly in a registry
    experiments = [
        # --- TASK 1: Custom Elman RNN ---
        {
            "run": True,
            "factory": lambda: ExRNN(config.EMBEDDING_SIZE, config.OUTPUT_SIZE, config.SMALL_HIDDEN_SIZE),
        },
        {
            "run": True,
            "factory": lambda: ExRNN(config.EMBEDDING_SIZE, config.OUTPUT_SIZE, config.LARGE_HIDDEN_SIZE),
        },
        
        # --- TASK 1: Custom GRU ---
        {
            "run": True,
            "factory": lambda: ExGRU(config.EMBEDDING_SIZE, config.OUTPUT_SIZE, config.SMALL_HIDDEN_SIZE),
        },
        {
            "run": True,
            "factory": lambda: ExGRU(config.EMBEDDING_SIZE, config.OUTPUT_SIZE, config.LARGE_HIDDEN_SIZE),
        },
        
        # --- TASK 2: Per-token MLP with global average pooling ---
        {
            "run": True,
            "factory": lambda: ExMLP(config.EMBEDDING_SIZE, config.OUTPUT_SIZE, config.DEFAULT_HIDDEN_SIZE),
        },
        
        # --- TASK 4: Local self-attention + MLP head ---
        {
            "run": True,
            "factory": lambda: ExLRestSelfAtten(
                config.EMBEDDING_SIZE,
                config.OUTPUT_SIZE,
                config.DEFAULT_HIDDEN_SIZE,
                window_size=config.ATTENTION_WINDOW_SIZE,
            ),
        },
    ]

    # Execute only the active experiments
    for exp in experiments:
        if exp["run"]:
            run_experiment(exp["factory"], train_loader, test_loader)


if __name__ == "__main__":
    main()
