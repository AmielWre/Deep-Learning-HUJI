"""Unified execution hub for Exercise 3 sentiment analysis.

This file coordinates the whole practical pipeline: data loading, model
selection, training, evaluation, plotting, checkpointing, and diagnostic
printing. It intentionally reads no command-line arguments. To change an
experiment, edit ``config.py`` or toggle the ``run`` values in ``main()``.

Call flow:
    main()
    |- get_data_loaders()
    |  |- load IMDB records and optional custom diagnostics
    |  |_ create train_loader and test_loader
    |
    |- build the experiments list
    |  |- ExRNN small / large
    |  |- ExGRU small / large
    |  |- ExMLP
    |  |_ ExLRestSelfAtten
    |
    |_ for each active experiment
       |_ run_experiment()
          |- set_seed()
          |- create a fresh model from its factory
          |_ train_and_evaluate()
             |- move model to config.DEVICE
             |- try load_model_checkpoint()
             |  |- if checkpoint exists and SKIP_TRAINING_IF_MODEL_EXISTS=True
             |  |  |- evaluate on the test set
             |  |  |_ print diagnostics
             |  |_ otherwise continue to training
             |
             |- for each epoch
             |  |- run_epoch(... train_loader ..., optimizer)
             |  |_ run_epoch(... test_loader ..., optimizer=None)
             |
             |- save loss/accuracy plots
             |- save model checkpoint when SAVE_MODELS=True
             |_ print_diagnostics()

Checkpoint behavior:
    Models are saved to ``config.OUTPUT_DIR / f"{model.name()}.pth"``. When
    ``LOAD_MODELS`` is true, existing weights are loaded before training. When
    ``SKIP_TRAINING_IF_MODEL_EXISTS`` is also true, the script skips the slow
    training loop and only runs test evaluation plus diagnostics. Set
    ``SKIP_TRAINING_IF_MODEL_EXISTS=False`` to continue training from a saved
    checkpoint.
"""

from __future__ import annotations

import random
from pathlib import Path
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
    checkpoint_path = config.OUTPUT_DIR / f"{model.name()}.pth"
    loaded_checkpoint = load_model_checkpoint(model, checkpoint_path)
    if loaded_checkpoint and config.SKIP_TRAINING_IF_MODEL_EXISTS:
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None)
        print(
            f"Using model: {model.name()} on {config.DEVICE} | "
            f"loaded checkpoint, skipped training"
        )
        print(f"Test loss {test_loss:.4f}, acc {test_acc:.4f}")
        print_diagnostics(model, test_loader)
        return

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
        torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved curves to {plot_path}")
    print_diagnostics(model, test_loader)


def load_model_checkpoint(model: nn.Module, checkpoint_path: Path) -> bool:
    """Load existing model weights when configured to do so.

    Args:
        model: Model instance whose weights should be loaded.
        checkpoint_path: Path to the saved ``state_dict``.

    Returns:
        ``True`` if a checkpoint was loaded, otherwise ``False``.
    """

    if not config.LOAD_MODELS or not checkpoint_path.exists():
        return False
    state_dict = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    return True


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Batch],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Run one train or evaluation epoch.

    Args:
        model: Sentiment model.
        loader: DataLoader of batches. Can be train_loader or test_loader.
        criterion: Loss function.
        optimizer: Optimizer for training; ``None`` for evaluation (test).

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
            if token_scores is None:  # meaning the model does not support diagnostics (RNN and GRU since they do not have per-token scores)
                continue
            for index in diagnostic_indices:
                target = int(batch.labels[index].item())
                prediction = int(logits[index].argmax().item())
                actual_outcome = diagnostic_outcome(target, prediction)
                print(
                    f"\nDiagnostic sample: intended {batch.split_names[index]} | "
                    f"actual {actual_outcome}"
                )
                print_review(
                    batch.tokens[index],
                    token_scores[index],
                    logits[index],
                    target,
                )
                printed += 1
            if printed >= len(config.CUSTOM_DIAGNOSTIC_REVIEWS):
                return


def diagnostic_outcome(target: int, prediction: int) -> str:
    """Return the TP/TN/FP/FN outcome for one binary prediction.

    Args:
        target: Ground-truth class index where 0 is negative and 1 is positive.
        prediction: Predicted class index where 0 is negative and 1 is positive.

    Returns:
        One of ``TP``, ``TN``, ``FP``, or ``FN``.
    """

    if target == 1 and prediction == 1:
        return "TP"
    if target == 0 and prediction == 0:
        return "TN"
    if target == 0 and prediction == 1:
        return "FP"
    return "FN"


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
            "run": False,
            "factory": lambda: ExRNN(config.EMBEDDING_SIZE, config.OUTPUT_SIZE, config.SMALL_HIDDEN_SIZE),
        },
        {
            "run": False,
            "factory": lambda: ExRNN(config.EMBEDDING_SIZE, config.OUTPUT_SIZE, config.LARGE_HIDDEN_SIZE),
        },
        
        # --- TASK 1: Custom GRU ---
        {
            "run": False,
            "factory": lambda: ExGRU(config.EMBEDDING_SIZE, config.OUTPUT_SIZE, config.SMALL_HIDDEN_SIZE),
        },
        {
            "run": False,
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
