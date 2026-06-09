"""Executable training script for Exercise 3 IMDB sentiment models.

Run examples:
    ``python ex3/main.py --smoke-test --model all --epochs 1``
    ``python ex3/main.py --imdb-root data/aclImdb --glove-path data/glove.6B.100d.txt``

Args:
    None.

Returns:
    None.

Raises:
    FileNotFoundError: If requested local IMDB or GloVe files are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .dataset import (
        IMDBReviewDataset,
        Vocabulary,
        build_vocabulary,
        collate_reviews,
        iter_custom_text_batches,
        load_glove_embeddings,
        load_imdb_split,
        make_smoke_test_examples,
        tokenize,
    )
    from .models import ContextualizedPerWordClassifier, PerWordMLPClassifier, RecurrentSentimentClassifier
    from .train_utils import print_subprediction_diagnostics, train_for_fixed_window
except ImportError:
    from dataset import (
        IMDBReviewDataset,
        Vocabulary,
        build_vocabulary,
        collate_reviews,
        iter_custom_text_batches,
        load_glove_embeddings,
        load_imdb_split,
        make_smoke_test_examples,
        tokenize,
    )
    from models import ContextualizedPerWordClassifier, PerWordMLPClassifier, RecurrentSentimentClassifier
    from train_utils import print_subprediction_diagnostics, train_for_fixed_window


# === TEMPLATE EXECUTION BLOCK: CUSTOM DIAGNOSTIC TEXTS ===
my_test_texts: list[str] = [
    "not boring",
    "not good",
    "this movie is not boring and actually good",
    "this movie is not good and painfully boring",
]


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Args:
        None.

    Returns:
        Parsed arguments.

    Raises:
        SystemExit: If argument parsing fails.
    """
    parser = argparse.ArgumentParser(description="Exercise 3 IMDB sentiment models.")
    parser.add_argument("--imdb-root", type=Path, default=None, help="Path to aclImdb root directory.")
    parser.add_argument("--glove-path", type=Path, default=None, help="Path to a GloVe text file.")
    parser.add_argument("--smoke-test", action="store_true", help="Use a tiny local dataset and random frozen embeddings.")
    parser.add_argument("--embedding-dim", type=int, default=100, help="GloVe embedding dimension.")
    parser.add_argument("--max-vocab-size", type=int, default=30_000, help="Maximum vocabulary size.")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum token frequency for vocabulary inclusion.")
    parser.add_argument("--max-len", type=int, default=256, help="Maximum review length in tokens.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Fixed training/evaluation window.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 128],
        choices=[64, 128],
        help="Task 1 recurrent hidden sizes to sweep.",
    )
    parser.add_argument(
        "--model",
        choices=["rnn", "gru", "pool", "attention", "all"],
        default="all",
        help="Model to train.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Sets deterministic random seeds where practical.

    Args:
        seed: Random seed.

    Returns:
        None.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError("seed cannot be negative.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(args: argparse.Namespace) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Loads IMDB examples according to command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Train and test example lists.

    Raises:
        FileNotFoundError: If no usable dataset source is available.
    """
    if args.smoke_test:
        return make_smoke_test_examples()
    if args.imdb_root is None:
        raise FileNotFoundError("Provide --imdb-root pointing to aclImdb, or pass --smoke-test.")
    return load_imdb_split(args.imdb_root, "train"), load_imdb_split(args.imdb_root, "test")


def make_dataloaders(
    train_examples: list[tuple[str, int]],
    test_examples: list[tuple[str, int]],
    args: argparse.Namespace,
) -> tuple[
    DataLoader[dict[str, torch.Tensor | list[list[str]]]],
    DataLoader[dict[str, torch.Tensor | list[list[str]]]],
    torch.Tensor,
    Vocabulary,
]:
    """Builds vocabulary, embeddings, datasets, and dataloaders.

    Args:
        train_examples: Training examples.
        test_examples: Test examples.
        args: Parsed command-line arguments.

    Returns:
        Train loader, test loader, embedding matrix, and vocabulary.

    Raises:
        FileNotFoundError: If a requested GloVe file is missing.
        ValueError: If dataset or embedding settings are invalid.
    """
    vocab = build_vocabulary(
        (tokenize(text) for text, _ in train_examples),
        max_vocab_size=args.max_vocab_size,
        min_freq=1 if args.smoke_test else args.min_freq,
    )
    embedding_matrix = load_glove_embeddings(
        None if args.smoke_test else args.glove_path,
        vocab=vocab,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )
    train_dataset = IMDBReviewDataset(train_examples, vocab, max_len=args.max_len)
    test_dataset = IMDBReviewDataset(test_examples, vocab, max_len=args.max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_reviews,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_reviews,
    )
    return train_loader, test_loader, embedding_matrix, vocab


def build_models(args: argparse.Namespace, embedding_matrix: torch.Tensor, pad_id: int) -> list[tuple[str, nn.Module]]:
    """Constructs requested Exercise 3 models.

    Args:
        args: Parsed command-line arguments.
        embedding_matrix: Frozen embedding matrix.
        pad_id: Padding token id.

    Returns:
        List of ``(name, model)`` pairs.

    Raises:
        ValueError: If ``args.model`` is unsupported.
    """
    selected = {"rnn", "gru", "pool", "attention"} if args.model == "all" else {args.model}
    models: list[tuple[str, nn.Module]] = []
    if "rnn" in selected:
        for hidden_dim in args.hidden_dims:
            models.append(
                (
                    f"Task1-ElmanRNN-hidden{hidden_dim}",
                    RecurrentSentimentClassifier(embedding_matrix, "rnn", hidden_dim, pad_id=pad_id),
                )
            )
    if "gru" in selected:
        for hidden_dim in args.hidden_dims:
            models.append(
                (
                    f"Task1-GRU-hidden{hidden_dim}",
                    RecurrentSentimentClassifier(embedding_matrix, "gru", hidden_dim, pad_id=pad_id),
                )
            )
    if "pool" in selected:
        models.append(("Task2-PerWordMLP-GAP", PerWordMLPClassifier(embedding_matrix, pad_id=pad_id)))
    if "attention" in selected:
        models.append(
            (
                "Task4-RestrictedAttention-PerWordMLP",
                ContextualizedPerWordClassifier(embedding_matrix, pad_id=pad_id, max_len=args.max_len),
            )
        )
    if not models:
        raise ValueError(f"Unsupported model selection: {args.model}")
    return models


def run_custom_diagnostics(model: nn.Module, vocab: Vocabulary, args: argparse.Namespace, device: torch.device) -> None:
    """Runs custom semantic-shift diagnostics for sub-prediction models.

    Args:
        model: Trained model.
        vocab: Vocabulary object.
        args: Parsed command-line arguments.
        device: Target torch device.

    Returns:
        None.

    Raises:
        AttributeError: If ``vocab`` does not implement the expected encoder.
    """
    if not hasattr(vocab, "encode"):
        raise AttributeError("vocab must provide an encode method.")
    for batch in iter_custom_text_batches(my_test_texts, vocab, max_len=args.max_len):
        print_subprediction_diagnostics(model, batch, device)


def main() -> None:
    """Runs the complete Exercise 3 training and diagnostic workflow.

    Args:
        None.

    Returns:
        None.

    Raises:
        FileNotFoundError: If requested local files are missing.
        RuntimeError: If training fails.
    """
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    train_examples, test_examples = load_examples(args)
    train_loader, test_loader, embedding_matrix, vocab = make_dataloaders(train_examples, test_examples, args)

    print(f"Loaded {len(train_examples)} train and {len(test_examples)} test examples.")
    print(f"Vocabulary size: {len(vocab)} | Frozen embedding matrix: {tuple(embedding_matrix.shape)}")

    for model_name, model in build_models(args, embedding_matrix, pad_id=vocab.pad_id):
        print(f"\n=== Training {model_name} ===")
        train_for_fixed_window(model, train_loader, test_loader, args.epochs, args.lr, device, model_name)
        if hasattr(model, "subprediction_scores"):
            print(f"\n=== Crossword Reasoning Diagnostics: {model_name} ===")
            run_custom_diagnostics(model, vocab, args, device)


if __name__ == "__main__":
    main()
