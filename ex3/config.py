"""Configuration for Exercise 3 sentiment-analysis experiments.

This module is the single source of truth for paths, hyperparameters,
hardware selection, model sizes, and diagnostic review strings. The training
script intentionally reads no command-line arguments so experiments are
reproducible by editing this file only.
"""

from pathlib import Path

import torch


BASE_DIR: Path = Path(__file__).resolve().parent
PROVIDED_DIR: Path = BASE_DIR / "provided files"
DATASET_PATH: Path = PROVIDED_DIR / "IMDB Dataset.csv"
OUTPUT_DIR: Path = BASE_DIR / "outputs"

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED: int = 7
NUM_WORKERS: int = 0

MAX_LENGTH: int = 100
EMBEDDING_SIZE: int = 100
OUTPUT_SIZE: int = 2
TRAIN_SIZE: int = 30_000
BATCH_SIZE: int = 32

SMALL_HIDDEN_SIZE: int = 64
LARGE_HIDDEN_SIZE: int = 128
DEFAULT_HIDDEN_SIZE: int = SMALL_HIDDEN_SIZE
ATTENTION_WINDOW_SIZE: int = 5

NUM_EPOCHS: int = 30
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 0.0

USE_CUSTOM_DIAGNOSTICS_IN_TEST: bool = True
SAVE_MODELS: bool = True
LOAD_MODELS: bool = True
SKIP_TRAINING_IF_MODEL_EXISTS: bool = True

LABEL_TO_INDEX: dict[str, int] = {"negative": 0, "positive": 1}
INDEX_TO_LABEL: dict[int, str] = {0: "negative", 1: "positive"}

CUSTOM_TP_REVIEWS: list[str] = [
    "A warm and funny movie with excellent acting and a wonderful ending.",
    "The story is charming, the characters are lovable, and I enjoyed every minute.",
]
CUSTOM_TN_REVIEWS: list[str] = [
    "A dull and painfully slow film with terrible dialogue and no emotional payoff.",
    "The plot is boring, the acting is weak, and the ending is a complete mess.",
]
CUSTOM_FP_REVIEWS: list[str] = [
    "The trailer looked amazing, good and exciting. It is a shame that the movie itself was none of those things.",
    "Beautiful music and bright colors cannot hide below standard and not good film.",
]
CUSTOM_FN_REVIEWS: list[str] = [
    "It starts badly, yet becomes a good and touching film.",
    "This movie is awfully good.",
]
NEGATION_SHIFT_REVIEWS: list[str] = [
    "not boring and not bad",
    "The film is not good.",
]

CUSTOM_DIAGNOSTIC_REVIEWS: list[tuple[str, str, str]] = [
    *[("TP", text, "positive") for text in CUSTOM_TP_REVIEWS],
    *[("TN", text, "negative") for text in CUSTOM_TN_REVIEWS],
    *[("FP", text, "negative") for text in CUSTOM_FP_REVIEWS],
    *[("FN", text, "positive") for text in CUSTOM_FN_REVIEWS],
    ("NEGATION_POS", NEGATION_SHIFT_REVIEWS[0], "positive"),
    ("NEGATION_NEG", NEGATION_SHIFT_REVIEWS[1], "negative"),
]