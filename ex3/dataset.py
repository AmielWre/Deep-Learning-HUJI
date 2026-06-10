"""Dataset and text-vectorization utilities .

This file converts the raw IMDB CSV (or any other CSV with similar structure) into PyTorch batches ready for the models.
The public entry point is ``get_data_loaders()``, which is called from
``main.py``. The call flow is:

``main.py`` -> ``get_data_loaders()`` -> ``load_records()`` -> ``_rows_to_records()``
-> ``ReviewRecord`` objects -> ``IMDBReviewDataset`` -> ``DataLoader`` ->
``SentimentCollator`` -> ``Batch``.

Classes:
    ``ReviewRecord``:
        Stores one raw example: review text, numeric label, and ``split_name``.
        This is for using record.review/label/split_name instead of record[0]/[1]/[2].
        Normal IMDB rows use ``split_name="imdb"``; custom diagnostic rows use
        names such as ``TP``, ``FN``, or ``NEGATION_POS`` so diagnostics can find
        them later. See ``config.CUSTOM_DIAGNOSTIC_REVIEWS`` 
    ``Batch``:
        Stores one mini-batch returned by the DataLoader. It contains labels,
        padded embeddings, padding masks, token lists, and split names.
    ``TextVectorizer``:
        The embedding maker.
        Loads GloVe and converts token lists into fixed-size tensors. Short
        reviews are padded with zero vectors here.
    ``IMDBReviewDataset``:
        A thin PyTorch ``Dataset`` wrapper around a list of ``ReviewRecord``
        objects.
    ``SentimentCollator``:
        Converts a list of ``ReviewRecord`` objects into one ``Batch`` by
        cleaning, tokenizing, embedding, padding, and stacking examples.

Shape conventions:
    labels:
        ``[batch_size]`` integer labels, where 0 is negative and 1 is positive.
    embeddings:
        ``[batch_size, MAX_LENGTH, EMBEDDING_SIZE]``. With the assignment
        default this is ``[batch_size, 100, 100]``.
    mask:
        ``[batch_size, MAX_LENGTH]``. Real token positions are 1.0 and padding
        positions are 0.0. Models use this mask to ignore padded zero vectors
        during pooling or recurrent updates.

The ``MAX_LENGTH`` rule is applied in ``tokenize()`` with
``split()[:max_length]``. Padding happens later in ``TextVectorizer.encode()``:
it first creates zero tensors of length ``MAX_LENGTH``, then fills the first
positions with GloVe vectors for the real tokens.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchtext.vocab import GloVe

import config


TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[^A-Za-z]+")
SPACE_PATTERN: re.Pattern[str] = re.compile(r"\s+")


class ReviewRecord:
    """Container for one raw review and its class label.

    Args:
        review: Raw review text.
        label: Integer label where 0 is negative and 1 is positive.
        split_name: Optional source marker used by diagnostics.
    """

    def __init__(self, review: str, label: int, split_name: str = "imdb") -> None:
        self.review = review
        self.label = label
        self.split_name = split_name


class Batch:
    """Mini-batch returned by the custom collate function.

    Args:
        labels: Tensor of integer class labels with shape ``[batch]``. 1 for positive, 0 for negative.
        embeddings: Embedded reviews with shape ``[batch, seq_len, emb_dim]``.
        mask: Float mask with shape ``[batch, seq_len]`` where real tokens are 1 and padding tokens are 0.
        tokens: Original token lists after cleaning and truncation.
        split_names: Source markers for each row.
    """

    def __init__(
        self,
        labels: torch.Tensor,
        embeddings: torch.Tensor,  # e.g. for embedding: [[[0.1, 0.2, ...], [0.3, 0.4, ... (emb_dim)], ... (seq_len)], [[0.5, 0.6, ...], [0.7, 0.8, ...], ...], ... (batch)]
        mask: torch.Tensor,  # e.g. for mask: [[1, 1, 1, 0, 0, ... (seq_len)], [1, 1, 0, 0, ...], ... (batch)]
        tokens: list[list[str]],  # List of token lists with length ``batch`` and inner lists of length at most ``seq_len``. e.g. [["a", "good", "movie", ...], ["bad", "film", "that", "i", "hated", ...], ...]
        split_names: list[str],  # List of source markers with length ``batch``. e.g. ["imdb", "imdb", "custom_tp", "custom_fn", ...] (custom for manual diagnostics, see config.CUSTOM_DIAGNOSTIC_REVIEWS, imdb for regular samples)
    ) -> None:
        self.labels = labels
        self.embeddings = embeddings
        self.mask = mask
        self.tokens = tokens
        self.split_names = split_names

    def to(self, device: torch.device) -> "Batch":
        """Move tensor fields to the requested device.

        Args:
            device: Target PyTorch device.

        Returns:
            The same batch instance with tensor fields moved in-place.
        """

        self.labels = self.labels.to(device)
        self.embeddings = self.embeddings.to(device)
        self.mask = self.mask.to(device)
        return self


def clean_review(text: str) -> str:
    """Normalize a movie review before tokenization.

    Args:
        text: Raw input review.

    Returns:
        Lower-cased alphabetic text with collapsed spaces.

    Examples:
        - "This movie is <b>great</b>! Check it out: https://example.com" ->
          "this movie is great check it out"
        - "A    bad movie with 1 star." -> "a bad movie with star"
        - "Not a good movie." -> "not good movie"
        - "A film with a single letter title: X." -> "a film with single letter title"
        - " I hate this movie! " -> "i hate this movie"
    """

    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("<br />", " ")
    text = TOKEN_PATTERN.sub(" ", text)
    text = re.sub(r"\s+[A-Za-z]\s+", " ", text)
    return SPACE_PATTERN.sub(" ", text).strip().lower()


def tokenize(text: str, max_length: int = config.MAX_LENGTH) -> list[str]:
    """Clean, split, and truncate a review.

    Args:
        text: Raw input review.
        max_length: Maximum number of tokens to keep.

    Returns:
        A token list of length at most ``max_length``.
    """

    return clean_review(text).split()[:max_length]


class TextVectorizer:
    """Convert token lists into fixed-size embedding tensors.

    The vectorizer requires torchtext GloVe vectors. Missing torchtext 
    or unavailable GloVe vectors are treated as setup errors rather than
    silently changing the representation.

    Args:
        embedding_size: Token vector width.
        max_length: Number of sequence positions per review.

    Raises:
        ValueError: If ``embedding_size`` or ``max_length`` are not positve.
        RuntimeError: If torchtext GloVe cannot be loaded.
    """

    def __init__(self, embedding_size: int, max_length: int) -> None:
        if embedding_size <= 0 or max_length <= 0:
            raise ValueError("embedding_size and max_length must be positve.")
        self.embedding_size = embedding_size
        self.max_length = max_length
        self._cache: dict[str, torch.Tensor] = {}
        self._glove = self._load_glove()

    def encode(self, tokens: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed and pad a token sequence.

        Args:
            tokens: Clean token sequence.

        Returns:
            A pair ``(embeddings, mask)`` with shapes
            ``[max_length, embedding_size]`` and ``[max_length]``.
        """

        embeddings = torch.zeros(self.max_length, self.embedding_size, dtype=torch.float32)  # Padding is here.
        mask = torch.zeros(self.max_length, dtype=torch.float32)
        for index, token in enumerate(tokens[: self.max_length]):
            embeddings[index] = self._token_vector(token)
            mask[index] = 1.0
        return embeddings, mask

    def _load_glove(self) -> object:
        """Load required GloVe vectors.

        Returns:
            A torchtext GloVe object.

        Raises:
            RuntimeError: If torchtext or the requested GloVe vectors are
                unavailable.
        """

        try:
            return GloVe(name="6B", dim=self.embedding_size)
        except Exception as error:
            raise RuntimeError(
                "Failed to load torchtext GloVe vectors. Install torchtext and "
                "make sure GloVe 6B vectors are available before running."
            ) from error

    def _token_vector(self, token: str) -> torch.Tensor:
        """Return a cached vector for one token.

        Args:
            token: Clean token string.

        Returns:
            A one-dimensional embedding tensor.
        """

        if token in self._cache:
            return self._cache[token]
        vector = self._glove.get_vecs_by_tokens([token]).squeeze(0).float()
        self._cache[token] = vector
        return vector


class IMDBReviewDataset(Dataset[ReviewRecord]):
    """PyTorch dataset for raw IMDB review records.

    Args:
        records: Sequence of review records.
    """

    def __init__(self, records: Sequence[ReviewRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        """Return the number of records."""

        return len(self.records)

    def __getitem__(self, index: int) -> ReviewRecord:
        """Fetch one record.

        Args:
            index: Dataset index.

        Returns:
            A ``ReviewRecord``.
        """

        return self.records[index]


class SentimentCollator:
    """Collate raw review records into tensors.

    Args:
        vectorizer: Text vectorizer used for token embeddings.
    """

    def __init__(self, vectorizer: TextVectorizer) -> None:
        self.vectorizer = vectorizer

    def __call__(self, records: Sequence[ReviewRecord]) -> Batch:
        """Build one mini-batch.

        Args:
            records: Raw review records sampled by a DataLoader.

        Returns:
            A tensorized ``Batch``.
        """

        labels: list[int] = []
        token_lists: list[list[str]] = []
        split_names: list[str] = []
        embeddings: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for record in records:
            tokens = tokenize(record.review)
            review_embeddings, mask = self.vectorizer.encode(tokens)
            labels.append(record.label)
            token_lists.append(tokens)
            split_names.append(record.split_name)
            embeddings.append(review_embeddings)
            masks.append(mask)
        return Batch(
            labels=torch.tensor(labels, dtype=torch.long),
            embeddings=torch.stack(embeddings),
            mask=torch.stack(masks),
            tokens=token_lists,
            split_names=split_names,
        )


def load_records(
    dataset_path: Path = config.DATASET_PATH,
    include_custom_diagnostics: bool = False,
) -> list[ReviewRecord]:
    """Load IMDB records from disk into Pandas DataFrame and then convert them to review records list.

    Args:
        dataset_path: Path to the IMDB CSV file.
        include_custom_diagnostics: Whether to append the configured custom
            diagnostic reviews using ``pd.concat``.

    Returns:
        A list of review records.

    Raises:
        FileNotFoundError: If the CSV file is missing.
        ModuleNotFoundError: If pandas is not installed.
        ValueError: If a sentiment label is unknown.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {dataset_path}")

    data = pd.read_csv(dataset_path)
    data = data[["review", "sentiment"]].copy()
    data["split_name"] = "imdb"
    if include_custom_diagnostics:
        custom_data = pd.DataFrame(
            {
                "review": [text for _, text, _ in config.CUSTOM_DIAGNOSTIC_REVIEWS],
                "sentiment": [label for _, _, label in config.CUSTOM_DIAGNOSTIC_REVIEWS],
                "split_name": [name for name, _, _ in config.CUSTOM_DIAGNOSTIC_REVIEWS],
            }
        )
        data = pd.concat([data, custom_data], ignore_index=True)
    return _rows_to_records(
        data[["review", "sentiment", "split_name"]].itertuples(index=False, name=None)
    )


def get_data_loaders() -> tuple[DataLoader[Batch], DataLoader[Batch], TextVectorizer]:
    """Create train/test loaders from ``config.py`` only.

    Returns:
        Train loader, test loader, and the shared vectorizer.
    """

    all_records: list[ReviewRecord] = load_records(
        include_custom_diagnostics=config.USE_CUSTOM_DIAGNOSTICS_IN_TEST
    )
    train_records = all_records[: config.TRAIN_SIZE]
    test_records = all_records[config.TRAIN_SIZE :]
    vectorizer = TextVectorizer(config.EMBEDDING_SIZE, config.MAX_LENGTH)
    collator = SentimentCollator(vectorizer)
    train_loader: DataLoader[Batch] = DataLoader(
        IMDBReviewDataset(train_records),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collator,
    )
    test_loader: DataLoader[Batch] = DataLoader(
        IMDBReviewDataset(test_records),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collator,
    )
    return train_loader, test_loader, vectorizer


def _rows_to_records(rows: Iterable[tuple[str, str, str]]) -> list[ReviewRecord]:
    """Convert raw CSV rows into typed records.

    Args:
        rows: Iterable of ``(review, sentiment, split_name)`` tuples.

    Returns:
        Parsed review records.

    Raises:
        ValueError: If a sentiment label is unknown.
    """

    records: list[ReviewRecord] = []
    for review, sentiment, split_name in rows:
        label_name = sentiment.strip().lower()
        if label_name not in config.LABEL_TO_INDEX:
            raise ValueError(f"Unknown sentiment label: {sentiment}")
        records.append(
            ReviewRecord(
                review=review,
                label=config.LABEL_TO_INDEX[label_name],
                split_name=split_name,
            )
        )
    return records
