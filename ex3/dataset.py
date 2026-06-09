"""Dataset and vocabulary utilities for Exercise 3.

The code in this module keeps tensors in the assignment convention:
``N_batch x N_word x N_features`` for embedded batches and
``N_batch x N_word`` for integer token batches.

Args:
    None.

Returns:
    None.

Raises:
    FileNotFoundError: If required local dataset or GloVe files are missing.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch.utils.data import Dataset


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]")


# === SHARED DATA UTILITIES ===


def tokenize(text: str) -> list[str]:
    """Tokenizes a movie review into lowercase tokens.

    Args:
        text: Raw review text.

    Returns:
        A list of lowercase tokens.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Vocabulary:
    """Maps tokens to integer ids and back.

    Args:
        token_to_id: Mapping from token string to integer id.
        pad_token: Token used for padding.
        unk_token: Token used for unknown words.

    Returns:
        None.

    Raises:
        ValueError: If required special tokens are absent.
    """

    token_to_id: dict[str, int]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    def __post_init__(self) -> None:
        """Validates the vocabulary special tokens.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If the padding or unknown token is missing.
        """
        if self.pad_token not in self.token_to_id:
            raise ValueError("pad_token is missing from token_to_id.")
        if self.unk_token not in self.token_to_id:
            raise ValueError("unk_token is missing from token_to_id.")

    @property
    def pad_id(self) -> int:
        """Returns the padding id.

        Args:
            None.

        Returns:
            The integer id of the padding token.

        Raises:
            None.
        """
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        """Returns the unknown-token id.

        Args:
            None.

        Returns:
            The integer id of the unknown token.

        Raises:
            None.
        """
        return self.token_to_id[self.unk_token]

    def __len__(self) -> int:
        """Returns the vocabulary size.

        Args:
            None.

        Returns:
            Number of vocabulary entries.

        Raises:
            None.
        """
        return len(self.token_to_id)

    def encode(self, tokens: Sequence[str], max_len: int) -> tuple[list[int], list[int]]:
        """Encodes tokens as padded ids and an attention mask.

        Args:
            tokens: Token sequence.
            max_len: Maximum encoded sequence length.

        Returns:
            A tuple ``(ids, mask)`` with both lists of length ``max_len``.

        Raises:
            ValueError: If ``max_len`` is not positive.
        """
        if max_len <= 0:
            raise ValueError("max_len must be positive.")
        clipped = list(tokens[:max_len])
        ids = [self.token_to_id.get(token, self.unk_id) for token in clipped]
        mask = [1] * len(ids)
        pad_count = max_len - len(ids)
        ids.extend([self.pad_id] * pad_count)
        mask.extend([0] * pad_count)
        return ids, mask

    def decode(self, ids: Sequence[int], mask: Sequence[int] | None = None) -> list[str]:
        """Decodes ids into token strings.

        Args:
            ids: Integer token ids.
            mask: Optional mask where ``0`` positions are skipped.

        Returns:
            Decoded token strings.

        Raises:
            None.
        """
        id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        tokens: list[str] = []
        for index, token_id in enumerate(ids):
            if mask is not None and int(mask[index]) == 0:
                continue
            tokens.append(id_to_token.get(int(token_id), self.unk_token))
        return tokens


def build_vocabulary(
    tokenized_texts: Iterable[Sequence[str]],
    max_vocab_size: int = 30_000,
    min_freq: int = 2,
) -> Vocabulary:
    """Builds a vocabulary from tokenized texts.

    Args:
        tokenized_texts: Iterable of token sequences.
        max_vocab_size: Maximum number of tokens including special tokens.
        min_freq: Minimum frequency required for a token to be included.

    Returns:
        A ``Vocabulary`` instance.

    Raises:
        ValueError: If ``max_vocab_size`` is smaller than two or ``min_freq`` is negative.
    """
    if max_vocab_size < 2:
        raise ValueError("max_vocab_size must be at least 2.")
    if min_freq < 0:
        raise ValueError("min_freq cannot be negative.")

    counter: Counter[str] = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    token_to_id = {"<pad>": 0, "<unk>": 1}
    for token, count in counter.most_common(max_vocab_size - 2):
        if count < min_freq:
            continue
        token_to_id[token] = len(token_to_id)
    return Vocabulary(token_to_id=token_to_id)


def load_imdb_split(root: Path, split: str) -> list[tuple[str, int]]:
    """Loads an IMDB split from the standard ``aclImdb`` directory layout.

    Args:
        root: Path to the ``aclImdb`` directory.
        split: Split name, either ``"train"`` or ``"test"``.

    Returns:
        A list of ``(review_text, label)`` pairs where label ``1`` is positive.

    Raises:
        FileNotFoundError: If the split directories do not exist.
        ValueError: If ``split`` is unsupported.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")
    examples: list[tuple[str, int]] = []
    for sentiment, label in (("neg", 0), ("pos", 1)):
        folder = root / split / sentiment
        if not folder.exists():
            raise FileNotFoundError(f"Missing IMDB folder: {folder}")
        for path in sorted(folder.glob("*.txt")):
            examples.append((path.read_text(encoding="utf-8", errors="replace"), label))
    return examples


def make_smoke_test_examples() -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Creates a tiny deterministic sentiment dataset for smoke tests.

    Args:
        None.

    Returns:
        Train and test examples with IMDB-like binary labels.

    Raises:
        None.
    """
    train = [
        ("this movie is good and enjoyable", 1),
        ("a wonderful film with great acting", 1),
        ("not boring and surprisingly good", 1),
        ("this movie is bad and boring", 0),
        ("a terrible film with awful acting", 0),
        ("not good and painfully dull", 0),
    ]
    test = [
        ("good acting and enjoyable story", 1),
        ("bad acting and boring story", 0),
        ("not boring", 1),
        ("not good", 0),
    ]
    return train, test


class IMDBReviewDataset(Dataset[dict[str, torch.Tensor | list[str]]]):
    """PyTorch dataset for tokenized IMDB reviews.

    Args:
        examples: Review and label pairs.
        vocab: Vocabulary used for encoding.
        max_len: Maximum token sequence length.

    Returns:
        None.

    Raises:
        ValueError: If ``max_len`` is not positive.
    """

    def __init__(self, examples: Sequence[tuple[str, int]], vocab: Vocabulary, max_len: int) -> None:
        """Initializes the dataset.

        Args:
            examples: Review and label pairs.
            vocab: Vocabulary used for encoding.
            max_len: Maximum token sequence length.

        Returns:
            None.

        Raises:
            ValueError: If ``max_len`` is not positive.
        """
        if max_len <= 0:
            raise ValueError("max_len must be positive.")
        self.examples = list(examples)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        """Returns the number of examples.

        Args:
            None.

        Returns:
            Dataset length.

        Raises:
            None.
        """
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[str]]:
        """Returns a single encoded review example.

        Args:
            index: Example index.

        Returns:
            Dictionary with ``input_ids``, ``attention_mask``, ``label``, and ``tokens``.

        Raises:
            IndexError: If ``index`` is outside the dataset.
        """
        text, label = self.examples[index]
        tokens = tokenize(text)
        ids, mask = self.vocab.encode(tokens, self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "tokens": tokens[: self.max_len],
        }


def collate_reviews(batch: Sequence[dict[str, torch.Tensor | list[str]]]) -> dict[str, torch.Tensor | list[list[str]]]:
    """Collates encoded reviews into a mini-batch.

    Args:
        batch: Sequence of dataset items.

    Returns:
        Batched tensors and token lists.

    Raises:
        ValueError: If ``batch`` is empty.
    """
    if not batch:
        raise ValueError("batch cannot be empty.")
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch if isinstance(item["input_ids"], torch.Tensor)]),
        "attention_mask": torch.stack(
            [item["attention_mask"] for item in batch if isinstance(item["attention_mask"], torch.Tensor)]
        ),
        "label": torch.stack([item["label"] for item in batch if isinstance(item["label"], torch.Tensor)]),
        "tokens": [item["tokens"] for item in batch if isinstance(item["tokens"], list)],
    }


def load_glove_embeddings(
    glove_path: Path | None,
    vocab: Vocabulary,
    embedding_dim: int,
    seed: int = 17,
) -> torch.Tensor:
    """Loads a frozen embedding matrix from a GloVe text file.

    Args:
        glove_path: Path to a GloVe ``.txt`` file, or ``None`` for a smoke-test matrix.
        vocab: Vocabulary whose rows should be initialized.
        embedding_dim: Expected embedding dimension.
        seed: Random seed used for missing words.

    Returns:
        Tensor with shape ``N_vocab x N_features``.

    Raises:
        FileNotFoundError: If ``glove_path`` is provided but absent.
        ValueError: If ``embedding_dim`` is not positive or the GloVe file has incompatible rows.
    """
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.empty(len(vocab), embedding_dim).uniform_(-0.05, 0.05, generator=generator)
    matrix[vocab.pad_id].zero_()

    if glove_path is None:
        return matrix
    if not glove_path.exists():
        raise FileNotFoundError(f"GloVe file not found: {glove_path}")

    wanted = set(vocab.token_to_id)
    with glove_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip().split(" ")
            if len(parts) <= 2:
                continue
            token = parts[0]
            if token not in wanted:
                continue
            values = parts[1:]
            if len(values) != embedding_dim:
                raise ValueError(
                    f"GloVe row {line_number} has dimension {len(values)}, expected {embedding_dim}."
                )
            matrix[vocab.token_to_id[token]] = torch.tensor([float(value) for value in values], dtype=torch.float32)
    matrix[vocab.pad_id].zero_()
    return matrix


def split_train_validation(
    examples: Sequence[tuple[str, int]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Splits examples into train and validation subsets.

    Args:
        examples: Input examples.
        validation_fraction: Fraction assigned to validation.
        seed: Random seed.

    Returns:
        Train and validation example lists.

    Raises:
        ValueError: If ``validation_fraction`` is outside ``[0, 1)``.
    """
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_size = int(len(shuffled) * validation_fraction)
    return shuffled[val_size:], shuffled[:val_size]


def iter_custom_text_batches(
    texts: Sequence[str],
    vocab: Vocabulary,
    max_len: int,
) -> Iterator[dict[str, torch.Tensor | list[list[str]]]]:
    """Yields one encoded batch for custom text diagnostics.

    Args:
        texts: Raw texts to encode.
        vocab: Vocabulary used for encoding.
        max_len: Maximum token length.

    Returns:
        Iterator yielding a single batch dictionary.

    Raises:
        ValueError: If ``texts`` is empty.
    """
    if not texts:
        raise ValueError("texts cannot be empty.")
    ids_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    tokens_list: list[list[str]] = []
    for text in texts:
        tokens = tokenize(text)
        ids, mask = vocab.encode(tokens, max_len)
        ids_list.append(torch.tensor(ids, dtype=torch.long))
        mask_list.append(torch.tensor(mask, dtype=torch.float32))
        tokens_list.append(tokens[:max_len])
    yield {
        "input_ids": torch.stack(ids_list),
        "attention_mask": torch.stack(mask_list),
        "tokens": tokens_list,
    }

