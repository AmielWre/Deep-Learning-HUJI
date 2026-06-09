"""Neural architectures for Exercise 3 sentiment analysis.

All model inputs follow the assignment tensor convention. Token ids have shape
``N_batch x N_word`` and embeddings/contextual states have shape
``N_batch x N_word x N_features``. Recurrent models explicitly iterate over the
second axis, the word/time dimension.

Args:
    None.

Returns:
    None.

Raises:
    ValueError: If model dimensions are invalid.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


def masked_average(values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Averages sequence values over valid tokens only.

    Args:
        values: Tensor with shape ``N_batch x N_word x N_features``.
        attention_mask: Tensor with shape ``N_batch x N_word`` where nonzero entries are valid.

    Returns:
        Masked mean with shape ``N_batch x N_features``.

    Raises:
        ValueError: If tensor ranks are incompatible.
    """
    if values.ndim != 3:
        raise ValueError("values must have shape N_batch x N_word x N_features.")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape N_batch x N_word.")
    if values.shape[:2] != attention_mask.shape:
        raise ValueError("values and attention_mask disagree on batch or word dimensions.")
    mask = attention_mask.to(values.dtype).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (values * mask).sum(dim=1) / denominator


def build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, dropout: float) -> nn.Sequential:
    """Builds a feed-forward MLP.

    Args:
        input_dim: Input feature size.
        hidden_dims: Hidden layer sizes.
        output_dim: Output feature size.
        dropout: Dropout probability after hidden activations.

    Returns:
        A sequential MLP module.

    Raises:
        ValueError: If dimensions or dropout are invalid.
    """
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive.")
    if any(dim <= 0 for dim in hidden_dims):
        raise ValueError("hidden_dims must contain only positive dimensions.")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1).")

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


# === TASK 1: RECURRENT NETWORKS ===


class ElmanRNNCell(nn.Module):
    """Scratch implementation of an Elman RNN cell.

    Args:
        input_dim: Input embedding dimension.
        hidden_dim: Hidden state dimension.

    Returns:
        None.

    Raises:
        ValueError: If either dimension is non-positive.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initializes the cell.

        Args:
            input_dim: Input embedding dimension.
            hidden_dim: Hidden state dimension.

        Returns:
            None.

        Raises:
            ValueError: If either dimension is non-positive.
        """
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Advances the Elman recurrence by one token.

        Args:
            x_t: Current token embedding with shape ``N_batch x N_features``.
            h_prev: Previous hidden state with shape ``N_batch x N_hidden``.

        Returns:
            Next hidden state with shape ``N_batch x N_hidden``.

        Raises:
            ValueError: If input ranks are not two-dimensional.
        """
        if x_t.ndim != 2 or h_prev.ndim != 2:
            raise ValueError("x_t and h_prev must both be rank-2 tensors.")
        return torch.tanh(self.fc(torch.cat([x_t, h_prev], dim=-1)))


class GRUCellScratch(nn.Module):
    """Scratch implementation of a GRU cell using explicit equations.

    Args:
        input_dim: Input embedding dimension.
        hidden_dim: Hidden state dimension.

    Returns:
        None.

    Raises:
        ValueError: If either dimension is non-positive.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initializes the scratch GRU cell.

        Args:
            input_dim: Input embedding dimension.
            hidden_dim: Hidden state dimension.

        Returns:
            None.

        Raises:
            ValueError: If either dimension is non-positive.
        """
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gate_fc = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim)
        self.candidate_fc = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Advances the GRU recurrence by one token.

        Args:
            x_t: Current token embedding with shape ``N_batch x N_features``.
            h_prev: Previous hidden state with shape ``N_batch x N_hidden``.

        Returns:
            Next hidden state with shape ``N_batch x N_hidden``.

        Raises:
            ValueError: If input ranks are not two-dimensional.
        """
        if x_t.ndim != 2 or h_prev.ndim != 2:
            raise ValueError("x_t and h_prev must both be rank-2 tensors.")
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        z_t, r_t = torch.sigmoid(self.gate_fc(gate_input)).chunk(2, dim=-1)
        candidate_input = torch.cat([x_t, r_t * h_prev], dim=-1)
        h_tilde = torch.tanh(self.candidate_fc(candidate_input))
        return (1.0 - z_t) * h_prev + z_t * h_tilde


class RecurrentSentimentClassifier(nn.Module):
    """Task 1 RNN/GRU sentiment classifier with frozen GloVe embeddings.

    Args:
        embedding_matrix: Precomputed embedding weights with shape ``N_vocab x N_features``.
        cell_type: Recurrent cell type, either ``"rnn"`` or ``"gru"``.
        hidden_dim: Hidden state size, typically 64 to 128 for this assignment.
        mlp_hidden_dims: Hidden dimensions for the output MLP head.
        dropout: Dropout probability.
        pad_id: Padding token id.

    Returns:
        None.

    Raises:
        ValueError: If dimensions or cell type are invalid.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        cell_type: Literal["rnn", "gru"],
        hidden_dim: int,
        mlp_hidden_dims: tuple[int, ...] = (64,),
        dropout: float = 0.2,
        pad_id: int = 0,
    ) -> None:
        """Initializes the classifier.

        Args:
            embedding_matrix: Precomputed embedding weights.
            cell_type: Recurrent cell type.
            hidden_dim: Hidden state size.
            mlp_hidden_dims: Hidden dimensions for the output MLP head.
            dropout: Dropout probability.
            pad_id: Padding token id.

        Returns:
            None.

        Raises:
            ValueError: If embedding matrix rank, hidden size, or cell type is invalid.
        """
        super().__init__()
        if embedding_matrix.ndim != 2:
            raise ValueError("embedding_matrix must have shape N_vocab x N_features.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix.float(), freeze=True, padding_idx=pad_id)
        input_dim = embedding_matrix.shape[1]
        if cell_type == "rnn":
            self.cell: nn.Module = ElmanRNNCell(input_dim, hidden_dim)
        elif cell_type == "gru":
            self.cell = GRUCellScratch(input_dim, hidden_dim)
        else:
            raise ValueError("cell_type must be 'rnn' or 'gru'.")
        self.hidden_dim = hidden_dim
        self.classifier = build_mlp(hidden_dim, mlp_hidden_dims, 2, dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Computes two-class logits from a token batch.

        Args:
            input_ids: Token ids with shape ``N_batch x N_word``.
            attention_mask: Valid-token mask with shape ``N_batch x N_word``.

        Returns:
            Logits with shape ``N_batch x 2``.

        Raises:
            ValueError: If input shapes are invalid.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape N_batch x N_word.")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")
        embeddings = self.embedding(input_ids)
        batch_size, word_count, _ = embeddings.shape
        h_t = embeddings.new_zeros(batch_size, self.hidden_dim)
        for word_index in range(word_count):
            x_t = embeddings[:, word_index, :]
            proposed_h = self.cell(x_t, h_t)
            mask_t = attention_mask[:, word_index].to(embeddings.dtype).unsqueeze(-1)
            h_t = mask_t * proposed_h + (1.0 - mask_t) * h_t
        return self.classifier(h_t)


# === TASK 2: GLOBAL AVERAGE POOLING ===


class PerWordMLPClassifier(nn.Module):
    """Task 2 per-word MLP with global average pooling over logits.

    Args:
        embedding_matrix: Precomputed embedding weights with shape ``N_vocab x N_features``.
        word_hidden_dims: Hidden dimensions of the shared per-word MLP.
        dropout: Dropout probability.
        pad_id: Padding token id.

    Returns:
        None.

    Raises:
        ValueError: If dimensions are invalid.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        word_hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        pad_id: int = 0,
    ) -> None:
        """Initializes the pooling classifier.

        Args:
            embedding_matrix: Precomputed embedding weights.
            word_hidden_dims: Hidden dimensions of the shared per-word MLP.
            dropout: Dropout probability.
            pad_id: Padding token id.

        Returns:
            None.

        Raises:
            ValueError: If embedding matrix rank is invalid.
        """
        super().__init__()
        if embedding_matrix.ndim != 2:
            raise ValueError("embedding_matrix must have shape N_vocab x N_features.")
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix.float(), freeze=True, padding_idx=pad_id)
        self.word_scorer = build_mlp(embedding_matrix.shape[1], word_hidden_dims, 2, dropout)

    def subprediction_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Computes raw per-word two-class sub-prediction logits.

        Args:
            input_ids: Token ids with shape ``N_batch x N_word``.
            attention_mask: Optional valid-token mask, accepted for API compatibility.

        Returns:
            Per-token logits with shape ``N_batch x N_word x 2``.

        Raises:
            ValueError: If ``input_ids`` is not rank two or ``attention_mask`` has an incompatible shape.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape N_batch x N_word.")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")
        return self.word_scorer(self.embedding(input_ids))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Computes pooled two-class logits.

        Args:
            input_ids: Token ids with shape ``N_batch x N_word``.
            attention_mask: Valid-token mask with shape ``N_batch x N_word``.

        Returns:
            Logits with shape ``N_batch x 2`` after averaging raw logits.

        Raises:
            ValueError: If input shapes are invalid.
        """
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")
        per_word_logits = self.subprediction_scores(input_ids)
        return masked_average(per_word_logits, attention_mask)


# === TASK 3: RESTRICTED SELF-ATTENTION LAYER ===


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for word sequences.

    Args:
        embedding_dim: Feature dimension.
        max_len: Maximum supported sequence length.

    Returns:
        None.

    Raises:
        ValueError: If dimensions are invalid.
    """

    def __init__(self, embedding_dim: int, max_len: int = 512) -> None:
        """Initializes positional encodings.

        Args:
            embedding_dim: Feature dimension.
            max_len: Maximum supported sequence length.

        Returns:
            None.

        Raises:
            ValueError: If dimensions are invalid.
        """
        super().__init__()
        if embedding_dim <= 0 or max_len <= 0:
            raise ValueError("embedding_dim and max_len must be positive.")
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / embedding_dim))
        encodings = torch.zeros(max_len, embedding_dim, dtype=torch.float32)
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term[: encodings[:, 1::2].shape[1]])
        self.register_buffer("encodings", encodings.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encodings to an embedded batch.

        Args:
            x: Tensor with shape ``N_batch x N_word x N_features``.

        Returns:
            Tensor with positional encodings added.

        Raises:
            ValueError: If ``x`` has invalid rank or exceeds ``max_len``.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape N_batch x N_word x N_features.")
        if x.shape[1] > self.encodings.shape[1]:
            raise ValueError("sequence length exceeds positional encoding max_len.")
        return x + self.encodings[:, : x.shape[1], :].to(dtype=x.dtype)


class ExLRestSelfAtten(nn.Module):
    """Task 3 restricted single-head self-attention layer.

    The attention neighborhood contains the closest five words on each side and
    the query token itself. ``torch.roll`` is used on explicitly zero-padded key
    and value tensors to construct local neighborhoods while masking invalid
    boundary positions.

    Args:
        embedding_dim: Input and output feature dimension.
        window_size: Number of neighbors on each side of the query token.
        max_len: Maximum sequence length for positional encoding.

    Returns:
        None.

    Raises:
        ValueError: If dimensions are invalid.
    """

    def __init__(self, embedding_dim: int, window_size: int = 5, max_len: int = 512) -> None:
        """Initializes restricted self-attention.

        Args:
            embedding_dim: Input and output feature dimension.
            window_size: Number of neighbors on each side.
            max_len: Maximum sequence length for positional encoding.

        Returns:
            None.

        Raises:
            ValueError: If dimensions are invalid.
        """
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if window_size != 5:
            raise ValueError("The assignment requires window_size=5.")
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.position = PositionalEncoding(embedding_dim, max_len=max_len)
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def _local_stacks(self, tensor: torch.Tensor) -> torch.Tensor:
        """Builds a local window stack using explicit padding and ``torch.roll``.

        Args:
            tensor: Key or value tensor with shape ``N_batch x N_word x N_features``.

        Returns:
            Tensor with shape ``N_batch x N_word x 11 x N_features``.

        Raises:
            ValueError: If ``tensor`` is not rank three.
        """
        if tensor.ndim != 3:
            raise ValueError("tensor must have shape N_batch x N_word x N_features.")
        pad = self.window_size
        padded = F.pad(tensor, (0, 0, pad, pad), value=0.0)
        shifted = []
        for offset in range(-pad, pad + 1):
            rolled = torch.roll(padded, shifts=-offset, dims=1)
            shifted.append(rolled[:, pad : pad + tensor.shape[1], :])
        return torch.stack(shifted, dim=2)

    def _local_valid_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Builds a boolean local-neighborhood mask.

        Args:
            attention_mask: Valid-token mask with shape ``N_batch x N_word``.

        Returns:
            Boolean mask with shape ``N_batch x N_word x 11``.

        Raises:
            ValueError: If ``attention_mask`` is not rank two.
        """
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape N_batch x N_word.")
        pad = self.window_size
        padded = F.pad(attention_mask.to(torch.bool), (pad, pad), value=False)
        shifted = []
        for offset in range(-pad, pad + 1):
            rolled = torch.roll(padded, shifts=-offset, dims=1)
            shifted.append(rolled[:, pad : pad + attention_mask.shape[1]])
        local_mask = torch.stack(shifted, dim=2)
        query_mask = attention_mask.to(torch.bool).unsqueeze(-1)
        return local_mask & query_mask

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Applies restricted self-attention to a sequence.

        Args:
            x: Embedded input with shape ``N_batch x N_word x N_features``.
            attention_mask: Valid-token mask with shape ``N_batch x N_word``.

        Returns:
            Contextualized tensor with shape ``N_batch x N_word x N_features``.

        Raises:
            ValueError: If input shapes are invalid.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape N_batch x N_word x N_features.")
        if attention_mask.shape != x.shape[:2]:
            raise ValueError("attention_mask must match x batch and word dimensions.")

        positioned = self.position(x)
        q = self.query(positioned)
        k = self.key(positioned)
        v = self.value(positioned)
        local_k = self._local_stacks(k)
        local_v = self._local_stacks(v)
        local_mask = self._local_valid_mask(attention_mask)
        scores = (q.unsqueeze(2) * local_k).sum(dim=-1) / math.sqrt(self.embedding_dim)
        scores = scores.masked_fill(~local_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).masked_fill(~local_mask, 0.0)
        context = (weights.unsqueeze(-1) * local_v).sum(dim=2)
        return context * attention_mask.to(context.dtype).unsqueeze(-1)


# === TASK 4: CONTEXTUALIZED SENTIMENT PIPELINE ===


class ContextualizedPerWordClassifier(nn.Module):
    """Task 4 restricted-attention model feeding the Task 2 sub-predictor.

    Args:
        embedding_matrix: Precomputed embedding weights with shape ``N_vocab x N_features``.
        word_hidden_dims: Hidden dimensions of the shared per-word MLP.
        dropout: Dropout probability.
        pad_id: Padding token id.
        max_len: Maximum sequence length for positional encoding.

    Returns:
        None.

    Raises:
        ValueError: If dimensions are invalid.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        word_hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        pad_id: int = 0,
        max_len: int = 512,
    ) -> None:
        """Initializes the contextualized classifier.

        Args:
            embedding_matrix: Precomputed embedding weights.
            word_hidden_dims: Hidden dimensions of the shared per-word MLP.
            dropout: Dropout probability.
            pad_id: Padding token id.
            max_len: Maximum sequence length for positional encoding.

        Returns:
            None.

        Raises:
            ValueError: If embedding matrix rank is invalid.
        """
        super().__init__()
        if embedding_matrix.ndim != 2:
            raise ValueError("embedding_matrix must have shape N_vocab x N_features.")
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix.float(), freeze=True, padding_idx=pad_id)
        embedding_dim = embedding_matrix.shape[1]
        self.attention = ExLRestSelfAtten(embedding_dim=embedding_dim, window_size=5, max_len=max_len)
        self.word_scorer = build_mlp(embedding_dim, word_hidden_dims, 2, dropout)

    def subprediction_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Computes contextualized per-word raw logits.

        Args:
            input_ids: Token ids with shape ``N_batch x N_word``.
            attention_mask: Valid-token mask with shape ``N_batch x N_word``.

        Returns:
            Per-token logits with shape ``N_batch x N_word x 2``.

        Raises:
            ValueError: If input shapes are invalid.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape N_batch x N_word.")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")
        embeddings = self.embedding(input_ids)
        contextualized = self.attention(embeddings, attention_mask)
        return self.word_scorer(contextualized)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Computes pooled two-class logits from contextualized word scores.

        Args:
            input_ids: Token ids with shape ``N_batch x N_word``.
            attention_mask: Valid-token mask with shape ``N_batch x N_word``.

        Returns:
            Logits with shape ``N_batch x 2``.

        Raises:
            ValueError: If input shapes are invalid.
        """
        return masked_average(self.subprediction_scores(input_ids, attention_mask), attention_mask)
