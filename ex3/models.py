"""Neural network architectures for Exercise 3 sentiment analysis.

The module contains the custom Elman RNN, custom GRU, per-token MLP, and the
restricted local self-attention model requested by the assignment. All models
inherit from ``torch.nn.Module`` and expose a common forward interface.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class MatMul(nn.Module):
    """Matrix multiplication layer for arbitrary leading dimensions.

    Args:
        in_channels: Size of the final input dimension.
        out_channels: Size of the final output dimension.
        use_bias: Whether to add a learnable bias.
    """

    def __init__(self, in_channels: int, out_channels: int, use_bias: bool = True) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_normal_(self.matrix)
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1, 1, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the final dimension of ``x``.

        Args:
            x: Input tensor with final dimension ``in_channels``.

        Returns:
            Tensor with final dimension ``out_channels``.
        """

        output = torch.matmul(x, self.matrix)
        if self.use_bias:
            output = output + self.bias
        return output


class MaskedLogitAveraging(nn.Module):
    """Average token logits over non-padding positions.

    Notes:
        The averaging is deliberately done in logit space before softmax.
        Averaging probabilities would mix already-normalized distributions and
        can over-emphasize locally confident but contradictory token decisions.
        Cross-entropy expects unnormalized class logits, so the global pooling
        block returns logits and lets the loss or diagnostic code apply softmax.
    """

    def forward(self, token_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute global average pooled logits.

        Args:
            token_logits: Per-token scores with shape ``[batch, seq_len, classes]``.
            mask: Padding mask with shape ``[batch, seq_len]``.

        Returns:
            Review-level logits with shape ``[batch, classes]``.
        """

        expanded_mask = mask.unsqueeze(-1)
        denominator = expanded_mask.sum(dim=1).clamp_min(1.0)
        return (token_logits * expanded_mask).sum(dim=1) / denominator


class ExRNN(nn.Module):
    """Custom Elman RNN classifier implemented from first principles.

    Args:
        input_size: Word embedding size.
        output_size: Number of sentiment classes.
        hidden_size: Recurrent hidden-state size.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def name(self) -> str:
        """Return a short model name."""

        return f"rnn_h{self.hidden_size}"

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        """Create a zero initial hidden state.

        Args:
            batch_size: Batch size.
            device: Optional target device.

        Returns:
            Hidden-state tensor with shape ``[batch_size, hidden_size]``.
        """

        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Classify a batch of embedded reviews.

        Args:
            x: Embedded reviews with shape ``[batch, seq_len, input_size]``.
            mask: Optional padding mask. Recurrent models keep the last valid
                hidden state when padding is encountered.

        Returns:
            A tuple ``(logits, None, None)``.
        """

        hidden = self.init_hidden(x.size(0), x.device)
        for time_step in range(x.size(1)):
            candidate = torch.tanh(self.in2hidden(torch.cat([x[:, time_step, :], hidden], dim=1)))
            if mask is None:
                hidden = candidate
            else:
                keep = mask[:, time_step].unsqueeze(1)
                hidden = candidate * keep + hidden * (1.0 - keep)
        return self.classifier(hidden), None, None


class ExGRU(nn.Module):
    """Custom GRU classifier implemented from first principles.

    Args:
        input_size: Word embedding size.
        output_size: Number of sentiment classes.
        hidden_size: Recurrent hidden-state size.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        gate_size = input_size + hidden_size
        self.reset_gate = nn.Linear(gate_size, hidden_size)
        self.update_gate = nn.Linear(gate_size, hidden_size)
        self.candidate_state = nn.Linear(gate_size, hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def name(self) -> str:
        """Return a short model name."""

        return f"gru_h{self.hidden_size}"

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        """Create a zero initial hidden state.

        Args:
            batch_size: Batch size.
            device: Optional target device.

        Returns:
            Hidden-state tensor with shape ``[batch_size, hidden_size]``.
        """

        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Classify a batch of embedded reviews.

        Args:
            x: Embedded reviews with shape ``[batch, seq_len, input_size]``.
            mask: Optional padding mask.

        Returns:
            A tuple ``(logits, None, None)``.
        """

        hidden = self.init_hidden(x.size(0), x.device)
        for time_step in range(x.size(1)):
            current = x[:, time_step, :]
            gate_input = torch.cat([current, hidden], dim=1)
            reset = torch.sigmoid(self.reset_gate(gate_input))
            update = torch.sigmoid(self.update_gate(gate_input))
            candidate_input = torch.cat([current, reset * hidden], dim=1)
            candidate = torch.tanh(self.candidate_state(candidate_input))
            next_hidden = (1.0 - update) * hidden + update * candidate
            if mask is None:
                hidden = next_hidden
            else:
                keep = mask[:, time_step].unsqueeze(1)
                hidden = next_hidden * keep + hidden * (1.0 - keep)
        return self.classifier(hidden), None, None


class ExMLP(nn.Module):
    """Per-word MLP with explicit global average pooling.

    Args:
        input_size: Word embedding size.
        output_size: Number of sentiment classes.
        hidden_size: Token-level hidden size.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.scorer = MatMul(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.pool = MaskedLogitAveraging()

    def name(self) -> str:
        """Return a short model name."""

        return f"mlp_h{self.hidden_size}"

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Classify reviews from isolated token predictions.

        Args:
            x: Embedded reviews with shape ``[batch, seq_len, input_size]``.
            mask: Padding mask with shape ``[batch, seq_len]``.

        Returns:
            Tuple of review logits, token logits, and no attention weights.
        """

        hidden = self.activation(self.layer1(x))
        hidden = self.activation(self.layer2(hidden))
        token_logits = self.scorer(hidden)
        return self.pool(token_logits, mask), token_logits, None


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for fixed-length reviews.

    Args:
        embedding_size: Encoding width.
        max_length: Maximum sequence length.
    """

    def __init__(self, embedding_size: int, max_length: int) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisors = torch.exp(
            torch.arange(0, embedding_size, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / embedding_size)
        )
        encoding = torch.zeros(max_length, embedding_size, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * divisors)
        encoding[:, 1::2] = torch.cos(positions * divisors[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional information to embedded tokens.

        Args:
            x: Embedded tokens with shape ``[batch, seq_len, embedding_size]``.

        Returns:
            Position-aware token embeddings.
        """

        return x + self.encoding[:, : x.size(1), :]


class ExLRestSelfAtten(nn.Module):
    """Restricted local self-attention followed by per-token scoring.

    Each token attends only to the closest ``window_size`` tokens on both sides,
    giving an 11-token window for the assignment default of 5.

    Args:
        input_size: Word embedding size.
        output_size: Number of sentiment classes.
        hidden_size: Hidden size for token features and Q/K/V projections.
        window_size: Number of neighbors on each side.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        window_size: int = config.ATTENTION_WINDOW_SIZE,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.position = PositionalEncoding(input_size, config.MAX_LENGTH)
        self.layer1 = MatMul(input_size, hidden_size)
        self.query = MatMul(hidden_size, hidden_size, use_bias=False)
        self.key = MatMul(hidden_size, hidden_size, use_bias=False)
        self.value = MatMul(hidden_size, hidden_size, use_bias=False)
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.scorer = MatMul(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.pool = MaskedLogitAveraging()

    def name(self) -> str:
        """Return a short model name."""

        return f"local_attention_h{self.hidden_size}_w{self.window_size}"

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Classify reviews with restricted local self-attention.

        Args:
            x: Embedded reviews with shape ``[batch, seq_len, input_size]``.
            mask: Padding mask with shape ``[batch, seq_len]``.

        Returns:
            Tuple of review logits, token logits, and local attention weights
            with shape ``[batch, seq_len, 2 * window_size + 1]``.
        """

        hidden = self.activation(self.layer1(self.position(x)))
        local_context, attention_weights = self._local_attention(hidden, mask)
        token_hidden = self.activation(self.layer2(local_context))
        token_logits = self.scorer(token_hidden)
        return self.pool(token_logits, mask), token_logits, attention_weights

    def _local_attention(self, hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply sliding-window attention using pad and roll.

        Args:
            hidden: Token features with shape ``[batch, seq_len, hidden_size]``.
            mask: Padding mask with shape ``[batch, seq_len]``.

        Returns:
            Contextual token features and attention weights.
        """

        queries = self.query(hidden)
        keys = self.key(hidden)
        values = self.value(hidden)
        key_windows = self._local_windows(keys)
        value_windows = self._local_windows(values)
        mask_windows = self._local_windows(mask.unsqueeze(-1)).squeeze(-1)
        scores = (queries.unsqueeze(2) * key_windows).sum(dim=-1) / math.sqrt(self.hidden_size)
        scores = scores.masked_fill(mask_windows <= 0.0, -1e9)
        attention_weights = F.softmax(scores, dim=2)
        context = (attention_weights.unsqueeze(-1) * value_windows).sum(dim=2)
        return context * mask.unsqueeze(-1), attention_weights

    def _local_windows(self, x: torch.Tensor) -> torch.Tensor:
        """Build local windows around every sequence position.

        Args:
            x: Tensor with shape ``[batch, seq_len, channels]``.

        Returns:
            Tensor with shape ``[batch, seq_len, 2 * window_size + 1, channels]``.
        """

        padded = F.pad(x, (0, 0, self.window_size, self.window_size))
        shifted = []
        for offset in range(-self.window_size, self.window_size + 1):
            rolled = torch.roll(padded, shifts=-offset, dims=1)
            shifted.append(rolled[:, self.window_size : -self.window_size, :])
        return torch.stack(shifted, dim=2)
