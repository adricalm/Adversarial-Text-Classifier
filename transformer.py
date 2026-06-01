import dataclasses
from dataclasses import dataclass

import torch
from torch import nn

from self_attention import MultiHeadSelfAttention


class PositionwiseFFN(nn.Module):
    """
    Position-wise FFN after self-attention: expand 4x, ReLU, project back.
    """

    def __init__(self, vector_dim: int, dropout_prob: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(vector_dim, 4 * vector_dim, bias=True)
        self.fc2 = nn.Linear(4 * vector_dim, vector_dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class Block(nn.Module):
    """
    Transformer encoder block.

    This version differs from the original version in  [Vaswani et al. NeurIPS 2017],
    and applies the LayerNorm before the self-attention, and before the FFN, as this
    has proved to be beneficial (see [Nguyen and Salazar 2019]).
    """

    def __init__(
        self,
        vector_dim: int,
        n_heads: int,
        block_size: int,
        dropout_prob: float,
        *,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            vector_dim, n_heads, block_size, is_causal=is_causal
        )
        self.ffn = PositionwiseFFN(vector_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(vector_dim)
        self.ln2 = nn.LayerNorm(vector_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x1 = self.ln1(x)
        x2 = x + self.dropout(self.attn(x1, key_padding_mask=key_padding_mask))
        x3 = self.ln2(x2)
        x4 = x2 + self.dropout(self.ffn(x3))
        return x4


@dataclass
class Config:
    vocab_size: int = 5000
    number_of_transformer_blocks: int = 4
    number_of_attention_heads: int = 4
    vector_dim: int = 256
    block_size: int = 512
    dropout_prob: float = 0.1
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-6
    no_of_epochs: int = 1
    num_labels: int = 2        # benign vs injection
    pad_token_id: int | None = None  # positions equal to this id are treated as padding



class BinaryClassifier(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.vector_dim)
        self.positional = nn.Parameter(
            torch.randn(1, config.block_size, config.vector_dim)
        )
        self.transformers = nn.ModuleList(
            [
                Block(
                    config.vector_dim,
                    config.number_of_attention_heads,
                    config.block_size,
                    config.dropout_prob,
                    is_causal=False,
                )
                for _ in range(config.number_of_transformer_blocks)
            ]
        )
        self.ln_f = nn.LayerNorm(config.vector_dim)
        self.classifier = nn.Linear(config.vector_dim, config.num_labels)

    @staticmethod
    def _masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.float()
        mask_counts = mask.sum(dim=1).clamp(min=1.0)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask_counts.unsqueeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Token ids of shape ``(B, S)``. Rows are typically right-padded to ``S``.
            attention_mask: Optional mask of shape ``(B, S)``. Non-zero marks valid tokens;
                padding positions must be ``0``. If omitted, inferred from ``pad_token_id``.
            pad_token_id: Overrides ``config.pad_token_id`` for masking (positions equal to
                this id are pooling padding). Unused when ``attention_mask`` is given.

        Returns:
            Logits of shape ``(B, num_labels)`` — use with ``nn.CrossEntropyLoss``.
        The sequence embedding is always a **masked mean** over non-padding positions.
        """
        _, S = x.shape
        if S > self.config.block_size:
            raise ValueError(
                f"sequence length {S} exceeds config.block_size {self.config.block_size}"
            )
        token_embeddings = self.embed(x)
        positional_embeddings = self.positional[:, :S, :]
        hidden = token_embeddings + positional_embeddings

        if attention_mask is None:
            effective_pad_id = pad_token_id
            if effective_pad_id is None:
                effective_pad_id = self.config.pad_token_id
            if effective_pad_id is None:
                attention_mask = torch.ones_like(x, dtype=torch.float32)
            else:
                attention_mask = (x != effective_pad_id).to(dtype=torch.float32)
        else:
            attention_mask = attention_mask.to(dtype=torch.float32)

        # True where padding — fed into attention to zero out those key positions.
        key_padding_mask = attention_mask == 0

        for block in self.transformers:
            hidden = block(hidden, key_padding_mask=key_padding_mask)

        pooled = self._masked_mean_pool(hidden, attention_mask)
        pooled = self.ln_f(pooled)
        return self.classifier(pooled)


    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "BinaryClassifier":
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        known = {f.name for f in dataclasses.fields(Config)}
        raw_cfg = checkpoint["config"]
        config = Config(**{k: v for k, v in raw_cfg.items() if k in known})
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print(
            f"Model loaded from {checkpoint_path} "
            f"(Epoch {checkpoint['epoch']}, iteration {checkpoint['iteration']})"
        )
        return model
