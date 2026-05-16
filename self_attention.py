import math

import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, vector_dim, block_size, is_causal=False):
        super().__init__()
        self.vector_dim = vector_dim
        self.is_causal = is_causal
        self.wq = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wk = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wv = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wo = nn.Linear(vector_dim, vector_dim, bias=False)
        if self.is_causal:
            # Lower-triangular causal mask registered as a non-trainable buffer so
            # it moves to the correct device with the model parameters.
            causal_mask = torch.tril(torch.ones(block_size, block_size)).unsqueeze(0)
            self.register_buffer("mask", causal_mask)

    def compute_attention(self, q, k, v, key_padding_mask=None):
        """
        Args:
            q, k, v: query / key / value tensors.
                Single-head:  (B, S, D)
                Multi-head:   (B, n_heads, S, att_dim)
            key_padding_mask: bool tensor of shape (B, S), True where the token
                is padding. Those positions are set to -inf before softmax so
                they receive zero attention weight.
        """

        scale = math.sqrt(q.size(-1))
        scores = q @ k.transpose(-2, -1) / scale  # (..., S_q, S_k)

        if self.is_causal:
            seq_len = q.size(-2)
            scores = scores.masked_fill(
                self.mask[:, :seq_len, :seq_len] == 0, float("-inf")
            )

        if key_padding_mask is not None:
            # key_padding_mask is (B, S_k); expand to match scores by inserting
            # a leading dim for every extra batch/head dimension in scores.
            mask = key_padding_mask
            for _ in range(scores.ndim - mask.ndim):
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        return weights @ v

    def forward(self, x, key_padding_mask=None):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        values = self.compute_attention(q, k, v, key_padding_mask=key_padding_mask)
        return self.wo(values)


class MultiHeadSelfAttention(SelfAttention):
    def __init__(self, vector_dim, n_heads, block_size, is_causal=False):
        super().__init__(vector_dim, block_size, is_causal)
        self.att_dim = vector_dim // n_heads
        self.n_heads = n_heads

    def reshape_for_multihead_attention(self, x):
        """
        (batch_size, seq_length, vector_dim) ->
        (batch_size, n_heads, seq_length, att_dim)
        """
        batch_size, seq_length, _ = x.shape
        x = x.view(batch_size, seq_length, self.n_heads, self.att_dim)
        return x.transpose(1, 2)

    def reshape_after_multihead_attention(self, x):
        """
        (batch_size, n_heads, seq_length, att_dim) ->
        (batch_size, seq_length, vector_dim)
        """
        batch_size, _, seq_length, _ = x.shape
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_length, self.n_heads * self.att_dim)

    def forward(self, x, key_padding_mask=None):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self.reshape_for_multihead_attention(q)
        k = self.reshape_for_multihead_attention(k)
        v = self.reshape_for_multihead_attention(v)
        values = self.compute_attention(q, k, v, key_padding_mask=key_padding_mask)
        values = self.reshape_after_multihead_attention(values)
        return self.wo(values)
