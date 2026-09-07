"""Lab 2 starter: scaled dot-product attention and multi-head attention."""

import math
import torch
import torch.nn as nn


def attention(q, k, v, mask=None):
    """Compute scaled dot-product attention."""

    # 1. Compare each query with every key
    scores = torch.matmul(q, k.transpose(-2, -1))

    # 2. Scale the scores by sqrt(d_k)
    d_k = q.size(-1)
    scores = scores / math.sqrt(d_k)

    # 3. Apply a mask if one is provided
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask

    # 4. Convert scores into attention probabilities
    weights = torch.softmax(scores, dim=-1)

    # 5. Use those weights to combine the values
    output = torch.matmul(weights, v)

    return output



class MultiHeadAttention(nn.Module):
    """Simple multi-head self-attention module."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project the input into Q, K, and V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split d_model into multiple heads
        q = q.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Apply attention independently to every head
        out = attention(q, k, v, mask=mask)

        # Combine all heads again
        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # Final output projection
        return self.out_proj(out)
