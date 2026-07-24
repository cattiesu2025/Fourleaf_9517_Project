"""
src/transfer/attention.py
----------------------------
A multi-head self-attention block that operates on a CNN's spatial
feature map, following the "Bottleneck Transformer" pattern: treat each
spatial location as a token, run standard multi-head self-attention
across all tokens, reshape back to a feature map.

Inserted after ResNet18's layer4 (7x7 feature map = 49 tokens), before
global average pooling -- see model.py's build_model(..., use_attention=True).
"""

import torch
import torch.nn as nn


class MultiHeadSelfAttention2D(nn.Module):
    """Self-attention over the spatial positions of a (B, C, H, W) feature map.

    Uses a residual connection + LayerNorm (standard Transformer block
    pattern), so if the attention weights start near-uniform/useless early
    in training, the block can still pass the original features through
    largely unchanged rather than corrupting them.
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")

        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, C, H, W)"""
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        normed = self.norm(tokens)
        attn_out, attn_weights = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out  # residual connection

        return tokens.transpose(1, 2).reshape(b, c, h, w)
