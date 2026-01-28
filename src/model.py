"""CrystallineTransformer model."""

import torch
import torch.nn as nn

from .bottleneck import CrystallineBottleneck
from .config import ModelConfig


class CrystallineTransformer(nn.Module):
    """
    Transformer with Crystalline bottlenecks after attention and MLP.

    Architecture:
        Input Embeddings
            ↓
        [Transformer Block] × N
            - Attention → Crystalline Bottleneck → Residual
            - MLP → Crystalline Bottleneck → Residual
            ↓
        Output Logits
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.dim)

        # Transformer blocks with bottlenecks
        self.blocks = nn.ModuleList([
            TransformerBlockWithBottleneck(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Output head
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying
        self.output.weight = self.token_embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """
        Forward pass.

        Args:
            input_ids: Token ids of shape (batch, seq_len)
            attention_mask: Optional attention mask

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            bottleneck_infos: List of bottleneck info dicts per layer
        """
        batch, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # Transformer blocks
        bottleneck_infos = []
        for block in self.blocks:
            x, info = block(x, attention_mask)
            bottleneck_infos.append(info)

        # Output
        x = self.norm(x)
        logits = self.output(x)

        return logits, bottleneck_infos


class TransformerBlockWithBottleneck(nn.Module):
    """Single transformer block with crystalline bottlenecks."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Attention
        self.attn_norm = nn.LayerNorm(config.dim)
        self.attn = nn.MultiheadAttention(
            config.dim,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attn_bottleneck = CrystallineBottleneck(
            dim=config.dim,
            codebook_size=config.bottleneck.codebook_size,
            num_codes_k=config.bottleneck.num_codes_k,
            temp_init=config.bottleneck.temp_init,
            temp_min=config.bottleneck.temp_min,
        )

        # MLP
        self.mlp_norm = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, config.dim * 4),
            nn.GELU(),
            nn.Linear(config.dim * 4, config.dim),
            nn.Dropout(config.dropout),
        )
        self.mlp_bottleneck = CrystallineBottleneck(
            dim=config.dim,
            codebook_size=config.bottleneck.codebook_size,
            num_codes_k=config.bottleneck.num_codes_k,
            temp_init=config.bottleneck.temp_init,
            temp_min=config.bottleneck.temp_min,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass through block."""
        # Attention with bottleneck
        attn_out, _ = self.attn(
            self.attn_norm(x),
            self.attn_norm(x),
            self.attn_norm(x),
            key_padding_mask=attention_mask,
        )
        attn_out, attn_info = self.attn_bottleneck(attn_out)
        x = x + attn_out

        # MLP with bottleneck
        mlp_out = self.mlp(self.mlp_norm(x))
        mlp_out, mlp_info = self.mlp_bottleneck(mlp_out)
        x = x + mlp_out

        return x, {
            'layer': self.layer_idx,
            'attn': attn_info,
            'mlp': mlp_info,
        }
