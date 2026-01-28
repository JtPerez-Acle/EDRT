"""Tests for CrystallineTransformer model."""

import pytest
import torch

from src.model import CrystallineTransformer, TransformerBlockWithBottleneck
from src.config import ModelConfig, BottleneckConfig


class TestCrystallineTransformer:
    """Tests for the full transformer model."""

    @pytest.fixture
    def tiny_config(self):
        """Create a tiny config for fast testing."""
        return ModelConfig(
            vocab_size=100,
            dim=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=16,
            dropout=0.0,
            bottleneck=BottleneckConfig(
                codebook_size=64,
                num_codes_k=4,
                temp_init=1.0,
                temp_min=0.1,
            ),
        )

    def test_model_initialization(self, tiny_config):
        """Model should initialize with correct structure."""
        model = CrystallineTransformer(tiny_config)

        assert len(model.blocks) == 2
        assert model.token_embed.num_embeddings == 100
        assert model.token_embed.embedding_dim == 32

    def test_forward_shape(self, tiny_config):
        """Forward pass should produce correct output shape."""
        model = CrystallineTransformer(tiny_config)
        x = torch.randint(0, 100, (2, 8))

        logits, infos = model(x)

        assert logits.shape == (2, 8, 100)
        assert len(infos) == 2  # One per layer

    def test_bottleneck_info_structure(self, tiny_config):
        """Each layer should return attn and mlp bottleneck info."""
        model = CrystallineTransformer(tiny_config)
        x = torch.randint(0, 100, (2, 8))

        _, infos = model(x)

        for i, info in enumerate(infos):
            assert 'layer' in info
            assert info['layer'] == i
            assert 'attn' in info
            assert 'mlp' in info

            # Check bottleneck info contents
            for key in ['attn', 'mlp']:
                assert 'temperature' in info[key]
                assert 'soft_codes' in info[key]
                assert 'hard_codes' in info[key]

    def test_gradient_flow_to_temperatures(self, tiny_config):
        """Gradients should flow to all temperature parameters."""
        model = CrystallineTransformer(tiny_config)
        x = torch.randint(0, 100, (2, 8))

        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()

        for i, block in enumerate(model.blocks):
            assert block.attn_bottleneck._temperature.grad is not None, \
                f"Layer {i} attn temperature has no gradient"
            assert block.mlp_bottleneck._temperature.grad is not None, \
                f"Layer {i} mlp temperature has no gradient"

    def test_gradient_flow_to_codebooks(self, tiny_config):
        """Gradients should flow to all codebook parameters."""
        model = CrystallineTransformer(tiny_config)
        x = torch.randint(0, 100, (2, 8))

        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()

        for i, block in enumerate(model.blocks):
            assert block.attn_bottleneck.codebook.grad is not None, \
                f"Layer {i} attn codebook has no gradient"
            assert block.mlp_bottleneck.codebook.grad is not None, \
                f"Layer {i} mlp codebook has no gradient"

    def test_weight_tying(self, tiny_config):
        """Output projection should share weights with token embeddings."""
        model = CrystallineTransformer(tiny_config)

        assert model.output.weight is model.token_embed.weight

    def test_different_sequence_lengths(self, tiny_config):
        """Model should handle various sequence lengths up to max."""
        model = CrystallineTransformer(tiny_config)

        for seq_len in [1, 4, 8, 16]:
            x = torch.randint(0, 100, (1, seq_len))
            logits, _ = model(x)
            assert logits.shape == (1, seq_len, 100)


class TestTransformerBlock:
    """Tests for individual transformer blocks."""

    @pytest.fixture
    def tiny_config(self):
        return ModelConfig(
            vocab_size=100,
            dim=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=16,
            dropout=0.0,
            bottleneck=BottleneckConfig(codebook_size=64, num_codes_k=4),
        )

    def test_block_preserves_shape(self, tiny_config):
        """Block should preserve input shape."""
        block = TransformerBlockWithBottleneck(tiny_config, layer_idx=0)
        x = torch.randn(2, 8, 32)

        output, _ = block(x)

        assert output.shape == x.shape

    def test_block_residual_connection(self, tiny_config):
        """Output should be different from input (not identity)."""
        block = TransformerBlockWithBottleneck(tiny_config, layer_idx=0)
        x = torch.randn(2, 8, 32)

        output, _ = block(x)

        # Should not be identical (residual adds transformed input)
        assert not torch.allclose(output, x)
