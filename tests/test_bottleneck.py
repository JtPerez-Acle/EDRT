"""Tests for CrystallineBottleneck module."""

import pytest
import torch

from src.bottleneck import CrystallineBottleneck
from src.utils import gumbel_softmax, top_k_gumbel


class TestGumbelSoftmax:
    """Tests for Gumbel-Softmax utilities."""

    def test_gumbel_softmax_shape(self):
        """Output should have same shape as input."""
        logits = torch.randn(2, 4, 16)
        output = gumbel_softmax(logits, tau=1.0)
        assert output.shape == logits.shape

    def test_gumbel_softmax_sums_to_one(self):
        """Soft outputs should sum to 1 along last dimension."""
        logits = torch.randn(2, 4, 16)
        output = gumbel_softmax(logits, tau=1.0)
        sums = output.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gumbel_softmax_hard(self):
        """Hard mode should produce one-hot vectors."""
        logits = torch.randn(2, 4, 16)
        output = gumbel_softmax(logits, tau=1.0, hard=True)
        # Each row should have exactly one 1
        assert torch.allclose(output.sum(dim=-1), torch.ones(2, 4))
        # Values should be 0 or 1
        assert torch.all((output == 0) | (output == 1))

    def test_gumbel_softmax_temperature(self):
        """Lower temperature should produce sharper distributions."""
        logits = torch.randn(100, 16)

        soft_high = gumbel_softmax(logits, tau=5.0)
        soft_low = gumbel_softmax(logits, tau=0.1)

        # Entropy should be lower for low temperature
        entropy_high = -(soft_high * torch.log(soft_high + 1e-8)).sum(dim=-1).mean()
        entropy_low = -(soft_low * torch.log(soft_low + 1e-8)).sum(dim=-1).mean()

        assert entropy_low < entropy_high


class TestTopKGumbel:
    """Tests for top-k Gumbel selection."""

    def test_top_k_shape(self):
        """Output shapes should match input."""
        logits = torch.randn(2, 4, 16)
        soft, hard = top_k_gumbel(logits, k=4, tau=1.0)
        assert soft.shape == logits.shape
        assert hard.shape == logits.shape

    def test_top_k_selection_count(self):
        """Hard selection should have exactly k non-zero entries."""
        logits = torch.randn(2, 4, 16)
        _, hard = top_k_gumbel(logits, k=4, tau=1.0)
        # Count non-zero entries per position
        counts = (hard > 0.5).sum(dim=-1)
        assert torch.all(counts == 4)


class TestCrystallineBottleneck:
    """Tests for the CrystallineBottleneck module."""

    def test_initialization(self):
        """Bottleneck should initialize with correct shapes."""
        bottleneck = CrystallineBottleneck(
            dim=64,
            codebook_size=128,
            num_codes_k=8,
        )
        assert bottleneck.codebook.shape == (128, 64)
        assert bottleneck.temperature.shape == ()

    def test_temperature_clamping(self):
        """Temperature should be clamped to minimum."""
        bottleneck = CrystallineBottleneck(
            dim=64,
            codebook_size=128,
            temp_init=0.01,  # Below minimum
            temp_min=0.1,
        )
        assert bottleneck.temperature >= 0.1

    def test_forward_shape(self):
        """Forward pass should preserve input shape."""
        bottleneck = CrystallineBottleneck(dim=64, codebook_size=128)
        x = torch.randn(2, 4, 64)
        output, info = bottleneck(x)
        assert output.shape == x.shape

    def test_gradient_flow(self):
        """Gradients should flow through the bottleneck."""
        bottleneck = CrystallineBottleneck(dim=64, codebook_size=128)
        x = torch.randn(2, 4, 64, requires_grad=True)
        output, _ = bottleneck(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert bottleneck.codebook.grad is not None
        assert bottleneck._temperature.grad is not None

    def test_forward_info_dict(self):
        """Forward should return required info for loss computation."""
        bottleneck = CrystallineBottleneck(dim=64, codebook_size=128, num_codes_k=8)
        x = torch.randn(2, 4, 64)
        output, info = bottleneck(x)

        # Check all required keys exist
        assert 'temperature' in info
        assert 'soft_codes' in info
        assert 'hard_codes' in info
        assert 'input' in info
        assert 'output' in info
        assert 'entropy' in info

        # Check shapes
        assert info['soft_codes'].shape == (2, 4, 128)
        assert info['hard_codes'].shape == (2, 4, 128)
        assert info['input'].shape == x.shape
        assert info['output'].shape == x.shape

    def test_hard_codes_sparsity(self):
        """Hard codes should have exactly k non-zero entries per position."""
        bottleneck = CrystallineBottleneck(dim=64, codebook_size=128, num_codes_k=8)
        x = torch.randn(2, 4, 64)
        _, info = bottleneck(x)

        # Each position should select exactly k codes
        active_per_position = (info['hard_codes'] > 0.5).sum(dim=-1)
        assert torch.all(active_per_position == 8)
