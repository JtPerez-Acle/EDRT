"""CrystallineBottleneck module - learnable discrete bottleneck with adaptive temperature."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import top_k_gumbel


class CrystallineBottleneck(nn.Module):
    """
    Learnable discrete bottleneck with adaptive temperature.

    Key innovations:
    1. Temperature is LEARNED per-layer (or per-head, per-position)
    2. Gumbel-Softmax enables gradient flow through discretization
    3. Codebook is learned during training
    4. Compression pressure via entropy regularization
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 512,
        num_codes_k: int = 8,
        temp_init: float = 1.0,
        temp_min: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codes_k = num_codes_k
        self.temp_min = temp_min

        # Learnable codebook - initialize with unit-norm vectors
        codebook = torch.randn(codebook_size, dim)
        codebook = F.normalize(codebook, dim=-1)
        self.codebook = nn.Parameter(codebook)

        # Learned scale factor for reconstruction (starts at 1.0)
        self.output_scale = nn.Parameter(torch.ones(1))

        # Learned temperature (will be clamped to temp_min)
        self._temperature = nn.Parameter(torch.tensor(temp_init))

    @property
    def temperature(self) -> torch.Tensor:
        """Return temperature clamped to minimum value."""
        return torch.clamp(self._temperature, min=self.temp_min)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Forward pass through the crystalline bottleneck.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            output: Reconstructed tensor of shape (batch, seq_len, dim)
            info: Dictionary with temperature, active_codes, entropy
        """
        batch, seq_len, dim = x.shape

        # 1. Compute similarities to codebook: (batch, seq, codebook_size)
        # Normalize for stable dot products (cosine similarity)
        x_norm = F.normalize(x, dim=-1)
        codebook_norm = F.normalize(self.codebook, dim=-1)
        logits = x_norm @ codebook_norm.T

        # 2. Gumbel-Softmax with learned temperature + top-k selection
        soft_codes, hard_codes = top_k_gumbel(
            logits,
            k=self.num_codes_k,
            tau=self.temperature
        )

        # 3. Reconstruct from codebook with learned scale
        # Scale output to match input magnitude
        output = hard_codes @ self.codebook
        output = output * self.output_scale

        # 4. Compute entropy for monitoring
        eps = 1e-8
        entropy = -(soft_codes * torch.log(soft_codes + eps)).sum(dim=-1).mean()

        return output, {
            'temperature': self.temperature,
            'soft_codes': soft_codes,
            'hard_codes': hard_codes,
            'input': x,
            'output': output,
            'entropy': entropy,
            'output_scale': self.output_scale,
        }
