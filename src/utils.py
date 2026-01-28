"""Utility functions for Crystalline."""

import torch
import torch.nn.functional as F


def gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1,
) -> torch.Tensor:
    """
    Gumbel-Softmax sampling.

    Args:
        logits: Unnormalized log probabilities
        tau: Temperature parameter
        hard: If True, use straight-through estimator
        dim: Dimension to apply softmax

    Returns:
        Sampled tensor (same shape as logits)
    """
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim=dim)

    if hard:
        # Straight-through estimator
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft

    return y_soft


def top_k_gumbel(
    logits: torch.Tensor,
    k: int,
    tau: float = 1.0,
    dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k Gumbel-Softmax selection.

    Args:
        logits: Unnormalized log probabilities
        k: Number of codes to select
        tau: Temperature parameter
        dim: Dimension to apply softmax

    Returns:
        soft_codes: Soft selection probabilities
        hard_codes: Hard top-k selection (straight-through)
    """
    # Soft selection
    soft_codes = gumbel_softmax(logits, tau=tau, hard=False, dim=dim)

    # Hard top-k selection
    _, indices = torch.topk(soft_codes, k, dim=dim)
    hard_codes = torch.zeros_like(soft_codes).scatter_(dim, indices, 1.0)

    # Straight-through estimator
    hard_codes = hard_codes - soft_codes.detach() + soft_codes

    return soft_codes, hard_codes


def compute_codebook_usage(hard_codes: torch.Tensor) -> dict:
    """
    Compute codebook usage statistics.

    Args:
        hard_codes: Hard code selections of shape (..., codebook_size)

    Returns:
        Dictionary with usage statistics
    """
    # Flatten to (num_selections, codebook_size)
    flat_codes = hard_codes.view(-1, hard_codes.size(-1))

    # Count usage per code
    usage_counts = flat_codes.sum(dim=0)
    total_selections = flat_codes.sum()

    # Compute statistics
    num_used = (usage_counts > 0).sum().item()
    codebook_size = hard_codes.size(-1)

    return {
        'num_used': num_used,
        'total_codes': codebook_size,
        'usage_fraction': num_used / codebook_size,
        'usage_entropy': -(usage_counts / total_selections *
                          torch.log(usage_counts / total_selections + 1e-8)).sum().item(),
    }
