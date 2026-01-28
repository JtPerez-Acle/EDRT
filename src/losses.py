"""Loss functions for Crystalline training."""

import torch
import torch.nn.functional as F


def compression_loss(soft_codes: torch.Tensor) -> torch.Tensor:
    """
    Entropy regularization loss - encourages discrete (low-entropy) selections.

    Args:
        soft_codes: Soft code selections of shape (batch, seq_len, codebook_size)

    Returns:
        Mean entropy across all positions
    """
    # Compute entropy: -sum(p * log(p))
    eps = 1e-8
    entropy = -torch.sum(soft_codes * torch.log(soft_codes + eps), dim=-1)
    return entropy.mean()


def commitment_loss(
    x: torch.Tensor,
    quantized: torch.Tensor,
) -> torch.Tensor:
    """
    VQ-VAE style commitment loss - encourages inputs to commit to codes.

    Args:
        x: Original input tensor
        quantized: Quantized output tensor

    Returns:
        MSE between input and quantized (with stopped gradient on quantized)
    """
    return F.mse_loss(x, quantized.detach())


def total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bottleneck_info: dict,
    lambda_compress: float = 0.01,
    lambda_commit: float = 0.25,
) -> tuple[torch.Tensor, dict]:
    """
    Compute total training loss.

    L_total = L_prediction + λ_compress * L_compression + λ_commit * L_commitment

    Args:
        logits: Model output logits
        targets: Target token ids
        bottleneck_info: Dictionary from bottleneck forward pass
        lambda_compress: Weight for compression loss
        lambda_commit: Weight for commitment loss

    Returns:
        total_loss: Combined loss tensor
        loss_dict: Dictionary with individual loss components
    """
    # Prediction loss (cross-entropy)
    pred_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
    )

    # Compression loss (entropy regularization)
    compress_loss = compression_loss(bottleneck_info['soft_codes'])

    # Commitment loss
    commit_loss = commitment_loss(
        bottleneck_info['input'],
        bottleneck_info['output'],
    )

    # Total loss
    total = pred_loss + lambda_compress * compress_loss + lambda_commit * commit_loss

    return total, {
        'total': total.item(),
        'prediction': pred_loss.item(),
        'compression': compress_loss.item(),
        'commitment': commit_loss.item(),
    }
