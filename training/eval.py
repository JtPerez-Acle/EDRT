"""Evaluation metrics for Crystalline models."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import CrystallineTransformer


def compute_perplexity(
    model: CrystallineTransformer,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> float:
    """
    Compute perplexity on a dataset.

    Perplexity = exp(average cross-entropy loss per token)
    Lower is better. Random baseline for vocab_size V is ~V.

    Args:
        model: The model to evaluate
        dataloader: DataLoader with input_ids and labels
        device: Device to run evaluation on
        max_batches: Limit evaluation to this many batches (for speed)

    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits, _ = model(input_ids)

            # Cross-entropy per token (sum reduction)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction='sum',
                ignore_index=-100,  # Ignore padding
            )

            # Count non-padding tokens
            valid_tokens = (labels != -100).sum().item()

            total_loss += loss.item()
            total_tokens += valid_tokens

    model.train()

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


def compute_crystallization_metrics(
    model: CrystallineTransformer,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: int = 10,
) -> dict:
    """
    Compute metrics related to the crystalline bottleneck.

    Returns:
        Dictionary with:
        - temperature_mean/min/max: Temperature statistics across layers
        - entropy_mean: Average entropy of code selections
        - codebook_usage: Fraction of codebook being used
        - temps_per_layer: List of temperatures per layer
    """
    model.eval()

    # Get temperature stats
    temps = []
    for block in model.blocks:
        temps.append(block.attn_bottleneck.temperature.item())
        temps.append(block.mlp_bottleneck.temperature.item())

    # Collect code activations to measure usage
    all_code_counts = None
    codebook_size = model.blocks[0].attn_bottleneck.codebook_size
    total_activations = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            _, infos = model(input_ids)

            # Aggregate code usage across all bottlenecks
            for layer_info in infos:
                for bn_type in ['attn', 'mlp']:
                    hard_codes = layer_info[bn_type]['hard_codes']
                    # hard_codes shape: (batch, seq_len, codebook_size)

                    # Count which codes are active
                    active = (hard_codes > 0.5).float()
                    code_counts = active.sum(dim=[0, 1])  # (codebook_size,)

                    if all_code_counts is None:
                        all_code_counts = code_counts
                    else:
                        all_code_counts = all_code_counts + code_counts

                    total_activations += active.sum().item()

    model.train()

    # Compute metrics
    if all_code_counts is not None:
        codes_used = (all_code_counts > 0).sum().item()
        codebook_usage = codes_used / codebook_size
    else:
        codebook_usage = 0.0

    # Compute average entropy (use last batch)
    avg_entropy = 0.0
    if infos:
        entropies = []
        for layer_info in infos:
            for bn_type in ['attn', 'mlp']:
                entropies.append(layer_info[bn_type]['entropy'].item())
        avg_entropy = sum(entropies) / len(entropies)

    return {
        'temperature_mean': sum(temps) / len(temps),
        'temperature_min': min(temps),
        'temperature_max': max(temps),
        'temps_per_layer': temps,
        'entropy_mean': avg_entropy,
        'codebook_usage': codebook_usage,
    }


def evaluate_model(
    model: CrystallineTransformer,
    val_loader: DataLoader,
    device: str = "cpu",
    max_batches: Optional[int] = 50,
) -> dict:
    """
    Run full evaluation suite.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to run on
        max_batches: Limit batches for speed

    Returns:
        Dictionary with all metrics
    """
    perplexity = compute_perplexity(model, val_loader, device, max_batches)
    crystal_metrics = compute_crystallization_metrics(model, val_loader, device, min(10, max_batches or 10))

    return {
        'perplexity': perplexity,
        **crystal_metrics,
    }


if __name__ == "__main__":
    # Quick test
    from src.config import ModelConfig, BottleneckConfig
    from src.model import CrystallineTransformer

    print("Testing evaluation functions...")

    # Create tiny model
    model_config = ModelConfig(
        vocab_size=1000,
        dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=64,
        bottleneck=BottleneckConfig(
            codebook_size=32,
            num_codes_k=4,
        ),
    )

    model = CrystallineTransformer(model_config)

    # Create dummy data
    dummy_data = [
        {'input_ids': torch.randint(0, 1000, (4, 64)),
         'labels': torch.randint(0, 1000, (4, 64))}
        for _ in range(5)
    ]

    class DummyLoader:
        def __iter__(self):
            return iter(dummy_data)

    # Run evaluation
    metrics = evaluate_model(model, DummyLoader(), device='cpu', max_batches=3)

    print(f"\nMetrics:")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
    print(f"  Temperature: {metrics['temperature_mean']:.3f}")
    print(f"  Entropy: {metrics['entropy_mean']:.3f}")
    print(f"  Codebook usage: {metrics['codebook_usage']:.1%}")
    print("\nEvaluation test passed!")
