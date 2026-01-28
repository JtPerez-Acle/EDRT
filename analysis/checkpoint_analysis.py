"""
Checkpoint analysis utilities for Crystalline models.

Provides functions to load checkpoints, extract bottleneck statistics,
and run inference to collect code activation data.
"""

from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from src.model import CrystallineTransformer
from src.config import ModelConfig, BottleneckConfig


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    model: CrystallineTransformer
    config: dict
    step: int
    epoch: int
    metrics: dict
    bottleneck_stats: dict


def load_checkpoint_for_analysis(
    path: Union[str, Path],
    device: str = "cpu",
) -> AnalysisResult:
    """
    Load a checkpoint and reconstruct the model for analysis.

    Args:
        path: Path to checkpoint file (.pt)
        device: Device to load model onto

    Returns:
        AnalysisResult containing model, config, and checkpoint info
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Extract config
    config = checkpoint.get("config", {})

    # Reconstruct model config from checkpoint
    model_config_dict = config.get("model", {}).copy()

    # Handle both nested (model.bottleneck) and top-level (bottleneck) config formats
    bottleneck_config_dict = model_config_dict.pop("bottleneck", None)
    if bottleneck_config_dict is None:
        bottleneck_config_dict = config.get("bottleneck", {})

    bottleneck_config = BottleneckConfig(**bottleneck_config_dict)
    model_config = ModelConfig(
        **model_config_dict,
        bottleneck=bottleneck_config,
    )

    # Create and load model
    model = CrystallineTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Extract bottleneck statistics
    bottleneck_stats = extract_bottleneck_stats(model)

    return AnalysisResult(
        model=model,
        config=config,
        step=checkpoint.get("step", 0),
        epoch=checkpoint.get("epoch", 0),
        metrics=checkpoint.get("metrics", {}),
        bottleneck_stats=bottleneck_stats,
    )


def extract_bottleneck_stats(model: CrystallineTransformer) -> dict:
    """
    Extract statistics from all bottleneck modules in the model.

    Args:
        model: CrystallineTransformer model

    Returns:
        Dictionary with temperatures, codebook info per layer/type
    """
    stats = {
        "n_layers": len(model.blocks),
        "layers": [],
        "temperatures": {
            "attn": [],
            "mlp": [],
            "all": [],
        },
        "codebook_sizes": [],
        "num_codes_k": [],
    }

    for i, block in enumerate(model.blocks):
        layer_stats = {
            "layer": i,
            "attn": _get_bottleneck_info(block.attn_bottleneck),
            "mlp": _get_bottleneck_info(block.mlp_bottleneck),
        }
        stats["layers"].append(layer_stats)

        # Aggregate temperatures
        attn_temp = layer_stats["attn"]["temperature"]
        mlp_temp = layer_stats["mlp"]["temperature"]
        stats["temperatures"]["attn"].append(attn_temp)
        stats["temperatures"]["mlp"].append(mlp_temp)
        stats["temperatures"]["all"].extend([attn_temp, mlp_temp])

        # Codebook info (same for all bottlenecks in standard config)
        stats["codebook_sizes"].append(layer_stats["attn"]["codebook_size"])
        stats["num_codes_k"].append(layer_stats["attn"]["num_codes_k"])

    # Summary statistics
    all_temps = stats["temperatures"]["all"]
    stats["temperature_summary"] = {
        "mean": np.mean(all_temps),
        "min": np.min(all_temps),
        "max": np.max(all_temps),
        "std": np.std(all_temps),
    }

    return stats


def _get_bottleneck_info(bottleneck) -> dict:
    """Extract info from a single bottleneck module."""
    return {
        "temperature": bottleneck.temperature.item(),
        "temp_min": bottleneck.temp_min,
        "codebook_size": bottleneck.codebook_size,
        "num_codes_k": bottleneck.num_codes_k,
        "output_scale": bottleneck.output_scale.item(),
        "codebook_norm": bottleneck.codebook.norm(dim=-1).mean().item(),
    }


def run_inference_with_codes(
    model: CrystallineTransformer,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: Optional[int] = 10,
) -> dict:
    """
    Run inference and collect code activation statistics.

    Args:
        model: Model to analyze
        dataloader: DataLoader with input data
        device: Device to run on
        max_batches: Maximum batches to process (None for all)

    Returns:
        Dictionary with code activations, entropies, and usage statistics
    """
    model.eval()

    results = {
        "n_samples": 0,
        "layers": [],
        "entropies": {"attn": [], "mlp": []},
        "code_counts": None,  # Will be initialized on first batch
    }

    codebook_size = model.blocks[0].attn_bottleneck.codebook_size
    n_layers = len(model.blocks)

    # Initialize code counts per layer and type
    code_counts = {
        "attn": torch.zeros(n_layers, codebook_size),
        "mlp": torch.zeros(n_layers, codebook_size),
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
            else:
                input_ids = batch[0].to(device)

            batch_size = input_ids.shape[0]
            results["n_samples"] += batch_size

            # Forward pass
            _, infos = model(input_ids)

            # Collect statistics from each layer
            for layer_info in infos:
                layer_idx = layer_info["layer"]

                for bn_type in ["attn", "mlp"]:
                    info = layer_info[bn_type]

                    # Entropy
                    entropy = info["entropy"].item()
                    results["entropies"][bn_type].append(entropy)

                    # Code usage counts
                    hard_codes = info["hard_codes"]  # (batch, seq_len, codebook_size)
                    active = (hard_codes > 0.5).float()
                    counts = active.sum(dim=[0, 1])  # Sum over batch and seq
                    code_counts[bn_type][layer_idx] += counts.cpu()

    # Compute usage statistics
    total_activations = results["n_samples"] * input_ids.shape[1]  # samples * seq_len
    results["code_usage"] = {
        "attn": (code_counts["attn"] > 0).float().mean(dim=1).tolist(),  # Per layer
        "mlp": (code_counts["mlp"] > 0).float().mean(dim=1).tolist(),
    }
    results["code_counts"] = {
        "attn": code_counts["attn"].tolist(),
        "mlp": code_counts["mlp"].tolist(),
    }

    # Compute mean entropies
    results["entropy_summary"] = {
        "attn_mean": np.mean(results["entropies"]["attn"]),
        "mlp_mean": np.mean(results["entropies"]["mlp"]),
        "overall_mean": np.mean(
            results["entropies"]["attn"] + results["entropies"]["mlp"]
        ),
    }

    return results


def compare_checkpoints(paths: list[Union[str, Path]], device: str = "cpu") -> dict:
    """
    Compare statistics across multiple checkpoints (e.g., training progression).

    Args:
        paths: List of checkpoint paths
        device: Device to use

    Returns:
        Dictionary with comparison data
    """
    comparison = {
        "steps": [],
        "temperatures": [],
        "entropies": [],
        "configs": [],
    }

    for path in paths:
        result = load_checkpoint_for_analysis(path, device)

        comparison["steps"].append(result.step)
        comparison["temperatures"].append(
            result.bottleneck_stats["temperature_summary"]["mean"]
        )
        comparison["configs"].append(result.config)

        # Note: entropy requires running inference, so we just use stored metrics if available
        if "entropy" in result.metrics:
            comparison["entropies"].append(result.metrics["entropy"])
        else:
            comparison["entropies"].append(None)

    return comparison


def get_codebook_embeddings(model: CrystallineTransformer, layer: int = 0, bn_type: str = "attn") -> np.ndarray:
    """
    Extract codebook embeddings for visualization.

    Args:
        model: Model to extract from
        layer: Layer index
        bn_type: 'attn' or 'mlp'

    Returns:
        Codebook vectors as numpy array (codebook_size, dim)
    """
    block = model.blocks[layer]
    bottleneck = block.attn_bottleneck if bn_type == "attn" else block.mlp_bottleneck
    return bottleneck.codebook.detach().cpu().numpy()


if __name__ == "__main__":
    # Test with the existing checkpoint
    import sys

    checkpoint_path = Path("checkpoints/tinystories/checkpoint_final.pt")

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run a training experiment first to generate a checkpoint.")
        sys.exit(1)

    print("Loading checkpoint for analysis...")
    result = load_checkpoint_for_analysis(checkpoint_path)

    print(f"\nCheckpoint Info:")
    print(f"  Step: {result.step}")
    print(f"  Epoch: {result.epoch}")

    print(f"\nBottleneck Statistics:")
    stats = result.bottleneck_stats
    print(f"  Layers: {stats['n_layers']}")
    print(f"  Temperature (mean): {stats['temperature_summary']['mean']:.4f}")
    print(f"  Temperature (range): [{stats['temperature_summary']['min']:.4f}, {stats['temperature_summary']['max']:.4f}]")

    print(f"\nPer-layer temperatures:")
    for layer_stats in stats["layers"]:
        i = layer_stats["layer"]
        attn_t = layer_stats["attn"]["temperature"]
        mlp_t = layer_stats["mlp"]["temperature"]
        print(f"  Layer {i}: attn={attn_t:.4f}, mlp={mlp_t:.4f}")

    print("\nCheckpoint analysis complete!")
