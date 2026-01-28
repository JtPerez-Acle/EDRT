#!/usr/bin/env python3
"""
Toy experiment: Copy task

A minimal experiment to verify the Crystalline system works end-to-end.
The model learns to copy the input sequence (shifted by 1 position).

This validates:
1. Training loop works
2. Loss decreases
3. Temperature behavior over training
4. Codebook usage doesn't collapse
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.model import CrystallineTransformer
from src.config import ModelConfig, BottleneckConfig
from src.utils import compute_codebook_usage


def generate_copy_batch(batch_size: int, seq_len: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch for the copy task.

    Input: [BOS, x1, x2, ..., xn]
    Target: [x1, x2, ..., xn, EOS]

    Model learns to copy the sequence.
    """
    # Reserve 0 for BOS, 1 for EOS
    tokens = torch.randint(2, vocab_size, (batch_size, seq_len - 1))

    # Input: BOS + tokens
    bos = torch.zeros(batch_size, 1, dtype=torch.long)
    inputs = torch.cat([bos, tokens], dim=1)

    # Target: tokens + EOS
    eos = torch.ones(batch_size, 1, dtype=torch.long)
    targets = torch.cat([tokens, eos], dim=1)

    return inputs, targets


def generate_identity_batch(batch_size: int, seq_len: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch for the identity task (simplest possible).

    Input: [x1, x2, ..., xn]
    Target: [x1, x2, ..., xn]

    Each position just predicts itself. Tests basic learning.
    """
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    return tokens, tokens.clone()


def get_temperature_stats(model: CrystallineTransformer) -> dict:
    """Extract temperature statistics from all bottlenecks."""
    temps = []
    for block in model.blocks:
        temps.append(block.attn_bottleneck.temperature.item())
        temps.append(block.mlp_bottleneck.temperature.item())
    return {
        'mean': sum(temps) / len(temps),
        'min': min(temps),
        'max': max(temps),
        'all': temps,
    }


def get_codebook_stats(infos: list[dict]) -> dict:
    """Extract codebook usage statistics from forward pass info."""
    all_hard_codes = []
    for info in infos:
        all_hard_codes.append(info['attn']['hard_codes'])
        all_hard_codes.append(info['mlp']['hard_codes'])

    combined = torch.cat([h.view(-1, h.size(-1)) for h in all_hard_codes], dim=0)
    usage = compute_codebook_usage(combined)
    return usage


def train_copy_task(
    vocab_size: int = 32,
    seq_len: int = 16,
    batch_size: int = 32,
    dim: int = 64,
    n_layers: int = 2,
    n_heads: int = 2,
    codebook_size: int = 128,
    num_codes_k: int = 8,
    n_steps: int = 500,
    lr: float = 1e-3,
    log_every: int = 50,
    device: str = 'cpu',
    task: str = 'identity',  # 'identity' or 'copy'
):
    """
    Train on the copy task and monitor discretization.
    """
    print("=" * 60)
    print(f"TOY EXPERIMENT: {task.upper()} Task")
    print("=" * 60)
    print(f"Config: vocab={vocab_size}, seq_len={seq_len}, dim={dim}")
    print(f"        layers={n_layers}, codebook={codebook_size}, k={num_codes_k}")
    print(f"        steps={n_steps}, lr={lr}")
    print("=" * 60)

    # Select batch generator
    if task == 'identity':
        generate_batch = generate_identity_batch
    else:
        generate_batch = generate_copy_batch

    # Create model
    config = ModelConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=seq_len,
        dropout=0.0,
        bottleneck=BottleneckConfig(
            codebook_size=codebook_size,
            num_codes_k=num_codes_k,
            temp_init=2.0,  # Start warm
            temp_min=0.1,
        ),
    )
    model = CrystallineTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for step in range(n_steps):
        # Generate batch
        inputs, targets = generate_batch(batch_size, seq_len, vocab_size)
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward
        logits, infos = model(inputs)

        # Loss (cross-entropy)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        if step % log_every == 0 or step == n_steps - 1:
            temp_stats = get_temperature_stats(model)
            codebook_stats = get_codebook_stats(infos)

            # Compute accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == targets).float().mean().item()

            print(f"Step {step:4d} | Loss: {loss.item():.4f} | Acc: {acc:.3f} | "
                  f"Temp: {temp_stats['mean']:.3f} [{temp_stats['min']:.3f}-{temp_stats['max']:.3f}] | "
                  f"Codebook: {codebook_stats['usage_fraction']*100:.1f}% used")

    print("=" * 60)
    print("FINAL TEMPERATURE PER BOTTLENECK:")
    temp_stats = get_temperature_stats(model)
    for i, (attn_t, mlp_t) in enumerate(zip(temp_stats['all'][::2], temp_stats['all'][1::2])):
        print(f"  Layer {i}: attn={attn_t:.4f}, mlp={mlp_t:.4f}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Toy copy task experiment")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--codebook", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--task", type=str, default="identity", choices=["identity", "copy"])
    args = parser.parse_args()

    train_copy_task(
        n_steps=args.steps,
        dim=args.dim,
        n_layers=args.layers,
        codebook_size=args.codebook,
        device=args.device,
        task=args.task,
    )
