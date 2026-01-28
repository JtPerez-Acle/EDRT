#!/usr/bin/env python3
"""
FSM Experiment: Train CrystallineTransformer on Finite State Machine data.

This experiment tests whether the bottleneck codes learn to represent FSM states.
If successful, we should see:
1. Codes that specialize to specific states (high purity)
2. Temperature decreasing as the model finds discrete structure
3. Good prediction accuracy on next-token prediction

Based on the Stanford "Codebook Features" paper setup.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from experiments.fsm.generate_data import (
    FSMConfig,
    FiniteStateMachine,
    compute_state_code_alignment,
)
from src.model import CrystallineTransformer
from src.config import ModelConfig, BottleneckConfig
from src.losses import compression_loss, commitment_loss


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


def evaluate_state_alignment(
    model: CrystallineTransformer,
    fsm: FiniteStateMachine,
    num_batches: int = 10,
    batch_size: int = 32,
    seq_len: int = 64,
    device: str = 'cpu',
) -> dict:
    """
    Evaluate how well bottleneck codes align with FSM states.

    Returns alignment metrics for each layer's bottlenecks.
    """
    model.eval()
    all_results = []

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets, states = fsm.generate_batch(batch_size, seq_len)
            inputs = inputs.to(device)
            states = states.to(device)

            _, infos = model(inputs)

            # Analyze each layer's bottleneck codes
            for layer_info in infos:
                layer_idx = layer_info['layer']

                for bottleneck_type in ['attn', 'mlp']:
                    hard_codes = layer_info[bottleneck_type]['hard_codes']

                    alignment = compute_state_code_alignment(
                        hard_codes.cpu(),
                        states.cpu(),
                        fsm.num_states,
                    )

                    all_results.append({
                        'layer': layer_idx,
                        'type': bottleneck_type,
                        **alignment,
                    })

    model.train()

    # Aggregate results
    aggregated = {}
    for r in all_results:
        key = f"L{r['layer']}_{r['type']}"
        if key not in aggregated:
            aggregated[key] = {'purity': [], 'active_codes': []}
        aggregated[key]['purity'].append(r['purity'])
        aggregated[key]['active_codes'].append(r['active_codes'])

    summary = {}
    for key, vals in aggregated.items():
        summary[f'{key}_purity'] = sum(vals['purity']) / len(vals['purity'])
        summary[f'{key}_active'] = sum(vals['active_codes']) / len(vals['active_codes'])

    return summary


def train_fsm(
    # FSM config
    num_states: int = 10,
    tokens_per_state: int = 3,

    # Model config
    dim: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    codebook_size: int = 64,  # Intentionally small to encourage state mapping
    num_codes_k: int = 4,

    # Training config
    batch_size: int = 64,
    seq_len: int = 64,
    n_steps: int = 5000,
    lr: float = 3e-4,
    weight_decay: float = 0.01,

    # Loss weights - KEY FOR CRYSTALLIZATION
    lambda_compress: float = 0.01,  # Compression pressure (pushes temp down)
    lambda_commit: float = 0.25,    # Commitment loss (VQ-VAE style)

    # Temperature annealing
    temp_anneal: bool = True,       # Whether to anneal temperature over training
    temp_start: float = 2.0,        # Starting temperature
    temp_end: float = 0.5,          # Ending temperature

    # Logging
    log_every: int = 100,
    eval_every: int = 500,

    # Other
    device: str = 'cpu',
    seed: int = 42,
):
    """Train on FSM data and analyze code-state alignment."""

    torch.manual_seed(seed)

    vocab_size = num_states * tokens_per_state

    print("=" * 70)
    print("FSM EXPERIMENT: Code-State Alignment")
    print("=" * 70)
    print(f"FSM: {num_states} states, {tokens_per_state} tokens/state, vocab={vocab_size}")
    print(f"Model: dim={dim}, layers={n_layers}, codebook={codebook_size}, k={num_codes_k}")
    print(f"Training: steps={n_steps}, batch={batch_size}, seq_len={seq_len}, lr={lr}")
    print(f"Loss weights: λ_compress={lambda_compress}, λ_commit={lambda_commit}")
    if temp_anneal:
        print(f"Temperature annealing: {temp_start} -> {temp_end}")
    print("=" * 70)

    # Create FSM
    fsm_config = FSMConfig(
        num_states=num_states,
        tokens_per_state=tokens_per_state,
        vocab_size=vocab_size,
        seed=seed,
    )
    fsm = FiniteStateMachine(fsm_config)

    # Create model
    model_config = ModelConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=seq_len,
        dropout=0.0,
        bottleneck=BottleneckConfig(
            codebook_size=codebook_size,
            num_codes_k=num_codes_k,
            temp_init=2.0,
            temp_min=0.1,
        ),
    )
    model = CrystallineTransformer(model_config).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr / 10)

    # Training loop
    model.train()
    for step in range(n_steps):
        # Temperature annealing: linearly interpolate from temp_start to temp_end
        if temp_anneal:
            progress = step / max(n_steps - 1, 1)
            target_temp = temp_start + progress * (temp_end - temp_start)
            # Set temperature for all bottlenecks
            with torch.no_grad():
                for block in model.blocks:
                    block.attn_bottleneck._temperature.fill_(target_temp)
                    block.mlp_bottleneck._temperature.fill_(target_temp)

        # Generate batch
        inputs, targets, states = fsm.generate_batch(batch_size, seq_len)
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward
        logits, infos = model(inputs)

        # Prediction loss
        pred_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        # Aggregate compression and commitment losses from all bottlenecks
        compress_loss = torch.tensor(0.0, device=device)
        commit_loss = torch.tensor(0.0, device=device)
        n_bottlenecks = 0

        for layer_info in infos:
            for bn_type in ['attn', 'mlp']:
                bn_info = layer_info[bn_type]
                compress_loss = compress_loss + compression_loss(bn_info['soft_codes'])
                commit_loss = commit_loss + commitment_loss(bn_info['input'], bn_info['output'])
                n_bottlenecks += 1

        compress_loss = compress_loss / n_bottlenecks
        commit_loss = commit_loss / n_bottlenecks

        # Total loss with crystallization pressure
        loss = pred_loss + lambda_compress * compress_loss + lambda_commit * commit_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Log
        if step % log_every == 0:
            temp_stats = get_temperature_stats(model)

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == targets).float().mean().item()

            print(f"Step {step:5d} | Loss: {loss.item():.4f} (pred={pred_loss.item():.3f}, "
                  f"comp={compress_loss.item():.3f}, commit={commit_loss.item():.3f}) | "
                  f"Acc: {acc:.3f} | Temp: {temp_stats['mean']:.3f} [{temp_stats['min']:.3f}-{temp_stats['max']:.3f}]")

        # Evaluate alignment
        if step % eval_every == 0 and step > 0:
            print("-" * 70)
            print("Evaluating code-state alignment...")
            alignment = evaluate_state_alignment(model, fsm, device=device)

            # Print purity for each bottleneck
            for key, val in sorted(alignment.items()):
                if 'purity' in key:
                    print(f"  {key}: {val:.3f}")
            print("-" * 70)

    # Final evaluation
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Temperature per bottleneck
    temp_stats = get_temperature_stats(model)
    print("\nFinal temperatures:")
    for i in range(n_layers):
        attn_t = temp_stats['all'][i * 2]
        mlp_t = temp_stats['all'][i * 2 + 1]
        print(f"  Layer {i}: attn={attn_t:.4f}, mlp={mlp_t:.4f}")

    # Final alignment
    print("\nFinal code-state alignment (purity):")
    alignment = evaluate_state_alignment(model, fsm, num_batches=20, device=device)
    for key, val in sorted(alignment.items()):
        if 'purity' in key:
            print(f"  {key}: {val:.3f}")

    # Final accuracy
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for _ in range(20):
            inputs, targets, _ = fsm.generate_batch(batch_size, seq_len)
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += targets.numel()

    print(f"\nFinal accuracy: {total_correct / total_count:.4f}")
    print("=" * 70)

    return model, fsm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSM experiment")
    parser.add_argument("--states", type=int, default=10)
    parser.add_argument("--tokens-per-state", type=int, default=3)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--codebook", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lambda-compress", type=float, default=0.01)
    parser.add_argument("--lambda-commit", type=float, default=0.25)
    parser.add_argument("--temp-anneal", action="store_true", default=True)
    parser.add_argument("--no-temp-anneal", dest="temp_anneal", action="store_false")
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_fsm(
        num_states=args.states,
        tokens_per_state=args.tokens_per_state,
        dim=args.dim,
        n_layers=args.layers,
        codebook_size=args.codebook,
        n_steps=args.steps,
        lambda_compress=args.lambda_compress,
        lambda_commit=args.lambda_commit,
        temp_anneal=args.temp_anneal,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        device=args.device,
        seed=args.seed,
    )
