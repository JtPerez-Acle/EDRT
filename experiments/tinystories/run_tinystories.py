#!/usr/bin/env python3
"""
TinyStories Experiment: Train CrystallineTransformer on natural language.

This experiment tests whether:
1. Crystalline bottlenecks can learn meaningful discrete codes for language
2. Different layers develop different discretization levels
3. The model maintains reasonable perplexity while developing interpretable structure

Usage:
    python -m experiments.tinystories.run_tinystories --config configs/tinystories_tiny.yaml
    python -m experiments.tinystories.run_tinystories --config configs/tinystories_tiny.yaml --steps 1000
"""

import argparse
from pathlib import Path
from dataclasses import asdict

import yaml
import torch

from src.model import CrystallineTransformer
from src.config import ModelConfig, BottleneckConfig
from training.data import DataConfig, create_dataloaders, load_tokenizer
from training.train import Trainer, TrainConfig
from training.eval import evaluate_model
from training.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with command-line arguments."""
    # Training overrides
    if args.steps is not None:
        config['training']['max_steps'] = args.steps
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.lambda_compress is not None:
        config['training']['lambda_compress'] = args.lambda_compress
    if args.lambda_commit is not None:
        config['training']['lambda_commit'] = args.lambda_commit
    if args.temp_start is not None:
        config['training']['temp_start'] = args.temp_start
    if args.temp_end is not None:
        config['training']['temp_end'] = args.temp_end
    if args.device is not None:
        config['training']['device'] = args.device

    # Data overrides
    if args.seq_len is not None:
        config['data']['max_seq_len'] = args.seq_len
        config['model']['max_seq_len'] = args.seq_len

    return config


def create_model_from_config(config: dict) -> CrystallineTransformer:
    """Create model from config dict."""
    bottleneck_config = BottleneckConfig(
        codebook_size=config['bottleneck']['codebook_size'],
        num_codes_k=config['bottleneck']['num_codes_k'],
        temp_init=config['bottleneck'].get('temp_init', 2.0),
        temp_min=config['bottleneck'].get('temp_min', 0.1),
    )

    model_config = ModelConfig(
        vocab_size=config['model']['vocab_size'],
        dim=config['model']['dim'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=config['model'].get('dropout', 0.0),
        bottleneck=bottleneck_config,
    )

    return CrystallineTransformer(model_config)


def main():
    parser = argparse.ArgumentParser(
        description="TinyStories Crystalline Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from (or 'latest' to find latest)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/tinystories",
        help="Directory for checkpoints"
    )

    # Training overrides
    parser.add_argument("--steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--lambda-compress", type=float, default=None, help="Compression loss weight")
    parser.add_argument("--lambda-commit", type=float, default=None, help="Commitment loss weight")
    parser.add_argument("--temp-start", type=float, default=None, help="Starting temperature")
    parser.add_argument("--temp-end", type=float, default=None, help="Ending temperature")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-stories", type=int, default=None, help="Limit stories (for quick testing)")

    args = parser.parse_args()

    # Load and merge config
    config = load_config(args.config)
    config = merge_config_with_args(config, args)

    # Set seed
    torch.manual_seed(args.seed)

    # Determine device
    device = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer_name = config['data'].get('tokenizer_name', 'gpt2')
    tokenizer = load_tokenizer(tokenizer_name)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Verify vocab size matches
    if config['model']['vocab_size'] != tokenizer.vocab_size:
        print(f"Warning: Config vocab_size ({config['model']['vocab_size']}) != "
              f"tokenizer vocab_size ({tokenizer.vocab_size})")
        print(f"Using tokenizer vocab_size: {tokenizer.vocab_size}")
        config['model']['vocab_size'] = tokenizer.vocab_size

    # Create data loaders
    data_config = DataConfig(
        dataset_name=config['data'].get('dataset_name', 'roneneldan/TinyStories'),
        tokenizer_name=tokenizer_name,
        max_seq_len=config['data']['max_seq_len'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data'].get('num_workers', 4),
        pack_sequences=config['data'].get('pack_sequences', True),
        val_split_size=config['data'].get('val_split_size', 1000),
        max_stories=args.max_stories or config['data'].get('max_stories'),
    )

    train_loader, val_loader = create_dataloaders(data_config, tokenizer)

    # Create model
    model = create_model_from_config(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    train_config = TrainConfig(
        max_steps=config['training']['max_steps'],
        batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01),
        warmup_steps=config['training'].get('warmup_steps', 1000),
        lambda_compress=config['training']['lambda_compress'],
        lambda_commit=config['training']['lambda_commit'],
        temp_anneal=config['training'].get('temp_anneal', True),
        temp_start=config['training'].get('temp_start', 2.0),
        temp_end=config['training'].get('temp_end', 0.5),
        log_every=config['training'].get('log_every', 100),
        eval_every=config['training'].get('eval_every', 1000),
        save_every=config['training'].get('save_every', 5000),
        device=device,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        vocab_size=tokenizer.vocab_size,
    )

    # Resume from checkpoint if specified
    if args.resume:
        if args.resume == 'latest':
            ckpt_path = get_latest_checkpoint(args.output_dir)
            if ckpt_path is None:
                print("No checkpoint found, starting from scratch")
            else:
                args.resume = ckpt_path

        if args.resume and args.resume != 'latest':
            ckpt_info = load_checkpoint(
                args.resume, model, trainer.optimizer, trainer.scheduler, device
            )
            trainer.step = ckpt_info['step']
            trainer.epoch = ckpt_info['epoch']
            print(f"Resumed from step {trainer.step}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define callbacks
    def eval_callback(step: int):
        print("-" * 70)
        print(f"Evaluating at step {step}...")
        metrics = evaluate_model(model, val_loader, device, max_batches=50)
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  Temperature: {metrics['temperature_mean']:.3f} "
              f"[{metrics['temperature_min']:.3f}-{metrics['temperature_max']:.3f}]")
        print(f"  Entropy: {metrics['entropy_mean']:.3f}")
        print(f"  Codebook usage: {metrics['codebook_usage']:.1%}")
        print("-" * 70)

    def save_callback(step: int):
        ckpt_path = output_dir / f"checkpoint_{step}.pt"
        save_checkpoint(
            path=str(ckpt_path),
            model=model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            step=step,
            epoch=trainer.epoch,
            config=config,
        )

    # Train
    trainer.train(
        eval_fn=eval_callback,
        save_fn=save_callback,
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    final_metrics = evaluate_model(model, val_loader, device, max_batches=100)
    print(f"Final perplexity: {final_metrics['perplexity']:.2f}")
    print(f"Final temperature: {final_metrics['temperature_mean']:.3f}")
    print(f"Final entropy: {final_metrics['entropy_mean']:.3f}")
    print(f"Final codebook usage: {final_metrics['codebook_usage']:.1%}")

    # Save final checkpoint
    save_checkpoint(
        path=str(output_dir / "checkpoint_final.pt"),
        model=model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        step=trainer.step,
        epoch=trainer.epoch,
        config=config,
        metrics=final_metrics,
    )


if __name__ == "__main__":
    main()
