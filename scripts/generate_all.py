#!/usr/bin/env python3
"""
Generate All: Run experiments and create visualizations for Crystalline.

This script:
1. Runs FSM validation experiment
2. Runs TinyStories training (optional, slower)
3. Generates all visualization figures
4. Saves everything to docs/figures/

Usage:
    # Quick mode (FSM only, ~5 min)
    python scripts/generate_all.py --quick

    # Full mode (FSM + TinyStories, ~30 min on CPU)
    python scripts/generate_all.py --full

    # Just generate figures from existing checkpoints
    python scripts/generate_all.py --figures-only
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_step(step: str):
    """Print a step marker."""
    print(f"\n>>> {step}\n")


def run_fsm_experiment(
    states: int = 8,
    steps: int = 2000,
    dim: int = 128,
    layers: int = 3,
    codebook: int = 32,
    save_checkpoint: bool = True,
) -> dict:
    """Run FSM validation experiment and return results."""
    print_header("FSM VALIDATION EXPERIMENT")

    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW

    from experiments.fsm.generate_data import FSMConfig, FiniteStateMachine, compute_state_code_alignment
    from src.model import CrystallineTransformer
    from src.config import ModelConfig, BottleneckConfig
    from src.losses import compression_loss, commitment_loss

    # Config
    SEED = 42
    BATCH_SIZE = 64
    SEQ_LEN = 64
    LR = 3e-4
    LAMBDA_COMPRESS = 0.01
    LAMBDA_COMMIT = 0.25
    TEMP_START = 2.0
    TEMP_END = 0.2

    torch.manual_seed(SEED)

    vocab_size = states * 3  # 3 tokens per state

    print(f"Configuration:")
    print(f"  States: {states}")
    print(f"  Vocabulary: {vocab_size}")
    print(f"  Model: dim={dim}, layers={layers}, codebook={codebook}")
    print(f"  Training: {steps} steps, batch={BATCH_SIZE}")
    print(f"  Temperature: {TEMP_START} → {TEMP_END}")

    # Create FSM
    fsm_config = FSMConfig(
        num_states=states,
        tokens_per_state=3,
        vocab_size=vocab_size,
        seed=SEED,
    )
    fsm = FiniteStateMachine(fsm_config)

    # Create model
    model_config = ModelConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=layers,
        n_heads=4,
        max_seq_len=SEQ_LEN,
        dropout=0.0,
        bottleneck=BottleneckConfig(
            codebook_size=codebook,
            num_codes_k=4,
            temp_init=TEMP_START,
            temp_min=0.1,
        ),
    )
    model = CrystallineTransformer(model_config)
    optimizer = AdamW(model.parameters(), lr=LR)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training history
    history = {
        "steps": [],
        "loss": [],
        "loss_pred": [],
        "loss_compress": [],
        "loss_commit": [],
        "accuracy": [],
        "temperature": [],
        "entropy": [],
    }

    # Training loop
    print_step("Training...")
    model.train()

    for step in range(steps):
        # Temperature annealing
        progress = step / max(steps - 1, 1)
        target_temp = TEMP_START + progress * (TEMP_END - TEMP_START)

        with torch.no_grad():
            for block in model.blocks:
                block.attn_bottleneck._temperature.fill_(target_temp)
                block.mlp_bottleneck._temperature.fill_(target_temp)

        # Generate batch
        inputs, targets, states_batch = fsm.generate_batch(BATCH_SIZE, SEQ_LEN)

        # Forward
        logits, infos = model(inputs)

        # Losses
        pred_loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

        compress_loss_val = torch.tensor(0.0)
        commit_loss_val = torch.tensor(0.0)
        n_bn = 0
        for layer_info in infos:
            for bn_type in ['attn', 'mlp']:
                compress_loss_val = compress_loss_val + compression_loss(layer_info[bn_type]['soft_codes'])
                commit_loss_val = commit_loss_val + commitment_loss(
                    layer_info[bn_type]['input'], layer_info[bn_type]['output']
                )
                n_bn += 1

        compress_loss_val = compress_loss_val / n_bn
        commit_loss_val = commit_loss_val / n_bn

        loss = pred_loss + LAMBDA_COMPRESS * compress_loss_val + LAMBDA_COMMIT * commit_loss_val

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Log
        if step % (steps // 20) == 0 or step == steps - 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == targets).float().mean().item()
                entropies = [info['attn']['entropy'].item() for info in infos]
                avg_entropy = sum(entropies) / len(entropies)

            history["steps"].append(step)
            history["loss"].append(loss.item())
            history["loss_pred"].append(pred_loss.item())
            history["loss_compress"].append(compress_loss_val.item())
            history["loss_commit"].append(commit_loss_val.item())
            history["accuracy"].append(acc)
            history["temperature"].append(target_temp)
            history["entropy"].append(avg_entropy)

            print(f"  Step {step:5d}/{steps} | Loss: {loss.item():.4f} | "
                  f"Acc: {acc:.3f} | Temp: {target_temp:.2f} | Entropy: {avg_entropy:.3f}")

    # Final evaluation
    print_step("Evaluating...")
    model.eval()

    all_hard_codes = []
    all_states = []
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for _ in range(20):
            inputs, targets, states_batch = fsm.generate_batch(BATCH_SIZE, SEQ_LEN)
            logits, infos = model(inputs)

            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += targets.numel()

            all_hard_codes.append(infos[0]['attn']['hard_codes'])
            all_states.append(states_batch)

    all_hard_codes = torch.cat(all_hard_codes, dim=0)
    all_states = torch.cat(all_states, dim=0)

    alignment = compute_state_code_alignment(all_hard_codes, all_states, states)
    final_acc = total_correct / total_count

    print(f"\nFinal Results:")
    print(f"  Accuracy: {final_acc:.4f} (random: {1/states:.4f}, improvement: {final_acc/(1/states):.1f}x)")
    print(f"  Purity: {alignment['purity']:.4f} (random: {1/states:.4f})")
    print(f"  Active codes: {alignment['active_codes']}/{codebook}")
    print(f"  Final entropy: {history['entropy'][-1]:.4f}")

    # Save checkpoint
    if save_checkpoint:
        ckpt_dir = PROJECT_ROOT / "checkpoints" / "fsm"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "checkpoint_final.pt"

        torch.save({
            "step": steps,
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "config": {
                "model": {
                    "vocab_size": vocab_size,
                    "dim": dim,
                    "n_layers": layers,
                    "n_heads": 4,
                    "max_seq_len": SEQ_LEN,
                    "dropout": 0.0,
                    "bottleneck": {
                        "codebook_size": codebook,
                        "num_codes_k": 4,
                        "temp_init": TEMP_START,
                        "temp_min": 0.1,
                    },
                },
                "fsm": {"num_states": states},
            },
            "metrics": {
                "accuracy": final_acc,
                "purity": alignment["purity"],
                "entropy": history["entropy"][-1],
            },
        }, ckpt_path)
        print(f"\n  Checkpoint saved: {ckpt_path}")

    return {
        "history": history,
        "final_accuracy": final_acc,
        "final_purity": alignment["purity"],
        "final_entropy": history["entropy"][-1],
        "alignment_matrix": alignment.get("code_state_counts"),
        "model": model,
        "fsm": fsm,
        "n_states": states,
        "codebook_size": codebook,
    }


def run_tinystories_experiment(
    steps: int = 5000,
    dim: int = 128,
    layers: int = 4,
    max_stories: int = 10000,
) -> dict:
    """Run TinyStories training experiment."""
    print_header("TINYSTORIES EXPERIMENT")

    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from src.model import CrystallineTransformer
    from src.config import ModelConfig, BottleneckConfig
    from src.losses import compression_loss, commitment_loss
    from training.data import DataConfig, create_dataloaders, load_tokenizer

    # Config
    SEED = 42
    BATCH_SIZE = 8
    SEQ_LEN = 128
    LR = 3e-4
    LAMBDA_COMPRESS = 0.01
    LAMBDA_COMMIT = 0.25
    TEMP_START = 2.0
    TEMP_END = 0.3
    CODEBOOK_SIZE = 256

    torch.manual_seed(SEED)

    print(f"Configuration:")
    print(f"  Model: dim={dim}, layers={layers}")
    print(f"  Training: {steps} steps, batch={BATCH_SIZE}, seq_len={SEQ_LEN}")
    print(f"  Max stories: {max_stories}")

    # Create dataloader
    print_step("Loading TinyStories dataset...")
    try:
        tokenizer = load_tokenizer()
        data_config = DataConfig(
            max_seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE,
            num_workers=0,  # Avoid multiprocessing issues
            max_stories=max_stories,
            val_split_size=500,
        )
        train_loader, val_loader = create_dataloaders(data_config, tokenizer)
        vocab_size = tokenizer.vocab_size
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Batches: {len(train_loader)}")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        print("  Skipping TinyStories experiment.")
        return None

    # Create model
    model_config = ModelConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=layers,
        n_heads=4,
        max_seq_len=SEQ_LEN,
        dropout=0.1,
        bottleneck=BottleneckConfig(
            codebook_size=CODEBOOK_SIZE,
            num_codes_k=8,
            temp_init=TEMP_START,
            temp_min=0.1,
        ),
    )
    model = CrystallineTransformer(model_config)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training history
    history = {
        "steps": [],
        "loss": [],
        "temperature": [],
        "entropy": [],
    }

    # Training loop
    print_step("Training...")
    model.train()
    data_iter = iter(train_loader)

    for step in range(steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Temperature annealing
        progress = step / max(steps - 1, 1)
        target_temp = TEMP_START + progress * (TEMP_END - TEMP_START)

        with torch.no_grad():
            for block in model.blocks:
                block.attn_bottleneck._temperature.fill_(target_temp)
                block.mlp_bottleneck._temperature.fill_(target_temp)

        # Forward
        logits, infos = model(input_ids)

        # Losses
        pred_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )

        compress_loss_val = torch.tensor(0.0)
        commit_loss_val = torch.tensor(0.0)
        n_bn = 0
        for layer_info in infos:
            for bn_type in ['attn', 'mlp']:
                compress_loss_val = compress_loss_val + compression_loss(layer_info[bn_type]['soft_codes'])
                commit_loss_val = commit_loss_val + commitment_loss(
                    layer_info[bn_type]['input'], layer_info[bn_type]['output']
                )
                n_bn += 1

        compress_loss_val = compress_loss_val / n_bn
        commit_loss_val = commit_loss_val / n_bn

        loss = pred_loss + LAMBDA_COMPRESS * compress_loss_val + LAMBDA_COMMIT * commit_loss_val

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Log
        if step % (steps // 20) == 0 or step == steps - 1:
            with torch.no_grad():
                entropies = [info['attn']['entropy'].item() for info in infos]
                avg_entropy = sum(entropies) / len(entropies)

            history["steps"].append(step)
            history["loss"].append(loss.item())
            history["temperature"].append(target_temp)
            history["entropy"].append(avg_entropy)

            print(f"  Step {step:5d}/{steps} | Loss: {loss.item():.4f} | "
                  f"Temp: {target_temp:.2f} | Entropy: {avg_entropy:.3f}")

    # Save checkpoint
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "tinystories"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint_final.pt"

    torch.save({
        "step": steps,
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "model": {
                "vocab_size": vocab_size,
                "dim": dim,
                "n_layers": layers,
                "n_heads": 4,
                "max_seq_len": SEQ_LEN,
                "dropout": 0.1,
                "bottleneck": {
                    "codebook_size": CODEBOOK_SIZE,
                    "num_codes_k": 8,
                    "temp_init": TEMP_START,
                    "temp_min": 0.1,
                },
            },
        },
        "metrics": {
            "loss": history["loss"][-1],
            "entropy": history["entropy"][-1],
        },
    }, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    return {
        "history": history,
        "model": model,
    }


def generate_figures(fsm_results: dict = None, tinystories_results: dict = None):
    """Generate all visualization figures."""
    print_header("GENERATING FIGURES")

    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    from analysis.style import setup_style, COLORS
    from analysis.visualize import (
        plot_crystallization_curve,
        plot_layer_temperatures,
        plot_codebook_usage,
        plot_code_state_alignment,
        plot_architecture_diagram,
    )

    figures_dir = PROJECT_ROOT / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_style("paper")

    generated = []

    # 1. Architecture diagram
    print_step("Architecture diagram...")
    fig = plot_architecture_diagram(n_layers=4, save_path=figures_dir / "architecture.png")
    plt.close(fig)
    generated.append("architecture.png")

    # 2. FSM results
    if fsm_results:
        print_step("FSM visualizations...")

        # Crystallization curve
        h = fsm_results["history"]
        fig = plot_crystallization_curve(
            steps=h["steps"],
            losses={
                "total": h["loss"],
                "pred": h["loss_pred"],
                "compress": h["loss_compress"],
                "commit": h["loss_commit"],
            },
            temperatures=h["temperature"],
            entropies=h["entropy"],
            title="FSM Training: Crystallization Dynamics",
            save_path=figures_dir / "fsm_crystallization.png",
        )
        plt.close(fig)
        generated.append("fsm_crystallization.png")

        # Layer temperatures (from final model)
        if fsm_results.get("model"):
            model = fsm_results["model"]
            temps = {"attn": [], "mlp": []}
            for block in model.blocks:
                temps["attn"].append(block.attn_bottleneck.temperature.item())
                temps["mlp"].append(block.mlp_bottleneck.temperature.item())

            fig = plot_layer_temperatures(
                temps,
                title="FSM: Final Layer Temperatures",
                save_path=figures_dir / "fsm_layer_temps.png",
            )
            plt.close(fig)
            generated.append("fsm_layer_temps.png")

    # 3. TinyStories results
    if tinystories_results:
        print_step("TinyStories visualizations...")

        h = tinystories_results["history"]

        # Simple training curve (loss + entropy)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(h["steps"], h["loss"], color=COLORS["loss_total"])
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")

        axes[1].plot(h["steps"], h["entropy"], color=COLORS["entropy"])
        axes[1].fill_between(h["steps"], h["entropy"], alpha=0.2, color=COLORS["entropy"])
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Entropy")
        axes[1].set_title("Code Selection Entropy")

        plt.tight_layout()
        fig.savefig(figures_dir / "tinystories_training.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        generated.append("tinystories_training.png")

    # 4. Load existing checkpoints for additional figures
    print_step("Checkpoint-based figures...")

    try:
        from analysis.checkpoint_analysis import load_checkpoint_for_analysis

        # Try TinyStories checkpoint
        ts_ckpt = PROJECT_ROOT / "checkpoints" / "tinystories" / "checkpoint_final.pt"
        if ts_ckpt.exists():
            result = load_checkpoint_for_analysis(ts_ckpt)

            fig = plot_layer_temperatures(
                result.bottleneck_stats["temperatures"],
                title="TinyStories: Layer Temperatures",
                save_path=figures_dir / "tinystories_layer_temps.png",
            )
            plt.close(fig)
            generated.append("tinystories_layer_temps.png")

        # Try FSM checkpoint
        fsm_ckpt = PROJECT_ROOT / "checkpoints" / "fsm" / "checkpoint_final.pt"
        if fsm_ckpt.exists():
            result = load_checkpoint_for_analysis(fsm_ckpt)
            # Already generated above if fsm_results was provided

    except Exception as e:
        print(f"  Warning: Could not load checkpoints: {e}")

    print(f"\nGenerated {len(generated)} figures:")
    for fig_name in generated:
        print(f"  - docs/figures/{fig_name}")

    return generated


def generate_summary_json(fsm_results: dict = None, tinystories_results: dict = None):
    """Generate a JSON summary of results."""
    print_step("Generating results summary...")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiments": {},
    }

    if fsm_results:
        summary["experiments"]["fsm"] = {
            "accuracy": fsm_results["final_accuracy"],
            "purity": fsm_results["final_purity"],
            "entropy": fsm_results["final_entropy"],
            "improvement_over_random": fsm_results["final_accuracy"] * fsm_results["n_states"],
            "steps": len(fsm_results["history"]["steps"]),
        }

    if tinystories_results:
        h = tinystories_results["history"]
        summary["experiments"]["tinystories"] = {
            "final_loss": h["loss"][-1],
            "final_entropy": h["entropy"][-1],
            "steps": len(h["steps"]),
        }

    summary_path = PROJECT_ROOT / "docs" / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate all Crystalline results and figures")
    parser.add_argument("--quick", action="store_true", help="Quick mode: FSM only, fewer steps")
    parser.add_argument("--full", action="store_true", help="Full mode: FSM + TinyStories")
    parser.add_argument("--figures-only", action="store_true", help="Only generate figures from existing checkpoints")
    parser.add_argument("--fsm-steps", type=int, default=2000, help="FSM training steps")
    parser.add_argument("--ts-steps", type=int, default=1000, help="TinyStories training steps")
    args = parser.parse_args()

    print_header("CRYSTALLINE: GENERATE ALL")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Mode: {'quick' if args.quick else 'full' if args.full else 'figures-only' if args.figures_only else 'default'}")

    fsm_results = None
    tinystories_results = None

    if not args.figures_only:
        # Run FSM experiment
        fsm_steps = 1500 if args.quick else args.fsm_steps
        fsm_results = run_fsm_experiment(
            states=8,
            steps=fsm_steps,
            dim=128,
            layers=3,
            codebook=32,
        )

        # Run TinyStories (full mode only)
        if args.full:
            tinystories_results = run_tinystories_experiment(
                steps=args.ts_steps,
                dim=128,
                layers=4,
                max_stories=10000,
            )

    # Generate figures
    generate_figures(fsm_results, tinystories_results)

    # Generate summary
    if fsm_results or tinystories_results:
        generate_summary_json(fsm_results, tinystories_results)

    print_header("COMPLETE")
    print("Repository is now ready with:")
    print("  - Trained checkpoints in checkpoints/")
    print("  - Visualization figures in docs/figures/")
    print("  - Results summary in docs/results_summary.json")
    print("\nNext steps:")
    print("  - View figures: ls docs/figures/")
    print("  - Explore interactively: jupyter notebook analysis/notebooks/")
    print("  - Run tests: pytest tests/ -v")


if __name__ == "__main__":
    main()
