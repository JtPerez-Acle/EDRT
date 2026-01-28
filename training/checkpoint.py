"""Checkpoint saving and loading for training resumption."""

from pathlib import Path
from typing import Optional
import torch

from src.model import CrystallineTransformer


def save_checkpoint(
    path: str,
    model: CrystallineTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    epoch: int,
    config: dict,
    metrics: Optional[dict] = None,
):
    """
    Save full training state for resumption.

    Args:
        path: Path to save checkpoint
        model: The model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        step: Current training step
        epoch: Current epoch
        config: Training configuration dict
        metrics: Optional metrics to save
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'metrics': metrics or {},
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path} (step {step})")


def load_checkpoint(
    path: str,
    model: CrystallineTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
) -> dict:
    """
    Load checkpoint and restore state.

    Args:
        path: Path to checkpoint file
        model: Model to restore weights to
        optimizer: Optional optimizer to restore state to
        scheduler: Optional scheduler to restore state to
        device: Device to load to

    Returns:
        Checkpoint dict with step, epoch, config, metrics
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Loaded checkpoint from {path} (step {checkpoint.get('step', 'unknown')})")

    return {
        'step': checkpoint.get('step', 0),
        'epoch': checkpoint.get('epoch', 0),
        'config': checkpoint.get('config', {}),
        'metrics': checkpoint.get('metrics', {}),
    }


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search

    Returns:
        Path to latest checkpoint, or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None

    # Sort by step number extracted from filename
    def get_step(path):
        try:
            return int(path.stem.split('_')[1])
        except (IndexError, ValueError):
            return 0

    latest = max(checkpoints, key=get_step)
    return str(latest)


if __name__ == "__main__":
    import tempfile
    from src.config import ModelConfig, BottleneckConfig

    print("Testing checkpoint functions...")

    # Create model
    model_config = ModelConfig(
        vocab_size=1000,
        dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=64,
        bottleneck=BottleneckConfig(codebook_size=32, num_codes_k=4),
    )

    model = CrystallineTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Take a few optimizer steps to change state
    for _ in range(3):
        dummy_loss = model(torch.randint(0, 1000, (2, 32)))[0].mean()
        dummy_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = f"{tmpdir}/checkpoint_100.pt"

        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            epoch=5,
            config={'test': True},
            metrics={'loss': 1.5},
        )

        # Create new model and load
        model2 = CrystallineTransformer(model_config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=100)

        info = load_checkpoint(ckpt_path, model2, optimizer2, scheduler2)

        # Verify
        assert info['step'] == 100
        assert info['epoch'] == 5
        assert info['config']['test'] == True
        assert info['metrics']['loss'] == 1.5

        # Verify model weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"

        # Test latest checkpoint finder
        save_checkpoint(f"{tmpdir}/checkpoint_50.pt", model, optimizer, scheduler, 50, 2, {})
        save_checkpoint(f"{tmpdir}/checkpoint_200.pt", model, optimizer, scheduler, 200, 10, {})

        latest = get_latest_checkpoint(tmpdir)
        assert "200" in latest

    print("\nCheckpoint test passed!")
