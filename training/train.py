"""Main training loop for Crystalline models."""

from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.model import CrystallineTransformer
from src.losses import compression_loss, commitment_loss


@dataclass
class TrainConfig:
    """Training configuration."""
    # Training
    max_steps: int = 50000
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Crystallization losses
    lambda_compress: float = 0.01
    lambda_commit: float = 0.25

    # Temperature annealing
    temp_anneal: bool = True
    temp_start: float = 2.0
    temp_end: float = 0.5

    # Logging and checkpointing
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class Trainer:
    """
    Trainer for CrystallineTransformer models.

    Handles:
    - Training loop with gradient accumulation
    - Temperature annealing for crystallization
    - Loss computation (prediction + compression + commitment)
    - Logging and evaluation
    """

    def __init__(
        self,
        model: CrystallineTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        vocab_size: int,
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.vocab_size = vocab_size
        self.device = config.device

        # Set seed
        torch.manual_seed(config.seed)

        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate / 10,
        )

        # Track state
        self.step = 0
        self.epoch = 0

    def get_temperature_stats(self) -> dict:
        """Extract temperature statistics from all bottlenecks."""
        temps = []
        for block in self.model.blocks:
            temps.append(block.attn_bottleneck.temperature.item())
            temps.append(block.mlp_bottleneck.temperature.item())

        return {
            'mean': sum(temps) / len(temps),
            'min': min(temps),
            'max': max(temps),
            'all': temps,
        }

    def apply_temperature_annealing(self):
        """Anneal temperature across all bottlenecks based on training progress."""
        if not self.config.temp_anneal:
            return

        progress = self.step / max(self.config.max_steps - 1, 1)
        target_temp = (
            self.config.temp_start +
            progress * (self.config.temp_end - self.config.temp_start)
        )

        with torch.no_grad():
            for block in self.model.blocks:
                block.attn_bottleneck._temperature.fill_(target_temp)
                block.mlp_bottleneck._temperature.fill_(target_temp)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        infos: list[dict],
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute total loss with all components.

        Args:
            logits: Model output logits (batch, seq_len, vocab_size)
            labels: Target labels (batch, seq_len)
            infos: List of bottleneck info dicts from each layer

        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dictionary with individual loss components
        """
        # Prediction loss
        pred_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,  # Ignore padding
        )

        # Aggregate compression and commitment losses from all bottlenecks
        compress_loss_total = torch.tensor(0.0, device=self.device)
        commit_loss_total = torch.tensor(0.0, device=self.device)
        n_bottlenecks = 0

        for layer_info in infos:
            for bn_type in ['attn', 'mlp']:
                bn_info = layer_info[bn_type]
                compress_loss_total = compress_loss_total + compression_loss(bn_info['soft_codes'])
                commit_loss_total = commit_loss_total + commitment_loss(
                    bn_info['input'], bn_info['output']
                )
                n_bottlenecks += 1

        compress_loss_avg = compress_loss_total / n_bottlenecks
        commit_loss_avg = commit_loss_total / n_bottlenecks

        # Total loss
        total_loss = (
            pred_loss +
            self.config.lambda_compress * compress_loss_avg +
            self.config.lambda_commit * commit_loss_avg
        )

        return total_loss, {
            'total': total_loss.item(),
            'prediction': pred_loss.item(),
            'compression': compress_loss_avg.item(),
            'commitment': commit_loss_avg.item(),
        }

    def train_step(self, batch: dict) -> dict:
        """
        Execute a single training step.

        Args:
            batch: Dictionary with 'input_ids' and 'labels'

        Returns:
            Dictionary with loss values
        """
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        logits, infos = self.model(input_ids)

        # Compute loss
        loss, loss_dict = self.compute_loss(logits, labels, infos)

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        return loss_dict

    def train(
        self,
        log_fn: Optional[callable] = None,
        eval_fn: Optional[callable] = None,
        save_fn: Optional[callable] = None,
    ):
        """
        Main training loop.

        Args:
            log_fn: Optional callback for logging (receives step, losses, temps, lr)
            eval_fn: Optional callback for evaluation (receives step)
            save_fn: Optional callback for checkpointing (receives step)
        """
        self.model.train()
        data_iter = iter(self.train_loader)

        print("=" * 70)
        print("TINYSTORIES TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Lambda compress: {self.config.lambda_compress}")
        print(f"Lambda commit: {self.config.lambda_commit}")
        if self.config.temp_anneal:
            print(f"Temperature annealing: {self.config.temp_start} -> {self.config.temp_end}")
        print("=" * 70)

        while self.step < self.config.max_steps:
            # Get next batch, restart data iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Apply temperature annealing
            self.apply_temperature_annealing()

            # Training step
            loss_dict = self.train_step(batch)

            # Gradient accumulation
            if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Logging
            if self.step % self.config.log_every == 0:
                temp_stats = self.get_temperature_stats()
                lr = self.scheduler.get_last_lr()[0]

                if log_fn:
                    log_fn(self.step, loss_dict, temp_stats, lr)
                else:
                    # Default console logging
                    print(
                        f"Step {self.step:6d} | "
                        f"Loss: {loss_dict['total']:.4f} "
                        f"(pred={loss_dict['prediction']:.3f}, "
                        f"comp={loss_dict['compression']:.3f}, "
                        f"commit={loss_dict['commitment']:.3f}) | "
                        f"Temp: {temp_stats['mean']:.3f} "
                        f"[{temp_stats['min']:.3f}-{temp_stats['max']:.3f}]"
                    )

            # Evaluation
            if self.step % self.config.eval_every == 0 and self.step > 0:
                if eval_fn:
                    eval_fn(self.step)

            # Checkpointing
            if self.step % self.config.save_every == 0 and self.step > 0:
                if save_fn:
                    save_fn(self.step)

            self.step += 1

        print("=" * 70)
        print("Training complete!")
        print("=" * 70)


if __name__ == "__main__":
    # Quick test of training loop structure
    from src.config import ModelConfig, BottleneckConfig

    print("Testing Trainer initialization...")

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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy data loader
    dummy_data = [
        {'input_ids': torch.randint(0, 1000, (8, 64)),
         'labels': torch.randint(0, 1000, (8, 64))}
        for _ in range(10)
    ]

    class DummyLoader:
        def __iter__(self):
            return iter(dummy_data)

    train_config = TrainConfig(
        max_steps=5,
        batch_size=8,
        log_every=1,
    )

    trainer = Trainer(
        model=model,
        train_loader=DummyLoader(),
        val_loader=DummyLoader(),
        config=train_config,
        vocab_size=1000,
    )

    # Run a few steps
    trainer.train()
    print("\nTrainer test passed!")
