"""Configuration classes for Crystalline models and training."""

from dataclasses import dataclass


@dataclass
class BottleneckConfig:
    """Configuration for CrystallineBottleneck."""

    codebook_size: int = 512
    num_codes_k: int = 8
    temp_init: float = 1.0
    temp_min: float = 0.1


@dataclass
class ModelConfig:
    """Configuration for CrystallineTransformer."""

    vocab_size: int = 32000
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 512
    dropout: float = 0.1

    # Bottleneck config
    bottleneck: BottleneckConfig = None

    def __post_init__(self):
        if self.bottleneck is None:
            self.bottleneck = BottleneckConfig()


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 1000

    # Loss weights
    lambda_compress: float = 0.01
    lambda_commit: float = 0.25

    # Logging
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
