# Crystalline Architecture

Technical deep-dive into the Crystalline framework.

## Overview

Crystalline extends the standard transformer architecture by inserting **learnable discrete bottlenecks** after each attention and MLP layer. These bottlenecks force information through a sparse codebook, encouraging the model to develop discrete internal representations.

```
Standard Transformer Block:        Crystalline Transformer Block:

  Input                              Input
    │                                  │
    ▼                                  ▼
┌───────────┐                    ┌───────────┐
│ Attention │                    │ Attention │
└─────┬─────┘                    └─────┬─────┘
      │                                │
      │ + residual                     ▼
      │                          ┌────────────┐
      │                          │ Bottleneck │──→ + residual
      │                          └────────────┘
      ▼                                │
┌───────────┐                          ▼
│    MLP    │                    ┌───────────┐
└─────┬─────┘                    │    MLP    │
      │                          └─────┬─────┘
      │ + residual                     │
      │                                ▼
      │                          ┌────────────┐
      │                          │ Bottleneck │──→ + residual
      │                          └────────────┘
      ▼                                │
   Output                              ▼
                                    Output
```

## Crystalline Bottleneck

The core innovation is the `CrystallineBottleneck` module (`src/bottleneck.py`).

### Architecture

```python
class CrystallineBottleneck(nn.Module):
    def __init__(self, dim, codebook_size, num_codes_k, temp_init, temp_min):
        # Learnable codebook: (codebook_size, dim)
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))

        # Learned temperature (clamped to temp_min)
        self._temperature = nn.Parameter(torch.tensor(temp_init))

        # Output scaling factor
        self.output_scale = nn.Parameter(torch.ones(1))
```

### Forward Pass

```python
def forward(self, x):
    # 1. Compute cosine similarities to codebook
    x_norm = F.normalize(x, dim=-1)
    codebook_norm = F.normalize(self.codebook, dim=-1)
    logits = x_norm @ codebook_norm.T  # (batch, seq, codebook_size)

    # 2. Gumbel-Softmax with learned temperature
    soft_codes, hard_codes = top_k_gumbel(logits, k=self.num_codes_k, tau=self.temperature)

    # 3. Reconstruct from codebook
    output = hard_codes @ self.codebook * self.output_scale

    # 4. Return output and info for loss computation
    return output, {
        'temperature': self.temperature,
        'soft_codes': soft_codes,  # For compression loss
        'hard_codes': hard_codes,  # For analysis
        'entropy': -(soft_codes * log(soft_codes)).sum(-1).mean(),
        ...
    }
```

### Key Design Choices

#### 1. Cosine Similarity

We use cosine similarity (normalized dot product) instead of Euclidean distance:
- More stable gradients
- Codebook vectors naturally cluster on unit sphere
- Temperature has consistent effect regardless of magnitude

#### 2. Gumbel-Softmax

The Gumbel-Softmax trick enables gradient flow through discrete sampling:

```python
def gumbel_softmax(logits, tau, hard=True):
    # Add Gumbel noise for stochastic sampling
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    y_soft = F.softmax((logits + gumbels) / tau, dim=-1)

    if hard:
        # Straight-through estimator
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft  # Gradient flows through y_soft
    return y_soft
```

**Temperature (τ) controls discreteness**:
- τ → ∞: Uniform distribution (maximum entropy)
- τ → 0: One-hot (minimum entropy, fully discrete)

#### 3. Top-k Selection

Instead of selecting just one code, we select the top-k:
- Allows richer representations
- Reduces pressure on individual codes
- k is typically 4-16

```python
def top_k_gumbel(logits, k, tau):
    soft = gumbel_softmax(logits, tau, hard=False)

    # Select top-k indices
    _, indices = soft.topk(k, dim=-1)

    # Create sparse hard codes
    hard = torch.zeros_like(soft)
    hard.scatter_(-1, indices, soft.gather(-1, indices))

    # Normalize so hard codes sum to 1
    hard = hard / hard.sum(-1, keepdim=True)

    return soft, hard - soft.detach() + soft
```

#### 4. Learned Temperature

Unlike VQ-VAE which uses fixed discretization, we let temperature be **learned**:

```python
@property
def temperature(self):
    return torch.clamp(self._temperature, min=self.temp_min)
```

This allows different bottlenecks to learn different discretization levels. In practice, we override with temperature annealing during training.

## Loss Functions

The total loss combines three components (`src/losses.py`):

```
L_total = L_prediction + λ_compress × L_compression + λ_commit × L_commitment
```

### Prediction Loss

Standard cross-entropy for next-token prediction:

```python
L_prediction = CrossEntropy(logits, targets)
```

### Compression Loss (Entropy Regularization)

Encourages low-entropy (discrete) code selections:

```python
def compression_loss(soft_codes):
    # Entropy: -Σ p log p
    entropy = -(soft_codes * torch.log(soft_codes + eps)).sum(dim=-1)
    return entropy.mean()
```

**Lower entropy = more discrete**. The compression loss pushes the model toward sharper selections.

### Commitment Loss

VQ-VAE style loss that encourages the input to "commit" to chosen codes:

```python
def commitment_loss(input, output):
    # L2 distance between input and reconstruction
    return F.mse_loss(input, output.detach())
```

This helps stabilize training and prevents codebook drift.

## Temperature Annealing

**Key insight**: Temperature annealing is essential for crystallization.

Without annealing, the compression loss provides only weak pressure toward discretization. The model can satisfy the loss with slightly sharper (but still soft) selections.

With annealing, we externally force temperature down over training:

```python
def apply_temperature_annealing(self):
    progress = self.step / self.max_steps
    target_temp = temp_start + progress * (temp_end - temp_start)

    for block in self.model.blocks:
        block.attn_bottleneck._temperature.fill_(target_temp)
        block.mlp_bottleneck._temperature.fill_(target_temp)
```

**Typical schedule**: 2.0 → 0.2 over training.

## Gradient Flow

A critical concern is ensuring gradients flow through the discrete bottleneck.

### Straight-Through Estimator

We use the straight-through estimator (STE):

```python
# Forward: use hard (discrete) codes
# Backward: gradient flows through soft codes
output = hard_codes - soft_codes.detach() + soft_codes
```

This means:
- **Forward pass**: Uses discrete hard codes for computation
- **Backward pass**: Gradients computed as if soft codes were used

### Verified Gradient Flow

The test suite verifies gradients flow to all learnable parameters:

```python
def test_gradient_flow_to_temperatures(self):
    output = model(input_ids)
    loss = output.mean()
    loss.backward()

    for block in model.blocks:
        assert block.attn_bottleneck._temperature.grad is not None
        assert block.mlp_bottleneck._temperature.grad is not None
```

## Model Configuration

Configuration is managed via dataclasses (`src/config.py`):

```python
@dataclass
class BottleneckConfig:
    codebook_size: int = 512      # Number of codes
    num_codes_k: int = 8          # Top-k active codes
    temp_init: float = 1.0        # Initial temperature
    temp_min: float = 0.1         # Minimum temperature (clamp)

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 256                # Hidden dimension
    n_layers: int = 4             # Number of transformer blocks
    n_heads: int = 4              # Attention heads
    max_seq_len: int = 512
    dropout: float = 0.1
    bottleneck: BottleneckConfig  # Nested config
```

## Evaluation Metrics

### Perplexity

Standard language modeling metric:

```python
perplexity = exp(average_cross_entropy_per_token)
```

### Crystallization Metrics

```python
def compute_crystallization_metrics(model, dataloader):
    return {
        'temperature_mean': ...,    # Average temperature across bottlenecks
        'temperature_min': ...,
        'temperature_max': ...,
        'entropy_mean': ...,        # Average code selection entropy
        'codebook_usage': ...,      # Fraction of codes being used
    }
```

### Code-State Alignment (FSM)

For FSM experiments, we measure how well codes align with ground-truth states:

```python
def compute_state_code_alignment(hard_codes, states, n_states):
    # Build co-occurrence matrix: (codebook_size, n_states)
    # Compute purity: average max(P(state|code)) over codes
    return {'purity': ..., 'alignment_matrix': ...}
```

## File Organization

```
src/
├── bottleneck.py    # CrystallineBottleneck class
├── model.py         # CrystallineTransformer (blocks + embeddings + head)
├── config.py        # Configuration dataclasses
├── losses.py        # Loss functions
└── utils.py         # Gumbel-Softmax, top-k selection, helpers

training/
├── train.py         # Trainer class with annealing logic
├── data.py          # Dataset loading (TinyStories)
├── eval.py          # Evaluation metrics
└── checkpoint.py    # Save/load training state
```

## Extension Points

### Custom Bottleneck Variants

You can create custom bottleneck types by subclassing:

```python
class MyBottleneck(CrystallineBottleneck):
    def forward(self, x):
        # Custom discretization logic
        ...
```

### Per-Head Temperature

Current implementation has one temperature per bottleneck. For per-head temperature:

```python
self._temperature = nn.Parameter(torch.ones(n_heads) * temp_init)
```

### Hierarchical Codebooks

For larger codebooks, consider hierarchical product quantization:

```python
# Two-level codebook: select from coarse, then fine
coarse_codes = self.coarse_codebook(x)
fine_codes = self.fine_codebook(x, coarse_codes)
```

## References

1. **Gumbel-Softmax**: Jang et al., 2016 - [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)
2. **VQ-VAE**: van den Oord et al., 2017 - [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
3. **Straight-Through Estimator**: Bengio et al., 2013 - [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
