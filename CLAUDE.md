# CLAUDE.md - Crystalline: Emergent Discrete Representations in Transformers

## Project Vision

We are building **Crystalline**, a training framework that enables transformers to discover their own discrete internal representations through learned compression pressure. Instead of forcing discretization architecturally, we let the model learn *when*, *where*, and *how much* to crystallize its continuous representations into sparse, interpretable codes.

**Core Hypothesis**: The fuzziness in LLMs is a training convenience, not a fundamental requirement. Given the right pressure, models will discover discrete symbolic structure where discrete structure is useful, while preserving continuous representations where nuance matters.

## Research Goals

1. **Emergent Crystallization**: Train models where discretization emerges from optimization pressure, not architectural mandate
2. **Learned Sparsity**: Each layer/head learns its own discretization temperature—some go hard symbolic, others stay fuzzy
3. **Interpretable Internals**: Discrete codes should be readable, causal, and composable
4. **Maintained Performance**: Match or approach standard transformer performance despite the bottleneck

## Technical Approach

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     CRYSTALLINE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Embeddings                                                │
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Transformer Block                                        │    │
│  │   ┌──────────────┐    ┌──────────────┐                  │    │
│  │   │  Attention   │───→│ Crystalline  │───→ (residual)   │    │
│  │   │    Heads     │    │  Bottleneck  │                  │    │
│  │   └──────────────┘    │  (learnable  │                  │    │
│  │                       │  temperature)│                  │    │
│  │   ┌──────────────┐    └──────────────┘                  │    │
│  │   │     MLP      │───→│ Crystalline  │───→ (residual)   │    │
│  │   │              │    │  Bottleneck  │                  │    │
│  │   └──────────────┘    └──────────────┘                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│       ↓ (repeat N layers)                                        │
│  Output Logits                                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Crystalline Bottleneck Module

```python
# Conceptual structure - actual implementation in src/
class CrystallineBottleneck(nn.Module):
    """
    Learnable discrete bottleneck with adaptive temperature.
    
    Key innovations:
    1. Temperature is LEARNED per-layer (or per-head, per-position)
    2. Gumbel-Softmax enables gradient flow through discretization
    3. Codebook is learned during training
    4. Compression pressure via entropy regularization
    """
    
    def __init__(self, dim, codebook_size, num_codes_k):
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))
        self.temperature = nn.Parameter(torch.ones(1))  # Learned!
        self.k = num_codes_k  # Top-k codes to activate
    
    def forward(self, x):
        # Compute similarities to codebook
        similarities = x @ self.codebook.T
        
        # Gumbel-Softmax with LEARNED temperature
        # Lower temp → more discrete, higher temp → more fuzzy
        soft_codes = gumbel_softmax(similarities, tau=self.temperature)
        
        # Top-k selection (straight-through estimator for gradients)
        hard_codes = top_k_selection(soft_codes, self.k)
        
        # Reconstruct from codebook
        output = hard_codes @ self.codebook
        
        return output, {
            'temperature': self.temperature,
            'active_codes': hard_codes,
            'entropy': compute_entropy(soft_codes)
        }
```

### Training Objective

```
L_total = L_prediction + λ_compress * L_compression + λ_commit * L_commitment

Where:
- L_prediction: Standard cross-entropy next-token prediction
- L_compression: Entropy regularization (reward low-entropy = more discrete)
- L_commitment: VQ-VAE style commitment loss (codes should commit to clusters)
```

### Key Hyperparameters to Explore

| Parameter | Description | Initial Range |
|-----------|-------------|---------------|
| `codebook_size` | Number of discrete codes per bottleneck | 512 - 4096 |
| `num_codes_k` | Top-k codes active per forward pass | 4 - 32 |
| `lambda_compress` | Compression pressure strength | 0.001 - 0.1 |
| `lambda_commit` | Commitment loss weight | 0.1 - 1.0 |
| `temp_init` | Initial temperature | 1.0 - 5.0 |
| `temp_min` | Minimum temperature (clamp) | 0.1 - 0.5 |

## Development Infrastructure

### Local Development (Ollama + CPU)

For rapid iteration and debugging:

```bash
# Pull a small model for architecture testing
ollama pull tinyllama
ollama pull phi

# Our training will use custom PyTorch, but Ollama helps for:
# - Baseline comparisons
# - Tokenizer reuse
# - Sanity checks
```

**Local Use Cases**:
- Unit tests for bottleneck modules
- Small-scale training on toy datasets (FSM, synthetic)
- Architecture iteration
- Debugging gradient flow

### Compute (Runpod)

For serious training runs:

```bash
# Runpod setup script (to be created)
# - PyTorch with CUDA
# - Weights & Biases for tracking
# - Persistent storage for checkpoints
```

**Runpod Use Cases**:
- Training on TinyStories, WikiText
- Scaling experiments (model size, codebook size)
- Ablation studies
- Final benchmark runs

### Environment Setup

```bash
# Local environment
python -m venv .venv
source .venv/bin/activate
pip install torch numpy transformers datasets wandb einops

# Key dependencies
# - torch: Core framework
# - einops: Clean tensor operations
# - transformers: Tokenizers, baseline models
# - datasets: HuggingFace datasets
# - wandb: Experiment tracking
```

## Project Structure

```
crystalline/
├── CLAUDE.md                 # This file
├── README.md                 # Public-facing documentation
├── pyproject.toml            # Project config
│
├── src/
│   ├── __init__.py
│   ├── bottleneck.py         # CrystallineBottleneck module
│   ├── model.py              # Full CrystallineTransformer
│   ├── config.py             # Model/training configurations
│   ├── losses.py             # Custom loss functions
│   └── utils.py              # Helpers
│
├── training/
│   ├── train.py              # Main training loop
│   ├── data.py               # Dataset loading/processing
│   └── eval.py               # Evaluation and metrics
│
├── experiments/
│   ├── fsm/                  # Finite state machine experiments
│   │   ├── generate_data.py
│   │   └── run_fsm.py
│   ├── tinystories/          # TinyStories experiments
│   └── ablations/            # Hyperparameter studies
│
├── analysis/
│   ├── interpret_codes.py    # Code interpretation tools
│   ├── visualize.py          # Visualization utilities
│   └── interventions.py      # Causal intervention experiments
│
├── scripts/
│   ├── setup_runpod.sh       # Runpod initialization
│   └── download_data.sh      # Dataset download
│
└── tests/
    ├── test_bottleneck.py
    ├── test_gradients.py
    └── test_training.py
```

## Development Phases

### Phase 1: Foundation (Current)
- [ ] Implement CrystallineBottleneck module
- [ ] Verify gradient flow through Gumbel-Softmax
- [ ] Unit tests for core components
- [ ] Toy experiment: learn to discretize a known structure

### Phase 2: FSM Replication
- [ ] Replicate Stanford's FSM experiment setup
- [ ] Train CrystallineTransformer on FSM data
- [ ] Verify codes learn to represent states
- [ ] Compare learned temperatures across layers

### Phase 3: Language Modeling
- [ ] Integrate with TinyStories dataset
- [ ] Train small (10-50M param) models
- [ ] Evaluate perplexity vs. standard transformer
- [ ] Analyze code interpretability

### Phase 4: Scaling & Analysis
- [ ] Scale to larger models on Runpod
- [ ] Comprehensive ablation studies
- [ ] Intervention experiments (causal verification)
- [ ] Write up findings

## Key Experiments

### Experiment 1: Gradient Verification
**Goal**: Confirm gradients flow correctly through discretization
```python
# Test that temperature gradients exist and are meaningful
# Test that codebook gradients update correctly
# Test straight-through estimator behavior
```

### Experiment 2: FSM State Learning
**Goal**: Replicate codebook features result on finite state machine
- 100-state FSM with known transitions
- Train CrystallineTransformer to predict next token
- Measure: do codes correspond 1:1 with states?
- Measure: does temperature decrease (more discrete) over training?

### Experiment 3: Layer-wise Crystallization
**Goal**: Do different layers learn different discretization levels?
- Train on language modeling
- Plot temperature per layer over training
- Hypothesis: early layers stay warmer (fuzzy), late layers cool (discrete)

### Experiment 4: Compression-Performance Tradeoff
**Goal**: Map the Pareto frontier of compression vs. accuracy
- Sweep `lambda_compress` from 0 to 0.5
- Measure perplexity and average code entropy
- Find sweet spot where discrete structure emerges without killing performance

### Experiment 5: Causal Interventions
**Goal**: Verify codes are causally meaningful
- Identify codes that correlate with concepts (topics, syntax, etc.)
- Intervene by activating/deactivating codes
- Measure effect on generation

## Metrics to Track

### Training Metrics
- `loss/total`: Combined loss
- `loss/prediction`: Next-token prediction loss
- `loss/compression`: Entropy regularization
- `loss/commitment`: VQ commitment loss
- `temperature/layer_{i}`: Learned temperature per layer
- `codebook/usage`: Fraction of codebook being used
- `codebook/entropy`: Average entropy of code selections

### Evaluation Metrics
- `perplexity`: Standard language model metric
- `code_precision`: How precisely codes identify concepts (FSM states, linguistic features)
- `intervention_effect`: Causal effect size of code interventions

## Coding Standards

### Style
- Use type hints everywhere
- Docstrings for all public functions
- Keep modules focused and small
- Prefer einops for tensor operations

### Testing
- Test gradient flow explicitly
- Test numerical stability (especially Gumbel-Softmax at low temps)
- Test on tiny configs before scaling

### Experiment Tracking
- All runs logged to W&B
- Configs saved with checkpoints
- Seeds set for reproducibility

## References

### Must-Read Papers
1. **Codebook Features** (Tamkin et al., 2023) - Direct inspiration
   https://arxiv.org/abs/2310.17230
   
2. **VQ-VAE** (van den Oord et al., 2017) - Vector quantization foundation
   https://arxiv.org/abs/1711.00937
   
3. **Gumbel-Softmax** (Jang et al., 2016) - Differentiable discretization
   https://arxiv.org/abs/1611.01144
   
4. **Towards Monosemanticity** (Anthropic, 2023) - Why this matters
   https://transformer-circuits.pub/2023/monosemantic-features

### Related Work
- Discrete Key-Value Bottleneck (Träuble et al., 2023)
- Information Bottleneck Theory (Tishby et al.)
- Sparse Autoencoders for interpretability

## Quick Start Commands

```bash
# Setup
git clone <repo>
cd crystalline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/

# Train on FSM (local)
python experiments/fsm/run_fsm.py --config configs/fsm_tiny.yaml

# Train on TinyStories (Runpod)
python training/train.py --config configs/tinystories_small.yaml

# Analyze codes
python analysis/interpret_codes.py --checkpoint checkpoints/latest.pt
```

## Notes for Claude Code

When working on this project:

1. **Start small**: Always test on tiny configs first (2 layers, 64 dim, 128 codebook)
2. **Verify gradients**: The Gumbel-Softmax can be numerically unstable at low temperatures
3. **Track temperatures**: The learned temperature is our key signal - watch it closely
4. **Codebook collapse**: Watch for unused codes (common failure mode in VQ)
5. **Baseline first**: Always compare against a standard transformer baseline

The goal is not just to make this work, but to understand *why* it works and *what* the model learns. Interpretability is a first-class citizen, not an afterthought.

---

*"The continuous math is just how we found the discrete structure, not the structure itself."*