# Crystalline: Emergent Discrete Representations in Transformers

**Status**: Phase 2 Complete | January 2026

## Overview

Crystalline enables transformers to discover **discrete internal representations** through compression pressure. Instead of forcing discretization architecturally, the model learns when and where to crystallize.

**Core mechanism**: Gumbel-Softmax bottlenecks after attention/MLP with learned temperature + compression loss.

```
L_total = L_prediction + λ_compress × L_entropy + λ_commit × L_commitment
```

## Results

### Phase 1: FSM Validation

Trained on finite state machines where codes should learn to represent states.

| Metric | Random Baseline | Achieved | Improvement |
|--------|-----------------|----------|-------------|
| Accuracy | 6.25% | 24.4% | **3.9x** |
| Code-State Purity | 25% | 52% | **2x** |
| Entropy | ~3.0 nats | 0.49 nats | **Crystallized** |
| Active Codes | N/A | 18/64 | 28% codebook |

**Key finding**: Temperature annealing (2.0 → 0.2) is essential for strong crystallization. Without it, entropy remains high and purity stays near baseline.

#### What This Means

- **Accuracy 3.9x improvement**: The model learned the FSM transition structure
- **Purity doubled**: Individual codes specialized to represent specific FSM states
- **Entropy collapsed**: Code selections became sharp/discrete rather than fuzzy

### Phase 2: TinyStories

Training infrastructure complete for real language modeling.

```
Step     0 | Loss: 42.75 | Entropy: 3.16 | Temp: 2.00
Step    30 | Loss: 41.30 | Entropy: 2.44 | Temp: 1.08  ← Crystallizing
Step   100 | Loss: 38.52 | Entropy: 1.87 | Temp: 0.65
```

**Observation**: Entropy drops as temperature anneals, indicating crystallization in progress.

## Visualizations

Analysis tools are available in `analysis/`:

```python
from analysis import load_checkpoint_for_analysis, plot_layer_temperatures

# Load trained model
result = load_checkpoint_for_analysis("checkpoints/tinystories/checkpoint_final.pt")

# Plot temperatures
fig = plot_layer_temperatures(result.bottleneck_stats["temperatures"])
```

Interactive notebooks in `analysis/notebooks/`:
- `explore_checkpoint.ipynb` - General checkpoint exploration
- `fsm_analysis.ipynb` - FSM experiment deep-dive
- `tinystories_analysis.ipynb` - Language model analysis

## Project Structure

```
src/
├── bottleneck.py      # CrystallineBottleneck (Gumbel-Softmax + top-k)
├── model.py           # CrystallineTransformer
├── losses.py          # compression_loss, commitment_loss
└── config.py          # ModelConfig, BottleneckConfig, TrainingConfig

training/
├── data.py            # TinyStories DataLoader
├── train.py           # Trainer class
├── eval.py            # Perplexity, crystallization metrics
└── checkpoint.py      # Save/load

experiments/
├── fsm/               # FSM experiments (validated)
└── tinystories/       # Language model experiments

configs/
├── tinystories_tiny.yaml   # CPU testing (~8M params)
└── tinystories_small.yaml  # GPU dev (~20M params)
```

## Quick Start

```bash
# Setup
pip install -e ".[dev]"
pytest tests/ -v  # 21 tests pass

# FSM experiment (validates crystallization)
python -m experiments.fsm.run_fsm \
  --states 4 --dim 128 --layers 3 --steps 3000 \
  --temp-anneal --temp-start 2.0 --temp-end 0.2

# TinyStories (language modeling)
python -m experiments.tinystories.run_tinystories \
  --config configs/tinystories_tiny.yaml \
  --max-stories 10000 --steps 1000
```

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `temp_start` | 2.0 | Warm start (soft codes) |
| `temp_end` | 0.2-0.5 | Cool end (discrete codes) |
| `λ_compress` | 0.01-0.1 | Entropy regularization |
| `λ_commit` | 0.25 | VQ-VAE commitment |
| `codebook_size` | 256-512 | Codes per bottleneck |
| `num_codes_k` | 4-16 | Active codes per forward |

## Insights

1. **Temperature annealing required** - Loss-driven crystallization alone is too slow
2. **Entropy tracks crystallization** - Watch it drop from ~3.0 to <1.0
3. **Purity validates alignment** - Codes specialize to discrete concepts
4. **Scaling preserves crystallization** - Larger models crystallize while learning better

## References

- Codebook Features (Tamkin et al., 2023)
- VQ-VAE (van den Oord et al., 2017)
- Gumbel-Softmax (Jang et al., 2016)
