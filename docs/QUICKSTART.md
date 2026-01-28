# Crystalline Quick Start Guide

Get up and running with Crystalline in 5 minutes.

## Prerequisites

- Python 3.10+
- pip
- (Optional) CUDA-capable GPU for faster training

## Fastest Way: One-Command Setup

```bash
# This installs deps, runs tests, trains models, and generates figures
./scripts/setup_repo.sh

# Or for full setup including TinyStories
./scripts/setup_repo.sh --full
```

After this completes, you'll have:
- Trained checkpoints in `checkpoints/`
- Visualization figures in `docs/figures/`
- Results summary in `docs/results_summary.json`

## Manual Installation

```bash
# Clone the repository
git clone https://github.com/example/crystalline.git
cd crystalline

# Install with uv (recommended)
uv pip install -e ".[dev,analysis]"

# Or with pip in a venv
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,analysis]"
```

## Verify Installation

```bash
# Run the test suite
pytest tests/ -v

# Expected: 21 tests passed
```

## Your First Experiment: FSM Validation

The FSM (Finite State Machine) experiment is the best way to verify crystallization works. It trains a model to predict sequences from a state machine, where codes should learn to represent states.

### Quick Run (CPU, ~5 minutes)

```bash
python -m experiments.fsm.run_fsm \
    --states 4 \
    --dim 128 \
    --layers 3 \
    --steps 1000 \
    --temp-anneal \
    --temp-start 2.0 \
    --temp-end 0.2
```

### What to Watch For

As training progresses, you should see:

1. **Loss decreasing** - The model is learning
2. **Temperature dropping** - Following the annealing schedule (2.0 → 0.2)
3. **Accuracy improving** - From ~6% (random) toward 20%+

Sample output:
```
Step     0 | Loss: 3.4567 | Acc: 0.062 | Temp: 2.000
Step   100 | Loss: 2.8901 | Acc: 0.124 | Temp: 1.800
Step   500 | Loss: 1.5432 | Acc: 0.198 | Temp: 1.000
Step  1000 | Loss: 0.9876 | Acc: 0.244 | Temp: 0.200
```

### Full Experiment (CPU, ~15 minutes)

```bash
python -m experiments.fsm.run_fsm \
    --states 10 \
    --dim 128 \
    --layers 4 \
    --steps 5000 \
    --temp-anneal \
    --temp-start 2.0 \
    --temp-end 0.2
```

## TinyStories: Language Modeling

Once FSM validates crystallization works, try language modeling:

### CPU Testing

```bash
python -m experiments.tinystories.run_tinystories \
    --config configs/tinystories_tiny.yaml \
    --max-stories 10000 \
    --steps 1000
```

### GPU Training

```bash
python -m experiments.tinystories.run_tinystories \
    --config configs/tinystories_small.yaml \
    --steps 20000
```

## Analyzing Results

### Load a Checkpoint

```python
from analysis import load_checkpoint_for_analysis

result = load_checkpoint_for_analysis("checkpoints/tinystories/checkpoint_final.pt")

print(f"Training step: {result.step}")
print(f"Mean temperature: {result.bottleneck_stats['temperature_summary']['mean']:.3f}")
print(f"Layers: {result.bottleneck_stats['n_layers']}")
```

### Generate Visualizations

```python
from analysis import plot_layer_temperatures, setup_style

setup_style("paper")
fig = plot_layer_temperatures(
    result.bottleneck_stats["temperatures"],
    save_path="layer_temps.png"
)
```

### Interactive Exploration (Jupyter)

```bash
jupyter notebook analysis/notebooks/explore_checkpoint.ipynb
```

## Next Steps

1. **Read the full README** - [README.md](../README.md)
2. **Understand the architecture** - [ARCHITECTURE.md](ARCHITECTURE.md)
3. **View detailed results** - [CRYSTALLINE_RESULTS.md](CRYSTALLINE_RESULTS.md)
4. **Explore in Jupyter** - `analysis/notebooks/`

## Common Issues

### "Module not found" errors

Make sure you installed the package in editable mode:
```bash
pip install -e .
```

### Out of memory on GPU

Use the tiny config or reduce batch size:
```bash
python -m experiments.tinystories.run_tinystories \
    --config configs/tinystories_tiny.yaml \
    --batch-size 4
```

### Training seems stuck

Check that temperature annealing is enabled (`--temp-anneal`). Without it, crystallization is much slower.
