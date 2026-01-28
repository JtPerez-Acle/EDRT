#!/bin/bash
#
# Setup Crystalline Repository
#
# This script installs dependencies, runs experiments, and generates
# all visualizations to leave the repository in a ready state.
#
# Usage:
#   ./scripts/setup_repo.sh          # Quick setup (FSM only)
#   ./scripts/setup_repo.sh --full   # Full setup (FSM + TinyStories)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================"
echo "  Crystalline Repository Setup"
echo "========================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo ">>> Installing package and dependencies with uv..."
uv pip install -e ".[dev,analysis]"

echo ""
echo ">>> Running tests..."
uv run pytest tests/ -v --tb=short

echo ""
echo ">>> Generating results and figures..."
if [ "$1" = "--full" ]; then
    uv run python scripts/generate_all.py --full
else
    uv run python scripts/generate_all.py --quick
fi

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "What was created:"
echo "  - checkpoints/fsm/checkpoint_final.pt"
if [ "$1" = "--full" ]; then
    echo "  - checkpoints/tinystories/checkpoint_final.pt"
fi
echo "  - docs/figures/*.png"
echo "  - docs/results_summary.json"
echo ""
echo "Try these commands:"
echo "  uv run pytest tests/ -v       # Run tests"
echo "  uv run jupyter notebook       # Explore notebooks"
echo "  cat docs/results_summary.json # View results"
echo ""
