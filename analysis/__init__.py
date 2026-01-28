"""
Analysis and visualization tools for Crystalline models.

This module provides tools for:
- Loading and analyzing checkpoints
- Generating static (matplotlib) and interactive (plotly) visualizations
- Interpreting learned codes and their meanings
"""

from .checkpoint_analysis import (
    load_checkpoint_for_analysis,
    extract_bottleneck_stats,
    run_inference_with_codes,
)
from .visualize import (
    plot_crystallization_curve,
    plot_layer_temperatures,
    plot_codebook_usage,
    plot_code_state_alignment,
    plot_architecture_diagram,
)
from .style import setup_style, COLORS, FIGSIZE

# Interactive visualizations (require plotly)
try:
    from .visualize_interactive import (
        plot_crystallization_curve_interactive,
        plot_layer_temperatures_interactive,
        plot_codebook_usage_interactive,
        plot_code_state_alignment_interactive,
    )
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False

__all__ = [
    # Checkpoint analysis
    "load_checkpoint_for_analysis",
    "extract_bottleneck_stats",
    "run_inference_with_codes",
    # Static visualization
    "plot_crystallization_curve",
    "plot_layer_temperatures",
    "plot_codebook_usage",
    "plot_code_state_alignment",
    "plot_architecture_diagram",
    # Interactive visualization
    "plot_crystallization_curve_interactive",
    "plot_layer_temperatures_interactive",
    "plot_codebook_usage_interactive",
    "plot_code_state_alignment_interactive",
    "INTERACTIVE_AVAILABLE",
    # Style
    "setup_style",
    "COLORS",
    "FIGSIZE",
]
