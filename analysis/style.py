"""
Plotting style configuration for Crystalline visualizations.

Provides consistent colors, fonts, and figure sizes for publication-quality plots.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Color palette - colorblind-friendly
COLORS = {
    # Loss components
    "loss_total": "#1f77b4",      # Blue
    "loss_pred": "#ff7f0e",       # Orange
    "loss_compress": "#2ca02c",   # Green
    "loss_commit": "#d62728",     # Red

    # Crystallization metrics
    "temperature": "#9467bd",      # Purple
    "entropy": "#8c564b",          # Brown
    "codebook_usage": "#e377c2",   # Pink

    # Layer types
    "attn": "#17becf",             # Cyan
    "mlp": "#bcbd22",              # Olive

    # General
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "highlight": "#d62728",
}

# Figure sizes (width, height) in inches
FIGSIZE = {
    "single": (6, 4),        # Single plot
    "wide": (10, 4),         # Wide single plot
    "double": (10, 4),       # Two plots side by side
    "quad": (10, 8),         # 2x2 grid
    "tall": (6, 8),          # Tall single plot
    "presentation": (12, 6), # For slides
}

# Font sizes
FONTSIZE = {
    "title": 14,
    "label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
}


def setup_style(context: str = "paper"):
    """
    Apply consistent matplotlib style for Crystalline visualizations.

    Args:
        context: One of 'paper', 'notebook', 'presentation'
                - paper: Publication-quality, smaller fonts
                - notebook: Medium size for Jupyter
                - presentation: Larger fonts for slides
    """
    # Base style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Context-specific scaling
    scale = {"paper": 1.0, "notebook": 1.2, "presentation": 1.5}.get(context, 1.0)

    # Update rcParams
    plt.rcParams.update({
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,

        # Fonts
        "font.size": FONTSIZE["tick"] * scale,
        "axes.titlesize": FONTSIZE["title"] * scale,
        "axes.labelsize": FONTSIZE["label"] * scale,
        "legend.fontsize": FONTSIZE["legend"] * scale,
        "xtick.labelsize": FONTSIZE["tick"] * scale,
        "ytick.labelsize": FONTSIZE["tick"] * scale,

        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 6,

        # Axes
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",

        # Colors
        "axes.prop_cycle": mpl.cycler(color=[
            COLORS["primary"],
            COLORS["secondary"],
            COLORS["accent"],
            COLORS["highlight"],
            COLORS["temperature"],
            COLORS["entropy"],
        ]),
    })


def get_layer_colors(n_layers: int, bottleneck_type: str = "both") -> list:
    """
    Generate colors for layer-wise plots.

    Args:
        n_layers: Number of transformer layers
        bottleneck_type: 'attn', 'mlp', or 'both'

    Returns:
        List of colors for each layer/bottleneck
    """
    if bottleneck_type == "both":
        # Alternating attn/mlp colors with varying lightness
        colors = []
        for i in range(n_layers):
            # Lighter to darker as layers increase
            alpha = 0.4 + 0.6 * (i / max(n_layers - 1, 1))
            colors.append((*mpl.colors.to_rgb(COLORS["attn"]), alpha))
            colors.append((*mpl.colors.to_rgb(COLORS["mlp"]), alpha))
        return colors
    else:
        base_color = COLORS[bottleneck_type]
        cmap = mpl.cm.get_cmap("Blues" if bottleneck_type == "attn" else "Greens")
        return [cmap(0.3 + 0.7 * i / max(n_layers - 1, 1)) for i in range(n_layers)]


def format_metric_name(metric: str) -> str:
    """Convert internal metric names to display names."""
    names = {
        "loss_total": "Total Loss",
        "loss_pred": "Prediction Loss",
        "loss_compress": "Compression Loss",
        "loss_commit": "Commitment Loss",
        "temperature": "Temperature",
        "entropy": "Entropy",
        "codebook_usage": "Codebook Usage",
        "perplexity": "Perplexity",
        "purity": "Code-State Purity",
    }
    return names.get(metric, metric.replace("_", " ").title())


if __name__ == "__main__":
    # Demo the style
    import numpy as np

    setup_style("paper")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["double"])

    # Demo line plot
    x = np.linspace(0, 10, 100)
    axes[0].plot(x, np.exp(-x/3) * np.sin(x), label="Temperature", color=COLORS["temperature"])
    axes[0].plot(x, np.exp(-x/2), label="Entropy", color=COLORS["entropy"])
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Crystallization Metrics")
    axes[0].legend()

    # Demo bar plot with layer colors
    n_layers = 4
    colors = get_layer_colors(n_layers)
    bars = np.random.rand(n_layers * 2) * 0.5 + 0.5
    labels = [f"L{i//2} {'attn' if i%2==0 else 'mlp'}" for i in range(n_layers * 2)]
    axes[1].bar(range(len(bars)), bars, color=colors[:len(bars)])
    axes[1].set_xticks(range(len(bars)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("Temperature")
    axes[1].set_title("Layer-wise Temperature")

    plt.tight_layout()
    print("Style demo created. Close the plot to continue.")
    plt.show()
