"""
Static matplotlib visualizations for Crystalline analysis.

Provides publication-quality plots for:
- Crystallization curves (loss, temperature, entropy over training)
- Layer-wise temperature analysis
- Codebook usage distributions
- Code-state alignment (FSM experiments)
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from .style import setup_style, COLORS, FIGSIZE, get_layer_colors, format_metric_name


def plot_crystallization_curve(
    steps: list,
    losses: dict,
    temperatures: list,
    entropies: list,
    codebook_usage: Optional[list] = None,
    title: str = "Crystallization During Training",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot the crystallization curve showing training dynamics.

    This is the main result figure showing how crystallization emerges.

    Args:
        steps: Training step numbers
        losses: Dict with 'total', 'pred', 'compress', 'commit' lists
        temperatures: Mean temperature at each step
        entropies: Mean entropy at each step
        codebook_usage: Optional codebook usage fraction at each step
        title: Figure title
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style("paper")

    n_panels = 4 if codebook_usage else 3
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["quad"])
    axes = axes.flatten()

    # Panel 1: Loss curves
    ax = axes[0]
    ax.plot(steps, losses.get("total", []), label="Total", color=COLORS["loss_total"], linewidth=2)
    ax.plot(steps, losses.get("pred", []), label="Prediction", color=COLORS["loss_pred"], alpha=0.8)
    ax.plot(steps, losses.get("compress", []), label="Compression", color=COLORS["loss_compress"], alpha=0.8)
    ax.plot(steps, losses.get("commit", []), label="Commitment", color=COLORS["loss_commit"], alpha=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")

    # Panel 2: Temperature
    ax = axes[1]
    ax.plot(steps, temperatures, color=COLORS["temperature"], linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Temperature")
    ax.set_title("Temperature Annealing")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="τ=1.0")
    ax.fill_between(steps, temperatures, alpha=0.2, color=COLORS["temperature"])

    # Panel 3: Entropy
    ax = axes[2]
    ax.plot(steps, entropies, color=COLORS["entropy"], linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Code Selection Entropy")
    ax.fill_between(steps, entropies, alpha=0.2, color=COLORS["entropy"])

    # Annotation for crystallization
    if len(entropies) > 0:
        final_entropy = entropies[-1]
        ax.axhline(y=final_entropy, color=COLORS["entropy"], linestyle="--", alpha=0.5)
        ax.annotate(
            f"Final: {final_entropy:.2f}",
            xy=(steps[-1], final_entropy),
            xytext=(steps[-1] * 0.7, final_entropy * 1.2),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    # Panel 4: Codebook usage (or hide)
    ax = axes[3]
    if codebook_usage:
        ax.plot(steps, codebook_usage, color=COLORS["codebook_usage"], linewidth=2)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Fraction Used")
        ax.set_title("Codebook Utilization")
        ax.set_ylim(0, 1.05)
        ax.fill_between(steps, codebook_usage, alpha=0.2, color=COLORS["codebook_usage"])
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Codebook usage\nnot recorded",
                ha="center", va="center", fontsize=12, color="gray",
                transform=ax.transAxes)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_layer_temperatures(
    temperatures: dict,
    title: str = "Temperature per Layer",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot temperatures for each layer's attention and MLP bottlenecks.

    Args:
        temperatures: Dict with 'attn' and 'mlp' lists of temperatures per layer
        title: Figure title
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style("paper")

    n_layers = len(temperatures["attn"])
    fig, ax = plt.subplots(figsize=FIGSIZE["wide"])

    x = np.arange(n_layers)
    width = 0.35

    bars_attn = ax.bar(x - width/2, temperatures["attn"], width,
                       label="Attention", color=COLORS["attn"])
    bars_mlp = ax.bar(x + width/2, temperatures["mlp"], width,
                      label="MLP", color=COLORS["mlp"])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Temperature")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax.legend()

    # Add value labels on bars
    for bars in [bars_attn, bars_mlp]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha="center", va="bottom", fontsize=8)

    # Add horizontal line at τ=1.0
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_codebook_usage(
    code_counts: Union[list, np.ndarray],
    codebook_size: int,
    layer: int = 0,
    bn_type: str = "attn",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot codebook usage distribution showing which codes are used.

    Args:
        code_counts: Activation counts per code
        codebook_size: Total codebook size
        layer: Layer index for title
        bn_type: 'attn' or 'mlp' for title
        title: Custom title (overrides auto-generated)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style("paper")

    counts = np.array(code_counts)
    if counts.sum() > 0:
        counts = counts / counts.sum()  # Normalize to probabilities

    # Sort by frequency
    sorted_idx = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_idx]

    fig, ax = plt.subplots(figsize=FIGSIZE["wide"])

    # Color by usage level
    colors = plt.cm.viridis(sorted_counts / (sorted_counts.max() + 1e-8))

    ax.bar(range(len(sorted_counts)), sorted_counts, color=colors)
    ax.set_xlabel("Code Index (sorted by frequency)")
    ax.set_ylabel("Activation Frequency")

    if title is None:
        title = f"Codebook Usage - Layer {layer} ({bn_type.upper()})"
    ax.set_title(title)

    # Add statistics
    used_codes = (counts > 0).sum()
    usage_frac = used_codes / codebook_size
    ax.text(0.95, 0.95, f"Used: {used_codes}/{codebook_size} ({usage_frac:.1%})",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_code_state_alignment(
    alignment_matrix: np.ndarray,
    n_states: int,
    codebook_size: int,
    purity: Optional[float] = None,
    title: str = "Code-State Alignment",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot heatmap showing alignment between codes and FSM states.

    Args:
        alignment_matrix: Matrix of shape (codebook_size, n_states) with co-occurrence counts
        n_states: Number of FSM states
        codebook_size: Size of codebook
        purity: Optional purity score to display
        title: Figure title
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style("paper")

    fig, ax = plt.subplots(figsize=FIGSIZE["single"])

    # Normalize rows (codes) to show state distribution per code
    row_sums = alignment_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized = alignment_matrix / row_sums

    im = ax.imshow(normalized, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("FSM State")
    ax.set_ylabel("Code Index")
    ax.set_title(title)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(state | code)")

    # Add purity annotation
    if purity is not None:
        ax.text(0.02, 0.98, f"Purity: {purity:.3f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_entropy_comparison(
    entropies_with_annealing: list,
    entropies_without_annealing: list,
    steps: list,
    title: str = "Effect of Temperature Annealing",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Compare entropy curves with and without temperature annealing.

    Args:
        entropies_with_annealing: Entropy values with annealing
        entropies_without_annealing: Entropy values without annealing
        steps: Training steps
        title: Figure title
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style("paper")

    fig, ax = plt.subplots(figsize=FIGSIZE["single"])

    ax.plot(steps, entropies_with_annealing, label="With Annealing",
            color=COLORS["primary"], linewidth=2)
    ax.plot(steps, entropies_without_annealing, label="Without Annealing",
            color=COLORS["secondary"], linewidth=2, linestyle="--")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title(title)
    ax.legend()

    # Add annotation for the gap
    if len(steps) > 0:
        final_with = entropies_with_annealing[-1]
        final_without = entropies_without_annealing[-1]
        reduction = (final_without - final_with) / final_without * 100
        ax.annotate(f"{reduction:.0f}% reduction",
                   xy=(steps[-1], (final_with + final_without) / 2),
                   xytext=(steps[-1] * 0.7, (final_with + final_without) / 2),
                   fontsize=10,
                   arrowprops=dict(arrowstyle="->", color="gray"))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_architecture_diagram(
    n_layers: int = 4,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a visual architecture diagram of the Crystalline Transformer.

    Args:
        n_layers: Number of layers to show
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style("paper")

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Colors
    emb_color = "#E8E8E8"
    attn_color = COLORS["attn"]
    mlp_color = COLORS["mlp"]
    bottleneck_color = COLORS["highlight"]

    # Helper function to draw a box
    def draw_box(x, y, w, h, text, color, fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor="black", linewidth=1
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")

    # Draw input
    draw_box(3.5, 11, 3, 0.6, "Input Embeddings", emb_color)
    ax.annotate("", xy=(5, 10.9), xytext=(5, 10.4),
                arrowprops=dict(arrowstyle="->", color="black"))

    # Draw transformer blocks
    block_height = 2.2
    for i in range(min(n_layers, 3)):
        y_base = 10.2 - (i + 1) * block_height

        # Block container
        block_rect = mpatches.FancyBboxPatch(
            (1, y_base), 8, block_height - 0.3,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor="white", edgecolor="gray", linewidth=1.5, linestyle="--"
        )
        ax.add_patch(block_rect)
        ax.text(0.5, y_base + block_height/2, f"Block {i}", rotation=90,
                ha="center", va="center", fontsize=10)

        # Attention
        draw_box(1.5, y_base + 1.1, 2.5, 0.6, "Attention", attn_color)

        # Bottleneck after attention
        draw_box(4.5, y_base + 1.1, 2, 0.6, "Bottleneck", bottleneck_color, fontsize=8)

        # MLP
        draw_box(1.5, y_base + 0.3, 2.5, 0.6, "MLP", mlp_color)

        # Bottleneck after MLP
        draw_box(4.5, y_base + 0.3, 2, 0.6, "Bottleneck", bottleneck_color, fontsize=8)

        # Residual arrows
        ax.annotate("", xy=(4.3, y_base + 1.4), xytext=(6.7, y_base + 1.4),
                    arrowprops=dict(arrowstyle="->", color="black"))
        ax.annotate("", xy=(4.3, y_base + 0.6), xytext=(6.7, y_base + 0.6),
                    arrowprops=dict(arrowstyle="->", color="black"))

        # Residual text
        ax.text(7.2, y_base + 1.4, "+ residual", fontsize=7, va="center")
        ax.text(7.2, y_base + 0.6, "+ residual", fontsize=7, va="center")

        # Arrow to next block
        if i < min(n_layers, 3) - 1:
            ax.annotate("", xy=(5, y_base), xytext=(5, y_base - 0.3),
                        arrowprops=dict(arrowstyle="->", color="black"))

    # Ellipsis for more layers
    if n_layers > 3:
        ax.text(5, 3.3, "...", fontsize=16, ha="center", va="center")

    # Output
    draw_box(3.5, 0.5, 3, 0.6, "Output Logits", emb_color)
    ax.annotate("", xy=(5, 2.8), xytext=(5, 1.2),
                arrowprops=dict(arrowstyle="->", color="black"))

    # Legend
    legend_items = [
        (attn_color, "Attention"),
        (mlp_color, "MLP"),
        (bottleneck_color, "Crystalline Bottleneck"),
    ]
    for i, (color, label) in enumerate(legend_items):
        rect = mpatches.Rectangle((7.5, 10.5 - i*0.5), 0.4, 0.3, facecolor=color, edgecolor="black")
        ax.add_patch(rect)
        ax.text(8.1, 10.65 - i*0.5, label, fontsize=8, va="center")

    ax.set_title("Crystalline Transformer Architecture", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


if __name__ == "__main__":
    # Demo visualizations with synthetic data
    import numpy as np

    print("Generating demo visualizations...")

    # Create output directory
    output_dir = Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Demo crystallization curve
    steps = list(range(0, 5001, 100))
    n_steps = len(steps)
    losses = {
        "total": [10 * np.exp(-s/2000) + 0.5 for s in steps],
        "pred": [8 * np.exp(-s/1500) + 0.3 for s in steps],
        "compress": [1.5 * np.exp(-s/3000) + 0.1 for s in steps],
        "commit": [0.5 * np.exp(-s/2500) + 0.1 for s in steps],
    }
    temperatures = [2.0 - 1.5 * s/5000 for s in steps]
    entropies = [3.0 * np.exp(-s/2000) + 0.5 for s in steps]
    codebook_usage = [0.3 + 0.5 * (1 - np.exp(-s/1000)) for s in steps]

    fig = plot_crystallization_curve(
        steps, losses, temperatures, entropies, codebook_usage,
        save_path=output_dir / "crystallization_curve_demo.png"
    )
    plt.close(fig)

    # Demo layer temperatures
    temperatures_dict = {
        "attn": [0.8, 0.6, 0.5, 0.4],
        "mlp": [0.9, 0.7, 0.6, 0.5],
    }
    fig = plot_layer_temperatures(
        temperatures_dict,
        save_path=output_dir / "layer_temperatures_demo.png"
    )
    plt.close(fig)

    # Demo codebook usage
    np.random.seed(42)
    code_counts = np.random.exponential(scale=1.0, size=64)
    code_counts[:10] *= 5  # Some codes used more
    fig = plot_codebook_usage(
        code_counts, codebook_size=64,
        save_path=output_dir / "codebook_usage_demo.png"
    )
    plt.close(fig)

    # Demo architecture diagram
    fig = plot_architecture_diagram(
        n_layers=4,
        save_path=output_dir / "architecture_diagram.png"
    )
    plt.close(fig)

    print(f"\nDemo figures saved to {output_dir}/")
