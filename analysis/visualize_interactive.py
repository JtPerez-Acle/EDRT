"""
Interactive Plotly visualizations for Crystalline analysis.

Provides interactive plots for use in Jupyter notebooks with:
- Hover information
- Zoom and pan
- Toggle traces
"""

from typing import Optional, Union
from pathlib import Path
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Install with: pip install plotly")

# Color palette matching static visualizations
COLORS = {
    "loss_total": "#1f77b4",
    "loss_pred": "#ff7f0e",
    "loss_compress": "#2ca02c",
    "loss_commit": "#d62728",
    "temperature": "#9467bd",
    "entropy": "#8c564b",
    "codebook_usage": "#e377c2",
    "attn": "#17becf",
    "mlp": "#bcbd22",
}


def check_plotly():
    """Check if Plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive visualizations. "
            "Install with: pip install plotly"
        )


def plot_crystallization_curve_interactive(
    steps: list,
    losses: dict,
    temperatures: list,
    entropies: list,
    codebook_usage: Optional[list] = None,
    title: str = "Crystallization During Training",
) -> "go.Figure":
    """
    Interactive crystallization curve with hover information.

    Args:
        steps: Training step numbers
        losses: Dict with 'total', 'pred', 'compress', 'commit' lists
        temperatures: Mean temperature at each step
        entropies: Mean entropy at each step
        codebook_usage: Optional codebook usage fraction at each step
        title: Figure title

    Returns:
        Plotly Figure
    """
    check_plotly()

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Loss Components", "Temperature", "Entropy", "Codebook Usage"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Panel 1: Loss curves
    for loss_name, loss_values in losses.items():
        color_key = f"loss_{loss_name}" if loss_name != "total" else "loss_total"
        fig.add_trace(
            go.Scatter(
                x=steps, y=loss_values,
                name=loss_name.capitalize(),
                mode="lines",
                line=dict(color=COLORS.get(color_key, "#333")),
                hovertemplate=f"{loss_name.capitalize()}: %{{y:.4f}}<br>Step: %{{x}}<extra></extra>",
            ),
            row=1, col=1,
        )
    fig.update_yaxes(type="log", row=1, col=1)

    # Panel 2: Temperature
    fig.add_trace(
        go.Scatter(
            x=steps, y=temperatures,
            name="Temperature",
            mode="lines",
            line=dict(color=COLORS["temperature"], width=2),
            fill="tozeroy",
            fillcolor=f"rgba(148, 103, 189, 0.2)",
            hovertemplate="Temperature: %{y:.3f}<br>Step: %{x}<extra></extra>",
        ),
        row=1, col=2,
    )
    # Add τ=1.0 reference line
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)

    # Panel 3: Entropy
    fig.add_trace(
        go.Scatter(
            x=steps, y=entropies,
            name="Entropy",
            mode="lines",
            line=dict(color=COLORS["entropy"], width=2),
            fill="tozeroy",
            fillcolor=f"rgba(140, 86, 75, 0.2)",
            hovertemplate="Entropy: %{y:.3f} nats<br>Step: %{x}<extra></extra>",
        ),
        row=2, col=1,
    )

    # Panel 4: Codebook usage
    if codebook_usage:
        fig.add_trace(
            go.Scatter(
                x=steps, y=codebook_usage,
                name="Codebook Usage",
                mode="lines",
                line=dict(color=COLORS["codebook_usage"], width=2),
                fill="tozeroy",
                fillcolor=f"rgba(227, 119, 194, 0.2)",
                hovertemplate="Usage: %{y:.1%}<br>Step: %{x}<extra></extra>",
            ),
            row=2, col=2,
        )
        fig.update_yaxes(range=[0, 1.05], row=2, col=2)
    else:
        fig.add_annotation(
            text="Codebook usage not recorded",
            xref="x4", yref="y4",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
            row=2, col=2,
        )

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=600,
        template="plotly_white",
    )

    # Update axis labels
    fig.update_xaxes(title_text="Training Step", row=2, col=1)
    fig.update_xaxes(title_text="Training Step", row=2, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Temperature", row=1, col=2)
    fig.update_yaxes(title_text="Entropy (nats)", row=2, col=1)
    fig.update_yaxes(title_text="Fraction Used", row=2, col=2)

    return fig


def plot_layer_temperatures_interactive(
    temperatures: dict,
    title: str = "Temperature per Layer",
) -> "go.Figure":
    """
    Interactive bar chart of per-layer temperatures.

    Args:
        temperatures: Dict with 'attn' and 'mlp' lists of temperatures per layer
        title: Figure title

    Returns:
        Plotly Figure
    """
    check_plotly()

    n_layers = len(temperatures["attn"])
    layers = [f"L{i}" for i in range(n_layers)]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Attention",
            x=layers,
            y=temperatures["attn"],
            marker_color=COLORS["attn"],
            text=[f"{t:.3f}" for t in temperatures["attn"]],
            textposition="outside",
            hovertemplate="Layer %{x}<br>Attention: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="MLP",
            x=layers,
            y=temperatures["mlp"],
            marker_color=COLORS["mlp"],
            text=[f"{t:.3f}" for t in temperatures["mlp"]],
            textposition="outside",
            hovertemplate="Layer %{x}<br>MLP: %{y:.4f}<extra></extra>",
        )
    )

    # Add τ=1.0 reference line
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="τ=1.0", annotation_position="right")

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        barmode="group",
        xaxis_title="Layer",
        yaxis_title="Temperature",
        template="plotly_white",
        height=400,
    )

    return fig


def plot_codebook_usage_interactive(
    code_counts: Union[list, np.ndarray],
    codebook_size: int,
    layer: int = 0,
    bn_type: str = "attn",
    title: Optional[str] = None,
) -> "go.Figure":
    """
    Interactive codebook usage distribution.

    Args:
        code_counts: Activation counts per code
        codebook_size: Total codebook size
        layer: Layer index for title
        bn_type: 'attn' or 'mlp' for title
        title: Custom title (overrides auto-generated)

    Returns:
        Plotly Figure
    """
    check_plotly()

    counts = np.array(code_counts)
    if counts.sum() > 0:
        counts = counts / counts.sum()

    # Sort by frequency
    sorted_idx = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_idx]

    # Statistics
    used_codes = (counts > 0).sum()
    usage_frac = used_codes / codebook_size

    if title is None:
        title = f"Codebook Usage - Layer {layer} ({bn_type.upper()})"

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(range(len(sorted_counts))),
            y=sorted_counts,
            marker=dict(
                color=sorted_counts,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Frequency"),
            ),
            hovertemplate="Code %{customdata}<br>Frequency: %{y:.4f}<extra></extra>",
            customdata=sorted_idx,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Code Index (sorted by frequency)",
        yaxis_title="Activation Frequency",
        template="plotly_white",
        height=400,
        annotations=[
            dict(
                text=f"Used: {used_codes}/{codebook_size} ({usage_frac:.1%})",
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                showarrow=False,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
            )
        ],
    )

    return fig


def plot_code_state_alignment_interactive(
    alignment_matrix: np.ndarray,
    n_states: int,
    codebook_size: int,
    purity: Optional[float] = None,
    title: str = "Code-State Alignment",
) -> "go.Figure":
    """
    Interactive heatmap of code-state alignment.

    Args:
        alignment_matrix: Matrix of shape (codebook_size, n_states)
        n_states: Number of FSM states
        codebook_size: Size of codebook
        purity: Optional purity score to display
        title: Figure title

    Returns:
        Plotly Figure
    """
    check_plotly()

    # Normalize rows
    row_sums = alignment_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized = alignment_matrix / row_sums

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=normalized,
            x=[f"S{i}" for i in range(n_states)],
            y=[f"C{i}" for i in range(codebook_size)],
            colorscale="YlOrRd",
            colorbar=dict(title="P(state|code)"),
            hovertemplate="Code %{y}<br>State %{x}<br>P: %{z:.3f}<extra></extra>",
        )
    )

    annotations = []
    if purity is not None:
        annotations.append(
            dict(
                text=f"Purity: {purity:.3f}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=12, color="black"),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="FSM State",
        yaxis_title="Code Index",
        template="plotly_white",
        height=500,
        annotations=annotations,
    )

    return fig


def plot_training_comparison_interactive(
    results: list[dict],
    labels: list[str],
    metric: str = "entropy",
    title: Optional[str] = None,
) -> "go.Figure":
    """
    Compare training curves across multiple experiments.

    Args:
        results: List of result dicts with 'steps' and metric values
        labels: Labels for each experiment
        metric: Which metric to compare ('entropy', 'loss', 'temperature')
        title: Figure title

    Returns:
        Plotly Figure
    """
    check_plotly()

    if title is None:
        title = f"{metric.capitalize()} Comparison"

    fig = go.Figure()

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["highlight"]]

    for i, (result, label) in enumerate(zip(results, labels)):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=result["steps"],
                y=result[metric],
                name=label,
                mode="lines",
                line=dict(color=color, width=2),
                hovertemplate=f"{label}<br>{metric}: %{{y:.4f}}<br>Step: %{{x}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Training Step",
        yaxis_title=metric.capitalize(),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=400,
    )

    return fig


if __name__ == "__main__":
    # Demo interactive visualizations
    import numpy as np

    print("Generating demo interactive visualizations...")

    # Check plotly availability
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping demo.")
        exit(1)

    # Demo data
    steps = list(range(0, 5001, 100))
    losses = {
        "total": [10 * np.exp(-s/2000) + 0.5 for s in steps],
        "pred": [8 * np.exp(-s/1500) + 0.3 for s in steps],
        "compress": [1.5 * np.exp(-s/3000) + 0.1 for s in steps],
        "commit": [0.5 * np.exp(-s/2500) + 0.1 for s in steps],
    }
    temperatures = [2.0 - 1.5 * s/5000 for s in steps]
    entropies = [3.0 * np.exp(-s/2000) + 0.5 for s in steps]
    codebook_usage = [0.3 + 0.5 * (1 - np.exp(-s/1000)) for s in steps]

    # Create figures
    fig1 = plot_crystallization_curve_interactive(
        steps, losses, temperatures, entropies, codebook_usage
    )
    fig1.write_html("docs/figures/crystallization_interactive.html")
    print("Saved: docs/figures/crystallization_interactive.html")

    # Layer temperatures
    temperatures_dict = {
        "attn": [0.8, 0.6, 0.5, 0.4],
        "mlp": [0.9, 0.7, 0.6, 0.5],
    }
    fig2 = plot_layer_temperatures_interactive(temperatures_dict)
    fig2.write_html("docs/figures/layer_temps_interactive.html")
    print("Saved: docs/figures/layer_temps_interactive.html")

    print("\nDemo complete! Open the HTML files in a browser to view.")
