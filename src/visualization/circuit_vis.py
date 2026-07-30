"""
src/visualization/circuit_vis.py
==================================
Research-quality visualization suite for mechanistic interpretability analyses.

Visualization Inventory
-----------------------
1.  plot_logit_lens_curve          — Line plot: logit diff by layer
2.  plot_logit_lens_heatmap        — Heatmap: logit lens per token position
3.  plot_layer_ablation_bars       — Bar chart: LD drop by layer × component
4.  plot_layer_ablation_heatmap    — 2D heatmap: layer × component
5.  plot_head_importance_heatmap   — 12×12 head importance heatmap
6.  plot_head_ranking_bar          — Horizontal bar: top-N heads ranked
7.  plot_activation_patching_heatmap — Restoration score heatmap
8.  plot_all_patching_comparison   — Side-by-side: resid/attn/MLP patching
9.  plot_circuit_graph             — Directed circuit graph (Plotly)
10. plot_attention_patterns        — Attention weight visualization

Design System
-------------
All plots follow the CircuitScope dark theme:
  - Background: near-black (#0D1117 / #161B22)
  - Primary accent: electric blue (#58A6FF)
  - Highlight: teal (#00C9A7), coral (#FF6B6B)
  - Heatmap: "RdBu_r" for signed values, "Viridis" for unsigned
  - Font: "Inter, Arial, sans-serif" throughout
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.utils.io_utils import save_figure

logger = logging.getLogger(__name__)

# ── Design constants ────────────────────────────────────────────────────────
_TEMPLATE = "plotly_dark"
_BG_PLOT  = "rgba(13, 17, 23, 0.95)"
_BG_PAPER = "rgba(10, 13, 18, 1.0)"
_FONT     = "Inter, Arial, sans-serif"
_W, _H    = 1000, 600
_BLUE     = "#58A6FF"
_TEAL     = "#00C9A7"
_CORAL    = "#FF6B6B"
_AMBER    = "#FFC75F"
_PURPLE   = "#845EC2"
_GREEN    = "#3DDC84"

# ── Colorscales ─────────────────────────────────────────────────────────────
# RdBu_r: Red (negative/bad) → White (zero) → Blue (positive/good)
# Good for signed quantities like logit diff, importance scores.
_CS_SIGNED    = "RdBu_r"
# Blues: White → Blue (for unsigned quantities like probability)
_CS_UNSIGNED  = "Blues"
# Viridis: Purple → Yellow (perceptually uniform, colorblind-safe)
_CS_VIRIDIS   = "Viridis"


def _base_layout(title: str, width: int = _W, height: int = _H, **kwargs) -> dict:
    """Return consistent base layout kwargs for all CircuitScope plots."""
    return {
        "template": _TEMPLATE,
        "title": {
            "text": title,
            "x": 0.5, "xanchor": "center",
            "font": {"size": 17, "family": _FONT, "color": "#E8E8E8"},
        },
        "font": {"family": _FONT, "size": 12, "color": "#CCCCCC"},
        "plot_bgcolor": _BG_PLOT,
        "paper_bgcolor": _BG_PAPER,
        "margin": {"l": 80, "r": 40, "t": 80, "b": 70},
        "width": width, "height": height,
        **kwargs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Logit Lens Visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_logit_lens_curve(
    lens_df: pd.DataFrame,
    title: str = "Logit Lens: When Does GPT-2 Small Identify the Indirect Object?",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    Line chart showing logit diff and IO/S probabilities at each layer.

    The x-axis is the layer index (Embed, L0, …, L11).
    The y-axis is logit_diff (IO - S). A second y-axis shows prob_io.

    The point where logit_diff first exceeds 0 indicates which layer
    first "decides" the correct answer.

    Parameters
    ----------
    lens_df : pd.DataFrame
        Output of `LogitLensAnalyzer.run()`. Must have columns:
        layer_label, logit_diff, prob_io, prob_s, fraction_correct.

    title : str
        Plot title.

    save_path : str or Path, optional
        Base path for saving (no extension). Extensions added per format.

    formats : list[str], optional
        ["html", "png"] by default.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with dual y-axes.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    x = lens_df["layer_label"].tolist()

    # ── Primary: Logit Difference ─────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x, y=lens_df["logit_diff"],
            name="Logit Diff (IO − S)",
            line={"color": _BLUE, "width": 3},
            mode="lines+markers",
            marker={"size": 8, "color": _BLUE, "line": {"color": "white", "width": 1}},
            hovertemplate="<b>%{x}</b><br>Logit Diff: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # ── Secondary: P(IO) and P(S) ─────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x, y=lens_df["prob_io"],
            name="P(IO) probability",
            line={"color": _TEAL, "width": 2, "dash": "dot"},
            mode="lines+markers",
            marker={"size": 6, "color": _TEAL},
            hovertemplate="<b>%{x}</b><br>P(IO): %{y:.4f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=lens_df["prob_s"],
            name="P(S) probability",
            line={"color": _CORAL, "width": 2, "dash": "dot"},
            mode="lines+markers",
            marker={"size": 6, "color": _CORAL},
            hovertemplate="<b>%{x}</b><br>P(S): %{y:.4f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # ── Decision boundary at LD=0 ─────────────────────────────────────────
    fig.add_hline(
        y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)",
        line_width=1.5, secondary_y=False,
    )

    # ── Fraction correct (right axis, shaded area) ────────────────────────
    if "fraction_correct" in lens_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=lens_df["fraction_correct"],
                name="Fraction Correct",
                line={"color": _AMBER, "width": 1.5, "dash": "dash"},
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(255, 199, 95, 0.08)",
                hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1%}<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        **_base_layout(title, height=_H),
        legend={
            "bgcolor": "rgba(255,255,255,0.05)",
            "bordercolor": "rgba(255,255,255,0.15)",
            "borderwidth": 1,
            "x": 0.02, "y": 0.98,
        },
    )
    fig.update_xaxes(
        title_text="Layer",
        gridcolor="rgba(255,255,255,0.07)",
        tickangle=-30,
    )
    fig.update_yaxes(
        title_text="Logit Difference (IO − S)",
        gridcolor="rgba(255,255,255,0.07)",
        secondary_y=False,
        zeroline=True, zerolinecolor="rgba(255,255,255,0.3)",
    )
    fig.update_yaxes(
        title_text="Probability / Fraction Correct",
        secondary_y=True,
        range=[0, 1],
        gridcolor="rgba(0,0,0,0)",
    )

    if save_path:
        save_figure(fig, save_path, formats=formats or ["html", "png"],
                    width=_W, height=_H)
    return fig


def plot_logit_lens_heatmap(
    pos_df: pd.DataFrame,
    prompt_str: str = "",
    title: str = "Logit Lens Heatmap: Logit Diff by (Layer, Token Position)",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    Heatmap of logit diff at every (layer, token_position) from the logit lens.

    Each cell shows logit(IO) - logit(S) at that position after that layer.
    Red = negative (S preferred), Blue = positive (IO preferred).

    Parameters
    ----------
    pos_df : pd.DataFrame
        Output of `LogitLensAnalyzer.run_per_token_position()`.
        Columns: position, token_str, logit_diff, prob_io.

    prompt_str : str
        The prompt string (for annotation in title).

    title : str
        Plot title.

    save_path, formats : optional save parameters.

    Returns
    -------
    go.Figure
    """
    token_labels = [
        f"{r['token_str']!r}" for _, r in pos_df.iterrows()
    ]
    z_vals = pos_df["logit_diff"].values.reshape(-1, 1)  # [n_tokens, 1]

    fig = go.Figure(
        go.Heatmap(
            z=z_vals.T,
            x=token_labels,
            y=["Logit Diff"],
            colorscale=_CS_SIGNED,
            zmid=0,
            colorbar={
                "title": "Logit Diff",
                "thickness": 15,
            },
            hovertemplate=(
                "Token: %{x}<br>"
                "Logit Diff: %{z:.4f}<extra></extra>"
            ),
        )
    )
    display_title = f"{title}<br><sub>Prompt: {prompt_str[:80]}…</sub>" if prompt_str else title
    fig.update_layout(
        **_base_layout(display_title, width=_W + 200, height=300),
        xaxis={"tickangle": -45, "tickfont": {"size": 11}},
    )

    if save_path:
        save_figure(fig, save_path, formats=formats or ["html", "png"],
                    width=_W + 200, height=300)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Layer Ablation Visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_layer_ablation_bars(
    ablation_df: pd.DataFrame,
    title: str = "Layer Ablation: Logit Diff Drop by Layer and Component",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    Grouped bar chart: for each layer, show LD drop from ablating attn, MLP, full.

    Taller bars = more important component. Bars that reach the full
    baseline height indicate a critical layer for the IOI task.

    Parameters
    ----------
    ablation_df : pd.DataFrame
        Output of `LayerAblationAnalyzer.run_full_sweep()`.
        Columns: layer, component, ld_drop_norm, is_critical.

    Returns
    -------
    go.Figure
    """
    component_colors = {
        "attn": _BLUE,
        "mlp": _CORAL,
        "full_layer": _AMBER,
    }
    component_names = {
        "attn": "Attention Only",
        "mlp": "MLP Only",
        "full_layer": "Full Layer (Attn + MLP)",
    }

    fig = go.Figure()

    for component, color in component_colors.items():
        subset = ablation_df[ablation_df["component"] == component]
        if subset.empty:
            continue

        fig.add_trace(
            go.Bar(
                name=component_names.get(component, component),
                x=[f"L{l}" for l in subset["layer"]],
                y=subset["ld_drop_norm"],
                marker_color=color,
                marker_opacity=0.85,
                marker_line={"color": "rgba(255,255,255,0.2)", "width": 0.5},
                hovertemplate=(
                    "<b>Layer %{x} — " + component_names.get(component, component) + "</b><br>"
                    "Normalised LD drop: %{y:.3f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Mark the threshold for "critical"
    fig.add_hline(
        y=0.10, line_dash="dash",
        line_color="rgba(255,255,255,0.35)",
        annotation_text=" Critical threshold (10%)",
        annotation_position="right",
        annotation_font={"size": 11, "color": "rgba(200,200,200,0.6)"},
    )

    fig.update_layout(
        **_base_layout(title),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
    )
    fig.update_xaxes(title_text="Layer", gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(
        title_text="Normalised LD Drop (fraction of baseline)",
        gridcolor="rgba(255,255,255,0.07)",
        tickformat=".0%",
    )

    if save_path:
        save_figure(fig, save_path, formats=formats or ["html", "png"],
                    width=_W, height=_H)
    return fig


def plot_layer_ablation_heatmap(
    ablation_df: pd.DataFrame,
    metric: str = "ld_drop_norm",
    title: str = "Layer Ablation Heatmap: Component × Layer Importance",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    2D heatmap: rows = components, columns = layers.
    Each cell shows the normalised logit diff drop when that component is ablated.

    Parameters
    ----------
    ablation_df : pd.DataFrame
        Output of `LayerAblationAnalyzer.run_full_sweep()`.

    metric : str
        Which metric to display. Default: "ld_drop_norm".

    Returns
    -------
    go.Figure
    """
    # Pivot: rows=component, cols=layer
    pivot = ablation_df.pivot(index="component", columns="layer", values=metric)
    pivot.columns = [f"L{c}" for c in pivot.columns]

    component_order = ["attn", "mlp", "full_layer"]
    component_labels = {
        "attn": "Attention", "mlp": "MLP", "full_layer": "Full Layer"
    }
    pivot = pivot.reindex([c for c in component_order if c in pivot.index])
    pivot.index = [component_labels.get(c, c) for c in pivot.index]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=_CS_SIGNED,
            zmid=0,
            text=np.round(pivot.values, 3),
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar={"title": "LD Drop (norm.)", "thickness": 15},
            hovertemplate=(
                "<b>%{y} — %{x}</b><br>"
                "LD drop: %{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(**_base_layout(title, height=350))
    fig.update_xaxes(title_text="Layer", side="bottom")
    fig.update_yaxes(title_text="Component")

    if save_path:
        save_figure(fig, save_path, formats=formats or ["html", "png"],
                    width=_W, height=350)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Head Ablation Visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_head_importance_heatmap(
    importance_matrix: pd.DataFrame,
    title: str = "Attention Head Importance: IOI Logit Diff Drop (144 Heads)",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    12×12 heatmap of per-head importance scores.

    Rows = layers (0–11), columns = heads (0–11).
    Color: blue = positive importance (head helps IOI), red = suppressor.

    The 12×12 grid directly corresponds to the GPT-2 Small architecture.
    High-importance cells (dark blue) are the IOI circuit heads.

    Parameters
    ----------
    importance_matrix : pd.DataFrame
        12×12 DataFrame from `HeadAblationAnalyzer.pivot_importance_matrix()`.
        Index = layer (0–11), columns = head (0–11).

    Returns
    -------
    go.Figure
    """
    z = importance_matrix.values  # [12, 12]
    x_labels = [f"H{h}" for h in range(importance_matrix.shape[1])]
    y_labels = [f"L{l}" for l in range(importance_matrix.shape[0])]

    # Text annotations: show score inside each cell
    text_vals = [[f"{v:+.3f}" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=_CS_SIGNED,
            zmid=0,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 9, "color": "rgba(255,255,255,0.85)"},
            colorbar={
                "title": "Importance<br>(LD drop norm.)",
                "thickness": 18,
            },
            hovertemplate=(
                "<b>Layer %{y}, Head %{x}</b><br>"
                "Importance: %{z:+.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **_base_layout(title, width=_W + 100, height=_H + 50),
        xaxis={"title": "Attention Head", "side": "bottom"},
        yaxis={"title": "Layer", "autorange": "reversed"},
    )

    if save_path:
        save_figure(fig, save_path, formats=formats or ["html", "png"],
                    width=_W + 100, height=_H + 50)
    return fig


def plot_head_ranking_bar(
    results_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top Attention Heads by IOI Causal Importance",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    Horizontal bar chart ranking the top-N attention heads by importance.

    Color-coded by head type: Name Mover (teal), Helper (blue),
    Suppressor (coral), Neutral (grey).

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of `HeadAblationAnalyzer.run_full_sweep()`.
        Must have: head_label, importance, head_type.

    top_n : int
        Number of heads to display (sorted by |importance|).

    Returns
    -------
    go.Figure
    """
    # Show top_n by absolute importance (includes both positive and negative)
    top = results_df.reindex(results_df["importance"].abs().nlargest(top_n).index)
    top = top.sort_values("importance", ascending=True)  # ascending for horizontal bar

    type_colors = {
        "Name Mover": _TEAL,
        "Helper": _BLUE,
        "Neutral": "rgba(150,150,150,0.7)",
        "Suppressor": _CORAL,
        "Strong Suppressor": "#FF0033",
    }

    colors = [type_colors.get(t, _BLUE) for t in top["head_type"]]

    fig = go.Figure(
        go.Bar(
            y=top["head_label"],
            x=top["importance"],
            orientation="h",
            marker_color=colors,
            marker_line={"color": "rgba(255,255,255,0.15)", "width": 0.5},
            text=[f"{v:+.4f}" for v in top["importance"]],
            textposition="outside",
            textfont={"size": 11},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Importance: %{x:+.4f}<br>"
                "Type: %{customdata}<extra></extra>"
            ),
            customdata=top["head_type"],
        )
    )

    fig.add_vline(
        x=0, line_color="rgba(255,255,255,0.3)", line_width=1.5
    )

    # Add legend (manual since colors don't auto-generate legend for Bar)
    for head_type, color in type_colors.items():
        if head_type in top["head_type"].values:
            fig.add_trace(go.Bar(
                x=[None], y=[None],
                marker_color=color,
                name=head_type,
                showlegend=True,
            ))

    fig.update_layout(
        **_base_layout(title, height=max(400, top_n * 30)),
        xaxis_title="Normalised Importance (LD drop / baseline)",
        yaxis_title="Attention Head",
        barmode="overlay",
    )

    if save_path:
        save_figure(fig, save_path, formats=formats or ["html", "png"],
                    width=_W, height=max(500, top_n * 30))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Activation Patching Visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_activation_patching_heatmap(
    patching_df: pd.DataFrame,
    title: str = "Activation Patching: Restoration Score (Layer × Token Position)",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
    clim: tuple[float, float] = (-0.3, 1.0),
) -> go.Figure:
    """
    Heatmap of restoration scores at every (token_position, layer) pair.

    Rows = token positions, columns = layers.
    Color: dark blue = high restoration (critical location),
           white = no effect, red = negative restoration.

    A cell with restoration ≈ 1.0 means that patching the activation at
    that (layer, position) fully restores the clean IOI behaviour.

    Parameters
    ----------
    patching_df : pd.DataFrame
        Output of `ActivationPatchingAnalyzer.run_layer_position_sweep()`.
        Index = token position labels, columns = layer labels ("L0", …, "L11").

    title : str
        Plot title.

    save_path, formats : optional save.

    clim : tuple of (min, max)
        Color axis limits. Default: (-0.3, 1.0).

    Returns
    -------
    go.Figure
    """
    import numpy as np
    z = patching_df.values  # [n_positions, n_layers]
    y_labels = patching_df.index.tolist()
    x_labels = patching_df.columns.tolist()

    # Truncate long position labels for readability
    y_labels_short = [
        label[:20] + "…" if len(label) > 20 else label
        for label in y_labels
    ]
    text_vals = [[f"{v:+.2f}" if not np.isnan(v) else "" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels_short,
            colorscale="RdBu",
            zmin=clim[0], zmax=clim[1],
            zmid=0,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 8},
            colorbar={
                "title": "Restoration<br>Score",
                "thickness": 15,
                "tickformat": ".2f",
            },
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Layer: %{x}<br>"
                "Restoration: %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **_base_layout(title, width=_W, height=max(_H, len(y_labels) * 22 + 120)),
        xaxis={"title": "Layer", "side": "bottom"},
        yaxis={"title": "Token Position", "autorange": "reversed"},
    )

    if save_path:
        h = max(_H, len(y_labels) * 22 + 120)
        save_figure(fig, save_path, formats=formats or ["html", "png"],
                    width=_W, height=h)
    return fig


def plot_all_patching_comparison(
    resid_df: pd.DataFrame,
    attn_df: pd.DataFrame,
    mlp_df: pd.DataFrame,
    title: str = "Activation Patching Comparison: Resid vs Attn vs MLP",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    Side-by-side heatmaps comparing restoration scores for the three patch types.

    Columns: Residual Stream | Attention Output | MLP Output
    This comparison reveals whether the IOI circuit is primarily attention-
    mediated (attn >> mlp), MLP-mediated (mlp >> attn), or residual-stream-
    mediated (resid > both).

    Parameters
    ----------
    resid_df, attn_df, mlp_df : pd.DataFrame
        Patching results from the three experiment types.
        All must have the same index (token positions) and columns (layers).

    Returns
    -------
    go.Figure
    """
    titles_sub = ["Residual Stream", "Attention Output", "MLP Output"]
    dfs = [resid_df, attn_df, mlp_df]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=titles_sub,
        horizontal_spacing=0.08,
    )

    for col_idx, (df, sub_title) in enumerate(zip(dfs, titles_sub), start=1):
        y_labels = df.index.tolist()
        y_labels_short = [l[:15] + "…" if len(l) > 15 else l for l in y_labels]

        fig.add_trace(
            go.Heatmap(
                z=df.values,
                x=df.columns.tolist(),
                y=y_labels_short,
                colorscale="RdBu",
                zmin=-0.3, zmax=1.0,
                zmid=0,
                showscale=(col_idx == 3),
                colorbar={
                    "x": 1.02, "thickness": 12,
                    "title": "Score",
                },
                hovertemplate=(
                    f"<b>{sub_title}</b><br>"
                    "%{y}<br>Layer: %{x}<br>"
                    "Restoration: %{z:.4f}<extra></extra>"
                ),
            ),
            row=1, col=col_idx,
        )

    n_rows = len(resid_df)
    fig.update_layout(
        **_base_layout(title, width=_W * 2, height=max(_H, n_rows * 18 + 150)),
        showlegend=False,
    )

    for col_idx in range(1, 4):
        fig.update_yaxes(autorange="reversed", row=1, col=col_idx)
        if col_idx == 1:
            fig.update_yaxes(title_text="Token Position", row=1, col=col_idx)
        fig.update_xaxes(title_text="Layer", row=1, col=col_idx)

    if save_path:
        h = max(_H, n_rows * 18 + 150)
        save_figure(fig, save_path, formats=formats or ["html", "png"],
                    width=_W * 2, height=h)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Circuit Graph Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_circuit_graph(
    graph_df: pd.DataFrame,
    sender_results: pd.DataFrame,
    title: str = "IOI Circuit Graph: Information Flow Between Attention Heads",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    Directed graph of the IOI circuit showing information flow between heads.

    Nodes = attention heads (positioned by layer × head in a grid).
    Edges = estimated information flow (thickness = edge weight).
    Node color = sender importance score.
    Node size = |importance|.

    The layered layout places earlier layers on the left and later layers
    on the right, making the left-to-right flow of information intuitive.

    Parameters
    ----------
    graph_df : pd.DataFrame
        Edge list from `PathPatchingAnalyzer.build_circuit_graph()`.
        Columns: source_label, target_label, estimated_edge_weight,
                 source_layer, source_head, target_layer, target_head.

    sender_results : pd.DataFrame
        Output of `PathPatchingAnalyzer.run_sender_patching()`.
        Used to determine node sizes and colors.

    Returns
    -------
    go.Figure
    """
    if graph_df.empty:
        logger.warning("[plot_circuit_graph] Empty graph_df; returning empty figure.")
        return go.Figure()

    # ── Node positions: x = layer, y = head ──────────────────────────────
    # Collect all unique nodes from the graph
    all_nodes = set(graph_df["source_label"].tolist() + graph_df["target_label"].tolist())

    # Build node metadata from sender_results
    node_info: dict[str, dict] = {}
    for _, row in sender_results.iterrows():
        if row["head_label"] in all_nodes:
            node_info[row["head_label"]] = {
                "layer": row["layer"],
                "head": row["head"],
                "importance": row["restoration_score"],
            }

    if not node_info:
        logger.warning("[plot_circuit_graph] No node info found.")
        return go.Figure()

    # ── Edge traces ───────────────────────────────────────────────────────
    edge_traces = []
    max_weight = graph_df["estimated_edge_weight"].max()

    for _, edge in graph_df.iterrows():
        src = edge["source_label"]
        tgt = edge["target_label"]

        if src not in node_info or tgt not in node_info:
            continue

        x0 = float(node_info[src]["layer"])
        y0 = float(node_info[src]["head"])
        x1 = float(node_info[tgt]["layer"])
        y1 = float(node_info[tgt]["head"])

        weight = float(edge["estimated_edge_weight"])
        abs_weight = abs(weight)
        abs_max = max(abs(float(max_weight)), 1e-8)
        opacity = min(0.9, max(0.1, 0.2 + 0.7 * abs_weight / abs_max))
        width = max(0.5, float(1.0 + 4.0 * abs_weight / abs_max))

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line={"width": width, "color": f"rgba(88, 166, 255, {opacity:.2f})"},
                hoverinfo="none",
                showlegend=False,
            )
        )

    # ── Node trace ────────────────────────────────────────────────────────
    node_x, node_y, node_text, node_color, node_size, node_hover = (
        [], [], [], [], [], []
    )

    for label, info in node_info.items():
        node_x.append(float(info["layer"]))
        node_y.append(float(info["head"]))
        node_text.append(label)
        node_color.append(info["importance"])
        node_size.append(10 + 30 * abs(info["importance"]))
        node_hover.append(
            f"<b>{label}</b><br>"
            f"Layer: {info['layer']}, Head: {info['head']}<br>"
            f"Restoration Score: {info['importance']:+.4f}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont={"size": 10, "color": "white"},
        marker={
            "size": node_size,
            "color": node_color,
            "colorscale": "RdBu",
            "cmid": 0,
            "line": {"color": "white", "width": 1},
            "showscale": True,
            "colorbar": {
                "title": "Restoration<br>Score",
                "thickness": 12,
                "x": 1.02,
            },
        },
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=node_hover,
        name="Attention Heads",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        **_base_layout(title, width=_W + 200, height=600),
        xaxis={
            "title": "Layer",
            "tickmode": "array",
            "tickvals": list(range(12)),
            "ticktext": [f"L{l}" for l in range(12)],
            "gridcolor": "rgba(255,255,255,0.08)",
            "range": [-0.5, 11.5],
        },
        yaxis={
            "title": "Head",
            "tickmode": "array",
            "tickvals": list(range(12)),
            "ticktext": [f"H{h}" for h in range(12)],
            "gridcolor": "rgba(255,255,255,0.08)",
            "range": [-0.5, 11.5],
        },
        showlegend=False,
    )

    if save_path:
        save_figure(fig, save_path, formats=formats or ["html", "png"],
                    width=_W + 200, height=600)
    return fig


def plot_sender_importance_heatmap(
    sender_results: pd.DataFrame,
    title: str = "Path Patching: Sender Head Restoration Scores (12×12 Grid)",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    12×12 heatmap of sender restoration scores from path patching.

    Similar to head ablation heatmap but with restoration scores instead
    of ablation drops. High values = head is a strong sender of IOI-relevant
    information.

    Parameters
    ----------
    sender_results : pd.DataFrame
        Output of `PathPatchingAnalyzer.run_sender_patching()`.
        Columns: layer, head, restoration_score.

    Returns
    -------
    go.Figure
    """
    pivot = sender_results.pivot(
        index="layer", columns="head", values="restoration_score"
    )
    z = pivot.values  # [12, 12]
    x_labels = [f"H{h}" for h in pivot.columns]
    y_labels = [f"L{l}" for l in pivot.index]

    text_vals = [[f"{v:+.3f}" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=_CS_SIGNED,
            zmid=0,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 9, "color": "rgba(255,255,255,0.85)"},
            colorbar={
                "title": "Restoration<br>Score",
                "thickness": 15,
            },
            hovertemplate=(
                "<b>Layer %{y}, Head %{x}</b><br>"
                "Restoration: %{z:+.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **_base_layout(title, width=_W + 100, height=_H + 50),
        yaxis={"autorange": "reversed"},
    )

    if save_path:
        save_figure(fig, save_path, formats=formats or ["html", "png"],
                    width=_W + 100, height=_H + 50)
    return fig


def plot_attention_patterns(
    model,
    prompt: str,
    layer: int,
    head: int,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    Visualize the attention pattern of a specific head on a specific prompt.

    The heatmap shows attn_pattern[head, query, key]:
    - Rows = query positions (each token attends to...)
    - Cols = key positions (...these tokens)
    - Color intensity = attention weight (0 to 1)

    For Name Mover heads, we expect the final "to" position to strongly
    attend to the IO name position.

    Parameters
    ----------
    model : HookedTransformer
        Loaded GPT-2 Small model.

    prompt : str
        The input prompt string to visualize attention for.

    layer : int
        Layer index (0–11).

    head : int
        Head index (0–11).

    title : str, optional
        Custom title. If None, auto-generated.

    save_path, formats : optional save.

    Returns
    -------
    go.Figure
    """
    import torch

    hook_name = f"blocks.{layer}.attn.hook_pattern"
    tokens = model.to_tokens(prompt, prepend_bos=True)
    str_tokens = model.to_str_tokens(prompt, prepend_bos=True)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])

    # pattern shape: [batch, n_heads, query, key]
    pattern = cache[hook_name][0, head, :, :]  # [query, key]
    pattern_np = pattern.cpu().float().numpy()

    display_tokens = [t.replace("<|endoftext|>", "BOS") for t in str_tokens]

    auto_title = (
        title or
        f"Attention Pattern — Layer {layer}, Head {head} (L{layer}H{head})"
    )

    fig = go.Figure(
        go.Heatmap(
            z=pattern_np,
            x=display_tokens,
            y=display_tokens,
            colorscale=_CS_UNSIGNED,
            zmin=0, zmax=1,
            colorbar={"title": "Attention<br>Weight", "thickness": 15},
            hovertemplate=(
                "<b>%{y} attends to %{x}</b><br>"
                "Weight: %{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **_base_layout(auto_title, width=_W, height=_W - 50),
        xaxis={
            "title": "Key (source) token",
            "tickangle": -45,
            "tickfont": {"size": 10},
        },
        yaxis={
            "title": "Query (destination) token",
            "autorange": "reversed",
            "tickfont": {"size": 10},
        },
    )

    if save_path:
        save_figure(fig, save_path, formats=formats or ["html", "png"],
                    width=_W, height=_W - 50)
    return fig
