"""
src/visualization/plots.py
============================
Baseline visualization suite for the IOI circuit analysis.

All plots are built with Plotly for interactive HTML output plus static PNG
(via kaleido). Matplotlib fallback functions are also provided for
environments where Plotly rendering is unavailable (e.g., some HPC clusters).

Plot Inventory
--------------
1. plot_accuracy_bar
   Grouped bar chart: overall accuracy + per-template (ABB/BAB) accuracy.
   Shows the model's success rate at identifying the indirect object.

2. plot_logit_diff_histogram
   Distribution of per-prompt logit differences (logit_IO - logit_S).
   The vertical line at x=0 divides correct (right) from incorrect (left).
   Shape: we expect a right-skewed distribution centered around +2 to +4
   for a well-trained model on IOI.

3. plot_confidence_histogram
   Distribution of softmax probability assigned to the IO token (prob_IO).
   High-confidence correct answers cluster near 1.0.

4. plot_top_k_distribution
   Bar chart of the most frequently predicted top-1 tokens across all prompts.
   Ideally the IO names dominate; if "the" appears, the model is confused.

5. save_all_baseline_plots
   Convenience function: generates and saves all four plots in one call.

Design System
-------------
All plots use the "plotly_dark" template for a research-quality dark theme.
Color palette: teal (correct), coral (incorrect), purple (logit diff),
amber (confidence). All plots include proper axis labels, titles, and
hover information.
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

# ── Design constants (match experiment_config.yaml) ────────────────────────
_TEMPLATE = "plotly_dark"
_WIDTH = 900
_HEIGHT = 550
_COLOR_CORRECT = "#00C9A7"      # Teal
_COLOR_INCORRECT = "#FF6B6B"    # Coral
_COLOR_LOGIT_DIFF = "#845EC2"   # Purple
_COLOR_CONFIDENCE = "#FFC75F"   # Amber
_COLOR_TOP_K = "#4D8AF0"        # Blue
_FONT_FAMILY = "Inter, Arial, sans-serif"


def _apply_base_layout(fig: go.Figure, title: str, **kwargs) -> go.Figure:
    """
    Apply consistent base layout to all CircuitScope plots.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to update.
    title : str
        Plot title (will be bold and slightly larger than axes labels).
    **kwargs
        Additional layout overrides passed to `fig.update_layout()`.

    Returns
    -------
    go.Figure
        Updated figure with base layout applied.
    """
    fig.update_layout(
        template=_TEMPLATE,
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "family": _FONT_FAMILY, "color": "#E8E8E8"},
        },
        font={"family": _FONT_FAMILY, "size": 13, "color": "#CCCCCC"},
        plot_bgcolor="rgba(20, 20, 30, 0.9)",
        paper_bgcolor="rgba(15, 15, 25, 1.0)",
        margin={"l": 70, "r": 40, "t": 80, "b": 70},
        legend={
            "bgcolor": "rgba(255,255,255,0.05)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "borderwidth": 1,
        },
        **kwargs,
    )
    return fig


def plot_accuracy_bar(
    results_df: pd.DataFrame,
    title: str = "IOI Accuracy: GPT-2 Small Baseline",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    Create a grouped bar chart showing IOI accuracy overall and by template type.

    The chart displays three bars:
      - Overall accuracy (all 1000 prompts)
      - ABB template accuracy
      - BAB template accuracy

    Each bar shows the fraction of prompts where logit(IO) > logit(S).
    Error bar annotations show the raw counts (n_correct / n_total).

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of `IOIEvaluator.evaluate()`. Must contain columns:
        "is_correct", "template_type".

    title : str
        Chart title. Can be overridden.

    save_path : str or Path, optional
        If provided, save the figure to this path (without extension).
        Multiple formats written per `formats`.

    formats : list of str, optional
        Output formats. Default: ["html", "png"].

    Returns
    -------
    go.Figure
        Interactive Plotly figure.

    Examples
    --------
    >>> fig = plot_accuracy_bar(results_df, save_path="outputs/figures/accuracy")
    >>> fig.show()
    """
    # ── Compute accuracy per group ─────────────────────────────────────────
    groups = {
        "Overall": results_df,
        "ABB": results_df[results_df["template_type"] == "ABB"],
        "BAB": results_df[results_df["template_type"] == "BAB"],
    }

    labels, accuracies, counts, totals = [], [], [], []
    for name, subset in groups.items():
        if len(subset) == 0:
            continue
        acc = subset["is_correct"].mean()
        n_correct = subset["is_correct"].sum()
        n_total = len(subset)
        labels.append(name)
        accuracies.append(acc)
        counts.append(n_correct)
        totals.append(n_total)

    # ── Colors: green for ≥0.7, yellow for 0.5–0.7, red below 0.5 ────────
    bar_colors = [
        _COLOR_CORRECT if a >= 0.7 else (_COLOR_CONFIDENCE if a >= 0.5 else _COLOR_INCORRECT)
        for a in accuracies
    ]

    # ── Build figure ──────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=[a * 100 for a in accuracies],  # Display as percentages
            marker_color=bar_colors,
            marker_line={"color": "rgba(255,255,255,0.3)", "width": 1.5},
            text=[f"{a:.1%}<br>({c:,}/{t:,})" for a, c, t in zip(accuracies, counts, totals)],
            textposition="outside",
            textfont={"size": 13, "color": "#E8E8E8"},
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Accuracy: %{y:.1f}%<br>"
                "Correct: %{customdata[0]:,} / %{customdata[1]:,}<extra></extra>"
            ),
            customdata=list(zip(counts, totals)),
            name="Accuracy",
            width=0.45,
        )
    )

    # Reference line at 50% (chance level)
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="Chance (50%)",
        annotation_position="right",
        annotation_font={"color": "rgba(200,200,200,0.7)", "size": 11},
    )

    # Reference line at 100%
    fig.add_hline(
        y=100,
        line_dash="dot",
        line_color="rgba(255,255,255,0.15)",
    )

    fig = _apply_base_layout(fig, title)
    fig.update_xaxes(title_text="Template Type", tickfont={"size": 14})
    fig.update_yaxes(
        title_text="Accuracy (%)",
        range=[0, 115],
        ticksuffix="%",
        gridcolor="rgba(255,255,255,0.08)",
    )

    # Save if path provided
    if save_path is not None:
        save_figure(fig, save_path, formats=formats, width=_WIDTH, height=_HEIGHT)

    logger.info("[plot_accuracy_bar] ✓ Accuracy bar chart created.")
    return fig


def plot_logit_diff_histogram(
    results_df: pd.DataFrame,
    title: str = "Logit Difference Distribution (IOI vs. S Token)",
    n_bins: int = 60,
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    Plot the distribution of per-prompt logit differences.

    logit_diff = logit(IO) - logit(S)

    Prompts are color-coded:
      - Green  (logit_diff > 0) : Model correctly prefers IO
      - Coral  (logit_diff < 0) : Model incorrectly prefers S
      - Dashed vertical line at 0 divides the two regions

    This plot is the most important diagnostic for the IOI baseline.
    A well-functioning model should show a right-shifted distribution,
    meaning most prompts have positive logit difference.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: "logit_diff", "template_type", "is_correct".

    title : str
        Chart title.

    n_bins : int
        Number of histogram bins. Default: 60 (fine-grained view).

    save_path : str or Path, optional
        Base save path without extension.

    formats : list of str, optional
        Output formats.

    Returns
    -------
    go.Figure
        Interactive Plotly histogram.

    Examples
    --------
    >>> fig = plot_logit_diff_histogram(results_df,
    ...         save_path="outputs/figures/logit_diff_hist")
    """
    # ── Split into correct and incorrect groups ───────────────────────────
    correct = results_df[results_df["is_correct"]]
    incorrect = results_df[~results_df["is_correct"]]

    # Compute bin range once so both histograms use the same bins
    all_diffs = results_df["logit_diff"]
    x_min, x_max = all_diffs.min() - 0.5, all_diffs.max() + 0.5

    fig = go.Figure()

    # ── Incorrect (negative logit diff) ──────────────────────────────────
    if len(incorrect) > 0:
        fig.add_trace(
            go.Histogram(
                x=incorrect["logit_diff"],
                nbinsx=n_bins,
                name=f"Incorrect (n={len(incorrect):,})",
                marker_color=_COLOR_INCORRECT,
                opacity=0.8,
                xbins={"start": x_min, "end": x_max},
                hovertemplate="Logit Diff: %{x:.3f}<br>Count: %{y}<extra>Incorrect</extra>",
            )
        )

    # ── Correct (positive logit diff) ─────────────────────────────────────
    if len(correct) > 0:
        fig.add_trace(
            go.Histogram(
                x=correct["logit_diff"],
                nbinsx=n_bins,
                name=f"Correct (n={len(correct):,})",
                marker_color=_COLOR_CORRECT,
                opacity=0.8,
                xbins={"start": x_min, "end": x_max},
                hovertemplate="Logit Diff: %{x:.3f}<br>Count: %{y}<extra>Correct</extra>",
            )
        )

    # ── Vertical line at x=0 (decision boundary) ─────────────────────────
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="white",
        line_width=2,
        annotation_text=" Decision boundary (0)",
        annotation_position="top right",
        annotation_font={"color": "rgba(255,255,255,0.8)", "size": 11},
    )

    # ── Mean logit diff annotation ────────────────────────────────────────
    mean_ld = results_df["logit_diff"].mean()
    fig.add_vline(
        x=mean_ld,
        line_dash="dot",
        line_color=_COLOR_LOGIT_DIFF,
        line_width=2,
        annotation_text=f" μ = {mean_ld:+.3f}",
        annotation_position="top left",
        annotation_font={"color": _COLOR_LOGIT_DIFF, "size": 11},
    )

    fig.update_layout(barmode="overlay")
    fig = _apply_base_layout(fig, title)
    fig.update_xaxes(
        title_text="Logit Difference (IO − S)",
        gridcolor="rgba(255,255,255,0.08)",
    )
    fig.update_yaxes(
        title_text="Number of Prompts",
        gridcolor="rgba(255,255,255,0.08)",
    )

    # Accuracy annotation in top-right
    accuracy = results_df["is_correct"].mean()
    fig.add_annotation(
        x=0.97, y=0.95,
        xref="paper", yref="paper",
        text=f"<b>Accuracy: {accuracy:.1%}</b>",
        showarrow=False,
        font={"size": 14, "color": _COLOR_CORRECT},
        bgcolor="rgba(0,201,167,0.15)",
        bordercolor=_COLOR_CORRECT,
        borderwidth=1,
        borderpad=8,
    )

    if save_path is not None:
        save_figure(fig, save_path, formats=formats, width=_WIDTH, height=_HEIGHT)

    logger.info("[plot_logit_diff_histogram] ✓ Logit diff histogram created.")
    return fig


def plot_confidence_histogram(
    results_df: pd.DataFrame,
    title: str = "Model Confidence: P(IO Token) Distribution",
    n_bins: int = 50,
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    Plot the distribution of softmax probabilities assigned to the IO token.

    High probability = model is confident the IO is correct.
    Low probability = model is uncertain or wrong.

    Color mapping:
      - prob_io > 0.5 → amber (model is confident, usually correct)
      - prob_io ≤ 0.5 → coral (model is uncertain or wrong)

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: "prob_io", "is_correct".

    title : str
        Chart title.

    n_bins : int
        Number of histogram bins.

    save_path : str or Path, optional
        Base save path without extension.

    formats : list of str, optional
        Output formats.

    Returns
    -------
    go.Figure
        Interactive Plotly histogram.
    """
    high_conf = results_df[results_df["prob_io"] > 0.1]
    low_conf = results_df[results_df["prob_io"] <= 0.1]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=results_df["prob_io"],
            nbinsx=n_bins,
            name="P(IO Token)",
            marker_color=_COLOR_CONFIDENCE,
            marker_line={"color": "rgba(255,255,255,0.2)", "width": 0.5},
            opacity=0.85,
            hovertemplate="P(IO): %{x:.4f}<br>Count: %{y}<extra></extra>",
        )
    )

    # Mean probability line
    mean_prob = results_df["prob_io"].mean()
    fig.add_vline(
        x=mean_prob,
        line_dash="dash",
        line_color=_COLOR_CORRECT,
        line_width=2,
        annotation_text=f" μ = {mean_prob:.4f}",
        annotation_position="top right",
        annotation_font={"color": _COLOR_CORRECT, "size": 11},
    )

    # Stats box
    stats_text = (
        f"<b>Statistics</b><br>"
        f"Mean P(IO): {mean_prob:.4f}<br>"
        f"Median P(IO): {results_df['prob_io'].median():.4f}<br>"
        f"Std: {results_df['prob_io'].std():.4f}"
    )
    fig.add_annotation(
        x=0.97, y=0.95,
        xref="paper", yref="paper",
        text=stats_text,
        showarrow=False,
        font={"size": 12, "color": "#E8E8E8"},
        bgcolor="rgba(255,255,255,0.05)",
        bordercolor="rgba(255,255,255,0.2)",
        borderwidth=1,
        borderpad=10,
        align="left",
    )

    fig = _apply_base_layout(fig, title)
    fig.update_xaxes(
        title_text="Softmax Probability P(IO)",
        range=[0, 1],
        gridcolor="rgba(255,255,255,0.08)",
    )
    fig.update_yaxes(
        title_text="Number of Prompts",
        gridcolor="rgba(255,255,255,0.08)",
    )

    if save_path is not None:
        save_figure(fig, save_path, formats=formats, width=_WIDTH, height=_HEIGHT)

    logger.info("[plot_confidence_histogram] ✓ Confidence histogram created.")
    return fig


def plot_top_k_distribution(
    results_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top-1 Prediction Token Frequency",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    Bar chart of the most common top-1 predicted tokens across all prompts.

    In a well-functioning IOI circuit, the most common top-1 predictions
    should be the IO names (Alice, Bob, Mary, etc.). If generic tokens
    like " the", " a", or " he" dominate, the circuit is not functioning.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain column "top_1_prediction".

    top_n : int
        Number of most frequent tokens to show. Default: 20.

    title : str
        Chart title.

    save_path : str or Path, optional
        Base save path without extension.

    formats : list of str, optional
        Output formats.

    Returns
    -------
    go.Figure
        Interactive horizontal bar chart.
    """
    # Count frequency of each top-1 prediction
    token_counts = (
        results_df["top_1_prediction"]
        .value_counts()
        .head(top_n)
    )

    tokens = token_counts.index.tolist()
    counts = token_counts.values.tolist()

    # Fraction of total prompts
    total = len(results_df)
    fractions = [c / total for c in counts]

    fig = go.Figure(
        go.Bar(
            y=tokens[::-1],    # Reverse for ascending order (highest at top)
            x=counts[::-1],
            orientation="h",
            marker={
                "color": [_COLOR_CORRECT if t in results_df["io_name"].values else _COLOR_TOP_K
                          for t in tokens[::-1]],
                "line": {"color": "rgba(255,255,255,0.2)", "width": 0.8},
            },
            text=[f"{c:,} ({f:.1%})" for c, f in zip(counts[::-1], fractions[::-1])],
            textposition="outside",
            textfont={"size": 11, "color": "#CCCCCC"},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Count: %{x:,}<br>"
                "Fraction: %{customdata:.1%}<extra></extra>"
            ),
            customdata=fractions[::-1],
        )
    )

    fig = _apply_base_layout(
        fig, title,
        height=max(400, top_n * 28),
    )
    fig.update_xaxes(
        title_text="Number of Prompts",
        gridcolor="rgba(255,255,255,0.08)",
    )
    fig.update_yaxes(
        title_text="Token",
        tickfont={"size": 12},
    )

    if save_path is not None:
        save_figure(
            fig, save_path, formats=formats,
            width=_WIDTH, height=max(500, top_n * 28)
        )

    logger.info("[plot_top_k_distribution] ✓ Top-K token distribution created.")
    return fig


def plot_logit_diff_by_template(
    results_df: pd.DataFrame,
    title: str = "Logit Difference by Template Type (ABB vs. BAB)",
    save_path: Optional[Union[str, Path]] = None,
    formats: Optional[list[str]] = None,
) -> go.Figure:
    """
    Side-by-side violin plots of logit difference split by template type.

    Violin plots show the full distribution shape (not just quartiles)
    which is important for detecting multi-modal distributions that might
    indicate the circuit works for some templates but not others.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: "logit_diff", "template_type".

    title : str
        Chart title.

    save_path : str or Path, optional
        Base save path.

    formats : list of str, optional
        Output formats.

    Returns
    -------
    go.Figure
        Interactive Plotly violin figure.
    """
    template_colors = {"ABB": "#4D8AF0", "BAB": "#F0A04D"}

    fig = go.Figure()

    for tmpl, color in template_colors.items():
        subset = results_df[results_df["template_type"] == tmpl]
        if len(subset) == 0:
            continue

        fig.add_trace(
            go.Violin(
                x=[tmpl] * len(subset),
                y=subset["logit_diff"],
                name=tmpl,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                opacity=0.7,
                line_color=color,
                hoverinfo="y",
            )
        )

    # Decision boundary
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(255,255,255,0.4)",
        annotation_text=" Correct threshold (0)",
        annotation_position="right",
        annotation_font={"size": 11, "color": "rgba(200,200,200,0.7)"},
    )

    fig = _apply_base_layout(fig, title)
    fig.update_xaxes(title_text="Template Type")
    fig.update_yaxes(
        title_text="Logit Difference (IO − S)",
        gridcolor="rgba(255,255,255,0.08)",
    )

    if save_path is not None:
        save_figure(fig, save_path, formats=formats, width=_WIDTH, height=_HEIGHT)

    logger.info("[plot_logit_diff_by_template] ✓ Violin plot created.")
    return fig


def save_all_baseline_plots(
    results_df: pd.DataFrame,
    figures_dir: Union[str, Path],
    formats: Optional[list[str]] = None,
) -> dict[str, list[Path]]:
    """
    Generate and save all baseline IOI plots in one call.

    This is the main entry point for the visualization pipeline. It calls
    every plot function above and saves each to `figures_dir`.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full evaluation results from `IOIEvaluator.evaluate()`.

    figures_dir : str or Path
        Output directory for all figures. Created if it doesn't exist.

    formats : list of str, optional
        Output formats for each figure. Default: ["html", "png"].

    Returns
    -------
    dict[str, list[Path]]
        Mapping of plot name → list of saved file paths.

    Examples
    --------
    >>> from src.visualization import save_all_baseline_plots
    >>> saved = save_all_baseline_plots(results_df, "outputs/figures")
    >>> for name, paths in saved.items():
    ...     print(f"{name}: {paths}")
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ["html", "png"]

    saved: dict[str, list] = {}

    # ── 1. Accuracy bar chart ─────────────────────────────────────────────
    fig1 = plot_accuracy_bar(
        results_df,
        save_path=figures_dir / "01_accuracy_bar",
        formats=formats,
    )
    saved["accuracy_bar"] = [figures_dir / f"01_accuracy_bar.{fmt}" for fmt in formats]

    # ── 2. Logit difference histogram ─────────────────────────────────────
    fig2 = plot_logit_diff_histogram(
        results_df,
        save_path=figures_dir / "02_logit_diff_histogram",
        formats=formats,
    )
    saved["logit_diff_histogram"] = [
        figures_dir / f"02_logit_diff_histogram.{fmt}" for fmt in formats
    ]

    # ── 3. Confidence histogram ───────────────────────────────────────────
    fig3 = plot_confidence_histogram(
        results_df,
        save_path=figures_dir / "03_confidence_histogram",
        formats=formats,
    )
    saved["confidence_histogram"] = [
        figures_dir / f"03_confidence_histogram.{fmt}" for fmt in formats
    ]

    # ── 4. Top-K token distribution ───────────────────────────────────────
    fig4 = plot_top_k_distribution(
        results_df,
        save_path=figures_dir / "04_top_k_distribution",
        formats=formats,
    )
    saved["top_k_distribution"] = [
        figures_dir / f"04_top_k_distribution.{fmt}" for fmt in formats
    ]

    # ── 5. Logit diff by template (violin) ───────────────────────────────
    fig5 = plot_logit_diff_by_template(
        results_df,
        save_path=figures_dir / "05_logit_diff_by_template",
        formats=formats,
    )
    saved["logit_diff_by_template"] = [
        figures_dir / f"05_logit_diff_by_template.{fmt}" for fmt in formats
    ]

    logger.info(
        f"[save_all_baseline_plots] ✓ All {len(saved)} baseline plots saved to "
        f"{figures_dir.resolve()}"
    )
    return saved
