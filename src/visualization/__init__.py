"""
src.visualization — Plotting sub-package
=========================================
Exports baseline plots and mechanistic interpretability circuit visualizations.

Modules
-------
plots       : Baseline IOI evaluation plots (accuracy, logit diff histogram, etc.)
circuit_vis : Circuit analysis plots (logit lens, ablation heatmaps, circuit graph)
"""
from .plots import (
    plot_accuracy_bar,
    plot_logit_diff_histogram,
    plot_confidence_histogram,
    plot_top_k_distribution,
    save_all_baseline_plots,
)
from .circuit_vis import (
    plot_logit_lens_curve,
    plot_logit_lens_heatmap,
    plot_layer_ablation_bars,
    plot_layer_ablation_heatmap,
    plot_head_importance_heatmap,
    plot_head_ranking_bar,
    plot_activation_patching_heatmap,
    plot_all_patching_comparison,
    plot_circuit_graph,
    plot_sender_importance_heatmap,
    plot_attention_patterns,
)

__all__ = [
    # Baseline
    "plot_accuracy_bar",
    "plot_logit_diff_histogram",
    "plot_confidence_histogram",
    "plot_top_k_distribution",
    "save_all_baseline_plots",
    # Circuit analysis
    "plot_logit_lens_curve",
    "plot_logit_lens_heatmap",
    "plot_layer_ablation_bars",
    "plot_layer_ablation_heatmap",
    "plot_head_importance_heatmap",
    "plot_head_ranking_bar",
    "plot_activation_patching_heatmap",
    "plot_all_patching_comparison",
    "plot_circuit_graph",
    "plot_sender_importance_heatmap",
    "plot_attention_patterns",
]
