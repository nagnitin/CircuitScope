"""
experiments/10_statistical_analysis.py
========================================
Part 3: Statistical Analysis — Confidence Intervals, Effect Sizes, Correlations

Computes research-grade statistics over all experiment results:
  1. Bootstrap 95% CI for accuracy and logit diff (IOI and pronoun tasks)
  2. Cohen's d for head ablation importance scores vs. neutral heads
  3. Layer depth × importance Spearman correlation (with permutation test)
  4. Comprehensive stats tables per template type (ABB vs. BAB)
  5. IOI vs. pronoun task comparison

Requires previously run experiments (at minimum 01 baseline and 04 head ablation).
Loads results from CSV files in outputs/results/.

Produces:
  outputs/results/stats_bootstrap_ci.csv
  outputs/results/stats_effect_sizes.csv
  outputs/results/stats_layer_correlation.json
  outputs/results/stats_comprehensive.csv
  outputs/figures/28_bootstrap_ci_plot.html/.png
  outputs/figures/29_effect_sizes_plot.html/.png
  outputs/figures/30_layer_correlation_scatter.html/.png

Usage:
  python experiments/10_statistical_analysis.py
  python experiments/10_statistical_analysis.py --n-bootstrap 5000
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical Analysis")
    parser.add_argument("--config",
                        default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.utils.logger import get_logger, silence_external_loggers
    from src.utils.io_utils import ensure_dirs, save_csv, save_json, save_figure
    from src.analysis.statistics import (
        bootstrap_ci,
        cohen_d,
        classify_effect_size,
        permutation_test,
        layer_depth_correlation,
        compute_comprehensive_stats,
        ioi_vs_pronoun_comparison,
        analyse_head_importance_statistics,
    )

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger("circuitscope.statistics", level=log_level,
                        log_dir=config["paths"]["logs_dir"])
    silence_external_loggers()

    paths = config["paths"]
    # Per-experiment output subdirectories
    paths["figures_dir"] = paths["outputs_dir"] + "/10_statistical_analysis/figures"
    paths["results_dir"] = paths["outputs_dir"] + "/10_statistical_analysis/results"
    ensure_dirs(paths["figures_dir"], paths["results_dir"], paths["logs_dir"])

    _BG_PLOT  = "rgba(13, 17, 23, 0.95)"
    _BG_PAPER = "rgba(10, 13, 18, 1.0)"
    _FONT     = "Inter, Arial, sans-serif"
    formats = config.get("plotting", {}).get("export_formats", ["html", "png"])

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    # -- Load results files ------------------------------------------------
    results_dir = Path(paths["results_dir"])

    def load_if_exists(fname: str) -> pd.DataFrame | None:
        p = results_dir / fname
        if p.exists():
            return pd.read_csv(p)
        logger.warning(f"File not found: {p}. Run the corresponding experiment first.")
        return None

    ioi_df      = load_if_exists("ioi_results.csv")
    head_df     = load_if_exists("head_ablation.csv")
    layer_df    = load_if_exists("layer_ablation.csv")
    pronoun_df  = load_if_exists("pronoun_baseline.csv")

    all_stats = {}

    # ====================================================================
    # 1. Bootstrap CI for IOI Task Metrics
    # ====================================================================
    print("\n===============================================================")
    print("  STATISTICAL ANALYSIS")
    print("===============================================================")

    ci_rows = []
    if ioi_df is not None:
        print("\n[1] Bootstrap 95% Confidence Intervals — IOI Task")
        for metric_col, label in [
            ("logit_diff", "Logit Difference"),
            ("is_correct", "Accuracy"),
            ("prob_io", "P(IO Token)"),
        ]:
            if metric_col not in ioi_df.columns:
                continue
            data = ioi_df[metric_col].dropna().values.astype(float)
            obs, lo, hi = bootstrap_ci(data, n_bootstrap=args.n_bootstrap, seed=42)
            ci_rows.append({
                "task": "IOI",
                "metric": label,
                "estimate": round(obs, 6),
                "ci_lower": round(lo, 6),
                "ci_upper": round(hi, 6),
                "ci_width": round(hi - lo, 6),
                "n": len(data),
            })
            print(f"  {label:<25}: {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    if pronoun_df is not None:
        print("\n[1b] Bootstrap 95% Confidence Intervals — Pronoun Task")
        for metric_col, label in [
            ("logit_diff", "Logit Difference"),
            ("is_correct", "Accuracy"),
        ]:
            if metric_col not in pronoun_df.columns:
                continue
            data = pronoun_df[metric_col].dropna().values.astype(float)
            obs, lo, hi = bootstrap_ci(data, n_bootstrap=args.n_bootstrap, seed=42)
            ci_rows.append({
                "task": "Pronoun",
                "metric": label,
                "estimate": round(obs, 6),
                "ci_lower": round(lo, 6),
                "ci_upper": round(hi, 6),
                "ci_width": round(hi - lo, 6),
                "n": len(data),
            })
            print(f"  {label:<25}: {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    if ci_rows:
        ci_df = pd.DataFrame(ci_rows)
        save_csv(ci_df, paths["results_dir"] + "/stats_bootstrap_ci.csv")

    # ════════════════════════════════════════════════════════════════════
    # 2. Effect Sizes: Head Ablation (circuit vs. neutral heads)
    # ════════════════════════════════════════════════════════════════════
    effect_rows = []
    if head_df is not None and "importance" in head_df.columns:
        print("\n[2] Cohen's d Effect Sizes — Circuit vs. Neutral Heads")

        neutral_importances = head_df[
            head_df["head_type"] == "Neutral"
        ]["importance"].values
        circuit_importances = head_df[
            head_df["head_type"].isin(["Name Mover", "Helper"])
        ]["importance"].values
        suppressor_importances = head_df[
            head_df["head_type"].isin(["Suppressor", "Strong Suppressor"])
        ]["importance"].values

        for group_name, group_imps in [
            ("Name Mover / Helper", circuit_importances),
            ("Suppressor", suppressor_importances),
        ]:
            if len(group_imps) >= 2 and len(neutral_importances) >= 2:
                d = cohen_d(group_imps, neutral_importances)
                obs, lo, hi = bootstrap_ci(group_imps, n_bootstrap=args.n_bootstrap)
                stat, p = permutation_test(
                    group_imps, neutral_importances, alternative="greater"
                )
                cat = classify_effect_size(d)
                print(f"  {group_name:<25}: d={d:+.4f} ({cat}), p={p:.4f}")
                effect_rows.append({
                    "group": group_name,
                    "n_heads": len(group_imps),
                    "mean_importance": round(float(np.mean(group_imps)), 6),
                    "cohens_d": round(d, 6),
                    "effect_category": cat,
                    "permutation_p": round(p, 6),
                    "ci_lower": round(lo, 6),
                    "ci_upper": round(hi, 6),
                })

        # Late vs. early layer comparison
        late_heads = head_df[head_df["layer"] >= 9]["importance"].values
        early_heads = head_df[head_df["layer"] <= 4]["importance"].values
        if len(late_heads) >= 2 and len(early_heads) >= 2:
            d_late_early = cohen_d(late_heads, early_heads)
            stat, p = permutation_test(
                late_heads, early_heads, alternative="greater"
            )
            cat = classify_effect_size(d_late_early)
            print(f"  {'Late (9-11) vs. Early (0-4)':<25}: d={d_late_early:+.4f} ({cat}), p={p:.4f}")
            effect_rows.append({
                "group": "Late layers (9-11) vs. Early (0-4)",
                "n_heads": len(late_heads),
                "mean_importance": round(float(np.mean(late_heads)), 6),
                "cohens_d": round(d_late_early, 6),
                "effect_category": cat,
                "permutation_p": round(p, 6),
                "ci_lower": round(float(np.percentile(late_heads, 2.5)), 6),
                "ci_upper": round(float(np.percentile(late_heads, 97.5)), 6),
            })

    if effect_rows:
        effect_df = pd.DataFrame(effect_rows)
        save_csv(effect_df, paths["results_dir"] + "/stats_effect_sizes.csv")

    # ════════════════════════════════════════════════════════════════════
    # 3. Layer Depth × Importance Correlation
    # ════════════════════════════════════════════════════════════════════
    if head_df is not None and "importance" in head_df.columns:
        print("\n[3] Spearman Correlation: Layer Depth × Head Importance")
        corr_result = layer_depth_correlation(head_df)
        print(f"  ρ = {corr_result['spearman_r']:.4f}, {corr_result['significance']}")
        print(f"  {corr_result['interpretation']}")
        save_json(corr_result, paths["results_dir"] + "/stats_layer_correlation.json")
        all_stats["layer_correlation"] = corr_result

    # ════════════════════════════════════════════════════════════════════
    # 4. Comprehensive Stats by Template Type
    # ════════════════════════════════════════════════════════════════════
    if ioi_df is not None and "template_type" in ioi_df.columns:
        print("\n[4] Comprehensive Stats by Template Type (IOI)")
        for metric in ["logit_diff", "is_correct"]:
            if metric not in ioi_df.columns:
                continue
            stats = compute_comprehensive_stats(
                ioi_df, metric_col=metric,
                group_col="template_type",
                n_bootstrap=args.n_bootstrap,
            )
            print(f"\n  {metric}:")
            print(stats[["group", "n", "mean", "std", "ci_lower_95", "ci_upper_95"]].to_string(index=False))
            save_csv(stats, paths["results_dir"] + f"/stats_{metric}_by_template.csv")

    # ════════════════════════════════════════════════════════════════════
    # 5. IOI vs. Pronoun Task Comparison
    # ════════════════════════════════════════════════════════════════════
    if ioi_df is not None and pronoun_df is not None:
        print("\n[5] IOI vs. Pronoun Task Comparison")
        comparison = ioi_vs_pronoun_comparison(ioi_df, pronoun_df)
        print(f"  IOI LD: {comparison['ioi_mean']:+.4f} [{comparison['ioi_ci_lower']:+.4f}, {comparison['ioi_ci_upper']:+.4f}]")
        print(f"  Pronoun LD: {comparison['pronoun_mean']:+.4f} [{comparison['pronoun_ci_lower']:+.4f}, {comparison['pronoun_ci_upper']:+.4f}]")
        print(f"  Cohen's d: {comparison['cohens_d']:.4f} ({comparison['effect_category']})")
        print(f"  p-value: {comparison['p_value']:.4f}")
        all_stats["task_comparison"] = comparison
        save_json(all_stats, paths["results_dir"] + "/stats_summary.json")

    # ════════════════════════════════════════════════════════════════════
    # PLOTS
    # ════════════════════════════════════════════════════════════════════

    # Plot 1: Bootstrap CI bar chart
    if ci_rows:
        ci_df = pd.DataFrame(ci_rows)
        tasks = ci_df["task"].unique()
        task_colors = {"IOI": "#58A6FF", "Pronoun": "#00C9A7"}

        fig1 = go.Figure()
        for task in tasks:
            tdf = ci_df[ci_df["task"] == task]
            fig1.add_trace(go.Bar(
                name=task,
                x=tdf["metric"],
                y=tdf["estimate"],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (tdf["ci_upper"] - tdf["estimate"]).tolist(),
                    "arrayminus": (tdf["estimate"] - tdf["ci_lower"]).tolist(),
                    "color": "rgba(255,255,255,0.7)",
                    "thickness": 2,
                    "width": 6,
                },
                marker_color=task_colors.get(task, "#FFC75F"),
                opacity=0.85,
            ))
        fig1.update_layout(
            title={"text": "Bootstrap 95% Confidence Intervals by Task and Metric",
                   "x": 0.5, "xanchor": "center"},
            template="plotly_dark",
            plot_bgcolor=_BG_PLOT, paper_bgcolor=_BG_PAPER,
            font={"family": _FONT},
            barmode="group",
            yaxis_title="Estimate (95% CI)",
            width=900, height=500,
        )
        save_figure(fig1, paths["figures_dir"] + "/28_bootstrap_ci_plot",
                    formats=formats, width=900, height=500)

    # Plot 2: Layer depth vs. head importance scatter
    if head_df is not None and "importance" in head_df.columns:
        type_colors = {
            "Name Mover": "#00C9A7",
            "Helper": "#58A6FF",
            "Neutral": "rgba(150,150,150,0.5)",
            "Suppressor": "#FF6B6B",
            "Strong Suppressor": "#FF0033",
        }

        fig2 = go.Figure()
        for head_type, color in type_colors.items():
            subset = head_df[head_df["head_type"] == head_type]
            if subset.empty:
                continue
            jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(subset))
            fig2.add_trace(go.Scatter(
                x=subset["layer"] + jitter,
                y=subset["importance"],
                mode="markers",
                name=head_type,
                marker={
                    "size": 8,
                    "color": color,
                    "opacity": 0.8,
                    "line": {"color": "rgba(255,255,255,0.3)", "width": 0.5},
                },
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Layer: %{x:.1f}<br>"
                    "Importance: %{y:+.4f}<extra></extra>"
                ),
                customdata=subset["head_label"],
            ))

        # Trend line
        if corr_result and len(head_df) > 0:
            from scipy import stats as scipy_stats
            m, b, *_ = scipy_stats.linregress(
                head_df["layer"].values, head_df["importance"].values
            )
            x_line = np.arange(0, 12)
            fig2.add_trace(go.Scatter(
                x=x_line, y=m * x_line + b,
                mode="lines", name=f"Trend (ρ={corr_result['spearman_r']:.3f})",
                line={"color": "#FFC75F", "width": 2, "dash": "dot"},
            ))

        fig2.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
        fig2.update_layout(
            title={"text": "Head Importance vs. Layer Depth",
                   "x": 0.5, "xanchor": "center"},
            template="plotly_dark",
            plot_bgcolor=_BG_PLOT, paper_bgcolor=_BG_PAPER,
            font={"family": _FONT},
            xaxis={"title": "Layer (with jitter)", "range": [-0.5, 11.5],
                   "tickvals": list(range(12)), "ticktext": [f"L{l}" for l in range(12)]},
            yaxis_title="Normalised Importance (LD drop / baseline)",
            width=1000, height=600,
        )
        save_figure(fig2, paths["figures_dir"] + "/29_layer_importance_correlation",
                    formats=formats, width=1000, height=600)

    # Plot 3: Effect sizes
    if effect_rows:
        effect_df = pd.DataFrame(effect_rows)
        fig3 = go.Figure()
        colors_ef = ["#00C9A7" if d > 0 else "#FF6B6B" for d in effect_df["cohens_d"]]
        fig3.add_trace(go.Bar(
            x=effect_df["group"],
            y=effect_df["cohens_d"],
            marker_color=colors_ef,
            text=[f"d={d:+.3f}<br>({c})" for d, c in
                  zip(effect_df["cohens_d"], effect_df["effect_category"])],
            textposition="outside",
        ))
        for threshold, label in [(0.2, "Small"), (0.5, "Medium"), (0.8, "Large")]:
            fig3.add_hline(y=threshold, line_dash="dot",
                           line_color="rgba(255,255,255,0.25)",
                           annotation_text=f" {label}")
        fig3.update_layout(
            title={"text": "Cohen's d Effect Sizes: Circuit Heads vs. Neutral",
                   "x": 0.5, "xanchor": "center"},
            template="plotly_dark",
            plot_bgcolor=_BG_PLOT, paper_bgcolor=_BG_PAPER,
            font={"family": _FONT},
            yaxis_title="Cohen's d",
            width=800, height=500,
        )
        save_figure(fig3, paths["figures_dir"] + "/30_effect_sizes_plot",
                    formats=formats, width=800, height=500)

    logger.info("✓ Statistical analysis complete.")
    print(f"\n✓ Statistics saved to {paths['results_dir']}/")
    print(f"✓ Plots saved to {paths['figures_dir']}/")


if __name__ == "__main__":
    main()
