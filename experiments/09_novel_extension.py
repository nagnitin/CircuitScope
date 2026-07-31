"""
experiments/09_novel_extension.py
===================================
Part 2: Novel Extension — Pronoun Resolution

Investigates whether the IOI circuit (Name Mover Heads, S-Inhibition Heads)
also mediates pronoun-based coreference resolution.

Task: Given "Alice met Bob. She bought a gift for ___", the model should
predict "Bob" (the recipient) rather than "Alice" (the speaker).

Hypothesis: Since both IOI and pronoun resolution require:
  (a) Name token identification in context
  (b) Predicting the "other" named entity
  (c) Suppressing the repeated/incorrect name

...the same late-layer Name Mover heads should be activated for both tasks.

Analysis Pipeline (applied to pronoun dataset):
  1. Baseline evaluation (accuracy, logit diff, rank)
  2. Layer ablation (which layers are causally important?)
  3. Head ablation (do the same heads matter?)
  4. Comparison with IOI results

Produces:
  outputs/results/pronoun_baseline.csv
  outputs/results/pronoun_head_ablation.csv
  outputs/results/task_comparison.json
  outputs/figures/25_pronoun_baseline.html/.png
  outputs/figures/26_task_head_comparison.html/.png
  outputs/figures/27_ioi_vs_pronoun_heatmap.html/.png

Usage:
  python experiments/09_novel_extension.py
  python experiments/09_novel_extension.py --n-samples 100
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main() -> None:
    parser = argparse.ArgumentParser(description="Novel Extension: Pronoun Resolution")
    parser.add_argument("--config",
                        default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--n-prompts", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Samples for ablation experiments (matches Exp 04)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip head ablation (much faster)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.utils.logger import get_logger, silence_external_loggers
    from src.utils.reproducibility import set_seed
    from src.utils.io_utils import ensure_dirs, save_csv, save_json, save_figure
    from src.model.loader import load_model
    from src.data.ioi_dataset import IOIDataset
    from src.data.pronoun_dataset import PronounDataset
    from src.analysis.head_ablation import HeadAblationAnalyzer
    from src.analysis.layer_ablation import LayerAblationAnalyzer
    from src.evaluation.metrics import IOIEvaluator
    from src.analysis.statistics import (
        ioi_vs_pronoun_comparison,
        compute_comprehensive_stats,
    )
    import torch

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger("circuitscope.pronoun", level=log_level,
                        log_dir=config["paths"]["logs_dir"])
    silence_external_loggers()

    paths = config["paths"]
    # Per-experiment output subdirectories
    paths["figures_dir"] = paths["outputs_dir"] + "/09_novel_extension/figures"
    paths["results_dir"] = paths["outputs_dir"] + "/09_novel_extension/results"
    ensure_dirs(paths["figures_dir"], paths["results_dir"], paths["logs_dir"])
    set_seed(config.get("seed", 42))

    _BG_PLOT  = "rgba(13, 17, 23, 0.95)"
    _BG_PAPER = "rgba(10, 13, 18, 1.0)"
    _FONT     = "Inter, Arial, sans-serif"

    # ── Load model ────────────────────────────────────────────────────────
    logger.info("Loading model…")
    model = load_model(config["model"]["name"], device=config["model"]["device"])
    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # ── Generate pronoun dataset ──────────────────────────────────────────
    logger.info(f"Generating pronoun dataset ({args.n_prompts} prompts)…")
    pronoun_ds = PronounDataset(
        model=model, n_prompts=args.n_prompts, seed=config.get("seed", 42)
    ).generate()

    save_csv(pronoun_ds.to_dataframe(),
             paths["results_dir"] + "/pronoun_dataset.csv")
    logger.info(f"Pronoun dataset: {len(pronoun_ds)} prompts generated")

    # ── Baseline evaluation on pronoun task ───────────────────────────────
    logger.info("Running pronoun task baseline evaluation…")
    t0 = time.time()

    from src.evaluation.metrics import compute_logit_diff, compute_token_rank
    import torch.nn.functional as F

    bos_id = model.tokenizer.bos_token_id
    prompts = pronoun_ds.get_clean_prompts()
    io_ids = pronoun_ds.get_io_token_ids()
    s_ids = pronoun_ds.get_s_token_ids()
    n = len(prompts)

    pronoun_rows = []
    batch_size = args.batch_size

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_prompts = prompts[batch_start:batch_end]
        batch_io = io_ids[batch_start:batch_end]
        batch_s = s_ids[batch_start:batch_end]

        token_lists = [
            model.to_tokens(p, prepend_bos=True)[0].tolist()
            for p in batch_prompts
        ]
        seq_lengths = [len(t) for t in token_lists]
        max_len = max(seq_lengths)
        padded = [t + [bos_id] * (max_len - len(t)) for t in token_lists]
        tokens = torch.tensor(padded, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(tokens)

        for i, (io_id, s_id, seq_len) in enumerate(
            zip(batch_io, batch_s, seq_lengths)
        ):
            final_logits = logits[i, seq_len - 1, :]
            ld = compute_logit_diff(final_logits, io_id, s_id)
            probs = F.softmax(final_logits, dim=-1)
            rank_io = compute_token_rank(final_logits, io_id)

            pronoun_rows.append({
                "prompt": batch_prompts[i],
                "speaker": pronoun_ds.prompts[batch_start + i].speaker_name,
                "recipient": pronoun_ds.prompts[batch_start + i].recipient_name,
                "pronoun": pronoun_ds.prompts[batch_start + i].pronoun,
                "logit_diff": ld,
                "prob_io": probs[io_id].item(),
                "prob_s": probs[s_id].item(),
                "is_correct": ld > 0,
                "rank_io": rank_io,
                "template_idx": pronoun_ds.prompts[batch_start + i].template_idx,
            })

    pronoun_eval_df = pd.DataFrame(pronoun_rows)
    save_csv(pronoun_eval_df, paths["results_dir"] + "/pronoun_baseline.csv")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    pron_acc = pronoun_eval_df["is_correct"].mean()
    pron_mean_ld = pronoun_eval_df["logit_diff"].mean()
    logger.info(
        f"Pronoun task baseline: acc={pron_acc:.1%}, mean_ld={pron_mean_ld:+.4f}"
    )
    print(f"\n-- Pronoun Task Baseline --------------------------------------")
    print(f"  Accuracy:        {pron_acc:.1%}")
    print(f"  Mean logit diff: {pron_mean_ld:+.4f}")
    print(f"  Mean P(target):  {pronoun_eval_df['prob_io'].mean():.4f}")
    print(f"  Mean rank:       {pronoun_eval_df['rank_io'].mean():.1f}")

    # -- Head ablation on pronoun task -------------------------------------
    pronoun_head_df = None
    if not args.skip_ablation:
        logger.info("Running head ablation on pronoun task...")
        head_analyzer = HeadAblationAnalyzer(
            model, pronoun_ds,
            n_samples=args.n_samples, batch_size=args.batch_size
        )
        mean_z = head_analyzer.compute_mean_z()
        pronoun_head_df = head_analyzer.run_full_sweep(mean_z)
        save_csv(pronoun_head_df, paths["results_dir"] + "/pronoun_head_ablation.csv")

    # -- Load IOI results for comparison ----------------------------------
    ioi_csv = Path(paths["outputs_dir"]) / "01_baseline" / "results" / "ioi_results.csv"
    ioi_head_csv = Path(paths["outputs_dir"]) / "04_head_ablation" / "results" / "head_ablation.csv"
    ioi_eval_df = None
    ioi_head_df = None

    if ioi_csv.exists():
        ioi_eval_df = pd.read_csv(ioi_csv)
        logger.info(f"Loaded IOI eval results: {len(ioi_eval_df)} rows")
    if ioi_head_csv.exists():
        ioi_head_df = pd.read_csv(ioi_head_csv)
        logger.info(f"Loaded IOI head ablation: {len(ioi_head_df)} rows")

    # -- Statistical comparison --------------------------------------------
    comparison_results = {}
    if ioi_eval_df is not None and "logit_diff" in ioi_eval_df.columns:
        comparison = ioi_vs_pronoun_comparison(ioi_eval_df, pronoun_eval_df)
        comparison_results["logit_diff_comparison"] = comparison
        save_json(comparison_results, paths["results_dir"] + "/task_comparison.json")

        print(f"\n-- IOI vs. Pronoun Comparison ---------------------------------")
        print(f"  IOI mean LD    : {comparison['ioi_mean']:+.4f} "
              f"[{comparison['ioi_ci_lower']:+.4f}, {comparison['ioi_ci_upper']:+.4f}]")
        print(f"  Pronoun mean LD: {comparison['pronoun_mean']:+.4f} "
              f"[{comparison['pronoun_ci_lower']:+.4f}, {comparison['pronoun_ci_upper']:+.4f}]")
        print(f"  Cohen's d      : {comparison['cohens_d']:.4f} ({comparison['effect_category']})")
        print(f"  p-value        : {comparison['p_value']:.4f} "
              f"({'significant' if comparison['significant_at_05'] else 'not significant'})")
        print(f"  {comparison['interpretation']}")

    # ── Plots ─────────────────────────────────────────────────────────────
    formats = config.get("plotting", {}).get("export_formats", ["html", "png"])

    # Plot 1: Pronoun task baseline (logit diff histogram)
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(
        x=pronoun_eval_df["logit_diff"],
        nbinsx=40,
        name="Pronoun Task",
        marker_color="#00C9A7",
        opacity=0.8,
        hovertemplate="Logit diff: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))
    if ioi_eval_df is not None:
        fig1.add_trace(go.Histogram(
            x=ioi_eval_df["logit_diff"],
            nbinsx=40,
            name="IOI Task",
            marker_color="#58A6FF",
            opacity=0.6,
            hovertemplate="Logit diff: %{x:.3f}<br>Count: %{y}<extra></extra>",
        ))
    fig1.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                   annotation_text=" Decision boundary")
    fig1.update_layout(
        title={"text": "Logit Diff Distribution: Pronoun Resolution vs. IOI Task",
               "x": 0.5, "xanchor": "center"},
        template="plotly_dark",
        plot_bgcolor=_BG_PLOT, paper_bgcolor=_BG_PAPER,
        font={"family": _FONT},
        xaxis_title="Logit Difference (Target − Foil)",
        yaxis_title="Count",
        barmode="overlay",
        width=1000, height=500,
        legend={"bgcolor": "rgba(255,255,255,0.05)"},
    )
    save_figure(fig1, paths["figures_dir"] + "/25_pronoun_vs_ioi_distribution",
                formats=formats, width=1000, height=500)

    # Plot 2: Head importance comparison (if both available)
    if ioi_head_df is not None and pronoun_head_df is not None:
        # Merge on (layer, head)
        merged = ioi_head_df[["layer", "head", "importance"]].rename(
            columns={"importance": "ioi_importance"}
        ).merge(
            pronoun_head_df[["layer", "head", "importance"]].rename(
                columns={"importance": "pronoun_importance"}
            ),
            on=["layer", "head"]
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=merged["ioi_importance"],
            y=merged["pronoun_importance"],
            mode="markers+text",
            text=[f"L{r['layer']}H{r['head']}" for _, r in merged.iterrows()],
            textposition="top center",
            textfont={"size": 9},
            marker={
                "size": 10,
                "color": merged["ioi_importance"],
                "colorscale": "RdBu",
                "cmid": 0,
                "showscale": True,
                "colorbar": {"title": "IOI Importance"},
            },
            hovertemplate=(
                "<b>L%{customdata[0]}H%{customdata[1]}</b><br>"
                "IOI importance: %{x:+.4f}<br>"
                "Pronoun importance: %{y:+.4f}<extra></extra>"
            ),
            customdata=merged[["layer", "head"]].values,
        ))

        # Identity line (perfect correlation)
        lim = max(abs(merged["ioi_importance"].max()),
                  abs(merged["pronoun_importance"].max()))
        fig2.add_shape(
            type="line", x0=-lim, y0=-lim, x1=lim, y1=lim,
            line={"color": "rgba(255,255,255,0.3)", "dash": "dot"},
        )
        fig2.add_vline(x=0, line_color="rgba(255,255,255,0.2)")
        fig2.add_hline(y=0, line_color="rgba(255,255,255,0.2)")

        from scipy import stats as scipy_stats
        r, p = scipy_stats.pearsonr(
            merged["ioi_importance"], merged["pronoun_importance"]
        )

        fig2.update_layout(
            title={"text": f"Head Importance: IOI vs. Pronoun Resolution (r={r:.3f}, p={p:.4f})",
                   "x": 0.5, "xanchor": "center"},
            template="plotly_dark",
            plot_bgcolor=_BG_PLOT, paper_bgcolor=_BG_PAPER,
            font={"family": _FONT},
            xaxis_title="IOI Head Importance",
            yaxis_title="Pronoun Head Importance",
            width=800, height=700,
        )
        save_figure(fig2, paths["figures_dir"] + "/26_ioi_vs_pronoun_head_scatter",
                    formats=formats, width=800, height=700)

        # Save correlation stats
        comparison_results["head_importance_correlation"] = {
            "pearson_r": float(r), "p_value": float(p)
        }
        save_json(comparison_results, paths["results_dir"] + "/task_comparison.json")

        print(f"\n── Head Importance Correlation ─────────────────────────────────")
        print(f"  Pearson r = {r:.4f}, p = {p:.6f}")
        if r > 0.6:
            print(f"  ✓ Strong positive correlation — same heads important for both tasks")
        elif r > 0.3:
            print(f"  ~ Moderate correlation — partial circuit overlap")
        else:
            print(f"  ✗ Weak correlation — different circuits for different tasks")

    logger.info("✓ Novel extension experiment complete.")
    print(f"\n✓ Results saved to {paths['results_dir']}/")
    print(f"✓ Plots saved to {paths['figures_dir']}/")


if __name__ == "__main__":
    main()
