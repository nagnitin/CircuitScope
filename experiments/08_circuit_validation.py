"""
experiments/08_circuit_validation.py
======================================
Part 1: Circuit Validation — Necessity, Sufficiency, and Generalization

This script validates the discovered IOI circuit through three canonical tests:

  Necessity:     Ablate ONLY circuit heads → large drop = necessary
  Sufficiency:   Ablate everything EXCEPT circuit → retained perf = sufficient
  Generalization: Test across held-out prompts, ABB, and BAB templates

To run this script, you must first run experiment 04_head_ablation.py
so that head_ablation.csv exists. The circuit is loaded from that file.

Produces:
  outputs/results/circuit_validation.csv
  outputs/results/circuit_spec.json
  outputs/figures/23_necessity_vs_sufficiency.html/.png
  outputs/figures/24_generalization_comparison.html/.png

Usage:
  python experiments/08_circuit_validation.py
  python experiments/08_circuit_validation.py --n-samples 150
  python experiments/08_circuit_validation.py --threshold 0.10
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
    parser = argparse.ArgumentParser(description="Circuit Validation Experiment")
    parser.add_argument("--config",
                        default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Importance threshold for circuit membership")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.utils.logger import get_logger, silence_external_loggers
    from src.utils.reproducibility import set_seed
    from src.utils.io_utils import ensure_dirs, save_csv, save_json, save_figure
    from src.model.loader import load_model
    from src.data.ioi_dataset import IOIDataset
    from src.analysis.head_ablation import HeadAblationAnalyzer
    from src.analysis.circuit_validation import CircuitSpec, CircuitValidator

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger("circuitscope.validation", level=log_level,
                        log_dir=config["paths"]["logs_dir"])
    silence_external_loggers()

    paths = config["paths"]
    # Per-experiment output subdirectories
    paths["figures_dir"] = paths["outputs_dir"] + "/08_circuit_validation/figures"
    paths["results_dir"] = paths["outputs_dir"] + "/08_circuit_validation/results"
    ensure_dirs(paths["figures_dir"], paths["results_dir"], paths["logs_dir"])
    set_seed(config.get("seed", 42))

    # ── Load model & dataset ──────────────────────────────────────────────
    logger.info("Loading model…")
    model = load_model(config["model"]["name"], device=config["model"]["device"])

    logger.info("Generating dataset…")
    dataset = IOIDataset(
        model=model,
        n_prompts=config["dataset"]["n_prompts"],
        seed=config.get("seed", 42),
    ).generate()

    # ── Load or compute head ablation results ─────────────────────────────
    head_csv = Path(paths["results_dir"]) / "head_ablation.csv"
    if head_csv.exists():
        logger.info(f"Loading head ablation results from {head_csv}…")
        head_df = pd.read_csv(head_csv)
    else:
        logger.info("head_ablation.csv not found — running head ablation now…")
        head_analyzer = HeadAblationAnalyzer(model, dataset,
                                             n_samples=args.n_samples,
                                             batch_size=args.batch_size)
        mean_z = head_analyzer.compute_mean_z()
        head_df = head_analyzer.run_full_sweep(mean_z)
        save_csv(head_df, str(head_csv))

    # ── Build circuit specification ───────────────────────────────────────
    circuit = CircuitSpec.from_head_ablation_df(
        head_df, importance_threshold=args.threshold
    )
    logger.info(f"Circuit: {circuit}")

    circuit_dict = {
        "name": circuit.name,
        "threshold": args.threshold,
        "n_heads": len(circuit),
        "heads": [{"layer": l, "head": h} for l, h in circuit.heads],
    }
    save_json(circuit_dict, paths["results_dir"] + "/circuit_spec.json")

    # ── Compute mean z for validation ─────────────────────────────────────
    logger.info("Computing mean_z cache for validation…")
    head_analyzer = HeadAblationAnalyzer(model, dataset,
                                         n_samples=args.n_samples,
                                         batch_size=args.batch_size)
    mean_z = head_analyzer.compute_mean_z()

    # ── Run validation tests ──────────────────────────────────────────────
    validator = CircuitValidator(
        model, dataset, circuit, mean_z,
        n_samples=args.n_samples, batch_size=args.batch_size
    )
    t0 = time.time()
    validation_results = validator.run_all_tests()
    logger.info(f"Validation complete in {time.time()-t0:.1f}s")

    # ── Build results DataFrame ───────────────────────────────────────────
    val_df = pd.DataFrame([r.to_dict() for r in validation_results])
    save_csv(val_df, paths["results_dir"] + "/circuit_validation.csv")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    # -- Print report ------------------------------------------------------
    print("\n===============================================================")
    print("  CIRCUIT VALIDATION RESULTS")
    print("===============================================================")
    print(f"\n  Circuit: {circuit.name}")
    print(f"  Number of heads in circuit: {len(circuit)}")
    print(f"  Circuit heads: {sorted(circuit.heads)}")
    print(f"\n  {'Test':<35} {'Score':>8} {'Acc Change':>12} {'LD Change':>12}")
    print("  " + "-"*67)
    for r in validation_results:
        acc_chg = r.experimental_acc - r.baseline_acc
        ld_chg = r.experimental_ld - r.baseline_ld
        score_bar = "#" * int(r.score * 20) if 0 <= r.score <= 1 else "?"
        print(f"  {r.test_name:<35} {r.score:>8.4f} {acc_chg:>+12.1%} {ld_chg:>+12.4f}")

    nec_result = next((r for r in validation_results if r.test_name == "necessity"), None)
    suf_result = next((r for r in validation_results if r.test_name == "sufficiency"), None)
    if nec_result and suf_result:
        print(f"\n  Summary:")
        print(f"    Necessity score  = {nec_result.score:.4f} "
              f"({'HIGH' if nec_result.score > 0.5 else 'LOW'})")
        print(f"    Sufficiency score= {suf_result.score:.4f} "
              f"({'HIGH' if suf_result.score > 0.5 else 'LOW'})")
        verdict = (
            "[OK] FULL CIRCUIT" if (nec_result.score > 0.5 and suf_result.score > 0.5)
            else "[PARTIAL] PARTIAL CIRCUIT" if (nec_result.score > 0.3 or suf_result.score > 0.3)
            else "[FAIL] CIRCUIT NOT CONFIRMED"
        )
        print(f"    Verdict: {verdict}")

    # ── Plots ─────────────────────────────────────────────────────────────
    formats = config.get("plotting", {}).get("export_formats", ["html", "png"])
    _BG_PLOT  = "rgba(13, 17, 23, 0.95)"
    _BG_PAPER = "rgba(10, 13, 18, 1.0)"
    _FONT     = "Inter, Arial, sans-serif"

    # Plot 1: Necessity vs Sufficiency bar
    fig1 = go.Figure()
    test_names_display = {
        "necessity": "Necessity",
        "sufficiency": "Sufficiency",
        "generalization_held_out": "Generalization\n(Held-out)",
        "generalization_abb_template": "Generalization\n(ABB)",
        "generalization_bab_template": "Generalization\n(BAB)",
    }
    colors = ["#58A6FF", "#00C9A7", "#FFC75F", "#FF6B6B", "#845EC2"]

    for i, r in enumerate(validation_results):
        fig1.add_trace(go.Bar(
            name=test_names_display.get(r.test_name, r.test_name),
            x=[test_names_display.get(r.test_name, r.test_name)],
            y=[r.score],
            marker_color=colors[i % len(colors)],
            text=[f"{r.score:.3f}"],
            textposition="outside",
            textfont={"size": 14, "color": "white"},
        ))

    fig1.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                   annotation_text=" 0.5 threshold")

    fig1.update_layout(
        title={"text": f"Circuit Validation Results — {circuit.name} ({len(circuit)} heads)",
               "x": 0.5, "xanchor": "center",
               "font": {"size": 16, "family": _FONT}},
        template="plotly_dark",
        plot_bgcolor=_BG_PLOT, paper_bgcolor=_BG_PAPER,
        font={"family": _FONT},
        yaxis={"title": "Score", "range": [0, 1.1], "tickformat": ".0%"},
        xaxis={"title": "Validation Test"},
        barmode="group", showlegend=False,
        width=900, height=500,
    )
    save_figure(fig1, paths["figures_dir"] + "/23_circuit_validation_scores",
                formats=formats, width=900, height=500)

    # Plot 2: Baseline vs Experimental LD comparison
    fig2 = go.Figure()
    test_labels = [test_names_display.get(r.test_name, r.test_name)
                   for r in validation_results]
    fig2.add_trace(go.Bar(
        name="Baseline LD", x=test_labels,
        y=[r.baseline_ld for r in validation_results],
        marker_color="#58A6FF", opacity=0.85,
    ))
    fig2.add_trace(go.Bar(
        name="Experimental LD", x=test_labels,
        y=[r.experimental_ld for r in validation_results],
        marker_color="#FF6B6B", opacity=0.85,
    ))
    fig2.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
    fig2.update_layout(
        title={"text": "Logit Difference: Baseline vs. Experimental Condition",
               "x": 0.5, "xanchor": "center",
               "font": {"size": 16, "family": _FONT}},
        template="plotly_dark",
        plot_bgcolor=_BG_PLOT, paper_bgcolor=_BG_PAPER,
        font={"family": _FONT},
        barmode="group",
        yaxis={"title": "Logit Difference (IO − S)"},
        width=900, height=500,
    )
    save_figure(fig2, paths["figures_dir"] + "/24_validation_ld_comparison",
                formats=formats, width=900, height=500)

    logger.info("✓ Circuit validation experiment complete.")
    print(f"\n✓ Saved to {paths['results_dir']}/circuit_validation.csv")
    print(f"✓ Plots saved to {paths['figures_dir']}/")


if __name__ == "__main__":
    main()
