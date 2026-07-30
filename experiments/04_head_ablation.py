"""
experiments/04_head_ablation.py
=================================
Part 3: Attention Head Ablation

Mean-ablates each of the 144 attention heads independently and measures
causal importance for the IOI task. Produces a ranked CSV, a 12×12 heatmap,
and a horizontal bar chart of the top heads.

Expected runtime:
  GPU: ~10–20 min for 200 samples
  CPU: ~60–120 min for 200 samples (use --n-samples 50 for quick test)

Produces:
  outputs/results/head_ablation.csv          (144 rows, ranked)
  outputs/results/head_importance_matrix.csv  (12×12 pivot)
  outputs/figures/10_head_importance_heatmap.html/.png
  outputs/figures/11_head_ranking_bar.html/.png

Usage:
  python experiments/04_head_ablation.py
  python experiments/04_head_ablation.py --n-samples 50  # quick test
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention Head Ablation Analysis")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Prompts for mean_z and evaluation. Use 50 for quick test.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of heads to show in ranking bar chart")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.utils.logger import get_logger, silence_external_loggers
    from src.utils.reproducibility import set_seed
    from src.utils.io_utils import ensure_dirs, save_csv
    from src.model.loader import load_model
    from src.data.ioi_dataset import IOIDataset
    from src.analysis.head_ablation import HeadAblationAnalyzer
    from src.visualization.circuit_vis import (
        plot_head_importance_heatmap,
        plot_head_ranking_bar,
    )

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger("circuitscope.head_ablation", level=log_level,
                        log_dir=config["paths"]["logs_dir"])
    silence_external_loggers()

    paths = config["paths"]
    # Per-experiment output subdirectories
    paths["figures_dir"] = paths["outputs_dir"] + "/04_head_ablation/figures"
    paths["results_dir"] = paths["outputs_dir"] + "/04_head_ablation/results"
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

    # ── Run head ablation ─────────────────────────────────────────────────
    logger.info(
        f"Running head ablation sweep: 144 heads × {args.n_samples} prompts…\n"
        f"  This will take ~{args.n_samples * 144 // 200} minutes on GPU."
    )
    t0 = time.time()

    analyzer = HeadAblationAnalyzer(
        model, dataset,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )

    # Step 1: Compute mean z-vectors (reused for all head ablations)
    logger.info("Computing mean z-cache…")
    mean_z = analyzer.compute_mean_z()

    # Step 2: Full 144-head sweep
    logger.info("Running 144-head ablation sweep…")
    results_df = analyzer.run_full_sweep(mean_z)

    elapsed = time.time() - t0
    logger.info(f"Head ablation complete in {elapsed:.1f}s")

    # ── Report ────────────────────────────────────────────────────────────
    print("\n── Head Ablation Results ──────────────────────────────────────")
    print(f"\nTop 15 Most Important Heads:")
    top15 = results_df.head(15)
    print(top15[["rank", "head_label", "importance", "head_type"]].to_string(index=False))

    print(f"\nHead Type Distribution:")
    print(results_df["head_type"].value_counts().to_string())

    print(f"\nName Mover Heads (importance > 0.15):")
    name_movers = results_df[results_df["head_type"] == "Name Mover"]
    if not name_movers.empty:
        print(name_movers[["head_label", "importance"]].to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    save_csv(results_df, paths["results_dir"] + "/head_ablation.csv")

    # Pivot to 12×12 matrix and save
    pivot_df = analyzer.pivot_importance_matrix(results_df)
    save_csv(pivot_df, paths["results_dir"] + "/head_importance_matrix.csv", index=True)

    # ── Plots ─────────────────────────────────────────────────────────────
    formats = config.get("plotting", {}).get("export_formats", ["html", "png"])
    logger.info("Generating head ablation plots…")

    plot_head_importance_heatmap(
        pivot_df,
        save_path=paths["figures_dir"] + "/10_head_importance_heatmap",
        formats=formats,
    )
    plot_head_ranking_bar(
        results_df,
        top_n=args.top_n,
        save_path=paths["figures_dir"] + "/11_head_ranking_bar",
        formats=formats,
    )

    logger.info("✓ Head ablation experiment complete.")
    print(f"\n✓ Saved head_ablation.csv ({len(results_df)} heads ranked)")
    print(f"✓ Saved head_importance_matrix.csv (12×12)")
    print(f"✓ Plots saved to {paths['figures_dir']}/")


if __name__ == "__main__":
    main()
