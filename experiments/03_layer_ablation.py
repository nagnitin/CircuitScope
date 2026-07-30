"""
experiments/03_layer_ablation.py
==================================
Part 2: Layer Ablation Experiment

Mean-ablates every transformer layer (attention, MLP, full) independently
and measures the resulting drop in IOI logit difference and accuracy.

Produces:
  outputs/results/layer_ablation.csv
  outputs/figures/08_layer_ablation_bars.html/.png
  outputs/figures/09_layer_ablation_heatmap.html/.png

Usage:
  python experiments/03_layer_ablation.py
  python experiments/03_layer_ablation.py --n-samples 150
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer Ablation Analysis")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.utils.logger import get_logger, silence_external_loggers
    from src.utils.reproducibility import set_seed
    from src.utils.io_utils import ensure_dirs, save_csv
    from src.model.loader import load_model
    from src.data.ioi_dataset import IOIDataset
    from src.analysis.layer_ablation import LayerAblationAnalyzer
    from src.visualization.circuit_vis import (
        plot_layer_ablation_bars,
        plot_layer_ablation_heatmap,
    )

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger("circuitscope.layer_ablation", level=log_level,
                        log_dir=config["paths"]["logs_dir"])
    silence_external_loggers()

    paths = config["paths"]
    # Per-experiment output subdirectories
    paths["figures_dir"] = paths["outputs_dir"] + "/03_layer_ablation/figures"
    paths["results_dir"] = paths["outputs_dir"] + "/03_layer_ablation/results"
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

    # ── Run layer ablation ────────────────────────────────────────────────
    logger.info(f"Running layer ablation sweep ({args.n_samples} prompts)…")
    t0 = time.time()

    analyzer = LayerAblationAnalyzer(
        model, dataset,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )

    # Step 1: Compute mean cache and clean cache
    logger.info("Computing mean activation cache for mean ablation…")
    mean_cache = analyzer.compute_mean_cache()

    logger.info("Computing clean activation cache for resample ablation control…")
    clean_cache = analyzer.compute_clean_cache()

    # Step 2: Run full sweep for both ablation modes (36 ablations each)
    logger.info("Running 36-ablation sweep (mean ablation)…")
    results_df = analyzer.run_full_sweep(mean_cache, ablation_mode="mean")

    logger.info("Running 36-ablation sweep (resample ablation control)…")
    resample_df = analyzer.run_full_sweep(ablation_mode="resample", clean_cache=clean_cache)

    elapsed = time.time() - t0
    logger.info(f"Layer ablation complete in {elapsed:.1f}s")

    # ── Report ────────────────────────────────────────────────────────────
    print("\n── Mean Layer Ablation Results ─────────────────────────────────────")
    critical = results_df[results_df["is_critical"]].sort_values(
        "ld_drop_norm", ascending=False
    )
    print(f"Critical components ({len(critical)} total):")
    print(critical[["layer", "component", "ld_drop_norm", "ablated_ld"]].to_string(index=False))

    print("\n── Resample Layer Ablation Control Results ─────────────────────────")
    res_critical = resample_df[resample_df["is_critical"]].sort_values(
        "ld_drop_norm", ascending=False
    )
    print(f"Critical components ({len(res_critical)} total):")
    print(res_critical[["layer", "component", "ld_drop_norm", "ablated_ld"]].to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    save_csv(results_df, paths["results_dir"] + "/layer_ablation.csv")
    save_csv(resample_df, paths["results_dir"] + "/layer_ablation_resample.csv")

    # ── Plots ─────────────────────────────────────────────────────────────
    formats = config.get("plotting", {}).get("export_formats", ["html", "png"])
    logger.info("Generating ablation plots…")

    plot_layer_ablation_bars(
        results_df,
        save_path=paths["figures_dir"] + "/08_layer_ablation_bars",
        formats=formats,
    )
    plot_layer_ablation_heatmap(
        results_df,
        save_path=paths["figures_dir"] + "/09_layer_ablation_heatmap",
        formats=formats,
    )

    logger.info("✓ Layer ablation experiment complete.")
    print(f"\n✓ Saved to {paths['results_dir']}/layer_ablation.csv")
    print(f"✓ Plots saved to {paths['figures_dir']}/")


if __name__ == "__main__":
    main()
