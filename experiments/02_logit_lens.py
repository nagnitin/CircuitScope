"""
experiments/02_logit_lens.py
==============================
Part 1: Logit Lens Analysis

This script projects the residual stream at every transformer layer into
vocabulary space (using LayerNorm_final + W_U) and tracks when the model
develops a preference for the IO over the S token.

Produces:
  outputs/results/logit_lens_by_layer.csv
  outputs/results/logit_lens_per_token.csv
  outputs/figures/06_logit_lens_curve.html/.png
  outputs/figures/07_logit_lens_heatmap.html/.png

Usage:
  python experiments/02_logit_lens.py
  python experiments/02_logit_lens.py --n-samples 100
  python experiments/02_logit_lens.py --verbose
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Logit Lens Analysis")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prompt-idx", type=int, default=0,
                        help="Prompt index for per-token position analysis")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.utils.logger import get_logger, silence_external_loggers
    from src.utils.reproducibility import set_seed
    from src.utils.io_utils import ensure_dirs, save_csv, save_json
    from src.model.loader import load_model
    from src.data.ioi_dataset import IOIDataset
    from src.analysis.logit_lens import LogitLensAnalyzer
    from src.visualization.circuit_vis import (
        plot_logit_lens_curve,
        plot_logit_lens_heatmap,
    )

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger("circuitscope.logit_lens", level=log_level,
                        log_dir=config["paths"]["logs_dir"])
    silence_external_loggers()

    paths = config["paths"]
    # Per-experiment output subdirectories
    paths["figures_dir"] = paths["outputs_dir"] + "/02_logit_lens/figures"
    paths["results_dir"] = paths["outputs_dir"] + "/02_logit_lens/results"
    ensure_dirs(paths["figures_dir"], paths["results_dir"], paths["logs_dir"])
    set_seed(config.get("seed", 42))

    # ── Load model & dataset ──────────────────────────────────────────────
    logger.info("Loading model…")
    model = load_model(
        config["model"]["name"],
        device=config["model"]["device"],
    )

    logger.info("Generating dataset…")
    dataset = IOIDataset(
        model=model,
        n_prompts=config["dataset"]["n_prompts"],
        seed=config.get("seed", 42),
    ).generate()

    # ── Run logit lens by layer ───────────────────────────────────────────
    logger.info(f"Running logit lens over {args.n_samples} prompts…")
    t0 = time.time()

    analyzer = LogitLensAnalyzer(model, dataset, n_samples=args.n_samples)
    lens_df = analyzer.run(batch_size=args.batch_size)

    logger.info(f"Logit lens complete in {time.time()-t0:.1f}s")
    print("\n── Logit Lens Results ─────────────────────────────────────────")
    print(lens_df[["layer_label", "logit_diff", "prob_io", "fraction_correct"]].to_string(index=False))

    # Save CSV
    save_csv(lens_df, paths["results_dir"] + "/logit_lens_by_layer.csv")

    # ── Per-token position analysis ───────────────────────────────────────
    logger.info(f"Running per-token logit lens for prompt #{args.prompt_idx}…")
    pos_df = analyzer.run_per_token_position(
        prompt_idx=args.prompt_idx,
        layer=model.cfg.n_layers - 1,  # final layer
    )
    save_csv(pos_df, paths["results_dir"] + "/logit_lens_per_token.csv")

    prompt_str = dataset.prompts[args.prompt_idx].prompt_clean
    logger.info(f"Sample prompt: {prompt_str!r}")
    print("\n── Per-Token Logit Lens (Final Layer) ──────────────────────────")
    print(pos_df[["position", "token_str", "logit_diff", "prob_io"]].to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────
    formats = config.get("plotting", {}).get("export_formats", ["html", "png"])

    logger.info("Generating logit lens plots…")
    plot_logit_lens_curve(
        lens_df,
        save_path=paths["figures_dir"] + "/06_logit_lens_curve",
        formats=formats,
    )
    plot_logit_lens_heatmap(
        pos_df,
        prompt_str=prompt_str,
        save_path=paths["figures_dir"] + "/07_logit_lens_token_heatmap",
        formats=formats,
    )

    # Summary
    first_positive = lens_df[lens_df["logit_diff"] > 0]
    if not first_positive.empty:
        first_layer = first_positive.iloc[0]["layer_label"]
        logger.info(f"✓ IO preference first emerges at: {first_layer}")
    logger.info("✓ Logit lens experiment complete.")
    print(f"\n✓ Saved plots to {paths['figures_dir']}/")
    print(f"✓ Saved CSVs to {paths['results_dir']}/")


if __name__ == "__main__":
    main()
