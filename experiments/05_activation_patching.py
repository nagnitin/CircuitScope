"""
experiments/05_activation_patching.py
=======================================
Part 4: Activation Patching

Implements clean→corrupted activation patching at every (layer, token_position)
for three component types:
  - Residual stream (blocks.{l}.hook_resid_post)
  - Attention output (blocks.{l}.hook_attn_out)
  - MLP output (blocks.{l}.hook_mlp_out)

For each patch site, the restoration score = how much the clean-behaviour
is recovered by inserting the clean activation into the corrupted run.

Note: This is computationally intensive. Each prompt requires:
  - 2 forward passes with cache (clean + corrupted)
  - n_layers × seq_len forward passes with hooks (per patch experiment)
  - Total: ≈ n_samples × n_layers × seq_len × 3 forward passes

Recommended: --n-samples 30–50 on GPU, 10–20 on CPU.

Produces:
  outputs/results/patching_resid.csv
  outputs/results/patching_attn.csv
  outputs/results/patching_mlp.csv
  outputs/figures/12_patching_resid_heatmap.html/.png
  outputs/figures/13_patching_attn_heatmap.html/.png
  outputs/figures/14_patching_mlp_heatmap.html/.png
  outputs/figures/15_patching_comparison.html/.png

Usage:
  python experiments/05_activation_patching.py --n-samples 30
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation Patching Analysis")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of prompt pairs. 30–50 recommended for reasonable runtime.")
    parser.add_argument("--resid-only", action="store_true",
                        help="Only run residual stream patching (fastest)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.utils.logger import get_logger, silence_external_loggers
    from src.utils.reproducibility import set_seed
    from src.utils.io_utils import ensure_dirs, save_csv
    from src.model.loader import load_model
    from src.data.ioi_dataset import IOIDataset
    from src.analysis.activation_patching import ActivationPatchingAnalyzer
    from src.visualization.circuit_vis import (
        plot_activation_patching_heatmap,
        plot_all_patching_comparison,
    )

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger("circuitscope.patching", level=log_level,
                        log_dir=config["paths"]["logs_dir"])
    silence_external_loggers()

    paths = config["paths"]
    # Per-experiment output subdirectories
    paths["figures_dir"] = paths["outputs_dir"] + "/05_activation_patching/figures"
    paths["results_dir"] = paths["outputs_dir"] + "/05_activation_patching/results"
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

    # ── Run activation patching ───────────────────────────────────────────
    analyzer = ActivationPatchingAnalyzer(
        model, dataset,
        n_samples=args.n_samples,
        batch_size=1,  # must be 1 for per-prompt patching
    )

    formats = config.get("plotting", {}).get("export_formats", ["html", "png"])
    results: dict = {}

    # ── Part A: Residual stream patching ──────────────────────────────────
    logger.info(f"Running residual stream patching ({args.n_samples} prompts)…")
    t0 = time.time()
    resid_df = analyzer.run_resid_patching()
    logger.info(f"Residual patching complete in {time.time()-t0:.1f}s")
    save_csv(resid_df, paths["results_dir"] + "/patching_resid.csv", index=True)
    results["resid"] = resid_df

    plot_activation_patching_heatmap(
        resid_df,
        title="Residual Stream Patching: Restoration Score (Layer × Token Position)",
        save_path=paths["figures_dir"] + "/12_patching_resid_heatmap",
        formats=formats,
    )

    if not args.resid_only:
        # ── Part B: Attention output patching ─────────────────────────────
        logger.info("Running attention output patching…")
        t1 = time.time()
        attn_df = analyzer.run_attn_patching()
        logger.info(f"Attention patching complete in {time.time()-t1:.1f}s")
        save_csv(attn_df, paths["results_dir"] + "/patching_attn.csv", index=True)
        results["attn"] = attn_df

        plot_activation_patching_heatmap(
            attn_df,
            title="Attention Output Patching: Restoration Score (Layer × Token Position)",
            save_path=paths["figures_dir"] + "/13_patching_attn_heatmap",
            formats=formats,
        )

        # ── Part C: MLP output patching ───────────────────────────────────
        logger.info("Running MLP output patching…")
        t2 = time.time()
        mlp_df = analyzer.run_mlp_patching()
        logger.info(f"MLP patching complete in {time.time()-t2:.1f}s")
        save_csv(mlp_df, paths["results_dir"] + "/patching_mlp.csv", index=True)
        results["mlp"] = mlp_df

        plot_activation_patching_heatmap(
            mlp_df,
            title="MLP Output Patching: Restoration Score (Layer × Token Position)",
            save_path=paths["figures_dir"] + "/14_patching_mlp_heatmap",
            formats=formats,
        )

        # ── Part D: Combined comparison plot ──────────────────────────────
        logger.info("Generating comparison plot…")
        plot_all_patching_comparison(
            resid_df, attn_df, mlp_df,
            save_path=paths["figures_dir"] + "/15_patching_comparison",
            formats=formats,
        )

    # ── Report ────────────────────────────────────────────────────────────
    import numpy as np
    print("\n── Activation Patching Results ────────────────────────────────")
    resid_max = float(resid_df.values.max()) if not resid_df.empty else 0.0
    print(f"  Residual stream: max restoration = {resid_max:.4f}")
    if "attn" in results:
        attn_max = float(results["attn"].values.max())
        mlp_max = float(results["mlp"].values.max())
        print(f"  Attention out  : max restoration = {attn_max:.4f}")
        print(f"  MLP out        : max restoration = {mlp_max:.4f}")
        dominance = "attention" if attn_max > mlp_max else "MLP"
        print(f"  ➜ IOI circuit is primarily {dominance}-mediated")

    logger.info("✓ Activation patching experiment complete.")
    print(f"\n✓ Results saved to {paths['results_dir']}/")
    print(f"✓ Plots saved to {paths['figures_dir']}/")


if __name__ == "__main__":
    main()
