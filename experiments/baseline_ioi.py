"""
experiments/baseline_ioi.py
=============================
Main experiment runner for the CircuitScope IOI baseline analysis.

This script orchestrates the complete end-to-end pipeline:

    1. Load configuration from config/experiment_config.yaml
    2. Set global random seed for reproducibility
    3. Set up structured logging (console + file)
    4. Create output directories
    5. Load GPT-2 Small via TransformerLens HookedTransformer
    6. Generate the IOI dataset (1000 clean + corrupted prompt pairs)
    7. Save the dataset to CSV
    8. Run the evaluation pipeline (logits, probs, logit-diff, top-k)
    9. Save evaluation results to CSV
   10. Compute and log aggregate statistics
   11. Save experiment metadata (config, stats) as JSON
   12. Generate and save all baseline visualization plots

Usage
-----
From the project root directory:

    # Activate your virtual environment first:
    python experiments/baseline_ioi.py

With a custom config:
    python experiments/baseline_ioi.py --config config/my_config.yaml

With verbosity:
    python experiments/baseline_ioi.py --verbose

On Google Colab:
    !python experiments/baseline_ioi.py

Expected Runtime
----------------
  GPU (CUDA)  : ~3-5 minutes for 1000 prompts
  CPU         : ~15-30 minutes for 1000 prompts
  Apple M1/M2 : ~5-10 minutes

Expected Output
---------------
outputs/
├── figures/
│   ├── 01_accuracy_bar.html + .png
│   ├── 02_logit_diff_histogram.html + .png
│   ├── 03_confidence_histogram.html + .png
│   ├── 04_top_k_distribution.html + .png
│   └── 05_logit_diff_by_template.html + .png
├── results/
│   ├── ioi_dataset.csv          (1000 rows × 11 columns)
│   ├── ioi_results.csv          (1000 rows × 21 columns)
│   └── experiment_metadata.json
└── logs/
    └── circuitscope_YYYYMMDD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any

# ── Ensure project root is in Python path ─────────────────────────────────
# This allows `python experiments/baseline_ioi.py` to work without installing
# the package, as long as the script is run from the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def load_config(config_path: str) -> dict:
    """
    Load the YAML experiment configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.resolve()}\n"
            f"Expected at: {PROJECT_ROOT / 'config' / 'experiment_config.yaml'}"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the baseline experiment.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace with attributes:
        - config : str path to YAML config
        - verbose : bool for DEBUG logging
        - no_plots: bool to skip plotting (faster for CI)
    """
    parser = argparse.ArgumentParser(
        description=(
            "CircuitScope: IOI Baseline Evaluation\n"
            "Runs the complete IOI evaluation pipeline on GPT-2 Small."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"),
        help="Path to YAML configuration file (default: config/experiment_config.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging (very verbose)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip generating visualization plots (useful for faster CI runs)",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=None,
        help="Override dataset size from config (e.g., --n-prompts 100 for quick test)",
    )
    return parser.parse_args()


def run_experiment(config: dict, verbose: bool = False, generate_plots: bool = True) -> dict[str, Any]:
    """
    Execute the complete IOI baseline experiment.

    This function is the heart of the script. It is designed to be callable
    from both the command line (via `main()`) and from a Jupyter notebook
    (by importing and calling directly).

    Parameters
    ----------
    config : dict
        Experiment configuration (loaded from YAML).

    verbose : bool
        If True, use DEBUG logging level.

    generate_plots : bool
        If True, generate and save all baseline plots.

    Returns
    -------
    dict
        Summary dictionary with keys:
        - "results_df" : pd.DataFrame (full evaluation results)
        - "dataset"    : IOIDataset
        - "model"      : HookedTransformer
        - "stats"      : dict (aggregate statistics)
        - "saved_paths": dict (all saved file paths)
    """
    # ── Step 0: Imports (inside function so Colab cell imports are clean) ──
    import torch
    import pandas as pd

    from src.utils.logger import get_logger, silence_external_loggers
    from src.utils.reproducibility import set_seed, configure_determinism, get_reproducibility_state
    from src.utils.io_utils import ensure_dirs, save_csv, save_json
    from src.model.loader import load_model, get_tokenizer_info
    from src.data.ioi_dataset import IOIDataset
    from src.evaluation.metrics import IOIEvaluator
    from src.visualization.plots import save_all_baseline_plots

    # ── Step 1: Set up logging ─────────────────────────────────────────────
    log_level = "DEBUG" if verbose else config.get("logging", {}).get("level", "INFO")
    log_dir = Path(config["paths"]["logs_dir"])

    logger = get_logger(
        "circuitscope.experiment",
        level=log_level,
        log_dir=log_dir,
        console=config.get("logging", {}).get("console", True),
        file=config.get("logging", {}).get("file", True),
    )

    # Silence noisy external libraries
    silence_external_loggers()

    logger.info("=" * 60)
    logger.info("CircuitScope: IOI Baseline Experiment")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ── Step 2: Create output directories ────────────────────────────────
    paths = config["paths"]
    # Per-experiment output subdirectories
    paths["figures_dir"] = paths["outputs_dir"] + "/01_baseline/figures"
    paths["results_dir"] = paths["outputs_dir"] + "/01_baseline/results"
    ensure_dirs(
        paths["figures_dir"],
        paths["results_dir"],
        paths["logs_dir"],
    )
    logger.info(f"[Setup] Output directories created under: {paths['outputs_dir']}/")

    # ── Step 3: Set random seed + determinism ────────────────────────────
    seed = config.get("seed", 42)
    set_seed(seed)
    configure_determinism(enabled=True, warn_only=True)
    repro_state = get_reproducibility_state()
    logger.info(f"[Reproducibility] Seed={seed}, state={repro_state}")

    # ── Step 4: Load GPT-2 Small ──────────────────────────────────────────
    logger.info("[Model] Loading GPT-2 Small via TransformerLens…")
    t0 = time.time()

    model_cfg = config.get("model", {})
    model = load_model(
        model_name=model_cfg.get("name", "gpt2"),
        device=model_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "float32"),
        cache_dir=model_cfg.get("cache_dir", None),
    )

    load_time = time.time() - t0
    logger.info(f"[Model] ✓ Loaded in {load_time:.1f}s")

    tokenizer_info = get_tokenizer_info(model)
    logger.info(f"[Tokenizer] Metadata: {tokenizer_info}")

    # ── Step 5: Generate IOI Dataset ──────────────────────────────────────
    logger.info("[Dataset] Generating IOI dataset…")
    t1 = time.time()

    dataset_cfg = config.get("dataset", {})
    n_prompts = dataset_cfg.get("n_prompts", 1000)

    dataset = IOIDataset(
        model=model,
        n_prompts=n_prompts,
        seed=seed,
    )
    dataset.generate()

    gen_time = time.time() - t1
    logger.info(f"[Dataset] ✓ Generated {len(dataset):,} prompts in {gen_time:.1f}s")
    print(dataset.summary())

    # Save dataset CSV
    dataset_csv_path = save_csv(dataset.df, paths["dataset_csv"])
    logger.info(f"[Dataset] ✓ Saved to {dataset_csv_path}")

    # ── Step 6: Evaluate (clean prompts) ─────────────────────────────────
    logger.info("[Evaluation] Running IOI evaluation on clean prompts…")
    t2 = time.time()

    eval_cfg = config.get("evaluation", {})
    evaluator = IOIEvaluator(
        model=model,
        dataset=dataset,
        batch_size=eval_cfg.get("batch_size", 32),
        top_k=eval_cfg.get("top_k", 5),
    )

    results_df = evaluator.evaluate(use_corrupted=False)
    eval_time = time.time() - t2

    logger.info(
        f"[Evaluation] ✓ Complete in {eval_time:.1f}s\n"
        f"  Shape: {results_df.shape}\n"
        f"  Columns: {results_df.columns.tolist()}"
    )

    # ── Step 7: Evaluate (corrupted prompts) ─────────────────────────────
    logger.info("[Evaluation] Running IOI evaluation on corrupted prompts…")
    corrupted_df = evaluator.evaluate(use_corrupted=True)

    # Combine clean + corrupted results
    all_results_df = pd.concat([results_df, corrupted_df], ignore_index=True)

    # Save results CSV (clean only is the primary output)
    results_csv_path = save_csv(results_df, paths["results_csv"])
    logger.info(f"[Evaluation] ✓ Results saved to {results_csv_path}")

    # Also save corrupted results
    corrupted_csv_path = save_csv(
        corrupted_df,
        str(paths["results_dir"]) + "/ioi_corrupted_results.csv",
    )

    # ── Step 8: Compute aggregate statistics ─────────────────────────────
    logger.info("[Stats] Computing aggregate statistics…")
    stats = evaluator.compute_aggregate_stats(results_df)

    logger.info(
        "\n" + "=" * 60 + "\n"
        "BASELINE RESULTS SUMMARY\n"
        + "=" * 60 + "\n"
        f"  Model               : GPT-2 Small\n"
        f"  Dataset size        : {stats['n_prompts']:,} prompts\n"
        f"  ─────────────────────────────────────\n"
        f"  Overall Accuracy    : {stats['accuracy']:.1%}\n"
        f"  ABB Accuracy        : {stats.get('accuracy_abb', 'N/A')}\n"
        f"  BAB Accuracy        : {stats.get('accuracy_bab', 'N/A')}\n"
        f"  ─────────────────────────────────────\n"
        f"  Mean Logit Diff     : {stats['mean_logit_diff']:+.4f}\n"
        f"  Std Logit Diff      : {stats['std_logit_diff']:.4f}\n"
        f"  Median Logit Diff   : {stats['median_logit_diff']:+.4f}\n"
        f"  ─────────────────────────────────────\n"
        f"  Mean P(IO)          : {stats['mean_prob_io']:.4f}\n"
        f"  Mean IO Rank        : {stats['mean_rank_io']:.1f} / 50,257\n"
        + "=" * 60
    )

    # ── Step 9: Save metadata JSON ────────────────────────────────────────
    metadata = {
        "experiment": "IOI Baseline",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "reproducibility": repro_state,
        "tokenizer_info": tokenizer_info,
        "results": {k: (float(v) if v is not None else None) for k, v in stats.items()},
        "timing": {
            "model_load_seconds": round(load_time, 2),
            "dataset_gen_seconds": round(gen_time, 2),
            "eval_seconds": round(eval_time, 2),
            "total_seconds": round(time.time() - t0, 2),
        },
    }

    metadata_path = save_json(
        metadata,
        Path(paths["results_dir"]) / "experiment_metadata.json",
    )

    # ── Step 10: Generate plots ───────────────────────────────────────────
    saved_plot_paths = {}
    if generate_plots:
        logger.info("[Plots] Generating all baseline plots…")
        t3 = time.time()

        plot_cfg = config.get("plotting", {})
        saved_plot_paths = save_all_baseline_plots(
            results_df=results_df,
            figures_dir=paths["figures_dir"],
            formats=plot_cfg.get("export_formats", ["html", "png"]),
        )

        plot_time = time.time() - t3
        logger.info(f"[Plots] ✓ All plots saved in {plot_time:.1f}s")
    else:
        logger.info("[Plots] Skipped (--no-plots flag set).")

    # ── Final summary ─────────────────────────────────────────────────────
    total_time = time.time() - t0
    logger.info(
        f"\n✓ Experiment complete in {total_time:.1f}s.\n"
        f"  Results CSV    : {results_csv_path}\n"
        f"  Dataset CSV    : {dataset_csv_path}\n"
        f"  Metadata JSON  : {metadata_path}\n"
        f"  Figures dir    : {Path(paths['figures_dir']).resolve()}"
    )

    return {
        "results_df": results_df,
        "corrupted_df": corrupted_df,
        "dataset": dataset,
        "model": model,
        "stats": stats,
        "saved_paths": {
            "results_csv": str(results_csv_path),
            "dataset_csv": str(dataset_csv_path),
            "metadata_json": str(metadata_path),
            "plots": saved_plot_paths,
        },
    }


def main() -> None:
    """
    Command-line entry point for the IOI baseline experiment.

    Parses CLI arguments, loads config, and runs the full pipeline.
    Handles keyboard interrupts and unexpected errors gracefully.
    """
    args = parse_args()

    print(
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        "║  CircuitScope: Mechanistic Interpretability Research ║\n"
        "║  GPT-2 Small — IOI Baseline Experiment               ║\n"
        "╚══════════════════════════════════════════════════════╝\n"
    )

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Override n_prompts if provided via CLI
    if args.n_prompts is not None:
        config["dataset"]["n_prompts"] = args.n_prompts
        print(f"[Override] Dataset size set to {args.n_prompts} prompts.")

    # Run experiment
    try:
        results = run_experiment(
            config=config,
            verbose=args.verbose,
            generate_plots=not args.no_plots,
        )
        print("\n✓ Experiment completed successfully.")
        print(f"  Accuracy : {results['stats']['accuracy']:.1%}")
        print(f"  Mean LD  : {results['stats']['mean_logit_diff']:+.4f}")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Experiment stopped by user.")
        sys.exit(0)

    except Exception as exc:
        print(f"\n[FATAL ERROR] {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
