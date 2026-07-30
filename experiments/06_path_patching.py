"""
experiments/06_path_patching.py
=================================
Part 5: Path Patching — Circuit Tracing

Runs sender-side path patching to identify which attention heads carry
IOI-relevant information (senders). Then builds a circuit graph of the
most important heads and their estimated information-flow edges.

Produces:
  outputs/results/path_patching_senders.csv
  outputs/results/circuit_graph_edges.csv
  outputs/results/circuit_summary.json
  outputs/figures/16_sender_importance_heatmap.html/.png
  outputs/figures/17_circuit_graph.html/.png

Also generates attention pattern visualizations for the top-5 identified
circuit heads.

Usage:
  python experiments/06_path_patching.py
  python experiments/06_path_patching.py --n-samples 30
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Path Patching / Circuit Analysis")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Prompt pairs for path patching. Each requires 2+N_heads forward passes.")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Minimum restoration score to classify as circuit node")
    parser.add_argument("--top-n-senders", type=int, default=15)
    parser.add_argument("--attn-layers", nargs="+", type=int, default=None,
                        help="Layers for attention pattern visualization. Default: top-5 circuit heads")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.utils.logger import get_logger, silence_external_loggers
    from src.utils.reproducibility import set_seed
    from src.utils.io_utils import ensure_dirs, save_csv, save_json
    from src.model.loader import load_model
    from src.data.ioi_dataset import IOIDataset
    from src.analysis.path_patching import PathPatchingAnalyzer
    from src.visualization.circuit_vis import (
        plot_sender_importance_heatmap,
        plot_circuit_graph,
        plot_attention_patterns,
    )

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger("circuitscope.path_patching", level=log_level,
                        log_dir=config["paths"]["logs_dir"])
    silence_external_loggers()

    paths = config["paths"]
    # Per-experiment output subdirectories
    paths["figures_dir"] = paths["outputs_dir"] + "/06_path_patching/figures"
    paths["results_dir"] = paths["outputs_dir"] + "/06_path_patching/results"
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

    # ── Run sender path patching ──────────────────────────────────────────
    logger.info(
        f"Running sender path patching: "
        f"{args.n_samples} prompts × 144 heads = "
        f"{args.n_samples * 144:,} forward passes…"
    )
    t0 = time.time()

    analyzer = PathPatchingAnalyzer(
        model, dataset,
        n_samples=args.n_samples,
        importance_threshold=args.threshold,
    )

    sender_df = analyzer.run_sender_patching()
    elapsed = time.time() - t0
    logger.info(f"Sender patching complete in {elapsed:.1f}s")

    # ── Build circuit graph ───────────────────────────────────────────────
    logger.info("Building circuit graph…")
    graph_df = analyzer.build_circuit_graph(
        sender_df, top_n_senders=args.top_n_senders
    )

    circuit_summary = analyzer.get_circuit_summary(sender_df)

    # ── Report ────────────────────────────────────────────────────────────
    print("\n── Path Patching / Circuit Analysis ────────────────────────────")
    print(f"\nCircuit Summary:")
    print(f"  Total circuit nodes  : {circuit_summary['n_circuit_nodes']}")
    print(f"  Early layers (0–4)   : {circuit_summary['n_early_heads']} heads")
    print(f"  Middle layers (5–8)  : {circuit_summary['n_middle_heads']} heads")
    print(f"  Late layers (9–11)   : {circuit_summary['n_late_heads']} heads")
    print(f"\n  Top circuit heads:")
    for entry in circuit_summary["top_heads"][:10]:
        print(f"    {entry['head']}: restoration = {entry['score']:+.4f}")
    print(f"\n  Late-layer heads (likely Name Movers): {circuit_summary['late_circuit_heads']}")
    print(f"  Middle-layer heads (likely S-Inhibition): {circuit_summary['middle_circuit_heads']}")

    if not graph_df.empty:
        print(f"\n  Circuit edges: {len(graph_df)}")
        print(f"  Top 5 edges by weight:")
        print(graph_df[["source_label", "target_label", "estimated_edge_weight"]].head(5).to_string(index=False))

    # ── Save results ──────────────────────────────────────────────────────
    save_csv(sender_df, paths["results_dir"] + "/path_patching_senders.csv")
    if not graph_df.empty:
        save_csv(graph_df, paths["results_dir"] + "/circuit_graph_edges.csv")
    save_json(circuit_summary, paths["results_dir"] + "/circuit_summary.json")

    # ── Plots ─────────────────────────────────────────────────────────────
    formats = config.get("plotting", {}).get("export_formats", ["html", "png"])
    logger.info("Generating circuit visualization plots…")

    plot_sender_importance_heatmap(
        sender_df,
        save_path=paths["figures_dir"] + "/16_sender_importance_heatmap",
        formats=formats,
    )

    if not graph_df.empty:
        plot_circuit_graph(
            graph_df, sender_df,
            save_path=paths["figures_dir"] + "/17_circuit_graph",
            formats=formats,
        )

    # ── Attention patterns for top circuit heads ───────────────────────────
    top_circuit_heads = sender_df[sender_df["is_circuit_node"]].head(5)
    sample_prompt = dataset.prompts[0].prompt_clean
    logger.info(f"Visualizing attention patterns for top {len(top_circuit_heads)} circuit heads…")

    for fig_idx, (_, row) in enumerate(top_circuit_heads.iterrows(), start=18):
        layer = int(row["layer"])
        head = int(row["head"])
        label = row["head_label"]

        try:
            fig = plot_attention_patterns(
                model=model,
                prompt=sample_prompt,
                layer=layer,
                head=head,
                title=f"Attention Pattern — {label} (restoration={row['restoration_score']:+.4f})",
                save_path=paths["figures_dir"] + f"/{fig_idx:02d}_attn_{label}",
                formats=formats,
            )
            logger.info(f"  ✓ Saved attention pattern for {label}")
        except Exception as e:
            logger.warning(f"  Attention pattern for {label} failed: {e}")

    logger.info("✓ Path patching experiment complete.")
    print(f"\n✓ Saved results to {paths['results_dir']}/")
    print(f"✓ Plots saved to {paths['figures_dir']}/")


if __name__ == "__main__":
    main()
