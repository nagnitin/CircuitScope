"""
experiments/07_full_pipeline.py
=================================
Master Runner: Execute the Complete IOI Mechanistic Interpretability Pipeline

This script orchestrates all 6 analysis parts sequentially with a single command.
It handles shared state (model, dataset, mean_z) efficiently — computing expensive
caches once and reusing them across experiments.

Pipeline:
  Part 0: Baseline IOI evaluation (from experiments/01)
  Part 1: Logit Lens
  Part 2: Layer Ablation
  Part 3: Head Ablation
  Part 4: Activation Patching (residual only for speed; full with --full-patching)
  Part 5: Path Patching
  Part 6: Final report generation

Recommended: Run on GPU for reasonable time.
  GPU (~T4):  ~60–90 minutes total
  CPU:        ~4–8 hours total (use --quick for short versions)

Usage:
  python experiments/07_full_pipeline.py
  python experiments/07_full_pipeline.py --quick   # n_samples=50, resid only
  python experiments/07_full_pipeline.py --full-patching  # all 3 patch types
  python experiments/07_full_pipeline.py --skip 3 5  # skip parts 3 and 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def run_pipeline(config: dict, quick: bool, full_patching: bool, skip_parts: set) -> dict:
    """
    Execute the full mechanistic interpretability pipeline.

    Parameters
    ----------
    config : dict
        Loaded YAML configuration.
    quick : bool
        If True, use reduced n_samples for all experiments.
    full_patching : bool
        If True, run all three patching experiments (resid + attn + MLP).
    skip_parts : set of int
        Part numbers to skip (1-5).

    Returns
    -------
    dict
        Summary of all results.
    """
    from src.utils.logger import get_logger, silence_external_loggers
    from src.utils.reproducibility import set_seed, configure_determinism
    from src.utils.io_utils import ensure_dirs, save_csv, save_json
    from src.model.loader import load_model
    from src.data.ioi_dataset import IOIDataset
    from src.evaluation.metrics import IOIEvaluator
    from src.analysis.logit_lens import LogitLensAnalyzer
    from src.analysis.layer_ablation import LayerAblationAnalyzer
    from src.analysis.head_ablation import HeadAblationAnalyzer
    from src.analysis.activation_patching import ActivationPatchingAnalyzer
    from src.analysis.path_patching import PathPatchingAnalyzer
    from src.visualization.plots import save_all_baseline_plots
    from src.visualization.circuit_vis import (
        plot_logit_lens_curve, plot_logit_lens_heatmap,
        plot_layer_ablation_bars, plot_layer_ablation_heatmap,
        plot_head_importance_heatmap, plot_head_ranking_bar,
        plot_activation_patching_heatmap, plot_all_patching_comparison,
        plot_sender_importance_heatmap, plot_circuit_graph,
        plot_attention_patterns,
    )

    silence_external_loggers()
    logger = get_logger(
        "circuitscope.pipeline",
        level="INFO",
        log_dir=config["paths"]["logs_dir"],
    )
    paths = config["paths"]
    # Per-experiment output subdirectories
    paths["figures_dir"] = paths["outputs_dir"] + "/07_full_pipeline/figures"
    paths["results_dir"] = paths["outputs_dir"] + "/07_full_pipeline/results"
    formats = config.get("plotting", {}).get("export_formats", ["html", "png"])
    seed = config.get("seed", 42)

    ensure_dirs(
        paths["figures_dir"], paths["results_dir"], paths["logs_dir"]
    )
    set_seed(seed)
    configure_determinism(enabled=True, warn_only=True)

    # ── Shared settings based on --quick flag ────────────────────────────
    n_ablation = 50 if quick else 200
    n_patching = 20 if quick else 50
    n_lens     = 100 if quick else 200

    summary = {
        "pipeline_start": datetime.now().isoformat(),
        "quick_mode": quick,
        "parts_run": [],
        "timing": {},
        "results": {},
    }

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    # --------------------------------------------------------------------
    # SHARED: Load model and dataset (used by all parts)
    # --------------------------------------------------------------------
    print("\n" + "="*60)
    print(" CircuitScope -- Full Pipeline")
    print("="*60)
    print(f" Mode: {'QUICK' if quick else 'FULL'}")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    logger.info("Loading GPT-2 Small…")
    t = time.time()
    model = load_model(
        config["model"]["name"],
        device=config["model"]["device"],
    )
    summary["timing"]["model_load"] = round(time.time() - t, 1)
    logger.info(f"Model loaded in {summary['timing']['model_load']}s")

    logger.info("Generating IOI dataset…")
    t = time.time()
    dataset = IOIDataset(
        model=model,
        n_prompts=config["dataset"]["n_prompts"],
        seed=seed,
    ).generate()
    summary["timing"]["dataset_gen"] = round(time.time() - t, 1)
    logger.info(f"Dataset generated: {len(dataset)} prompts in {summary['timing']['dataset_gen']}s")

    # ════════════════════════════════════════════════════════════════════
    # PART 0: Baseline Evaluation
    # ════════════════════════════════════════════════════════════════════
    print("\n[Part 0] Baseline IOI Evaluation…")
    t = time.time()
    evaluator = IOIEvaluator(model, dataset, batch_size=32)
    results_df = evaluator.evaluate()
    stats = evaluator.compute_aggregate_stats(results_df)
    save_csv(results_df, paths["results_csv"])
    save_all_baseline_plots(results_df, paths["figures_dir"], formats=formats)
    summary["results"]["baseline"] = {
        k: float(v) if v is not None else None for k, v in stats.items()
    }
    summary["timing"]["part0"] = round(time.time() - t, 1)
    print(f"  ✓ Accuracy: {stats['accuracy']:.1%}, Mean LD: {stats['mean_logit_diff']:+.4f}")
    print(f"  Time: {summary['timing']['part0']}s")

    # ════════════════════════════════════════════════════════════════════
    # PART 1: Logit Lens
    # ════════════════════════════════════════════════════════════════════
    if 1 not in skip_parts:
        print("\n[Part 1] Logit Lens…")
        t = time.time()
        lens_analyzer = LogitLensAnalyzer(model, dataset, n_samples=n_lens)
        lens_df = lens_analyzer.run(batch_size=10)
        save_csv(lens_df, paths["results_dir"] + "/logit_lens_by_layer.csv")

        pos_df = lens_analyzer.run_per_token_position(prompt_idx=0, layer=11)
        save_csv(pos_df, paths["results_dir"] + "/logit_lens_per_token.csv")

        plot_logit_lens_curve(lens_df,
            save_path=paths["figures_dir"] + "/06_logit_lens_curve", formats=formats)
        plot_logit_lens_heatmap(pos_df,
            prompt_str=dataset.prompts[0].prompt_clean,
            save_path=paths["figures_dir"] + "/07_logit_lens_token_heatmap", formats=formats)

        first_pos = lens_df[lens_df["logit_diff"] > 0]["layer_label"].values
        first_layer = first_pos[0] if len(first_pos) > 0 else "none"
        summary["results"]["logit_lens"] = {
            "first_positive_layer": first_layer,
            "max_logit_diff": float(lens_df["logit_diff"].max()),
            "max_layer": lens_df.loc[lens_df["logit_diff"].idxmax(), "layer_label"],
        }
        summary["timing"]["part1"] = round(time.time() - t, 1)
        summary["parts_run"].append(1)
        print(f"  ✓ IO preference first emerges at: {first_layer}")
        print(f"  Time: {summary['timing']['part1']}s")

    # ════════════════════════════════════════════════════════════════════
    # PART 2: Layer Ablation
    # ════════════════════════════════════════════════════════════════════
    if 2 not in skip_parts:
        print("\n[Part 2] Layer Ablation…")
        t = time.time()
        layer_analyzer = LayerAblationAnalyzer(model, dataset,
                                               n_samples=n_ablation, batch_size=16)
        mean_cache = layer_analyzer.compute_mean_cache()
        layer_df = layer_analyzer.run_full_sweep(mean_cache)
        save_csv(layer_df, paths["results_dir"] + "/layer_ablation.csv")

        plot_layer_ablation_bars(layer_df,
            save_path=paths["figures_dir"] + "/08_layer_ablation_bars", formats=formats)
        plot_layer_ablation_heatmap(layer_df,
            save_path=paths["figures_dir"] + "/09_layer_ablation_heatmap", formats=formats)

        n_critical = layer_df["is_critical"].sum()
        top_attn = int(layer_df[layer_df["component"]=="attn"].nlargest(1,"ld_drop_norm")["layer"].values[0])
        summary["results"]["layer_ablation"] = {
            "n_critical_components": int(n_critical),
            "top_attn_layer": top_attn,
        }
        summary["timing"]["part2"] = round(time.time() - t, 1)
        summary["parts_run"].append(2)
        print(f"  ✓ Critical components: {n_critical}, top attention layer: L{top_attn}")
        print(f"  Time: {summary['timing']['part2']}s")

    # ════════════════════════════════════════════════════════════════════
    # PART 3: Head Ablation (most expensive — 144 heads)
    # ════════════════════════════════════════════════════════════════════
    if 3 not in skip_parts:
        print("\n[Part 3] Head Ablation (144 heads)…")
        t = time.time()
        head_analyzer = HeadAblationAnalyzer(model, dataset,
                                             n_samples=n_ablation, batch_size=16)
        mean_z = head_analyzer.compute_mean_z()
        head_df = head_analyzer.run_full_sweep(mean_z)
        save_csv(head_df, paths["results_dir"] + "/head_ablation.csv")

        pivot_df = head_analyzer.pivot_importance_matrix(head_df)
        save_csv(pivot_df, paths["results_dir"] + "/head_importance_matrix.csv", index=True)

        plot_head_importance_heatmap(pivot_df,
            save_path=paths["figures_dir"] + "/10_head_importance_heatmap", formats=formats)
        plot_head_ranking_bar(head_df, top_n=20,
            save_path=paths["figures_dir"] + "/11_head_ranking_bar", formats=formats)

        name_movers = head_df[head_df["head_type"] == "Name Mover"]["head_label"].tolist()
        top_head = head_df.iloc[0]["head_label"] if not head_df.empty else "none"
        summary["results"]["head_ablation"] = {
            "top_head": top_head,
            "top_head_importance": float(head_df.iloc[0]["importance"]) if not head_df.empty else 0,
            "name_mover_heads": name_movers,
            "n_name_movers": len(name_movers),
        }
        summary["timing"]["part3"] = round(time.time() - t, 1)
        summary["parts_run"].append(3)
        print(f"  ✓ Top head: {top_head}, Name Mover heads: {name_movers}")
        print(f"  Time: {summary['timing']['part3']}s")
    else:
        head_df = None

    # ════════════════════════════════════════════════════════════════════
    # PART 4: Activation Patching
    # ════════════════════════════════════════════════════════════════════
    if 4 not in skip_parts:
        print("\n[Part 4] Activation Patching…")
        t = time.time()
        patch_analyzer = ActivationPatchingAnalyzer(model, dataset,
                                                    n_samples=n_patching, batch_size=1)
        resid_df = patch_analyzer.run_resid_patching()
        save_csv(resid_df, paths["results_dir"] + "/patching_resid.csv", index=True)
        plot_activation_patching_heatmap(resid_df,
            save_path=paths["figures_dir"] + "/12_patching_resid_heatmap", formats=formats)

        if full_patching:
            attn_df = patch_analyzer.run_attn_patching()
            mlp_df = patch_analyzer.run_mlp_patching()
            save_csv(attn_df, paths["results_dir"] + "/patching_attn.csv", index=True)
            save_csv(mlp_df, paths["results_dir"] + "/patching_mlp.csv", index=True)
            plot_activation_patching_heatmap(attn_df,
                save_path=paths["figures_dir"] + "/13_patching_attn_heatmap", formats=formats)
            plot_activation_patching_heatmap(mlp_df,
                save_path=paths["figures_dir"] + "/14_patching_mlp_heatmap", formats=formats)
            plot_all_patching_comparison(resid_df, attn_df, mlp_df,
                save_path=paths["figures_dir"] + "/15_patching_comparison", formats=formats)

        import numpy as np
        summary["results"]["activation_patching"] = {
            "resid_max_restoration": float(resid_df.values.max()),
        }
        summary["timing"]["part4"] = round(time.time() - t, 1)
        summary["parts_run"].append(4)
        print(f"  ✓ Max restoration (resid): {resid_df.values.max():.4f}")
        print(f"  Time: {summary['timing']['part4']}s")

    # ════════════════════════════════════════════════════════════════════
    # PART 5: Path Patching
    # ════════════════════════════════════════════════════════════════════
    if 5 not in skip_parts:
        print("\n[Part 5] Path Patching…")
        t = time.time()
        path_analyzer = PathPatchingAnalyzer(model, dataset,
                                             n_samples=n_patching,
                                             importance_threshold=0.05)
        sender_df = path_analyzer.run_sender_patching()
        graph_df = path_analyzer.build_circuit_graph(sender_df, top_n_senders=15)
        circuit_summary = path_analyzer.get_circuit_summary(sender_df)

        save_csv(sender_df, paths["results_dir"] + "/path_patching_senders.csv")
        if not graph_df.empty:
            save_csv(graph_df, paths["results_dir"] + "/circuit_graph_edges.csv")
        save_json(circuit_summary, paths["results_dir"] + "/circuit_summary.json")

        plot_sender_importance_heatmap(sender_df,
            save_path=paths["figures_dir"] + "/16_sender_importance_heatmap", formats=formats)
        if not graph_df.empty:
            plot_circuit_graph(graph_df, sender_df,
                save_path=paths["figures_dir"] + "/17_circuit_graph", formats=formats)

        # Attention patterns for top circuit heads
        top_circuit = sender_df[sender_df["is_circuit_node"]].head(5)
        sample_prompt = dataset.prompts[0].prompt_clean
        for fig_idx, (_, row) in enumerate(top_circuit.iterrows(), start=18):
            try:
                plot_attention_patterns(
                    model=model, prompt=sample_prompt,
                    layer=int(row["layer"]), head=int(row["head"]),
                    title=f"Attention — {row['head_label']} (score={row['restoration_score']:+.4f})",
                    save_path=paths["figures_dir"] + f"/{fig_idx:02d}_attn_{row['head_label']}",
                    formats=formats,
                )
            except Exception as exc:
                logger.warning(f"Attention pattern failed for {row['head_label']}: {exc}")

        summary["results"]["path_patching"] = {
            "n_circuit_nodes": circuit_summary["n_circuit_nodes"],
            "top_senders": [e["head"] for e in circuit_summary["top_heads"][:5]],
        }
        summary["timing"]["part5"] = round(time.time() - t, 1)
        summary["parts_run"].append(5)
        print(f"  ✓ Circuit nodes: {circuit_summary['n_circuit_nodes']}")
        print(f"  ✓ Top senders: {[e['head'] for e in circuit_summary['top_heads'][:5]]}")
        print(f"  Time: {summary['timing']['part5']}s")

    # ════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ════════════════════════════════════════════════════════════════════
    total_time = round(time.time() - pipeline_start, 1)
    summary["pipeline_end"] = datetime.now().isoformat()
    summary["total_time_seconds"] = total_time
    save_json(summary, paths["results_dir"] + "/pipeline_summary.json")

    # Count total files produced
    figs_dir = Path(paths["figures_dir"])
    results_dir = Path(paths["results_dir"])
    n_figs = len(list(figs_dir.glob("*.html"))) + len(list(figs_dir.glob("*.png")))
    n_csvs = len(list(results_dir.glob("*.csv")))
    n_jsons = len(list(results_dir.glob("*.json")))

    print("\n" + "="*60)
    print(" [OK] Full Pipeline Complete!")
    print("="*60)
    print(f" Total time      : {total_time}s ({total_time/60:.1f} min)")
    print(f" Parts executed  : {summary['parts_run']}")
    print(f" Figures saved   : {n_figs}")
    print(f" CSVs saved      : {n_csvs}")
    print(f" JSON files      : {n_jsons}")
    print(f"\n Key Results:")
    if "baseline" in summary["results"]:
        b = summary["results"]["baseline"]
        print(f"   Baseline accuracy    : {b.get('accuracy', 0):.1%}")
        print(f"   Mean logit diff      : {b.get('mean_logit_diff', 0):+.4f}")
    if "head_ablation" in summary["results"]:
        h = summary["results"]["head_ablation"]
        print(f"   Top head             : {h.get('top_head', 'N/A')}")
        print(f"   Name Mover heads     : {h.get('name_mover_heads', [])}")
    print("="*60)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CircuitScope: Full Mechanistic Interpretability Pipeline"
    )
    parser.add_argument("--config",
                        default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--quick", action="store_true",
                        help="Use small n_samples for all experiments (good for testing)")
    parser.add_argument("--full-patching", action="store_true",
                        help="Run all 3 activation patching types (resid + attn + MLP)")
    parser.add_argument("--skip", nargs="*", type=int, default=[],
                        metavar="PART",
                        help="Skip specified parts (e.g., --skip 3 5)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    try:
        run_pipeline(
            config=config,
            quick=args.quick,
            full_patching=args.full_patching,
            skip_parts=set(args.skip),
        )
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Pipeline stopped by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FATAL ERROR] {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
