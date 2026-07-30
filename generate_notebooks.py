import json
from pathlib import Path

OUT = Path("notebooks")
OUT.mkdir(exist_ok=True)

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

def notebook(cells, title="CircuitScope"):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4", "name": title},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }

def save_nb(name, nb):
    p = OUT / name
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  created: {p}")

SETUP = '''import os, sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.isdir("CircuitScope"):
        os.system("git clone https://github.com/nagnitin/CircuitScope.git")
    os.chdir("CircuitScope")
    os.system("pip install -q -r requirements.txt")
    print("Colab setup complete.")
else:
    ROOT = os.path.abspath(".")
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    print("Local setup complete.")
'''

CFG = '''import yaml
with open("config/experiment_config.yaml") as f:
    config = yaml.safe_load(f)
from src.utils.logger import get_logger, silence_external_loggers
from src.utils.reproducibility import set_seed
from src.utils.io_utils import ensure_dirs
silence_external_loggers()
set_seed(config.get("seed", 42))
print("Config loaded | seed:", config.get("seed", 42))
'''

MODEL = '''from src.model.loader import load_model
model = load_model(config["model"]["name"], device=config["model"]["device"])
print(f"Loaded {config['model']['name']}")
'''

DATA = '''from src.data.ioi_dataset import IOIDataset
dataset = IOIDataset(model=model, n_prompts=config["dataset"]["n_prompts"], seed=config.get("seed",42)).generate()
print(f"Dataset: {len(dataset)} prompts")
'''

# ── 01 Baseline ───────────────────────────────────────────────────────────────
cells_01 = [
    md("# Experiment 01 — Baseline IOI Evaluation\n**CircuitScope · GPT-2 Small**\n\nEvaluates GPT-2 on 1,000 IOI prompts and records accuracy, logit difference, and vocabulary rank.\n\n**Runtime:** ~3 min (GPU) / ~15 min (CPU)"),
    md("## 1 · Environment Setup"), code(SETUP),
    md("## 2 · Config & Seed"),     code(CFG),
    md("## 3 · Load GPT-2 Small"),  code(MODEL),
    md("## 4 · Generate IOI Dataset"), code(DATA),
    md("## 5 · Run Evaluation"),
    code('''from src.evaluation.metrics import IOIEvaluator
evaluator = IOIEvaluator(model=model, dataset=dataset, batch_size=32, top_k=5)
results_df = evaluator.evaluate(use_corrupted=False)
stats = evaluator.compute_aggregate_stats(results_df)
print(f"""
Results
=======
Accuracy   : {stats["accuracy"]:.1%}
Mean LD    : {stats["mean_logit_diff"]:+.4f}
Std LD     : {stats["std_logit_diff"]:.4f}
Mean P(IO) : {stats["mean_prob_io"]:.4f}
IO Rank    : {stats["mean_rank_io"]:.1f} / 50,257
""")
'''),
    md("## 6 · Save Results & Figures"),
    code('''from src.utils.io_utils import save_csv, save_json, ensure_dirs
from src.visualization.plots import save_all_baseline_plots
from datetime import datetime
paths = config["paths"]
paths["figures_dir"] = paths["outputs_dir"] + "/01_baseline/figures"
paths["results_dir"] = paths["outputs_dir"] + "/01_baseline/results"
ensure_dirs(paths["figures_dir"], paths["results_dir"])
save_csv(results_df, paths["results_dir"] + "/ioi_results.csv")
save_json({"timestamp": datetime.now().isoformat(), "stats": {k: (float(v) if v else None) for k,v in stats.items()}}, paths["results_dir"] + "/experiment_metadata.json")
save_all_baseline_plots(results_df, paths["figures_dir"], formats=["html","png"])
print("Saved to", paths["results_dir"])
'''),
    md("## 7 · Quick Inspect"),
    code('''print("Top 5 easiest prompts:")
print(results_df.nlargest(5,"logit_diff")[["io_name","s_name","logit_diff","rank_io"]].to_string())
print("\\nTop 5 hardest prompts:")
print(results_df.nsmallest(5,"logit_diff")[["io_name","s_name","logit_diff","is_correct"]].to_string())
'''),
]
save_nb("01_baseline_ioi.ipynb", notebook(cells_01, "01 Baseline IOI"))

# ── 02 Logit Lens ─────────────────────────────────────────────────────────────
cells_02 = [
    md("# Experiment 02 — Logit Lens\n**CircuitScope · GPT-2 Small**\n\nProjects each layer's residual stream into vocabulary space. Reveals *when* the model develops a preference for the correct IO token.\n\n**Runtime:** ~5 min (GPU)"),
    md("## 1 · Setup"), code(SETUP), code(CFG),
    md("## 2 · Load Model & Data"), code(MODEL), code(DATA),
    md("## 3 · Run Logit Lens"),
    code('''import time
from src.analysis.logit_lens import LogitLensAnalyzer
N_SAMPLES = 200  # reduce to 50 for quick test
t0 = time.time()
analyzer = LogitLensAnalyzer(model, dataset, n_samples=N_SAMPLES)
lens_df = analyzer.run(batch_size=10)
print(f"Done in {time.time()-t0:.1f}s")
print(lens_df[["layer_label","logit_diff","prob_io","fraction_correct"]].to_string())
'''),
    md("## 4 · When Does IO Preference Emerge?"),
    code('''first_pos = lens_df[lens_df["logit_diff"] > 0]
layer = first_pos.iloc[0]["layer_label"] if not first_pos.empty else "N/A"
print(f"IO preference first emerges at: {layer}")
'''),
    md("## 5 · Per-Token Analysis"),
    code('''pos_df = analyzer.run_per_token_position(prompt_idx=0, layer=model.cfg.n_layers-1)
sample = dataset.prompts[0].prompt_clean
print("Prompt:", sample)
print(pos_df[["position","token_str","logit_diff","prob_io"]].to_string())
'''),
    md("## 6 · Save & Plot"),
    code('''from src.utils.io_utils import save_csv, ensure_dirs
from src.visualization.circuit_vis import plot_logit_lens_curve, plot_logit_lens_heatmap
paths = config["paths"]
paths["figures_dir"] = paths["outputs_dir"] + "/02_logit_lens/figures"
paths["results_dir"] = paths["outputs_dir"] + "/02_logit_lens/results"
ensure_dirs(paths["figures_dir"], paths["results_dir"])
save_csv(lens_df, paths["results_dir"] + "/logit_lens_by_layer.csv")
save_csv(pos_df,  paths["results_dir"] + "/logit_lens_per_token.csv")
plot_logit_lens_curve(lens_df, save_path=paths["figures_dir"]+"/06_logit_lens_curve", formats=["html","png"])
plot_logit_lens_heatmap(pos_df, prompt_str=sample, save_path=paths["figures_dir"]+"/07_logit_lens_token_heatmap", formats=["html","png"])
print("Saved to", paths["figures_dir"])
'''),
]
save_nb("02_logit_lens.ipynb", notebook(cells_02, "02 Logit Lens"))

# ── 03 Layer Ablation ─────────────────────────────────────────────────────────
cells_03 = [
    md("# Experiment 03 — Layer Ablation\n**CircuitScope · GPT-2 Small**\n\nMean-ablates each layer's attention output, MLP output, and both combined.\nIdentifies which layers are causally necessary for IOI behavior.\n\n**Runtime:** ~10 min (GPU)"),
    md("## 1 · Setup"), code(SETUP), code(CFG),
    md("## 2 · Load Model & Data"), code(MODEL), code(DATA),
    md("## 3 · Compute Mean Cache"),
    code('''from src.analysis.layer_ablation import LayerAblationAnalyzer
N_SAMPLES = 200
analyzer = LayerAblationAnalyzer(model, dataset, n_samples=N_SAMPLES, batch_size=16)
print("Computing mean activation cache…")
mean_cache = analyzer.compute_mean_cache()
print("Mean cache ready.")
'''),
    md("## 4 · Run 36-Ablation Sweep (12 layers x 3 components)"),
    code('''import time
t0 = time.time()
results_df = analyzer.run_full_sweep(mean_cache)
print(f"Done in {time.time()-t0:.1f}s")
'''),
    md("## 5 · Results"),
    code('''critical = results_df[results_df["is_critical"]].sort_values("ld_drop_norm", ascending=False)
print(f"Critical components ({len(critical)}):")
print(critical[["layer","component","ld_drop_norm","ablated_ld"]].to_string())
attn = results_df[results_df["component"]=="attn"]
mlp  = results_df[results_df["component"]=="mlp"]
print("\\nTop 3 critical attention layers:", attn.nlargest(3,"ld_drop_norm")["layer"].tolist())
print("Top 3 critical MLP layers:", mlp.nlargest(3,"ld_drop_norm")["layer"].tolist())
'''),
    md("## 6 · Save & Plot"),
    code('''from src.utils.io_utils import save_csv, ensure_dirs
from src.visualization.circuit_vis import plot_layer_ablation_bars, plot_layer_ablation_heatmap
paths = config["paths"]
paths["figures_dir"] = paths["outputs_dir"] + "/03_layer_ablation/figures"
paths["results_dir"] = paths["outputs_dir"] + "/03_layer_ablation/results"
ensure_dirs(paths["figures_dir"], paths["results_dir"])
save_csv(results_df, paths["results_dir"] + "/layer_ablation.csv")
plot_layer_ablation_bars(results_df, save_path=paths["figures_dir"]+"/08_layer_ablation_bars", formats=["html","png"])
plot_layer_ablation_heatmap(results_df, save_path=paths["figures_dir"]+"/09_layer_ablation_heatmap", formats=["html","png"])
print("Saved to", paths["results_dir"])
'''),
]
save_nb("03_layer_ablation.ipynb", notebook(cells_03, "03 Layer Ablation"))

# ── 04 Head Ablation ──────────────────────────────────────────────────────────
cells_04 = [
    md("# Experiment 04 — Attention Head Ablation (144 Heads)\n**CircuitScope · GPT-2 Small**\n\nMean-ablates each of the 144 attention heads individually. Ranks them by causal importance and classifies them as Name Mover, S-Inhibition, Helper, or Neutral.\n\n**Runtime:** ~20 min (GPU, n=200) | Use n=50 for quick test"),
    md("## 1 · Setup"), code(SETUP), code(CFG),
    md("## 2 · Load Model & Data"), code(MODEL), code(DATA),
    md("## 3 · Compute Mean Z Cache"),
    code('''from src.analysis.head_ablation import HeadAblationAnalyzer
N_SAMPLES = 50  # increase to 200 for full results
analyzer = HeadAblationAnalyzer(model, dataset, n_samples=N_SAMPLES, batch_size=16)
print("Computing mean z-vectors for all 144 heads…")
mean_z = analyzer.compute_mean_z()
print("Mean z cache ready.")
'''),
    md("## 4 · Run Full 144-Head Sweep"),
    code('''import time
t0 = time.time()
print("Running 144-head sweep…")
results_df = analyzer.run_full_sweep(mean_z)
print(f"Done in {time.time()-t0:.1f}s")
'''),
    md("## 5 · Results"),
    code('''print("Top 20 Most Important Heads:")
print(results_df.head(20)[["rank","head_label","importance","head_type"]].to_string())
print("\\nHead Type Distribution:")
print(results_df["head_type"].value_counts().to_string())
'''),
    code('''nm = results_df[results_df["head_type"]=="Name Mover"]
si = results_df[results_df["head_type"]=="S-Inhibition"]
print("Name Mover Heads:", nm["head_label"].tolist())
print("S-Inhibition Heads:", si["head_label"].tolist())
'''),
    md("## 6 · Save & Plot"),
    code('''from src.utils.io_utils import save_csv, ensure_dirs
from src.visualization.circuit_vis import plot_head_importance_heatmap, plot_head_ranking_bar
paths = config["paths"]
paths["figures_dir"] = paths["outputs_dir"] + "/04_head_ablation/figures"
paths["results_dir"] = paths["outputs_dir"] + "/04_head_ablation/results"
ensure_dirs(paths["figures_dir"], paths["results_dir"])
pivot_df = analyzer.pivot_importance_matrix(results_df)
save_csv(results_df, paths["results_dir"] + "/head_ablation.csv")
save_csv(pivot_df,   paths["results_dir"] + "/head_importance_matrix.csv", index=True)
plot_head_importance_heatmap(pivot_df, save_path=paths["figures_dir"]+"/10_head_importance_heatmap", formats=["html","png"])
plot_head_ranking_bar(results_df, top_n=20, save_path=paths["figures_dir"]+"/11_head_ranking_bar", formats=["html","png"])
print("Saved to", paths["results_dir"])
'''),
]
save_nb("04_head_ablation.ipynb", notebook(cells_04, "04 Head Ablation"))

# ── 05 Activation Patching ────────────────────────────────────────────────────
cells_05 = [
    md("# Experiment 05 — Activation Patching\n**CircuitScope · GPT-2 Small**\n\nAt every (layer, token position) pair, replaces the corrupted activation with the clean activation and measures behavior recovery. Topographically maps where IOI information is stored.\n\n**Runtime:** ~15–30 min (GPU)"),
    md("## 1 · Setup"), code(SETUP), code(CFG),
    md("## 2 · Load Model & Data"), code(MODEL), code(DATA),
    md("## 3 · Residual Stream Patching"),
    code('''import time
from src.analysis.activation_patching import ActivationPatchingAnalyzer
N_SAMPLES = 30  # 50 for full run
analyzer = ActivationPatchingAnalyzer(model, dataset, n_samples=N_SAMPLES, batch_size=1)
print("Running residual stream patching…")
t0 = time.time()
resid_df = analyzer.run_resid_patching()
print(f"Done in {time.time()-t0:.1f}s | Max restoration: {resid_df.values.max():.4f}")
'''),
    md("## 4 · Attention & MLP Patching (optional)"),
    code('''RUN_FULL = True
if RUN_FULL:
    print("Running attention patching…")
    attn_df = analyzer.run_attn_patching()
    print("Running MLP patching…")
    mlp_df  = analyzer.run_mlp_patching()
    print(f"Attn max: {attn_df.values.max():.4f} | MLP max: {mlp_df.values.max():.4f}")
else:
    attn_df = mlp_df = None
    print("Skipped.")
'''),
    md("## 5 · Save & Plot"),
    code('''from src.utils.io_utils import save_csv, ensure_dirs
from src.visualization.circuit_vis import plot_activation_patching_heatmap, plot_all_patching_comparison
paths = config["paths"]
paths["figures_dir"] = paths["outputs_dir"] + "/05_activation_patching/figures"
paths["results_dir"] = paths["outputs_dir"] + "/05_activation_patching/results"
ensure_dirs(paths["figures_dir"], paths["results_dir"])
save_csv(resid_df, paths["results_dir"] + "/patching_resid.csv", index=True)
plot_activation_patching_heatmap(resid_df, title="Residual Stream Patching", save_path=paths["figures_dir"]+"/12_patching_resid_heatmap", formats=["html","png"])
if attn_df is not None:
    save_csv(attn_df, paths["results_dir"] + "/patching_attn.csv", index=True)
    save_csv(mlp_df,  paths["results_dir"] + "/patching_mlp.csv",  index=True)
    plot_activation_patching_heatmap(attn_df, title="Attention Patching", save_path=paths["figures_dir"]+"/13_patching_attn_heatmap", formats=["html","png"])
    plot_activation_patching_heatmap(mlp_df,  title="MLP Patching",       save_path=paths["figures_dir"]+"/14_patching_mlp_heatmap",  formats=["html","png"])
    plot_all_patching_comparison(resid_df, attn_df, mlp_df,               save_path=paths["figures_dir"]+"/15_patching_comparison",   formats=["html","png"])
print("Saved to", paths["figures_dir"])
'''),
]
save_nb("05_activation_patching.ipynb", notebook(cells_05, "05 Activation Patching"))

# ── 06 Path Patching ──────────────────────────────────────────────────────────
cells_06 = [
    md("# Experiment 06 — Path Patching & Circuit Graph\n**CircuitScope · GPT-2 Small**\n\nMeasures how much each head's output carries IOI-relevant information (sender-side path patching) and builds a directed circuit graph.\n\n**Runtime:** ~15 min (GPU)"),
    md("## 1 · Setup"), code(SETUP), code(CFG),
    md("## 2 · Load Model & Data"), code(MODEL), code(DATA),
    md("## 3 · Run Sender Path Patching"),
    code('''import time
from src.analysis.path_patching import PathPatchingAnalyzer
N_SAMPLES = 50
analyzer = PathPatchingAnalyzer(model, dataset, n_samples=N_SAMPLES, importance_threshold=0.1)
print(f"Running sender patching: {N_SAMPLES} prompts x 144 heads…")
t0 = time.time()
sender_df = analyzer.run_sender_patching()
print(f"Done in {time.time()-t0:.1f}s")
'''),
    md("## 4 · Build Circuit Graph"),
    code('''graph_df = analyzer.build_circuit_graph(sender_df, top_n_senders=20)
summary  = analyzer.get_circuit_summary(sender_df)
print(f"Circuit nodes  : {summary['n_circuit_nodes']}")
print(f"Early (0-4)    : {summary['n_early_heads']} heads")
print(f"Middle (5-8)   : {summary['n_middle_heads']} heads — likely S-Inhibition")
print(f"Late (9-11)    : {summary['n_late_heads']} heads — likely Name Movers")
print(f"Top late heads : {summary['late_circuit_heads']}")
print("\\nTop 10 circuit heads:")
print(sender_df[sender_df["is_circuit_node"]].head(10)[["head_label","restoration_score"]].to_string())
'''),
    md("## 5 · Save & Plot"),
    code('''from src.utils.io_utils import save_csv, save_json, ensure_dirs
from src.visualization.circuit_vis import plot_sender_importance_heatmap, plot_circuit_graph
paths = config["paths"]
paths["figures_dir"] = paths["outputs_dir"] + "/06_path_patching/figures"
paths["results_dir"] = paths["outputs_dir"] + "/06_path_patching/results"
ensure_dirs(paths["figures_dir"], paths["results_dir"])
save_csv(sender_df, paths["results_dir"] + "/path_patching_senders.csv")
save_json(summary,  paths["results_dir"] + "/circuit_summary.json")
if not graph_df.empty:
    save_csv(graph_df, paths["results_dir"] + "/circuit_graph_edges.csv")
plot_sender_importance_heatmap(sender_df, save_path=paths["figures_dir"]+"/16_sender_importance_heatmap", formats=["html","png"])
if not graph_df.empty:
    plot_circuit_graph(graph_df, sender_df, save_path=paths["figures_dir"]+"/17_circuit_graph", formats=["html","png"])
print("Saved to", paths["results_dir"])
'''),
]
save_nb("06_path_patching.ipynb", notebook(cells_06, "06 Path Patching"))

# ── 07 Full Pipeline ──────────────────────────────────────────────────────────
cells_07 = [
    md("# Experiment 07 — Full End-to-End Pipeline\n**CircuitScope · GPT-2 Small**\n\nRuns all 6 analysis stages with a single shared model instance.\n\n| Part | Experiment | GPU Time |\n|------|-----------|----------|\n| 0 | Baseline | ~3 min |\n| 1 | Logit Lens | ~5 min |\n| 2 | Layer Ablation | ~10 min |\n| 3 | Head Ablation | ~20 min |\n| 4 | Activation Patching | ~15 min |\n| 5 | Path Patching | ~15 min |\n| **Total** | | **~70 min** |"),
    md("## 1 · Setup"), code(SETUP), code(CFG),
    md("## 2 · Run Pipeline (subprocess — easiest)"),
    code('''import subprocess, sys
QUICK = True   # set False for full sample sizes (~70 min)
cmd = [sys.executable, "experiments/07_full_pipeline.py"] + (["--quick"] if QUICK else ["--full-patching"])
print("Running:", " ".join(cmd))
subprocess.run(cmd)
print("Done. Check outputs/07_full_pipeline/")
'''),
    md("## — OR — Run Each Part Manually in This Notebook"),
    code(MODEL), code(DATA),
    code('''# Part 0: Baseline
from src.evaluation.metrics import IOIEvaluator
evaluator = IOIEvaluator(model=model, dataset=dataset, batch_size=32, top_k=5)
results_df = evaluator.evaluate(use_corrupted=False)
stats = evaluator.compute_aggregate_stats(results_df)
print(f"Accuracy: {stats['accuracy']:.1%} | Mean LD: {stats['mean_logit_diff']:+.4f}")
'''),
    code('''# Part 1: Logit Lens
from src.analysis.logit_lens import LogitLensAnalyzer
lens_df = LogitLensAnalyzer(model, dataset, n_samples=100).run(batch_size=10)
first = lens_df[lens_df["logit_diff"] > 0]
print("IO preference emerges at:", first.iloc[0]["layer_label"] if not first.empty else "N/A")
'''),
    code('''# Part 2: Layer Ablation
from src.analysis.layer_ablation import LayerAblationAnalyzer
la = LayerAblationAnalyzer(model, dataset, n_samples=50, batch_size=16)
layer_df = la.run_full_sweep(la.compute_mean_cache())
print("Critical layers:", layer_df[layer_df["is_critical"]][["layer","component"]].values.tolist())
'''),
    code('''# Part 3: Head Ablation
from src.analysis.head_ablation import HeadAblationAnalyzer
ha = HeadAblationAnalyzer(model, dataset, n_samples=50, batch_size=16)
head_df = ha.run_full_sweep(ha.compute_mean_z())
print("Top 5 heads:")
print(head_df.head(5)[["head_label","importance","head_type"]].to_string())
'''),
    md("## 3 · Save All"),
    code('''from src.utils.io_utils import save_csv, ensure_dirs
paths = config["paths"]
paths["figures_dir"] = paths["outputs_dir"] + "/07_full_pipeline/figures"
paths["results_dir"] = paths["outputs_dir"] + "/07_full_pipeline/results"
ensure_dirs(paths["figures_dir"], paths["results_dir"])
save_csv(results_df, paths["results_dir"] + "/ioi_results.csv")
save_csv(lens_df,    paths["results_dir"] + "/logit_lens_by_layer.csv")
save_csv(layer_df,   paths["results_dir"] + "/layer_ablation.csv")
save_csv(head_df,    paths["results_dir"] + "/head_ablation.csv")
print("All results saved to", paths["results_dir"])
'''),
]
save_nb("07_full_pipeline.ipynb", notebook(cells_07, "07 Full Pipeline"))

# ── 08 Circuit Validation ─────────────────────────────────────────────────────
cells_08 = [
    md("# Experiment 08 — Circuit Validation\n**CircuitScope · GPT-2 Small**\n\nTests the discovered IOI circuit for:\n- **Necessity**: Ablating circuit heads should cause large performance drop\n- **Sufficiency**: Keeping only circuit heads should retain most performance\n- **Generalization**: Circuit should work on held-out prompts and templates\n\n**Runtime:** ~10 min (GPU)"),
    md("## 1 · Setup"), code(SETUP), code(CFG),
    md("## 2 · Load Model & Data"), code(MODEL), code(DATA),
    md("## 3 · Define Circuit"),
    code('''# Edit these based on your head ablation results (experiment 04)
CIRCUIT_HEADS = [
    (9, 6), (9, 9), (10, 0),   # Name Mover heads  (layers 9-11)
    (7, 3), (7, 9), (8, 6),    # S-Inhibition heads (layers 7-8)
    (3, 0), (4, 11), (5, 5),   # Helper heads       (layers 1-5)
]
from src.analysis.circuit_validation import CircuitSpec
circuit = CircuitSpec(head_list=CIRCUIT_HEADS, name="IOI Circuit")
print(f"Circuit defined: {len(CIRCUIT_HEADS)} heads")
for l, h in CIRCUIT_HEADS:
    print(f"  L{l}H{h}")
'''),
    md("## 4 · Run Validation Tests"),
    code('''import time
from src.analysis.head_ablation import HeadAblationAnalyzer
from src.analysis.circuit_validation import CircuitValidator
ha = HeadAblationAnalyzer(model, dataset, n_samples=100, batch_size=16)
mean_z = ha.compute_mean_z()
validator = CircuitValidator(model, dataset, mean_z, n_samples=100)
print("Running necessity, sufficiency & generalization tests…")
t0 = time.time()
val_df = validator.run_all_tests(circuit, threshold=0.05)
print(f"Done in {time.time()-t0:.1f}s")
print(val_df[["test_name","metric","value","passes"]].to_string())
'''),
    md("## 5 · Save"),
    code('''from src.utils.io_utils import save_csv, save_json, ensure_dirs
paths = config["paths"]
paths["figures_dir"] = paths["outputs_dir"] + "/08_circuit_validation/figures"
paths["results_dir"] = paths["outputs_dir"] + "/08_circuit_validation/results"
ensure_dirs(paths["figures_dir"], paths["results_dir"])
save_csv(val_df, paths["results_dir"] + "/circuit_validation.csv")
save_json({"circuit_heads": CIRCUIT_HEADS}, paths["results_dir"] + "/circuit_spec.json")
print("Saved to", paths["results_dir"])
'''),
]
save_nb("08_circuit_validation.ipynb", notebook(cells_08, "08 Circuit Validation"))

# ── 09 Novel Extension ────────────────────────────────────────────────────────
cells_09 = [
    md("# Experiment 09 — Pronoun Resolution (Novel Extension)\n**CircuitScope · GPT-2 Small**\n\nOriginal contribution: applies the same head ablation pipeline to pronoun resolution.\n\nExample: *'Sarah met James at the cafe. She bought a gift for ___'* → James\n\nIf the same heads matter for both tasks, they implement a general *name-moving* operation.\n\n**Runtime:** ~20 min (GPU)"),
    md("## 1 · Setup"), code(SETUP), code(CFG),
    md("## 2 · Load Model"), code(MODEL),
    md("## 3 · Generate Pronoun Dataset"),
    code('''from src.data.pronoun_dataset import PronounDataset
pronoun_ds = PronounDataset(model=model, n_prompts=500, seed=config.get("seed",42)).generate()
print(f"Pronoun dataset: {len(pronoun_ds)} prompts")
print(pronoun_ds.summary())
'''),
    md("## 4 · Head Ablation on Pronoun Task"),
    code('''import time
from src.analysis.head_ablation import HeadAblationAnalyzer
N_SAMPLES = 100
pa = HeadAblationAnalyzer(model, pronoun_ds, n_samples=N_SAMPLES, batch_size=16)
mean_z = pa.compute_mean_z()
print("Running 144-head sweep on pronoun task…")
t0 = time.time()
pronoun_df = pa.run_full_sweep(mean_z)
print(f"Done in {time.time()-t0:.1f}s")
print(pronoun_df.head(10)[["head_label","importance","head_type"]].to_string())
'''),
    md("## 5 · Compare with IOI (Pearson r)"),
    code('''import pandas as pd
from pathlib import Path
ioi_path = Path("outputs/04_head_ablation/results/head_ablation.csv")
if ioi_path.exists():
    ioi_df = pd.read_csv(ioi_path)
    merged = pronoun_df[["head_label","importance"]].rename(columns={"importance":"pronoun"})
    merged = merged.merge(ioi_df[["head_label","importance"]].rename(columns={"importance":"ioi"}), on="head_label")
    from scipy.stats import pearsonr
    r, p = pearsonr(merged["ioi"], merged["pronoun"])
    print(f"Pearson r = {r:.4f}  p = {p:.4e}")
    print("Strong correlation (r>0.5): circuit partially reused!" if abs(r)>0.5 else "Weak correlation.")
else:
    print("Run experiment 04 first to compare. Showing pronoun results only.")
    print(pronoun_df.head(15)[["head_label","importance"]].to_string())
'''),
    md("## 6 · Save"),
    code('''from src.utils.io_utils import save_csv, ensure_dirs
paths = config["paths"]
paths["results_dir"] = paths["outputs_dir"] + "/09_novel_extension/results"
ensure_dirs(paths["results_dir"])
save_csv(pronoun_df, paths["results_dir"] + "/pronoun_head_ablation.csv")
print("Saved to", paths["results_dir"])
'''),
]
save_nb("09_novel_extension.ipynb", notebook(cells_09, "09 Novel Extension"))

# ── 10 Statistical Analysis ───────────────────────────────────────────────────
cells_10 = [
    md("# Experiment 10 — Statistical Analysis\n**CircuitScope · GPT-2 Small**\n\nRigorous statistical validation of circuit findings:\n- Bootstrap 95% confidence intervals\n- Cohen's d effect sizes (Name Movers vs. Neutral)\n- Spearman correlation (layer depth vs. importance)\n- Permutation tests\n\n**Prerequisite:** Run experiments 01 and 04 first.\n**Runtime:** ~10 min"),
    md("## 1 · Setup"), code(SETUP), code(CFG),
    md("## 2 · Load CSVs"),
    code('''import pandas as pd
from pathlib import Path
results_df = pd.read_csv("outputs/01_baseline/results/ioi_results.csv")   if Path("outputs/01_baseline/results/ioi_results.csv").exists()  else None
head_df    = pd.read_csv("outputs/04_head_ablation/results/head_ablation.csv") if Path("outputs/04_head_ablation/results/head_ablation.csv").exists() else None
print("IOI results:", results_df.shape if results_df is not None else "NOT FOUND — run exp 01 first")
print("Head ablation:", head_df.shape  if head_df    is not None else "NOT FOUND — run exp 04 first")
'''),
    md("## 3 · Bootstrap Confidence Intervals"),
    code('''import numpy as np
from src.analysis.statistics import bootstrap_ci
N_BOOTSTRAP = 2000
if results_df is not None:
    ld = results_df["logit_diff"].dropna().values
    mean_ld, lo_ld, hi_ld = bootstrap_ci(ld, statistic=np.mean, n_bootstrap=N_BOOTSTRAP)
    acc = results_df["is_correct"].astype(float).values
    mean_acc, lo_acc, hi_acc = bootstrap_ci(acc, statistic=np.mean, n_bootstrap=N_BOOTSTRAP)
    print(f"Mean Logit Diff : {mean_ld:.4f}  [95% CI: {lo_ld:.4f}, {hi_ld:.4f}]")
    print(f"Accuracy        : {mean_acc:.3f}  [95% CI: {lo_acc:.3f}, {hi_acc:.3f}]")
'''),
    md("## 4 · Cohen's d — Name Movers vs. Neutral"),
    code('''from src.analysis.statistics import cohens_d
if head_df is not None:
    nm = head_df[head_df["head_type"]=="Name Mover"]["importance"].values
    nt = head_df[head_df["head_type"]=="Neutral"]["importance"].values
    if len(nm)>0 and len(nt)>0:
        d = cohens_d(nm, nt)
        print(f"Cohen s d = {d:.4f}  ({'Large' if abs(d)>0.8 else 'Medium' if abs(d)>0.5 else 'Small'} effect)")
        print(f"Name Mover mean: {nm.mean():.4f} | Neutral mean: {nt.mean():.4f}")
'''),
    md("## 5 · Spearman Correlation (Layer Depth vs. Importance)"),
    code('''from src.analysis.statistics import spearman_layer_correlation
if head_df is not None:
    rho, p_val = spearman_layer_correlation(head_df)
    print(f"Spearman rho = {rho:.4f}  (p = {p_val:.4e})")
    if p_val < 0.001:
        print("Significant: later layers tend to have more important heads.")
'''),
    md("## 6 · Save"),
    code('''from src.utils.io_utils import save_json, ensure_dirs
paths = config["paths"]
paths["results_dir"] = paths["outputs_dir"] + "/10_statistical_analysis/results"
ensure_dirs(paths["results_dir"])
summary = {}
if results_df is not None:
    summary["logit_diff_ci"] = {"mean": float(mean_ld), "lower": float(lo_ld), "upper": float(hi_ld)}
    summary["accuracy_ci"]   = {"mean": float(mean_acc),"lower": float(lo_acc),"upper": float(hi_acc)}
if head_df is not None:
    summary["cohens_d"] = float(d)
    summary["spearman_rho"] = float(rho)
    summary["spearman_p"]   = float(p_val)
save_json(summary, paths["results_dir"] + "/stats_summary.json")
print("Saved summary:", summary)
'''),
]
save_nb("10_statistical_analysis.ipynb", notebook(cells_10, "10 Statistical Analysis"))

print("\nAll 10 notebooks generated successfully in notebooks/")
