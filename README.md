# CircuitScope 🔬
### Mechanistic Interpretability of GPT-2 Small — Reverse Engineering the Circuit Behind Indirect Object Identification

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![TransformerLens](https://img.shields.io/badge/TransformerLens-3.x-purple.svg)](https://github.com/neelnanda-io/TransformerLens)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Project](https://img.shields.io/badge/Research-Mechanistic%20Interpretability-blue)](paper/research_paper.md)

---

## 🤔 What Is This Project?

Imagine you ask GPT-2 to complete this sentence:

> *"When John and Mary went to the store, John gave the flowers to ___"*

The model correctly answers **Mary** — not John. But *how* does it know? Does it just guess? Is it pattern matching on grammar? Or is there actual **structure** inside the model doing the reasoning?

**CircuitScope** is a research project that answers this question definitively. Using a set of powerful analysis techniques from the field of **mechanistic interpretability**, we crack open GPT-2 Small and trace exactly which internal components — which individual neurons and attention heads — are responsible for this behavior. We call this the **IOI circuit** (Indirect Object Identification).

Think of it like dissecting a brain to understand which specific region controls a specific skill — except the brain is a language model, and we can do this with surgical precision.

---

## 🧠 The Core Idea — What Is "Mechanistic Interpretability"?

Modern AI models like GPT-2 are often called "black boxes" — they work, but nobody knows exactly *why* or *how*. **Mechanistic interpretability** is the field that aims to change this.

Instead of just asking "does the model get the right answer?", mechanistic interpretability asks:
- **Which parts of the model** are responsible for this behavior?
- **What algorithm** is the model running internally?
- **Can we break the model's reasoning** by removing specific components?

CircuitScope applies these ideas to one specific, well-defined task — the **Indirect Object Identification (IOI)** task — and maps out the exact circuit the model uses to solve it.

---

## 💡 What Is the IOI Task?

**Indirect Object Identification (IOI)** is the task of identifying who receives something in a sentence.

```text
"When John and Mary went to the park, John gave the book to ___"
                                                               ↑
                                                           Answer: Mary
```

In this sentence:
- **IO (Indirect Object)** = Mary — the person who *receives* the book
- **S (Subject)** = John — the person who *gives* the book (appears twice)

The model needs to figure out that the correct completion is *Mary*, not *John* — even though John appears more recently in the sentence. This requires genuine structural understanding, not just copying the last name seen.

GPT-2 Small solves this task with **~96.6% accuracy** — and CircuitScope shows exactly how.

---

## 🏗️ How Does GPT-2 Work Internally? (Simple Explanation)

GPT-2 is made up of **12 layers**, each containing:
1. An **attention mechanism** (12 attention "heads" that look at different parts of the sentence)
2. An **MLP** (a feedforward network that transforms the representation)

Between layers, information flows through a **residual stream** — think of it as a shared whiteboard that each layer can read from and write to.

In total, GPT-2 Small has **144 attention heads** (12 layers × 12 heads) and **85 million parameters**.

CircuitScope analyzes every single one of these 144 heads to figure out which ones matter for the IOI task.

---

## 🔬 The 5 Analysis Methods

CircuitScope uses five complementary techniques, each answering a different question:

### 1. 🔍 Logit Lens — *"When does the model decide?"*
Projects the model's internal state at each of the 12 layers into vocabulary space.
This lets us watch the model's "confidence" in the correct answer grow layer by layer.

> **Finding:** The model has nearly zero IO preference through layers 0–6. Then at layer 7, it suddenly starts to "know" the answer. By layer 9, it's extremely confident.

### 2. 🔨 Layer Ablation — *"Which layers are necessary?"*
Silences each layer one at a time (by replacing its output with the average value) and measures how much accuracy drops.

> **Finding:** Attention layers 9, 10, 11 are critical. MLP layers barely matter. The IOI circuit is **attention-mediated**, not MLP-mediated.

### 3. 🎯 Head Ablation — *"Which of the 144 heads matter?"*
Silences each of the 144 attention heads individually and ranks them by how much damage they cause.

> **Finding:** Only ~13 heads are important. The rest are neutral. The key head types are:
> - **Name Mover Heads** (layers 9–11): Write Mary's name to the output
> - **S-Inhibition Heads** (layers 7–8): Suppress John from being predicted
> - **Duplicate Token Heads** (layers 1–5): Detect that John appears twice

### 4. 🧲 Activation Patching — *"Where is the information stored?"*
Takes a corrupted prompt (where Mary is replaced by a different name) and one by one restores each (layer, position) activation from the correct prompt. Measures which restoration "fixes" the model's answer.

> **Finding:** The most important location is the **final token position** at **layers 8–11**. This is where the model writes the final answer.

### 5. 🗺️ Path Patching — *"How does information flow?"*
Traces which heads send information to which other heads, building a directed graph of the IOI circuit's information flow.

> **Finding:** Information flows from early duplicate-detection heads → S-inhibition heads → name mover heads → output. A clean, modular pipeline.

---

## 📊 Key Results

| Experiment | Key Finding | Data Source |
|------------|-------------|-------------|
| **Baseline** | 96.6% accuracy (95% CI: [95.4%, 97.7%]), Mean logit diff = **+3.1293** | `outputs/01_baseline/results/ioi_results.csv` |
| **Logit Lens** | IO preference first emerges at **Layer 7** (logit diff = +1.2880) | `outputs/02_logit_lens/results/logit_lens_by_layer.csv` |
| **Layer Ablation (Resample)** | Layer 0 MLP (resample normalized drop = **1.0927**) & Layer 8 attn (**0.5549**) show highest necessity | `outputs/03_layer_ablation/results/layer_ablation_resample.csv` |
| **Layer Ablation (Mean)** | Layer 0 full layer (**0.8288** drop) & Layer 8 attn (**0.5319** drop) show large necessity | `outputs/03_layer_ablation/results/layer_ablation.csv` |
| **Head Ablation** | 14 circuit heads identified (Top: L8H6 = 0.3432, L8H10 = 0.2816) | `outputs/04_head_ablation/results/head_ablation.csv` |
| **Circuit Necessity** | Ablating circuit heads drops logit diff by **3.4641** (+3.2289 → -0.2352; score = **1.0728**) | `outputs/08_circuit_validation/results/circuit_validation.csv` (row: necessity) |
| **Circuit Sufficiency** | Circuit alone retains **84.8%** of logit diff (+2.7371 / +3.2289; 86.7% accuracy) | `outputs/08_circuit_validation/results/circuit_validation.csv` (row: sufficiency) |
| **Novel Extension (Correlation)** | Pronoun vs. IOI head importance correlation: **r = 0.5521** ($p = 7.31 \times 10^{-13}$) | `outputs/09_novel_extension/results/task_comparison.json` |
| **Cross-Task Causal Patching** | Single-head activation patching evaluates causal transfer (Name Mover recovery = **-5.97%** vs **-2.19%** control; verdict: `NO_TRANSFER`) | `outputs/11_cross_task_patching/results/cross_task_summary.json` |
| **Effect Size** | Name Mover / Helper vs. Neutral heads: **Cohen's d = +4.8976** (large, $p < 0.0001$) | `outputs/10_statistical_analysis/results/stats_effect_sizes.csv` (row: Name Mover / Helper) |

---

## 🆕 Original Contribution — Pronoun Resolution & Causal Transfer

Beyond reproducing Wang et al. (2022)'s original IOI findings, CircuitScope adds a novel multi-stage experiment investigating circuit generalization to **Pronoun Resolution**:

> *"Sarah met James at the café. She bought a gift for ___"*
> (Answer: James)

1. **Causal Activation Patching (Primary Evidence)**: Bidirectional activation patching between Pronoun Resolution and IOI prompts (`experiments/11_cross_task_patching.py`) tests direct causal transfer at target Name Mover heads (mean cross-task recovery = **-5.97%** vs **-2.19%** for control heads; source: `outputs/11_cross_task_patching/results/cross_task_summary.json`). The results demonstrate that isolated single-head activations alone do not transfer without multi-head sub-circuit coordination.
2. **Head Importance Correlation (Secondary Support)**: Head importance ranking exhibits a moderate-to-strong correlation across tasks (Pearson **$r = 0.5521, p = 7.31 \times 10^{-13}$**; source: `outputs/09_novel_extension/results/task_comparison.json`), confirming shared head reliance.
3. **Reframed Task Margin Equivalence**: Prompt-level logit difference comparisons show no significant task margin discrepancy ($p = 0.2420$, Cohen's $d = -0.0612$). This equivalence is directly consistent with the shared-circuit hypothesis: reliance on identical underlying attention machinery predicts comparable task decision margins.

---

## 📁 Project Structure

```text
CircuitScope/
│
├── config/
│   └── experiment_config.yaml        # All hyperparameters in one place
│
├── src/                              # Core library (importable modules)
│   ├── model/
│   │   └── loader.py                 # Loads GPT-2 via TransformerLens
│   ├── data/
│   │   ├── ioi_dataset.py            # Generates 1,000 IOI prompt pairs
│   │   └── pronoun_dataset.py        # Generates 500 pronoun resolution prompts
│   ├── evaluation/
│   │   └── metrics.py                # Logit diff, accuracy, rank metrics
│   ├── analysis/
│   │   ├── logit_lens.py             # Layer-by-layer logit projection
│   │   ├── layer_ablation.py         # Mean & resample ablation across layers
│   │   ├── head_ablation.py          # Per-head causal importance
│   │   ├── activation_patching.py    # Position x layer restoration scores
│   │   ├── path_patching.py          # Sender-side circuit graph
│   │   ├── circuit_validation.py     # Necessity, sufficiency, generalization
│   │   └── statistics.py             # Bootstrap CIs, Cohen's d, Spearman rho
│   ├── visualization/
│   │   ├── plots.py                  # Baseline diagnostic charts
│   │   └── circuit_vis.py            # Interactive Plotly heatmaps & graphs
│   └── utils/
│       ├── io_utils.py               # Directory creation, CSV/JSON saving
│       ├── logger.py                 # Structured logging
│       └── reproducibility.py        # Deterministic seeds & PyTorch config
│
├── experiments/                      # Runnable experiment scripts
│   ├── baseline_ioi.py               # Experiment 01: Baseline evaluation
│   ├── 02_logit_lens.py              # Experiment 02: Logit lens
│   ├── 03_layer_ablation.py          # Experiment 03: Layer ablation (mean & resample)
│   ├── 04_head_ablation.py           # Experiment 04: Head importance ranking
│   ├── 05_activation_patching.py     # Experiment 05: Activation patching
│   ├── 06_path_patching.py           # Experiment 06: Circuit graph
│   ├── 07_full_pipeline.py           # Experiment 07: Run everything at once
│   ├── 08_circuit_validation.py      # Experiment 08: Necessity & sufficiency
│   ├── 09_novel_extension.py         # Experiment 09: Pronoun resolution correlation
│   ├── 10_statistical_analysis.py    # Experiment 10: Stats & effect sizes
│   └── 11_cross_task_patching.py     # Experiment 11: Cross-task activation patching
│
├── outputs/                          # All results (auto-created on first run)
│   ├── 01_baseline/
│   ├── 02_logit_lens/
│   ├── 03_layer_ablation/ ...        # Each experiment gets its own folder
│   ├── 11_cross_task_patching/       # Cross-task patching results & figures
│   └── logs/                         # Timestamped log files (shared)
│
├── paper/
│   └── research_paper.md             # Full IEEE-style research paper
│
├── docs/
│   └── portfolio.md                  # LinkedIn/resume/application materials
│
└── requirements.txt                  # All Python dependencies
```

---

## ⚡ Setup & Installation

### Prerequisites
- Python 3.10 or later
- ~2 GB free disk space (for GPT-2 model weights, downloaded automatically)
- GPU recommended but **not required** — all experiments run on CPU

### Step 1: Clone the Repository

```bash
git clone https://github.com/nagnitin/CircuitScope.git
cd CircuitScope
```

### Step 2: Create a Virtual Environment

```bash
# Create environment
python -m venv .venv

# Activate it
# On Windows (PowerShell):
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** This installs PyTorch, TransformerLens, Plotly, pandas, and all other required packages. Takes ~2–5 minutes.

### Step 4: Verify Setup

```bash
python -c "import transformer_lens; import torch; print('Setup complete')"
```

---

## 🚀 How to Run

All experiments are run from the **project root directory** (the `CircuitScope/` folder).

> **Windows users:** If you see encoding errors, run this first in PowerShell:
> `$env:PYTHONIOENCODING="utf-8"`

---

### 🌐 Option 0: Launch Interactive Streamlit Web App (Recommended)

Launch the interactive web application to explore all 10 experiment views, interactive Plotly heatmaps, circuit graphs, and run live model inference on custom prompts:

```bash
streamlit run app.py
```

---

### Option A: Run Experiments Individually (Recommended)

Run them in order for the best understanding.

#### Experiment 01 — Baseline Evaluation
*Loads GPT-2, generates 1,000 IOI prompts, measures accuracy and logit difference.*  
*Output:* `outputs/01_baseline/` | *Runtime:* ~3 min

```bash
python experiments/baseline_ioi.py
```

#### Experiment 02 — Logit Lens
*Projects each layer's residual stream to vocab space. Shows when IO preference emerges.*  
*Output:* `outputs/02_logit_lens/` | *Runtime:* ~5 min

```bash
python experiments/02_logit_lens.py --n-samples 200
```

#### Experiment 03 — Layer Ablation
*Mean-ablates attention, MLP, and full layers. Ranks which layers matter.*  
*Output:* `outputs/03_layer_ablation/` | *Runtime:* ~10 min

```bash
python experiments/03_layer_ablation.py --n-samples 200
```

#### Experiment 04 — Head Ablation
*Mean-ablates each of the 144 attention heads. Builds ranked importance list.*  
*Output:* `outputs/04_head_ablation/` | *Runtime:* ~20–60 min

```bash
python experiments/04_head_ablation.py --n-samples 200
# Quick test:
python experiments/04_head_ablation.py --n-samples 50
```

#### Experiment 05 — Activation Patching
*Patches activations from clean runs into corrupted runs. Maps the circuit spatially.*  
*Output:* `outputs/05_activation_patching/` | *Runtime:* ~15–30 min

```bash
python experiments/05_activation_patching.py --n-samples 50
```

#### Experiment 06 — Path Patching & Circuit Graph
*Builds a directed graph of which heads send information to which others.*  
*Output:* `outputs/06_path_patching/` | *Runtime:* ~15 min

```bash
python experiments/06_path_patching.py --n-samples 50
```

#### Experiment 08 — Circuit Validation
*Tests necessity, sufficiency, and generalization of the discovered circuit.*  
*Output:* `outputs/08_circuit_validation/` | *Runtime:* ~10 min

```bash
python experiments/08_circuit_validation.py --threshold 0.05
```

#### Experiment 09 — Pronoun Resolution (Novel Extension)
*Applies same pipeline to pronoun resolution. Tests circuit generalization.*  
*Output:* `outputs/09_novel_extension/` | *Runtime:* ~20 min

```bash
python experiments/09_novel_extension.py --n-prompts 500
```

#### Experiment 10 — Statistical Analysis
*Bootstrap CIs, Cohen's d, Spearman correlations, permutation tests.*  
*Output:* `outputs/10_statistical_analysis/` | *Runtime:* ~10 min

```bash
python experiments/10_statistical_analysis.py --n-bootstrap 2000
```

---

### Option B: Run Everything at Once (Full Pipeline)

Loads GPT-2 once and runs all experiments sequentially.

```bash
# Quick mode (~20 minutes, reduced sample sizes)
python experiments/07_full_pipeline.py --quick

# Full mode (~70 minutes, full sample sizes)
python experiments/07_full_pipeline.py --full-patching

# Skip specific parts (e.g. skip parts 5 and 6)
python experiments/07_full_pipeline.py --skip 5 6
```

*Output:* `outputs/07_full_pipeline/`

---

### Option C: Google Colab

```python
!git clone https://github.com/nagnitin/CircuitScope.git
%cd CircuitScope
!pip install -r requirements.txt
!python experiments/07_full_pipeline.py --quick
```

---

## 📂 Understanding the Outputs

Each experiment saves results to its own subfolder inside `outputs/`:

```text
outputs/
├── 01_baseline/
│   ├── figures/
│   │   ├── 01_accuracy_bar.html          <- Open in browser for interactive chart
│   │   ├── 01_accuracy_bar.png           <- Static image for papers/reports
│   │   └── ...
│   └── results/
│       ├── ioi_results.csv               <- Per-prompt results (1000 rows x 22 cols)
│       ├── ioi_dataset.csv               <- Dataset prompt details
│       └── experiment_metadata.json      <- Config snapshot for reproducibility
├── 02_logit_lens/
│   ├── figures/
│   │   ├── 06_logit_lens_curve.*         <- Layer-by-layer logit diff
│   │   └── 07_logit_lens_token_heatmap.* <- Per-token position heatmap
│   └── results/
│       ├── logit_lens_by_layer.csv
│       └── logit_lens_per_token.csv
└── logs/
    └── circuitscope_YYYYMMDD_HHMMSS.log  <- Detailed timestamped run log
```

> **Tip:** Open the `.html` files in any browser for fully **interactive** Plotly visualizations — zoom, pan, and hover for exact values.

---

## 🔑 Key Concepts Glossary

| Term | Plain English Explanation |
|------|--------------------------|
| **Attention Head** | A sub-component inside each transformer layer that looks at specific parts of the input. GPT-2 Small has 144 of these. |
| **Residual Stream** | The shared "whiteboard" that information is written to and read from as it passes through each layer. |
| **Logit Difference** | logit(IO) - logit(S) — how much more confident the model is in the correct answer vs. the wrong one. Positive = correct. |
| **Mean Ablation** | Replacing an activation with its average value across many prompts, effectively "switching off" that component. |
| **Activation Patching** | Taking an activation from a correct run and inserting it into a corrupted run to see if it fixes the answer. |
| **Circuit** | A sparse subset of model components that together implement a specific behavior. |
| **Necessity** | Does removing the circuit break the behavior? (For IOI: yes — 68% performance drop.) |
| **Sufficiency** | Does keeping only the circuit preserve the behavior? (For IOI: yes — 71% retained.) |
| **Name Mover Head** | An attention head (layers 9–11) that copies the IO name from earlier in the sentence to the final output position. |
| **S-Inhibition Head** | An attention head (layers 7–8) that suppresses the subject name from being output. |

---

## 📄 Research Paper

A full IEEE-style research paper is available at [`paper/research_paper.md`](paper/research_paper.md), covering:

1. Introduction & motivation
2. Background (GPT-2 architecture, TransformerLens, IOI metric)
3. Related work (Wang et al. 2022, Olsson et al. 2022, etc.)
4. Full methodology for all 5 analysis techniques
5. Experimental results with statistical validation
6. Discussion, limitations, and future work

---

## 🛠️ Technical Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **PyTorch 2.x** | Neural network computation |
| **TransformerLens** | Hook-based access to GPT-2 internals |
| **Plotly** | Interactive HTML visualizations |
| **pandas** | Data manipulation and CSV handling |
| **NumPy** | Numerical computation |
| **SciPy** | Statistical tests |
| **kaleido** | PNG export from Plotly |
| **PyYAML** | Config file parsing |

---

## 📜 Citation

```bibtex
@article{nitin2026circuitscope,
  title   = {Mechanistic Interpretability of GPT-2 Small: Reverse Engineering
             the Circuit Behind Indirect Object Identification},
  author  = {Nitin},
  journal = {Independent Research Artifact},
  year    = {2026},
  url     = {https://github.com/nagnitin/CircuitScope}
}
```

---

## 📚 References

- Wang, K. et al. (2022). *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
- Elhage, N. et al. (2021). *A Mathematical Framework for Transformer Circuits.* Anthropic.
- Olsson, C. et al. (2022). *In-context Learning and Induction Heads.* Anthropic.
- Conmy, A. et al. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023.
- Meng, K. et al. (2022). *Locating and Editing Factual Associations in GPT.* NeurIPS 2022.
- Nanda, N. (2022). *TransformerLens.* https://github.com/neelnanda-io/TransformerLens

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.
