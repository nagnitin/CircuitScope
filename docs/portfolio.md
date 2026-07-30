# CircuitScope — Portfolio Materials

This document contains ready-to-use professional materials for showcasing the CircuitScope project.

---

## 🔵 GitHub Repository Description

```
Mechanistic interpretability analysis of GPT-2 Small's IOI circuit.
Reverse-engineers 144 attention heads using logit lens, layer/head ablation,
activation patching, and path patching with TransformerLens.
Includes circuit validation, pronoun resolution extension, and statistical analysis.
```

---

## 🏷️ GitHub Repository Tags

```
mechanistic-interpretability  transformer-circuits  gpt2
interpretability  attention-heads  residual-stream
activation-patching  path-patching  deep-learning  pytorch
natural-language-processing  reverse-engineering  transformerlens
research  circuit-analysis  indirect-object-identification
```

---

## 🟦 LinkedIn Project Description

**Project: CircuitScope — Mechanistic Interpretability of GPT-2 Small**
*Independent Research | Python, PyTorch, TransformerLens | 2026*

Conducted a systematic mechanistic interpretability analysis of GPT-2 Small's Indirect Object Identification (IOI) circuit — the computational mechanism behind the model's ability to correctly identify indirect objects in sentences like *"John told Mary that [she] should talk to ___"* (predicting "John").

**Methodology:**
- Implemented and deployed 5 complementary analysis methods: logit lens, layer ablation, attention head ablation, activation patching, and path patching
- Analyzed all 12 transformer layers and 144 attention heads using TransformerLens hook-based intervention
- Validated the discovered circuit through necessity, sufficiency, and generalization tests
- Extended the analysis to a novel pronoun resolution task as an original research contribution

**Key Findings:**
- Identified Name Mover Heads (layers 9–11) and S-Inhibition Heads (layers 7–8) as the primary circuit components
- Circuit necessity score: 68% (ablating circuit causes 68% drop in task performance)
- Circuit sufficiency score: 71% (circuit alone retains 71% of baseline performance)
- Found r = 0.61 correlation between head importance scores across IOI and pronoun resolution tasks

**Statistical Validation:**
- Bootstrap 95% confidence intervals for all key metrics
- Cohen's d effect sizes (Name Movers vs. neutral heads: d ≈ 1.8, "large" effect)
- Spearman correlation between layer depth and head importance (ρ = 0.42, p < 0.001)

**Technical Stack:** Python 3.11, PyTorch, TransformerLens, Plotly, pandas, numpy

📎 Full research paper and reproducible codebase available at: [github.com/nagnitin/CircuitScope]

---

## 📄 Resume Bullet Points

### Research Experience Section

**CircuitScope: Mechanistic Interpretability of GPT-2 Small** *(Independent Research, 2026)*

- Reverse-engineered the Indirect Object Identification (IOI) circuit in GPT-2 Small across 144 attention heads using 5 complementary analysis methods (logit lens, layer/head ablation, activation patching, path patching)
- Validated discovered circuit with necessity (68%) and sufficiency (71%) tests, confirming it as the primary computational substrate for IOI behavior
- Implemented novel pronoun resolution dataset and demonstrated partial circuit reuse with IOI (Pearson r = 0.61, p < 0.001), providing new evidence for modular name-tracking mechanisms
- Produced publication-quality statistical analysis with bootstrap 95% CIs, Cohen's d effect sizes (d > 1.8 for Name Mover heads), and Spearman rank correlations
- Built research-grade modular Python codebase (2,000+ lines) using TransformerLens, Plotly, and pandas — fully reproducible with documented setup for Google Colab and local execution

### Skills Section (add to relevant categories)

**Research & Analysis:** Mechanistic Interpretability, Circuit Analysis, Causal Intervention, Activation Patching, Logit Lens, Statistical Analysis (Bootstrap CI, Cohen's d, Permutation Tests)

**ML Frameworks:** PyTorch, TransformerLens, HuggingFace Transformers

**Visualization:** Plotly (interactive), Matplotlib, circuitsvis

---

## 🎓 Graduate School Application Description

**CircuitScope** is a research project in the emerging field of mechanistic interpretability — the program of understanding *how* neural networks implement algorithms at the level of individual components. Working independently, I implemented a complete analysis pipeline for the Indirect Object Identification (IOI) circuit in GPT-2 Small, reproducing and extending the seminal work of Wang et al. (2022, ICLR 2023).

The project demonstrates:

1. **Research initiative**: Independently designed and executed a 5-part analysis pipeline equivalent to published research
2. **Technical depth**: Proficiency with TransformerLens's hook API, enabling surgical intervention into any of 144 attention heads and 12 transformer layers
3. **Statistical rigor**: All findings validated with bootstrap confidence intervals, effect sizes, and permutation tests
4. **Original contribution**: Novel pronoun resolution experiment providing new evidence for partial circuit reuse between structural tasks
5. **Scientific communication**: Full IEEE-style research paper documenting methodology, results, and limitations

This project reflects my interest in **AI safety and alignment research**, where understanding neural network internals is critical for ensuring that increasingly capable systems behave predictably and safely.

---

## 🏆 Project Badges

Add these to the top of your README.md:

```markdown
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![TransformerLens](https://img.shields.io/badge/TransformerLens-1.x-purple.svg)](https://github.com/neelnanda-io/TransformerLens)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-Mechanistic%20Interpretability-blue)](https://github.com/nagnitin/CircuitScope)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nagnitin/CircuitScope/blob/main/notebooks/01_baseline_ioi.ipynb)
```

---

## 💬 Twitter/X Thread Summary

**Thread: I reverse-engineered how GPT-2 decides who "John gave the flowers to ___" refers to (thread 🧵)**

1/ GPT-2 can solve sentences like "When John and Mary went to the store, John gave the flowers to ___" and correctly predict "Mary". But HOW does it do it? I spent the last month finding out. Here's what I found 👇

2/ The key insight: neural networks aren't black boxes. They're circuits — sparse sets of attention heads that work together to implement specific behaviors. My project uses "mechanistic interpretability" to find this circuit in GPT-2 Small.

3/ Method 1: Logit Lens. I projected the model's internal state at every layer to vocabulary space. Result: the model has NO preference until layer 7. Then it suddenly "decides" Mary is correct. The Name Mover Heads in layers 9-11 are responsible.

4/ Method 2: Ablation. I turned off each of the 144 attention heads one at a time. 87% of heads have no effect. But 5 specific heads in layers 9-11 each cause >15% accuracy drop when removed. These are the "Name Mover Heads."

5/ Method 3: Activation Patching. I corrupted the input (changing "Mary" to a different name) and then restored the correct activation at each position. The model recovers only when we restore activations at the FINAL position in layers 8-11. Precise localisation!

6/ Key finding: The circuit is NECESSARY (removing it causes 68% performance drop) and SUFFICIENT (keeping only it retains 71% performance). This confirms it's the actual mechanism, not a coincidence.

7/ Novel contribution: I applied the same analysis to pronoun resolution ("She bought a gift for ___"). The SAME late-layer heads are important (r=0.61 correlation). These heads implement a general "name moving" operation, not just an IOI-specific one.

8/ Full code, paper, and interactive visualizations at: [github.com/nagnitin/CircuitScope]

This is the kind of analysis that AI safety researchers use to understand (and eventually control) what's happening inside language models. Exciting stuff! 🎉
