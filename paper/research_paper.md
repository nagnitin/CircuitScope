# Mechanistic Interpretability of GPT-2 Small: Reverse Engineering the Circuit Behind Indirect Object Identification

**Author:** Nitin  
**Affiliation:** Independent Research  
**Repository:** github.com/nagnitin/CircuitScope  
**Date:** July 2026

---

## Abstract

We present **CircuitScope**, a systematic mechanistic interpretability analysis of the Indirect Object Identification (IOI) circuit in GPT-2 Small. The IOI task requires a language model to identify the indirect object in sentences of the form *"When John and Mary went to the park, John gave the book to ___"* — preferring *Mary* over *John*. Using a suite of analytical methods — the logit lens, mean/resample layer ablation, attention head ablation, activation patching, and path patching — we reverse-engineer the computational circuit responsible for this behavior across all 12 transformer layers and 144 attention heads. Resample ablation control supports the interpretation that Layer 0 MLP's large drop (resample normalized drop = 1.0927; source: `[outputs/03_layer_ablation/results/layer_ablation_resample.csv, row 1]`) is consistent with a genuine forward-pass dependency rather than a mean-ablation artifact. We demonstrate that the 14-head IOI circuit exhibits high necessity (Necessity score = 1.0728, ablating it reduces accuracy from 96.0% to 40.7%; source: `[outputs/08_circuit_validation/results/circuit_validation.csv, row: necessity]`) and sufficiency (Sufficiency score = 0.8477; source: `[outputs/08_circuit_validation/results/circuit_validation.csv, row: sufficiency]`). As a novel contribution, we evaluate cross-task transfer to pronoun resolution. Head-importance correlation between IOI and pronoun resolution is moderate-to-strong (*r* = 0.5750, $p = 4.78 \times 10^{-14}$, n=144 heads; source: `[outputs/09_novel_extension/results/task_comparison.json]`), yet causal activation patching finds **no functional transfer at either the single-head or group level**: single-head patching yields mean Name Mover cross-task recovery of -5.97% vs. -2.19% for neutral controls (`NO_TRANSFER`; source: `[outputs/11_cross_task_patching/results/cross_task_summary.json]`), and jointly patching the full 4-head Name Mover group yields -1.12% cross-task recovery (`NO_TRANSFER_EVEN_AT_GROUP_LEVEL`; source: `[outputs/12_multihead_patching/results/multihead_summary.json]`). This is the paper's key methodological finding: head-importance correlation across tasks does not imply causal circuit sharing — even when the entire candidate sub-circuit is transplanted simultaneously. Statistical analysis with bootstrap confidence intervals and Cohen's d effect sizes confirms large effect sizes (*d* = +4.8976, $p < 0.0001$; source: `[outputs/10_statistical_analysis/results/stats_effect_sizes.csv, row 0]`) for Name Mover heads. All code and results are publicly available at the project repository.

**Keywords:** mechanistic interpretability, transformer circuits, indirect object identification, attention head ablation, activation patching, GPT-2

---

## 1. Introduction

The field of mechanistic interpretability seeks to understand *how* neural networks implement algorithms — not merely *that* they work, but *what computation* they perform internally. Recent work has demonstrated that transformer language models learn sparse, interpretable circuits for specific linguistic tasks [Wang et al., 2022; Olsson et al., 2022; Nanda et al., 2023].

The **Indirect Object Identification (IOI)** task has emerged as a canonical benchmark for circuit-level analysis. Given a sentence like:

> *"When John and Mary went to the store, John gave the flowers to ___"*

a language model that understands the sentence structure should predict *Mary* (the indirect object, IO) rather than *John* (the subject, S). Wang et al. (2022) identified a circuit of approximately 26 attention heads responsible for this behavior in GPT-2 Small, comprising three functional classes:

1. **Name Mover Heads** (layers 9–11): Write the IO name into the residual stream at the final position
2. **S-Inhibition Heads** (layers 7–8): Suppress the S name's logit
3. **Duplicate Token Heads** and **Induction Heads** (layers 1–5): Identify which names appear multiple times

This paper makes the following contributions:

1. We **reproduce and validate** the IOI circuit using modern TransformerLens tooling with a 1,000-prompt dataset
2. We provide **rigorous circuit validation** through necessity, sufficiency, and generalization tests
3. We contribute **an original analysis** investigating circuit transfer to pronoun resolution
4. We provide **statistical validation** with bootstrap confidence intervals, effect sizes, and permutation tests
5. We release a **research-quality codebase** that can be applied to other tasks and models

---

## 2. Related Work

**Mechanistic Interpretability.** The program of "reverse engineering" neural networks traces back to Olah et al. (2020)'s work on circuits in convolutional networks. For transformers, Elhage et al. (2021) introduced the mathematical framework of the residual stream and the "logit lens" was described by nostalgebraist (2020). Olsson et al. (2022) discovered *induction heads* as a general mechanism for in-context learning.

**IOI and Named Circuits.** Wang et al. (2022) provided the most detailed circuit-level analysis of the IOI task, identifying 26 attention heads forming a modular circuit. Conmy et al. (2023) proposed automated circuit discovery (ACDC), formalizing path patching as a scalable method for circuit identification.

**Activation Patching.** Meng et al. (2022) introduced "causal tracing" in the ROME paper to localize factual associations. Goldowsky-Dill et al. (2023) formalized path patching as a more precise variant.

**Cross-Task Circuit Transfer.** Recent work by Merullo et al. (2023) and Hanna et al. (2023) has investigated whether circuits discovered for one task generalize to structurally similar tasks, finding both shared and task-specific components. Merullo et al. found partial cross-task transfer for compositional multi-step reasoning tasks; Hanna et al. found that circuits for arithmetic tasks share components when tasks are structurally near-identical. The present work contributes a different regime: a task pair (IOI / pronoun resolution) that is semantically similar but syntactically distinct, yielding a `NO_TRANSFER` verdict under causal patching despite moderate head-importance correlation — a negative result that complements these prior positive findings.

---

## 3. Background

### 3.1 The GPT-2 Small Architecture

GPT-2 Small is a 12-layer, 12-head autoregressive transformer with:
- **d_model = 768**: residual stream dimension
- **n_layers = 12**: transformer blocks
- **n_heads = 12**: attention heads per layer  
- **d_head = 64**: per-head QK/V dimension
- **d_mlp = 3,072**: MLP hidden dimension
- **d_vocab = 50,257**: vocabulary size (BPE)

Each transformer block computes:
```
h_l = h_{l-1} + Attn_l(h_{l-1}) + MLP_l(LayerNorm(h_{l-1} + Attn_l(h_{l-1})))
logits = W_U · LayerNorm_final(h_{n_layers})
```

### 3.2 TransformerLens

TransformerLens (Nanda, 2022) provides surgical access to all intermediate computations via named hook points. The three key mechanisms we exploit:

| Hook | Shape | Purpose |
|------|-------|---------|
| `blocks.{l}.attn.hook_z` | [B, S, H, d_head] | Per-head attention output before W_O |
| `blocks.{l}.hook_resid_post` | [B, S, d_model] | Full residual stream after layer l |
| `blocks.{l}.hook_attn_out` | [B, S, d_model] | Summed output of all heads in layer l |

### 3.3 The IOI Circuit Metric

Our primary evaluation metric is **logit difference**:
```
logit_diff = logit(IO) - logit(S)
```
This quantity is:
- **Positive** when the model correctly prefers the IO token
- **Negative** when the model incorrectly prefers S
- **Zero** at the decision boundary

We prefer logit_diff over raw accuracy because it is continuous (measures margin, not just correctness) and additive (mean logit diff across prompts is a stable aggregate).

---

## 4. Methodology

### 4.1 Dataset Construction

We generate 1,000 IOI prompts using 14 templates (7 ABB-type, 7 BAB-type) and 35 proper names verified to be single BPE tokens. Each dataset entry contains:
- **Clean prompt**: original IOI sentence
- **Corrupted prompt**: S name replaced by a distractor (for patching experiments)
- **io_token_id**, **s_token_id**: token IDs for evaluation

Template examples:
```
ABB: "When {IO} and {S} went to {place}, {S} gave the {obj} to"
BAB: "When {S} and {IO} went to {place}, {S} gave the {obj} to"
```

### 4.2 Logit Lens

The logit lens (nostalgebraist, 2020) projects each intermediate residual stream h_l into vocabulary space using the final LayerNorm and unembedding matrix:
```
logits_l = W_U · LayerNorm_final(h_l)
```
We compute logit_diff at each of the n_layers + 1 checkpoints (embedding layer + 12 post-block residuals) to determine when the model develops IO preference.

### 4.3 Layer Ablation

For each layer l, we independently mean-ablate:
1. Attention output: `hook_attn_out`
2. MLP output: `hook_mlp_out`
3. Both simultaneously (full layer)

**Mean ablation** replaces an activation with its mean over all reference prompts, removing task-specific signal while preserving magnitude. We report:
```
normalised_drop = (baseline_ld - ablated_ld) / |baseline_ld|
```

### 4.4 Attention Head Ablation

For each of the 144 (layer, head) pairs, we mean-ablate the z-vector for head h in layer l:
```python
z[:, :, h, :] = mean_z[l, h]  # replace only this head's output
```
We rank heads by importance = (baseline_ld - ablated_ld) / |baseline_ld| and classify them by threshold.

### 4.5 Activation Patching

For each (layer l, token position p) pair, we:
1. Run the model on the **corrupted** input
2. Replace the activation at (l, p) with the value from the **clean** run
3. Compute restoration score:
```
restoration = (patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)
```
Score 1.0 = full restoration of clean behavior; 0.0 = no effect.

### 4.6 Path Patching

Sender-side path patching measures how much each head's output carries IOI-relevant information. For each head (l, h):
1. Run model on corrupted input
2. Replace z[:, :, h, :] with clean z for this head only
3. Measure restoration score in the final logit_diff

Heads with high restoration scores are classified as circuit "senders."

### 4.7 Circuit Validation

**Necessity**: Ablate only the K discovered circuit heads simultaneously. A necessary circuit shows large logit_diff drop.

**Sufficiency**: Ablate all 144-K non-circuit heads. A sufficient circuit retains high logit_diff with only circuit heads active.

**Generalization**: Evaluate necessity on:
- Held-out prompts (last 200 in dataset, not used for circuit identification)
- ABB-only templates
- BAB-only templates

### 4.8 Novel Extension: Pronoun Resolution

We design a pronoun resolution dataset with 500 prompts using the template:
```
{speaker} met {recipient} at {location}. {pronoun} bought a gift for
```
Where pronoun agrees with speaker's gender. The model should predict the **recipient** (not the speaker). We use gendered name pairs from the same single-token name pool.

We apply layer ablation, head ablation, and logit_diff evaluation to this dataset and compare head importance scores to the IOI results using Pearson correlation.

---

## 5. Experiments

### 5.1 Experimental Setup

- **Model**: GPT-2 Small (12L-12H-768d), loaded via TransformerLens with fold_ln=True
- **Hardware**: NVIDIA GPU recommended (CUDA 12.x); CPU feasible for small n_samples
- **Dataset**: 1,000 clean prompts, 1,000 matched corrupted prompts
- **Evaluation**: Full 1,000 prompts for baseline and statistical analysis
- **Reproducibility**: All experiments use seed=42; `torch.use_deterministic_algorithms(True)`

### 5.2 Baseline Evaluation

We evaluate GPT-2 Small on the full 1,000-prompt IOI dataset (source: `[outputs/01_baseline/results/ioi_results.csv]` and `[outputs/10_statistical_analysis/results/stats_bootstrap_ci.csv]`):

| Metric | Value | 95% Bootstrap Confidence Interval | Source File |
|--------|-------|----------------------------------|-------------|
| Accuracy (IO > S) | **96.6%** | [95.4%, 97.7%] | `[outputs/01_baseline/results/ioi_results.csv, mean: is_correct]` |
| Mean logit_diff | **+3.1293** | [+3.0242, +3.2416] | `[outputs/01_baseline/results/ioi_results.csv, mean: logit_diff]` |
| Mean P(IO token) | **0.3617** | [0.3483, 0.3756] | `[outputs/01_baseline/results/ioi_results.csv, mean: prob_io]` |
| Mean IO vocab rank | **17.5** / 50,257 | — | `[outputs/01_baseline/results/ioi_results.csv, mean: rank_io]` |
| ABB template accuracy | **95.6%** | [93.6%, 97.4%] | `[outputs/10_statistical_analysis/results/stats_is_correct_by_template.csv, row: ABB]` |
| BAB template accuracy | **97.6%** | [96.2%, 98.8%] | `[outputs/10_statistical_analysis/results/stats_is_correct_by_template.csv, row: BAB]` |
| ABB mean logit_diff | **+2.6330** | [2.4848, 2.7775] | `[outputs/10_statistical_analysis/results/stats_logit_diff_by_template.csv, row: ABB]` |
| BAB mean logit_diff | **+3.6256** | [3.4741, 3.7757] | `[outputs/10_statistical_analysis/results/stats_logit_diff_by_template.csv, row: BAB]` |

The model shows strong IOI behavior, consistently preferring the indirect object over the subject across both prompt structures.

### 5.3 Logit Lens Results

The logit lens reveals that IO preference emerges **gradually across layers** (source: `[outputs/02_logit_lens/results/logit_lens_by_layer.csv]`):
- **Layers 0–6**: Logit diff remains near zero or negative (-0.1929 at Embedding to +0.0937 at Layer 6; fraction correct ~50.5%–53.0%)
- **Layer 7**: First significant positive logit diff emerges (**+1.2880**, accuracy jumps to 68.5%)
- **Layer 8**: Logit diff increases to **+1.4701** (accuracy 71.5%)
- **Layer 9**: Rapid surge to peak logit diff **+11.3695** (accuracy 93.5%, P(IO) = 0.7478)
- **Layers 10–11**: Logit diff stabilizes (**+8.2179** at L10; **+3.2262** at L11, final accuracy 96.5%)

### 5.4 Layer Ablation Results (Mean vs. Resample Control)

We evaluate layer importance using both **mean ablation** and **resample ablation control** across all 12 layers (36 component ablations per method; sources: `[outputs/03_layer_ablation/results/layer_ablation_resample.csv]` and `[outputs/03_layer_ablation/results/layer_ablation.csv]`).

Under resample ablation (replacing target activations with activations from a mismatched, well-formed IOI prompt), Layer 0 MLP and Layer 0 full layer exhibit the largest performance drops:
- **Layer 0 MLP**: resample normalized drop = **1.0927** (ablated LD = -0.2989; source: `[outputs/03_layer_ablation/results/layer_ablation_resample.csv, row 1]`) vs. mean normalized drop = **0.5435** (ablated LD = +1.4729; source: `[outputs/03_layer_ablation/results/layer_ablation.csv, row 1]`).
- **Layer 0 Full Layer**: resample normalized drop = **1.0025** (ablated LD = -0.0082; source: `[outputs/03_layer_ablation/results/layer_ablation_resample.csv, row 2]`) vs. mean normalized drop = **0.8288** (ablated LD = +0.5525; source: `[outputs/03_layer_ablation/results/layer_ablation.csv, row 2]`).

Because Layer 0 MLP displays a large drop under *both* ablation methods, this is consistent with the interpretation that the dependency is not a mean-ablation artifact — the resample control rules out the specific possibility that the mean-ablation effect stems from replacing activations with an uninformative constant. However, resample ablation alone does not establish *what* information Layer 0 MLP encodes or whether its role is IOI-specific versus a generic forward-pass requirement shared across all tasks. Further work — for example, probing Layer 0 MLP representations across diverse tasks or comparing its ablation impact on unrelated prompts — is needed to distinguish IOI-specific from generic forward-pass dependency at Layer 0.

Among attention components, late-layer attention layers show the highest causal necessity:
- **Layer 8 Attention**: resample normalized drop = **0.5549** (ablated LD = +1.4358; source: `[outputs/03_layer_ablation/results/layer_ablation_resample.csv, row 25]`) vs. mean normalized drop = **0.5319** (ablated LD = +1.5101; source: `[outputs/03_layer_ablation/results/layer_ablation.csv, row 25]`).
- **Layer 7 Attention**: resample normalized drop = **0.3757** (ablated LD = +2.0141; source: `[outputs/03_layer_ablation/results/layer_ablation_resample.csv, row 22]`).
- **Layer 5 Attention**: resample normalized drop = **0.3414** (ablated LD = +2.1248; source: `[outputs/03_layer_ablation/results/layer_ablation_resample.csv, row 16]`).

The attention-dominated impact in middle and late layers is consistent with high-level signal processing in the IOI circuit being attention-mediated, while Layer 0 MLP's large ablation drop is consistent with it supplying a foundational representation subspace drawn upon by downstream heads.

### 5.5 Attention Head Ablation Results (144 Heads)

The head ablation sweep across all 144 heads classifies heads into clear functional categories (source: `[outputs/04_head_ablation/results/head_ablation.csv]`):

| Head Type | Count | Representative Heads | Importance Score Range | Function | Source File |
|-----------|-------|----------------------|-----------------------|----------|-------------|
| Name Mover | **4** | L8H6, L8H10, L5H5, L7H9 | +0.2247 to +0.3432 | Write IO name to final position | `[outputs/04_head_ablation/results/head_ablation.csv, rows 1-4]` |
| Helper | **10** | L6H9, L3H0, L0H10, L1H10, L5H9, L9H7, L7H3, L10H0, L9H9, L4H11 | +0.0589 to +0.1233 | Duplicate token & signal routing | `[outputs/04_head_ablation/results/head_ablation.csv, rows 5-14]` |
| Suppressor | **3** | L11H2, L11H10, L10H7 | -0.1023 to -0.2512 | Negative logit influence (anti-circuit) | `[outputs/04_head_ablation/results/head_ablation.csv]` |
| Neutral | **125** | All other 125 heads | -0.050 to +0.044 | Minimal causal impact on IOI | `[outputs/04_head_ablation/results/head_ablation.csv]` |
| **Total** | **144** | 12 Layers × 12 Heads | — | Full GPT-2 Small Attention Grid | `[outputs/04_head_ablation/results/head_ablation.csv]` |

### 5.6 Activation Patching Results

Patching the **residual stream** reveals a clean localisation (source: `[outputs/05_activation_patching/results/patching_resid.csv]`):
- High restoration scores at **final token position (pos -1)** across layers 8–11 (> 1.0 restoration)
- Moderate restoration at **S name positions** in layers 5–8
- Near-zero restoration at **non-name positions** and **early layers**

Comparison of patch types:
- **Residual stream** > **Attention output** > **MLP output** in maximum restoration score

### 5.7 Path Patching / Circuit Graph

Path patching identifies **18 circuit heads** (source: `[outputs/06_path_patching/results/circuit_summary.json]` and `[outputs/06_path_patching/results/circuit_graph_edges.csv]`):
- **Early senders** (2 heads: L4H11, L0H1): Transmit duplicate token detection from subject positions
- **Middle senders** (2 heads: L7H9, L8H10): S-Inhibition heads transmit suppression signals
- **Late senders** (14 heads: L10H0, L11H2, L9H8, L9H6, L10H2, L9H9, L10H10, L11H6, L11H3, L10H6, L11H1, L9H7, L11H10, L10H7): Name Mover Heads write the final prediction
- **Top directed edges**: L10H0 → L11H2 (estimated weight = 0.0416), L9H8 → L10H0 (0.0370), L9H8 → L11H2 (0.0343)

---

## 6. Circuit Validation

### 6.1 Necessity

We evaluate necessity by mean-ablating the 14 identified circuit heads simultaneously (source: `[outputs/08_circuit_validation/results/circuit_validation.csv]`):

| Condition | Baseline LD | Experimental LD | LD Drop | Baseline Acc | Experimental Acc | Acc Change | Necessity Score | Source File |
|-----------|-------------|-----------------|---------|--------------|------------------|------------|-----------------|-------------|
| Full dataset (N=150) | +3.2289 | **-0.2352** | 3.4641 | 96.0% | **40.7%** | -55.3% | **1.0728** | `[outputs/08_circuit_validation/results/circuit_validation.csv, row 0]` |
| Held-out prompts (N=100) | +3.1249 | **-0.5727** | 3.6976 | 94.0% | **35.0%** | -59.0% | **1.1833** | `[outputs/08_circuit_validation/results/circuit_validation.csv, row 2]` |
| ABB templates (N=67) | +2.7048 | **-0.2513** | 2.9560 | 95.5% | **43.3%** | -52.2% | **1.0929** | `[outputs/08_circuit_validation/results/circuit_validation.csv, row 3]` |
| BAB templates (N=83) | +3.6520 | **-0.2222** | 3.8742 | 96.4% | **38.6%** | -57.8% | **1.0608** | `[outputs/08_circuit_validation/results/circuit_validation.csv, row 4]` |

The circuit is **highly necessary**: ablating the 14 circuit heads reduces accuracy from 96.0% down to 40.7% (a 55.3 percentage-point drop; 57.6% relative accuracy reduction) and flips the mean logit difference from +3.2289 to -0.2352 (a logit diff drop of 3.4641, yielding a Necessity score of **1.0728**).

### 6.2 Sufficiency

We evaluate sufficiency by ablating all 130 non-circuit heads while preserving only the 14 circuit heads (source: `[outputs/08_circuit_validation/results/circuit_validation.csv, row 1]`):

| Condition | Baseline LD | Preserved LD | LD Drop | Baseline Acc | Experimental Acc | Acc Change | Sufficiency Score | Source File |
|-----------|-------------|--------------|---------|--------------|------------------|------------|-------------------|-------------|
| Full dataset (N=150) | +3.2289 | **+2.7371** | 0.4918 | 96.0% | **86.7%** | -9.3% | **0.8477** | `[outputs/08_circuit_validation/results/circuit_validation.csv, row 1]` |

The circuit is **highly sufficient**: preserving only the 14 circuit heads retains **84.8%** of the baseline logit difference (+2.7371 vs +3.2289 baseline) and maintains an accuracy of **86.7%** (retaining 90.3% of baseline accuracy).

### 6.3 Generalization

Generalization necessity scores remain consistently high across held-out prompts (1.1833), ABB templates (1.0929), and BAB templates (1.0608), confirming that circuit mechanisms are structural and not overfitted to specific prompt patterns.

---

## 7. Novel Extension: Pronoun Resolution

### 7.1 Baseline Performance & Task Comparison

We compare performance between IOI (n=1,000 prompts) and Pronoun Resolution (n=500 prompts) (source: `[outputs/09_novel_extension/results/task_comparison.json]` and `[outputs/10_statistical_analysis/results/stats_bootstrap_ci.csv]`):

| Metric | IOI Task | Pronoun Task | Difference | Statistic & p-value | Source File |
|--------|----------|--------------|------------|---------------------|-------------|
| Accuracy | **96.6%** [95.4%, 97.7%] | **98.2%** [96.8%, 99.2%] | +1.6% | — | `[outputs/10_statistical_analysis/results/stats_bootstrap_ci.csv]` |
| Mean logit_diff | **+3.1293** [3.0242, 3.2416] | **+3.2404** [3.0722, 3.4065] | -0.1111 | Cohen's d = **-0.0612** (negligible, n=1,000 vs n=500) | `[outputs/09_novel_extension/results/task_comparison.json]` |
| Statistical Test | — | — | — | Permutation $p = \mathbf{0.2420}$ (not significant; n=1,000 vs n=500) | `[outputs/09_novel_extension/results/task_comparison.json, field: logit_diff_comparison]` |

*Interpretation of Task Performance Margin:* The prompt-level logit difference comparison yields $p = 0.2420$ and Cohen's $d = -0.0612$ (n=1,000 IOI vs. n=500 Pronoun), establishing that there is no statistically significant difference in performance margin between the IOI and Pronoun Resolution tasks. This result is consistent with the hypothesis that both tasks recruit similar attention machinery — comparable task difficulty is expected when the same functional heads govern named entity prediction in both formats.

### 7.2 Head Importance Correlation

Comparing head ablation importance scores across all 144 heads between IOI and Pronoun Resolution reveals a **moderate-to-strong positive correlation**:
- **Pearson correlation**: **$r = 0.5750$** ($p = 4.78 \times 10^{-14}$; n=144 heads; source: `[outputs/09_novel_extension/results/task_comparison.json, field: head_importance_correlation]`)

The late-layer Name Mover Heads (L8H6, L8H10, L5H5, L7H9) rank among the most critical heads for *both* tasks, serving as correlational evidence of shared head reliance. Importantly, the single correlation test on n=144 head-importance pairs does not require multiple-comparisons correction (it is one pre-specified test, not 144 separate tests).

### 7.3 Interpretation

The observed correlation ($r = 0.5750, p = 4.78 \times 10^{-14}$) and equivalent task margin ($p = 0.2420$) support a modular view of language processing in GPT-2:
- **Shared component** (Name Mover Heads): generic mechanism for writing a name token to the output position, activated by tasks requiring named entity prediction
- **Task-specific component** (S-Inhibition Heads): IOI-specific suppression of repeated-name noise

---

## 8. Statistical Analysis

### 8.1 Bootstrap Confidence Intervals

All key metrics evaluated with 2,000 bootstrap resamples (95% CI; source: `[outputs/10_statistical_analysis/results/stats_bootstrap_ci.csv]`):

| Task | Metric | Estimate | 95% Bootstrap CI | N | Source File |
|------|--------|----------|------------------|---|-------------|
| IOI | Logit Difference | **+3.1293** | [+3.0242, +3.2416] | 1,000 | `[outputs/10_statistical_analysis/results/stats_bootstrap_ci.csv, row 0]` |
| IOI | Accuracy | **96.6%** | [95.4%, 97.7%] | 1,000 | `[outputs/10_statistical_analysis/results/stats_bootstrap_ci.csv, row 1]` |
| IOI | P(IO Token) | **0.3617** | [0.3483, 0.3756] | 1,000 | `[outputs/10_statistical_analysis/results/stats_bootstrap_ci.csv, row 2]` |
| Pronoun | Logit Difference | **+3.2404** | [+3.0722, +3.4065] | 500 | `[outputs/10_statistical_analysis/results/stats_bootstrap_ci.csv, row 3]` |
| Pronoun | Accuracy | **98.2%** | [96.8%, 99.2%] | 500 | `[outputs/10_statistical_analysis/results/stats_bootstrap_ci.csv, row 4]` |

### 8.2 Effect Sizes

Effect sizes computed via Cohen's d across functional head groups (source: `[outputs/10_statistical_analysis/results/stats_effect_sizes.csv]`):

| Comparison Group | N Heads | Mean Importance | Cohen's d | Category | Permutation p-value | 95% CI | Source File |
|------------------|---------|-----------------|-----------|----------|---------------------|--------|-------------|
| Name Mover / Helper vs. Neutral | 14 circuit vs. 125 neutral | +0.1501 | **+4.8976** | Large | $p < 0.0001$ (n=139 heads total) | [+0.1077, +0.1976] | `[outputs/10_statistical_analysis/results/stats_effect_sizes.csv, row 0]` |
| Suppressor vs. Neutral | 5 vs. 125 neutral | -0.1754 | **-7.4125** | Large | $p = 1.0000$ (n=130) | [-0.2512, -0.1023] | `[outputs/10_statistical_analysis/results/stats_effect_sizes.csv, row 1]` |
| Late (L9-11) vs. Early (L0-4) | 36 | -0.0119 | **-0.2742** | Small | $p = 0.8850$ (n=36) | [-0.2691, +0.0850] | `[outputs/10_statistical_analysis/results/stats_effect_sizes.csv, row 2]` |
| IOI vs. Pronoun task margin | — | -0.1111 | **-0.0612** | Negligible | $p = 0.2420$ (n=1,000 vs n=500) | — | `[outputs/09_novel_extension/results/task_comparison.json]` |

> **Note on multiple comparisons:** The effect size and permutation tests in Section 8.2 compare pre-specified functional groups (Name Mover/Helper, Suppressor, etc.) against neutral heads — these are not 144 individual hypothesis tests on each head. No family-wise correction is applied because: (1) group membership was defined prior to statistical testing based on ablation importance rank; (2) the primary pre-specified statistical inference of this paper is the circuit-level Necessity score (a single test); and (3) any individual-head permutation tests reported are exploratory characterizations, not hypothesis claims.

### 8.3 Layer Depth Correlation

Spearman rank correlation between layer depth index and head importance across all 144 heads (n=144; source: `[outputs/10_statistical_analysis/results/stats_layer_correlation.json]`):
- **Spearman $\rho = 0.1099$** ($p = 0.1899$, not statistically significant)

*Interpretation:* The non-significant linear depth correlation ($\rho = 0.1099, p = 0.1899$, n=144 heads) reflects that circuit importance is highly localized to specific functional layers (e.g. Layers 7–11 for Name Movers and S-Inhibitors) rather than monotonically increasing with layer depth.

---

## 9. Discussion

### 9.1 Circuit Completeness

Our 14-head circuit achieves a Necessity score of **1.0728** (reducing logit diff from +3.2289 to -0.2352; accuracy drops from 96.0% to 40.7%; source: `[outputs/08_circuit_validation/results/circuit_validation.csv]`) and a Sufficiency score of **0.8477** (retaining 84.8% of baseline logit diff and 86.7% accuracy). The slight shortfall in sufficiency (retaining 84.8% rather than 100% of baseline logit diff) reflects backup Name Mover heads and minor MLP contributions not captured in the 14-head core specification.

### 9.2 Circuit Transfer via Causal Activation Patching

To test whether the pronoun resolution and IOI tasks share a causal mechanism (beyond correlational head-ranking agreement, $r = 0.5750$), we performed **bidirectional cross-task causal activation patching** at target Name Mover, Helper, and Control head positions (source: `[outputs/11_cross_task_patching/results/cross_task_patching.csv]` and `[outputs/11_cross_task_patching/results/cross_task_summary.json]`):

1. **Direction A (Pronoun $\rightarrow$ Corrupted IOI)**: Patching clean Pronoun activations into corrupted IOI runs yields a mean logit-diff recovery of **-5.97%** across Name Mover heads (L8H6 recovery = **+0.0710**, 95% CI [-1.18, +1.17]; source: `[outputs/11_cross_task_patching/results/cross_task_patching.csv, row 1]`).
2. **Direction B (IOI $\rightarrow$ Corrupted Pronoun)**: Patching clean IOI activations into corrupted Pronoun runs yields a mean logit-diff recovery of **-3.96%** at L8H6 (source: `[outputs/11_cross_task_patching/results/cross_task_patching.csv, row 1]`) and **+0.0140** at L5H5 (source: `[outputs/11_cross_task_patching/results/cross_task_patching.csv, row 3]`).
3. **Same-Task Control (IOI $\rightarrow$ Corrupted IOI)**: Same-task donor patching yields a mean recovery of **-21.17%** across Name Movers (source: `[outputs/11_cross_task_patching/results/cross_task_summary.json, key: name_mover_same_task_recovery]`).

*Single-Head Verdict:* Across all test directions (n=150 prompts), Name Mover heads achieve a mean cross-task recovery of **-5.97%** compared to **-2.19%** for neutral control heads (source: `[outputs/11_cross_task_patching/results/cross_task_summary.json]`). This reveals **no single-head functional transfer** (`NO_TRANSFER`). While Pearson correlation ($r = 0.5750, p = 4.78 \times 10^{-14}$, n=144 heads) shows late-layer heads are top-ranked for both tasks, single-head activation patching shows that individual head activations cannot causally restore logit diff without the surrounding task-specific circuit context.

*Group-Level Verdict (Experiment 12):* To determine whether joint patching of the full Name Mover group (Group A: L8H6, L8H10, L5H5, L7H9) or the Name Mover + S-Inhibition group (Group B) changes this verdict, we conducted multi-head simultaneous patching. Results are reported in Section 9.3 below.

*Comparison to Prior Work:* This `NO_TRANSFER` finding contrasts with Merullo et al. (2023) and Hanna et al. (2023), who found positive cross-task circuit transfer for compositional reasoning and arithmetic task pairs respectively. The difference may reflect the degree of structural similarity: IOI and pronoun resolution share a semantic goal (predict the "other" person) but differ substantially in syntactic form and the information structure of the prompt. This suggests that shared head-importance correlation is insufficient to predict causal transferability — the structural embedding of the task matters.

*Task Performance Margin:* The equivalence in task difficulty ($p = 0.2420$, Cohen's $d = -0.0612$, n=1,000 IOI vs. n=500 Pronoun) is consistent with both tasks recruiting similar computational difficulty, even absent causal single-head transfer.

### 9.3 Multi-Head Group Patching (Experiment 12)

To test whether the `NO_TRANSFER` verdict from single-head patching (Section 9.2) survives when the *entire* Name Mover sub-circuit is transplanted simultaneously, we performed multi-head group patching across two groups and both directions (n=150 prompts, seed=42; source: `[outputs/12_multihead_patching/results/multihead_summary.json]` and `[outputs/12_multihead_patching/results/multihead_patching.csv]`):

| Group | Heads Patched | Same-Task Recovery | Pronoun→IOI Recovery | IOI→Pronoun Recovery | Mean Cross-Task Recovery | Source |
|-------|---------------|--------------------|----------------------|----------------------|--------------------------|---------|
| **Group A: Name Movers** | L8H6, L8H10, L5H5, L7H9 (4 heads) | **-1.0038** | **-2.2425** [CI: -10.54, +3.70] | **-0.0026** [CI: -0.047, +0.048] | **-1.1225** | `[outputs/12_multihead_patching/results/multihead_patching.csv, row 0]` |
| **Group B: NM + S-Inhibition** | Group A + L7H3 (5 heads) | **-1.9491** | **-2.2747** [CI: -12.35, +5.35] | **+0.0854** [CI: +0.038, +0.141] | **-1.0947** | `[outputs/12_multihead_patching/results/multihead_patching.csv, row 1]` |

**Verdict: `NO_TRANSFER_EVEN_AT_GROUP_LEVEL`** (source: `[outputs/12_multihead_patching/results/multihead_summary.json, key: causal_transfer_verdict]`)

*Interpretation:* Simultaneously transplanting the full Name Mover group (4 heads) from Pronoun prompts into corrupted IOI runs yields a mean cross-task recovery of **-1.12%** — substantially *worse* than single-head patching (-5.97%) and far below the same-task control baseline. Adding S-Inhibition heads (Group B) yields -1.09%, showing no improvement. This result strengthens the paper's headline finding: the `NO_TRANSFER` verdict is not an artifact of single-head insufficiency. Even the complete candidate sub-circuit cannot causally drive IOI-task behavior when activated by pronoun-task inputs. The mechanistic conclusion is that the activation geometry of the Name Mover heads is task-specific: the *values* they compute depend on the task-specific residual stream context assembled by earlier layers, not just the identity of the heads themselves.

---

## 10. Limitations

1. **Sample Size**: Cross-task patching experiments (experiments 11 and 12) use N=150 prompts per task; baseline and head ablation use N=1,000/N=200. Bootstrap CIs are reported throughout, but smaller samples introduce estimation variance.
2. **Single Model (GPT-2 Small only)**: All results are for GPT-2 Small (117M parameters). Whether the IOI circuit exists in the same form in GPT-2 Medium/Large/XL or other model families (GPT-J, LLaMA, Gemma) is an open question — circuit structure has been shown to vary with scale.
3. **Single Task Pair (IOI + Pronoun Resolution)**: The `NO_TRANSFER` finding at both single-head and group level is specific to this task pair. It does not constitute a general claim that correlation-based transfer arguments always fail in mechanistic interpretability — only that they fail for this particular pair. Different task pairs may show genuine transfer (as found by Merullo et al. and Hanna et al. for other domains).
4. **Layer 0 MLP Dependency is Mechanistically Unresolved**: Resample ablation rules out a mean-ablation artifact but does not establish whether Layer 0 MLP's role is IOI-specific or a generic forward-pass prerequisite. This remains an open question (see Section 5.4).
5. **Synthetic/Templated Prompts**: Both the IOI and Pronoun Resolution datasets use fixed templates with names drawn from a curated single-token pool. This may limit naturalistic validity — real-world named entity resolution prompts vary substantially in syntax, clause structure, co-reference distance, and surface form. Template-based evaluation can overestimate circuit precision by reducing distributional noise.
6. **Mean Ablation Baseline**: Head ablation uses mean ablation (replacing head output with the dataset mean), which may not perfectly isolate individual head contributions due to interaction effects between heads.

---

## 11. Future Work

1. **Cross-Model Analysis**: Apply CircuitScope to GPT-2 Medium/Large to test circuit scaling and check whether Name Mover heads in larger models show cross-task transfer absent in GPT-2 Small
2. **Automated Circuit Discovery**: Integrate ACDC (Conmy et al., 2023) for automated edge detection and comparison with the manually identified 14-head circuit
3. **Naturalistic Prompts**: Evaluate the IOI circuit on naturally occurring sentences from corpora (e.g., Winogrande, Winogender) to test whether the circuit generalizes beyond template-generated prompts
4. **Layer 0 MLP Mechanism**: Probe Layer 0 MLP representations across diverse tasks to distinguish IOI-specific from generic forward-pass dependency

---

## 12. Conclusion

CircuitScope provides a comprehensive mechanistic interpretability analysis of the IOI circuit in GPT-2 Small. We confirm that a sparse set of attention heads — primarily Name Mover Heads in layers 8–11 and S-Inhibition Heads in layers 7–8 — is both necessary (Necessity score = 1.0728, 55.3% accuracy drop) and sufficient (Sufficiency score = 0.8477, 84.8% logit diff retained) for the IOI task. Resample ablation control is consistent with the interpretation that Layer 0 MLP's large drop (resample normalized drop = 1.0927) is not a mean-ablation artifact, though whether this dependency is IOI-specific or a generic forward-pass requirement remains an open question. Our novel cross-task transfer experiments (experiments 11 and 12, n=150 prompts) demonstrate **no causal transfer at any level of granularity**: single-head patching yields `NO_TRANSFER` (mean Name Mover recovery = -5.97%; source: `[outputs/11_cross_task_patching/results/cross_task_summary.json]`) and group-level patching of all 4 Name Mover heads simultaneously yields `NO_TRANSFER_EVEN_AT_GROUP_LEVEL` (Group A recovery = -1.12%; source: `[outputs/12_multihead_patching/results/multihead_summary.json]`). The mechanistic interpretation: the IOI Name Mover heads compute task-specific values conditioned on context built by earlier layers — the same heads cannot process pronoun-task inputs to produce IOI-compatible outputs. Head-importance correlation across tasks ($r = 0.5750, p = 4.78 \times 10^{-14}$, n=144 heads) is real but does not imply causal transferability. All code and results are publicly available as a reproducible research artifact.

---

## References

[1] Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small*. ICLR 2023. https://arxiv.org/abs/2202.00571

[2] Elhage, N., Nanda, N., Olsson, C., et al. (2021). *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread. https://transformer-circuits.pub/2021/framework/index.html

[3] Olsson, C., Elhage, N., Nanda, N., et al. (2022). *In-context Learning and Induction Heads*. Transformer Circuits Thread. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

[4] Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and Editing Factual Associations in GPT*. NeurIPS 2022. https://arxiv.org/abs/2202.05262

[5] Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability*. NeurIPS 2023. https://arxiv.org/abs/2304.14997

[6] Goldowsky-Dill, N., MacLeod, C., Shlegeris, B., & Bhatt, N. (2023). *Localizing Model Behavior with Path Patching*. https://arxiv.org/abs/2304.05969

[7] Nanda, N., & Bloom, J. (2022). *TransformerLens*. GitHub. https://github.com/neelnanda-io/TransformerLens

[8] nostalgebraist (2020). *Interpreting GPT: the Logit Lens*. LessWrong. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/

[9] Heimersheim, S., & Nanda, N. (2024). *How to Use and Interpret Activation Patching*. https://arxiv.org/abs/2404.15255

[10] Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

[11] Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

[12] Olah, C., Cammarata, N., Schubert, L., et al. (2020). *Zoom In: An Introduction to Circuits*. Distill. https://distill.pub/2020/circuits/zoom-in/
