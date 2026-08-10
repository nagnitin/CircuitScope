# FEASIBILITY VERDICT v3: Multi-Token Full-Name Log-Probability Gate

**Generated:** 2026-08-10 22:42:37
**Model Evaluated:** `Qwen/Qwen2.5-0.5B` (Qwen2.5 0.5B, 24 layers, 14 heads, d_model=896)
**Experiment:** 15 — Multi-Token Full-Name Log-Probability Scoring Gate
**Methodology:** Exact full-name sequence log-probability comparison: `logP(IO|Prompt) > logP(S|Prompt)`
**Bootstrap CI:** 95%, 10,000 resamples
**PASS Threshold:** English Sanity Check ≥ 85%, Indic Languages Lower 95% CI > 55%

---

## 📊 Side-by-Side Methodology Comparison: v2 (Proxy Token) vs v3 (Full-Name LogProb)

| Language | v2 Proxy Accuracy | v2 Logit Diff | v3 Full-Name Accuracy | v3 95% Bootstrap CI | v3 Mean LogProb Diff | Gate Verdict v3 | Proceed to Circuit Analysis? |
|----------|-------------------|---------------|-----------------------|---------------------|----------------------|-----------------|------------------------------|
| English | 100.0% | +4.8310 | **100.0%** | [100.0%, 100.0%] | +4.8408 | ✅ PASS | Sanity Check Passed |
| Hindi | 36.0% | +0.0790 | **96.0%** | [88.0%, 100.0%] | +5.2576 | ✅ PASS | Yes (in Exp 16+) |
| Bengali | 0.0% | +0.0000 | **100.0%** | [100.0%, 100.0%] | +6.7130 | ✅ PASS | Yes (in Exp 16+) |
| Assamese | 16.0% | +0.0528 | **100.0%** | [100.0%, 100.0%] | +7.0069 | ✅ PASS | Yes (in Exp 16+) |

---

## Compute & Runtime Profile

| Metric | Measured Value |
|--------|----------------|
| Model Loading Time | 42.90 seconds |
| Gate Runtime | 279.75 seconds |
| Peak Process Memory (RSS) | 2543.8 MB |
| Device Used | cpu |

---

## Detailed Language Breakdown (Full-Name LogProb Method)

### English

**Gate Verdict v3: ✅ PASS**

#### Accuracy & Log-Probability Metrics

| Metric | Value |
|--------|-------|
| Accuracy (Total LogProb) | 100.0% |
| 95% Bootstrap CI (Total LogProb) | [100.0%, 100.0%] |
| Accuracy (Length-Normalized Avg LogProb) | 100.0% |
| Mean Total LogProb Diff | +4.8408 |
| Std Total LogProb Diff | 1.9766 |
| Mean Avg LogProb Diff | +4.7484 |
| Mean Sub-word Tokens per IO Name | 1.2 |
| Mean Sub-word Tokens per S Name | 1.0 |

#### Honest Diagnosis & Interpretation

✅ **English Sanity Check PASSED**: Qwen2.5-0.5B achieves 100.0% accuracy on English IOI prompts using full-name sequence scoring. The model demonstrates robust IOI reasoning in English.

---

### Hindi

**Gate Verdict v3: ✅ PASS**

#### Accuracy & Log-Probability Metrics

| Metric | Value |
|--------|-------|
| Accuracy (Total LogProb) | 96.0% |
| 95% Bootstrap CI (Total LogProb) | [88.0%, 100.0%] |
| Accuracy (Length-Normalized Avg LogProb) | 88.0% |
| Mean Total LogProb Diff | +5.2576 |
| Std Total LogProb Diff | 3.7878 |
| Mean Avg LogProb Diff | +1.2297 |
| Mean Sub-word Tokens per IO Name | 5.0 |
| Mean Sub-word Tokens per S Name | 4.8 |

#### Honest Diagnosis & Interpretation

✅ **PASS**: Switching from single-token proxy to full-name log-probability scoring improved Hindi accuracy from 36.0% (v2 proxy) to **96.0%** (v3 full-name). Lower 95% CI bound (88.0%) exceeds 55% threshold.
**Conclusion**: The prior FAIL verdict in Exp 14 was a **measurement artifact** caused by leading-space proxy token collapse. Qwen2.5-0.5B demonstrates genuine IOI capability in Hindi.

---

### Bengali

**Gate Verdict v3: ✅ PASS**

#### Accuracy & Log-Probability Metrics

| Metric | Value |
|--------|-------|
| Accuracy (Total LogProb) | 100.0% |
| 95% Bootstrap CI (Total LogProb) | [100.0%, 100.0%] |
| Accuracy (Length-Normalized Avg LogProb) | 96.0% |
| Mean Total LogProb Diff | +6.7130 |
| Std Total LogProb Diff | 2.3342 |
| Mean Avg LogProb Diff | +1.0736 |
| Mean Sub-word Tokens per IO Name | 6.3 |
| Mean Sub-word Tokens per S Name | 6.4 |

#### Honest Diagnosis & Interpretation

✅ **PASS**: Switching from single-token proxy to full-name log-probability scoring improved Bengali accuracy from 0.0% (v2 proxy) to **100.0%** (v3 full-name). Lower 95% CI bound (100.0%) exceeds 55% threshold.
**Conclusion**: The prior FAIL verdict in Exp 14 was a **measurement artifact** caused by leading-space proxy token collapse. Qwen2.5-0.5B demonstrates genuine IOI capability in Bengali.

---

### Assamese

**Gate Verdict v3: ✅ PASS**

#### Accuracy & Log-Probability Metrics

| Metric | Value |
|--------|-------|
| Accuracy (Total LogProb) | 100.0% |
| 95% Bootstrap CI (Total LogProb) | [100.0%, 100.0%] |
| Accuracy (Length-Normalized Avg LogProb) | 92.0% |
| Mean Total LogProb Diff | +7.0069 |
| Std Total LogProb Diff | 3.8722 |
| Mean Avg LogProb Diff | +0.9484 |
| Mean Sub-word Tokens per IO Name | 6.3 |
| Mean Sub-word Tokens per S Name | 6.8 |

#### Honest Diagnosis & Interpretation

✅ **PASS**: Switching from single-token proxy to full-name log-probability scoring improved Assamese accuracy from 16.0% (v2 proxy) to **100.0%** (v3 full-name). Lower 95% CI bound (100.0%) exceeds 55% threshold.
**Conclusion**: The prior FAIL verdict in Exp 14 was a **measurement artifact** caused by leading-space proxy token collapse. Qwen2.5-0.5B demonstrates genuine IOI capability in Assamese.

---

## Overall Summary & Scientific Conclusion

✅ **All target languages PASSED under full-name log-probability scoring.** The prior failures were measurement artifacts of proxy collapse.

---

## Methodological Disclosure

1. **Sequence Likelihood Formulation**: Evaluated exact conditioned sequence log-probabilities $\sum \log P(t_j \mid P, t_1 \dots t_{j-1})$ across all name sub-tokens.
2. **Dataset**: Prompts remain MT-assisted without native-speaker review. Limitation retained.
3. **Scope Enforcement**: Experiment 15 is strictly diagnostic. No circuit ablation or patching has been performed.
