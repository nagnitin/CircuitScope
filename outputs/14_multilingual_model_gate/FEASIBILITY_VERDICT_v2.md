# FEASIBILITY VERDICT v2: Multilingual Model Gate

**Generated:** 2026-08-10 22:27:13
**Model Evaluated:** `Qwen/Qwen2.5-0.5B` (Qwen2.5 0.5B, 24 layers, 14 heads, d_model=896)
**Experiment:** 14 — Multilingual Model Feasibility Gate
**Dataset Status:** MT-assisted (25 prompts per language). No native-speaker review logged yet.
**Bootstrap CI:** 95%, 10,000 resamples
**PASS Threshold:** English Sanity Check ≥ 85%, Indic Languages Lower 95% CI > 55%

---

## Compute & Runtime Profile

| Metric | Measured Value |
|--------|----------------|
| Model Loading Time | 16.18 seconds |
| Gate Runtime | 33.98 seconds |
| Peak Process Memory (RSS) | 3806.9 MB |
| Device Used | cpu |
| Feasibility for Full Pipeline (Exp 15+) | **HIGH** (Fast load and low memory footprint) |

---

## Results Summary Across Languages

| Language | Accuracy | 95% Bootstrap CI | Proxy Collapse? | Gate Verdict | Proceed to Circuit Analysis? |
|----------|----------|-----------------|-----------------|--------------|------------------------------|
| English | 100.0% | [100.0%, 100.0%] | No | ✅ PASS | Sanity Check Passed |
| Hindi | 36.0% | [16.0%, 56.0%] | YES ⚠ | ❌ FAIL | No |
| Bengali | 0.0% | [0.0%, 0.0%] | YES ⚠ | ❌ FAIL | No |
| Assamese | 16.0% | [4.0%, 32.0%] | YES ⚠ | ❌ FAIL | No |

---

## Detailed Language Breakdown

### English

**Gate Verdict: ✅ PASS**

#### Accuracy & Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 100.0% |
| 95% Bootstrap CI | [100.0%, 100.0%] |
| Mean logit-diff | +4.8310 |
| Std logit-diff | 1.9686 |
| Mean tokens / prompt | 14.8 |
| Token inflation vs English | 1.00x |
| Byte-level fallback ratio | 0.0% |
| Unique space-prefix first tokens | 34 / 34 |
| Unique raw-name first tokens | 33 / 34 |

#### Findings & Root Cause Analysis

✅ **English Sanity Check PASSED**: Qwen2.5-0.5B achieves 100.0% accuracy on English IOI prompts. The model demonstrates clear capability to perform indirect object identification in English.

---

### Hindi

**Gate Verdict: ❌ FAIL**

#### Accuracy & Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 36.0% |
| 95% Bootstrap CI | [16.0%, 56.0%] |
| Mean logit-diff | +0.0790 |
| Std logit-diff | 2.6089 |
| Mean tokens / prompt | 27.6 |
| Token inflation vs English | 1.86x |
| Byte-level fallback ratio | 96.3% |
| Unique space-prefix first tokens | 5 / 20 |
| Unique raw-name first tokens | 12 / 20 |

#### Findings & Root Cause Analysis

⚠ **Proxy Token Collapse Detected**: Mid-sentence space-prefix tokenization collapses 20 unique Hindi names down to only 5 distinct first token IDs (e.g. sharing space byte token ID 35178). This causes IO and S to share identical target token IDs in evaluation.
❌ **FAIL**: Accuracy (36.0%) and lower 95% CI bound (16.0%) do not demonstrate reliable IOI task capability.

---

### Bengali

**Gate Verdict: ❌ FAIL**

#### Accuracy & Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.0% |
| 95% Bootstrap CI | [0.0%, 0.0%] |
| Mean logit-diff | +0.0000 |
| Std logit-diff | 0.0000 |
| Mean tokens / prompt | 37.7 |
| Token inflation vs English | 2.54x |
| Byte-level fallback ratio | 97.3% |
| Unique space-prefix first tokens | 1 / 20 |
| Unique raw-name first tokens | 12 / 20 |

#### Findings & Root Cause Analysis

⚠ **Proxy Token Collapse Detected**: Mid-sentence space-prefix tokenization collapses 20 unique Bengali names down to only 1 distinct first token IDs (e.g. sharing space byte token ID 35178). This causes IO and S to share identical target token IDs in evaluation.
❌ **FAIL**: Accuracy (0.0%) and lower 95% CI bound (0.0%) do not demonstrate reliable IOI task capability.

---

### Assamese

**Gate Verdict: ❌ FAIL**

#### Accuracy & Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 16.0% |
| 95% Bootstrap CI | [4.0%, 32.0%] |
| Mean logit-diff | +0.0528 |
| Std logit-diff | 1.0466 |
| Mean tokens / prompt | 44.7 |
| Token inflation vs English | 3.01x |
| Byte-level fallback ratio | 96.7% |
| Unique space-prefix first tokens | 2 / 15 |
| Unique raw-name first tokens | 11 / 15 |

#### Findings & Root Cause Analysis

⚠ **Proxy Token Collapse Detected**: Mid-sentence space-prefix tokenization collapses 15 unique Assamese names down to only 2 distinct first token IDs (e.g. sharing space byte token ID 35178). This causes IO and S to share identical target token IDs in evaluation.
❌ **FAIL**: Accuracy (16.0%) and lower 95% CI bound (4.0%) do not demonstrate reliable IOI task capability.

---

## Methodological Disclosure & Next Steps

1. **Dataset Status**: All Indic prompts are MT-translated without native-speaker verification. This caveat must be retained in all downstream publications.
2. **Proxy Token Behavior**: Qwen2.5 uses a 151k vocabulary. While it tokenizes sentences efficiently (only 2.9x - 4.9x token count vs 4.5x+ in GPT-2), multi-token Indic names require careful handling to avoid leading-space proxy collapse.
3. **Scope Enforcement**: Experiment 14 is strictly a feasibility gate. No circuit ablation or activation patching has been executed in this step.
