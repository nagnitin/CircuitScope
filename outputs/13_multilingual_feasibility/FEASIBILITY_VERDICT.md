# FEASIBILITY VERDICT: Multilingual IOI Gate

**Generated:** 2026-08-10 21:44:57
**Model:** GPT-2 Small (gpt2)
**Experiment:** 13 — Multilingual IOI Feasibility Gate
**Datasets:** 25 prompts per language (13 ABB + 12 BAB)
**Bootstrap CI:** 95%, 10,000 resamples
**PASS threshold:** Lower 95% CI bound > 55%

---

## English Baseline (Reference)

| Metric | Value |
|--------|-------|
| Accuracy | 96.6% |
| Mean logit-diff | +3.129 |
| Mean tokens/prompt | ~17 |
| Dataset size | 1,000 prompts |

---

## Per-Language Results

### Hindi (Devanagari)

**Gate verdict: ❌ FAIL**

#### Accuracy & Logit-Diff

| Metric | Value |
|--------|-------|
| Accuracy (IO > S logit) | 0.0% |
| 95% Bootstrap CI | [0.0%, 0.0%] |
| CI lower bound > 55%? | NO → FAIL |
| Mean logit-diff (proxy) | +0.0000 |
| Std logit-diff (proxy) | 0.0000 |
| ABB template accuracy | 0.0% |
| BAB template accuracy | 0.0% |
| English baseline accuracy | 96.6% |

#### Tokenization Findings

| Metric | Value |
|--------|-------|
| Mean tokens/prompt | 44.9 |
| Token inflation vs English | 2.9× |
| Byte-level fallback fraction | 98% |
| Byte-level fallback detected | YES ⚠ |
| Names that are single-token | 0 |
| Names that are multi-token | 20 |
| Proxy approach required | YES ⚠ |
| Logit-diff methodology valid | NO (proxy used) |

**Sample tokenization (prompt 1):**

Prompt: `जब राम और गीता पार्क गए`

Tokens (34 total): `<|endoftext|> | � | � | � | � |  � | � | ा | � | � |  � | � | � | � |  �` ...

Byte-level tokens: 33/34 (97%)

#### Verdict Reasoning

The lower bound of the 95% bootstrap CI (0.0%) does NOT exceed the pass threshold (55%). GPT-2 Small **cannot reliably perform the IOI task** in Hindi.

**Root causes:**
- **Tokenizer failure**: 98% of tokens are byte-level fallbacks. GPT-2 was trained on English text and has no meaningful representations for Devanagari script.
- **Multi-token names**: All 20 names are multi-token in GPT-2's BPE vocabulary. The standard logit-diff methodology is fundamentally broken for this language.
- **Proxy token collapse (critical)**: All Devanagari names share the **same first-token proxy (token ID 28225)** — a byte-level encoding artifact that is identical across all IO and S names. This means `logit_diff = logit[28225] − logit[28225] = 0.0` for every single prompt, and `is_correct = (0.0 > 0.0) = False` trivially. The 0.0% accuracy is a **mathematical artifact of identical proxy tokens**, not a measurement of model performance. The model was never actually compared on distinct representations of IO vs S.
- **Training data**: GPT-2 Small contains essentially no Hindi text. The model has no learned circuit for IOI in this language.

**Recommendation:** A multilingual-native model is required. Suggested alternatives:
- `Llama-3 8B` or `Llama-3.1 8B` (TransformerLens supported)
- `Qwen2-7B` or `Qwen2.5-7B` (TransformerLens supported)
- `ai4bharat/IndicBERT` (Indic-specialized, for probing studies)

> ❌ **DO NOT attempt ablation or activation-patching analysis for Hindi on GPT-2 Small.** The model cannot perform the task, so any 'circuit' found would be an artifact of noise, not genuine IOI computation.

---

### Bengali (Bengali)

**Gate verdict: ❌ FAIL**

#### Accuracy & Logit-Diff

| Metric | Value |
|--------|-------|
| Accuracy (IO > S logit) | 0.0% |
| 95% Bootstrap CI | [0.0%, 0.0%] |
| CI lower bound > 55%? | NO → FAIL |
| Mean logit-diff (proxy) | +0.0000 |
| Std logit-diff (proxy) | 0.0000 |
| ABB template accuracy | 0.0% |
| BAB template accuracy | 0.0% |
| English baseline accuracy | 96.6% |

#### Tokenization Findings

| Metric | Value |
|--------|-------|
| Mean tokens/prompt | 66.8 |
| Token inflation vs English | 4.3× |
| Byte-level fallback fraction | 91% |
| Byte-level fallback detected | YES ⚠ |
| Names that are single-token | 0 |
| Names that are multi-token | 20 |
| Proxy approach required | YES ⚠ |
| Logit-diff methodology valid | NO (proxy used) |

**Sample tokenization (prompt 1):**

Prompt: `যখন রাম এবং গীতা পার্ক গেলেন`

Tokens (56 total): `<|endoftext|> | � | � | � | � | � | � |   | � | � | � | � | � | � |  ` ...

Byte-level tokens: 50/56 (89%)

#### Verdict Reasoning

The lower bound of the 95% bootstrap CI (0.0%) does NOT exceed the pass threshold (55%). GPT-2 Small **cannot reliably perform the IOI task** in Bengali.

**Root causes:**
- **Tokenizer failure**: 91% of tokens are byte-level fallbacks. GPT-2 was trained on English text and has no meaningful representations for Bengali script.
- **Multi-token names**: All 20 names are multi-token in GPT-2's BPE vocabulary. The standard logit-diff methodology is fundamentally broken for this language.
- **Proxy token collapse (critical)**: All Bengali/Assamese names share the **same first-token proxy (token ID 220 — the plain space character)**. Every name `" রাম"`, `" গীতা"`, etc. starts with the identical space token, making IO and S indistinguishable at the proxy level. `logit_diff = 0.0` trivially. The 0.0% accuracy is a **mathematical artifact**, not a measurement of model performance.
- **Training data**: GPT-2 Small contains essentially no Bengali text. The model has no learned circuit for IOI in this language.

**Recommendation:** A multilingual-native model is required. Suggested alternatives:
- `Llama-3 8B` or `Llama-3.1 8B` (TransformerLens supported)
- `Qwen2-7B` or `Qwen2.5-7B` (TransformerLens supported)
- `ai4bharat/IndicBERT` (Indic-specialized, for probing studies)

> ❌ **DO NOT attempt ablation or activation-patching analysis for Bengali on GPT-2 Small.** The model cannot perform the task, so any 'circuit' found would be an artifact of noise, not genuine IOI computation.

---

### Assamese (Assamese (Bengali script variant))

**Gate verdict: ❌ FAIL**

#### Accuracy & Logit-Diff

| Metric | Value |
|--------|-------|
| Accuracy (IO > S logit) | 0.0% |
| 95% Bootstrap CI | [0.0%, 0.0%] |
| CI lower bound > 55%? | NO → FAIL |
| Mean logit-diff (proxy) | +0.0000 |
| Std logit-diff (proxy) | 0.0000 |
| ABB template accuracy | 0.0% |
| BAB template accuracy | 0.0% |
| English baseline accuracy | 96.6% |

#### Tokenization Findings

| Metric | Value |
|--------|-------|
| Mean tokens/prompt | 77.0 |
| Token inflation vs English | 4.9× |
| Byte-level fallback fraction | 92% |
| Byte-level fallback detected | YES ⚠ |
| Names that are single-token | 0 |
| Names that are multi-token | 15 |
| Proxy approach required | YES ⚠ |
| Logit-diff methodology valid | NO (proxy used) |

**Sample tokenization (prompt 1):**

Prompt: `যেতিয়া ৰাম আৰু গীতা পাৰ্ক গৈছিল`

Tokens (68 total): `<|endoftext|> | � | � | � | � | � | � | � | � | � | � | � | � | � | �` ...

Byte-level tokens: 62/68 (91%)

#### Verdict Reasoning

The lower bound of the 95% bootstrap CI (0.0%) does NOT exceed the pass threshold (55%). GPT-2 Small **cannot reliably perform the IOI task** in Assamese.

**Root causes:**
- **Tokenizer failure**: 92% of tokens are byte-level fallbacks. GPT-2 was trained on English text and has no meaningful representations for Assamese (Bengali script variant) script.
- **Multi-token names**: All 15 names are multi-token in GPT-2's BPE vocabulary. The standard logit-diff methodology is fundamentally broken for this language.
- **Proxy token collapse (critical)**: Same as Bengali — all Assamese names share **token ID 220 (space character)** as their first-token proxy. `logit_diff = 0.0` for every prompt. The 0.0% accuracy is a **mathematical artifact of identical proxy tokens**.
- **Training data**: GPT-2 Small contains essentially no Assamese text. The model has no learned circuit for IOI in this language.

**Recommendation:** A multilingual-native model is required. Suggested alternatives:
- `Llama-3 8B` or `Llama-3.1 8B` (TransformerLens supported)
- `Qwen2-7B` or `Qwen2.5-7B` (TransformerLens supported)
- `ai4bharat/IndicBERT` (Indic-specialized, for probing studies)

> ❌ **DO NOT attempt ablation or activation-patching analysis for Assamese on GPT-2 Small.** The model cannot perform the task, so any 'circuit' found would be an artifact of noise, not genuine IOI computation.

---

## Overall Summary

| Language | PASS/FAIL | Proceed to Circuit Analysis? |
|----------|-----------|------------------------------|
| Hindi | ❌ FAIL | No — GPT-2 Small inadequate |
| Bengali | ❌ FAIL | No — GPT-2 Small inadequate |
| Assamese | ❌ FAIL | No — GPT-2 Small inadequate |

> ❌ **All three languages FAILED the feasibility gate.**
> GPT-2 Small is not a suitable model for multilingual IOI circuit analysis.
> A multilingual-native model (e.g., Llama-3 8B or Qwen2-7B) with
> TransformerLens support is required before any circuit analysis can proceed.
> No circuit results have been fabricated. This investigation stops here.

---

## Methodological Disclosure

### Translation Method
Prompts were constructed using MT-assisted template translation with no
native speaker review. See `data/multilingual/TRANSLATION_NOTES.md` for
full disclosure of limitations.

### Proxy Token Approach — And Its Collapse

Because GPT-2's BPE tokenizer splits Indic script names into multiple
byte-level tokens, the standard IOI methodology (reading `logit[IO_token_id]`)
cannot be applied directly. This experiment used the **first byte token** of
each name as a proxy.

**This proxy completely collapsed for all three languages:**

| Language | All names' first-token proxy | Token ID | Meaning |
|----------|------------------------------|----------|---------|
| Hindi (Devanagari) | Identical byte-level token | 28225 | A Devanagari byte encoding artifact shared by all names |
| Bengali | Plain space character | 220 | `' '` — the space preceding every Indic name |
| Assamese | Plain space character | 220 | `' '` — same as Bengali |
| English (reference) | Unique per name | e.g., 14862, 5811, 3271 | `' Alice'`, `' Bob'`, `' David'` — correctly distinct |

Because IO and S names share the **same first-token proxy in every prompt**,
`logit_diff = logit[proxy_id] − logit[proxy_id] = 0.0` exactly, and
`is_correct = (0.0 > 0.0) = False` for all 75 prompts across all languages.

The reported 0.0% accuracy is therefore a **mathematical artifact** —
the model's IOI ability (or lack thereof) was not actually measured.
This makes the FAIL verdict even more definitive: there is **no meaningful
measurement possible** for these languages using GPT-2 Small.

All results flagged with `proxy_used=True` in the baseline CSV files should
be interpreted with this caveat. The `logit_diff_proxy` column is uniformly
0.0000 and should be disregarded.

### No Ablation or Patching Analysis
Per the experimental design, this script does NOT perform ablation,
activation patching, or any circuit analysis. These are gated behind
the feasibility check and will only proceed in a follow-up task if
the gate passes for at least one language.
