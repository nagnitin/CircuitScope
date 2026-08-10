# Translation Notes — Multilingual IOI Dataset
# Experiment 13: Multilingual Feasibility Gate

## Overview

These datasets contain IOI (Indirect Object Identification) prompts translated 
into Hindi, Bengali, and Assamese for a **feasibility diagnostic only**. 
They are not intended for production circuit analysis.

---

## Translation Method

**Method used: Machine Translation (MT) assisted template construction.**

- English templates were translated using Google Translate and manually 
  reviewed for grammatical plausibility by the experiment author.
- **No native speaker review was performed** for this feasibility run.
- Name pools use common South Asian names rendered in native script.
- Structural elements (conjunctions, verb forms, postpositions) were 
  cross-checked against standard Hindi/Bengali/Assamese grammar references.

> **Disclosure**: Because no native speaker reviewed these translations, 
> minor grammatical errors may exist (especially in verb agreement and 
> postposition placement). This is acceptable for a feasibility diagnostic 
> where the goal is to detect model capability (or lack thereof) rather 
> than to produce publication-quality stimuli. Any follow-up study 
> should use native speaker-verified stimuli.

---

## IOI Task Structure Preservation

Each prompt follows the IOI structural pattern:
- Two names (IO and S) are introduced early in the sentence
- The S name repeats as the agent of the transfer action
- The prompt ends BEFORE the IO name is mentioned in the transfer context
- The model's final-token logit is read to measure P(IO) vs P(S)

### Template Endings by Language

**Hindi** — Prompts end with a past-tense verb:
- `दिया` (gave) — Template 0
- `सौंपा` (handed over) — Template 1  
- `भेजा` (sent) — Template 2
- `दे दिया` (gave away) — Template 3

Note: Hindi SOV order means the recipient (IO) would naturally come 
BEFORE the verb in a complete sentence. The templates end with the verb,
creating a slightly non-standard truncation. The model's logit at the 
verb position is used as a proxy for IOI capability.

**Bengali** — Prompts end with a past-tense verb:
- `দিলেন` (gave) — Templates 0 and 1
- `পাঠালেন` (sent) — Template 2
- `দিয়ে গেলেন` (gave and went) — Template 3

**Assamese** — Prompts end with a past-tense verb:
- `দিলে` (gave) — Templates 0 and 1
- `পঠালে` (sent) — Template 2
- `দি গৈছিল` (gave and went) — Template 3

---

## Limitations of This Dataset

1. **GPT-2 tokenizer will almost certainly fail for all three scripts.**
   The tokenizer was trained on English text and uses byte-level BPE 
   fallback for non-Latin scripts. Each Devanagari/Bengali/Assamese 
   character may split into 2-3 byte tokens, causing massive token 
   sequence inflation and loss of linguistic structure.

2. **Names are multi-token in GPT-2's vocabulary.**
   The English IOI baseline relies on names being single tokens (e.g., 
   " Alice" → token 7653). Indic script names will be 3-8+ tokens each. 
   The evaluation uses the FIRST token of the name as a proxy — this is 
   a disclosed approximation that undermines the logit-diff methodology.

3. **Template truncation is grammatically unusual.**
   Ending the sentence at the verb (SOV languages) is not the same as 
   ending at a preposition (English "gave...to"). The probe position 
   may not elicit name predictions in the same way.

4. **Small dataset (25 prompts per language).**
   Statistical power is limited. Bootstrap CIs will be wide. This is 
   intentional — the diagnostic only needs to determine PASS/FAIL at 
   a coarse level before scaling up.

---

## Name Pools

### Hindi (Devanagari) — 20 names
राम, श्याम, गीता, सीता, मोहन, रवि, अनिल, सुनीता, कमल, विजय,
अर्जुन, देवी, प्रिया, रोहन, नीता, अजय, ललिता, सुरेश, मीना, पूजा

### Bengali (Bengali script) — 20 names
রাম, শ্যাম, গীতা, সীতা, মোহন, রবি, অনিল, সুনীতা, কমল, বিজয়,
অর্জুন, দেবী, প্রিয়া, রোহান, নীতা, অজয়, ললিতা, সুরেশ, মীনা, পূজা

### Assamese (Assamese script) — 15 names
ৰাম, শ্যাম, গীতা, সীতা, মোহন, ৰবি, অনিল, সুনীতা, কমল, বিজয়,
অৰ্জুন, দেৱী, প্ৰিয়া, ৰোহান, নীতা

---

## Recommended Follow-up

If any language passes the feasibility gate, a rigorous follow-up should:
1. Have all prompts reviewed by a native speaker
2. Use a larger prompt set (≥100 prompts)
3. Consider a multilingual-native model (Llama-3 8B, Qwen2-7B)
   with TransformerLens support for circuit analysis
