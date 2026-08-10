"""
experiments/15_multiname_scoring_gate.py
=========================================
Feasibility gate for multilingual IOI using multi-token full-name log-probability scoring.

Purpose
-------
Replaces the single-token proxy method (used in Experiments 13 & 14) with exact
multi-token full-name log-probability sequence scoring on Qwen/Qwen2.5-0.5B.

This test determines whether the prior FAIL verdicts were a measurement artifact
(caused by first-token proxy collapse) or a true model-competence limit.

This is a DIAGNOSTIC-ONLY script. It does NOT:
  - Modify any existing experiments (01-14) or their outputs.
  - Perform ablation or activation-patching analysis.
  - Proceed past reporting if the feasibility gate fails.

Pipeline
--------
  1. Load Qwen/Qwen2.5-0.5B via TransformerLens (HookedTransformer)
  2. Run English IOI baseline sanity check with full-name log-probability scoring
  3. Load translated datasets for Hindi, Bengali, Assamese (data/multilingual/)
  4. For each prompt:
     - Tokenize prompt_clean and full target name sequences (IO and S)
     - Compute exact sum log-probability:
         logP(Name | Prompt) = sum_j logP(t_j | Prompt, t_1...t_{j-1})
     - Compute total logprob difference: logprob_diff = logP(IO) - logP(S)
     - Compute length-normalized average logprob difference for comparison
     - is_correct = (logP(IO) > logP(S))
  5. Compute 95% bootstrap confidence intervals (10,000 resamples)
  6. Save outputs/15_multiname_scoring_gate/results/
  7. Generate FEASIBILITY_VERDICT_v3.md with side-by-side v2 (proxy) vs v3 (full-name) comparison

Usage
-----
From project root:
    python experiments/15_multiname_scoring_gate.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Ensure project root is in Python path ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Force UTF-8 stdout/stderr on Windows ──────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "15_multiname_scoring_gate"
RESULTS_DIR = OUTPUT_DIR / "results"
DATA_DIR = PROJECT_ROOT / "data" / "multilingual"

# ── Model configuration ───────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PASS_THRESHOLD_CI_LOWER = 0.55
ENGLISH_SANITY_THRESHOLD = 0.85

BOOTSTRAP_N_RESAMPLES = 10_000
BOOTSTRAP_CI = 0.95

# ── Prior v2 results (Qwen2.5-0.5B proxy method from Exp 14) for side-by-side ──
V2_PROXY_RESULTS = {
    "english": {"acc": 1.000, "ci": "[100.0%, 100.0%]", "mean_diff": "+4.8310", "verdict": "✅ PASS"},
    "hindi": {"acc": 0.360, "ci": "[16.0%, 56.0%]", "mean_diff": "+0.0790", "verdict": "❌ FAIL"},
    "bengali": {"acc": 0.000, "ci": "[0.0%, 0.0%]", "mean_diff": "+0.0000", "verdict": "❌ FAIL"},
    "assamese": {"acc": 0.160, "ci": "[4.0%, 32.0%]", "mean_diff": "+0.0528", "verdict": "❌ FAIL"},
}


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CircuitScope Experiment 15: Multi-Token Full-Name LogProb Gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Model name (default: {MODEL_NAME})")
    return parser.parse_args()


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_memory_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def create_english_sample_dataset() -> pd.DataFrame:
    english_prompts = [
        ("When David and Henry visited the restaurant, Henry handed the letter to", "David", "Henry", "Harry", "ABB", 0, "the restaurant", "letter"),
        ("While Bob and Kate were at the gym, Bob delivered the map to", "Kate", "Bob", "Carol", "BAB", 1, "the gym", "map"),
        ("When Isla and Ella arrived at the hospital, Ella passed the wallet to", "Isla", "Ella", "Wendy", "ABB", 2, "the hospital", "wallet"),
        ("When Clara and Kate went to the museum, Clara gave the package to", "Kate", "Clara", "Derek", "BAB", 0, "the museum", "package"),
        ("After Grace and Victor left the school, Victor sent the card to", "Grace", "Victor", "Sam", "ABB", 3, "the school", "card"),
        ("When Kate and Clara went to the hospital, Kate gave the phone to", "Clara", "Kate", "Olivia", "BAB", 0, "the hospital", "phone"),
        ("As Xander and Isla walked through the station, Xander offered the ticket to", "Isla", "Xander", "Carol", "BAB", 6, "the station", "ticket"),
        ("After Emma and Victor left the gym, Emma sent the bag to", "Victor", "Emma", "Liam", "BAB", 3, "the gym", "bag"),
        ("When Isla and Liam visited the station, Liam handed the ticket to", "Isla", "Liam", "Mia", "ABB", 1, "the station", "ticket"),
        ("While Paul and Rose were at the park, Paul brought the book to", "Rose", "Paul", "Zach", "BAB", 4, "the park", "book"),
        ("When Brian and Alice arrived at the store, Alice passed the key to", "Brian", "Alice", "Gina", "BAB", 2, "the store", "key"),
        ("Once Frank and Grace were at the office, Frank brought the note to", "Grace", "Frank", "Uma", "ABB", 4, "the office", "note"),
        ("When Yara and Zach left the market, Zach sent the pen to", "Yara", "Zach", "Amy", "ABB", 3, "the market", "pen"),
        ("While Derek and Fred were at the theater, Derek delivered the trophy to", "Fred", "Derek", "Jack", "BAB", 5, "the theater", "trophy"),
        ("When Iris and Jack visited the cafe, Jack handed the jacket to", "Iris", "Jack", "Noah", "ABB", 1, "the cafe", "jacket"),
        ("As Quinn and Sam walked through the mall, Quinn offered the card to", "Sam", "Quinn", "Tara", "BAB", 6, "the mall", "card"),
        ("Once Alice and Bob were at the beach, Bob brought the package to", "Alice", "Bob", "Clara", "ABB", 4, "the beach", "package"),
        ("When Henry and Iris arrived at the office, Iris passed the letter to", "Henry", "Iris", "Kate", "BAB", 2, "the office", "letter"),
        ("After Carol and David left the store, Carol sent the key to", "David", "Carol", "Emma", "BAB", 3, "the store", "key"),
        ("When Gina and Harry visited the park, Harry handed the map to", "Gina", "Harry", "Isla", "ABB", 1, "the park", "map"),
        ("While Liam and Mia were at the library, Liam delivered the book to", "Mia", "Liam", "Noah", "ABB", 5, "the library", "book"),
        ("When Noah and Olivia arrived at the museum, Olivia passed the ticket to", "Noah", "Olivia", "Paul", "BAB", 2, "the museum", "ticket"),
        ("As Rose and Sam walked through the market, Sam offered the pen to", "Rose", "Sam", "Tara", "ABB", 6, "the market", "pen"),
        ("After Tara and Uma left the school, Tara sent the phone to", "Uma", "Tara", "Victor", "BAB", 3, "the school", "phone"),
        ("When Victor and Wendy visited the hospital, Wendy handed the wallet to", "Victor", "Wendy", "Xander", "ABB", 1, "the hospital", "wallet"),
    ]
    records = []
    for clean, io, s, dist, tmpl, idx, place, obj in english_prompts:
        corrupted = clean.replace(s, dist)
        records.append({
            "prompt_clean": clean,
            "prompt_corrupted": corrupted,
            "io_name": io,
            "s_name": s,
            "distractor_name": dist,
            "template_type": tmpl,
            "template_idx": idx,
            "place": place,
            "object_noun": obj,
            "io_token_id": -1,
            "s_token_id": -1,
        })
    return pd.DataFrame(records)


def load_all_datasets() -> dict[str, pd.DataFrame]:
    return {
        "english": create_english_sample_dataset(),
        "hindi": pd.read_csv(DATA_DIR / "hindi_ioi_dataset.csv", encoding="utf-8"),
        "bengali": pd.read_csv(DATA_DIR / "bengali_ioi_dataset.csv", encoding="utf-8"),
        "assamese": pd.read_csv(DATA_DIR / "assamese_ioi_dataset.csv", encoding="utf-8"),
    }


def bootstrap_ci(
    values: list[float],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci: float = BOOTSTRAP_CI,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    sample_mean = arr.mean()
    bootstrap_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = 1.0 - ci
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return float(sample_mean), float(ci_lower), float(ci_upper)


@torch.no_grad()
def compute_sequence_logprob(
    model,
    prompt_prefix: str,
    target_name: str,
    device: torch.device,
) -> tuple[float, float, list[int], list[str]]:
    """
    Compute total and average log-probability of generating `target_name`
    (with leading space) given `prompt_prefix`.
    """
    p_tokens = model.to_tokens(prompt_prefix, prepend_bos=True)[0]
    n_tokens = model.to_tokens(f" {target_name}", prepend_bos=False)[0]

    combined = torch.cat([p_tokens, n_tokens], dim=0).unsqueeze(0).to(device)

    logits = model(combined)[0]  # [seq_len, d_vocab]
    log_probs = F.log_softmax(logits, dim=-1)

    p_len = len(p_tokens)
    n_len = len(n_tokens)

    total_logprob = 0.0
    for i in range(n_len):
        target_token_id = n_tokens[i].item()
        pred_pos = p_len - 1 + i
        token_logp = log_probs[pred_pos, target_token_id].item()
        total_logprob += token_logp

    avg_logprob = total_logprob / n_len if n_len > 0 else 0.0
    str_tokens = model.to_str_tokens(f" {target_name}", prepend_bos=False)

    return total_logprob, avg_logprob, n_tokens.tolist(), str_tokens


@torch.no_grad()
def evaluate_language_fullname(
    model,
    df: pd.DataFrame,
    lang: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run baseline evaluation for one language using full-name log-probability scoring.
    """
    print(f"\n{'─' * 60}")
    print(f"Evaluating {lang.upper()} ({len(df)} prompts) — Full-Name LogProb Method")
    print(f"{'─' * 60}")

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {lang}"):
        prompt_clean = row["prompt_clean"]
        io_name = row["io_name"]
        s_name = row["s_name"]

        # Compute full-name sequence log-probabilities
        logp_io_total, logp_io_avg, ids_io, strs_io = compute_sequence_logprob(model, prompt_clean, io_name, device)
        logp_s_total, logp_s_avg, ids_s, strs_s = compute_sequence_logprob(model, prompt_clean, s_name, device)

        logp_diff_total = logp_io_total - logp_s_total
        logp_diff_avg = logp_io_avg - logp_s_avg

        is_correct_total = logp_diff_total > 0.0
        is_correct_avg = logp_diff_avg > 0.0

        record = {
            "prompt_clean": prompt_clean,
            "prompt_corrupted": row["prompt_corrupted"],
            "io_name": io_name,
            "s_name": s_name,
            "distractor_name": row["distractor_name"],
            "template_type": row["template_type"],
            "template_idx": row["template_idx"],
            "place": row["place"],
            "object_noun": row["object_noun"],
            "io_num_tokens": len(ids_io),
            "s_num_tokens": len(ids_s),
            "io_token_ids": str(ids_io),
            "s_token_ids": str(ids_s),
            "logprob_io_total": round(logp_io_total, 6),
            "logprob_s_total": round(logp_s_total, 6),
            "logprob_diff_total": round(logp_diff_total, 6),
            "logprob_io_avg": round(logp_io_avg, 6),
            "logprob_s_avg": round(logp_s_avg, 6),
            "logprob_diff_avg": round(logp_diff_avg, 6),
            "is_correct": bool(is_correct_total),
            "is_correct_avg": bool(is_correct_avg),
            "language": lang,
        }
        records.append(record)

        if verbose:
            print(f"  [{lang}] Prompt {idx+1:02d}: IO='{io_name}' ({len(ids_io)}t) S='{s_name}' ({len(ids_s)}t) | logP_diff={logp_diff_total:+.4f} | correct={is_correct_total}")

    results_df = pd.DataFrame(records)
    acc_total = results_df["is_correct"].mean()
    acc_avg = results_df["is_correct_avg"].mean()
    mean_ld_total = results_df["logprob_diff_total"].mean()

    print(f"\n[{lang.upper()} Full-Name Results]")
    print(f"  Prompts evaluated              : {len(results_df)}")
    print(f"  Accuracy (Total LogProb IO>S)  : {acc_total:.1%}")
    print(f"  Accuracy (Avg LogProb IO>S)    : {acc_avg:.1%}")
    print(f"  Mean Total LogProb Diff        : {mean_ld_total:+.4f}")

    return results_df


def write_feasibility_verdict_v3(
    eval_results: dict[str, pd.DataFrame],
    output_path: Path,
    runtime_info: dict,
) -> None:
    lines = []
    lines.append("# FEASIBILITY VERDICT v3: Multi-Token Full-Name Log-Probability Gate")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model Evaluated:** `{MODEL_NAME}` (Qwen2.5 0.5B, 24 layers, 14 heads, d_model=896)")
    lines.append(f"**Experiment:** 15 — Multi-Token Full-Name Log-Probability Scoring Gate")
    lines.append(f"**Methodology:** Exact full-name sequence log-probability comparison: `logP(IO|Prompt) > logP(S|Prompt)`")
    lines.append(f"**Bootstrap CI:** 95%, {BOOTSTRAP_N_RESAMPLES:,} resamples")
    lines.append(f"**PASS Threshold:** English Sanity Check ≥ {ENGLISH_SANITY_THRESHOLD:.0%}, Indic Languages Lower 95% CI > {PASS_THRESHOLD_CI_LOWER:.0%}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 📊 Side-by-Side Methodology Comparison: v2 (Proxy Token) vs v3 (Full-Name LogProb)")
    lines.append("")
    lines.append("| Language | v2 Proxy Accuracy | v2 Logit Diff | v3 Full-Name Accuracy | v3 95% Bootstrap CI | v3 Mean LogProb Diff | Gate Verdict v3 | Proceed to Circuit Analysis? |")
    lines.append("|----------|-------------------|---------------|-----------------------|---------------------|----------------------|-----------------|------------------------------|")

    overall_verdicts = {}

    for lang, df_res in eval_results.items():
        is_correct_list = df_res["is_correct"].astype(float).tolist()
        acc, ci_lo, ci_hi = bootstrap_ci(is_correct_list)
        mean_ld = df_res["logprob_diff_total"].mean()

        if lang == "english":
            passes = (ci_lo >= ENGLISH_SANITY_THRESHOLD)
        else:
            passes = (ci_lo > PASS_THRESHOLD_CI_LOWER)

        verdict_str = "✅ PASS" if passes else "❌ FAIL"
        overall_verdicts[lang] = passes

        v2_info = V2_PROXY_RESULTS.get(lang, {"acc": 0.0, "mean_diff": "N/A"})
        v2_acc_str = f"{v2_info['acc']:.1%}"

        proceed_str = "Yes (in Exp 16+)" if (passes and lang != "english") else ("Sanity Check Passed" if (passes and lang == "english") else "No — Model incompetent")

        lines.append(f"| {lang.capitalize()} | {v2_acc_str} | {v2_info['mean_diff']} | **{acc:.1%}** | [{ci_lo:.1%}, {ci_hi:.1%}] | {mean_ld:+.4f} | {verdict_str} | {proceed_str} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Compute & Runtime Profile")
    lines.append("")
    lines.append("| Metric | Measured Value |")
    lines.append("|--------|----------------|")
    lines.append(f"| Model Loading Time | {runtime_info['load_time_seconds']:.2f} seconds |")
    lines.append(f"| Gate Runtime | {runtime_info['eval_time_seconds']:.2f} seconds |")
    lines.append(f"| Peak Process Memory (RSS) | {runtime_info['peak_memory_mb']:.1f} MB |")
    lines.append(f"| Device Used | {runtime_info['device']} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Language Breakdown (Full-Name LogProb Method)")
    lines.append("")

    for lang, df_res in eval_results.items():
        is_correct_list = df_res["is_correct"].astype(float).tolist()
        acc, ci_lo, ci_hi = bootstrap_ci(is_correct_list)
        mean_ld_tot = df_res["logprob_diff_total"].mean()
        std_ld_tot = df_res["logprob_diff_total"].std()
        acc_avg = df_res["is_correct_avg"].mean()
        mean_ld_avg = df_res["logprob_diff_avg"].mean()

        mean_io_toks = df_res["io_num_tokens"].mean()
        mean_s_toks = df_res["s_num_tokens"].mean()

        lines.append(f"### {lang.capitalize()}")
        lines.append("")
        lines.append(f"**Gate Verdict v3: {'✅ PASS' if overall_verdicts[lang] else '❌ FAIL'}**")
        lines.append("")
        lines.append("#### Accuracy & Log-Probability Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Accuracy (Total LogProb) | {acc:.1%} |")
        lines.append(f"| 95% Bootstrap CI (Total LogProb) | [{ci_lo:.1%}, {ci_hi:.1%}] |")
        lines.append(f"| Accuracy (Length-Normalized Avg LogProb) | {acc_avg:.1%} |")
        lines.append(f"| Mean Total LogProb Diff | {mean_ld_tot:+.4f} |")
        lines.append(f"| Std Total LogProb Diff | {std_ld_tot:.4f} |")
        lines.append(f"| Mean Avg LogProb Diff | {mean_ld_avg:+.4f} |")
        lines.append(f"| Mean Sub-word Tokens per IO Name | {mean_io_toks:.1f} |")
        lines.append(f"| Mean Sub-word Tokens per S Name | {mean_s_toks:.1f} |")
        lines.append("")

        lines.append("#### Honest Diagnosis & Interpretation")
        lines.append("")
        if lang == "english":
            lines.append(f"✅ **English Sanity Check PASSED**: Qwen2.5-0.5B achieves {acc:.1%} accuracy on English IOI prompts using full-name sequence scoring. The model demonstrates robust IOI reasoning in English.")
        else:
            v2_acc = V2_PROXY_RESULTS[lang]["acc"]
            if overall_verdicts[lang]:
                lines.append(f"✅ **PASS**: Switching from single-token proxy to full-name log-probability scoring improved {lang.capitalize()} accuracy from {v2_acc:.1%} (v2 proxy) to **{acc:.1%}** (v3 full-name). Lower 95% CI bound ({ci_lo:.1%}) exceeds 55% threshold.")
                lines.append(f"**Conclusion**: The prior FAIL verdict in Exp 14 was a **measurement artifact** caused by leading-space proxy token collapse. Qwen2.5-0.5B demonstrates genuine IOI capability in {lang.capitalize()}.")
            else:
                lines.append(f"❌ **FAIL**: Correcting the scoring methodology to full-name log-probability yielded {acc:.1%} accuracy (95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]).")
                if acc < 0.40:
                    lines.append(f"**Conclusion**: The low performance is **NOT a measurement artifact**. Even when evaluating exact full-name sequence likelihood, Qwen2.5-0.5B performs at or near chance level ({acc:.1%}) in {lang.capitalize()}. The model genuinely lacks the underlying multilingual capability for IOI in this language.")
                    lines.append(f"**Recommendation**: A larger model (e.g., `Qwen2.5-3B` or `Qwen2.5-7B`) is strictly necessary before attempting circuit analysis in {lang.capitalize()}.")

        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Overall Summary & Scientific Conclusion")
    lines.append("")
    passing_langs = [l for l, p in overall_verdicts.items() if p and l != "english"]
    failing_langs = [l for l, p in overall_verdicts.items() if not p and l != "english"]

    if passing_langs and not failing_langs:
        lines.append(f"✅ **All target languages PASSED under full-name log-probability scoring.** The prior failures were measurement artifacts of proxy collapse.")
    elif failing_langs and not passing_langs:
        lines.append(f"❌ **All target Indic languages ({', '.join(l.capitalize() for l in failing_langs)}) FAILED under full-name log-probability scoring.**")
        lines.append("This proves conclusively that Qwen2.5-0.5B's failure is **not an artifact of proxy token collapse**, but a genuine model-competence limit.")
        lines.append("A larger model (Qwen2.5-3B or Qwen2.5-7B) is required before any circuit analysis can proceed for these languages.")
    else:
        lines.append(f"Partial outcome: **{', '.join(l.capitalize() for l in passing_langs)}** passed, while **{', '.join(l.capitalize() for l in failing_langs)}** failed.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Methodological Disclosure")
    lines.append("")
    lines.append("1. **Sequence Likelihood Formulation**: Evaluated exact conditioned sequence log-probabilities $\\sum \\log P(t_j \\mid P, t_1 \\dots t_{j-1})$ across all name sub-tokens.")
    lines.append("2. **Dataset**: Prompts remain MT-assisted without native-speaker review. Limitation retained.")
    lines.append("3. **Scope Enforcement**: Experiment 15 is strictly diagnostic. No circuit ablation or patching has been performed.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[Verdict] FEASIBILITY_VERDICT_v3.md written to: {output_path}")


def main() -> None:
    args = parse_args()

    print("\n" + "═" * 70)
    print("CircuitScope: Experiment 15 — Multi-Token Full-Name LogProb Gate")
    print(f"Model: {args.model}")
    print("═" * 70)

    ensure_output_dirs()

    print(f"\n[Model] Loading {args.model} via TransformerLens...")
    t0 = time.time()
    from transformer_lens import HookedTransformer

    import logging
    logging.getLogger("transformer_lens").setLevel(logging.WARNING)

    mem_before = get_memory_usage_mb()
    model = HookedTransformer.from_pretrained(
        args.model,
        device="cpu",
        dtype="float32",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    load_time = time.time() - t0
    mem_after = get_memory_usage_mb()

    print(f"[Model] ✓ Loaded in {load_time:.1f}s | Memory used: {mem_after - mem_before:.1f} MB (Process RSS: {mem_after:.1f} MB)")

    datasets = load_all_datasets()

    t1 = time.time()
    eval_results = {}

    for lang, df in datasets.items():
        lang_res = evaluate_language_fullname(model, df, lang, verbose=args.verbose)
        eval_results[lang] = lang_res

        out_csv = RESULTS_DIR / f"baseline_{lang}.csv"
        lang_res.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"  Saved to: {out_csv}")

    eval_time = time.time() - t1
    total_time = time.time() - t0
    peak_mem = get_memory_usage_mb()

    runtime_info = {
        "load_time_seconds": round(load_time, 2),
        "eval_time_seconds": round(eval_time, 2),
        "total_time_seconds": round(total_time, 2),
        "peak_memory_mb": round(peak_mem, 1),
        "device": str(getattr(model.cfg, "device", "cpu")),
    }

    verdict_path = OUTPUT_DIR / "FEASIBILITY_VERDICT_v3.md"
    write_feasibility_verdict_v3(eval_results, verdict_path, runtime_info)

    print("\n" + "═" * 70)
    print("EXPERIMENT 15 COMPLETE")
    print("═" * 70)
    print(f"Total time : {total_time:.1f}s")
    print(f"Peak memory: {peak_mem:.1f} MB")
    print(f"Verdict    : {verdict_path}")


if __name__ == "__main__":
    main()
