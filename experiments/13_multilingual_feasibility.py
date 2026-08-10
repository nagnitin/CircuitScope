"""
experiments/13_multilingual_feasibility.py
==========================================
Feasibility gate for multilingual extension of CircuitScope IOI analysis.

Purpose
-------
Tests whether GPT-2 Small can perform the Indirect Object Identification (IOI)
task at all in Hindi (Devanagari), Bengali, and Assamese BEFORE any circuit
or ablation analysis is attempted in those languages.

This is a DIAGNOSTIC-ONLY script. It does NOT:
  - Modify any existing experiments (01-12) or their outputs.
  - Perform ablation or activation-patching analysis.
  - Proceed past reporting if the feasibility gate fails.

Pipeline
--------
  1. Load GPT-2 Small via TransformerLens
  2. Load translated IOI datasets for each language
  3. Run tokenization diagnostic (MUST be done before model evaluation)
     - Print actual token strings for sample prompts
     - Detect byte-level fallback for non-Latin scripts
     - Check if IO/S names are single tokens (required for clean logit-diff)
     - Report mean/max tokens per prompt vs English baseline
  4. Save tokenization_report.json
  5. For each language: run baseline accuracy evaluation
     - Compute logit_diff (IO logit - S logit) at final token position
     - NOTE: Indic names are multi-token; first-token proxy is used (disclosed)
     - Compute accuracy (logit_io > logit_s) and bootstrap 95% CI
     - PASS threshold: lower CI bound clearly > 50%
  6. Save baseline_{language}.csv for each language
  7. Write FEASIBILITY_VERDICT.md with honest per-language PASS/FAIL

Usage
-----
From the project root directory:

    python experiments/13_multilingual_feasibility.py

With verbose logging:
    python experiments/13_multilingual_feasibility.py --verbose

Expected Output
---------------
outputs/13_multilingual_feasibility/
├── results/
│   ├── tokenization_report.json
│   ├── baseline_hindi.csv
│   ├── baseline_bengali.csv
│   └── baseline_assamese.csv
└── FEASIBILITY_VERDICT.md

data/multilingual/
├── hindi_ioi_dataset.csv     (pre-built, 25 prompts)
├── bengali_ioi_dataset.csv   (pre-built, 25 prompts)
├── assamese_ioi_dataset.csv  (pre-built, 25 prompts)
└── TRANSLATION_NOTES.md

Key Technical Limitations (disclosed)
--------------------------------------
1. GPT-2 tokenizer uses byte-level BPE fallback for Indic scripts.
   Each Devanagari/Bengali/Assamese character encodes as 2-3 UTF-8 bytes,
   each becoming a separate token. A 17-token English prompt may become
   40-70 tokens in these scripts.

2. IO/S names will almost certainly be multi-token in GPT-2's vocabulary.
   The standard logit-diff methodology (logit[IO_token_id] - logit[S_token_id])
   requires each name to be a SINGLE token. For multi-token Indic names,
   this script uses the FIRST TOKEN of the name as a disclosed proxy.
   This approximation is explicitly flagged in all outputs.

3. GPT-2 Small was trained on English text (~0% Indic script content).
   The model has no meaningful representations for IOI in these languages.
   All three languages are expected to FAIL the gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

# ── Force UTF-8 stdout/stderr on Windows (avoids cp1252 UnicodeEncodeError) ──
# Windows cmd/PowerShell default to cp1252 which cannot encode Unicode box
# characters or any Indic script characters printed to console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "13_multilingual_feasibility"
RESULTS_DIR = OUTPUT_DIR / "results"
DATA_DIR = PROJECT_ROOT / "data" / "multilingual"

# ── English baseline reference (from experiment 01) ──────────────────────────
ENGLISH_BASELINE = {
    "accuracy": 0.966,
    "mean_logit_diff": 3.129,
    "mean_tokens_per_prompt": 17.0,  # approximate from English IOI prompts
}

# ── Language metadata ─────────────────────────────────────────────────────────
LANGUAGES = {
    "hindi": {
        "csv": DATA_DIR / "hindi_ioi_dataset.csv",
        "script": "Devanagari",
        "unicode_range": (0x0900, 0x097F),
    },
    "bengali": {
        "csv": DATA_DIR / "bengali_ioi_dataset.csv",
        "script": "Bengali",
        "unicode_range": (0x0980, 0x09FF),
    },
    "assamese": {
        "csv": DATA_DIR / "assamese_ioi_dataset.csv",
        "script": "Assamese (Bengali script variant)",
        "unicode_range": (0x0980, 0x09FF),
    },
}

# ── Custom JSON encoder (handles numpy scalars) ─────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy scalar types.

    numpy comparison operators (e.g. np.mean(arr) > 0.3) return numpy.bool_,
    not Python's built-in bool. Similarly, numpy integers / floats are not
    serialisable by the default encoder. This encoder converts them on the fly.
    """

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


# ── Feasibility gate threshold ─────────────────────────────────────────────────
# PASS if the LOWER bound of the 95% bootstrap CI on accuracy is clearly > 50%.
# "Clearly" = at least 5 percentage points above chance, i.e., CI_lower > 0.55.
PASS_THRESHOLD_CI_LOWER = 0.55

# ── Bootstrap parameters ──────────────────────────────────────────────────────
BOOTSTRAP_N_RESAMPLES = 10_000
BOOTSTRAP_CI = 0.95


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CircuitScope Experiment 13: Multilingual IOI Feasibility Gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable verbose output"
    )
    parser.add_argument(
        "--n-samples-tok", type=int, default=3,
        help="Number of sample prompts to print in tokenization diagnostic (default: 3)"
    )
    return parser.parse_args()


def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Output directory: {OUTPUT_DIR}")


def load_language_dataset(lang: str) -> pd.DataFrame:
    """
    Load a multilingual IOI dataset CSV.

    Parameters
    ----------
    lang : str
        Language key: 'hindi', 'bengali', or 'assamese'.

    Returns
    -------
    pd.DataFrame
        Dataset with columns matching the English IOI schema.
    """
    csv_path = LANGUAGES[lang]["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found for {lang}: {csv_path}\n"
            f"Expected at: {csv_path.resolve()}"
        )
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"[Dataset] Loaded {lang}: {len(df)} prompts from {csv_path.name}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Tokenization Diagnostic
# ══════════════════════════════════════════════════════════════════════════════

def is_byte_level_token(token_str: str) -> bool:
    """
    Detect whether a token string is a byte-level fallback token.

    GPT-2's byte-level BPE represents non-ASCII characters using a
    special Unicode mapping (see GPT-2's bytes_to_unicode() function).
    Byte-level tokens appear as strange multi-character sequences that
    don't look like English words.

    Heuristic: if the decoded token contains only non-printable or
    high-unicode characters, it's likely a byte-level representation.
    We also check for the GPT-2 byte-to-unicode mapping artifacts.

    Parameters
    ----------
    token_str : str
        The string representation of a token (as decoded by the tokenizer).

    Returns
    -------
    bool
        True if this token looks like a byte-level fallback token.
    """
    if not token_str:
        return False
    # Remove GPT-2's special space marker (Ġ = space prefix)
    cleaned = token_str.lstrip("Ġ").strip()
    if not cleaned:
        return False
    # If token has non-ASCII characters that aren't from the target script,
    # it's likely a byte-level encoding artifact.
    # GPT-2 maps bytes 0-255 to specific Unicode characters (bytes_to_unicode).
    # These appear as single characters in the range U+0100 to U+017E (approx)
    # plus some others. More practically: if cleaned is a single character and
    # not ASCII printable, it's a byte-level token.
    if len(cleaned) <= 2 and not cleaned.isascii():
        return True
    # Check for GPT-2 byte-mapping characters (they appear as Latin Extended
    # characters when non-ASCII bytes are encoded)
    ascii_printable_count = sum(1 for c in cleaned if c.isascii() and c.isprintable())
    if len(cleaned) > 0 and ascii_printable_count == 0 and all(ord(c) < 0x250 for c in cleaned):
        return True
    return False


def is_indic_script_token(token_str: str, unicode_range: tuple[int, int]) -> bool:
    """
    Check if a token contains characters from the target Indic script.

    Parameters
    ----------
    token_str : str
        Token string to check.
    unicode_range : tuple[int, int]
        (start, end) Unicode code point range for the target script.

    Returns
    -------
    bool
        True if any character is in the target Unicode range.
    """
    lo, hi = unicode_range
    return any(lo <= ord(c) <= hi for c in token_str)


def tokenize_prompt_with_strings(model, prompt: str) -> tuple[list[int], list[str]]:
    """
    Tokenize a prompt and return both token IDs and their string representations.

    Parameters
    ----------
    model : HookedTransformer
        GPT-2 model with tokenizer.
    prompt : str
        Input text to tokenize.

    Returns
    -------
    tuple of (token_ids, token_strings)
        - token_ids: list of integer token IDs (with BOS)
        - token_strings: list of decoded string representations
    """
    token_ids = model.to_tokens(prompt, prepend_bos=True)[0].tolist()
    token_strings = model.to_str_tokens(prompt, prepend_bos=True)
    return token_ids, token_strings


def get_first_token_id(model, name: str) -> tuple[int, bool]:
    """
    Get the first token ID for a name (with leading space, as it appears mid-sentence).

    For English names, this is typically the ONLY token (single-token names).
    For Indic script names, the tokenizer will produce multiple byte-level tokens.
    This function returns the FIRST token ID as a proxy, flagging multi-token cases.

    Parameters
    ----------
    model : HookedTransformer
        GPT-2 model.
    name : str
        Name string WITHOUT leading space.

    Returns
    -------
    tuple of (int, bool)
        - first_token_id: the first token ID for " {name}"
        - is_multi_token: True if the name encodes to more than 1 token
    """
    # Names appear mid-sentence with a preceding space in GPT-2 BPE
    token_ids = model.to_tokens(f" {name}", prepend_bos=False)[0].tolist()
    is_multi_token = len(token_ids) > 1
    return token_ids[0], is_multi_token


def run_tokenization_diagnostic(
    model,
    datasets: dict[str, pd.DataFrame],
    n_samples: int = 3,
    verbose: bool = False,
) -> dict:
    """
    Run the tokenization diagnostic for all three languages.

    For each language:
      - Prints actual token strings for n_samples prompts
      - Computes mean/max tokens per prompt
      - Detects byte-level fallback tokens
      - Checks if IO/S names tokenize to single tokens

    Parameters
    ----------
    model : HookedTransformer
        GPT-2 Small model.
    datasets : dict[str, pd.DataFrame]
        Mapping from language name to its dataset DataFrame.
    n_samples : int
        Number of sample prompts to show detailed token strings for.
    verbose : bool
        Print extra detail.

    Returns
    -------
    dict
        Tokenization report (to be saved as JSON).
    """
    print("\n" + "=" * 70)
    print("STEP 1: TOKENIZATION DIAGNOSTIC")
    print("=" * 70)
    print("Checking GPT-2 tokenizer behavior for each language.")
    print("This MUST be done before any model evaluation.\n")

    # English baseline: compute mean tokens for comparison
    english_sample_prompts = [
        "When David and Henry visited the restaurant, Henry handed the letter to",
        "While Bob and Kate were at the gym, Bob delivered the map to",
        "When Isla and Ella arrived at the hospital, Ella passed the wallet to",
    ]
    eng_token_counts = [
        len(model.to_tokens(p, prepend_bos=True)[0]) for p in english_sample_prompts
    ]
    eng_mean_tokens = np.mean(eng_token_counts)

    print(f"[English baseline] Mean tokens per prompt: {eng_mean_tokens:.1f}")
    print(f"[English baseline] (From 3 sample prompts: {eng_token_counts})\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "gpt2 (GPT-2 Small)",
        "tokenizer": "GPT-2 Byte-Level BPE",
        "english_baseline": {
            "mean_tokens_per_prompt": round(eng_mean_tokens, 2),
            "sample_token_counts": eng_token_counts,
        },
        "languages": {},
    }

    for lang, df in datasets.items():
        print(f"\n{'─' * 60}")
        print(f"Language: {lang.upper()} ({LANGUAGES[lang]['script']})")
        print(f"{'─' * 60}")

        unicode_range = LANGUAGES[lang]["unicode_range"]
        lang_report: dict[str, Any] = {
            "script": LANGUAGES[lang]["script"],
            "n_prompts": len(df),
            "sample_tokenizations": [],
            "token_count_stats": {},
            "byte_level_analysis": {},
            "name_token_analysis": {},
        }

        # ── 1. Sample tokenizations ───────────────────────────────────────
        token_counts = []
        byte_level_fractions = []

        for idx, row in df.iterrows():
            prompt = row["prompt_clean"]
            ids, strs = tokenize_prompt_with_strings(model, prompt)
            token_counts.append(len(ids))

            # Count byte-level tokens
            n_byte_level = sum(1 for s in strs if is_byte_level_token(s))
            frac_byte = n_byte_level / len(strs) if strs else 0.0
            byte_level_fractions.append(frac_byte)

        # ── 2. Print detailed token strings for n_samples prompts ─────────
        print(f"\nSample tokenizations ({n_samples} prompts):")
        for i in range(min(n_samples, len(df))):
            row = df.iloc[i]
            prompt = row["prompt_clean"]
            ids, strs = tokenize_prompt_with_strings(model, prompt)

            n_byte_level = sum(1 for s in strs if is_byte_level_token(s))
            frac_byte = n_byte_level / len(strs) if strs else 0.0

            print(f"\n  Prompt {i+1}: {prompt!r}")
            print(f"  Token count: {len(ids)} (English baseline: ~{eng_mean_tokens:.0f})")
            print(f"  Token strings: {strs}")
            print(f"  Byte-level tokens: {n_byte_level}/{len(strs)} ({frac_byte:.0%})")

            sample_entry = {
                "prompt": prompt,
                "token_count": len(ids),
                "token_strings": strs,
                "n_byte_level_tokens": n_byte_level,
                "fraction_byte_level": round(frac_byte, 4),
            }
            lang_report["sample_tokenizations"].append(sample_entry)

        # ── 3. Token count statistics ──────────────────────────────────────
        mean_tok = np.mean(token_counts)
        max_tok = np.max(token_counts)
        min_tok = np.min(token_counts)
        inflation = mean_tok / eng_mean_tokens

        lang_report["token_count_stats"] = {
            "mean_tokens_per_prompt": round(mean_tok, 2),
            "max_tokens_per_prompt": int(max_tok),
            "min_tokens_per_prompt": int(min_tok),
            "english_baseline_mean": round(eng_mean_tokens, 2),
            "token_inflation_factor": round(inflation, 2),
        }

        print(f"\nToken count statistics:")
        print(f"  Mean tokens per prompt : {mean_tok:.1f} "
              f"(English: {eng_mean_tokens:.1f}, "
              f"inflation: {inflation:.1f}×)")
        print(f"  Max / Min              : {max_tok} / {min_tok}")

        # ── 4. Byte-level analysis ─────────────────────────────────────────
        mean_byte_frac = np.mean(byte_level_fractions)
        lang_report["byte_level_analysis"] = {
            "mean_fraction_byte_level_tokens": round(float(mean_byte_frac), 4),
            "byte_level_fallback_detected": bool(mean_byte_frac > 0.3),
            "interpretation": (
                "SEVERE: Most tokens are byte-level fallbacks. "
                "GPT-2 tokenizer cannot handle this script natively."
                if mean_byte_frac > 0.3
                else "MILD: Some byte-level tokens detected."
                if mean_byte_frac > 0.05
                else "MINIMAL: Few byte-level tokens."
            ),
        }

        print(f"\nByte-level fallback analysis:")
        print(f"  Mean fraction of byte-level tokens: {mean_byte_frac:.0%}")
        if mean_byte_frac > 0.3:
            print("  ⚠ SEVERE byte-level fallback detected!")
            print("  GPT-2 tokenizer cannot handle this script natively.")

        # ── 5. Name tokenization analysis ─────────────────────────────────
        unique_io_names = df["io_name"].unique().tolist()
        unique_s_names = df["s_name"].unique().tolist()
        all_unique_names = list(set(unique_io_names + unique_s_names))

        multi_token_names = []
        single_token_names = []
        name_details = {}

        for name in all_unique_names:
            name_token_ids = model.to_tokens(f" {name}", prepend_bos=False)[0].tolist()
            name_strs = model.to_str_tokens(f" {name}", prepend_bos=False)
            is_multi = len(name_token_ids) > 1
            first_tok_id = name_token_ids[0]

            name_details[name] = {
                "n_tokens": len(name_token_ids),
                "token_strings": name_strs,
                "is_multi_token": is_multi,
                "first_token_id": first_tok_id,
            }

            if is_multi:
                multi_token_names.append(name)
            else:
                single_token_names.append(name)

        frac_multi = len(multi_token_names) / len(all_unique_names) if all_unique_names else 0.0

        lang_report["name_token_analysis"] = {
            "n_unique_names_tested": len(all_unique_names),
            "n_single_token_names": len(single_token_names),
            "n_multi_token_names": len(multi_token_names),
            "fraction_multi_token": round(frac_multi, 4),
            "multi_token_names": multi_token_names,
            "single_token_names": single_token_names,
            "name_details": name_details,
            "logit_diff_methodology_valid": len(multi_token_names) == 0,
            "proxy_approach_required": len(multi_token_names) > 0,
        }

        print(f"\nName tokenization analysis:")
        print(f"  Unique names tested       : {len(all_unique_names)}")
        print(f"  Single-token names        : {len(single_token_names)}")
        print(f"  Multi-token names (proxy) : {len(multi_token_names)} ({frac_multi:.0%})")
        if multi_token_names:
            print(f"  ⚠ Multi-token names require first-token proxy approach.")
            print(f"  ⚠ Logit-diff methodology is APPROXIMATE for this language.")

        # Show a few name tokenizations
        print(f"\n  Sample name tokenizations:")
        for name in all_unique_names[:5]:
            details = name_details[name]
            print(f"    {name!r} → {details['token_strings']} "
                  f"({'MULTI-TOKEN' if details['is_multi_token'] else 'single-token'})")

        report["languages"][lang] = lang_report

    return report


# ══════════════════════════════════════════════════════════════════════════════
# Baseline Accuracy Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    values: list[float],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci: float = BOOTSTRAP_CI,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for the mean of a list of values.

    Parameters
    ----------
    values : list[float]
        Sample values (e.g., list of is_correct booleans cast to float).
    n_resamples : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g., 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (mean, ci_lower, ci_upper)
        - mean: sample mean
        - ci_lower: lower bound of the CI
        - ci_upper: upper bound of the CI
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    sample_mean = arr.mean()

    # Draw n_resamples bootstrap samples and compute their means
    bootstrap_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_resamples)
    ])

    alpha = 1.0 - ci
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(sample_mean), float(ci_lower), float(ci_upper)


@torch.no_grad()
def evaluate_language(
    model,
    df: pd.DataFrame,
    lang: str,
    tok_report: dict,
    batch_size: int = 16,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run the IOI baseline evaluation for one language.

    Methodology
    -----------
    For each prompt:
      1. Tokenize the clean prompt (with BOS)
      2. Run a forward pass through GPT-2 Small
      3. Extract logits at the FINAL token position
      4. Look up logit for IO name's first token ID (proxy for multi-token names)
      5. Look up logit for S name's first token ID (proxy for multi-token names)
      6. Compute logit_diff = logit_io_proxy - logit_s_proxy
      7. is_correct = logit_diff > 0

    For multi-token names, the "first token" of each name is used as the
    logit target. This is a DISCLOSED APPROXIMATION — the full name is
    multi-token, so the single logit position does not represent the full
    name probability. Results are flagged accordingly.

    Parameters
    ----------
    model : HookedTransformer
        GPT-2 Small model.
    df : pd.DataFrame
        Language IOI dataset.
    lang : str
        Language name (for logging).
    tok_report : dict
        Tokenization report (used to look up pre-computed first-token IDs).
    batch_size : int
        Batch size for forward passes.
    verbose : bool
        Print per-prompt details.

    Returns
    -------
    pd.DataFrame
        Results with one row per prompt, including all metadata and metrics.
    """
    print(f"\n{'─' * 60}")
    print(f"Evaluating {lang.upper()} ({len(df)} prompts)")
    print(f"{'─' * 60}")

    name_details = tok_report["languages"][lang]["name_token_analysis"]["name_details"]

    # Build a lookup: name -> (first_token_id, is_multi_token)
    # We need this for every io_name and s_name in the dataset
    def get_name_proxy_info(name: str) -> tuple[int, bool]:
        if name in name_details:
            d = name_details[name]
            return d["first_token_id"], d["is_multi_token"]
        else:
            # Compute on the fly if not cached (distractor names, etc.)
            ids = model.to_tokens(f" {name}", prepend_bos=False)[0].tolist()
            return ids[0], len(ids) > 1

    records = []
    prompts = df["prompt_clean"].tolist()
    n = len(prompts)

    # Determine device
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    for batch_start in tqdm(
        range(0, n, batch_size),
        desc=f"Evaluating {lang}",
        unit="batch",
    ):
        batch_end = min(batch_start + batch_size, n)
        batch_prompts = prompts[batch_start:batch_end]
        batch_rows = df.iloc[batch_start:batch_end]

        # ── Tokenize batch (variable-length, pad with BOS) ────────────────
        token_lists = [
            model.to_tokens(p, prepend_bos=True)[0].tolist()
            for p in batch_prompts
        ]
        seq_lengths = [len(t) for t in token_lists]
        max_len = max(seq_lengths)
        bos_id = model.tokenizer.bos_token_id

        padded = [t + [bos_id] * (max_len - len(t)) for t in token_lists]
        tokens_tensor = torch.tensor(padded, dtype=torch.long, device=device)

        # ── Forward pass ──────────────────────────────────────────────────
        all_logits = model(tokens_tensor)  # [batch, seq_len, d_vocab]

        # Extract logits at the final REAL token position for each sequence
        final_logits = torch.stack([
            all_logits[i, seq_lengths[i] - 1, :]
            for i in range(len(batch_prompts))
        ])  # [batch, d_vocab]

        probs = F.softmax(final_logits, dim=-1)

        # ── Per-prompt metrics ────────────────────────────────────────────
        for i, (_, row) in enumerate(batch_rows.iterrows()):
            io_name = row["io_name"]
            s_name = row["s_name"]

            io_token_id, io_is_multi = get_name_proxy_info(io_name)
            s_token_id, s_is_multi = get_name_proxy_info(s_name)

            logits_i = final_logits[i]  # [d_vocab]
            probs_i = probs[i]

            logit_io = logits_i[io_token_id].item()
            logit_s = logits_i[s_token_id].item()
            logit_diff = logit_io - logit_s

            prob_io = probs_i[io_token_id].item()
            prob_s = probs_i[s_token_id].item()

            is_correct = logit_diff > 0

            # Top-5 predictions
            top5_vals, top5_ids = torch.topk(logits_i, k=5)
            top5_tokens = model.tokenizer.batch_decode(
                top5_ids.unsqueeze(-1)
            )
            top5_logits = top5_vals.tolist()

            # IO token rank
            sorted_ids = logits_i.argsort(descending=True)
            rank_io = ((sorted_ids == io_token_id).nonzero(as_tuple=True)[0].item() + 1)

            record = {
                # Dataset metadata
                "prompt_clean": row["prompt_clean"],
                "prompt_corrupted": row["prompt_corrupted"],
                "io_name": io_name,
                "s_name": s_name,
                "distractor_name": row["distractor_name"],
                "template_type": row["template_type"],
                "template_idx": row["template_idx"],
                "place": row["place"],
                "object_noun": row["object_noun"],
                # Token ID info (proxy approach disclosure)
                "io_token_id_proxy": io_token_id,
                "s_token_id_proxy": s_token_id,
                "io_name_is_multi_token": io_is_multi,
                "s_name_is_multi_token": s_is_multi,
                "proxy_used": io_is_multi or s_is_multi,
                # Metrics
                "logit_io_proxy": round(logit_io, 6),
                "logit_s_proxy": round(logit_s, 6),
                "logit_diff_proxy": round(logit_diff, 6),
                "prob_io_proxy": round(prob_io, 8),
                "prob_s_proxy": round(prob_s, 8),
                "is_correct": bool(is_correct),
                "rank_io_proxy": int(rank_io),
                "top_5_predictions": " | ".join(t.strip() for t in top5_tokens),
                "top_5_logits": " | ".join(f"{v:.4f}" for v in top5_logits),
                "seq_length_tokens": seq_lengths[i],
                "language": lang,
            }
            records.append(record)

            if verbose:
                print(
                    f"  [{lang}] Prompt {batch_start + i + 1:02d}: "
                    f"io={io_name!r} s={s_name!r} "
                    f"logit_diff={logit_diff:+.3f} "
                    f"correct={is_correct}"
                )

    results_df = pd.DataFrame(records)
    accuracy = results_df["is_correct"].mean()
    mean_ld = results_df["logit_diff_proxy"].mean()
    n_proxy = results_df["proxy_used"].sum()

    print(f"\n[{lang.upper()} results]")
    print(f"  Prompts evaluated   : {len(results_df)}")
    print(f"  Using proxy tokens  : {n_proxy}/{len(results_df)} prompts")
    print(f"  Accuracy (io>s)     : {accuracy:.1%}")
    print(f"  Mean logit-diff     : {mean_ld:+.4f}")

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# Verdict Writer
# ══════════════════════════════════════════════════════════════════════════════

def write_feasibility_verdict(
    eval_results: dict[str, pd.DataFrame],
    tok_report: dict,
    output_path: Path,
) -> None:
    """
    Write the FEASIBILITY_VERDICT.md file with honest per-language PASS/FAIL
    determination, accuracy/CI numbers, and tokenization findings.

    Parameters
    ----------
    eval_results : dict[str, pd.DataFrame]
        Evaluation results per language.
    tok_report : dict
        Tokenization report (from run_tokenization_diagnostic).
    output_path : Path
        Path to write the verdict file.
    """
    lines = []

    lines.append("# FEASIBILITY VERDICT: Multilingual IOI Gate")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model:** GPT-2 Small (gpt2)")
    lines.append(f"**Experiment:** 13 — Multilingual IOI Feasibility Gate")
    lines.append(f"**Datasets:** 25 prompts per language (13 ABB + 12 BAB)")
    lines.append(f"**Bootstrap CI:** 95%, {BOOTSTRAP_N_RESAMPLES:,} resamples")
    lines.append(f"**PASS threshold:** Lower 95% CI bound > {PASS_THRESHOLD_CI_LOWER:.0%}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## English Baseline (Reference)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Accuracy | {ENGLISH_BASELINE['accuracy']:.1%} |")
    lines.append(f"| Mean logit-diff | {ENGLISH_BASELINE['mean_logit_diff']:+.3f} |")
    lines.append(f"| Mean tokens/prompt | ~{ENGLISH_BASELINE['mean_tokens_per_prompt']:.0f} |")
    lines.append(f"| Dataset size | 1,000 prompts |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Per-Language Results")
    lines.append("")

    overall_verdicts = {}

    for lang, results_df in eval_results.items():
        lang_tok = tok_report["languages"][lang]
        script = LANGUAGES[lang]["script"]

        # Compute stats
        is_correct_list = results_df["is_correct"].astype(float).tolist()
        acc, ci_lo, ci_hi = bootstrap_ci(is_correct_list)

        logit_diffs = results_df["logit_diff_proxy"].tolist()
        mean_ld = np.mean(logit_diffs)
        std_ld = np.std(logit_diffs)

        # Per-template accuracy
        abb_mask = results_df["template_type"] == "ABB"
        bab_mask = results_df["template_type"] == "BAB"
        acc_abb = results_df.loc[abb_mask, "is_correct"].mean() if abb_mask.any() else float("nan")
        acc_bab = results_df.loc[bab_mask, "is_correct"].mean() if bab_mask.any() else float("nan")

        # Verdict determination
        passes_gate = ci_lo > PASS_THRESHOLD_CI_LOWER
        verdict = "✅ PASS" if passes_gate else "❌ FAIL"
        overall_verdicts[lang] = passes_gate

        # Token stats
        tok_stats = lang_tok["token_count_stats"]
        byte_analysis = lang_tok["byte_level_analysis"]
        name_analysis = lang_tok["name_token_analysis"]

        lines.append(f"### {lang.capitalize()} ({script})")
        lines.append("")
        lines.append(f"**Gate verdict: {verdict}**")
        lines.append("")

        # Accuracy table
        lines.append("#### Accuracy & Logit-Diff")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Accuracy (IO > S logit) | {acc:.1%} |")
        lines.append(f"| 95% Bootstrap CI | [{ci_lo:.1%}, {ci_hi:.1%}] |")
        lines.append(f"| CI lower bound > {PASS_THRESHOLD_CI_LOWER:.0%}? | "
                     f"{'YES → PASS' if ci_lo > PASS_THRESHOLD_CI_LOWER else 'NO → FAIL'} |")
        lines.append(f"| Mean logit-diff (proxy) | {mean_ld:+.4f} |")
        lines.append(f"| Std logit-diff (proxy) | {std_ld:.4f} |")
        lines.append(f"| ABB template accuracy | {acc_abb:.1%} |")
        lines.append(f"| BAB template accuracy | {acc_bab:.1%} |")
        lines.append(f"| English baseline accuracy | {ENGLISH_BASELINE['accuracy']:.1%} |")
        lines.append("")

        # Tokenization findings
        lines.append("#### Tokenization Findings")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Mean tokens/prompt | {tok_stats['mean_tokens_per_prompt']:.1f} |")
        lines.append(f"| Token inflation vs English | {tok_stats['token_inflation_factor']:.1f}× |")
        lines.append(f"| Byte-level fallback fraction | {byte_analysis['mean_fraction_byte_level_tokens']:.0%} |")
        lines.append(f"| Byte-level fallback detected | "
                     f"{'YES ⚠' if byte_analysis['byte_level_fallback_detected'] else 'No'} |")
        lines.append(f"| Names that are single-token | {name_analysis['n_single_token_names']} |")
        lines.append(f"| Names that are multi-token | {name_analysis['n_multi_token_names']} |")
        lines.append(f"| Proxy approach required | "
                     f"{'YES ⚠' if name_analysis['proxy_approach_required'] else 'No'} |")
        lines.append(f"| Logit-diff methodology valid | "
                     f"{'NO (proxy used)' if name_analysis['proxy_approach_required'] else 'YES'} |")
        lines.append("")

        # Sample token strings
        if lang_tok["sample_tokenizations"]:
            sample = lang_tok["sample_tokenizations"][0]
            lines.append("**Sample tokenization (prompt 1):**")
            lines.append("")
            lines.append(f"Prompt: `{sample['prompt']}`")
            lines.append("")
            lines.append(f"Tokens ({sample['token_count']} total): "
                         f"`{' | '.join(sample['token_strings'][:15])}` ...")
            lines.append("")
            lines.append(f"Byte-level tokens: {sample['n_byte_level_tokens']}/{sample['token_count']} "
                         f"({sample['fraction_byte_level']:.0%})")
            lines.append("")

        # Verdict reasoning
        lines.append("#### Verdict Reasoning")
        lines.append("")
        if passes_gate:
            lines.append(
                f"The lower bound of the 95% bootstrap CI ({ci_lo:.1%}) exceeds the pass "
                f"threshold ({PASS_THRESHOLD_CI_LOWER:.0%}), indicating GPT-2 Small shows "
                f"above-chance IOI performance in {lang.capitalize()}. "
                f"**Circuit analysis may be attempted in a follow-up task.**"
            )
            lines.append("")
            lines.append("> ⚠ **Note**: Even with a PASS verdict, results should be "
                         "interpreted cautiously due to:")
            if name_analysis["proxy_approach_required"]:
                lines.append("> - Multi-token name proxy approach (logit-diff is approximate)")
            if byte_analysis["byte_level_fallback_detected"]:
                lines.append("> - Byte-level tokenizer fallback (GPT-2 was not trained on this script)")
        else:
            lines.append(
                f"The lower bound of the 95% bootstrap CI ({ci_lo:.1%}) does NOT exceed "
                f"the pass threshold ({PASS_THRESHOLD_CI_LOWER:.0%}). "
                f"GPT-2 Small **cannot reliably perform the IOI task** in {lang.capitalize()}."
            )
            lines.append("")
            lines.append("**Root causes:**")
            if byte_analysis["byte_level_fallback_detected"]:
                lines.append(
                    f"- **Tokenizer failure**: {byte_analysis['mean_fraction_byte_level_tokens']:.0%} "
                    f"of tokens are byte-level fallbacks. GPT-2 was trained on English text and "
                    f"has no meaningful representations for {script} script."
                )
            if name_analysis["proxy_approach_required"]:
                lines.append(
                    f"- **Multi-token names**: All {name_analysis['n_multi_token_names']} names "
                    f"are multi-token in GPT-2's BPE vocabulary. The standard logit-diff "
                    f"methodology is fundamentally broken for this language."
                )
            lines.append(
                f"- **Training data**: GPT-2 Small contains essentially no {lang.capitalize()} "
                f"text. The model has no learned circuit for IOI in this language."
            )
            lines.append("")
            lines.append("**Recommendation:** A multilingual-native model is required. "
                         "Suggested alternatives:")
            lines.append("- `Llama-3 8B` or `Llama-3.1 8B` (TransformerLens supported)")
            lines.append("- `Qwen2-7B` or `Qwen2.5-7B` (TransformerLens supported)")
            lines.append("- `ai4bharat/IndicBERT` (Indic-specialized, for probing studies)")
            lines.append("")
            lines.append("> ❌ **DO NOT attempt ablation or activation-patching analysis "
                         f"for {lang.capitalize()} on GPT-2 Small.** The model cannot perform "
                         "the task, so any 'circuit' found would be an artifact of noise, "
                         "not genuine IOI computation.")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Overall summary
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Language | PASS/FAIL | Proceed to Circuit Analysis? |")
    lines.append("|----------|-----------|------------------------------|")
    for lang, passed in overall_verdicts.items():
        verdict_str = "✅ PASS" if passed else "❌ FAIL"
        proceed = "Yes (in follow-up task)" if passed else "No — GPT-2 Small inadequate"
        lines.append(f"| {lang.capitalize()} | {verdict_str} | {proceed} |")

    lines.append("")

    if not any(overall_verdicts.values()):
        lines.append("> ❌ **All three languages FAILED the feasibility gate.**")
        lines.append("> GPT-2 Small is not a suitable model for multilingual IOI circuit analysis.")
        lines.append("> A multilingual-native model (e.g., Llama-3 8B or Qwen2-7B) with")
        lines.append("> TransformerLens support is required before any circuit analysis can proceed.")
        lines.append("> No circuit results have been fabricated. This investigation stops here.")
    elif all(overall_verdicts.values()):
        lines.append("> ✅ All three languages PASSED the feasibility gate.")
        lines.append("> Circuit analysis may be attempted in a follow-up task.")
        lines.append("> Note: Tokenization limitations should be disclosed in any publication.")
    else:
        passing = [l for l, v in overall_verdicts.items() if v]
        failing = [l for l, v in overall_verdicts.items() if not v]
        lines.append(f"> Partial results: **{', '.join(l.capitalize() for l in passing)}** passed; "
                     f"**{', '.join(l.capitalize() for l in failing)}** failed.")
        lines.append("> Circuit analysis may proceed only for passing languages in a follow-up task.")
        lines.append("> Failing languages require a multilingual-native model.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Methodological Disclosure")
    lines.append("")
    lines.append("### Translation Method")
    lines.append("Prompts were constructed using MT-assisted template translation with no")
    lines.append("native speaker review. See `data/multilingual/TRANSLATION_NOTES.md` for")
    lines.append("full disclosure of limitations.")
    lines.append("")
    lines.append("### Proxy Token Approach")
    lines.append("Because GPT-2's BPE tokenizer splits Indic script names into multiple")
    lines.append("byte-level tokens, the standard IOI methodology (reading `logit[IO_token_id]`)")
    lines.append("cannot be applied directly. This experiment uses the **first byte token** of")
    lines.append("each name as a proxy. This proxy:")
    lines.append("- Underestimates the model's true probability for the full name")
    lines.append("- May conflate names that share the same first byte token")
    lines.append("- Produces logit-diff values that are not directly comparable to the English baseline")
    lines.append("")
    lines.append("All results flagged with `proxy_used=True` in the baseline CSV files should")
    lines.append("be interpreted with this caveat.")
    lines.append("")
    lines.append("### No Ablation or Patching Analysis")
    lines.append("Per the experimental design, this script does NOT perform ablation,")
    lines.append("activation patching, or any circuit analysis. These are gated behind")
    lines.append("the feasibility check and will only proceed in a follow-up task if")
    lines.append("the gate passes for at least one language.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"\n[Verdict] Written to: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Main entry point for the multilingual feasibility gate experiment.
    """
    args = parse_args()

    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  CircuitScope: Mechanistic Interpretability Research             ║")
    print("║  Experiment 13 — Multilingual IOI Feasibility Gate              ║")
    print("║  Model: GPT-2 Small │ Languages: Hindi, Bengali, Assamese       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print("")
    print("PURPOSE: Diagnostic feasibility check ONLY.")
    print("This experiment does NOT perform circuit or ablation analysis.")
    print("Gate failure → No further analysis for that language on GPT-2 Small.")
    print("")

    # ── Create output directories ──────────────────────────────────────────
    ensure_output_dirs()

    # ── Load model ────────────────────────────────────────────────────────
    print("\n[Model] Loading GPT-2 Small via TransformerLens…")
    t0 = time.time()

    # Import here (keeps import errors local to execution)
    from transformer_lens import HookedTransformer

    # Suppress noisy logs
    import logging
    logging.getLogger("transformer_lens").setLevel(logging.WARNING)

    model = HookedTransformer.from_pretrained("gpt2", cache_dir=None)
    model.eval()
    load_time = time.time() - t0
    print(f"[Model] ✓ Loaded GPT-2 Small in {load_time:.1f}s")
    print(f"[Model] Vocab size: {model.cfg.d_vocab:,} | "
          f"Layers: {model.cfg.n_layers} | "
          f"Heads: {model.cfg.n_heads}")

    # ── Load all datasets ─────────────────────────────────────────────────
    print("\n[Datasets] Loading multilingual IOI datasets…")
    datasets: dict[str, pd.DataFrame] = {}
    for lang in LANGUAGES:
        datasets[lang] = load_language_dataset(lang)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Tokenization Diagnostic
    # CRITICAL: Must run before evaluation to surface tokenizer failures.
    # ══════════════════════════════════════════════════════════════════════
    t1 = time.time()
    tok_report = run_tokenization_diagnostic(
        model=model,
        datasets=datasets,
        n_samples=args.n_samples_tok,
        verbose=args.verbose,
    )
    tok_time = time.time() - t1

    # Save tokenization report
    tok_report_path = RESULTS_DIR / "tokenization_report.json"
    with open(tok_report_path, "w", encoding="utf-8") as f:
        json.dump(tok_report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\n[Tokenization] ✓ Report saved to: {tok_report_path}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Baseline Accuracy Gate
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: BASELINE ACCURACY GATE")
    print("=" * 70)
    print("Running GPT-2 Small on each language's prompt set.")
    print("Using same logit-diff methodology as English baseline (experiment 01).")
    print("NOTE: Multi-token names use first-token proxy (disclosed).\n")

    eval_results: dict[str, pd.DataFrame] = {}
    t2 = time.time()

    for lang, df in datasets.items():
        lang_results = evaluate_language(
            model=model,
            df=df,
            lang=lang,
            tok_report=tok_report,
            batch_size=16,
            verbose=args.verbose,
        )
        eval_results[lang] = lang_results

        # Compute and print bootstrap CI immediately
        is_correct = lang_results["is_correct"].astype(float).tolist()
        acc, ci_lo, ci_hi = bootstrap_ci(is_correct)
        passes = ci_lo > PASS_THRESHOLD_CI_LOWER

        print(f"\n[{lang.upper()} GATE RESULT]")
        print(f"  Accuracy       : {acc:.1%}")
        print(f"  95% Bootstrap CI : [{ci_lo:.1%}, {ci_hi:.1%}]")
        print(f"  CI lower > {PASS_THRESHOLD_CI_LOWER:.0%}? : {'YES' if passes else 'NO'}")
        print(f"  VERDICT        : {'✅ PASS' if passes else '❌ FAIL'}")

        # Save per-language CSV
        output_csv = RESULTS_DIR / f"baseline_{lang}.csv"
        lang_results.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"  Saved to       : {output_csv}")

    eval_time = time.time() - t2

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Write Feasibility Verdict
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: WRITING FEASIBILITY VERDICT")
    print("=" * 70)

    verdict_path = OUTPUT_DIR / "FEASIBILITY_VERDICT.md"
    write_feasibility_verdict(
        eval_results=eval_results,
        tok_report=tok_report,
        output_path=verdict_path,
    )

    # ── Final summary ──────────────────────────────────────────────────────
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("EXPERIMENT 13 COMPLETE")
    print("=" * 70)
    print(f"  Total runtime            : {total_time:.1f}s")
    print(f"  Model load time          : {load_time:.1f}s")
    print(f"  Tokenization diagnostic  : {tok_time:.1f}s")
    print(f"  Evaluation               : {eval_time:.1f}s")
    print("")
    print(f"  Output directory         : {OUTPUT_DIR}")
    print(f"  Tokenization report      : {RESULTS_DIR / 'tokenization_report.json'}")
    print(f"  Feasibility verdict      : {verdict_path}")
    print("")

    # Print final verdict summary
    print("FINAL VERDICT SUMMARY:")
    for lang, results_df in eval_results.items():
        is_correct = results_df["is_correct"].astype(float).tolist()
        acc, ci_lo, ci_hi = bootstrap_ci(is_correct)
        passes = ci_lo > PASS_THRESHOLD_CI_LOWER
        print(f"  {lang.upper():<12}: {'✅ PASS' if passes else '❌ FAIL'} "
              f"(acc={acc:.1%}, 95% CI=[{ci_lo:.1%}, {ci_hi:.1%}])")

    print("")
    print("NOTE: This script has NOT proceeded to circuit or ablation analysis.")
    print("See FEASIBILITY_VERDICT.md for the full verdict and recommendations.")


if __name__ == "__main__":
    main()
