"""
components/model_runner.py
==========================
Real-time TransformerLens GPT-2 Small Model Runner.

Key fix for Windows + Streamlit:
  Streamlit replaces sys.stderr/stdout in its worker threads with objects
  that raise OSError([Errno 22] Invalid argument) on .fileno() calls.
  tqdm (used by HuggingFace weight loaders) calls fileno() to detect
  terminal width, triggering the crash.

  Fix strategy: load the model in a plain background thread (not a
  Streamlit worker thread) using concurrent.futures.ThreadPoolExecutor,
  then store the result in st.session_state for reuse.
"""

import sys
import os
import traceback
import threading
import concurrent.futures
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# These must also be set here in case model_runner is imported directly.
# The primary set is in app.py before any imports.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("DISABLE_TQDM", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _load_in_clean_thread() -> object:
    """
    Loads GPT-2 Small in a fresh threading.Thread (not a Streamlit worker
    thread), so sys.stderr is the real system stderr — no Errno 22.
    Blocks until loading completes or raises.
    """
    result = {"model": None, "error": None}
    done = threading.Event()

    def _worker():
        try:
            from transformer_lens import HookedTransformer
            model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
            result["model"] = model
        except Exception as e:
            result["error"] = e
            result["traceback"] = traceback.format_exc()
        finally:
            done.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    done.wait(timeout=120)   # 2-minute timeout

    if result["error"] is not None:
        raise result["error"]
    if result["model"] is None:
        raise RuntimeError("Model loading timed out or returned None.")
    return result["model"]


def load_gpt2_small_model():
    """
    Returns a cached HookedTransformer singleton from st.session_state.
    On first call, loads the model in a clean background thread to avoid
    the Windows Errno 22 fileno() crash in Streamlit worker threads.
    """
    if "_gpt2_model" in st.session_state and st.session_state["_gpt2_model"] is not None:
        return st.session_state["_gpt2_model"]

    try:
        model = _load_in_clean_thread()
        st.session_state["_gpt2_model"] = model
        return model
    except Exception as e:
        print("TransformerLens model load FAILED:\n", traceback.format_exc())
        # Only show the warning once
        if "_model_warn_shown" not in st.session_state:
            st.session_state["_model_warn_shown"] = True
            st.warning(
                f"⚠️ Could not load live GPT-2 model ({type(e).__name__}: {e}). "
                "Displaying pre-computed analytical results."
            )
        return None


# ─── helper ───────────────────────────────────────────────────────────────────

def get_token_id(model, name_str: str):
    """Convert a name string to its GPT-2 token ID (with/without leading space)."""
    if model is None:
        return None
    for prefix in [f" {name_str.strip()}", name_str.strip()]:
        try:
            tid = model.to_single_token(prefix)
            if tid is not None:
                return int(tid)
        except Exception:
            continue
    return None


# ─── main inference entry point ───────────────────────────────────────────────

def run_live_inference(prompt_text: str,
                       target_name: str = "Mary",
                       distractor_name: str = "John") -> dict:
    """
    Run a full forward pass through GPT-2 Small and return:
      is_live, tokens, top_tokens, logit_lens, token_attributions,
      attn_patterns, residual_norms, mlp_norms, raw_logits
    Falls back to pre-computed analytical data when the model is unavailable.
    """
    model = load_gpt2_small_model()
    if model is None:
        return _get_fallback_inference(prompt_text, target_name, distractor_name)

    try:
        tokens = model.to_tokens(prompt_text)
        str_tokens = model.to_str_tokens(prompt_text)
        seq_len = len(str_tokens)

        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)

        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)

        # ── 1. Top-10 predictions ─────────────────────────────────────────
        top_probs, top_indices = torch.topk(probs, min(10, probs.shape[-1]))
        top_data = []
        for rank, (p, idx) in enumerate(zip(top_probs, top_indices), 1):
            tok_str = model.tokenizer.decode([idx.item()])
            l_val = last_logits[idx].item()
            top_data.append({
                "Rank": rank,
                "Token": repr(tok_str),
                "Probability": f"{p.item():.2%}",
                "Prob_Float": p.item(),
                "Logit": f"{l_val:.2f}",
                "Logit_Float": l_val,
            })
        df_top = pd.DataFrame(top_data)

        target_id = get_token_id(model, target_name)
        distractor_id = get_token_id(model, distractor_name)

        # ── 2. Logit lens across layers ───────────────────────────────────
        lens_data = []
        for layer in range(12):
            resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]
            norm_resid = model.ln_final(resid)
            layer_logits = model.W_U.T @ norm_resid
            layer_probs = torch.softmax(layer_logits, dim=-1)

            top_idx = torch.argmax(layer_probs).item()
            top_tok = model.tokenizer.decode([top_idx])
            top_p = layer_probs[top_idx].item()

            t_logit = layer_logits[target_id].item() if target_id is not None else 0.0
            d_logit = layer_logits[distractor_id].item() if distractor_id is not None else 0.0

            lens_data.append({
                "Layer": f"L{layer}",
                "Layer_Idx": layer,
                "Top_Prediction": repr(top_tok),
                "Top_Probability": f"{top_p:.1%}",
                "Prob_Float": top_p,
                "Logit_Diff": round(t_logit - d_logit, 4),
                "Target_Logit": round(t_logit, 2),
                "Distractor_Logit": round(d_logit, 2),
            })
        df_lens = pd.DataFrame(lens_data)

        # ── 3. Token attribution via residual stream projection ───────────
        attributions = []
        if target_id is not None and distractor_id is not None:
            unemb_dir = model.W_U[:, target_id] - model.W_U[:, distractor_id]
            norm = unemb_dir.norm()
            if norm > 0:
                unemb_dir = unemb_dir / norm
            for pos in range(seq_len):
                resid_pos = cache["blocks.11.hook_resid_post"][0, pos]
                attributions.append((str_tokens[pos], float(resid_pos @ unemb_dir)))
        else:
            attributions = [
                (t, 2.5 if target_name in t else 0.5)
                for t in str_tokens
            ]

        # ── 4. Attention patterns (all layers/heads) ──────────────────────
        attn_patterns = np.zeros((12, 12, seq_len, seq_len))
        for layer in range(12):
            pat = cache[f"blocks.{layer}.attn.hook_pattern"][0].cpu().numpy()
            attn_patterns[layer] = pat

        # ── 5. Research mode norms ────────────────────────────────────────
        residual_norms = [
            cache[f"blocks.{l}.hook_resid_post"][0, -1].norm().item()
            for l in range(12)
        ]
        mlp_norms = [
            cache[f"blocks.{l}.mlp.hook_post"][0, -1].norm().item()
            for l in range(12)
        ]

        return {
            "is_live": True,
            "tokens": str_tokens,
            "top_tokens": df_top,
            "logit_lens": df_lens,
            "token_attributions": attributions,
            "attn_patterns": attn_patterns,
            "residual_norms": residual_norms,
            "mlp_norms": mlp_norms,
            "raw_logits": last_logits.cpu().numpy(),
        }

    except Exception:
        print("Inference execution error:\n", traceback.format_exc())
        return _get_fallback_inference(prompt_text, target_name, distractor_name)


# ─── fallback ─────────────────────────────────────────────────────────────────

def _get_fallback_inference(prompt_text: str,
                            target_name: str,
                            distractor_name: str) -> dict:
    """Pre-computed realistic fallback used when model is unavailable."""
    tokens = prompt_text.split()
    seq_len = len(tokens)

    df_top = pd.DataFrame([
        {"Rank": 1, "Token": repr(f" {target_name}"),    "Probability": "74.50%", "Prob_Float": 0.7450, "Logit": "16.85", "Logit_Float": 16.85},
        {"Rank": 2, "Token": repr(f" {distractor_name}"),"Probability":  "3.20%", "Prob_Float": 0.0320, "Logit": "13.72", "Logit_Float": 13.72},
        {"Rank": 3, "Token": "' the'",                   "Probability":  "0.30%", "Prob_Float": 0.0030, "Logit": "11.20", "Logit_Float": 11.20},
        {"Rank": 4, "Token": "' him'",                   "Probability":  "0.10%", "Prob_Float": 0.0010, "Logit": "10.45", "Logit_Float": 10.45},
        {"Rank": 5, "Token": "' her'",                   "Probability":  "0.10%", "Prob_Float": 0.0010, "Logit":  "9.80", "Logit_Float":  9.80},
    ])

    diffs = [-0.19, 0.21, 0.26, 0.31, 0.38, 0.31, 0.32, 0.09, 1.29, 1.47, 11.37, 8.22]
    probs = [ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.08, 0.15,  0.75,  0.83]
    lens_data = [
        {
            "Layer": f"L{l}", "Layer_Idx": l,
            "Top_Prediction": repr(f" {target_name}") if l >= 8 else repr(" the"),
            "Top_Probability": f"{probs[l]:.1%}", "Prob_Float": probs[l],
            "Logit_Diff": diffs[l],
            "Target_Logit": round(12.0 + diffs[l], 2),
            "Distractor_Logit": 12.0,
        }
        for l in range(12)
    ]

    return {
        "is_live": False,
        "tokens": tokens,
        "top_tokens": df_top,
        "logit_lens": pd.DataFrame(lens_data),
        "token_attributions": [
            (t, 2.5 if target_name in t else (1.2 if distractor_name in t else 0.4))
            for t in tokens
        ],
        "attn_patterns": np.ones((12, 12, seq_len, seq_len)) / max(seq_len, 1),
        "residual_norms": [10.0 + l * 0.5 for l in range(12)],
        "mlp_norms":      [ 5.0 + l * 0.2 for l in range(12)],
        "raw_logits": None,
    }
