"""
components/model_runner.py
==========================
Real-time TransformerLens GPT-2 Small Model Runner & Cache Management.
Executes live forward passes, logit lens projections, attention pattern extraction,
token attributions, and research mode activation diagnostics.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@st.cache_resource
def load_gpt2_small_model():
    """Loads GPT-2 Small via TransformerLens (cached across sessions)."""
    try:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
        return model
    except Exception as e:
        st.warning(f"Unable to load live HookedTransformer model ({e}). Using analytical fallback.")
        return None

def get_token_id(model, name_str: str) -> int | None:
    """Helper to convert token string to token ID in GPT-2 vocabulary."""
    if model is None:
        return None
    for prefix in [f" {name_str.strip()}", name_str.strip()]:
        try:
            tid = model.to_single_token(prefix)
            if tid is not None:
                return tid
        except Exception:
            continue
    return None

def run_live_inference(prompt_text: str, target_name: str = "Mary", distractor_name: str = "John"):
    """
    Executes live forward pass and extracts all internal activations for a prompt.
    
    Returns a dict containing:
    - tokens: List of string tokens
    - top_tokens: DataFrame of top 10 next token predictions
    - logit_lens: DataFrame of layer 0..11 logit lens evolution
    - token_attributions: List of (token_str, score) tuples
    - attention_patterns: Tensor of shape (12, 12, seq, seq)
    - residual_norms: List of layer-by-layer residual stream L2 norms
    - mlp_norms: List of layer-by-layer MLP activation L2 norms
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
        
        # 1. Top 10 Predictions
        top_probs, top_indices = torch.topk(probs, 10)
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
        
        # Identify Target and Distractor token IDs
        target_id = get_token_id(model, target_name)
        distractor_id = get_token_id(model, distractor_name)
        
        # 2. Logit Lens Evolution across Layers
        lens_data = []
        for layer in range(12):
            resid_key = f"blocks.{layer}.hook_resid_post"
            resid = cache[resid_key][0, -1] # (768,)
            norm_resid = model.ln_final(resid)
            layer_logits = model.W_U.T @ norm_resid # (vocab,)
            layer_probs = torch.softmax(layer_logits, dim=-1)
            
            top_idx = torch.argmax(layer_probs).item()
            top_tok = model.tokenizer.decode([top_idx])
            top_p = layer_probs[top_idx].item()
            
            t_logit = layer_logits[target_id].item() if target_id is not None else 0.0
            d_logit = layer_logits[distractor_id].item() if distractor_id is not None else 0.0
            diff = t_logit - d_logit
            
            lens_data.append({
                "Layer": f"L{layer}",
                "Layer_Idx": layer,
                "Top_Prediction": repr(top_tok),
                "Top_Probability": f"{top_p:.1%}",
                "Prob_Float": top_p,
                "Logit_Diff": round(diff, 4),
                "Target_Logit": round(t_logit, 2),
                "Distractor_Logit": round(d_logit, 2),
            })
        df_lens = pd.DataFrame(lens_data)
        
        # 3. Direct Token Attribution (Residual Stream per token onto unemb dir)
        attributions = []
        if target_id is not None and distractor_id is not None:
            unemb_dir = model.W_U[:, target_id] - model.W_U[:, distractor_id] # (768,)
            unemb_dir = unemb_dir / unemb_dir.norm()
            
            # Use final layer residual contributions per token
            for pos in range(seq_len):
                token_resid = cache["blocks.11.hook_resid_post"][0, pos] # (768,)
                attr_score = (token_resid @ unemb_dir).item()
                attributions.append((str_tokens[pos], float(attr_score)))
        else:
            for pos in range(seq_len):
                attributions.append((str_tokens[pos], 1.0 if target_name in str_tokens[pos] else 0.1))
                
        # 4. Attention Patterns (12 layers, 12 heads)
        attn_patterns = np.zeros((12, 12, seq_len, seq_len))
        for layer in range(12):
            pat = cache[f"blocks.{layer}.attn.hook_pattern"][0].cpu().numpy() # (12, seq, seq)
            attn_patterns[layer] = pat
            
        # 5. Research Mode Diagnostics (norms)
        residual_norms = [cache[f"blocks.{l}.hook_resid_post"][0, -1].norm().item() for l in range(12)]
        mlp_norms = [cache[f"blocks.{l}.mlp.hook_post"][0, -1].norm().item() for l in range(12)]
        
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
        
    except Exception as e:
        st.warning(f"Live inference fallback activated: {e}")
        return _get_fallback_inference(prompt_text, target_name, distractor_name)

def _get_fallback_inference(prompt_text: str, target_name: str, distractor_name: str):
    """Provides structured realistic fallback inference data."""
    tokens = prompt_text.split()
    seq_len = len(tokens)
    
    df_top = pd.DataFrame([
        {"Rank": 1, "Token": repr(f" {target_name}"), "Probability": "74.50%", "Prob_Float": 0.745, "Logit": "16.85", "Logit_Float": 16.85},
        {"Rank": 2, "Token": repr(f" {distractor_name}"), "Probability": "3.20%", "Prob_Float": 0.032, "Logit": "13.72", "Logit_Float": 13.72},
        {"Rank": 3, "Token": "' the'", "Probability": "0.30%", "Prob_Float": 0.003, "Logit": "11.20", "Logit_Float": 11.20},
        {"Rank": 4, "Token": "' him'", "Probability": "0.10%", "Prob_Float": 0.001, "Logit": "10.45", "Logit_Float": 10.45},
        {"Rank": 5, "Token": "' her'", "Probability": "0.10%", "Prob_Float": 0.001, "Logit": "9.80", "Logit_Float": 9.80},
    ])
    
    diffs = [-0.19, 0.21, 0.26, 0.31, 0.38, 0.31, 0.32, 0.09, 1.29, 1.47, 11.37, 8.22]
    probs = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.08, 0.15, 0.75, 0.83]
    
    lens_data = []
    for l in range(12):
        lens_data.append({
            "Layer": f"L{l}",
            "Layer_Idx": l,
            "Top_Prediction": repr(f" {target_name}") if l >= 8 else repr(" the"),
            "Top_Probability": f"{probs[l]:.1%}",
            "Prob_Float": probs[l],
            "Logit_Diff": diffs[l],
            "Target_Logit": round(12.0 + diffs[l], 2),
            "Distractor_Logit": 12.0,
        })
    df_lens = pd.DataFrame(lens_data)
    
    attributions = [(t, 2.5 if target_name in t else (1.2 if distractor_name in t else 0.4)) for t in tokens]
    attn_patterns = np.ones((12, 12, seq_len, seq_len)) / max(seq_len, 1)
    
    return {
        "is_live": False,
        "tokens": tokens,
        "top_tokens": df_top,
        "logit_lens": df_lens,
        "token_attributions": attributions,
        "attn_patterns": attn_patterns,
        "residual_norms": [10.0 + l*0.5 for l in range(12)],
        "mlp_norms": [5.0 + l*0.2 for l in range(12)],
        "raw_logits": None,
    }
