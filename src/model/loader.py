"""
src/model/loader.py
====================
Model and tokenizer loading utilities for CircuitScope.

Key TransformerLens Objects Explained
--------------------------------------
HookedTransformer
    The central class in TransformerLens. It wraps a pre-trained transformer
    (GPT-2 Small in our case) and adds:
      - `run_with_cache()` : forward pass that records every intermediate
        activation (residual stream, attention patterns, MLP outputs) into
        an `ActivationCache` object.
      - `add_hook()` / `run_with_hooks()` : register arbitrary Python
        callables that read or write activations at any named hook point
        during the forward pass. This is the core mechanism for activation
        patching and circuit analysis.
      - `to_tokens()` : tokenise a string using the model's own tokenizer.
      - `to_str_tokens()` : human-readable token list, useful for debugging.

HookedTransformerConfig (cfg)
    A dataclass living at `model.cfg` that exposes every architectural
    hyper-parameter: n_layers, n_heads, d_model, d_head, d_mlp, etc.
    Used throughout evaluation to know tensor shapes without hard-coding.

ActivationCache
    Returned by `run_with_cache()`. Behaves like a dict but is keyed by
    hook names such as:
      - "blocks.{layer}.attn.hook_pattern"   — attention pattern [B, H, Q, K]
      - "blocks.{layer}.hook_resid_post"     — residual stream [B, S, D]
      - "blocks.{layer}.mlp.hook_post"       — MLP post-activation [B, S, D_mlp]
    Supports slicing and common tensor operations.

FactoredMatrix
    TransformerLens represents weight matrices like W_OV = W_O @ W_V as
    `FactoredMatrix` to avoid materialising huge intermediate tensors.
    Relevant when computing eigenvalue / SVD-based circuit analyses.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformer_lens import HookedTransformer

# ── Module-level logger ────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def get_device(device: str = "auto") -> torch.device:
    """
    Resolve the target compute device.

    Parameters
    ----------
    device : str
        One of:
        - "auto"  → prefers CUDA, then MPS (Apple Silicon), then CPU.
        - "cuda"  → NVIDIA GPU; raises if unavailable.
        - "mps"   → Apple Silicon GPU; raises if unavailable.
        - "cpu"   → Always available fallback.

    Returns
    -------
    torch.device
        The resolved PyTorch device object.

    Notes
    -----
    TransformerLens models can be moved between devices via `.to(device)`.
    For CUDA, batch sizes of 32–64 are typically memory-safe for GPT-2 Small.
    For CPU, use smaller batches (8–16) to avoid long runtimes.
    """
    if device == "auto":
        if torch.cuda.is_available():
            resolved = torch.device("cuda")
        elif torch.backends.mps.is_available():
            resolved = torch.device("mps")
        else:
            resolved = torch.device("cpu")
    else:
        resolved = torch.device(device)

    logger.info(f"[get_device] Resolved device: {resolved}")
    return resolved


def load_model(
    model_name: str = "gpt2",
    device: str = "auto",
    dtype: str = "float32",
    cache_dir: Optional[str] = None,
    fold_ln: bool = True,
    center_writing_weights: bool = True,
    center_unembed: bool = True,
) -> HookedTransformer:
    """
    Load a pre-trained HookedTransformer model from TransformerLens.

    This is the primary entry point for obtaining the model. We apply several
    TransformerLens-specific transformations that simplify circuit analysis:

    Parameters
    ----------
    model_name : str
        Name of the model on HuggingFace Hub. For this project: "gpt2"
        (GPT-2 Small: 12 layers, 12 heads, d_model=768, 117M parameters).

    device : str
        Target device. See `get_device()` for valid options.

    dtype : str
        Floating-point precision. Options: "float32" (safest), "float16"
        (faster on GPU), "bfloat16" (numerically stable on modern GPUs).

    cache_dir : str, optional
        Custom HuggingFace model cache directory. None uses the default
        (~/.cache/huggingface/hub/).

    fold_ln : bool
        If True, fold LayerNorm parameters (weight and bias) into the
        subsequent linear layer's weights. This makes the model *exactly*
        equivalent but removes the non-linearity of LayerNorm's learned
        parameters, simplifying the residual stream analysis.
        Default: True (recommended for mechanistic interpretability).

    center_writing_weights : bool
        If True, subtract the mean of every weight matrix that *writes*
        into the residual stream (W_E, W_pos, W_O, W_in) so that all
        writing operations are mean-zero. Combined with fold_ln=True, this
        means LayerNorm becomes a pure normalisation with no learned shift.
        Default: True.

    center_unembed : bool
        If True, subtract the mean of the unembedding matrix W_U along the
        vocabulary dimension. This makes the logits mean-zero across tokens
        for every position, which is useful when computing logit differences
        (the metric cancels out bias terms automatically).
        Default: True.

    Returns
    -------
    HookedTransformer
        The fully loaded and configured model, placed on the target device,
        in evaluation mode (gradients disabled, dropout off).

    Examples
    --------
    >>> from src.model import load_model
    >>> model = load_model("gpt2", device="auto")
    >>> print(model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_model)
    12 12 768

    TransformerLens Key Architecture (GPT-2 Small)
    -----------------------------------------------
    n_layers  = 12   — Number of transformer blocks
    n_heads   = 12   — Attention heads per layer
    d_model   = 768  — Residual stream dimension
    d_head    = 64   — Per-head QKV dimension  (d_model / n_heads)
    d_mlp     = 3072 — MLP hidden dimension    (4 × d_model)
    d_vocab   = 50257— Vocabulary size
    n_ctx     = 1024 — Maximum context length
    """
    # ── 1. Resolve device ──────────────────────────────────────────────────
    target_device = get_device(device)

    # ── 2. Map dtype string → torch dtype ─────────────────────────────────
    dtype_map: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Choose from: {list(dtype_map.keys())}"
        )
    torch_dtype = dtype_map[dtype]

    # ── 3. Load model from HuggingFace via TransformerLens ────────────────
    logger.info(f"[load_model] Loading '{model_name}' on {target_device} ({dtype})…")

    model = HookedTransformer.from_pretrained(
        model_name,
        # fold_ln, center_writing_weights, center_unembed are passed as
        # keyword arguments to the TransformerLens loading pipeline.
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
        # dtype is set after loading via .to() to ensure correct precision
    )

    # ── 4. Move to device and set precision ───────────────────────────────
    model = model.to(target_device)
    if torch_dtype != torch.float32:
        model = model.to(torch_dtype)

    # ── 5. Set to evaluation mode ─────────────────────────────────────────
    # eval() disables dropout and sets BatchNorm to inference mode.
    # GPT-2 has no BatchNorm but this is good practice.
    model.eval()

    # ── 6. Log model summary ──────────────────────────────────────────────
    cfg = model.cfg
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[load_model] ✓ Model loaded successfully.\n"
        f"  Architecture : {cfg.model_name}\n"
        f"  Layers       : {cfg.n_layers}\n"
        f"  Heads        : {cfg.n_heads}\n"
        f"  d_model      : {cfg.d_model}\n"
        f"  d_head       : {cfg.d_head}\n"
        f"  d_mlp        : {cfg.d_mlp}\n"
        f"  d_vocab      : {cfg.d_vocab}\n"
        f"  n_ctx        : {cfg.n_ctx}\n"
        f"  Parameters   : {n_params / 1e6:.1f}M\n"
        f"  Device       : {target_device}\n"
        f"  dtype        : {torch_dtype}"
    )

    return model


def get_tokenizer_info(model: HookedTransformer) -> dict:
    """
    Extract and return tokenizer metadata from a loaded HookedTransformer.

    TransformerLens bundles the tokenizer inside the model object. You do
    not need a separate `AutoTokenizer` — use `model.to_tokens()` and
    `model.to_str_tokens()` directly.

    Parameters
    ----------
    model : HookedTransformer
        A loaded HookedTransformer instance.

    Returns
    -------
    dict
        Dictionary containing:
        - "vocab_size"    : int  — Total vocabulary size
        - "bos_token"     : str  — Beginning-of-sequence token string
        - "bos_token_id"  : int  — BOS token index
        - "eos_token"     : str  — End-of-sequence token string
        - "pad_token_id"  : int or None
        - "prepend_bos"   : bool — Whether TransformerLens prepends BOS

    Examples
    --------
    >>> info = get_tokenizer_info(model)
    >>> print(info["vocab_size"])    # 50257 for GPT-2
    >>> print(info["bos_token"])     # "<|endoftext|>"

    Notes on TransformerLens tokenisation
    --------------------------------------
    By default, TransformerLens prepends a BOS token to every input string
    when calling `model.to_tokens(text, prepend_bos=True)`. This is the
    default and is important: IOI prompts should always be processed with
    BOS prepended to match the training distribution of GPT-2.

    `model.to_str_tokens(" John")` returns [' John'] — note the leading
    space. GPT-2 uses a byte-pair encoding (BPE) where spaces are part of
    the token. Always include the leading space when looking up name tokens:
        correct_token_id = model.to_single_token(" Mary")
    """
    tokenizer = model.tokenizer

    info = {
        "vocab_size": model.cfg.d_vocab,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "pad_token_id": tokenizer.pad_token_id,
        "prepend_bos": model.cfg.default_prepend_bos,
    }

    logger.debug(f"[get_tokenizer_info] Tokenizer metadata: {info}")
    return info
