"""
src/analysis/logit_lens.py
===========================
Logit Lens analysis for the IOI circuit in GPT-2 Small.

Theory: The Logit Lens
-----------------------
The "logit lens" (nostalgebraist, 2020) is a technique for inspecting what
a transformer is "thinking" at each intermediate layer by projecting the
residual stream into vocabulary space.

Formally, GPT-2's forward pass computes:

    h_0 = Embed(tokens) + PosEmbed(tokens)
    h_l = h_{l-1} + Attn_l(h_{l-1}) + MLP_l(h_{l-1} + Attn_l(h_{l-1}))
    logits = W_U · LayerNorm_final(h_{n_layers})

The logit lens asks: what if we stopped at layer l and projected h_l directly?

    logits_l = W_U · LayerNorm_final(h_l)

This gives us a "peek" at what the model predicts using only information
accumulated up to layer l. By tracking how logit_diff (IO - S) evolves
across layers, we can identify:

  - Which layer first "encodes" the IO preference
  - How confidence builds as information propagates
  - Whether early layers actively harm or help the task

Key Implementation Notes
-------------------------
1. With `fold_ln=True` (our default), LayerNorm parameters are pre-folded
   into subsequent weight matrices. The final LayerNorm (`model.ln_final`)
   is still intact and must be applied before the unembed.

2. We cache ALL residual streams in a single forward pass using
   `run_with_cache()`, then apply logit lens offline. This is much more
   efficient than running 13 separate forward passes.

3. The residual stream hook points are:
     hook_embed                       → h_0 (before any layers)
     blocks.0.hook_resid_post         → h_1 (after layer 0)
     ...
     blocks.11.hook_resid_post        → h_12 (after layer 11 = final)

4. We extract the logit at the LAST token position (the "to" token's
   successor position) because IOI prompts end mid-sentence.

5. We average across multiple prompts to get stable per-layer estimates.

References
----------
nostalgebraist (2020). "interpreting GPT: the logit lens".
  https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/

Dar et al. (2022). "Analyzing Transformers in Embedding Space".
  https://arxiv.org/abs/2209.02535
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformer_lens import HookedTransformer, ActivationCache
from src.data.ioi_dataset import IOIDataset

logger = logging.getLogger(__name__)


class LogitLensAnalyzer:
    """
    Computes the logit lens projection for every transformer layer.

    For each layer l ∈ {0, 1, ..., n_layers}, we apply:
        logits_l = W_U · LayerNorm_final(h_l)
    and extract the per-prompt logit difference (IO - S) at the final
    token position.

    Parameters
    ----------
    model : HookedTransformer
        Loaded GPT-2 Small in eval mode.

    dataset : IOIDataset
        Generated IOI dataset with at least `n_samples` prompts.

    n_samples : int
        Number of prompts to average over. More → more stable estimates.
        Recommended: 100–300 for speed, 500+ for publication quality.

    Examples
    --------
    >>> analyzer = LogitLensAnalyzer(model, dataset, n_samples=200)
    >>> results_df = analyzer.run()
    >>> print(results_df.head())
    """

    def __init__(
        self,
        model: HookedTransformer,
        dataset: IOIDataset,
        n_samples: int = 200,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.n_samples = min(n_samples, len(dataset))
        self.device = next(model.parameters()).device
        self.n_layers = model.cfg.n_layers

        logger.info(
            f"[LogitLensAnalyzer] Initialized: "
            f"n_layers={self.n_layers}, n_samples={self.n_samples}"
        )

    def _get_residual_stream_names(self) -> list[str]:
        """
        Return ordered list of hook names for every residual stream checkpoint.

        The order matches the computation graph:
          - "hook_embed"               : embedding output (token + positional)
          - "blocks.0.hook_resid_post" : after layer 0 (attn + MLP)
          - "blocks.1.hook_resid_post" : after layer 1
          - ...
          - "blocks.11.hook_resid_post": after layer 11 (final)

        Note: "hook_embed" here refers to the combined embedding (token + pos).
        In TransformerLens, this is accessed as:
            resid_0 = cache["hook_embed"] + cache["hook_pos_embed"]

        Returns
        -------
        list of str
            Hook names for layers 0 through n_layers (n_layers+1 checkpoints).
        """
        names = []
        # Layer 0: pure embeddings (before any attention/MLP)
        names.append("pre_layer_0")   # synthetic label; handled specially
        # Layers 1 through n_layers: after each transformer block
        for l in range(self.n_layers):
            names.append(f"blocks.{l}.hook_resid_post")
        return names

    @torch.no_grad()
    def _compute_logit_lens_single_cache(
        self,
        cache: ActivationCache,
        io_token_id: int,
        s_token_id: int,
        final_token_pos: int,
    ) -> dict[int, dict[str, float]]:
        """
        Apply logit lens to a single prompt's activation cache.

        For each layer checkpoint, extracts the residual stream at
        `final_token_pos`, applies LayerNorm_final, projects through W_U,
        and computes logit_diff, prob_io, prob_s.

        Parameters
        ----------
        cache : ActivationCache
            Output of `model.run_with_cache(tokens)`. Contains all
            intermediate activations for a batch of prompts.

        io_token_id : int
            Token ID for the correct (IO) answer.

        s_token_id : int
            Token ID for the incorrect (S) answer.

        final_token_pos : int
            Token position index of the last real token (not padding).

        Returns
        -------
        dict[int, dict[str, float]]
            Mapping: layer_idx → {"logit_diff": ..., "prob_io": ..., "prob_s": ...}
            layer_idx=0 means "pre-layer 0" (pure embeddings).
        """

        results: dict[int, dict[str, float]] = {}

        for layer_idx in range(self.n_layers + 1):
            # ── Extract residual stream at this checkpoint ────────────────
            if layer_idx == 0:
                # Embedding layer: combine token + positional embeddings
                # Both have shape [batch, seq_len, d_model]
                resid = cache["hook_embed"] + cache["hook_pos_embed"]
                # resid shape: [batch, seq_len, d_model]
            else:
                # After block (layer_idx - 1)
                resid = cache[f"blocks.{layer_idx - 1}.hook_resid_post"]
                # resid shape: [batch, seq_len, d_model]

            # ── Extract at final token position ───────────────────────────
            # Shape: [batch, d_model]
            h = resid[:, final_token_pos, :]

            # ── Apply final LayerNorm ─────────────────────────────────────
            # model.ln_final is the final LayerNorm layer.
            # Even with fold_ln=True, this exists and must be applied.
            # Shape: [batch, d_model]
            h_normed = self.model.ln_final(h)

            # ── Project to vocabulary space via unembedding matrix ────────
            # model.unembed applies W_U: [batch, d_model] → [batch, d_vocab]
            # W_U shape: [d_model, d_vocab]
            logits = self.model.unembed(h_normed.unsqueeze(1)).squeeze(1)
            # logits shape: [batch, d_vocab]

            # ── Extract IO and S logits ───────────────────────────────────
            logit_io = logits[:, io_token_id]   # [batch]
            logit_s = logits[:, s_token_id]     # [batch]
            logit_diff = (logit_io - logit_s).mean().item()

            # ── Softmax probabilities ─────────────────────────────────────
            probs = F.softmax(logits, dim=-1)   # [batch, d_vocab]
            prob_io = probs[:, io_token_id].mean().item()
            prob_s = probs[:, s_token_id].mean().item()

            results[layer_idx] = {
                "logit_diff": logit_diff,
                "prob_io": prob_io,
                "prob_s": prob_s,
                "logit_io": logit_io.mean().item(),
                "logit_s": logit_s.mean().item(),
            }

        return results

    @torch.no_grad()
    def run(
        self,
        batch_size: int = 20,
        include_embedding_layer: bool = True,
    ) -> pd.DataFrame:
        """
        Run the full logit lens analysis over `n_samples` prompts.

        The pipeline:
          1. For each batch of prompts, run `model.run_with_cache()` to
             capture all residual stream activations.
          2. Apply logit lens at each of the n_layers+1 checkpoints.
          3. Average results across all prompts to get stable estimates.
          4. Return a tidy DataFrame indexed by layer.

        Parameters
        ----------
        batch_size : int
            Prompts processed per forward pass. Keep small (10–20) to
            avoid OOM errors since we cache ALL activations.

        include_embedding_layer : bool
            If True, include layer_idx=0 (pure embeddings). This is
            before any attention or MLP has processed the input.

        Returns
        -------
        pd.DataFrame
            Columns: layer_idx, layer_label, logit_diff, prob_io, prob_s,
                     logit_io, logit_s, mean_is_correct.
            One row per layer checkpoint (n_layers + 1 rows).

        Examples
        --------
        >>> df = analyzer.run(batch_size=20)
        >>> print(df[["layer_label", "logit_diff", "prob_io"]])
        """
        logger.info(
            f"[LogitLensAnalyzer.run] Running logit lens on "
            f"{self.n_samples} prompts…"
        )

        # Only cache the hook points we need to reduce memory
        names_to_cache = (
            ["hook_embed", "hook_pos_embed"]
            + [f"blocks.{l}.hook_resid_post" for l in range(self.n_layers)]
        )

        # Accumulator: layer_idx → list of per-prompt metrics
        layer_accumulator: dict[int, list[dict]] = {
            l: [] for l in range(self.n_layers + 1)
        }

        prompts = self.dataset.get_clean_prompts()[:self.n_samples]
        io_ids = self.dataset.get_io_token_ids()[:self.n_samples]
        s_ids = self.dataset.get_s_token_ids()[:self.n_samples]

        for batch_start in range(0, self.n_samples, batch_size):
            batch_end = min(batch_start + batch_size, self.n_samples)
            batch_prompts = prompts[batch_start:batch_end]
            batch_io = io_ids[batch_start:batch_end]
            batch_s = s_ids[batch_start:batch_end]

            # ── Tokenise batch ────────────────────────────────────────────
            token_lists = [
                self.model.to_tokens(p, prepend_bos=True)[0].tolist()
                for p in batch_prompts
            ]
            seq_lengths = [len(t) for t in token_lists]
            max_len = max(seq_lengths)
            bos_id = self.model.tokenizer.bos_token_id

            padded = [
                t + [bos_id] * (max_len - len(t)) for t in token_lists
            ]
            tokens = torch.tensor(padded, dtype=torch.long, device=self.device)

            # ── Run with cache (only the needed hook points) ──────────────
            # `names_filter` limits which activations are stored → saves memory
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=names_to_cache,
            )

            # ── Apply logit lens per prompt ───────────────────────────────
            for i in range(len(batch_prompts)):
                io_id = batch_io[i]
                s_id = batch_s[i]
                final_pos = seq_lengths[i] - 1  # last real token index

                # Build a single-prompt ActivationCache by slicing each
                # cached tensor along the batch dimension.
                single_cache_dict = {
                    key: cache[key][i:i+1]  # shape: [1, seq, ...]
                    for key in names_to_cache
                    if key in cache.cache_dict
                }
                single_cache = ActivationCache(single_cache_dict, self.model)

                results = self._compute_logit_lens_single_cache(
                    single_cache, io_id, s_id, final_pos
                )
                for layer_idx, metrics in results.items():
                    layer_accumulator[layer_idx].append(metrics)

            # Free GPU memory
            del cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            logger.debug(
                f"[LogitLensAnalyzer] Processed {batch_end}/{self.n_samples} prompts"
            )

        # ── Aggregate across prompts ──────────────────────────────────────
        rows = []
        for layer_idx in range(self.n_layers + 1):
            metrics_list = layer_accumulator[layer_idx]
            if not metrics_list:
                continue

            avg = {
                k: float(np.mean([m[k] for m in metrics_list]))
                for k in metrics_list[0]
            }
            std_ld = float(np.std([m["logit_diff"] for m in metrics_list]))
            fraction_correct = float(
                np.mean([m["logit_diff"] > 0 for m in metrics_list])
            )

            if layer_idx == 0:
                label = "Embed"
            else:
                label = f"L{layer_idx - 1}"  # L0 = after block 0

            rows.append({
                "layer_idx": layer_idx,
                "layer_label": label,
                "logit_diff": avg["logit_diff"],
                "logit_diff_std": std_ld,
                "prob_io": avg["prob_io"],
                "prob_s": avg["prob_s"],
                "logit_io": avg["logit_io"],
                "logit_s": avg["logit_s"],
                "fraction_correct": fraction_correct,
            })

        df = pd.DataFrame(rows)

        logger.info(
            f"[LogitLensAnalyzer.run] ✓ Complete.\n"
            f"  Layers analysed: {len(df)}\n"
            f"  Max logit diff (layer {df['logit_diff'].idxmax()}): "
            f"{df['logit_diff'].max():+.4f}\n"
            f"  First layer > 0: "
            f"{df[df['logit_diff'] > 0]['layer_label'].values[0] if (df['logit_diff'] > 0).any() else 'none'}"
        )

        return df

    @torch.no_grad()
    def run_per_token_position(
        self,
        prompt_idx: int = 0,
        layer: int = 11,
    ) -> pd.DataFrame:
        """
        Run logit lens at a specific layer across ALL token positions.

        This reveals which token positions contribute most to the final
        IOI decision at a given layer. Important positions include:
          - Position of IO name (first occurrence)
          - Position of S name (first and second occurrences)
          - Final "to" token position

        Parameters
        ----------
        prompt_idx : int
            Index into the dataset. Default: 0 (first prompt).

        layer : int
            Which transformer layer to examine (0 = after block 0,
            n_layers-1 = final layer).

        Returns
        -------
        pd.DataFrame
            Columns: position, token_str, logit_diff, prob_io, logit_io, logit_s.
            One row per token position in the prompt.

        Examples
        --------
        >>> df = analyzer.run_per_token_position(prompt_idx=0, layer=11)
        >>> print(df[["position", "token_str", "logit_diff"]])
        """
        import torch.nn.functional as F

        prompt = self.dataset.prompts[prompt_idx]
        io_id = prompt.io_token_id
        s_id = prompt.s_token_id

        tokens = self.model.to_tokens(prompt.prompt_clean, prepend_bos=True)
        str_tokens = self.model.to_str_tokens(prompt.prompt_clean, prepend_bos=True)
        seq_len = tokens.shape[1]

        # Run with cache (only need the specified layer's resid_post)
        hook_name = f"blocks.{layer}.hook_resid_post"
        _, cache = self.model.run_with_cache(
            tokens,
            names_filter=[hook_name],
        )

        # resid shape: [1, seq_len, d_model]
        resid = cache[hook_name][0]  # [seq_len, d_model]

        rows = []
        for pos in range(seq_len):
            h = resid[pos:pos+1]  # [1, d_model]
            h_normed = self.model.ln_final(h)
            logits = self.model.unembed(h_normed.unsqueeze(0)).squeeze(0).squeeze(0)

            probs = F.softmax(logits, dim=-1)
            logit_io = logits[io_id].item()
            logit_s = logits[s_id].item()

            rows.append({
                "position": pos,
                "token_str": str_tokens[pos] if pos < len(str_tokens) else "?",
                "logit_diff": logit_io - logit_s,
                "prob_io": probs[io_id].item(),
                "prob_s": probs[s_id].item(),
                "logit_io": logit_io,
                "logit_s": logit_s,
            })

        return pd.DataFrame(rows)
