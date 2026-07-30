"""
src/analysis/activation_patching.py
=====================================
Clean-to-corrupted activation patching experiments for the IOI circuit.

Theory: Activation Patching
-----------------------------
Activation patching (also called "causal tracing" by Meng et al., 2022)
is the core experimental technique in mechanistic interpretability. It works
as follows:

1. **Clean run**: Run the model on the clean IOI prompt (e.g.,
   "When Alice and Bob went to the park, Bob gave the book to").
   This produces a high logit_diff (model correctly prefers Alice).
   Record ALL activations: cache_clean.

2. **Corrupted run**: Run the model on the corrupted prompt (same structure,
   but subject Bob → distractor Carol:
   "When Alice and Carol went to the park, Carol gave the book to").
   This produces a low (often negative) logit_diff (model is confused).
   Record ALL activations: cache_corrupted.

3. **Patching**: Run the model on the CORRUPTED input, but replace one
   specific activation with the corresponding value from cache_clean.
   Measure the resulting logit_diff.

4. **Restoration Score**: How much did patching that activation restore
   the correct behaviour?
       score = (patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)
   - score ≈ 1.0 → patching this activation fully restores clean behaviour
                    → this activation is critical for the IOI circuit
   - score ≈ 0.0 → patching this activation has no effect
                    → this activation is not involved in the circuit
   - score < 0.0 → patching makes things worse (rare; indicates anti-correlation)

Patching Sites
--------------
We patch three types of activations:

A. Residual stream (`blocks.{l}.hook_resid_post`)
   The entire hidden state after layer l. Patching this at a specific
   (layer, token_position) pair tells us: "is the information at layer l,
   position p critical for the IOI task?"

B. Attention output (`blocks.{l}.hook_attn_out`)
   The summed output of all attention heads in layer l.
   Patching layer-by-layer reveals which attention layers process IOI-
   relevant information.

C. MLP output (`blocks.{l}.hook_mlp_out`)
   The output of the MLP in layer l.
   Patching reveals whether the IOI circuit runs through MLP computations
   or purely through attention.

Token Position Semantics (IOI)
-------------------------------
For the ABB template "When {IO} and {S} went to {place}, {S} gave the {obj} to":
  Position 0: <BOS>
  Position 1: "When"
  Position 2: " {IO}"       ← IO name first occurrence
  Position 3: " and"
  Position 4: " {S}"        ← S name first occurrence
  Position 5: " went"
  Position k: " {S}"        ← S name SECOND occurrence (critical!)
  Position -1: " to"        ← FINAL position where model predicts

The circuit operates primarily on the FINAL TOKEN POSITION: the model
integrates information from all previous positions to predict the IO name.

References
----------
Meng et al. (2022). "Locating and Editing Factual Associations in GPT."
  https://arxiv.org/abs/2202.05262 (ROME paper, introduced causal tracing)

Wang et al. (2022). "Interpretability in the Wild."
  https://arxiv.org/abs/2202.00571

Heimersheim & Nanda (2024). "How to use and interpret activation patching."
  https://arxiv.org/abs/2404.15255
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer, ActivationCache

from src.data.ioi_dataset import IOIDataset
from src.evaluation.metrics import compute_logit_diff

logger = logging.getLogger(__name__)


class ActivationPatchingAnalyzer:
    """
    Implements clean→corrupted activation patching for IOI circuit analysis.

    Patches residual stream, attention outputs, and MLP outputs at every
    (layer, token_position) pair and measures the restoration score.

    Parameters
    ----------
    model : HookedTransformer
        Loaded GPT-2 Small in eval mode.

    dataset : IOIDataset
        Generated IOI dataset with clean and corrupted prompt pairs.

    n_samples : int
        Number of (clean, corrupted) prompt pairs to average over.

    batch_size : int
        Prompts per forward pass. Keep small (8–16) since we run 3 forward
        passes per sample pair (clean cache, corrupted cache, patched run).

    Examples
    --------
    >>> analyzer = ActivationPatchingAnalyzer(model, dataset, n_samples=100)
    >>> results = analyzer.run_resid_patching()
    >>> print(results["restoration_score"].max())  # Should be ~1.0 at some layer
    """

    def __init__(
        self,
        model: HookedTransformer,
        dataset: IOIDataset,
        n_samples: int = 100,
        batch_size: int = 8,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.n_samples = min(n_samples, len(dataset))
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        self.n_layers = model.cfg.n_layers

        logger.info(
            f"[ActivationPatchingAnalyzer] Initialized: "
            f"n_samples={self.n_samples}, batch_size={batch_size}"
        )

    def _tokenize_single(self, prompt: str) -> torch.Tensor:
        """Tokenise a single prompt and return 1D tensor of token IDs."""
        return self.model.to_tokens(prompt, prepend_bos=True)[0]

    @torch.no_grad()
    def compute_clean_corrupted_caches(
        self,
        prompt_idx: int,
        names_filter: list[str],
    ) -> tuple[ActivationCache, ActivationCache, float, float, int]:
        """
        Run both clean and corrupted prompts for a single dataset entry.

        Caches all activations in `names_filter` for both runs, and computes
        the clean and corrupted logit differences.

        Parameters
        ----------
        prompt_idx : int
            Index into the dataset.

        names_filter : list[str]
            Hook names to cache (to limit memory).

        Returns
        -------
        tuple of:
            - cache_clean       : ActivationCache from clean prompt run
            - cache_corrupted   : ActivationCache from corrupted prompt run
            - clean_ld          : float, logit_diff on clean prompt
            - corrupted_ld      : float, logit_diff on corrupted prompt
            - seq_len           : int, sequence length (same for both prompts
                                  since only names change, not structure)
        """
        prompt = self.dataset.prompts[prompt_idx]
        io_id = prompt.io_token_id
        s_id = prompt.s_token_id

        # Tokenise both prompts
        tokens_clean = self.model.to_tokens(
            prompt.prompt_clean, prepend_bos=True
        )  # [1, seq]
        tokens_corrupted = self.model.to_tokens(
            prompt.prompt_corrupted, prepend_bos=True
        )  # [1, seq]

        seq_len_clean = tokens_clean.shape[1]
        seq_len_corrupted = tokens_corrupted.shape[1]

        # Use shorter seq_len for indexing (both should be same length
        # since only names change, but handle edge case)
        seq_len = min(seq_len_clean, seq_len_corrupted)

        # Run clean prompt, cache activations
        logits_clean, cache_clean = self.model.run_with_cache(
            tokens_clean, names_filter=names_filter
        )
        clean_ld = compute_logit_diff(
            logits_clean[0, seq_len_clean - 1, :], io_id, s_id
        )

        # Run corrupted prompt, cache activations
        logits_corrupted, cache_corrupted = self.model.run_with_cache(
            tokens_corrupted, names_filter=names_filter
        )
        corrupted_ld = compute_logit_diff(
            logits_corrupted[0, seq_len_corrupted - 1, :], io_id, s_id
        )

        return cache_clean, cache_corrupted, clean_ld, corrupted_ld, seq_len

    @torch.no_grad()
    def patch_and_measure(
        self,
        prompt_idx: int,
        hook_name: str,
        token_pos: int,
        cache_clean: ActivationCache,
        corrupted_tokens: torch.Tensor,
        seq_len: int,
    ) -> float:
        """
        Patch a single activation at a specific (hook_name, token_pos)
        and measure the resulting logit difference.

        The model runs on the CORRUPTED input but the activation at
        (hook_name, token_pos) is replaced by the CLEAN value.

        Parameters
        ----------
        prompt_idx : int
            Dataset index for io/s token IDs.

        hook_name : str
            The activation to patch (e.g., "blocks.5.hook_resid_post").

        token_pos : int
            Token position to patch (0 = BOS, -1 = last token, etc.).

        cache_clean : ActivationCache
            Activations from the clean prompt run.

        corrupted_tokens : torch.Tensor
            Tokenised corrupted prompt, shape [1, seq_len].

        seq_len : int
            Actual sequence length.

        Returns
        -------
        float
            The logit_diff (IO - S) at the final token position with
            the patched activation.

        Hook Implementation Details
        ----------------------------
        The hook function receives the activation during the corrupted forward
        pass and selectively replaces token position `token_pos` with the
        clean activation at the same position. All other positions are left
        unchanged.

        This is called "positional patching" — we patch one position at a time
        to localise exactly WHERE in the sequence the critical information lives.
        """
        prompt = self.dataset.prompts[prompt_idx]
        io_id = prompt.io_token_id
        s_id = prompt.s_token_id

        # Get clean activation for this hook_name at this position
        # clean_act shape: [1, seq, d_model] or [1, seq, n_heads, d_head]
        clean_act = cache_clean[hook_name]

        def patching_hook(corrupted_act: torch.Tensor, hook) -> torch.Tensor:
            """
            Replace position `token_pos` in the corrupted activation with
            the corresponding clean activation.

            This is the core patching operation. The hook is called during
            the corrupted forward pass. We modify only one token position
            to isolate the causal effect of that position's information.

            corrupted_act shape: [batch, seq, *dims]
            clean_act shape    : [1, seq, *dims]
            """
            result = corrupted_act.clone()

            # Handle negative indexing (token_pos=-1 → last real token)
            pos = token_pos if token_pos >= 0 else seq_len + token_pos

            if pos < 0 or pos >= seq_len:
                return result  # Out-of-bounds; no-op

            # Replace this position with clean activation
            # clean_act[0, pos, ...] has shape [*dims]
            # result[:, pos, ...] has shape [batch, *dims]
            result[:, pos, ...] = clean_act[0, pos, ...].unsqueeze(0).expand(
                result.shape[0], *clean_act.shape[2:]
            )
            return result

        logits = self.model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, patching_hook)],
        )

        patched_ld = compute_logit_diff(
            logits[0, seq_len - 1, :], io_id, s_id
        )
        return patched_ld

    @torch.no_grad()
    def run_layer_position_sweep(
        self,
        hook_pattern: str = "blocks.{l}.hook_resid_post",
        desc: str = "Residual stream patching",
    ) -> pd.DataFrame:
        """
        Patch a specified activation type at every (layer, token_position) pair.

        This is the core patching experiment. The result is a 2D matrix:
            rows = token positions (0 to seq_len-1)
            cols = layers (0 to n_layers-1)
            values = mean restoration score across n_samples prompts

        A high restoration score at (layer l, position p) means that the
        information at position p after layer l is causally important for IOI.

        Parameters
        ----------
        hook_pattern : str
            Hook name pattern with `{l}` as layer placeholder.
            Options:
              "blocks.{l}.hook_resid_post" → residual stream (default)
              "blocks.{l}.hook_attn_out"   → attention outputs
              "blocks.{l}.hook_mlp_out"    → MLP outputs

        desc : str
            Progress bar description.

        Returns
        -------
        pd.DataFrame
            Rows indexed by token position, columns indexed by layer.
            Values are mean restoration scores ∈ [-∞, +∞], typically [0, 1].
            Higher = more important.

        Examples
        --------
        >>> results = analyzer.run_layer_position_sweep("blocks.{l}.hook_resid_post")
        >>> print(results.shape)   # (seq_len, n_layers)
        """
        hook_names = [
            hook_pattern.format(l=l) for l in range(self.n_layers)
        ]

        # Accumulate restoration scores: {layer: {position: [list of scores]}}
        # We use the shortest prompt length as the common seq_len
        all_seq_lens = []
        for i in range(self.n_samples):
            prompt = self.dataset.prompts[i]
            t = self.model.to_tokens(prompt.prompt_clean, prepend_bos=True)
            all_seq_lens.append(t.shape[1])
        min_seq_len = min(all_seq_lens)

        # score_matrix[layer][pos] = list of restoration scores
        score_matrix: dict[int, dict[int, list[float]]] = {
            l: {p: [] for p in range(min_seq_len)}
            for l in range(self.n_layers)
        }
        clean_lds: list[float] = []
        corrupted_lds: list[float] = []

        logger.info(
            f"[ActivationPatchingAnalyzer] Running {desc} over "
            f"{self.n_samples} prompts × {self.n_layers} layers × "
            f"{min_seq_len} positions = "
            f"{self.n_samples * self.n_layers * min_seq_len:,} patches…"
        )

        for i in tqdm(range(self.n_samples), desc=desc, unit="prompt"):
            prompt = self.dataset.prompts[i]
            tokens_corrupted = self.model.to_tokens(
                prompt.prompt_corrupted, prepend_bos=True
            )

            # Run clean and corrupted, get caches
            cache_clean, _, clean_ld, corrupted_ld, seq_len = (
                self.compute_clean_corrupted_caches(i, hook_names)
            )
            clean_lds.append(clean_ld)
            corrupted_lds.append(corrupted_ld)

            # Skip if clean and corrupted are too similar (circuit didn't engage)
            ld_gap = clean_ld - corrupted_ld
            if abs(ld_gap) < 0.1:
                continue

            # Patch each (layer, position) combination
            for l, hook_name in enumerate(hook_names):
                for pos in range(min(min_seq_len, seq_len)):
                    patched_ld = self.patch_and_measure(
                        i, hook_name, pos, cache_clean, tokens_corrupted, seq_len
                    )
                    # Restoration score: 0 = no restoration, 1 = full restoration
                    restoration = (patched_ld - corrupted_ld) / ld_gap
                    score_matrix[l][pos].append(restoration)

            # Free memory
            del cache_clean
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── Build DataFrame ───────────────────────────────────────────────
        mean_scores = np.full((min_seq_len, self.n_layers), np.nan)
        for l in range(self.n_layers):
            for pos in range(min_seq_len):
                scores = score_matrix[l][pos]
                if scores:
                    mean_scores[pos, l] = np.mean(scores)

        # Build token label for first prompt (representative)
        sample_prompt = self.dataset.prompts[0].prompt_clean
        str_tokens = self.model.to_str_tokens(sample_prompt, prepend_bos=True)
        token_labels = [
            f"pos{p}: {str_tokens[p]!r}" if p < len(str_tokens) else f"pos{p}"
            for p in range(min_seq_len)
        ]

        df = pd.DataFrame(
            mean_scores,
            index=token_labels,
            columns=[f"L{l}" for l in range(self.n_layers)],
        )
        df.index.name = "token_position"

        logger.info(
            f"[ActivationPatchingAnalyzer] ✓ {desc} complete.\n"
            f"  Mean clean LD    : {np.mean(clean_lds):+.4f}\n"
            f"  Mean corrupted LD: {np.mean(corrupted_lds):+.4f}\n"
            f"  Max restoration  : {np.nanmax(mean_scores):.4f} at "
            f"(pos={np.unravel_index(np.nanargmax(mean_scores), mean_scores.shape)[0]}, "
            f"L{np.unravel_index(np.nanargmax(mean_scores), mean_scores.shape)[1]})"
        )

        return df

    def run_resid_patching(self, **kwargs) -> pd.DataFrame:
        """
        Patch residual stream at every (layer, position).
        Convenience wrapper around `run_layer_position_sweep`.
        """
        return self.run_layer_position_sweep(
            hook_pattern="blocks.{l}.hook_resid_post",
            desc="Residual stream patching",
            **kwargs,
        )

    def run_attn_patching(self, **kwargs) -> pd.DataFrame:
        """
        Patch attention output at every (layer, position).
        Convenience wrapper around `run_layer_position_sweep`.
        """
        return self.run_layer_position_sweep(
            hook_pattern="blocks.{l}.hook_attn_out",
            desc="Attention output patching",
            **kwargs,
        )

    def run_mlp_patching(self, **kwargs) -> pd.DataFrame:
        """
        Patch MLP output at every (layer, position).
        Convenience wrapper around `run_layer_position_sweep`.
        """
        return self.run_layer_position_sweep(
            hook_pattern="blocks.{l}.hook_mlp_out",
            desc="MLP output patching",
            **kwargs,
        )

    def run_all_patching_experiments(self) -> dict[str, pd.DataFrame]:
        """
        Run all three patching experiments (resid, attn, MLP) sequentially.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: "resid", "attn", "mlp".
            Each value: restoration score matrix [n_positions × n_layers].

        Examples
        --------
        >>> all_results = analyzer.run_all_patching_experiments()
        >>> resid_df = all_results["resid"]
        >>> attn_df = all_results["attn"]
        """
        logger.info("[ActivationPatchingAnalyzer] Running all patching experiments…")
        results = {
            "resid": self.run_resid_patching(),
            "attn": self.run_attn_patching(),
            "mlp": self.run_mlp_patching(),
        }
        logger.info("[ActivationPatchingAnalyzer] ✓ All patching experiments complete.")
        return results
