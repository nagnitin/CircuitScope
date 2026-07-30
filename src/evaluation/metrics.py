"""
src/evaluation/metrics.py
==========================
Metric computation for the IOI baseline evaluation.

Metrics Computed Per Prompt
-----------------------------
For each IOI prompt we run a forward pass through GPT-2 Small and record:

1. logit_io (float)
   The raw logit assigned to the IO token at the FINAL position.
   "Raw logit" = the pre-softmax score from the unembedding matrix W_U.
   Higher is better. This is the most direct signal of the model's belief.

2. logit_s (float)
   The raw logit assigned to the S (subject) token at the final position.

3. logit_diff (float)
   logit_io - logit_s
   THE primary metric in circuit analysis. Positive = model correctly
   prefers IO. Negative = model is wrong. Zero = equal probability.
   Logit difference is preferred over accuracy because it is:
     (a) differentiable (useful for gradient-based analysis)
     (b) scalar-valued (easy to aggregate)
     (c) sensitive to margin, not just argmax

4. prob_io (float)
   Softmax probability of the IO token: exp(logit_io) / Σ exp(logits).
   Range [0, 1]. Represents the model's "confidence" in the correct answer.

5. prob_s (float)
   Softmax probability of the S token.

6. is_correct (bool)
   True if logit_io > logit_s (i.e., model prefers IO over S).
   This is a binary accuracy metric — it does NOT check if IO is the
   argmax over the entire 50,257-token vocabulary, only vs. the S token.

7. top_1_prediction (str)
   The string token with the highest logit in the entire vocabulary.

8. top_k_predictions (list of str)
   The top-k token strings by logit (default k=5).

9. top_k_logits (list of float)
   Logits corresponding to top_k_predictions.

10. rank_io (int)
    The rank of the IO token in the full vocabulary sorted by logit.
    Rank 1 = IO is the top prediction. Rank 50257 = IO is last.

Implementation Notes
--------------------
Batched Forward Pass
~~~~~~~~~~~~~~~~~~~~
We tokenise all prompts first and then run them in batches through
`model(tokens)` rather than one at a time. This is 10–100× faster on GPU.

Final Token Position
~~~~~~~~~~~~~~~~~~~~
IOI prompts end with "... gave the {obj} to" (no completion). The model
predicts what comes AFTER the final "to". In TransformerLens:

    logits = model(tokens)                  # shape: [batch, seq_len, d_vocab]
    final_logits = logits[:, -1, :]         # shape: [batch, d_vocab]

We index `[:, -1, :]` to get the distribution at the last token position.

Padding for Batched Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
IOI prompts have different lengths depending on the place/object fillers.
We pad shorter sequences to the length of the longest sequence in each batch:
  - Padding is done with the BOS token (GPT-2 has no dedicated PAD token).
  - We only read logits at the LAST non-padding position for each prompt.
  - `attention_mask` is set so padding tokens don't attend.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.data.ioi_dataset import IOIDataset, IOIPrompt

logger = logging.getLogger(__name__)


def compute_logit_diff(
    logits: torch.Tensor,
    io_token_id: int,
    s_token_id: int,
) -> float:
    """
    Compute the logit difference between the IO and S tokens.

    This is the canonical metric for IOI circuit analysis, as introduced
    by Wang et al. (2022). It measures the model's preference for the
    correct answer over the incorrect one in raw (pre-softmax) logit space.

    Parameters
    ----------
    logits : torch.Tensor
        Shape: [d_vocab]. The logit vector at the final token position
        for a single prompt. Must be a 1D tensor.

    io_token_id : int
        Token ID of the indirect object (correct answer).

    s_token_id : int
        Token ID of the subject (incorrect answer).

    Returns
    -------
    float
        logit[io_token_id] - logit[s_token_id].
        Positive → model correctly prefers IO.
        Negative → model incorrectly prefers S.

    Examples
    --------
    >>> logits = torch.randn(50257)
    >>> logit_diff = compute_logit_diff(logits, io_token_id=5765, s_token_id=3329)
    >>> print(logit_diff > 0)  # True if model is correct
    """
    assert logits.ndim == 1, f"Expected 1D logit tensor, got shape {logits.shape}"
    return (logits[io_token_id] - logits[s_token_id]).item()


def compute_top_k(
    logits: torch.Tensor,
    tokenizer,
    k: int = 5,
) -> tuple[list[str], list[float]]:
    """
    Return the top-k token strings and their logit values.

    Parameters
    ----------
    logits : torch.Tensor
        Shape: [d_vocab]. Logit vector at the final token position.

    tokenizer
        The model's tokenizer (accessed via `model.tokenizer`).
        Used to decode token IDs to human-readable strings.

    k : int
        Number of top predictions to return. Default: 5.

    Returns
    -------
    tuple of (list[str], list[float])
        - top_k_tokens: decoded token strings (may include leading spaces)
        - top_k_logit_values: corresponding raw logit values

    Notes
    -----
    `torch.topk` is O(d_vocab) and faster than sorting the full vocab.
    We use it instead of `logits.argsort(descending=True)[:k]`.

    Examples
    --------
    >>> tokens, logit_vals = compute_top_k(final_logits[0], model.tokenizer)
    >>> print(tokens)    # e.g., [' Mary', ' Alice', ' the', ' Bob', ' she']
    """
    # Get top-k indices and values (faster than full sort)
    top_values, top_indices = torch.topk(logits, k=k)

    top_tokens = tokenizer.batch_decode(top_indices.unsqueeze(-1))
    top_logit_values = top_values.tolist()

    return top_tokens, top_logit_values


def compute_token_rank(logits: torch.Tensor, token_id: int) -> int:
    """
    Compute the rank of a specific token in the full vocabulary logit ranking.

    Rank 1 = highest logit (best prediction).
    Rank d_vocab = lowest logit (worst prediction).

    Parameters
    ----------
    logits : torch.Tensor
        Shape: [d_vocab]. Full logit vector.

    token_id : int
        The token whose rank we want.

    Returns
    -------
    int
        1-indexed rank of the token (1 = best).

    Examples
    --------
    >>> rank = compute_token_rank(logits, io_token_id)
    >>> print(f"IO token is rank {rank} out of {len(logits)}")
    """
    # argsort descending: index 0 = highest logit token
    sorted_indices = logits.argsort(descending=True)
    # Find position of our token_id in the sorted list
    rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
    return int(rank)


class IOIEvaluator:
    """
    Batch evaluator for computing all IOI metrics across the full dataset.

    This class handles:
      1. Tokenising prompts (with padding for batch processing)
      2. Running batched forward passes through the model
      3. Extracting per-prompt metrics at the final token position
      4. Assembling results into a pandas DataFrame

    Parameters
    ----------
    model : HookedTransformer
        Loaded GPT-2 Small model (eval mode, on target device).

    dataset : IOIDataset
        A fully generated IOIDataset (after calling `.generate()`).

    batch_size : int
        Number of prompts to process in each forward pass.
        Recommended: 32 for GPU, 8–16 for CPU.

    top_k : int
        Number of top-k predictions to record per prompt. Default: 5.

    Examples
    --------
    >>> evaluator = IOIEvaluator(model, dataset, batch_size=32, top_k=5)
    >>> results_df = evaluator.evaluate()
    >>> print(results_df.shape)   # (1000, 14)
    >>> print(results_df["logit_diff"].mean())   # Should be > 0 for a capable model
    """

    def __init__(
        self,
        model: HookedTransformer,
        dataset: IOIDataset,
        batch_size: int = 32,
        top_k: int = 5,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.top_k = top_k
        self.device = next(model.parameters()).device

        logger.info(
            f"[IOIEvaluator] Initialized: {len(dataset)} prompts, "
            f"batch_size={batch_size}, top_k={top_k}, device={self.device}."
        )

    def _tokenize_batch(
        self,
        prompts: list[str],
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Tokenise a list of prompt strings into a padded batch tensor.

        Handles variable-length sequences by padding with the BOS token
        (GPT-2 has no dedicated PAD token; BOS is the standard choice).

        Parameters
        ----------
        prompts : list[str]
            Batch of prompt strings to tokenise.

        Returns
        -------
        tuple of (tokens_tensor, seq_lengths)
            - tokens_tensor : [batch_size, max_seq_len] LongTensor on device
            - seq_lengths   : list of int — actual sequence length per prompt
                              (excluding padding), used to find the last real token

        Notes
        -----
        We prepend BOS (as TransformerLens does by default) because GPT-2
        was trained with BOS prepended. Skipping BOS would shift all token
        positions and produce incorrect logits.
        """
        # Tokenise each prompt individually (respects variable lengths)
        token_lists = [
            self.model.to_tokens(p, prepend_bos=True)[0].tolist()
            for p in prompts
        ]

        seq_lengths = [len(t) for t in token_lists]
        max_len = max(seq_lengths)
        bos_id = self.model.tokenizer.bos_token_id

        # Right-pad with BOS token to max_len
        padded = [
            t + [bos_id] * (max_len - len(t))
            for t in token_lists
        ]

        tokens_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
        return tokens_tensor, seq_lengths

    @torch.no_grad()
    def _forward_pass(
        self,
        tokens: torch.Tensor,
        seq_lengths: list[int],
    ) -> torch.Tensor:
        """
        Run a batched forward pass and return final-position logits.

        Parameters
        ----------
        tokens : torch.Tensor
            Shape: [batch, max_seq_len]. Padded token IDs.

        seq_lengths : list[int]
            Actual (unpadded) length of each sequence in the batch.

        Returns
        -------
        torch.Tensor
            Shape: [batch, d_vocab]. Logits at the final real token position
            for each prompt in the batch.

        TransformerLens Note
        --------------------
        `model(tokens)` returns logits of shape [batch, seq_len, d_vocab].
        We extract the logit at position `seq_len - 1` (last real token)
        for each sequence individually, since sequences may have different lengths.
        """
        # Full forward pass: logits shape = [batch, max_seq_len, d_vocab]
        all_logits = self.model(tokens)

        # Extract final real token position for each sequence in the batch
        final_logits = torch.stack(
            [all_logits[i, seq_lengths[i] - 1, :] for i in range(len(seq_lengths))]
        )  # shape: [batch, d_vocab]

        return final_logits

    @torch.no_grad()
    def evaluate(self, use_corrupted: bool = False) -> pd.DataFrame:
        """
        Run the full evaluation pipeline over the entire IOI dataset.

        For each prompt, computes:
          - logit_io, logit_s
          - logit_diff = logit_io - logit_s
          - prob_io, prob_s (softmax probabilities)
          - is_correct = logit_io > logit_s
          - rank_io (rank of IO in full vocabulary)
          - top_1_prediction (string)
          - top_k_predictions (pipe-separated string for CSV compatibility)
          - top_k_logits (pipe-separated string)

        Parameters
        ----------
        use_corrupted : bool
            If False (default): evaluate on clean prompts.
            If True: evaluate on corrupted prompts (for patching baseline).

        Returns
        -------
        pd.DataFrame
            One row per prompt. Includes all dataset metadata columns
            plus computed metric columns.

        Examples
        --------
        >>> results = evaluator.evaluate(use_corrupted=False)
        >>> print(results["is_correct"].mean())   # Accuracy
        >>> print(results["logit_diff"].mean())   # Mean logit difference
        """
        prompts_list = (
            self.dataset.get_corrupted_prompts()
            if use_corrupted
            else self.dataset.get_clean_prompts()
        )
        io_token_ids = self.dataset.get_io_token_ids()
        s_token_ids = self.dataset.get_s_token_ids()
        n = len(prompts_list)
        prompt_type = "corrupted" if use_corrupted else "clean"

        logger.info(
            f"[IOIEvaluator.evaluate] Evaluating {n} {prompt_type} prompts "
            f"in batches of {self.batch_size}…"
        )

        # ── Result containers ─────────────────────────────────────────────
        records: list[dict] = []

        # ── Batch loop ────────────────────────────────────────────────────
        for batch_start in tqdm(
            range(0, n, self.batch_size),
            desc=f"Evaluating {prompt_type} prompts",
            unit="batch",
        ):
            batch_end = min(batch_start + self.batch_size, n)
            batch_prompts = prompts_list[batch_start:batch_end]
            batch_io_ids = io_token_ids[batch_start:batch_end]
            batch_s_ids = s_token_ids[batch_start:batch_end]

            # Tokenise batch (handles variable lengths via padding)
            tokens, seq_lengths = self._tokenize_batch(batch_prompts)

            # Forward pass → final-position logits: [batch, d_vocab]
            final_logits = self._forward_pass(tokens, seq_lengths)

            # Softmax probabilities: [batch, d_vocab]
            probs = F.softmax(final_logits, dim=-1)

            # ── Per-prompt metrics ────────────────────────────────────────
            for i in range(len(batch_prompts)):
                logits_i = final_logits[i]   # shape: [d_vocab]
                probs_i = probs[i]
                io_id = batch_io_ids[i]
                s_id = batch_s_ids[i]
                prompt_idx = batch_start + i

                # Dataset metadata for this prompt
                meta = self.dataset.prompts[prompt_idx].to_dict()

                # ── Core IOI metrics ──────────────────────────────────────
                logit_io = logits_i[io_id].item()
                logit_s = logits_i[s_id].item()
                logit_diff = logit_io - logit_s

                prob_io = probs_i[io_id].item()
                prob_s = probs_i[s_id].item()

                is_correct = logit_diff > 0

                # ── Rank of IO in full vocabulary ─────────────────────────
                rank_io = compute_token_rank(logits_i, io_id)

                # ── Top-k predictions ─────────────────────────────────────
                top_k_tokens, top_k_logit_vals = compute_top_k(
                    logits_i, self.model.tokenizer, k=self.top_k
                )
                top_1_pred = top_k_tokens[0] if top_k_tokens else ""

                # ── Assemble record ───────────────────────────────────────
                record = {
                    **meta,  # spread all dataset metadata fields
                    "prompt_type": prompt_type,
                    "logit_io": round(logit_io, 6),
                    "logit_s": round(logit_s, 6),
                    "logit_diff": round(logit_diff, 6),
                    "prob_io": round(prob_io, 8),
                    "prob_s": round(prob_s, 8),
                    "is_correct": is_correct,
                    "rank_io": rank_io,
                    "top_1_prediction": top_1_pred.strip(),
                    "top_k_predictions": " | ".join(t.strip() for t in top_k_tokens),
                    "top_k_logits": " | ".join(f"{v:.4f}" for v in top_k_logit_vals),
                }
                records.append(record)

        # ── Assemble DataFrame ────────────────────────────────────────────
        df = pd.DataFrame(records)

        # ── Summary statistics ────────────────────────────────────────────
        accuracy = df["is_correct"].mean()
        mean_logit_diff = df["logit_diff"].mean()
        mean_rank_io = df["rank_io"].mean()

        logger.info(
            f"[IOIEvaluator.evaluate] ✓ Evaluation complete.\n"
            f"  Prompts evaluated : {len(df):,}\n"
            f"  Accuracy (IO > S) : {accuracy:.1%}\n"
            f"  Mean logit diff   : {mean_logit_diff:+.4f}\n"
            f"  Mean IO rank      : {mean_rank_io:.1f} / {self.model.cfg.d_vocab:,}"
        )

        return df

    def compute_aggregate_stats(self, results_df: pd.DataFrame) -> dict:
        """
        Compute aggregate statistics from an evaluation results DataFrame.

        Parameters
        ----------
        results_df : pd.DataFrame
            Output of `self.evaluate()`.

        Returns
        -------
        dict
            Dictionary with keys:
            - accuracy        : float, fraction of prompts where IO > S
            - mean_logit_diff : float
            - std_logit_diff  : float
            - median_logit_diff: float
            - mean_prob_io    : float, average softmax prob of correct token
            - mean_rank_io    : float
            - accuracy_abb    : float, accuracy on ABB templates only
            - accuracy_bab    : float, accuracy on BAB templates only

        Examples
        --------
        >>> stats = evaluator.compute_aggregate_stats(results_df)
        >>> print(f"Overall accuracy: {stats['accuracy']:.1%}")
        """
        stats = {
            "n_prompts": len(results_df),
            "accuracy": results_df["is_correct"].mean(),
            "mean_logit_diff": results_df["logit_diff"].mean(),
            "std_logit_diff": results_df["logit_diff"].std(),
            "median_logit_diff": results_df["logit_diff"].median(),
            "min_logit_diff": results_df["logit_diff"].min(),
            "max_logit_diff": results_df["logit_diff"].max(),
            "mean_prob_io": results_df["prob_io"].mean(),
            "mean_prob_s": results_df["prob_s"].mean(),
            "mean_rank_io": results_df["rank_io"].mean(),
            "median_rank_io": results_df["rank_io"].median(),
        }

        # Per-template accuracy
        for tmpl in ["ABB", "BAB"]:
            subset = results_df[results_df["template_type"] == tmpl]
            if len(subset) > 0:
                stats[f"accuracy_{tmpl.lower()}"] = subset["is_correct"].mean()
                stats[f"mean_logit_diff_{tmpl.lower()}"] = subset["logit_diff"].mean()
            else:
                stats[f"accuracy_{tmpl.lower()}"] = None
                stats[f"mean_logit_diff_{tmpl.lower()}"] = None

        return stats
