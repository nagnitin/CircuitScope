"""
src/analysis/head_ablation.py
==============================
Per-head causal importance analysis for all 144 attention heads in GPT-2 Small.

Theory: Attention Head Ablation
---------------------------------
GPT-2 Small has 12 layers × 12 heads = 144 attention heads. Each head
performs a distinct computation on the residual stream. Circuit analysis
identifies which heads are causally important for the IOI task.

Wang et al. (2022) identified three functionally distinct head types:
  - **Name Mover Heads**: Write the IO name into the residual stream at the
    final position. Critical for the final prediction. Found at layers 9–11.
  - **S-Inhibition Heads**: Suppress the S name's influence at the IO position.
    Found at layers 7–8.
  - **Duplicate Token Heads** & **Induction Heads**: Identify which names
    appear multiple times in the prompt. Found at early layers 1–5.
  - **Backup Name Mover Heads**: Redundant heads that fire when primary
    Name Movers are ablated.

Head Ablation Method
---------------------
We use **mean ablation via hook_z** (the attention output before the O matrix):

    hook_z shape: [batch, seq_len, n_heads, d_head]

For each head h in layer l:
  1. Compute mean_z[l, h] = mean of z[:, :, h, :] over all reference prompts.
  2. During evaluation, replace z[:, :, h, :] = mean_z[l, h] for that head.
  3. The O-matrix (W_O) then multiplies the replaced z, so the head's
     contribution to the residual stream is replaced by a constant vector.

Why hook_z instead of hook_attn_out?
  - hook_attn_out captures the summed output of ALL heads in a layer.
  - hook_z captures the value-weighted output of EACH head independently.
  - Using hook_z allows per-head ablation without affecting other heads.

Restoration Score
-----------------
For each head (l, h), we compute:
    importance = (baseline_ld - ablated_ld) / |baseline_ld|

Positive importance = the head helps the IOI task.
Negative importance = the head HURTS the task (suppressor head).
Near-zero = the head is not causally important for IOI.

References
----------
Wang et al. (2022). "Interpretability in the Wild: a Circuit for
Indirect Object Identification in GPT-2 Small."
https://arxiv.org/abs/2202.00571
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.data.ioi_dataset import IOIDataset
from src.evaluation.metrics import compute_logit_diff

logger = logging.getLogger(__name__)


class HeadAblationAnalyzer:
    """
    Scores the causal importance of all 144 attention heads via mean ablation.

    For each (layer, head) pair, independently ablates that head's output
    and measures the resulting drop in logit difference on the IOI task.

    Parameters
    ----------
    model : HookedTransformer
        Loaded GPT-2 Small model in eval mode.

    dataset : IOIDataset
        Generated IOI dataset.

    n_samples : int
        Prompts for mean cache computation and evaluation.

    batch_size : int
        Prompts per forward pass.

    Examples
    --------
    >>> analyzer = HeadAblationAnalyzer(model, dataset, n_samples=200)
    >>> mean_z = analyzer.compute_mean_z()
    >>> results_df = analyzer.run_full_sweep(mean_z)
    >>> print(results_df.nlargest(10, "importance"))
    """

    def __init__(
        self,
        model: HookedTransformer,
        dataset: IOIDataset,
        n_samples: int = 200,
        batch_size: int = 16,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.n_samples = min(n_samples, len(dataset))
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.d_head = model.cfg.d_head

        logger.info(
            f"[HeadAblationAnalyzer] Initialized: "
            f"{self.n_layers}L × {self.n_heads}H = "
            f"{self.n_layers * self.n_heads} heads, "
            f"n_samples={self.n_samples}"
        )

    def _tokenize_batch(self, prompts: list[str]) -> tuple[torch.Tensor, list[int]]:
        """Tokenise a batch of prompts with right-padding."""
        token_lists = [
            self.model.to_tokens(p, prepend_bos=True)[0].tolist()
            for p in prompts
        ]
        seq_lengths = [len(t) for t in token_lists]
        max_len = max(seq_lengths)
        bos_id = self.model.tokenizer.bos_token_id
        padded = [t + [bos_id] * (max_len - len(t)) for t in token_lists]
        tokens = torch.tensor(padded, dtype=torch.long, device=self.device)
        return tokens, seq_lengths

    @torch.no_grad()
    def compute_mean_z(self) -> dict[str, torch.Tensor]:
        """
        Compute mean z-vectors for every attention head across reference prompts.

        The z-vector is the attention-weighted sum of value vectors for
        each head before the output projection:
            z = pattern @ V    (per head)
            attn_out = z @ W_O (concatenate all heads, project)

        By computing mean_z[l, h] = E[z[:, :, h, :]] over prompts, we get
        a "neutral" value for each head that represents average behaviour.
        Replacing z with this mean removes the prompt-specific signal.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping: f"blocks.{l}.attn.hook_z" → mean_z tensor.
            Each tensor shape: [1, max_seq, n_heads, d_head].

        Memory Note
        -----------
        hook_z tensors are [batch, seq, n_heads, d_head] = large.
        We use online accumulation to avoid storing all activations in RAM.
        """
        logger.info(
            f"[HeadAblationAnalyzer.compute_mean_z] "
            f"Computing mean z-cache over {self.n_samples} prompts…"
        )

        hook_names = [f"blocks.{l}.attn.hook_z" for l in range(self.n_layers)]
        prompts = self.dataset.get_clean_prompts()[:self.n_samples]

        # Online accumulation: hook_name → [sum_tensor, count]
        running: dict[str, list] = {}

        for batch_start in tqdm(
            range(0, self.n_samples, self.batch_size),
            desc="Computing mean_z",
            unit="batch",
        ):
            batch_end = min(batch_start + self.batch_size, self.n_samples)
            batch_prompts = prompts[batch_start:batch_end]
            tokens, seq_lengths = self._tokenize_batch(batch_prompts)

            _, cache = self.model.run_with_cache(tokens, names_filter=hook_names)

            for hook_name in hook_names:
                if hook_name not in cache.cache_dict:
                    continue
                # z shape: [batch, seq, n_heads, d_head]
                z = cache[hook_name].float()

                for i, seq_len in enumerate(seq_lengths):
                    z_i = z[i, :seq_len, :, :]  # [seq, n_heads, d_head]

                    if hook_name not in running:
                        running[hook_name] = [z_i.clone(), 1]
                    else:
                        stored_len = running[hook_name][0].shape[0]
                        cur_len = z_i.shape[0]
                        if cur_len < stored_len:
                            pad_shape = (
                                stored_len - cur_len,
                                self.n_heads, self.d_head
                            )
                            z_i = torch.cat([z_i, torch.zeros(pad_shape, device=z_i.device)], dim=0)
                        elif cur_len > stored_len:
                            pad_shape = (
                                cur_len - stored_len,
                                self.n_heads, self.d_head
                            )
                            running[hook_name][0] = torch.cat(
                                [running[hook_name][0],
                                 torch.zeros(pad_shape, device=running[hook_name][0].device)],
                                dim=0
                            )
                        running[hook_name][0] += z_i
                        running[hook_name][1] += 1

            del cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Compute means, add batch dim
        mean_z: dict[str, torch.Tensor] = {
            k: (v[0] / v[1]).unsqueeze(0)  # [1, seq, n_heads, d_head]
            for k, v in running.items()
        }
        logger.info(
            f"[HeadAblationAnalyzer.compute_mean_z] ✓ Mean_z computed "
            f"for {len(mean_z)} layers."
        )
        return mean_z

    @torch.no_grad()
    def _compute_baseline_logit_diff(self) -> tuple[float, float]:
        """
        Compute baseline mean logit_diff and accuracy on clean prompts.

        Returns
        -------
        tuple of (mean_logit_diff, accuracy)
        """
        prompts = self.dataset.get_clean_prompts()[:self.n_samples]
        io_ids = self.dataset.get_io_token_ids()[:self.n_samples]
        s_ids = self.dataset.get_s_token_ids()[:self.n_samples]

        total_ld = 0.0
        n_correct = 0

        for batch_start in range(0, self.n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.n_samples)
            tokens, seq_lengths = self._tokenize_batch(
                prompts[batch_start:batch_end]
            )
            logits = self.model(tokens)
            for i, (io_id, s_id, seq_len) in enumerate(
                zip(io_ids[batch_start:batch_end],
                    s_ids[batch_start:batch_end],
                    seq_lengths)
            ):
                ld = compute_logit_diff(logits[i, seq_len - 1, :], io_id, s_id)
                total_ld += ld
                n_correct += 1 if ld > 0 else 0

        mean_ld = total_ld / self.n_samples
        accuracy = n_correct / self.n_samples
        return mean_ld, accuracy

    @torch.no_grad()
    def ablate_single_head(
        self,
        layer: int,
        head: int,
        mean_z: dict[str, torch.Tensor],
    ) -> tuple[float, float]:
        """
        Ablate a single attention head and return resulting logit_diff and accuracy.

        Replaces z[:, :, head, :] with mean_z[layer][:, :, head, :] during
        the forward pass, leaving all other heads untouched.

        Parameters
        ----------
        layer : int
            Layer index (0–11).

        head : int
            Head index within the layer (0–11).

        mean_z : dict[str, torch.Tensor]
            Pre-computed mean z-vectors from `compute_mean_z()`.

        Returns
        -------
        tuple of (mean_logit_diff, accuracy)
            The logit_diff and accuracy with this head ablated.
        """
        hook_name = f"blocks.{layer}.attn.hook_z"
        mean_z_layer = mean_z[hook_name]  # [1, seq, n_heads, d_head]

        def ablate_head_hook(z: torch.Tensor, hook) -> torch.Tensor:
            """
            Replace head `head`'s z-vector with its mean value.

            z shape: [batch, seq, n_heads, d_head]

            We only modify the slice [:, :, head, :] for head `head`.
            All other heads ([:, :, h, :] for h ≠ head) remain unchanged.

            The mean is trimmed or padded to match the current sequence length
            since sequences in each batch may be shorter than the mean cache.
            """
            cur_seq = z.shape[1]
            mean_seq = mean_z_layer.shape[1]

            # Trim/pad mean to current seq length
            if mean_seq >= cur_seq:
                mean_trimmed = mean_z_layer[:, :cur_seq, :, :]
            else:
                pad_len = cur_seq - mean_seq
                pad = torch.zeros(
                    1, pad_len, self.n_heads, self.d_head,
                    device=mean_z_layer.device, dtype=mean_z_layer.dtype
                )
                mean_trimmed = torch.cat([mean_z_layer, pad], dim=1)

            # Replace only this head's z-vector
            z_modified = z.clone()
            z_modified[:, :, head, :] = (
                mean_trimmed[:, :, head, :].to(z.dtype).expand(z.shape[0], -1, -1)
            )
            return z_modified

        prompts = self.dataset.get_clean_prompts()[:self.n_samples]
        io_ids = self.dataset.get_io_token_ids()[:self.n_samples]
        s_ids = self.dataset.get_s_token_ids()[:self.n_samples]

        total_ld = 0.0
        n_correct = 0

        for batch_start in range(0, self.n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.n_samples)
            tokens, seq_lengths = self._tokenize_batch(
                prompts[batch_start:batch_end]
            )

            logits = self.model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, ablate_head_hook)],
            )

            for i, (io_id, s_id, seq_len) in enumerate(
                zip(io_ids[batch_start:batch_end],
                    s_ids[batch_start:batch_end],
                    seq_lengths)
            ):
                ld = compute_logit_diff(logits[i, seq_len - 1, :], io_id, s_id)
                total_ld += ld
                n_correct += 1 if ld > 0 else 0

        return total_ld / self.n_samples, n_correct / self.n_samples

    @torch.no_grad()
    def run_full_sweep(
        self,
        mean_z: dict[str, torch.Tensor] | None = None,
    ) -> pd.DataFrame:
        """
        Run head ablation sweep over all 144 attention heads.

        For each of the 144 (layer, head) pairs:
          1. Ablate only that head (leave all others intact).
          2. Measure logit_diff and accuracy under the ablation.
          3. Compute importance = (baseline - ablated) / |baseline|.
          4. Classify by head type based on importance sign and magnitude.

        Parameters
        ----------
        mean_z : dict, optional
            Pre-computed mean z. If None, computed automatically.

        Returns
        -------
        pd.DataFrame
            144 rows, sorted by importance descending. Columns:
            - layer, head         : (int) position in the model
            - head_label          : str e.g. "L9H6"
            - ablated_ld          : float (logit diff after ablation)
            - ablated_acc         : float (accuracy after ablation)
            - baseline_ld         : float
            - baseline_acc        : float
            - ld_drop             : float (positive = head helps task)
            - importance          : float (ld_drop / |baseline|)
            - rank                : int (1 = most important)
            - head_type           : str classification

        Examples
        --------
        >>> results = analyzer.run_full_sweep()
        >>> print(results.nlargest(10, "importance")[["head_label", "importance", "head_type"]])
        """
        if mean_z is None:
            mean_z = self.compute_mean_z()

        baseline_ld, baseline_acc = self._compute_baseline_logit_diff()
        logger.info(
            f"[HeadAblationAnalyzer] Baseline: LD={baseline_ld:+.4f}, "
            f"acc={baseline_acc:.1%}"
        )

        rows = []
        total_heads = self.n_layers * self.n_heads

        with tqdm(total=total_heads, desc="Head ablation sweep", unit="head") as pbar:
            for layer in range(self.n_layers):
                for head in range(self.n_heads):
                    ablated_ld, ablated_acc = self.ablate_single_head(
                        layer, head, mean_z
                    )
                    ld_drop = baseline_ld - ablated_ld
                    importance = ld_drop / abs(baseline_ld) if baseline_ld != 0 else 0.0

                    rows.append({
                        "layer": layer,
                        "head": head,
                        "head_label": f"L{layer}H{head}",
                        "ablated_ld": round(ablated_ld, 6),
                        "ablated_acc": round(ablated_acc, 6),
                        "baseline_ld": round(baseline_ld, 6),
                        "baseline_acc": round(baseline_acc, 6),
                        "ld_drop": round(ld_drop, 6),
                        "importance": round(importance, 6),
                    })
                    pbar.set_postfix({
                        "L": layer, "H": head,
                        "imp": f"{importance:+.3f}"
                    })
                    pbar.update(1)

        df = pd.DataFrame(rows)

        # ── Assign ranks (1 = most important positive) ────────────────────
        df["rank"] = df["importance"].rank(ascending=False).astype(int)

        # ── Classify head types based on importance thresholds ────────────
        # Thresholds are empirically based on Wang et al. (2022):
        #   importance > 0.15  → Name Mover (very helpful)
        #   0.05 < imp ≤ 0.15  → Helper head (moderately helpful)
        #   -0.05 ≤ imp ≤ 0.05 → Neutral (not important for IOI)
        #   imp < -0.05        → Suppressor (negative token mover)
        def classify_head(imp: float) -> str:
            if imp > 0.15:
                return "Name Mover"
            elif imp > 0.05:
                return "Helper"
            elif imp < -0.15:
                return "Strong Suppressor"
            elif imp < -0.05:
                return "Suppressor"
            else:
                return "Neutral"

        df["head_type"] = df["importance"].apply(classify_head)
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        # ── Summary logging ───────────────────────────────────────────────
        type_counts = df["head_type"].value_counts().to_dict()
        top5 = df.head(5)[["head_label", "importance", "head_type"]].to_string()
        logger.info(
            f"[HeadAblationAnalyzer.run_full_sweep] ✓ Sweep complete.\n"
            f"  Head type distribution: {type_counts}\n"
            f"  Top 5 most important heads:\n{top5}"
        )

        return df

    def pivot_importance_matrix(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape results into a 12×12 matrix for heatmap visualization.

        Rows = layers (0–11), Columns = heads (0–11).
        Values = importance score.

        Parameters
        ----------
        results_df : pd.DataFrame
            Output of `run_full_sweep()`.

        Returns
        -------
        pd.DataFrame
            Shape [12, 12] with index=layer, columns=head.
            Values are importance scores.

        Examples
        --------
        >>> matrix = analyzer.pivot_importance_matrix(results_df)
        >>> print(matrix.shape)   # (12, 12)
        """
        pivot = results_df.pivot(
            index="layer", columns="head", values="importance"
        )
        pivot.index.name = "Layer"
        pivot.columns.name = "Head"
        return pivot
