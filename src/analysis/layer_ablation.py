"""
src/analysis/layer_ablation.py
================================
Layer-level mean ablation for the IOI circuit in GPT-2 Small.

Theory: Mean Ablation
----------------------
Ablation studies are the mechanistic interpretability equivalent of
surgical lesion studies in neuroscience. We "knock out" a model component
and measure how much the task performance drops.

**Mean ablation** replaces a component's output with the *average* of that
output computed over a large set of reference prompts. This is preferable
to zero-ablation because:
  - Zero-ablation can push activations to values never seen during training,
    creating unrealistic out-of-distribution inputs for downstream components.
  - Mean ablation keeps the magnitude roughly correct while removing the
    task-specific signal (the information that varies across prompts).

We ablate three types of components independently:

1. **Attention output** (`blocks.{l}.hook_attn_out`)
   The summed output of all 12 attention heads in layer l.
   Shape: [batch, seq, d_model].
   Ablating this removes all attention-mediated information flow in layer l.

2. **MLP output** (`blocks.{l}.hook_mlp_out`)
   The output of the two-layer MLP in layer l.
   Shape: [batch, seq, d_model].
   Ablating this removes all feature-to-feature transformations in layer l.

3. **Full layer** (ablate both attn_out and mlp_out simultaneously)
   The combined contribution of both components.

Hook Mechanism
--------------
TransformerLens's `run_with_hooks()` method accepts a list of
(hook_name, hook_function) tuples. Each hook function has signature:

    def hook_fn(value: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        ...
        return modified_value

The hook is called during the forward pass when the named activation is
computed. We replace `value` with our pre-computed mean and return it.

Algorithm
---------
1. Compute mean activations:
   a. Run all N reference prompts through the model.
   b. For each hook point, accumulate activations and divide by N.
   c. Result: `mean_cache[hook_name]` — shape [1, seq, d_model] or similar.

2. For each layer l and component type:
   a. Define a hook function that replaces the activation with mean_cache values.
   b. Run the model on the test prompts with this hook active.
   c. Compute logit_diff from the hooked model's output.
   d. Record: baseline_logit_diff - ablated_logit_diff = drop.

3. Normalise by baseline:
   normalised_drop = drop / |baseline_logit_diff|
   This gives a fraction: 1.0 = completely destroyed, 0 = no effect.

References
----------
Wang et al. (2022). Interpretability in the Wild.
Conmy et al. (2023). Towards Automated Circuit Discovery.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.data.ioi_dataset import IOIDataset
from src.evaluation.metrics import compute_logit_diff

logger = logging.getLogger(__name__)


class LayerAblationAnalyzer:
    """
    Measures the causal importance of each transformer layer via mean ablation.

    For each layer, independently ablates:
      - Attention output (attn_out)
      - MLP output (mlp_out)
      - Both simultaneously (full layer)

    Then measures the resulting logit difference on the IOI task to determine
    how much the layer contributes.

    Parameters
    ----------
    model : HookedTransformer
        Loaded GPT-2 Small model in eval mode.

    dataset : IOIDataset
        Generated IOI dataset.

    n_samples : int
        Number of prompts for mean computation AND evaluation.
        Larger = more stable estimates but slower.

    batch_size : int
        Prompts per forward pass.

    Examples
    --------
    >>> analyzer = LayerAblationAnalyzer(model, dataset, n_samples=200)
    >>> mean_cache = analyzer.compute_mean_cache()
    >>> results_df = analyzer.run_full_sweep(mean_cache)
    >>> print(results_df.sort_values("attn_logit_diff_drop", ascending=False))
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

        logger.info(
            f"[LayerAblationAnalyzer] Initialized: "
            f"n_layers={self.n_layers}, n_samples={self.n_samples}"
        )

    def _tokenize_batch(self, prompts: list[str]) -> tuple[torch.Tensor, list[int]]:
        """
        Tokenise a list of prompts into a padded batch tensor.

        Returns tokens tensor [batch, max_seq] and list of actual seq lengths.
        """
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
    def compute_mean_cache(self) -> dict[str, torch.Tensor]:
        """
        Compute mean activations across `n_samples` clean prompts.

        For each hook point (attn_out and mlp_out for every layer), we
        accumulate the sum of activations across all prompts, then divide
        by n_samples to get the mean.

        The mean cache is used as the ablation value: replacing a component's
        output with its mean removes task-specific signal while preserving
        the average magnitude.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping: hook_name → mean activation tensor.
            Tensors have shape [1, max_seq_len, d_model] where the batch
            dimension is 1 (broadcast-compatible with any batch size).

        Notes
        -----
        We use a simple online mean to avoid storing all activations:
            running_sum += activation_batch.sum(dim=0)
            mean = running_sum / n_prompts

        Attention: the max_seq_len in the mean cache corresponds to the
        longest sequence in the dataset. When ablating shorter sequences,
        we trim the mean cache to match the current sequence length.
        """
        logger.info(
            f"[LayerAblationAnalyzer.compute_mean_cache] "
            f"Computing mean cache over {self.n_samples} prompts…"
        )

        # Define which hook points to cache
        # attn_out: output of the full attention layer before residual add
        # mlp_out : output of the MLP layer before residual add
        hook_names = (
            [f"blocks.{l}.hook_attn_out" for l in range(self.n_layers)]
            + [f"blocks.{l}.hook_mlp_out" for l in range(self.n_layers)]
        )

        prompts = self.dataset.get_clean_prompts()[:self.n_samples]

        # Accumulate sum over all prompts
        # Key: hook_name, Value: running sum tensor [seq_len, d_model]
        running_sums: dict[str, torch.Tensor] = {}
        running_counts: dict[str, int] = {}

        for batch_start in tqdm(
            range(0, self.n_samples, self.batch_size),
            desc="Computing mean cache",
            unit="batch",
        ):
            batch_end = min(batch_start + self.batch_size, self.n_samples)
            batch_prompts = prompts[batch_start:batch_end]
            tokens, seq_lengths = self._tokenize_batch(batch_prompts)

            _, cache = self.model.run_with_cache(
                tokens, names_filter=hook_names
            )

            for hook_name in hook_names:
                if hook_name not in cache.cache_dict:
                    continue
                act = cache[hook_name]  # [batch, seq_len, d_model]

                # For each sequence in the batch, take the actual-length slice
                for i, seq_len in enumerate(seq_lengths):
                    act_i = act[i, :seq_len, :]  # [seq_len, d_model]

                    if hook_name not in running_sums:
                        running_sums[hook_name] = act_i.clone().float()
                        running_counts[hook_name] = 1
                    else:
                        # If lengths differ, pad shorter to match accumulated
                        stored_len = running_sums[hook_name].shape[0]
                        cur_len = act_i.shape[0]

                        if cur_len < stored_len:
                            pad = torch.zeros(
                                stored_len - cur_len, act_i.shape[-1],
                                device=self.device
                            )
                            act_i = torch.cat([act_i, pad], dim=0)
                        elif cur_len > stored_len:
                            pad = torch.zeros(
                                cur_len - stored_len, running_sums[hook_name].shape[-1],
                                device=self.device
                            )
                            running_sums[hook_name] = torch.cat(
                                [running_sums[hook_name], pad], dim=0
                            )
                        running_sums[hook_name] += act_i.float()
                        running_counts[hook_name] += 1

            del cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Compute means and add batch dimension for broadcasting
        mean_cache: dict[str, torch.Tensor] = {}
        for hook_name in hook_names:
            if hook_name in running_sums:
                mean = running_sums[hook_name] / running_counts[hook_name]
                # Shape: [1, seq_len, d_model] — broadcastable over batch dim
                mean_cache[hook_name] = mean.unsqueeze(0)

        logger.info(
            f"[LayerAblationAnalyzer.compute_mean_cache] ✓ Mean cache computed "
            f"for {len(mean_cache)} hook points."
        )
        return mean_cache

    @torch.no_grad()
    def _compute_baseline_logit_diff(self) -> float:
        """
        Compute the mean logit difference on clean prompts WITHOUT any ablation.

        This is the reference score. All ablated scores are compared to this
        baseline to measure the drop caused by each ablation.

        Returns
        -------
        float
            Mean logit_diff (IO - S) across all n_samples clean prompts.
        """
        prompts = self.dataset.get_clean_prompts()[:self.n_samples]
        io_ids = self.dataset.get_io_token_ids()[:self.n_samples]
        s_ids = self.dataset.get_s_token_ids()[:self.n_samples]

        total_ld = 0.0
        n_correct = 0

        for batch_start in range(0, self.n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.n_samples)
            batch_prompts = prompts[batch_start:batch_end]
            batch_io = io_ids[batch_start:batch_end]
            batch_s = s_ids[batch_start:batch_end]
            tokens, seq_lengths = self._tokenize_batch(batch_prompts)

            logits = self.model(tokens)

            for i, (io_id, s_id, seq_len) in enumerate(
                zip(batch_io, batch_s, seq_lengths)
            ):
                final_logits = logits[i, seq_len - 1, :]
                ld = compute_logit_diff(final_logits, io_id, s_id)
                total_ld += ld
                n_correct += 1 if ld > 0 else 0

        baseline = total_ld / self.n_samples
        accuracy = n_correct / self.n_samples
        logger.info(
            f"[LayerAblationAnalyzer] Baseline logit diff: {baseline:+.4f}, "
            f"accuracy: {accuracy:.1%}"
        )
        return baseline

    @torch.no_grad()
    def ablate_component(
        self,
        hook_names_to_ablate: list[str],
        mean_cache: dict[str, torch.Tensor],
    ) -> float:
        """
        Run the model with specified components mean-ablated.

        This is the core ablation function. It registers hooks that replace
        the named activations with their mean values during the forward pass.

        Parameters
        ----------
        hook_names_to_ablate : list[str]
            Hook names to ablate simultaneously. E.g.:
            ["blocks.5.hook_attn_out"] for attention in layer 5, or
            ["blocks.5.hook_attn_out", "blocks.5.hook_mlp_out"] for full layer 5.

        mean_cache : dict[str, torch.Tensor]
            Precomputed mean activations from `compute_mean_cache()`.
            Values shape: [1, max_seq, d_model] — batch dimension is 1 for broadcasting.

        Returns
        -------
        float
            Mean logit_diff across all test prompts with the ablation applied.

        Implementation Details
        ----------------------
        The hook function uses a closure to capture the specific mean value
        and sequence length for each prompt. This is necessary because:
          1. We need to trim the mean cache to the current sequence length
             (mean was computed on a padded average, but current seqs vary).
          2. The hook must be created fresh for each ablation to capture the
             correct mean cache reference.
        """
        prompts = self.dataset.get_clean_prompts()[:self.n_samples]
        io_ids = self.dataset.get_io_token_ids()[:self.n_samples]
        s_ids = self.dataset.get_s_token_ids()[:self.n_samples]

        total_ld = 0.0

        for batch_start in range(0, self.n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.n_samples)
            batch_prompts = prompts[batch_start:batch_end]
            batch_io = io_ids[batch_start:batch_end]
            batch_s = s_ids[batch_start:batch_end]
            tokens, seq_lengths = self._tokenize_batch(batch_prompts)
            batch_max_seq = tokens.shape[1]

            # ── Build hook functions (one per hook name to ablate) ────────
            fwd_hooks = []
            for hook_name in hook_names_to_ablate:
                if hook_name not in mean_cache:
                    logger.warning(f"Hook '{hook_name}' not in mean_cache; skipping.")
                    continue

                mean_act = mean_cache[hook_name]  # [1, mean_seq, d_model]
                mean_seq_len = mean_act.shape[1]

                def make_hook(mean_val: torch.Tensor, seq_len_val: int):
                    """
                    Factory function to create a closure with captured values.

                    Why use a factory? Python closures capture by reference, so
                    if we define the hook inline in a loop, all hooks would share
                    the same `mean_act` reference (pointing to the last iteration).
                    A factory function creates a new scope for each iteration.
                    """
                    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
                        # value shape: [batch, current_seq, d_model]
                        cur_seq = value.shape[1]

                        # Trim or pad mean to match current sequence length
                        if seq_len_val >= cur_seq:
                            mean_trimmed = mean_val[:, :cur_seq, :]
                        else:
                            # Pad with zeros if mean is shorter (rare edge case)
                            pad_len = cur_seq - seq_len_val
                            pad = torch.zeros(
                                1, pad_len, mean_val.shape[-1],
                                device=mean_val.device, dtype=mean_val.dtype
                            )
                            mean_trimmed = torch.cat([mean_val, pad], dim=1)

                        # Replace entire activation with mean (broadcast over batch)
                        return mean_trimmed.to(value.dtype).expand_as(value)
                    return hook_fn

                fwd_hooks.append(
                    (hook_name, make_hook(mean_act.to(self.device), mean_seq_len))
                )

            # ── Forward pass with hooks ────────────────────────────────────
            logits = self.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

            for i, (io_id, s_id, seq_len) in enumerate(
                zip(batch_io, batch_s, seq_lengths)
            ):
                final_logits = logits[i, seq_len - 1, :]
                ld = compute_logit_diff(final_logits, io_id, s_id)
                total_ld += ld

        return total_ld / self.n_samples

    @torch.no_grad()
    def compute_clean_cache(self) -> dict[str, torch.Tensor]:
        """
        Cache clean activations across all n_samples clean prompts for resample ablation.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping: hook_name → Tensor of shape [n_samples, max_seq_len, d_model].
        """
        logger.info(
            f"[LayerAblationAnalyzer.compute_clean_cache] "
            f"Caching clean activations over {self.n_samples} prompts…"
        )
        hook_names = (
            [f"blocks.{l}.hook_attn_out" for l in range(self.n_layers)]
            + [f"blocks.{l}.hook_mlp_out" for l in range(self.n_layers)]
        )
        prompts = self.dataset.get_clean_prompts()[:self.n_samples]

        clean_cache_lists: dict[str, list[torch.Tensor]] = {h: [] for h in hook_names}

        for batch_start in range(0, self.n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.n_samples)
            batch_prompts = prompts[batch_start:batch_end]
            tokens, seq_lengths = self._tokenize_batch(batch_prompts)

            _, cache = self.model.run_with_cache(tokens, names_filter=hook_names)

            for hook_name in hook_names:
                if hook_name in cache.cache_dict:
                    act = cache[hook_name]  # [batch, seq_len, d_model]
                    for i in range(len(batch_prompts)):
                        clean_cache_lists[hook_name].append(act[i].detach().clone())

            del cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        result_cache: dict[str, torch.Tensor] = {}
        for hook_name, tensor_list in clean_cache_lists.items():
            if not tensor_list:
                continue
            max_seq = max(t.shape[0] for t in tensor_list)
            padded_list = []
            for t in tensor_list:
                if t.shape[0] < max_seq:
                    pad = torch.zeros(
                        max_seq - t.shape[0], t.shape[1],
                        device=t.device, dtype=t.dtype
                    )
                    t_padded = torch.cat([t, pad], dim=0)
                else:
                    t_padded = t[:max_seq, :]
                padded_list.append(t_padded)
            result_cache[hook_name] = torch.stack(padded_list, dim=0)

        logger.info(
            f"[LayerAblationAnalyzer.compute_clean_cache] ✓ Clean cache stored "
            f"for {len(result_cache)} hook points."
        )
        return result_cache

    @torch.no_grad()
    def ablate_component_resample(
        self,
        hook_names_to_ablate: list[str],
        clean_cache: dict[str, torch.Tensor],
    ) -> float:
        """
        Run the model with specified components resample-ablated.

        Resample ablation replaces the activation of target component at prompt i
        with the clean activation of prompt (i + 1) % n_samples from clean_cache.
        This preserves valid activation geometry while breaking prompt-specific signal.
        """
        prompts = self.dataset.get_clean_prompts()[:self.n_samples]
        io_ids = self.dataset.get_io_token_ids()[:self.n_samples]
        s_ids = self.dataset.get_s_token_ids()[:self.n_samples]

        total_ld = 0.0

        for batch_start in range(0, self.n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.n_samples)
            batch_prompts = prompts[batch_start:batch_end]
            batch_io = io_ids[batch_start:batch_end]
            batch_s = s_ids[batch_start:batch_end]
            tokens, seq_lengths = self._tokenize_batch(batch_prompts)

            fwd_hooks = []
            for hook_name in hook_names_to_ablate:
                if hook_name not in clean_cache:
                    continue

                full_cache = clean_cache[hook_name]

                def make_resample_hook(b_start: int, b_end: int, cache_tensor: torch.Tensor):
                    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
                        cur_seq = value.shape[1]
                        indices = [(k + 1) % self.n_samples for k in range(b_start, b_end)]
                        resample_batch = cache_tensor[indices, :cur_seq, :]
                        return resample_batch.to(value.device, dtype=value.dtype)
                    return hook_fn

                fwd_hooks.append((hook_name, make_resample_hook(batch_start, batch_end, full_cache)))

            logits = self.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

            for i, (io_id, s_id, seq_len) in enumerate(zip(batch_io, batch_s, seq_lengths)):
                final_logits = logits[i, seq_len - 1, :]
                ld = compute_logit_diff(final_logits, io_id, s_id)
                total_ld += ld

        return total_ld / self.n_samples

    @torch.no_grad()
    def run_full_sweep(
        self,
        mean_cache: Optional[dict[str, torch.Tensor]] = None,
        ablation_mode: str = "mean",
        clean_cache: Optional[dict[str, torch.Tensor]] = None,
    ) -> pd.DataFrame:
        """
        Run layer ablation sweep over all 12 layers × 3 component types.

        Parameters
        ----------
        mean_cache : dict, optional
            Pre-computed mean cache for mean ablation.
        ablation_mode : str ("mean" or "resample")
            Mode of ablation to apply.
        clean_cache : dict, optional
            Pre-computed clean cache for resample ablation.
        """
        if ablation_mode == "mean" and mean_cache is None:
            mean_cache = self.compute_mean_cache()
        elif ablation_mode == "resample" and clean_cache is None:
            clean_cache = self.compute_clean_cache()

        baseline_ld = self._compute_baseline_logit_diff()

        rows = []
        total_ablations = self.n_layers * 3
        done = 0

        for layer in range(self.n_layers):
            for component in ["attn", "mlp", "full_layer"]:
                if component == "attn":
                    hooks = [f"blocks.{layer}.hook_attn_out"]
                elif component == "mlp":
                    hooks = [f"blocks.{layer}.hook_mlp_out"]
                else:  # full_layer
                    hooks = [
                        f"blocks.{layer}.hook_attn_out",
                        f"blocks.{layer}.hook_mlp_out",
                    ]

                if ablation_mode == "resample":
                    ablated_ld = self.ablate_component_resample(hooks, clean_cache)
                else:
                    ablated_ld = self.ablate_component(hooks, mean_cache)

                ld_drop = baseline_ld - ablated_ld
                ld_drop_norm = ld_drop / abs(baseline_ld) if baseline_ld != 0 else 0.0

                rows.append({
                    "layer": layer,
                    "component": component,
                    "ablation_mode": ablation_mode,
                    "ablated_ld": round(ablated_ld, 6),
                    "baseline_ld": round(baseline_ld, 6),
                    "ld_drop": round(ld_drop, 6),
                    "ld_drop_norm": round(ld_drop_norm, 6),
                    "is_critical": ld_drop_norm > 0.10,
                })

                done += 1
                logger.info(
                    f"[LayerAblation] [{ablation_mode}] [{done}/{total_ablations}] "
                    f"Layer {layer} {component}: "
                    f"LD={ablated_ld:+.4f} (drop={ld_drop:+.4f}, "
                    f"norm={ld_drop_norm:+.3f})"
                )

        df = pd.DataFrame(rows)
        return df

