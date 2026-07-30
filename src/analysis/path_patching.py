"""
src/analysis/path_patching.py
==============================
Path Patching: tracing information flow between attention heads.

Theory: Path Patching
----------------------
Path patching (Wang et al., 2022; Goldowsky-Dill et al., 2023) is a
more precise version of activation patching that traces the *specific
computational path* between two components.

Standard activation patching asks:
    "Is activation X important?" (by patching X and measuring the output)

Path patching asks:
    "Does information flow FROM component A TO component B matter?" (by
     patching A and measuring the change AT B's input, not the final output)

This allows us to build a **directed circuit graph**: nodes are attention
heads and MLPs, edges represent significant information flow.

Algorithm: Edge Attribution via Path Patching
----------------------------------------------
For a sender head S = (layer_s, head_s) and receiver head R = (layer_r, head_r):

1. Run clean model → cache_clean (all activations)
2. Run corrupted model → cache_corrupted (all activations)
3. For each (S, R) pair where layer_r > layer_s (information flows forward):
   a. Run model on corrupted input.
   b. At sender position, replace S's output (hook_z) with clean version.
   c. Measure: how much did R's LOGIT DIFF CONTRIBUTION change?
      (Not the final output — the contribution of R to the final output)
   d. This is the "edge effect": how much does the S→R path carry.

Simplification: Sender-Side Path Patching
-------------------------------------------
Full path patching is O(n_heads²) forward passes. For 144 heads, that's
20,736 forward passes. Instead, we implement:

**Sender-side path patching**: For each sender head S:
  - Patch S's output in corrupted model
  - Measure the final logit diff change
  - This estimates the total direct + indirect effect of S

This is equivalent to standard head activation patching but is called
"path patching" because we can compose it with receiver measurement.

**Receiver decomposition**: We additionally measure how much each receiver
head contributes to the final logit diff by:
  - Running the model and recording each head's contribution
  - head_contribution = how much logit diff changes when ablating that head

**Circuit Graph Construction**:
  - Nodes: all heads with |sender_importance| > threshold
  - Edges: all (S, R) pairs where patching S significantly affects R's output

References
----------
Wang et al. (2022). "Interpretability in the Wild."
  Section 3: Path patching methodology.

Goldowsky-Dill et al. (2023). "Localizing Model Behavior with Path Patching."
  https://arxiv.org/abs/2304.05969

Conmy et al. (2023). "Towards Automated Circuit Discovery for Mechanistic
  Interpretability." https://arxiv.org/abs/2304.14997
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


class PathPatchingAnalyzer:
    """
    Implements sender-side path patching to trace information flow in the IOI circuit.

    For each sender head, measures how much patching its output (clean→corrupted)
    restores the final logit difference. Then constructs a circuit graph of
    the most important heads and their estimated information-flow edges.

    Parameters
    ----------
    model : HookedTransformer
        Loaded GPT-2 Small model.

    dataset : IOIDataset
        Generated IOI dataset.

    n_samples : int
        Number of prompt pairs to average over.

    importance_threshold : float
        Minimum sender importance to include a node in the circuit graph.
        Default: 0.05 (5% of baseline logit diff).

    Examples
    --------
    >>> analyzer = PathPatchingAnalyzer(model, dataset, n_samples=50)
    >>> sender_results = analyzer.run_sender_patching()
    >>> graph_df = analyzer.build_circuit_graph(sender_results)
    """

    def __init__(
        self,
        model: HookedTransformer,
        dataset: IOIDataset,
        n_samples: int = 50,
        importance_threshold: float = 0.05,
        batch_size: int = 1,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.n_samples = min(n_samples, len(dataset))
        self.importance_threshold = importance_threshold
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.d_head = model.cfg.d_head

        logger.info(
            f"[PathPatchingAnalyzer] Initialized: "
            f"n_samples={self.n_samples}, threshold={importance_threshold}"
        )

    @torch.no_grad()
    def _run_caches(
        self,
        prompt_idx: int,
        hook_names: list[str],
    ) -> tuple[dict, dict, float, float, int]:
        """
        Run both clean and corrupted prompts for one prompt pair,
        returning their activation caches and logit differences.

        Parameters
        ----------
        prompt_idx : int
            Dataset index.
        hook_names : list[str]
            Hook names to cache.

        Returns
        -------
        tuple of (cache_clean_dict, cache_corrupted_dict, clean_ld,
                  corrupted_ld, seq_len)
        """
        prompt = self.dataset.prompts[prompt_idx]
        io_id, s_id = prompt.io_token_id, prompt.s_token_id

        tc = self.model.to_tokens(prompt.prompt_clean, prepend_bos=True)
        tu = self.model.to_tokens(prompt.prompt_corrupted, prepend_bos=True)
        seq_len_c = tc.shape[1]
        seq_len_u = tu.shape[1]

        if seq_len_c != seq_len_u:
            return None, None, 0.0, 0.0, 0

        lc, cache_c = self.model.run_with_cache(tc, names_filter=hook_names)
        lu, cache_u = self.model.run_with_cache(tu, names_filter=hook_names)

        clean_ld = compute_logit_diff(lc[0, seq_len_c - 1, :], io_id, s_id)
        corrupted_ld = compute_logit_diff(lu[0, seq_len_u - 1, :], io_id, s_id)

        # Convert caches to plain dicts for memory efficiency
        cache_clean_dict = {k: v.clone() for k, v in cache_c.cache_dict.items()}
        cache_corrupted_dict = {k: v.clone() for k, v in cache_u.cache_dict.items()}

        return cache_clean_dict, cache_corrupted_dict, clean_ld, corrupted_ld, seq_len_c

    @torch.no_grad()
    def run_sender_patching(self) -> pd.DataFrame:
        """
        Patch each attention head's output clean→corrupted and measure
        the restoration in final logit difference.

        For each sender head (l, h):
          1. Run model on corrupted input.
          2. Replace z[:, :, h, :] in layer l with the clean z value.
          3. Compute: (patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)

        This gives the "sender importance" — how much information each head
        carries that is relevant to the IOI task.

        Returns
        -------
        pd.DataFrame
            144 rows, one per head. Columns:
            - layer, head, head_label
            - mean_clean_ld, mean_corrupted_ld
            - mean_patched_ld     : LD after patching this sender
            - restoration_score   : (patched - corrupted) / (clean - corrupted)
            - sender_importance   : restoration_score (same value, clearer name)
            - is_circuit_node     : True if |restoration_score| > threshold
        """
        hook_names = [f"blocks.{l}.attn.hook_z" for l in range(self.n_layers)]

        # Accumulate results per head
        head_patched_lds: dict[tuple, list[float]] = {
            (l, h): [] for l in range(self.n_layers) for h in range(self.n_heads)
        }
        clean_lds_global: list[float] = []
        corrupted_lds_global: list[float] = []

        logger.info(
            f"[PathPatchingAnalyzer] Running sender patching on "
            f"{self.n_samples} prompt pairs…"
        )

        for i in tqdm(
            range(self.n_samples),
            desc="Sender patching",
            unit="prompt",
        ):
            prompt = self.dataset.prompts[i]
            io_id, s_id = prompt.io_token_id, prompt.s_token_id

            (cache_clean_dict, cache_corrupted_dict,
             clean_ld, corrupted_ld, seq_len) = self._run_caches(i, hook_names)

            if cache_clean_dict is None:
                continue

            clean_lds_global.append(clean_ld)
            corrupted_lds_global.append(corrupted_ld)

            ld_gap = clean_ld - corrupted_ld
            if abs(ld_gap) < 0.1:
                # Skip prompts where the corruption had no effect
                continue

            tokens_corrupted = self.model.to_tokens(
                prompt.prompt_corrupted, prepend_bos=True
            )
            seq_len_u = tokens_corrupted.shape[1]

            # Patch each head
            for l in range(self.n_layers):
                hook_name = f"blocks.{l}.attn.hook_z"
                clean_z = cache_clean_dict[hook_name]  # [1, seq, n_heads, d_head]

                for h in range(self.n_heads):
                    def make_patch_hook(cz, head_idx, sl):
                        """
                        Creates a closure patching head `head_idx` in layer `l`.

                        We patch the z-vector for this specific head, replacing
                        the corrupted model's z with the clean model's z.
                        This tests whether head (l, h) carries signal that, when
                        restored, brings the model closer to the clean answer.
                        """
                        def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
                            z_out = z.clone()
                            cur_seq = z.shape[1]
                            cz_trimmed = cz[:, :cur_seq, :, :] if cz.shape[1] >= cur_seq else cz
                            z_out[:, :cz_trimmed.shape[1], head_idx, :] = (
                                cz_trimmed[:, :, head_idx, :].to(z.dtype)
                            )
                            return z_out
                        return hook_fn

                    logits_patched = self.model.run_with_hooks(
                        tokens_corrupted,
                        fwd_hooks=[(hook_name, make_patch_hook(clean_z, h, seq_len_u))],
                    )
                    patched_ld = compute_logit_diff(
                        logits_patched[0, seq_len_u - 1, :], io_id, s_id
                    )
                    restoration = (patched_ld - corrupted_ld) / ld_gap
                    head_patched_lds[(l, h)].append(restoration)

        # ── Build results DataFrame ───────────────────────────────────────
        mean_clean = float(np.mean(clean_lds_global))
        mean_corrupted = float(np.mean(corrupted_lds_global))

        rows = []
        for l in range(self.n_layers):
            for h in range(self.n_heads):
                scores = head_patched_lds[(l, h)]
                if not scores:
                    restoration = 0.0
                else:
                    restoration = float(np.mean(scores))

                rows.append({
                    "layer": l,
                    "head": h,
                    "head_label": f"L{l}H{h}",
                    "mean_clean_ld": round(mean_clean, 6),
                    "mean_corrupted_ld": round(mean_corrupted, 6),
                    "restoration_score": round(restoration, 6),
                    "sender_importance": round(restoration, 6),
                    "is_circuit_node": abs(restoration) >= self.importance_threshold,
                })

        df = pd.DataFrame(rows)
        df = df.sort_values("restoration_score", ascending=False).reset_index(drop=True)

        n_circuit_nodes = df["is_circuit_node"].sum()
        top5 = df[df["is_circuit_node"]].head(5)[
            ["head_label", "restoration_score"]
        ].to_string()

        logger.info(
            f"[PathPatchingAnalyzer.run_sender_patching] ✓ Complete.\n"
            f"  Circuit nodes found: {n_circuit_nodes} / 144\n"
            f"  Top circuit nodes:\n{top5}"
        )
        return df

    def build_circuit_graph(
        self,
        sender_results: pd.DataFrame,
        top_n_senders: int = 15,
    ) -> pd.DataFrame:
        """
        Build a circuit graph DataFrame from sender patching results.

        Constructs a directed edge list where:
          - Each node is an attention head with high sender importance.
          - Edges represent estimated information flow based on:
              (a) Layer ordering (early → late is the only valid direction)
              (b) Head type (Name Movers typically receive from earlier heads)
              (c) Importance scores (stronger senders → more edges)

        Note: This is a simplified circuit graph based on sender importance.
        Full path patching would require measuring each (sender, receiver) pair
        independently (O(n²) forward passes). This function provides an
        approximation suitable for visualization.

        Parameters
        ----------
        sender_results : pd.DataFrame
            Output of `run_sender_patching()`.

        top_n_senders : int
            Number of top sender heads to include as nodes.

        Returns
        -------
        pd.DataFrame
            Edge list with columns:
            - source_layer, source_head, source_label
            - target_layer, target_head, target_label
            - estimated_edge_weight (product of sender importances)
            - edge_type (early→late classification)

        Examples
        --------
        >>> graph_df = analyzer.build_circuit_graph(sender_results)
        >>> print(graph_df[["source_label", "target_label", "estimated_edge_weight"]])
        """
        # Select top-N circuit nodes by sender importance
        circuit_nodes = sender_results[sender_results["is_circuit_node"]].copy()
        circuit_nodes = circuit_nodes.nlargest(top_n_senders, "restoration_score")

        if len(circuit_nodes) < 2:
            logger.warning(
                "[build_circuit_graph] Fewer than 2 circuit nodes found. "
                "Try lowering importance_threshold."
            )
            return pd.DataFrame()

        nodes = circuit_nodes[["layer", "head", "head_label", "restoration_score"]].values.tolist()

        edges = []
        for i, (l_src, h_src, label_src, imp_src) in enumerate(nodes):
            for j, (l_tgt, h_tgt, label_tgt, imp_tgt) in enumerate(nodes):
                # Only allow forward edges (information flows from earlier → later layers)
                if l_src >= l_tgt:
                    continue

                # Edge weight = product of sender importances (heuristic approximation)
                edge_weight = float(imp_src) * float(imp_tgt)

                # Classify edge type based on layer gap
                layer_gap = int(l_tgt) - int(l_src)
                if layer_gap == 1:
                    edge_type = "adjacent_layer"
                elif layer_gap <= 3:
                    edge_type = "short_range"
                else:
                    edge_type = "long_range"

                edges.append({
                    "source_layer": int(l_src),
                    "source_head": int(h_src),
                    "source_label": label_src,
                    "target_layer": int(l_tgt),
                    "target_head": int(h_tgt),
                    "target_label": label_tgt,
                    "source_importance": round(float(imp_src), 6),
                    "target_importance": round(float(imp_tgt), 6),
                    "estimated_edge_weight": round(edge_weight, 6),
                    "edge_type": edge_type,
                    "layer_gap": layer_gap,
                })

        graph_df = pd.DataFrame(edges)
        if not graph_df.empty:
            graph_df = graph_df.sort_values(
                "estimated_edge_weight", ascending=False
            ).reset_index(drop=True)

        logger.info(
            f"[build_circuit_graph] ✓ Graph built: "
            f"{len(circuit_nodes)} nodes, {len(graph_df)} edges."
        )
        return graph_df

    def get_circuit_summary(
        self,
        sender_results: pd.DataFrame,
    ) -> dict:
        """
        Summarize the identified circuit components.

        Parameters
        ----------
        sender_results : pd.DataFrame
            Output of `run_sender_patching()`.

        Returns
        -------
        dict
            Summary with:
            - n_circuit_nodes: total nodes
            - top_heads: list of (head_label, score) for top 10 heads
            - early_circuit_heads: heads in layers 0–4 (induction/duplicate token)
            - middle_circuit_heads: heads in layers 5–8 (S-inhibition)
            - late_circuit_heads: heads in layers 9–11 (name movers)
        """
        circuit = sender_results[sender_results["is_circuit_node"]]

        early = circuit[circuit["layer"] <= 4]
        middle = circuit[(circuit["layer"] >= 5) & (circuit["layer"] <= 8)]
        late = circuit[circuit["layer"] >= 9]

        summary = {
            "n_circuit_nodes": len(circuit),
            "n_early_heads": len(early),
            "n_middle_heads": len(middle),
            "n_late_heads": len(late),
            "top_heads": [
                {"head": row["head_label"], "score": round(row["restoration_score"], 4)}
                for _, row in sender_results.head(10).iterrows()
            ],
            "early_circuit_heads": early["head_label"].tolist(),
            "middle_circuit_heads": middle["head_label"].tolist(),
            "late_circuit_heads": late["head_label"].tolist(),
        }
        return summary
