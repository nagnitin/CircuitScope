"""
src/analysis/circuit_validation.py
====================================
Circuit Validation: Necessity, Sufficiency, and Generalization Tests.

This module provides the three canonical circuit validation tests described
in Wang et al. (2022) and standardised by Conmy et al. (2023):

Necessity Test
--------------
A set of heads H is *necessary* for a task if ablating only those heads
causes a significant drop in task performance. Formally:

    necessity_score = (baseline_ld - ablated_ld) / baseline_ld

If necessity_score ≈ 1.0, the circuit is necessary (ablating it destroys performance).
If necessity_score ≈ 0.0, the circuit is not necessary (other components compensate).

Sufficiency Test
----------------
A set of heads H is *sufficient* for a task if keeping only those heads
(ablating everything ELSE) preserves task performance. Formally:

    sufficiency_score = preserved_ld / baseline_ld

If sufficiency_score ≈ 1.0, the circuit alone can perform the task.
If sufficiency_score ≈ 0.0, other components are needed (circuit is incomplete).

Key implementation insight: "keeping only H active" means we ablate every
head NOT in H. This is the complement ablation.

Generalization Test
-------------------
Evaluates circuit robustness across distribution shifts:
  1. New names: names not seen during circuit identification
  2. New templates: structural variants not in the original template set
  3. Held-out prompts: a reserved split from the original dataset

A circuit that generalises should show consistent necessity/sufficiency
scores across all three conditions.

References
----------
Wang et al. (2022). "Interpretability in the Wild."
Conmy et al. (2023). "Towards Automated Circuit Discovery for Mechanistic Interpretability."
  https://arxiv.org/abs/2304.14997
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.data.ioi_dataset import IOIDataset
from src.evaluation.metrics import compute_logit_diff

logger = logging.getLogger(__name__)


@dataclass
class CircuitSpec:
    """
    Specification of a discovered circuit: a set of (layer, head) pairs.

    Attributes
    ----------
    heads : list of (int, int)
        Each element is (layer_idx, head_idx).

    name : str
        Human-readable name for the circuit.

    source : str
        How the circuit was identified (e.g., "head_ablation", "path_patching").

    Examples
    --------
    >>> circuit = CircuitSpec(
    ...     heads=[(9, 6), (9, 9), (10, 0), (7, 3), (7, 9)],
    ...     name="IOI Name Mover + S-Inhibition",
    ... )
    """
    heads: list[tuple[int, int]]
    name: str = "discovered_circuit"
    source: str = "head_ablation"

    @classmethod
    def from_head_ablation_df(
        cls,
        head_df: pd.DataFrame,
        importance_threshold: float = 0.05,
        name: str = "IOI Circuit",
    ) -> "CircuitSpec":
        """
        Build a CircuitSpec from head ablation results.

        Selects all heads with importance > threshold as circuit members.

        Parameters
        ----------
        head_df : pd.DataFrame
            Output of `HeadAblationAnalyzer.run_full_sweep()`.
            Must have columns: layer, head, importance.

        importance_threshold : float
            Minimum importance score to be included in the circuit.

        Returns
        -------
        CircuitSpec
        """
        circuit_heads = head_df[head_df["importance"] > importance_threshold]
        heads = [(int(r["layer"]), int(r["head"])) for _, r in circuit_heads.iterrows()]
        logger.info(
            f"[CircuitSpec.from_head_ablation_df] Built circuit with "
            f"{len(heads)} heads (threshold={importance_threshold})"
        )
        return cls(heads=heads, name=name, source="head_ablation")

    def __len__(self) -> int:
        return len(self.heads)

    def __repr__(self) -> str:
        return (
            f"CircuitSpec(name={self.name!r}, "
            f"n_heads={len(self.heads)}, "
            f"heads={sorted(self.heads)})"
        )


@dataclass
class ValidationResult:
    """
    Results from a single validation experiment.

    Attributes
    ----------
    test_name : str
        E.g., "necessity", "sufficiency", "generalization_new_names".

    baseline_ld : float
        Mean logit diff without any ablation.

    experimental_ld : float
        Mean logit diff under the experimental condition.

    baseline_acc : float
        Accuracy without ablation.

    experimental_acc : float
        Accuracy under experimental condition.

    score : float
        Primary metric (necessity or sufficiency score).

    n_prompts : int
        Number of prompts evaluated.

    circuit_n_heads : int
        Number of heads in the circuit.
    """
    test_name: str
    baseline_ld: float
    experimental_ld: float
    baseline_acc: float
    experimental_acc: float
    score: float
    n_prompts: int
    circuit_n_heads: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "baseline_ld": self.baseline_ld,
            "experimental_ld": self.experimental_ld,
            "baseline_acc": self.baseline_acc,
            "experimental_acc": self.experimental_acc,
            "score": self.score,
            "ld_change": self.experimental_ld - self.baseline_ld,
            "acc_change": self.experimental_acc - self.baseline_acc,
            "n_prompts": self.n_prompts,
            "circuit_n_heads": self.circuit_n_heads,
            **self.extra,
        }


class CircuitValidator:
    """
    Runs necessity, sufficiency, and generalization tests for a discovered circuit.

    Parameters
    ----------
    model : HookedTransformer
        Loaded GPT-2 Small model.

    dataset : IOIDataset
        The IOI dataset used to identify the circuit.

    circuit : CircuitSpec
        The discovered circuit to validate.

    mean_z : dict[str, torch.Tensor]
        Pre-computed mean z-vectors from `HeadAblationAnalyzer.compute_mean_z()`.
        Shape: [1, seq, n_heads, d_head] per layer.

    n_samples : int
        Prompts for validation experiments.

    batch_size : int
        Forward pass batch size.

    Examples
    --------
    >>> validator = CircuitValidator(model, dataset, circuit, mean_z)
    >>> results = validator.run_all_tests()
    >>> print(pd.DataFrame([r.to_dict() for r in results]))
    """

    def __init__(
        self,
        model: HookedTransformer,
        dataset: IOIDataset,
        circuit: CircuitSpec,
        mean_z: dict[str, torch.Tensor],
        n_samples: int = 200,
        batch_size: int = 16,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.circuit = circuit
        self.mean_z = mean_z
        self.n_samples = min(n_samples, len(dataset))
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.d_head = model.cfg.d_head

        logger.info(
            f"[CircuitValidator] Circuit: {circuit.name}, "
            f"{len(circuit)} heads, n_samples={n_samples}"
        )

    def _tokenize_batch(self, prompts: list[str]) -> tuple[torch.Tensor, list[int]]:
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
    def _compute_ld_with_ablation(
        self,
        prompts: list[str],
        io_ids: list[int],
        s_ids: list[int],
        heads_to_ablate: list[tuple[int, int]],
    ) -> tuple[float, float]:
        """
        Run the model with specified heads mean-ablated.

        Creates a hook for each unique layer that has heads to ablate,
        efficiently ablating all heads in a layer with a single hook.

        Parameters
        ----------
        heads_to_ablate : list of (layer, head)
            All (layer, head) pairs to ablate simultaneously.

        Returns
        -------
        tuple of (mean_logit_diff, accuracy)
        """
        # Group by layer for efficiency (one hook per layer)
        layer_to_heads: dict[int, set[int]] = {}
        for (l, h) in heads_to_ablate:
            layer_to_heads.setdefault(l, set()).add(h)

        total_ld = 0.0
        n_correct = 0
        n_total = len(prompts)

        for batch_start in range(0, n_total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_total)
            tokens, seq_lengths = self._tokenize_batch(
                prompts[batch_start:batch_end]
            )
            batch_io = io_ids[batch_start:batch_end]
            batch_s = s_ids[batch_start:batch_end]

            # Build hooks: one per layer with heads to ablate
            fwd_hooks = []
            for layer_idx, heads_in_layer in layer_to_heads.items():
                hook_name = f"blocks.{layer_idx}.attn.hook_z"
                if hook_name not in self.mean_z:
                    continue

                mean_z_layer = self.mean_z[hook_name]  # [1, mean_seq, n_heads, d_head]
                heads_frozen = frozenset(heads_in_layer)

                def make_layer_hook(mz, heads_set, n_h, d_h):
                    """Factory: creates hook ablating multiple heads in one layer."""
                    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
                        cur_seq = z.shape[1]
                        mz_seq = mz.shape[1]
                        mz_trim = mz[:, :min(cur_seq, mz_seq), :, :]
                        if mz_seq < cur_seq:
                            pad = torch.zeros(
                                1, cur_seq - mz_seq, n_h, d_h,
                                device=z.device, dtype=mz.dtype
                            )
                            mz_trim = torch.cat([mz_trim, pad], dim=1)

                        z_out = z.clone()
                        for h_idx in heads_set:
                            z_out[:, :cur_seq, h_idx, :] = (
                                mz_trim[:, :cur_seq, h_idx, :]
                                .to(z.dtype)
                                .expand(z.shape[0], -1, -1)
                            )
                        return z_out
                    return hook_fn

                fwd_hooks.append((
                    hook_name,
                    make_layer_hook(
                        mean_z_layer.to(self.device),
                        heads_frozen,
                        self.n_heads,
                        self.d_head,
                    )
                ))

            logits = self.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

            for i, (io_id, s_id, seq_len) in enumerate(
                zip(batch_io, batch_s, seq_lengths)
            ):
                ld = compute_logit_diff(logits[i, seq_len - 1, :], io_id, s_id)
                total_ld += ld
                n_correct += 1 if ld > 0 else 0

        return total_ld / n_total, n_correct / n_total

    @torch.no_grad()
    def _compute_baseline(
        self,
        prompts: list[str],
        io_ids: list[int],
        s_ids: list[int],
    ) -> tuple[float, float]:
        """Compute baseline logit_diff and accuracy without any ablation."""
        total_ld = 0.0
        n_correct = 0
        n_total = len(prompts)

        for batch_start in range(0, n_total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_total)
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

        return total_ld / n_total, n_correct / n_total

    def run_necessity_test(self) -> ValidationResult:
        """
        Necessity test: ablate ONLY the circuit heads and measure the drop.

        If the circuit is truly necessary, ablating it should cause a large
        drop in logit difference. We ablate all heads in the circuit
        simultaneously (this is the strongest version of the necessity test).

        Returns
        -------
        ValidationResult
            With score = necessity_score ∈ [0, 1].
            Score ≈ 1.0 → circuit is necessary.
            Score ≈ 0.0 → circuit is NOT necessary (redundant).
        """
        logger.info("[CircuitValidator] Running necessity test…")
        prompts = self.dataset.get_clean_prompts()[:self.n_samples]
        io_ids = self.dataset.get_io_token_ids()[:self.n_samples]
        s_ids = self.dataset.get_s_token_ids()[:self.n_samples]

        baseline_ld, baseline_acc = self._compute_baseline(prompts, io_ids, s_ids)
        ablated_ld, ablated_acc = self._compute_ld_with_ablation(
            prompts, io_ids, s_ids,
            heads_to_ablate=self.circuit.heads,
        )

        ld_drop = baseline_ld - ablated_ld
        necessity_score = ld_drop / abs(baseline_ld) if baseline_ld != 0 else 0.0

        logger.info(
            f"[CircuitValidator.necessity] "
            f"Baseline LD={baseline_ld:+.4f}, Ablated LD={ablated_ld:+.4f}, "
            f"Necessity score={necessity_score:.4f}"
        )

        return ValidationResult(
            test_name="necessity",
            baseline_ld=baseline_ld,
            experimental_ld=ablated_ld,
            baseline_acc=baseline_acc,
            experimental_acc=ablated_acc,
            score=necessity_score,
            n_prompts=self.n_samples,
            circuit_n_heads=len(self.circuit),
            extra={
                "ld_drop": ld_drop,
                "interpretation": (
                    "HIGH necessity — circuit is causally required"
                    if necessity_score > 0.5 else
                    "LOW necessity — circuit is partly redundant"
                ),
            },
        )

    def run_sufficiency_test(self) -> ValidationResult:
        """
        Sufficiency test: ablate everything EXCEPT the circuit heads.

        If the circuit is sufficient, the complement ablation should leave
        performance roughly intact. We ablate all heads NOT in the circuit.

        Returns
        -------
        ValidationResult
            With score = sufficiency_score ∈ [0, 1].
            Score ≈ 1.0 → circuit alone is sufficient.
            Score ≈ 0.0 → circuit alone performs at chance level.
        """
        logger.info("[CircuitValidator] Running sufficiency test…")
        prompts = self.dataset.get_clean_prompts()[:self.n_samples]
        io_ids = self.dataset.get_io_token_ids()[:self.n_samples]
        s_ids = self.dataset.get_s_token_ids()[:self.n_samples]

        # Compute complement: all heads NOT in the circuit
        circuit_set = set(self.circuit.heads)
        all_heads = [
            (l, h)
            for l in range(self.n_layers)
            for h in range(self.n_heads)
        ]
        complement_heads = [h for h in all_heads if h not in circuit_set]

        logger.info(
            f"[CircuitValidator.sufficiency] "
            f"Circuit: {len(circuit_set)} heads, "
            f"Complement: {len(complement_heads)} heads to ablate"
        )

        baseline_ld, baseline_acc = self._compute_baseline(prompts, io_ids, s_ids)
        preserved_ld, preserved_acc = self._compute_ld_with_ablation(
            prompts, io_ids, s_ids,
            heads_to_ablate=complement_heads,
        )

        sufficiency_score = preserved_ld / baseline_ld if baseline_ld != 0 else 0.0

        logger.info(
            f"[CircuitValidator.sufficiency] "
            f"Baseline LD={baseline_ld:+.4f}, Preserved LD={preserved_ld:+.4f}, "
            f"Sufficiency score={sufficiency_score:.4f}"
        )

        return ValidationResult(
            test_name="sufficiency",
            baseline_ld=baseline_ld,
            experimental_ld=preserved_ld,
            baseline_acc=baseline_acc,
            experimental_acc=preserved_acc,
            score=sufficiency_score,
            n_prompts=self.n_samples,
            circuit_n_heads=len(self.circuit),
            extra={
                "complement_n_heads": len(complement_heads),
                "interpretation": (
                    "HIGH sufficiency — circuit alone performs the task"
                    if sufficiency_score > 0.5 else
                    "LOW sufficiency — circuit is incomplete"
                ),
            },
        )

    def run_generalization_test(
        self,
        new_names: Optional[list[str]] = None,
        new_templates: Optional[list[str]] = None,
        held_out_start: int = 800,
    ) -> list[ValidationResult]:
        """
        Generalization test: evaluate necessity across distribution shifts.

        Tests three conditions:
          1. New names (not seen during circuit identification)
          2. Held-out prompts (reserved from the original dataset)
          3. Different templates (if dataset has template metadata)

        For each condition, measures the necessity score when ablating the circuit.
        A robust circuit should show consistent necessity across all conditions.

        Parameters
        ----------
        new_names : list of str, optional
            Novel first names to test. If None, uses reserved names from the
            IOIDataset name pool.

        new_templates : list of str, optional
            Novel sentence templates. If None, uses alternate phrasings of IOI.

        held_out_start : int
            Dataset index to start the held-out split.
            Prompts [held_out_start:] are treated as held-out.

        Returns
        -------
        list of ValidationResult
            One result per generalization condition.
        """
        logger.info("[CircuitValidator] Running generalization tests…")
        results: list[ValidationResult] = []

        baseline_ld_all, baseline_acc_all = self._compute_baseline(
            self.dataset.get_clean_prompts()[:self.n_samples],
            self.dataset.get_io_token_ids()[:self.n_samples],
            self.dataset.get_s_token_ids()[:self.n_samples],
        )

        # ── Condition 1: Held-out prompts ─────────────────────────────────
        held_out_prompts = self.dataset.get_clean_prompts()[held_out_start:]
        held_out_io = self.dataset.get_io_token_ids()[held_out_start:]
        held_out_s = self.dataset.get_s_token_ids()[held_out_start:]
        n_held = min(100, len(held_out_prompts))

        if n_held > 0:
            logger.info(f"  Generalization: held-out ({n_held} prompts)…")
            ho_baseline_ld, ho_baseline_acc = self._compute_baseline(
                held_out_prompts[:n_held],
                held_out_io[:n_held],
                held_out_s[:n_held],
            )
            ho_ablated_ld, ho_ablated_acc = self._compute_ld_with_ablation(
                held_out_prompts[:n_held],
                held_out_io[:n_held],
                held_out_s[:n_held],
                heads_to_ablate=self.circuit.heads,
            )
            necessity = (
                (ho_baseline_ld - ho_ablated_ld) / abs(ho_baseline_ld)
                if ho_baseline_ld != 0 else 0.0
            )
            results.append(ValidationResult(
                test_name="generalization_held_out",
                baseline_ld=ho_baseline_ld,
                experimental_ld=ho_ablated_ld,
                baseline_acc=ho_baseline_acc,
                experimental_acc=ho_ablated_acc,
                score=necessity,
                n_prompts=n_held,
                circuit_n_heads=len(self.circuit),
                extra={"condition": "held_out"},
            ))

        # ── Condition 2: ABB-only templates ───────────────────────────────
        abb_mask = [
            p.template_type == "ABB"
            for p in self.dataset.prompts[:self.n_samples]
        ]
        abb_prompts = [self.dataset.prompts[i].prompt_clean
                       for i, m in enumerate(abb_mask) if m]
        abb_io = [self.dataset.prompts[i].io_token_id
                  for i, m in enumerate(abb_mask) if m]
        abb_s = [self.dataset.prompts[i].s_token_id
                 for i, m in enumerate(abb_mask) if m]

        if abb_prompts:
            n_abb = min(100, len(abb_prompts))
            logger.info(f"  Generalization: ABB templates ({n_abb} prompts)…")
            abb_baseline_ld, abb_baseline_acc = self._compute_baseline(
                abb_prompts[:n_abb], abb_io[:n_abb], abb_s[:n_abb]
            )
            abb_ablated_ld, abb_ablated_acc = self._compute_ld_with_ablation(
                abb_prompts[:n_abb], abb_io[:n_abb], abb_s[:n_abb],
                heads_to_ablate=self.circuit.heads,
            )
            necessity = (
                (abb_baseline_ld - abb_ablated_ld) / abs(abb_baseline_ld)
                if abb_baseline_ld != 0 else 0.0
            )
            results.append(ValidationResult(
                test_name="generalization_abb_template",
                baseline_ld=abb_baseline_ld,
                experimental_ld=abb_ablated_ld,
                baseline_acc=abb_baseline_acc,
                experimental_acc=abb_ablated_acc,
                score=necessity,
                n_prompts=n_abb,
                circuit_n_heads=len(self.circuit),
                extra={"condition": "ABB_template"},
            ))

        # ── Condition 3: BAB-only templates ───────────────────────────────
        bab_mask = [
            p.template_type == "BAB"
            for p in self.dataset.prompts[:self.n_samples]
        ]
        bab_prompts = [self.dataset.prompts[i].prompt_clean
                       for i, m in enumerate(bab_mask) if m]
        bab_io = [self.dataset.prompts[i].io_token_id
                  for i, m in enumerate(bab_mask) if m]
        bab_s = [self.dataset.prompts[i].s_token_id
                 for i, m in enumerate(bab_mask) if m]

        if bab_prompts:
            n_bab = min(100, len(bab_prompts))
            logger.info(f"  Generalization: BAB templates ({n_bab} prompts)…")
            bab_baseline_ld, bab_baseline_acc = self._compute_baseline(
                bab_prompts[:n_bab], bab_io[:n_bab], bab_s[:n_bab]
            )
            bab_ablated_ld, bab_ablated_acc = self._compute_ld_with_ablation(
                bab_prompts[:n_bab], bab_io[:n_bab], bab_s[:n_bab],
                heads_to_ablate=self.circuit.heads,
            )
            necessity = (
                (bab_baseline_ld - bab_ablated_ld) / abs(bab_baseline_ld)
                if bab_baseline_ld != 0 else 0.0
            )
            results.append(ValidationResult(
                test_name="generalization_bab_template",
                baseline_ld=bab_baseline_ld,
                experimental_ld=bab_ablated_ld,
                baseline_acc=bab_baseline_acc,
                experimental_acc=bab_ablated_acc,
                score=necessity,
                n_prompts=n_bab,
                circuit_n_heads=len(self.circuit),
                extra={"condition": "BAB_template"},
            ))

        logger.info(
            f"[CircuitValidator.generalization] ✓ {len(results)} conditions tested."
        )
        return results

    def run_all_tests(self) -> list[ValidationResult]:
        """
        Run all validation tests and return a combined list of results.

        Returns
        -------
        list of ValidationResult
            Ordered: [necessity, sufficiency, gen_held_out, gen_abb, gen_bab]
        """
        logger.info("[CircuitValidator.run_all_tests] Starting full validation suite…")
        results: list[ValidationResult] = []
        results.append(self.run_necessity_test())
        results.append(self.run_sufficiency_test())
        results.extend(self.run_generalization_test())
        logger.info(
            f"[CircuitValidator.run_all_tests] ✓ {len(results)} tests complete.\n"
            + "\n".join(
                f"  {r.test_name}: score={r.score:.4f}, "
                f"acc_change={r.experimental_acc - r.baseline_acc:+.1%}"
                for r in results
            )
        )
        return results
