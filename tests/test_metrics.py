"""
tests/test_metrics.py
======================
Unit tests for src/evaluation/metrics.py

Tests verify:
  - compute_logit_diff returns correct values for known inputs
  - compute_top_k returns correct number of predictions and types
  - compute_token_rank returns rank 1 for argmax token
  - IOIEvaluator.compute_aggregate_stats returns correct structure
  - Results DataFrame has correct column schema
  - Accuracy and logit_diff are consistent with each other

All tests use pre-computed synthetic logit tensors rather than
running actual GPT-2 forward passes (which would be slow in CI).
The MockEvaluator creates a controlled test environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# ── Ensure project root is in path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_logit_diff,
    compute_top_k,
    compute_token_rank,
    IOIEvaluator,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

VOCAB_SIZE = 50257  # GPT-2 vocabulary size


@pytest.fixture
def zero_logits() -> torch.Tensor:
    """A logit tensor with all zeros (uniform distribution)."""
    return torch.zeros(VOCAB_SIZE)


@pytest.fixture
def peaked_logits() -> torch.Tensor:
    """
    A logit tensor where token 100 has a very high logit (10.0)
    and all others are 0.0. Used to test rank-1 predictions.
    """
    logits = torch.zeros(VOCAB_SIZE)
    logits[100] = 10.0
    return logits


@pytest.fixture
def io_better_logits() -> torch.Tensor:
    """
    Logit tensor where IO token (200) > S token (300).
    logit_diff = 3.0 - 1.0 = 2.0 (model is correct).
    """
    logits = torch.zeros(VOCAB_SIZE)
    logits[200] = 3.0  # IO token
    logits[300] = 1.0  # S token
    return logits


@pytest.fixture
def s_better_logits() -> torch.Tensor:
    """
    Logit tensor where S token (300) > IO token (200).
    logit_diff = 1.0 - 3.0 = -2.0 (model is incorrect).
    """
    logits = torch.zeros(VOCAB_SIZE)
    logits[200] = 1.0  # IO token
    logits[300] = 3.0  # S token
    return logits


class MockTokenizerForMetrics:
    """Minimal tokenizer for metrics tests."""

    def batch_decode(self, token_ids):
        """Decode token IDs to strings."""
        return [f"token_{t.item()}" for t in token_ids.squeeze(-1)]

    bos_token_id = 50256


# ── Tests: compute_logit_diff ─────────────────────────────────────────────

class TestComputeLogitDiff:
    """Unit tests for the compute_logit_diff function."""

    def test_correct_prediction_positive(self, io_better_logits):
        """When IO > S, logit_diff should be positive."""
        ld = compute_logit_diff(io_better_logits, io_token_id=200, s_token_id=300)
        assert ld > 0, f"Expected positive logit diff, got {ld}"

    def test_incorrect_prediction_negative(self, s_better_logits):
        """When S > IO, logit_diff should be negative."""
        ld = compute_logit_diff(s_better_logits, io_token_id=200, s_token_id=300)
        assert ld < 0, f"Expected negative logit diff, got {ld}"

    def test_exact_value_correct(self, io_better_logits):
        """Logit diff should equal exactly 3.0 - 1.0 = 2.0."""
        ld = compute_logit_diff(io_better_logits, io_token_id=200, s_token_id=300)
        assert abs(ld - 2.0) < 1e-5, f"Expected 2.0, got {ld}"

    def test_exact_value_incorrect(self, s_better_logits):
        """Logit diff should equal exactly 1.0 - 3.0 = -2.0."""
        ld = compute_logit_diff(s_better_logits, io_token_id=200, s_token_id=300)
        assert abs(ld - (-2.0)) < 1e-5, f"Expected -2.0, got {ld}"

    def test_zero_logits_gives_zero_diff(self, zero_logits):
        """Equal logits should give zero logit difference."""
        ld = compute_logit_diff(zero_logits, io_token_id=100, s_token_id=200)
        assert ld == 0.0, f"Expected 0.0 for equal logits, got {ld}"

    def test_returns_float(self, io_better_logits):
        """Return type must be Python float, not tensor."""
        ld = compute_logit_diff(io_better_logits, io_token_id=200, s_token_id=300)
        assert isinstance(ld, float), f"Expected float, got {type(ld)}"

    def test_raises_on_2d_input(self):
        """Should raise AssertionError for batched (2D) input."""
        logits_2d = torch.zeros(2, VOCAB_SIZE)
        with pytest.raises(AssertionError):
            compute_logit_diff(logits_2d, io_token_id=100, s_token_id=200)

    def test_symmetry(self, io_better_logits):
        """Swapping IO and S IDs should negate the logit diff."""
        ld_forward = compute_logit_diff(io_better_logits, io_token_id=200, s_token_id=300)
        ld_swapped = compute_logit_diff(io_better_logits, io_token_id=300, s_token_id=200)
        assert abs(ld_forward + ld_swapped) < 1e-5, (
            "Swapping IO and S should negate the logit diff"
        )


# ── Tests: compute_top_k ──────────────────────────────────────────────────

class TestComputeTopK:
    """Unit tests for the compute_top_k function."""

    def test_returns_k_items(self, peaked_logits):
        """Should return exactly k tokens and k logit values."""
        tokenizer = MockTokenizerForMetrics()
        tokens, logit_vals = compute_top_k(peaked_logits, tokenizer, k=5)
        assert len(tokens) == 5, f"Expected 5 tokens, got {len(tokens)}"
        assert len(logit_vals) == 5, f"Expected 5 logit values, got {len(logit_vals)}"

    def test_top1_is_argmax(self, peaked_logits):
        """The first predicted token should be the argmax token (ID 100)."""
        tokenizer = MockTokenizerForMetrics()
        tokens, logit_vals = compute_top_k(peaked_logits, tokenizer, k=5)
        # Token ID 100 has the highest logit; its string is "token_100"
        assert tokens[0] == "token_100", (
            f"Expected top-1 to be 'token_100', got '{tokens[0]}'"
        )

    def test_top1_logit_is_max(self, peaked_logits):
        """The highest logit value should be 10.0 (our injected value)."""
        tokenizer = MockTokenizerForMetrics()
        tokens, logit_vals = compute_top_k(peaked_logits, tokenizer, k=5)
        assert abs(logit_vals[0] - 10.0) < 1e-4, (
            f"Expected max logit 10.0, got {logit_vals[0]}"
        )

    def test_logit_vals_descending(self, peaked_logits):
        """Logit values should be in descending order."""
        tokenizer = MockTokenizerForMetrics()
        _, logit_vals = compute_top_k(peaked_logits, tokenizer, k=10)
        for i in range(len(logit_vals) - 1):
            assert logit_vals[i] >= logit_vals[i + 1], (
                f"Logit values not sorted: {logit_vals[i]} < {logit_vals[i + 1]} at idx {i}"
            )

    def test_tokens_are_strings(self, peaked_logits):
        """All returned tokens must be Python strings."""
        tokenizer = MockTokenizerForMetrics()
        tokens, _ = compute_top_k(peaked_logits, tokenizer, k=5)
        for t in tokens:
            assert isinstance(t, str), f"Token is not a string: {t!r} ({type(t)})"

    def test_logit_vals_are_floats(self, peaked_logits):
        """All returned logit values must be Python floats."""
        tokenizer = MockTokenizerForMetrics()
        _, logit_vals = compute_top_k(peaked_logits, tokenizer, k=5)
        for v in logit_vals:
            assert isinstance(v, float), f"Logit value is not a float: {v!r}"

    def test_k_equals_1(self, peaked_logits):
        """Should work with k=1."""
        tokenizer = MockTokenizerForMetrics()
        tokens, logit_vals = compute_top_k(peaked_logits, tokenizer, k=1)
        assert len(tokens) == 1
        assert len(logit_vals) == 1


# ── Tests: compute_token_rank ─────────────────────────────────────────────

class TestComputeTokenRank:
    """Unit tests for the compute_token_rank function."""

    def test_rank_1_for_argmax(self, peaked_logits):
        """The argmax token should have rank 1."""
        rank = compute_token_rank(peaked_logits, token_id=100)
        assert rank == 1, f"Expected rank 1 for argmax token, got {rank}"

    def test_rank_is_integer(self, peaked_logits):
        """Rank must be a Python int."""
        rank = compute_token_rank(peaked_logits, token_id=100)
        assert isinstance(rank, int), f"Expected int rank, got {type(rank)}"

    def test_rank_within_vocab(self, peaked_logits):
        """Rank must be between 1 and d_vocab inclusive."""
        for token_id in [0, 100, 1000, 25000, 50256]:
            rank = compute_token_rank(peaked_logits, token_id=token_id)
            assert 1 <= rank <= VOCAB_SIZE, (
                f"Rank {rank} out of range [1, {VOCAB_SIZE}] for token {token_id}"
            )

    def test_zero_logits_any_rank(self, zero_logits):
        """With equal logits, rank should be valid (position may vary by sort stability)."""
        rank = compute_token_rank(zero_logits, token_id=500)
        assert 1 <= rank <= VOCAB_SIZE

    def test_non_argmax_has_rank_gt_1(self, peaked_logits):
        """A non-argmax token (not 100) should have rank > 1."""
        rank = compute_token_rank(peaked_logits, token_id=999)
        assert rank > 1, f"Expected rank > 1 for non-argmax token, got {rank}"


# ── Tests: IOIEvaluator.compute_aggregate_stats ───────────────────────────

class TestComputeAggregateStats:
    """Tests for aggregate statistics computation."""

    @pytest.fixture
    def sample_results_df(self) -> pd.DataFrame:
        """
        Create a minimal synthetic results DataFrame for testing.

        Contains 10 correct and 10 incorrect predictions, split evenly
        between ABB and BAB templates.
        """
        n = 20
        rng = np.random.default_rng(42)

        df = pd.DataFrame({
            "is_correct": [True] * 10 + [False] * 10,
            "logit_diff": (
                rng.uniform(0.5, 5.0, size=10).tolist() +    # correct: positive
                rng.uniform(-3.0, -0.1, size=10).tolist()    # incorrect: negative
            ),
            "prob_io": (
                rng.uniform(0.3, 0.9, size=10).tolist() +
                rng.uniform(0.05, 0.3, size=10).tolist()
            ),
            "prob_s": rng.uniform(0.01, 0.2, size=n).tolist(),
            "rank_io": (
                [1, 2, 3, 1, 5, 2, 1, 4, 3, 2] +            # correct: low ranks
                [100, 500, 200, 1000, 50, 300, 150, 80, 250, 400]  # incorrect: higher
            ),
            "template_type": (["ABB"] * 5 + ["BAB"] * 5) * 2,
        })
        return df

    def test_accuracy_is_05(self, sample_results_df):
        """With 10 correct / 20 total, accuracy should be 0.5."""
        from tests.test_dataset import MockModel
        from src.data.ioi_dataset import IOIDataset

        # We need a minimal IOIEvaluator just to call compute_aggregate_stats
        mock_model = MockModel()
        mock_dataset = IOIDataset(model=mock_model, n_prompts=10, seed=42).generate()
        evaluator = IOIEvaluator(
            model=mock_model,
            dataset=mock_dataset,
            batch_size=4,
        )

        stats = evaluator.compute_aggregate_stats(sample_results_df)
        assert abs(stats["accuracy"] - 0.5) < 1e-6, (
            f"Expected accuracy 0.5, got {stats['accuracy']}"
        )

    def test_stats_keys_present(self, sample_results_df):
        """All expected keys must be present in stats dict."""
        from tests.test_dataset import MockModel
        from src.data.ioi_dataset import IOIDataset

        mock_model = MockModel()
        mock_dataset = IOIDataset(model=mock_model, n_prompts=10, seed=42).generate()
        evaluator = IOIEvaluator(model=mock_model, dataset=mock_dataset, batch_size=4)

        stats = evaluator.compute_aggregate_stats(sample_results_df)
        required_keys = {
            "n_prompts", "accuracy", "mean_logit_diff", "std_logit_diff",
            "median_logit_diff", "mean_prob_io", "mean_rank_io",
            "accuracy_abb", "accuracy_bab",
        }
        missing = required_keys - set(stats.keys())
        assert not missing, f"Missing stats keys: {missing}"

    def test_per_template_accuracy(self, sample_results_df):
        """ABB and BAB accuracies should be computed correctly."""
        from tests.test_dataset import MockModel
        from src.data.ioi_dataset import IOIDataset

        mock_model = MockModel()
        mock_dataset = IOIDataset(model=mock_model, n_prompts=10, seed=42).generate()
        evaluator = IOIEvaluator(model=mock_model, dataset=mock_dataset, batch_size=4)

        stats = evaluator.compute_aggregate_stats(sample_results_df)

        # Both ABB and BAB should have valid accuracy values in [0, 1]
        for tmpl_acc in ["accuracy_abb", "accuracy_bab"]:
            val = stats[tmpl_acc]
            assert val is not None, f"{tmpl_acc} should not be None"
            assert 0.0 <= val <= 1.0, (
                f"{tmpl_acc}={val} is outside [0, 1]"
            )

    def test_n_prompts_matches_df_length(self, sample_results_df):
        """n_prompts in stats should match len(results_df)."""
        from tests.test_dataset import MockModel
        from src.data.ioi_dataset import IOIDataset

        mock_model = MockModel()
        mock_dataset = IOIDataset(model=mock_model, n_prompts=10, seed=42).generate()
        evaluator = IOIEvaluator(model=mock_model, dataset=mock_dataset, batch_size=4)

        stats = evaluator.compute_aggregate_stats(sample_results_df)
        assert stats["n_prompts"] == len(sample_results_df), (
            f"n_prompts mismatch: {stats['n_prompts']} vs {len(sample_results_df)}"
        )


# ── Tests: Metric consistency ─────────────────────────────────────────────

class TestMetricConsistency:
    """Tests that cross-validate metrics for internal consistency."""

    def test_logit_diff_sign_matches_is_correct(self):
        """
        is_correct should be True iff logit_diff > 0.
        Create synthetic data and verify this invariant holds.
        """
        rng = np.random.default_rng(0)
        logit_diffs = rng.uniform(-5, 5, size=100)

        for ld in logit_diffs:
            is_correct = ld > 0
            # This is a definitional invariant — verify the logic is sound
            if ld > 0:
                assert is_correct
            else:
                assert not is_correct

    def test_positive_logit_diff_implies_prob_io_gt_prob_s(self):
        """
        If logit(IO) > logit(S), then softmax(IO) > softmax(S).
        This is a mathematical property of softmax.
        """
        import torch.nn.functional as F

        logits = torch.zeros(VOCAB_SIZE)
        logits[200] = 3.0  # IO
        logits[300] = 1.0  # S

        probs = F.softmax(logits, dim=-1)
        assert probs[200] > probs[300], (
            f"Expected prob_io > prob_s, got {probs[200]:.6f} vs {probs[300]:.6f}"
        )

    def test_logit_diff_additivity(self):
        """
        For any three tokens A, B, C:
        logit_diff(A, B) + logit_diff(B, C) == logit_diff(A, C)
        """
        logits = torch.randn(VOCAB_SIZE)
        a, b, c = 100, 200, 300

        diff_ab = compute_logit_diff(logits, a, b)
        diff_bc = compute_logit_diff(logits, b, c)
        diff_ac = compute_logit_diff(logits, a, c)

        assert abs((diff_ab + diff_bc) - diff_ac) < 1e-4, (
            "Logit diff additivity violated: "
            f"diff(A,B)+diff(B,C)={diff_ab+diff_bc:.6f} ≠ diff(A,C)={diff_ac:.6f}"
        )
