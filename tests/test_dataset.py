"""
tests/test_dataset.py
======================
Unit tests for src/data/ioi_dataset.py

Tests verify:
  - Dataset generation produces the correct number of prompts
  - Template type distribution is balanced (ABB/BAB split)
  - IO and S names are always distinct within a prompt
  - IO and S names are always distinct from the distractor name
  - All names come from the defined name pool
  - Clean and corrupted prompts differ correctly (S → distractor swap)
  - Token IDs are non-negative integers (valid vocabulary indices)
  - The DataFrame has the correct column schema
  - Reproducibility: same seed → identical dataset

These tests use a MockModel to avoid loading GPT-2 at test time.
The MockModel mimics the subset of the HookedTransformer API used
by IOIDataset (to_tokens, to_single_token, tokenizer).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure project root is in path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ioi_dataset import IOIDataset, ALL_NAMES, IOIPrompt


# ── Mock Model ─────────────────────────────────────────────────────────────
class MockTokenizer:
    """Minimal tokenizer mock for testing without loading GPT-2."""
    bos_token = "<|endoftext|>"
    bos_token_id = 50256
    eos_token = "<|endoftext|>"
    pad_token_id = None


class MockConfig:
    """Minimal model config mock."""
    d_vocab = 50257
    default_prepend_bos = True


class MockModel:
    """
    Minimal HookedTransformer mock for dataset tests.

    Simulates the token lookup behavior of GPT-2 by assigning a
    unique deterministic token ID to each name string.
    """
    tokenizer = MockTokenizer()
    cfg = MockConfig()

    # Build a fake vocabulary: each name " Name" → a fixed token ID
    _fake_vocab: dict[str, int] = {
        f" {name}": 1000 + i for i, name in enumerate(ALL_NAMES)
    }

    def to_tokens(self, text: str, prepend_bos: bool = True):
        """Return a mock token tensor (always single token for names)."""
        import torch
        # For testing: every string tokenises to exactly 1 token
        token_id = abs(hash(text)) % 40000 + 100
        if prepend_bos:
            return torch.tensor([[50256, token_id]])
        return torch.tensor([[token_id]])

    def to_single_token(self, text: str) -> int:
        """Return a fake single-token ID for the given string."""
        if text in self._fake_vocab:
            return self._fake_vocab[text]
        # Fallback: hash-based ID in safe range
        return abs(hash(text)) % 40000 + 100

    def to_str_tokens(self, text: str, prepend_bos: bool = True):
        """Return a list of token strings."""
        return [text]


@pytest.fixture(scope="module")
def mock_model() -> MockModel:
    """Shared MockModel instance for all tests in this module."""
    return MockModel()


@pytest.fixture(scope="module")
def dataset_100(mock_model: MockModel) -> IOIDataset:
    """A small generated dataset for fast testing."""
    ds = IOIDataset(model=mock_model, n_prompts=100, seed=42)
    ds.generate()
    return ds


@pytest.fixture(scope="module")
def dataset_1000(mock_model: MockModel) -> IOIDataset:
    """A full-size dataset matching the production configuration."""
    ds = IOIDataset(model=mock_model, n_prompts=1000, seed=42)
    ds.generate()
    return ds


# ── Test: Basic generation ────────────────────────────────────────────────

class TestDatasetGeneration:
    """Tests for basic dataset generation correctness."""

    def test_dataset_length_100(self, dataset_100: IOIDataset):
        """Dataset should contain exactly n_prompts prompts."""
        assert len(dataset_100) == 100, (
            f"Expected 100 prompts, got {len(dataset_100)}"
        )

    def test_dataset_length_1000(self, dataset_1000: IOIDataset):
        """Production-size dataset should have 1000 prompts."""
        assert len(dataset_1000) == 1000

    def test_dataframe_row_count(self, dataset_100: IOIDataset):
        """DataFrame row count must match number of prompts."""
        assert len(dataset_100.df) == len(dataset_100)

    def test_dataframe_columns(self, dataset_100: IOIDataset):
        """DataFrame must contain all expected columns."""
        expected_columns = {
            "prompt_clean", "prompt_corrupted",
            "io_name", "s_name", "distractor_name",
            "template_type", "template_idx",
            "place", "object_noun",
            "io_token_id", "s_token_id",
        }
        actual_columns = set(dataset_100.df.columns.tolist())
        assert expected_columns.issubset(actual_columns), (
            f"Missing columns: {expected_columns - actual_columns}"
        )

    def test_no_null_values(self, dataset_100: IOIDataset):
        """No column should have null values in a properly generated dataset."""
        null_counts = dataset_100.df.isnull().sum()
        assert null_counts.sum() == 0, (
            f"Found null values:\n{null_counts[null_counts > 0]}"
        )


# ── Test: Template distribution ───────────────────────────────────────────

class TestTemplateDistribution:
    """Tests for template type balance."""

    def test_abb_bab_split_100(self, dataset_100: IOIDataset):
        """With 100 prompts, expect 50 ABB and 50 BAB."""
        counts = dataset_100.df["template_type"].value_counts()
        assert counts.get("ABB", 0) == 50, f"Expected 50 ABB, got {counts.get('ABB', 0)}"
        assert counts.get("BAB", 0) == 50, f"Expected 50 BAB, got {counts.get('BAB', 0)}"

    def test_abb_bab_split_1000(self, dataset_1000: IOIDataset):
        """With 1000 prompts, expect 500 ABB and 500 BAB."""
        counts = dataset_1000.df["template_type"].value_counts()
        assert counts.get("ABB", 0) == 500
        assert counts.get("BAB", 0) == 500

    def test_template_types_valid(self, dataset_100: IOIDataset):
        """All template types must be either 'ABB' or 'BAB'."""
        valid_types = {"ABB", "BAB"}
        actual_types = set(dataset_100.df["template_type"].unique())
        assert actual_types.issubset(valid_types), (
            f"Invalid template types found: {actual_types - valid_types}"
        )


# ── Test: Name constraints ────────────────────────────────────────────────

class TestNameConstraints:
    """Tests for name sampling correctness."""

    def test_io_s_distinct(self, dataset_100: IOIDataset):
        """IO and S names must be distinct in every prompt."""
        for prompt in dataset_100.prompts:
            assert prompt.io_name != prompt.s_name, (
                f"IO and S names are the same: '{prompt.io_name}' "
                f"in prompt: {prompt.prompt_clean!r}"
            )

    def test_io_distractor_distinct(self, dataset_100: IOIDataset):
        """IO and distractor names must be distinct in every prompt."""
        for prompt in dataset_100.prompts:
            assert prompt.io_name != prompt.distractor_name, (
                f"IO and distractor are the same: '{prompt.io_name}'"
            )

    def test_s_distractor_distinct(self, dataset_100: IOIDataset):
        """S and distractor names must be distinct in every prompt."""
        for prompt in dataset_100.prompts:
            assert prompt.s_name != prompt.distractor_name, (
                f"S and distractor are the same: '{prompt.s_name}'"
            )

    def test_all_three_names_distinct(self, dataset_100: IOIDataset):
        """IO, S, and distractor must all be distinct (three-way check)."""
        for prompt in dataset_100.prompts:
            names = {prompt.io_name, prompt.s_name, prompt.distractor_name}
            assert len(names) == 3, (
                f"Expected 3 distinct names, got: {names}"
            )

    def test_names_from_pool(self, dataset_100: IOIDataset):
        """All names must come from the ALL_NAMES pool."""
        for prompt in dataset_100.prompts:
            assert prompt.io_name in ALL_NAMES, f"IO name not in pool: {prompt.io_name}"
            assert prompt.s_name in ALL_NAMES, f"S name not in pool: {prompt.s_name}"
            assert prompt.distractor_name in ALL_NAMES, (
                f"Distractor name not in pool: {prompt.distractor_name}"
            )

    def test_name_pool_coverage(self, dataset_1000: IOIDataset):
        """With 1000 prompts, most of the 35-name pool should appear."""
        io_names_used = set(dataset_1000.df["io_name"].unique())
        # Expect at least 25 of 35 names to appear as IO
        assert len(io_names_used) >= 25, (
            f"Only {len(io_names_used)} IO names used out of {len(ALL_NAMES)}"
        )


# ── Test: Prompt correctness ──────────────────────────────────────────────

class TestPromptCorrectness:
    """Tests for prompt string construction."""

    def test_io_name_in_clean_prompt(self, dataset_100: IOIDataset):
        """The IO name must appear in the clean prompt."""
        for prompt in dataset_100.prompts:
            assert prompt.io_name in prompt.prompt_clean, (
                f"IO name '{prompt.io_name}' not in prompt: {prompt.prompt_clean!r}"
            )

    def test_s_name_in_clean_prompt(self, dataset_100: IOIDataset):
        """The S name must appear in the clean prompt."""
        for prompt in dataset_100.prompts:
            assert prompt.s_name in prompt.prompt_clean, (
                f"S name '{prompt.s_name}' not in prompt: {prompt.prompt_clean!r}"
            )

    def test_distractor_in_corrupted_prompt(self, dataset_100: IOIDataset):
        """The distractor name must appear in the corrupted prompt."""
        for prompt in dataset_100.prompts:
            assert prompt.distractor_name in prompt.prompt_corrupted, (
                f"Distractor '{prompt.distractor_name}' not in corrupted: "
                f"{prompt.prompt_corrupted!r}"
            )

    def test_s_not_in_corrupted_prompt(self, dataset_100: IOIDataset):
        """The original S name must NOT appear in the corrupted prompt."""
        for prompt in dataset_100.prompts:
            assert prompt.s_name not in prompt.prompt_corrupted, (
                f"S name '{prompt.s_name}' found in corrupted prompt: "
                f"{prompt.prompt_corrupted!r}"
            )

    def test_clean_and_corrupted_differ(self, dataset_100: IOIDataset):
        """Clean and corrupted prompts must be different strings."""
        for prompt in dataset_100.prompts:
            assert prompt.prompt_clean != prompt.prompt_corrupted, (
                "Clean and corrupted prompts are identical!"
            )

    def test_prompt_ends_with_to(self, dataset_100: IOIDataset):
        """All prompts should end with ' to' (the cloze task setup)."""
        for prompt in dataset_100.prompts:
            assert prompt.prompt_clean.endswith(" to"), (
                f"Prompt does not end with ' to': {prompt.prompt_clean!r}"
            )


# ── Test: Token IDs ───────────────────────────────────────────────────────

class TestTokenIds:
    """Tests for token ID validity."""

    def test_io_token_id_positive(self, dataset_100: IOIDataset):
        """IO token IDs must be non-negative."""
        for prompt in dataset_100.prompts:
            assert prompt.io_token_id >= 0, (
                f"Invalid IO token ID: {prompt.io_token_id}"
            )

    def test_s_token_id_positive(self, dataset_100: IOIDataset):
        """S token IDs must be non-negative."""
        for prompt in dataset_100.prompts:
            assert prompt.s_token_id >= 0, (
                f"Invalid S token ID: {prompt.s_token_id}"
            )

    def test_io_s_token_ids_differ(self, dataset_100: IOIDataset):
        """IO and S token IDs should differ (since IO ≠ S names)."""
        for prompt in dataset_100.prompts:
            assert prompt.io_token_id != prompt.s_token_id, (
                f"IO and S token IDs are equal: {prompt.io_token_id}"
            )

    def test_token_ids_in_vocab_range(self, dataset_100: IOIDataset):
        """Token IDs must be within the GPT-2 vocabulary range [0, 50256]."""
        for prompt in dataset_100.prompts:
            assert 0 <= prompt.io_token_id <= 50256, (
                f"IO token ID out of range: {prompt.io_token_id}"
            )
            assert 0 <= prompt.s_token_id <= 50256, (
                f"S token ID out of range: {prompt.s_token_id}"
            )


# ── Test: Reproducibility ─────────────────────────────────────────────────

class TestReproducibility:
    """Tests that the same seed always produces identical datasets."""

    def test_same_seed_same_dataset(self, mock_model: MockModel):
        """Two datasets with the same seed must be identical."""
        ds1 = IOIDataset(model=mock_model, n_prompts=50, seed=42).generate()
        ds2 = IOIDataset(model=mock_model, n_prompts=50, seed=42).generate()

        for i, (p1, p2) in enumerate(zip(ds1.prompts, ds2.prompts)):
            assert p1.prompt_clean == p2.prompt_clean, (
                f"Prompt {i} differs between seeds:\n"
                f"  Run 1: {p1.prompt_clean!r}\n"
                f"  Run 2: {p2.prompt_clean!r}"
            )

    def test_different_seed_different_dataset(self, mock_model: MockModel):
        """Two datasets with different seeds should differ."""
        ds1 = IOIDataset(model=mock_model, n_prompts=50, seed=42).generate()
        ds2 = IOIDataset(model=mock_model, n_prompts=50, seed=99).generate()

        # At least some prompts should differ
        differences = sum(
            1 for p1, p2 in zip(ds1.prompts, ds2.prompts)
            if p1.prompt_clean != p2.prompt_clean
        )
        assert differences > 0, "Different seeds produced identical datasets!"


# ── Test: Filter by template ──────────────────────────────────────────────

class TestFilterByTemplate:
    """Tests for the filter_by_template convenience method."""

    def test_filter_abb(self, dataset_100: IOIDataset):
        """Filtering ABB should return only ABB prompts."""
        filtered = dataset_100.filter_by_template("ABB")
        assert all(p.template_type == "ABB" for p in filtered.prompts)
        assert all(filtered.df["template_type"] == "ABB")

    def test_filter_bab(self, dataset_100: IOIDataset):
        """Filtering BAB should return only BAB prompts."""
        filtered = dataset_100.filter_by_template("BAB")
        assert all(p.template_type == "BAB" for p in filtered.prompts)

    def test_filter_invalid_raises(self, dataset_100: IOIDataset):
        """Filtering with an invalid type should raise ValueError."""
        with pytest.raises(ValueError, match="ABB"):
            dataset_100.filter_by_template("XYZ")

    def test_filter_preserves_count(self, dataset_100: IOIDataset):
        """ABB + BAB filtered counts should sum to total."""
        n_abb = len(dataset_100.filter_by_template("ABB"))
        n_bab = len(dataset_100.filter_by_template("BAB"))
        assert n_abb + n_bab == len(dataset_100)


# ── Test: Error handling ──────────────────────────────────────────────────

class TestErrorHandling:
    """Tests for error conditions."""

    def test_too_few_names_raises(self, mock_model: MockModel):
        """Providing fewer than 3 names should raise ValueError."""
        with pytest.raises(ValueError, match="at least 3"):
            IOIDataset(model=mock_model, n_prompts=10, names=["Alice", "Bob"])

    def test_exactly_three_names_works(self, mock_model: MockModel):
        """Providing exactly 3 names should work (though diversity is low)."""
        ds = IOIDataset(
            model=mock_model,
            n_prompts=10,
            names=["Alice", "Bob", "Carol"]
        ).generate()
        assert len(ds) == 10
