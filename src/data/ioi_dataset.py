"""
src/data/ioi_dataset.py
========================
Complete Indirect Object Identification (IOI) dataset generator.

Background: The IOI Task
-------------------------
The IOI task was introduced by Wang et al. (2022) in:
  "Interpretability in the Wild: a Circuit for Indirect Object Identification
   in GPT-2 Small" (https://arxiv.org/abs/2202.00571)

The task probes whether a language model can correctly resolve indirect
object references. Consider the sentence:

    "When John and Mary went to the park, John gave the book to"

The model must predict " Mary" (the indirect object) over " John" (the
subject, who was already the giver). A model performing IOI correctly:
  1. Identifies that "John" is the SENDER (S) — the one doing the giving.
  2. Identifies that "Mary" is the INDIRECT OBJECT (IO) — the recipient.
  3. Predicts the IO token with higher probability than the S token.

Template Types
--------------
We implement two canonical template structures:

  ABB — IO appears first, S appears twice:
    "When {IO} and {S} went to {PLACE}, {S} gave the {OBJECT} to"
    Correct answer: {IO}

  BAB — S appears first, IO appears second (then S repeats):
    "When {S} and {IO} went to {PLACE}, {S} gave the {OBJECT} to"
    Correct answer: {IO}

Both templates test the same ability but with different token positions,
allowing us to verify that the circuit is robust to position.

Dataset Structure
-----------------
Each IOIPrompt contains:
  - prompt_clean    : The original IOI prompt
  - prompt_corrupted: A corrupted version where the S name is replaced
                      with a random distractor name (used for activation
                      patching — the corrupted model output is our baseline)
  - io_name         : The indirect object name (correct answer)
  - s_name          : The subject name (incorrect/distractor answer)
  - distractor_name : Name used in corrupted prompt (≠ IO, ≠ S)
  - template_type   : "ABB" or "BAB"
  - template_idx    : Which specific template string was used
  - place           : Location filler word
  - object          : Object filler word
  - io_token_id     : Token ID of " {IO}" (leading space included)
  - s_token_id      : Token ID of " {S}"

Corruption Strategy
-------------------
To run causal tracing / activation patching, we need a "corrupted" prompt
that produces the wrong answer. We replace the S name with a random
distractor (a third name not present in the prompt). This changes which
name the model should predict but keeps all other token positions fixed,
making the comparison clean.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)

# ── Name Pool ─────────────────────────────────────────────────────────────────
# 35 common English first names (gender-neutral for simplicity).
# Wang et al. (2022) used a similar English name set.
# Names are chosen to be single tokens in GPT-2's BPE vocabulary
# (confirmed: each tokenises to " Name" = 1 token with leading space).
ALL_NAMES: list[str] = [
    "Alice", "Bob", "Carol", "David", "Emma",
    "Frank", "Grace", "Henry", "Iris", "Jack",
    "Kate", "Liam", "Mia", "Noah", "Olivia",
    "Paul", "Quinn", "Rose", "Sam", "Tara",
    "Uma", "Victor", "Wendy", "Xander", "Yara",
    "Zach", "Amy", "Brian", "Clara", "Derek",
    "Ella", "Fred", "Gina", "Harry", "Isla",
]

# ── Location fillers ──────────────────────────────────────────────────────────
PLACES: list[str] = [
    "the store", "the park", "the library", "the office", "the restaurant",
    "the gym", "the museum", "the market", "the hospital", "the beach",
    "the school", "the station", "the theater", "the cafe", "the mall",
]

# ── Object fillers ────────────────────────────────────────────────────────────
OBJECTS: list[str] = [
    "book", "gift", "letter", "package", "key",
    "bag", "note", "ticket", "card", "pen",
    "phone", "jacket", "wallet", "map", "trophy",
]

# ── Template Strings ──────────────────────────────────────────────────────────
# Each template uses named placeholders:
#   {io}   → indirect object name (correct answer)
#   {s}    → subject name (wrong answer — the one doing the action)
#   {place}→ location filler
#   {obj}  → object filler
#
# The prompt is designed so that the LAST token predicts the IO.
# TransformerLens evaluates the logit at the FINAL token position.

ABB_TEMPLATES: list[str] = [
    "When {io} and {s} went to {place}, {s} gave the {obj} to",
    "When {io} and {s} visited {place}, {s} handed the {obj} to",
    "When {io} and {s} arrived at {place}, {s} passed the {obj} to",
    "After {io} and {s} left {place}, {s} sent the {obj} to",
    "Once {io} and {s} were at {place}, {s} brought the {obj} to",
    "While {io} and {s} were at {place}, {s} delivered the {obj} to",
    "As {io} and {s} walked through {place}, {s} offered the {obj} to",
]

BAB_TEMPLATES: list[str] = [
    "When {s} and {io} went to {place}, {s} gave the {obj} to",
    "When {s} and {io} visited {place}, {s} handed the {obj} to",
    "When {s} and {io} arrived at {place}, {s} passed the {obj} to",
    "After {s} and {io} left {place}, {s} sent the {obj} to",
    "Once {s} and {io} were at {place}, {s} brought the {obj} to",
    "While {s} and {io} were at {place}, {s} delivered the {obj} to",
    "As {s} and {io} walked through {place}, {s} offered the {obj} to",
]


@dataclass
class IOIPrompt:
    """
    A single IOI prompt with all associated metadata.

    Attributes
    ----------
    prompt_clean : str
        The natural-language IOI sentence. Example:
        "When Alice and Bob went to the park, Bob gave the book to"

    prompt_corrupted : str
        The same sentence with the subject name replaced by a random
        distractor. Used as the "counterfactual" in patching experiments.
        Example: "When Alice and Carol went to the park, Carol gave the book to"
        (where Carol is the distractor, replacing Bob)

    io_name : str
        The indirect object (IO) name — the correct completion.

    s_name : str
        The subject (S) name — the incorrect completion (wrong answer).

    distractor_name : str
        The name used in `prompt_corrupted` replacing the S name.

    template_type : str
        "ABB" or "BAB" indicating the structural variant.

    template_idx : int
        Index into the ABB_TEMPLATES or BAB_TEMPLATES list.

    place : str
        The location filler string used in the prompt.

    object_noun : str
        The object filler string used in the prompt.

    io_token_id : int
        GPT-2 token ID for " {io_name}" (with leading space).
        This is what we read from the final-position logit vector.

    s_token_id : int
        GPT-2 token ID for " {s_name}" (with leading space).
    """

    prompt_clean: str
    prompt_corrupted: str
    io_name: str
    s_name: str
    distractor_name: str
    template_type: str
    template_idx: int
    place: str
    object_noun: str
    io_token_id: int = field(default=-1)
    s_token_id: int = field(default=-1)

    def to_dict(self) -> dict:
        """Convert to a flat dictionary (for DataFrame construction)."""
        return asdict(self)


class IOIDataset:
    """
    Generator for a large-scale IOI (Indirect Object Identification) dataset.

    This class creates clean and corrupted prompt pairs for evaluating and
    patching the IOI circuit in GPT-2 Small. The dataset is reproducible
    given a fixed random seed.

    Parameters
    ----------
    model : HookedTransformer
        A loaded GPT-2 Small model. Used only for tokenising names to
        obtain token IDs. The model is not run during dataset construction.

    n_prompts : int
        Total number of IOI prompts to generate. Split equally between
        ABB and BAB template types (±1 for odd numbers).

    seed : int
        Random seed for reproducible name/template/filler sampling.
        Must match the seed passed to `set_seed()` for global consistency.

    names : list of str, optional
        Custom name pool. Defaults to `ALL_NAMES` (35 names). Must have
        at least 3 names (we need IO, S, and distractor simultaneously).

    Attributes
    ----------
    prompts : list[IOIPrompt]
        All generated IOI prompts after calling `generate()`.

    df : pd.DataFrame
        Tabular representation of all prompts (one row per prompt).

    Examples
    --------
    >>> from src.model import load_model
    >>> from src.data import IOIDataset
    >>> model = load_model("gpt2")
    >>> dataset = IOIDataset(model, n_prompts=1000, seed=42)
    >>> dataset.generate()
    >>> print(len(dataset))      # 1000
    >>> print(dataset.df.columns.tolist())
    """

    def __init__(
        self,
        model: HookedTransformer,
        n_prompts: int = 1000,
        seed: int = 42,
        names: Optional[list[str]] = None,
    ) -> None:
        if names is not None and len(names) < 3:
            raise ValueError("Name pool must contain at least 3 names.")

        self.model = model
        self.n_prompts = n_prompts
        self.seed = seed
        self.names = names or ALL_NAMES

        # Verify all names are single tokens in GPT-2's vocabulary.
        # GPT-2 BPE: " Alice" (with space) must be 1 token, not 2.
        self._verify_name_tokens()

        self.prompts: list[IOIPrompt] = []
        self.df: pd.DataFrame = pd.DataFrame()

        logger.info(
            f"[IOIDataset] Initialized: {n_prompts} prompts, "
            f"{len(self.names)} names, seed={seed}."
        )

    def _verify_name_tokens(self) -> None:
        """
        Verify that every name in the pool tokenises to exactly 1 token
        (with a leading space, as they appear in mid-sentence position).

        Names that tokenise to 2+ tokens would create a token-position
        mismatch in the evaluation (we'd read the wrong logit position).

        Issues a WARNING for any problematic names but does not remove them
        automatically — the user should update the name pool if needed.
        """
        multi_token_names: list[str] = []
        for name in self.names:
            # " Name" is how the name appears mid-sentence in GPT-2 BPE
            token_ids = self.model.to_tokens(f" {name}", prepend_bos=False)[0]
            if len(token_ids) != 1:
                multi_token_names.append(name)
                logger.warning(
                    f"[IOIDataset] Name '{name}' tokenises to "
                    f"{len(token_ids)} tokens: "
                    f"{self.model.to_str_tokens(' ' + name, prepend_bos=False)}. "
                    f"Consider removing it from the name pool."
                )

        if not multi_token_names:
            logger.info(
                f"[IOIDataset] ✓ All {len(self.names)} names are single-token."
            )

    def _get_token_id(self, name: str) -> int:
        """
        Return the GPT-2 token ID for " {name}" (leading space included).

        The leading space is crucial: in BPE, " Alice" (space+Alice) is a
        different token from "Alice". Mid-sentence names always have a
        preceding space in GPT-2's tokenisation.

        Parameters
        ----------
        name : str
            Name string WITHOUT leading space. The space is added internally.

        Returns
        -------
        int
            Token ID as a Python int (not a tensor).
        """
        # model.to_single_token raises if the string maps to multiple tokens
        try:
            return self.model.to_single_token(f" {name}")
        except Exception:
            # Fallback: use first token from full tokenisation
            token_ids = self.model.to_tokens(f" {name}", prepend_bos=False)[0]
            return token_ids[0].item()

    def _build_prompt(
        self,
        template: str,
        io: str,
        s: str,
        place: str,
        obj: str,
    ) -> str:
        """
        Fill a template string with concrete name/place/object values.

        Parameters
        ----------
        template : str
            Template with {io}, {s}, {place}, {obj} placeholders.
        io : str
            Indirect object name.
        s : str
            Subject name.
        place : str
            Location filler.
        obj : str
            Object noun.

        Returns
        -------
        str
            The completed prompt string.
        """
        return template.format(io=io, s=s, place=place, obj=obj)

    def _sample_names(self, rng: random.Random) -> tuple[str, str, str]:
        """
        Sample three distinct names from the pool: IO, S, and distractor.

        Parameters
        ----------
        rng : random.Random
            Seeded random instance for reproducible sampling.

        Returns
        -------
        tuple of (io_name, s_name, distractor_name)
            Three distinct names from the pool.
        """
        # Sample without replacement to guarantee all three are distinct
        io_name, s_name, distractor = rng.sample(self.names, 3)
        return io_name, s_name, distractor

    def generate(self) -> "IOIDataset":
        """
        Generate `n_prompts` IOI prompts and populate `self.prompts` and `self.df`.

        The generation process:
          1. Split n_prompts evenly between ABB and BAB templates.
          2. For each prompt:
             a. Sample IO, S, and distractor names (without replacement).
             b. Sample a template, place, and object randomly.
             c. Build the clean prompt using the template.
             d. Build the corrupted prompt by substituting S → distractor.
             e. Look up token IDs for IO and S.
          3. Convert to DataFrame.

        Returns
        -------
        IOIDataset
            Returns `self` for method chaining.

        Examples
        --------
        >>> dataset = IOIDataset(model, n_prompts=100, seed=42).generate()
        >>> print(dataset.df.shape)  # (100, 11)
        """
        logger.info(f"[IOIDataset.generate] Generating {self.n_prompts} prompts…")

        # Use a seeded Random instance (not the global random) for isolation
        rng = random.Random(self.seed)

        # Split evenly between template types
        n_abb = self.n_prompts // 2
        n_bab = self.n_prompts - n_abb

        self.prompts = []

        # ── Generate ABB prompts ──────────────────────────────────────────
        for i in range(n_abb):
            io, s, distractor = self._sample_names(rng)
            template_idx = rng.randrange(len(ABB_TEMPLATES))
            template = ABB_TEMPLATES[template_idx]
            place = rng.choice(PLACES)
            obj = rng.choice(OBJECTS)

            # Clean: S name used normally
            clean = self._build_prompt(template, io=io, s=s, place=place, obj=obj)
            # Corrupted: replace S with distractor throughout the prompt
            corrupted = self._build_prompt(
                template, io=io, s=distractor, place=place, obj=obj
            )

            prompt = IOIPrompt(
                prompt_clean=clean,
                prompt_corrupted=corrupted,
                io_name=io,
                s_name=s,
                distractor_name=distractor,
                template_type="ABB",
                template_idx=template_idx,
                place=place,
                object_noun=obj,
                io_token_id=self._get_token_id(io),
                s_token_id=self._get_token_id(s),
            )
            self.prompts.append(prompt)

        # ── Generate BAB prompts ──────────────────────────────────────────
        for i in range(n_bab):
            io, s, distractor = self._sample_names(rng)
            template_idx = rng.randrange(len(BAB_TEMPLATES))
            template = BAB_TEMPLATES[template_idx]
            place = rng.choice(PLACES)
            obj = rng.choice(OBJECTS)

            clean = self._build_prompt(template, io=io, s=s, place=place, obj=obj)
            corrupted = self._build_prompt(
                template, io=io, s=distractor, place=place, obj=obj
            )

            prompt = IOIPrompt(
                prompt_clean=clean,
                prompt_corrupted=corrupted,
                io_name=io,
                s_name=s,
                distractor_name=distractor,
                template_type="BAB",
                template_idx=template_idx,
                place=place,
                object_noun=obj,
                io_token_id=self._get_token_id(io),
                s_token_id=self._get_token_id(s),
            )
            self.prompts.append(prompt)

        # ── Shuffle to interleave ABB and BAB ────────────────────────────
        rng.shuffle(self.prompts)

        # ── Build DataFrame ───────────────────────────────────────────────
        self.df = pd.DataFrame([p.to_dict() for p in self.prompts])
        self.df.index.name = "prompt_id"

        # ── Log statistics ────────────────────────────────────────────────
        template_counts = self.df["template_type"].value_counts().to_dict()
        logger.info(
            f"[IOIDataset.generate] ✓ Generated {len(self.prompts)} prompts.\n"
            f"  Template type distribution: {template_counts}\n"
            f"  Unique IO names  : {self.df['io_name'].nunique()}\n"
            f"  Unique S names   : {self.df['s_name'].nunique()}\n"
            f"  Sample clean     : {self.prompts[0].prompt_clean!r}\n"
            f"  Sample corrupted : {self.prompts[0].prompt_corrupted!r}"
        )

        return self

    def __len__(self) -> int:
        """Return the number of generated prompts."""
        return len(self.prompts)

    def __getitem__(self, idx: int) -> IOIPrompt:
        """Return a single IOIPrompt by index."""
        return self.prompts[idx]

    def get_clean_prompts(self) -> list[str]:
        """Return list of all clean prompt strings."""
        return [p.prompt_clean for p in self.prompts]

    def get_corrupted_prompts(self) -> list[str]:
        """Return list of all corrupted prompt strings."""
        return [p.prompt_corrupted for p in self.prompts]

    def get_io_token_ids(self) -> list[int]:
        """Return list of IO token IDs (correct answer token IDs)."""
        return [p.io_token_id for p in self.prompts]

    def get_s_token_ids(self) -> list[int]:
        """Return list of S token IDs (incorrect answer token IDs)."""
        return [p.s_token_id for p in self.prompts]

    def filter_by_template(self, template_type: str) -> "IOIDataset":
        """
        Return a new IOIDataset-like object containing only prompts of
        a specific template type.

        Parameters
        ----------
        template_type : str
            "ABB" or "BAB".

        Returns
        -------
        IOIDataset
            A shallow-copied dataset with filtered prompts and df.
        """
        if template_type not in {"ABB", "BAB"}:
            raise ValueError(f"template_type must be 'ABB' or 'BAB', got '{template_type}'.")

        # Create a lightweight copy with filtered prompts
        filtered = object.__new__(IOIDataset)
        filtered.model = self.model
        filtered.n_prompts = len(self.prompts)
        filtered.seed = self.seed
        filtered.names = self.names
        filtered.prompts = [p for p in self.prompts if p.template_type == template_type]
        filtered.df = self.df[self.df["template_type"] == template_type].copy()
        return filtered

    def summary(self) -> str:
        """
        Return a human-readable summary of the dataset statistics.

        Returns
        -------
        str
            Multi-line summary string.
        """
        if not self.prompts:
            return "IOIDataset: empty (call .generate() first)"

        lines = [
            "=" * 60,
            "IOIDataset Summary",
            "=" * 60,
            f"  Total prompts     : {len(self.prompts):,}",
            f"  ABB prompts       : {sum(1 for p in self.prompts if p.template_type == 'ABB'):,}",
            f"  BAB prompts       : {sum(1 for p in self.prompts if p.template_type == 'BAB'):,}",
            f"  Unique IO names   : {len(set(p.io_name for p in self.prompts))}",
            f"  Unique S names    : {len(set(p.s_name for p in self.prompts))}",
            f"  Name pool size    : {len(self.names)}",
            f"  Template variants : {len(ABB_TEMPLATES)} ABB + {len(BAB_TEMPLATES)} BAB",
            f"  Location fillers  : {len(PLACES)}",
            f"  Object fillers    : {len(OBJECTS)}",
            "",
            "  Sample prompt (clean):",
            f"    {self.prompts[0].prompt_clean!r}",
            "  Sample prompt (corrupted):",
            f"    {self.prompts[0].prompt_corrupted!r}",
            "=" * 60,
        ]
        return "\n".join(lines)
