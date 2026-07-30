"""
src/data/pronoun_dataset.py
==============================
Pronoun Resolution Dataset for the Novel Extension Experiment.

Task Description
-----------------
Pronoun resolution (coreference resolution) is the task of identifying
which entity in a sentence is referred to by a pronoun.

Example (female pronoun, IO = Mary):
    "John met Mary at the store. She bought a gift for"
    → Model should predict "John" (the recipient, since She=Mary)

Example (male pronoun, IO = John):
    "Mary met John at the park. He thanked her for"
    → Model should predict "Mary" (the recipient, since He=John)

Relationship to IOI
--------------------
The pronoun resolution task shares structural properties with IOI:
  - Both require binding a name to a position in the sentence
  - Both require tracking which entity performed an action
  - Both depend on predicting the name of the "other" person

Key Difference from IOI:
  - In IOI, the model sees BOTH names explicitly and must prefer one
  - In pronoun resolution, the model must first RESOLVE the pronoun,
    then predict the referent's interaction partner
  - This adds an extra coreference step requiring gender agreement

This allows us to test whether the IOI circuit (Name Mover heads,
S-Inhibition heads) also mediates pronoun-based reference.

Dataset Structure
-----------------
For each prompt:
  - speaker_name: the person who performed the action (pronoun antecedent)
  - recipient_name: the person the speaker interacted with (correct answer)
  - pronoun: "He" or "She" matching the speaker's gender
  - template: sentence structure
  - target_token_id: recipient_name token ID (what the model should predict)
  - foil_token_id: speaker_name token ID (what the model should NOT predict)

Gender Assignment
-----------------
We assign gender to names based on a curated list. Only single-token names
with clear gender assignment are included. This avoids ambiguous cases that
would confound the analysis.

Female names: Alice, Emma, Grace, Iris, Kate, Mia, Olivia, Rose, Tara, Yara, Amy, Clara, Ella, Gina, Isla
Male names  : Bob, David, Frank, Henry, Jack, Liam, Noah, Paul, Sam, Victor, Xander, Zach, Brian, Derek, Fred, Harry
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Name pools with gender assignment ───────────────────────────────────────
FEMALE_NAMES = [
    "Alice", "Emma", "Grace", "Iris", "Kate",
    "Mia", "Olivia", "Rose", "Tara", "Yara",
    "Amy", "Clara", "Ella", "Gina", "Isla",
]

MALE_NAMES = [
    "Bob", "David", "Frank", "Henry", "Jack",
    "Liam", "Noah", "Paul", "Sam", "Victor",
    "Xander", "Zach", "Brian", "Derek", "Fred",
    "Harry",
]

GENDER_MAP: dict[str, str] = (
    {n: "F" for n in FEMALE_NAMES} | {n: "M" for n in MALE_NAMES}
)

PRONOUN_MAP: dict[str, str] = {"F": "She", "M": "He"}

# ── Sentence templates ───────────────────────────────────────────────────────
# {speaker} = the person who performed the action (pronoun antecedent)
# {recipient} = the person who received the action (model should predict)
# {pronoun} = gendered pronoun for speaker
PRONOUN_TEMPLATES = [
    "{speaker} met {recipient} at the store. {pronoun} bought a gift for",
    "{speaker} visited {recipient} at home. {pronoun} brought flowers for",
    "{speaker} called {recipient} on the phone. {pronoun} left a message for",
    "{speaker} helped {recipient} move. {pronoun} carried the boxes for",
    "{speaker} thanked {recipient} warmly. {pronoun} wrote a letter for",
    "{speaker} surprised {recipient} at the party. {pronoun} baked a cake for",
    "{speaker} recommended {recipient} for the job. {pronoun} wrote a reference for",
]


@dataclass
class PronounPrompt:
    """A single pronoun resolution prompt pair (correct + foil)."""
    prompt: str
    speaker_name: str
    recipient_name: str
    pronoun: str
    template_idx: int
    target_token_id: int       # ID of recipient_name (correct answer)
    foil_token_id: int         # ID of speaker_name (wrong answer)
    speaker_gender: str        # "M" or "F"

    @property
    def logit_diff_tokens(self) -> tuple[int, int]:
        """(target_token_id, foil_token_id) — for compute_logit_diff."""
        return self.target_token_id, self.foil_token_id


class PronounDataset:
    """
    Generates a pronoun resolution dataset for novel extension experiments.

    The dataset is structurally analogous to IOIDataset:
      - target_token_id ↔ io_token_id (correct answer)
      - foil_token_id ↔ s_token_id (wrong answer)
      - logit_diff = logit(target) - logit(foil)

    This allows the same analysis pipeline (layer ablation, head ablation,
    activation patching) to be applied without code changes.

    Parameters
    ----------
    model : HookedTransformer
        Used for tokenisation and name-to-token-id mapping.

    n_prompts : int
        Number of prompts to generate.

    seed : int
        Random seed for reproducibility.

    Examples
    --------
    >>> dataset = PronounDataset(model, n_prompts=500, seed=42).generate()
    >>> print(dataset.prompts[0].prompt)
    'Alice met Bob at the store. She bought a gift for'
    """

    def __init__(self, model, n_prompts: int = 500, seed: int = 42) -> None:
        self.model = model
        self.n_prompts = n_prompts
        self.seed = seed
        self.prompts: list[PronounPrompt] = []
        self._rng = random.Random(seed)

        # Verify all names tokenize as single tokens
        self._valid_female = self._filter_single_token(FEMALE_NAMES)
        self._valid_male = self._filter_single_token(MALE_NAMES)
        logger.info(
            f"[PronounDataset] Valid names: "
            f"{len(self._valid_female)} female, {len(self._valid_male)} male"
        )

    def _filter_single_token(self, names: list[str]) -> list[str]:
        """Keep only names that tokenize as exactly one token (with leading space)."""
        valid = []
        for name in names:
            try:
                tok = self.model.to_tokens(f" {name}", prepend_bos=False)
                if tok.shape[1] == 1:
                    valid.append(name)
            except Exception:
                pass
        return valid

    def _get_token_id(self, name: str) -> int:
        """Get token ID for a name (with leading space for mid-sentence usage)."""
        return self.model.to_tokens(f" {name}", prepend_bos=False)[0, 0].item()

    def generate(self) -> "PronounDataset":
        """
        Generate `n_prompts` pronoun resolution prompts.

        Each prompt is generated by:
          1. Sampling a speaker (male or female) and a different-gender recipient.
             (Same-gender pairs are excluded to ensure pronoun uniqueness.)
          2. Applying a random template.
          3. Computing token IDs for both names.

        Returns self for method chaining.
        """
        self.prompts = []
        all_female = self._valid_female
        all_male = self._valid_male

        for i in range(self.n_prompts):
            template_idx = i % len(PRONOUN_TEMPLATES)
            template = PRONOUN_TEMPLATES[template_idx]

            # Alternate speaker gender to balance the dataset
            if i % 2 == 0:
                # Female speaker, male recipient
                speaker = self._rng.choice(all_female)
                recipient = self._rng.choice(all_male)
            else:
                # Male speaker, female recipient
                speaker = self._rng.choice(all_male)
                recipient = self._rng.choice(all_female)

            # Ensure speaker ≠ recipient (names are different by construction
            # since we draw from different gender pools, but check anyway)
            if speaker == recipient:
                continue

            gender = GENDER_MAP[speaker]
            pronoun = PRONOUN_MAP[gender]

            prompt = template.format(
                speaker=speaker,
                recipient=recipient,
                pronoun=pronoun,
            )

            try:
                target_id = self._get_token_id(recipient)
                foil_id = self._get_token_id(speaker)
            except Exception as exc:
                logger.debug(f"Token ID error for {recipient}/{speaker}: {exc}")
                continue

            self.prompts.append(PronounPrompt(
                prompt=prompt,
                speaker_name=speaker,
                recipient_name=recipient,
                pronoun=pronoun,
                template_idx=template_idx,
                target_token_id=target_id,
                foil_token_id=foil_id,
                speaker_gender=gender,
            ))

        logger.info(
            f"[PronounDataset] Generated {len(self.prompts)} prompts "
            f"({self.n_prompts} requested)"
        )
        return self

    def get_clean_prompts(self) -> list[str]:
        """Return list of prompt strings."""
        return [p.prompt for p in self.prompts]

    def get_io_token_ids(self) -> list[int]:
        """Return list of target (recipient) token IDs. Mirrors IOIDataset API."""
        return [p.target_token_id for p in self.prompts]

    def get_s_token_ids(self) -> list[int]:
        """Return list of foil (speaker) token IDs. Mirrors IOIDataset API."""
        return [p.foil_token_id for p in self.prompts]

    def __len__(self) -> int:
        return len(self.prompts)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert dataset to a pandas DataFrame."""
        return pd.DataFrame([
            {
                "prompt": p.prompt,
                "speaker_name": p.speaker_name,
                "recipient_name": p.recipient_name,
                "pronoun": p.pronoun,
                "template_idx": p.template_idx,
                "target_token_id": p.target_token_id,
                "foil_token_id": p.foil_token_id,
                "speaker_gender": p.speaker_gender,
            }
            for p in self.prompts
        ])
