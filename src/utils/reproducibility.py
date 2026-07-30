"""
src/utils/reproducibility.py
==============================
Reproducibility utilities for deterministic experiment execution.

Why reproducibility matters in mechanistic interpretability
------------------------------------------------------------
IOI circuit experiments require exact re-runs to:
  1. Verify that the same 1000 prompts are generated each time.
  2. Ensure activation patching results are numerically identical.
  3. Allow collaborators to reproduce published figures.

We control randomness at four levels:
  - Python's `random` module (name sampling, template selection)
  - NumPy (array operations in dataset construction)
  - PyTorch CPU (tensor ops on CPU)
  - PyTorch CUDA (GPU kernel selection — requires deterministic algorithms)
"""

from __future__ import annotations

import os
import random
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set global random seeds for Python, NumPy, and PyTorch.

    This function must be called **before** any random operation in the
    experiment pipeline — ideally as the very first call in your script.

    Parameters
    ----------
    seed : int
        The integer seed value. The default (42) is widely used in ML
        research for reproducibility. Change only if generating multiple
        independent experimental runs.

    Side Effects
    ------------
    - Sets `random.seed(seed)`
    - Sets `np.random.seed(seed)`
    - Sets `torch.manual_seed(seed)`
    - Sets `torch.cuda.manual_seed_all(seed)` if CUDA is available
    - Sets the `PYTHONHASHSEED` environment variable (affects dict ordering
      in Python 3.3+)

    Notes
    -----
    This does NOT guarantee 100% reproducibility on GPU — CUDA operations
    like cuDNN convolutions use non-deterministic algorithms by default.
    Call `configure_determinism()` after this function for full GPU
    determinism (at a performance cost).

    Examples
    --------
    >>> from src.utils import set_seed
    >>> set_seed(42)
    """
    # Python built-in random (used in dataset name sampling)
    random.seed(seed)

    # NumPy random (array shuffling, statistical operations)
    np.random.seed(seed)

    # PyTorch CPU random number generator
    torch.manual_seed(seed)

    # PyTorch CUDA random number generators (all GPU devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Python hash seed (affects dict iteration order, set hashing)
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"[set_seed] Global random seed set to {seed}.")


def configure_determinism(
    enabled: bool = True,
    warn_only: bool = False,
) -> None:
    """
    Enable or disable PyTorch deterministic algorithm mode.

    When enabled, PyTorch forces every operation to use a deterministic
    algorithm if one is available. Operations without a deterministic
    implementation will raise a `RuntimeError` (unless `warn_only=True`).

    Parameters
    ----------
    enabled : bool
        If True, enable deterministic mode. If False, restore default
        (non-deterministic) mode for maximum performance.

    warn_only : bool
        If True, emit a warning instead of raising an error when a
        non-deterministic operation is encountered. Useful during
        development to identify which operations are non-deterministic
        without crashing the experiment.

    Performance Note
    ----------------
    Deterministic mode can significantly slow down CUDA operations
    (sometimes 2–5× slower for attention operations). For large-scale
    sweeps, disable this and rely on `set_seed()` alone.

    Examples
    --------
    >>> from src.utils import set_seed, configure_determinism
    >>> set_seed(42)
    >>> configure_determinism(enabled=True, warn_only=False)
    """
    # torch.use_deterministic_algorithms ensures that PyTorch selects
    # deterministic implementations wherever available.
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    # torch.backends.cudnn.deterministic affects the cuDNN library which
    # provides GPU-accelerated implementations of convolution etc.
    torch.backends.cudnn.deterministic = enabled

    # Disabling cuDNN benchmark mode prevents auto-tuning, which would
    # pick the fastest (potentially non-deterministic) kernel.
    torch.backends.cudnn.benchmark = not enabled

    status = "ENABLED" if enabled else "DISABLED"
    logger.info(
        f"[configure_determinism] Deterministic mode {status} "
        f"(warn_only={warn_only})."
    )


def get_reproducibility_state() -> dict:
    """
    Return a snapshot of the current reproducibility configuration.

    Useful for logging experiment metadata so results can be traced back
    to exact environment settings.

    Returns
    -------
    dict
        Keys: torch_version, cuda_available, cudnn_deterministic,
              use_deterministic_algorithms, python_hash_seed.
    """
    state = {
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not set"),
    }
    return state
