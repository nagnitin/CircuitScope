"""
src/utils/logger.py
====================
Structured logging utilities for CircuitScope experiments.

Design Rationale
----------------
We use Python's built-in `logging` module rather than a third-party library
because it is always available, integrates naturally with every library in
the stack (PyTorch, TransformerLens, etc.), and supports hierarchical loggers
that let us silence noisy sub-libraries without affecting our own output.

Logger Hierarchy
----------------
  root
  └── circuitscope              (level controlled by YAML config)
      ├── src.model.loader
      ├── src.data.ioi_dataset
      ├── src.evaluation.metrics
      ├── src.visualization.plots
      └── src.utils.*

All module loggers are children of the "circuitscope" namespace, so setting
the level on the parent controls the verbosity of all children.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Formatting constants ────────────────────────────────────────────────────
_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_FORMAT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
_DATE_FORMAT    = "%H:%M:%S"


def get_logger(
    name: str,
    level: str = "INFO",
    log_dir: Optional[Path | str] = None,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a named logger with console and/or file handlers.

    This function is idempotent: calling it multiple times with the same
    `name` returns the same logger with handlers added only once.

    Parameters
    ----------
    name : str
        Logger name. Use `__name__` inside module files so that the logger
        inherits the Python module hierarchy (e.g., "src.data.ioi_dataset").
        For the experiment runner, use "circuitscope.experiment".

    level : str
        Minimum logging level. One of: "DEBUG", "INFO", "WARNING", "ERROR".
        - DEBUG  : Very verbose; logs every tensor shape, token id, etc.
        - INFO   : Progress messages, model summaries, dataset stats.
        - WARNING: Only issues that might affect results.
        - ERROR  : Only failures.

    log_dir : Path or str, optional
        Directory where the log file will be written. If None or `file=False`,
        no file handler is attached. A timestamped file name is generated
        automatically: e.g., "circuitscope_20240101_120000.log".

    console : bool
        If True, attach a StreamHandler writing to sys.stdout.

    file : bool
        If True and `log_dir` is provided, attach a FileHandler.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> from src.utils import get_logger
    >>> logger = get_logger("circuitscope.experiment", level="INFO",
    ...                     log_dir="outputs/logs")
    >>> logger.info("Experiment started.")
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers are attached yet (idempotency guard).
    if logger.handlers:
        return logger

    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # ── Console Handler ───────────────────────────────────────────────────
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_formatter = logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # ── File Handler ──────────────────────────────────────────────────────
    if file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Timestamped log filename prevents overwriting previous runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"circuitscope_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Log the file path to console so users know where logs go
        logger.info(f"[get_logger] Log file: {log_file.resolve()}")

    # Prevent propagation to the root logger to avoid duplicate messages
    # when the root logger also has a StreamHandler (common in notebooks).
    logger.propagate = False

    return logger


def silence_external_loggers(loggers: Optional[list[str]] = None) -> None:
    """
    Set external library loggers to WARNING level to reduce noise.

    TransformerLens, HuggingFace Transformers, and PyTorch all emit INFO
    messages during model loading. This function silences them so only
    CircuitScope messages appear in output.

    Parameters
    ----------
    loggers : list of str, optional
        Additional logger names to silence. Default silenced loggers:
        ["transformers", "transformer_lens", "filelock", "urllib3"].

    Examples
    --------
    >>> from src.utils.logger import silence_external_loggers
    >>> silence_external_loggers()
    """
    default_noisy_loggers = [
        "transformers",
        "transformer_lens",
        "filelock",
        "urllib3",
        "PIL",
        "matplotlib",
        "accelerate",
    ]
    targets = default_noisy_loggers + (loggers or [])

    for logger_name in targets:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
