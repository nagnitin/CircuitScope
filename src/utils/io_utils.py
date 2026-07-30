"""
src/utils/io_utils.py
======================
I/O utilities for saving experiment outputs: CSVs, figures, and JSON metadata.

Design Principles
-----------------
1. All output paths are created automatically (no manual `mkdir` needed).
2. Functions are composable and return the Path of the saved file so callers
   can chain operations or log the save location.
3. Plotly figures are saved as both interactive HTML and static PNG.
4. Matplotlib figures use tight_layout and 300 dpi by default.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dirs(*paths: Union[str, Path]) -> None:
    """
    Create directories (and all parents) if they do not already exist.

    Parameters
    ----------
    *paths : str or Path
        One or more directory paths to create.

    Examples
    --------
    >>> from src.utils import ensure_dirs
    >>> ensure_dirs("outputs/figures", "outputs/results", "outputs/logs")
    """
    for p in paths:
        path = Path(p)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[ensure_dirs] Directory ready: {path.resolve()}")


def save_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    index: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Save a pandas DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save. Must not be empty.

    path : str or Path
        Destination file path. Parent directories are created automatically.
        Conventionally ends in ".csv".

    index : bool
        Whether to write the DataFrame index as a column. Default False
        (row numbers are not meaningful for IOI results).

    verbose : bool
        If True, log the shape and file path after saving.

    Returns
    -------
    Path
        Resolved absolute path of the saved CSV file.

    Raises
    ------
    ValueError
        If `df` is empty (likely indicates a bug in the pipeline).

    Examples
    --------
    >>> import pandas as pd
    >>> from src.utils import save_csv
    >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> path = save_csv(df, "outputs/results/test.csv")
    >>> print(path)
    """
    if df.empty:
        raise ValueError(
            f"[save_csv] Attempted to save an empty DataFrame to '{path}'. "
            "This likely indicates a bug in the evaluation pipeline."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)

    if verbose:
        logger.info(
            f"[save_csv] ✓ Saved {len(df):,} rows × {len(df.columns)} columns → {path.resolve()}"
        )
    return path.resolve()


def save_figure(
    fig: Any,
    path: Union[str, Path],
    formats: Optional[list[str]] = None,
    width: int = 900,
    height: int = 550,
    scale: float = 2.0,
) -> list[Path]:
    """
    Save a Plotly or Matplotlib figure to disk in one or more formats.

    Automatically detects the figure type and routes to the appropriate
    save method. Plotly HTML files are interactive (zoom, hover, pan).
    PNG files require the `kaleido` package (included in requirements.txt).

    Parameters
    ----------
    fig : plotly.graph_objects.Figure or matplotlib.figure.Figure
        The figure to save.

    path : str or Path
        Base path WITHOUT extension. Extensions are added per format.
        Example: "outputs/figures/logit_diff_histogram"
        Produces: "logit_diff_histogram.html", "logit_diff_histogram.png"

    formats : list of str, optional
        List of output formats. Default: ["html", "png"] for Plotly,
        ["png", "svg"] for Matplotlib.

    width : int
        Image width in pixels (Plotly only).

    height : int
        Image height in pixels (Plotly only).

    scale : float
        Pixel density multiplier for PNG export (Plotly only).
        scale=2.0 → retina-quality 1800×1100 at 72 dpi declared.

    Returns
    -------
    list of Path
        List of resolved paths for all saved files.

    Examples
    --------
    >>> import plotly.graph_objects as go
    >>> from src.utils import save_figure
    >>> fig = go.Figure(go.Bar(x=["a", "b"], y=[1, 2]))
    >>> paths = save_figure(fig, "outputs/figures/example", formats=["html", "png"])
    """
    import plotly.graph_objects as go

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    # ── Plotly figure ─────────────────────────────────────────────────────
    if isinstance(fig, go.Figure):
        formats = formats or ["html", "png"]
        for fmt in formats:
            out = path.with_suffix(f".{fmt}")
            if fmt == "html":
                fig.write_html(str(out))
            elif fmt in {"png", "svg", "pdf", "webp"}:
                try:
                    fig.write_image(
                        str(out),
                        width=width,
                        height=height,
                        scale=scale,
                    )
                except Exception as exc:
                    logger.warning(
                        f"[save_figure] Could not save {fmt} (kaleido installed?): {exc}"
                    )
                    continue
            saved_paths.append(out.resolve())
            logger.info(f"[save_figure] ✓ Saved → {out.resolve()}")

    # ── Matplotlib figure ─────────────────────────────────────────────────
    else:
        import matplotlib.pyplot as plt

        formats = formats or ["png", "svg"]
        for fmt in formats:
            out = path.with_suffix(f".{fmt}")
            fig.savefig(str(out), dpi=300, bbox_inches="tight")
            saved_paths.append(out.resolve())
            logger.info(f"[save_figure] ✓ Saved → {out.resolve()}")
        plt.close(fig)

    return saved_paths


def save_json(
    data: dict[str, Any],
    path: Union[str, Path],
    indent: int = 2,
    verbose: bool = True,
) -> Path:
    """
    Save a dictionary as a pretty-printed JSON file.

    Used to persist experiment metadata, configuration snapshots, and
    reproducibility state alongside result CSVs.

    Parameters
    ----------
    data : dict
        JSON-serialisable dictionary. Values must be Python primitives
        (str, int, float, bool, list, dict, None).

    path : str or Path
        Destination file path. Parent directories created automatically.

    indent : int
        JSON indentation level. Default 2 (compact but readable).

    verbose : bool
        If True, log the save path.

    Returns
    -------
    Path
        Resolved absolute path of the saved JSON file.

    Examples
    --------
    >>> from src.utils import save_json
    >>> save_json({"seed": 42, "model": "gpt2"}, "outputs/results/metadata.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serialisable types gracefully
    def _default_serialiser(obj: Any) -> Any:
        if hasattr(obj, "item"):     # numpy scalar → Python primitive
            return obj.item()
        if hasattr(obj, "tolist"):   # numpy array / torch tensor
            return obj.tolist()
        return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_default_serialiser)

    if verbose:
        logger.info(f"[save_json] ✓ Saved metadata → {path.resolve()}")
    return path.resolve()


def timestamped_path(base_dir: Union[str, Path], name: str, ext: str) -> Path:
    """
    Generate a timestamped file path to avoid overwriting previous runs.

    Parameters
    ----------
    base_dir : str or Path
        Parent directory.

    name : str
        Base filename stem (no extension).

    ext : str
        File extension (with or without leading dot).

    Returns
    -------
    Path
        e.g., Path("outputs/results/ioi_results_20240101_120000.csv")

    Examples
    --------
    >>> p = timestamped_path("outputs/results", "ioi_results", "csv")
    >>> print(p.name)   # "ioi_results_20240101_120000.csv"
    """
    ext = ext.lstrip(".")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / f"{name}_{ts}.{ext}"
