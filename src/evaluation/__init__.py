"""
src.evaluation — Metric computation sub-package
================================================
Exports the `IOIEvaluator` class and standalone metric helpers.
"""
from .metrics import IOIEvaluator, compute_logit_diff, compute_top_k

__all__ = ["IOIEvaluator", "compute_logit_diff", "compute_top_k"]
