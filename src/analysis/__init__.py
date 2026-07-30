"""
src/analysis — Mechanistic Interpretability Analysis Sub-Package
=================================================================
Exports all circuit analysis modules for CircuitScope.

Modules
-------
logit_lens           : Project residual stream to vocab space per layer
layer_ablation       : Mean-ablate every layer (attn + MLP) separately
head_ablation        : Causal importance scoring for all 144 attention heads
activation_patching  : Clean→corrupted activation patching experiments
path_patching        : Trace information flow between attention heads
circuit_validation   : Necessity, sufficiency, and generalization tests
statistics           : Bootstrap CI, effect sizes, correlations
"""

from .logit_lens import LogitLensAnalyzer
from .layer_ablation import LayerAblationAnalyzer
from .head_ablation import HeadAblationAnalyzer
from .activation_patching import ActivationPatchingAnalyzer
from .path_patching import PathPatchingAnalyzer
from .circuit_validation import CircuitSpec, CircuitValidator, ValidationResult
from .statistics import (
    bootstrap_ci,
    cohen_d,
    classify_effect_size,
    permutation_test,
    layer_depth_correlation,
    compute_comprehensive_stats,
    ioi_vs_pronoun_comparison,
)

__all__ = [
    "LogitLensAnalyzer",
    "LayerAblationAnalyzer",
    "HeadAblationAnalyzer",
    "ActivationPatchingAnalyzer",
    "PathPatchingAnalyzer",
    "CircuitSpec",
    "CircuitValidator",
    "ValidationResult",
    "bootstrap_ci",
    "cohen_d",
    "classify_effect_size",
    "permutation_test",
    "layer_depth_correlation",
    "compute_comprehensive_stats",
    "ioi_vs_pronoun_comparison",
]

