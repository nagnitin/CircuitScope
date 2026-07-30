"""
src.model — Model loading sub-package
======================================
Exports the primary `load_model` factory function.
"""
from .loader import load_model, get_device

__all__ = ["load_model", "get_device"]
