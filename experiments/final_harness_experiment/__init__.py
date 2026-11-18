"""Gate-controlled final harness experiment.

The confirmatory runner deliberately refuses to start while Gate A is blocked.
Unit and deterministic smoke paths remain available.
"""

from .optimizers import SequentialRLOptimizer, TailAwareRLOptimizer

__all__ = ["SequentialRLOptimizer", "TailAwareRLOptimizer"]
