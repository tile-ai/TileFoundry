"""Per-op evaluator registry: ``@register_eval(OpClass)`` handlers keyed by Op class."""
from __future__ import annotations

from typing import Callable

from tilefoundry.visitor_registry.registries import AnalysisRegistry

eval_registry: AnalysisRegistry = AnalysisRegistry("eval")


def register_eval(op_cls: type) -> Callable[[Callable], Callable]:
    """Register *fn* as the evaluator for ``op_cls``."""
    return eval_registry.decorator()(op_cls)
