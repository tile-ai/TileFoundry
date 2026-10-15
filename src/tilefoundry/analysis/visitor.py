"""The common inputs and lexical position used by analysis families."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.visitor_registry.contexts import CostContext

from .errors import AnalysisError


@dataclass
class AnalyzeContext(CostContext):
    """Read checked IR in one definition's local iteration scope."""

    options: object | None = None
    root: IterationScope | None = None
    current: IterationScope | None = None

    @property
    def module(self):
        return self.frame.module

    @property
    def target(self):
        return self.resolve_target()

    def for_region(self, region):
        return replace(
            super().for_region(region),
            current=next(child for child in self.current.children if child.owner is region),
        )

    def error(self, node, message):
        raise AnalysisError(message)


__all__ = ["AnalyzeContext"]
