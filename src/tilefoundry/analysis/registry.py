"""Analysis algorithm registration: dependencies and Metadata ownership.

An analysis declares what it needs and what it owns. The dependency names are
resolved under the same target as the root, so a closure never mixes
implementations from different hardware, and the owned Metadata types are what
the algorithm is permitted to replace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tilefoundry.ir.core import IRMetadata
from tilefoundry.registry import AlgorithmRegistry

# An algorithm runs over the selected Module and Function against one target.
AnalysisCallable = Callable[..., Any]


@dataclass(frozen=True)
class AnalysisAlgorithm:
    """One registered analysis: its identity, needs, and owned Metadata."""

    selector: str
    run: AnalysisCallable
    requires: tuple[str, ...] = ()
    produces: tuple[type[IRMetadata], ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.selector:
            raise ValueError("an analysis needs a non-empty selector")
        if len(set(self.requires)) != len(self.requires):
            raise ValueError(
                f"{self.selector}: duplicate entries in requires {list(self.requires)}"
            )
        if self.selector in self.requires:
            raise ValueError(f"{self.selector}: an analysis cannot require itself")
        for produced in self.produces:
            if not isinstance(produced, type) or not issubclass(produced, IRMetadata):
                raise ValueError(
                    f"{self.selector}: produces must name IRMetadata subclasses, "
                    f"got {produced!r}"
                )
        if len(set(self.produces)) != len(self.produces):
            raise ValueError(
                f"{self.selector}: the same Metadata type is produced twice"
            )


ANALYSES: AlgorithmRegistry[AnalysisAlgorithm] = AlgorithmRegistry("analyze")


def register_analysis(
    target_type: type,
    selector: str,
    *,
    requires: tuple[str, ...] = (),
    produces: tuple[type[IRMetadata], ...] = (),
) -> Callable[[AnalysisCallable], AnalysisCallable]:
    """Register an analysis for one exact target under *selector*.

    A target-independent analysis is registered once per supported target
    rather than inherited, so the support matrix stays exact.
    """

    def bind(run: AnalysisCallable) -> AnalysisCallable:
        ANALYSES.register(
            target_type,
            selector,
            AnalysisAlgorithm(
                selector=selector,
                run=run,
                requires=tuple(requires),
                produces=tuple(produces),
            ),
        )
        return run

    return bind


__all__ = [
    "ANALYSES",
    "AnalysisAlgorithm",
    "AnalysisCallable",
    "register_analysis",
]
