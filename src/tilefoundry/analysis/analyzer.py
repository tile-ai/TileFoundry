"""The boolean-selected authored-HIR entry, as a facade over the real one.

This entry exists only so the command line keeps working while it is rewritten
onto the composed operation. It measures nothing: it maps its option flags onto
root selectors, invokes the public operation once per root, and renders the
records the families left on the IR.

Everything in this module is scheduled for removal along with the option flags
and the rendered result fields.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import IRMetadata, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function

from .api import analyze as composed_analyze
from .errors import AnalysisError
from .metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    RooflineMetadata,
    TimelineMetadata,
    TrafficBytes,
)
from .walk import entry_function, postorder

# Which root selector each legacy flag asks for. The flag named after the
# footprint asks for the memory family, which is what absorbed it.
_ROOTS = (("roofline", "roofline"), ("footprint", "memory"), ("timeline", "timeline"))


@dataclass(frozen=True)
class AnalysisOptions:
    """Which analyses one legacy call selects."""

    roofline: bool = True
    footprint: bool = True
    timeline: bool = True

    @property
    def selectors(self) -> tuple[str, ...]:
        """The root selectors these flags ask for, in a stable order."""
        return tuple(
            selector for flag, selector in _ROOTS if getattr(self, flag)
        )


@dataclass(frozen=True)
class AnalysisResult:
    """The annotated IR, a rendered summary, and the records that were written."""

    ir: Module | Function
    summary_lines: tuple[str, ...]
    metadata_types: tuple[type[IRMetadata], ...]


def _flop_totals(function: Function) -> tuple[tuple[str, int], ...]:
    """The function's flops, summed from the per-Call work records.

    Compute cost has no function-level record because this sum is all there is
    to it, so summing here is a rendering step rather than a second analysis.
    """
    totals: dict[str, int] = {}
    for expr in postorder(function.body):
        record = get_metadata(expr, ComputeCostMetadata)
        if record is None:
            continue
        for name, value in record.flops:
            totals[name] = totals.get(name, 0) + value
    return tuple(sorted(totals.items()))


def _traffic_text(traffic: tuple[tuple[str, TrafficBytes], ...]) -> str:
    return (
        ", ".join(
            f"{level}=r{value.read_bytes}/w{value.write_bytes}"
            for level, value in traffic
        )
        or "0"
    )


def _summary(
    target: object, function: Function, options: AnalysisOptions
) -> tuple[str, ...]:
    """One stable line per selected measurement, read off the IR."""
    lines = [
        f"analysis target={getattr(target, 'name', type(target).__name__)} "
        f"analyses={','.join(options.selectors)}"
    ]
    if options.roofline:
        flops = _flop_totals(function)
        lines.append(
            "flops " + (", ".join(f"{name}={value}" for name, value in flops) or "0")
        )
        bound = get_metadata(function, RooflineMetadata)
        if bound is not None:
            lines.append(
                f"theoretical-bound={bound.theoretical_ns}ns by={bound.bound_by}"
            )
    # A selected root pulls its dependencies in, so a record can be on the IR
    # without its own flag being selected. What is rendered follows the flags
    # rather than what happens to be there: otherwise selecting the roofline
    # silently reports the memory measurement too.
    memory = get_metadata(function, MemoryMetadata) if options.footprint else None
    if memory is not None:
        lines.append("traffic " + _traffic_text(memory.traffic))
        lines.append(
            "peak-footprint "
            + (
                ", ".join(
                    f"{item.level}={item.peak_bytes}" for item in memory.footprint
                )
                or "0"
            )
        )
        lines.extend(f"advisory {note}" for note in memory.advisories)
    if options.timeline:
        placement = get_metadata(function, TimelineMetadata)
        makespan = 0 if placement is None else placement.end_ns
        lines.append(f"theoretical-makespan={makespan}ns")
    return tuple(lines)


def analyze(
    ir: Module,
    *,
    options: AnalysisOptions | None = None,
) -> AnalysisResult:
    """Run the selected analyses over *ir*'s entry function.

    One root per selected flag, because the composed operation takes one root
    per call. A root whose closure includes another selected root recomputes it,
    which is the cost of asking three separate questions.
    """
    if not isinstance(ir, Module):
        raise TypeError(
            f"analyze: expected a Module, got {type(ir).__name__}. A Function "
            "carries no execution context; select the Module that owns it."
        )
    options = options or AnalysisOptions()
    function = entry_function(ir)
    written: list[type[IRMetadata]] = []
    for selector in options.selectors:
        result = composed_analyze(ir, function, analysis=selector)
        for metadata_type in result.metadata_types:
            if metadata_type not in written:
                written.append(metadata_type)
    return AnalysisResult(
        ir, _summary(ir.resolve_target(), function, options), tuple(written)
    )


__all__ = [
    "AnalysisError",
    "AnalysisOptions",
    "AnalysisResult",
    "analyze",
]
