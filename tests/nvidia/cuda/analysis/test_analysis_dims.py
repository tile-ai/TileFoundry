"""Concrete dimensions select and preserve the checked analysis view."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.core.fixtures.placed.gqa_decode import GqaOnline
from tests.nvidia.cuda.models.qwen3_1_7b.case import CASE as QWEN3_1_7B
from tilefoundry.analysis import (
    ComputeCostMetadata,
    analyze,
)
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.ir.core import get_metadata
from tilefoundry.ir.hir.specialize import (
    SpecializationError,
    origin_of,
    residual_dims,
    variant_for,
)
from tilefoundry.ir.types import Topology
from tilefoundry.visitor_registry.contexts import ValuationContext
from tilefoundry.visitor_registry.valuation import ValuationVisitor
from tilefoundry_nvidia import CudaTarget

CONTEXT = 32


DIMS = {"ctx_len": CONTEXT}


FAMILIES = ("compute-cost", "memory", "roofline", "performance")


def _aimed():
    """The decode example, aimed at one machine."""
    return replace(
        GqaOnline, target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 8),)
    )


def test_analyze_at_stated_size() -> None:
    module = _aimed()
    authored = module.entry_function()
    checked, function = check_program(module, authored, dims=DIMS)
    result = analyze(checked, function, analysis=FAMILIES)
    assert result.metadata_types
    assert result.module is checked
    assert result.function is not authored
    assert result.function.name == authored.name
    assert all(
        residual_dims(frame.function, frame) == ()
        for frame in ValuationVisitor.definitions(ValuationContext(result.module, result.function))
    )


def test_static_view_records() -> None:
    """A static input remains authored while its analysis view carries records."""
    module = QWEN3_1_7B.build()
    function = module.lookup("mlp")

    checked, derived = check_program(module, function)
    result = analyze(checked, derived, analysis="compute-cost")

    assert result.module is checked
    assert result.function is derived
    assert result.function.name == function.name
    assert origin_of(result.function) is function
    assert get_metadata(result.function, ComputeCostMetadata) is not None
    assert get_metadata(function, ComputeCostMetadata) is None


def test_unknown_dim_refused() -> None:
    module = _aimed()

    with pytest.raises(SpecializationError, match="no dimension named"):
        check_program(module, module.entry_function(), dims={**DIMS, "batch": 2})


def test_unbound_dim_refused() -> None:
    """Test a dimension left unbound is refused.

    Stating some other dimension is useful while the choices are being made
    and useless to an analysis, which would meet the unbound one as an extent
    that is not a number.
    """
    module = _aimed()

    with pytest.raises(SpecializationError, match="was not given a size"):
        check_program(module, module.entry_function(), dims={"batch": 4})


def test_malformed_dims_refused() -> None:
    """A caller who believes they stated a size must not be left believing it."""
    module = _aimed()
    entry = module.entry_function()

    with pytest.raises(SpecializationError, match="non-empty mapping"):
        check_program(module, entry, dims={})
    with pytest.raises(SpecializationError, match="takes an integer extent"):
        check_program(module, entry, dims={"ctx_len": 32.0})


def test_foreign_function_refused() -> None:
    """Ownership is settled before a size is looked at.

    Ownership is settled before a size is looked at, so a foreign function
    is refused for being foreign rather than for its dimensions.
    """
    module = _aimed()
    foreign = QWEN3_1_7B.build().lookup("mlp")

    with pytest.raises(AnalysisError, match="is owned by no single node of module"):
        check_program(module, foreign, dims=DIMS)


def test_entry_keeps_name() -> None:
    """Choosing a size does not rename the entry.

    A function specialised from the entry is a different object and the same
    program, so anything that identifies the entry by name still finds it.
    """
    module = _aimed()
    variant = variant_for(module.entry_function(), DIMS)

    assert variant.name == module.entry_function().name
