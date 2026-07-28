"""Which functions a Module counts as its own."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.models.qwen3_5_30b_a3b.gqa_online import SMALL_CONTEXT_T, GqaOnline
from tilefoundry import func, module
from tilefoundry.analysis import analyze
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.dsl import Tensor, Topology, tf
from tilefoundry.ir.hir.specialize import variant_for
from tilefoundry.target import CudaTarget

DIMS = {"ctx_len": SMALL_CONTEXT_T, "seq_len": 1}


@module(entry="main", target="cuda")
class _Other:
    topologies = (Topology("cta", 1),)

    @func
    def main(source: Tensor[(8,), "f32"]):
        return tf.add(source, source)


def test_a_module_owns_the_functions_it_lists() -> None:
    assert GqaOnline.owns(GqaOnline.entry_function())
    assert GqaOnline.owns(GqaOnline.lookup("_ctx_partials"))


def test_a_module_owns_the_variants_of_its_functions() -> None:
    """A variant is reached through the prototype that dispatches to it rather
    than listed beside it, and it is what anything working at one chosen size
    holds. A module that disowned its own variants would disown all of them."""
    prototype = GqaOnline.entry_function()

    for variant in prototype.variants:
        assert GqaOnline.owns(variant)
    assert GqaOnline.owns(variant_for(prototype, DIMS))


def test_a_module_does_not_own_another_module_s_function() -> None:
    assert not GqaOnline.owns(_Other.entry_function())
    assert not _Other.owns(GqaOnline.entry_function())


def test_analysing_a_variant_gets_past_ownership() -> None:
    """The rejection a variant used to get was about membership. What stops it
    now is the thing that actually stops it: a variant still states its context
    length as a range, and an analysis has no answer for a range."""
    aimed = replace(
        GqaOnline, target=CudaTarget(), topologies=(Topology("cta", 8),)
    )
    variant = variant_for(aimed.entry_function(), DIMS)

    assert aimed.owns(variant)
    with pytest.raises(ValueError, match="not a concrete positive integer"):
        analyze(aimed, variant, analysis="compute-cost")


def test_analysing_a_function_from_elsewhere_is_still_refused() -> None:
    aimed = replace(
        GqaOnline, target=CudaTarget(), topologies=(Topology("cta", 8),)
    )

    with pytest.raises(AnalysisError, match="is not a function of module"):
        analyze(aimed, _Other.entry_function(), analysis="compute-cost")
