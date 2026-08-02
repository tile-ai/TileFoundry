"""Which functions a Module counts as its own."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from tests.fixtures.gqa_online import SMALL_CONTEXT_T, GqaOnline
from tilefoundry import func, module
from tilefoundry.analysis import analyze
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.dsl import Tensor, Topology, tf
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.specialize import (
    PROVENANCE,
    origin_of,
    specialize_function,
    variant_for,
)
from tilefoundry.schedule.partition.program import (
    PartitionProgramError,
    build_partition_program,
)
from tilefoundry.target import CudaTarget

DIMS = {"ctx_len": SMALL_CONTEXT_T}


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
        GqaOnline, target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 8),)
    )
    variant = variant_for(aimed.entry_function(), DIMS)

    assert aimed.owns(variant)
    with pytest.raises(ValueError, match="ctx_len.*is not concrete"):
        analyze(aimed, variant, analysis="compute-cost")


def test_analysing_a_function_from_elsewhere_is_still_refused() -> None:
    aimed = replace(
        GqaOnline, target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 8),)
    )

    with pytest.raises(AnalysisError, match="is not a function of module"):
        analyze(aimed, _Other.entry_function(), analysis="compute-cost")


def test_a_same_named_stranger_cannot_reach_an_algorithm_directly() -> None:
    """The precondition inside an algorithm follows a recorded origin, not a
    name. A name is shared by anything anybody chose to call the same, so
    answering by name would let a function from elsewhere in through the door
    the public boundary guards."""
    stranger = replace(
        _Other.entry_function(), name=GqaOnline.entry_function().name
    )

    assert stranger.name == GqaOnline.entry_function().name
    assert not GqaOnline.owns(stranger)
    assert not GqaOnline.owns(stranger, derived=True)

    with pytest.raises(PartitionProgramError, match="is not a function of module"):
        build_partition_program(GqaOnline, stranger)


def test_a_genuinely_derived_function_does_reach_it() -> None:
    derived = specialize_function(GqaOnline.entry_function(), DIMS)

    assert origin_of(derived) is variant_for(GqaOnline.entry_function(), DIMS)
    assert not GqaOnline.owns(derived)
    assert GqaOnline.owns(derived, derived=True)


def test_provenance_is_not_part_of_what_makes_a_function_what_it_is() -> None:
    """Where a function came from is recorded on it, not built into it. If it
    were a field, two functions specialised from different places would stop
    being equal even when they are the same program."""
    assert PROVENANCE not in {field.name for field in fields(Function)}

    derived = specialize_function(GqaOnline.entry_function(), DIMS)
    assert origin_of(derived) is not None
    assert origin_of(GqaOnline.entry_function()) is None


def test_a_copy_of_an_owned_function_is_not_owned() -> None:
    """A Function compares by structure, so a copy is equal to the original and
    is not the original. Answering ownership by equality lets the public
    boundary analyse a program the Module does not contain."""
    module = replace(
        GqaOnline, target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 8),)
    )
    owned = module.lookup("_ctx_partials")
    clone = replace(owned)

    assert clone is not owned
    assert clone == owned
    assert not module.owns(clone)
    assert not module.owns(clone, derived=True)

    with pytest.raises(AnalysisError, match="is not a function of module"):
        analyze(module, clone, analysis="compute-cost")


def test_a_copy_of_a_variant_is_not_owned_either() -> None:
    """The same rule at the level a specialisation is reached through."""
    prototype = GqaOnline.entry_function()
    clone = replace(prototype.variants[0])

    assert clone == prototype.variants[0]
    assert not GqaOnline.owns(clone)
    assert not GqaOnline.owns(clone, derived=True)
