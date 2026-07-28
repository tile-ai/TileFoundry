"""The Target Facts projection boundary: exact registration, the two call
shapes, and what the projection is not allowed to touch.

The Facts aggregates here are local to the test on purpose. A production
algorithm declares and registers its own, so inventing one before an algorithm
needs it would fix a shape no caller has asked for yet.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace

import pytest

from tilefoundry.target import CpuTarget
from tilefoundry.target.amx import AmxTarget
from tilefoundry.target.cuda import CudaTarget
from tilefoundry.target.facts import (
    TARGET_FACTS,
    DuplicateFactsConversionError,
    InvalidFactsTypeError,
    TargetFactsError,
    TargetFactsRegistry,
    UnknownFactsConversionError,
    register_target_facts,
)


@dataclass(frozen=True)
class _HardwareFacts:
    """A hardware-only aggregate: everything it holds the target already knows."""

    parallel_units: int
    bytes_per_second: int


@dataclass(frozen=True)
class _ShapeQuery:
    """One algorithm's private query. No common query base exists to share."""

    tile_elements: int


@dataclass(frozen=True)
class _ShapedFacts:
    """An aggregate whose value depends on the requesting algorithm's query."""

    parallel_units: int
    tiles_per_unit: int


def _hardware_facts(target: CudaTarget, query: object) -> _HardwareFacts:
    """A hardware-only conversion: it requires no query and rejects one."""
    if query is not None:
        raise TargetFactsError(
            f"_HardwareFacts is a hardware-only projection and takes no query, "
            f"got {type(query).__name__}"
        )
    return _HardwareFacts(
        parallel_units=target.device.sm_count,
        bytes_per_second=target.device.hbm_bandwidth_bytes_per_second,
    )


def _shaped_facts(target: CudaTarget, query: object) -> _ShapedFacts:
    """A program-dependent conversion validating its own private query type."""
    if not isinstance(query, _ShapeQuery):
        raise TargetFactsError(
            f"_ShapedFacts requires a _ShapeQuery, got {type(query).__name__}"
        )
    capacity = target.architecture.shared_memory_per_cta_bytes
    return _ShapedFacts(
        parallel_units=target.device.sm_count,
        tiles_per_unit=capacity // query.tile_elements,
    )


def _registry() -> TargetFactsRegistry:
    """A registry with both conversion shapes registered for CUDA."""
    registry = TargetFactsRegistry()
    registry.register(CudaTarget, _HardwareFacts, _hardware_facts)
    registry.register(CudaTarget, _ShapedFacts, _shaped_facts)
    return registry


def test_lookup_is_exact_with_no_subclass_or_by_name_fallback() -> None:
    """AC-2-1 and AC-2-2. A pair is unknown until its own registration has run,
    and known only by identity afterwards.

    Which is what makes a missing backend registration a hard failure rather than
    a silent default. Two targets sharing a base can describe different hardware,
    so a base registration must not serve a subclass; a Facts type is identified
    by the class itself, never by its name; and registering the same pair twice is
    refused rather than replacing, since replacing would make the projection
    depend on import order.
    """
    empty = TargetFactsRegistry()
    assert empty.registered_pairs() == ()
    with pytest.raises(UnknownFactsConversionError, match="no Facts conversion"):
        empty.project(CudaTarget(), _HardwareFacts)

    registry = _registry()
    assert ("CudaTarget", "_HardwareFacts") in registry.registered_pairs()
    assert registry.project(CudaTarget(), _HardwareFacts) == _HardwareFacts(
        parallel_units=132, bytes_per_second=4_800_000_000_000
    )
    with pytest.raises(DuplicateFactsConversionError, match="already registered"):
        registry.register(CudaTarget, _HardwareFacts, _hardware_facts)

    class _TunedCuda(CudaTarget):
        """A distinct target type that happens to share CudaTarget's base."""

    with pytest.raises(UnknownFactsConversionError, match="_TunedCuda"):
        registry.project(_TunedCuda(), _HardwareFacts)

    # A different target family is likewise unregistered rather than coerced.
    with pytest.raises(UnknownFactsConversionError, match="AmxTarget"):
        registry.project(AmxTarget(), _HardwareFacts)
    with pytest.raises(UnknownFactsConversionError, match="CpuTarget"):
        registry.project(CpuTarget(), _HardwareFacts)

    # Same name, different class: resolution is by identity, not by name.
    @dataclass(frozen=True)
    class _HardwareFactsImpostor:
        parallel_units: int
        bytes_per_second: int

    _HardwareFactsImpostor.__name__ = "_HardwareFacts"
    with pytest.raises(UnknownFactsConversionError):
        registry.project(CudaTarget(), _HardwareFactsImpostor)


def test_both_call_shapes_work_without_a_common_query_base() -> None:
    """AC-2-3. A hardware-only projection takes no query; a program-dependent
    one validates its own private query type. Neither shares a query base."""
    registry = _registry()
    target = CudaTarget()

    hardware = registry.project(target, _HardwareFacts)
    assert hardware.parallel_units == 132

    shaped = registry.project(target, _ShapedFacts, query=_ShapeQuery(tile_elements=1024))
    assert shaped.tiles_per_unit == 232448 // 1024

    # Each conversion rejects the other's call shape.
    with pytest.raises(TargetFactsError, match="takes no query"):
        registry.project(target, _HardwareFacts, query=_ShapeQuery(tile_elements=8))
    with pytest.raises(TargetFactsError, match="requires a _ShapeQuery"):
        registry.project(target, _ShapedFacts)

    # The two queries share no base beyond object.
    assert _ShapeQuery.__mro__[1:] == (object,)


def test_a_facts_aggregate_must_be_an_immutable_dataclass() -> None:
    """AC-2-4. Immutability is checked when the conversion is registered, so a
    mutable aggregate cannot reach a caller in the first place -- and the value
    that does reach one is frozen, because projection is a read: it converts what
    the target already knows and reports it, leaving the target and the registry
    exactly as they were.
    """
    projected = _registry()
    target = CudaTarget()
    before_target = replace(target)
    before_pairs = projected.registered_pairs()

    facts = projected.project(target, _HardwareFacts)

    assert target == before_target
    assert target.architecture_id == "nvidia.sm90"
    assert projected.registered_pairs() == before_pairs
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts.parallel_units = 1

    registry = TargetFactsRegistry()

    @dataclass
    class _Mutable:
        parallel_units: int

    class _NotADataclass:
        pass

    with pytest.raises(InvalidFactsTypeError, match="must be a frozen dataclass"):
        registry.register(CudaTarget, _Mutable, _hardware_facts)
    with pytest.raises(InvalidFactsTypeError, match="must be a dataclass"):
        registry.register(CudaTarget, _NotADataclass, _hardware_facts)
    with pytest.raises(InvalidFactsTypeError, match="must be a class"):
        registry.register(CudaTarget, "_HardwareFacts", _hardware_facts)


def test_a_conversion_returning_the_wrong_type_is_rejected() -> None:
    """A caller asked for one aggregate; handing back another would put the
    mismatch inside the algorithm rather than at the boundary."""
    registry = TargetFactsRegistry()
    registry.register(
        CudaTarget, _HardwareFacts, lambda target, query: _ShapeQuery(tile_elements=1)
    )
    with pytest.raises(TargetFactsError, match="returned _ShapeQuery"):
        registry.project(CudaTarget(), _HardwareFacts)


def test_as_facts_delegates_to_the_shared_registry() -> None:
    """The public boundary on Target is the same projection, so an algorithm
    never reaches for the registry itself."""

    @dataclass(frozen=True)
    class _SharedFacts:
        parallel_units: int

    register_target_facts(
        CudaTarget, _SharedFacts, lambda target, query: _SharedFacts(target.device.sm_count)
    )
    try:
        assert CudaTarget().as_facts(_SharedFacts) == _SharedFacts(132)
        with pytest.raises(UnknownFactsConversionError):
            AmxTarget().as_facts(_SharedFacts)
    finally:
        # There is deliberately no public unregister: a conversion is installed
        # for the process, so only this test's own locally-defined pair is undone.
        TARGET_FACTS._conversions.pop((CudaTarget, _SharedFacts))
