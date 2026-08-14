"""What a Target validates about a program, which scheduler it reaches.

What a Target validates about a program, which scheduler it reaches, and how
codegen groups by one.

The composed hardware facts themselves -- which documents a default target
resolves, and which limits belong to the architecture rather than the device --
are asserted where those facts are installed. What is left here is what a Target
is asked during a compilation.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, replace

import pytest

from tests.fixtures.placed.moe_mega_kernel import MoEMegaKernel
from tests.fixtures.placed.rmsnorm import RmsnormModule
from tests.fixtures.placed.square_cuda import Model as SquareCudaModel
from tests.fixtures.shapes.matmul_programs import scheduling_gemm
from tests.installed.smoke_target.vendor_npu import VendorNpuTarget
from tilefoundry import CompilerOptions, DType, build, jit, lower, module
from tilefoundry.analysis import AnalysisError, analyze
from tilefoundry.codegen.registry import group_functions_by_target
from tilefoundry.dsl import DimVar
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule import ScheduleError, schedule
from tilefoundry.schedule.plan import SchedulePlan
from tilefoundry.target import (
    AmxTarget,
    CpuTarget,
    CudaTarget,
    MemoryHierarchyFacts,
    ParallelCapacityFacts,
    Target,
    TargetFactsError,
    ThroughputFacts,
    TopologyLimitFacts,
    UnsupportedCapabilityError,
    facts_result,
    register_target,
    registered_targets,
    validate_cuda_topology_levels,
)
from tilefoundry.target.cuda.spec import SM90_ID
from tilefoundry.target.services import CodeGenerator, Scheduler


class ExternalCudaTarget(CudaTarget):
    name = "tests.target.external_cuda"


class _NoComputeUnitRateCudaTarget(CudaTarget):
    name = "tests.target.no_compute_unit_rate_cuda"

    def get_facts(self, facts_type: type, query: object | None = None):
        facts = super().get_facts(facts_type, query)
        if facts_type is ThroughputFacts:
            return replace(facts, peak_flops_per_second_per_unit=())
        return facts


class _NoBandwidthUnitRateCudaTarget(CudaTarget):
    name = "tests.target.no_bandwidth_unit_rate_cuda"

    def get_facts(self, facts_type: type, query: object | None = None):
        facts = super().get_facts(facts_type, query)
        if facts_type is ThroughputFacts:
            return replace(
                facts, memory_bandwidth_bytes_per_second_per_unit=None
            )
        return facts


@dataclass(frozen=True)
class _CustomSchedulePlan(SchedulePlan):
    topology: str

    def verify(self, module, function, topology) -> None:
        assert topology.name == self.topology

    def to_json(self) -> str:
        return self.topology

    def render(self) -> str:
        return self.topology


_CUSTOM_SOLVES: list[str] = []


def _solve_custom_topology(module, function, target, topology, options):
    _CUSTOM_SOLVES.append(topology.name)
    return _CustomSchedulePlan(topology.name)


class CustomSchedulerCudaTarget(CudaTarget):
    name = "tests.target.custom_scheduler_cuda"
    topology_levels = (*CudaTarget.topology_levels, "custom", "unknown")

    def get_facts(self, facts_type: type, query: object | None = None):
        if facts_type is TopologyLimitFacts and query in {"custom", "unknown"}:
            return TopologyLimitFacts(query, 1)
        return super().get_facts(facts_type, query)

    def get_scheduler(self, topology: str) -> Scheduler:
        if topology == "custom":
            return Scheduler("custom", _solve_custom_topology)
        return super().get_scheduler(topology)


class RefusingCudaTarget(CustomSchedulerCudaTarget):
    name = "tests.target.refusing_cuda"

    def get_scheduler(self, topology: str) -> Scheduler:
        if topology == "thread":
            raise UnsupportedCapabilityError(
                f"{type(self).__name__} ({type(self).name}): no scheduler for {topology!r}"
            )
        return super().get_scheduler(topology)


def _provider_target(module_name: str, provider_name: str, registered_name: str):
    return type(
        provider_name,
        (Target,),
        {
            "__module__": module_name,
            "__qualname__": provider_name,
            "name": registered_name,
        },
    )


def test_registration_is_one_class_boundary_and_reload_is_idempotent() -> None:
    registered_name = "tests.target.reload"
    first = _provider_target("tests.providers.reload", "ReloadTarget", registered_name)
    second = _provider_target("tests.providers.reload", "ReloadTarget", registered_name)

    assert register_target(first) is first
    assert register_target(second) is second
    assert registered_targets()[registered_name] is first
    assert isinstance(second(), Target)
    assert {"cpu", "cuda", "amx"} <= registered_targets().keys()
    with pytest.raises(TypeError):
        registered_targets()["tests.target.mutation"] = Target


def test_target_registration_and_service_annotations_resolve() -> None:
    assert typing.get_type_hints(register_target)
    assert typing.get_type_hints(PrimFunction)
    for getter in (
        Target.get_analyzer,
        Target.get_scheduler,
        Target.get_code_generator,
    ):
        assert typing.get_type_hints(getter)


def test_document_free_target_projects_facts_and_inherits_standard_analysis() -> None:
    target = VendorNpuTarget()

    assert target.get_analyzer("roofline").selector == "roofline"
    assert target.get_facts(TopologyLimitFacts, "core") == TopologyLimitFacts(
        "core", 256
    )
    memory = target.get_facts(MemoryHierarchyFacts)
    assert memory.explicit("gmem").capacity_bytes == 64_000_000_000
    assert target.get_facts(ThroughputFacts).peak_for(DType.f32) == 2_000_000_000_000_000
    assert target.get_facts(ParallelCapacityFacts) == ParallelCapacityFacts("core", 16)
    target.validate_program_topology(Topology("core", 256))
    with pytest.raises(ValueError, match="1 <= extent <= 256"):
        target.validate_program_topology(Topology("core", 257))


def test_cuda_projects_one_ctas_share_from_device_rates() -> None:
    target = CudaTarget("nvidia.h200_sxm")
    facts = target.get_facts(ThroughputFacts)

    assert facts.rate_unit == "cta"
    assert dict(facts.peak_flops_per_second_per_unit) == {
        dtype: rate // target.device.sm_count
        for dtype, rate in facts.peak_flops_per_second
    }
    assert facts.memory_bandwidth_bytes_per_second_per_unit == (
        target.device.hbm_bandwidth_bytes_per_second // target.device.sm_count
    )


@pytest.mark.parametrize(
    ("target_type", "missing"),
    (
        (_NoComputeUnitRateCudaTarget, r"dtype 'f32'.*'cta'"),
        (_NoBandwidthUnitRateCudaTarget, r"level 'gmem'.*'cta'"),
    ),
)
def test_timeline_refuses_a_target_without_its_required_one_unit_rate(
    target_type: type[CudaTarget], missing: str
) -> None:
    target = target_type("nvidia.h200_sxm")
    subject = replace(MoEMegaKernel, target=target)
    function = subject.entry_function()

    with pytest.raises(AnalysisError, match=missing):
        analyze(subject, function, analysis="timeline")


def test_document_free_target_enforces_projection_and_capability_boundaries() -> None:
    class BrokenFactsTarget(Target):
        name = "tests.target.broken_facts"

        def get_facts(self, facts_type: type, query: object | None = None):
            return facts_result(self, facts_type, object())

    target = VendorNpuTarget()

    with pytest.raises(
        TargetFactsError,
        match="BrokenFactsTarget: Facts projection for ThroughputFacts returned object",
    ):
        BrokenFactsTarget().get_facts(ThroughputFacts)
    with pytest.raises(UnsupportedCapabilityError) as error:
        target.get_code_generator()
    assert str(error.value) == "VendorNpuTarget (vendor.npu): no code generator"


def test_cuda_mesh_topology_validation_uses_the_emission_target() -> None:
    custom = CustomSchedulerCudaTarget("nvidia.h200_sxm")

    validate_cuda_topology_levels(custom, ("custom",))
    with pytest.raises(ValueError, match=r"supports \{cta, thread, custom, unknown\}"):
        validate_cuda_topology_levels(custom, ("warp",))


def test_registration_rejects_ambiguous_or_invalid_provider_classes() -> None:
    inherited = type(
        "InheritedTarget",
        (CudaTarget,),
        {"__module__": "tests.providers.inherited"},
    )
    empty = _provider_target("tests.providers.empty", "EmptyTarget", "")

    with pytest.raises(ValueError, match="declare a class-level"):
        register_target(inherited)
    with pytest.raises(ValueError, match="non-empty"):
        register_target(empty)
    with pytest.raises(TypeError, match="Target subclass"):
        register_target(object)
    with pytest.raises(TypeError, match="Target subclass"):
        register_target("tests.target.argument")

    owner = _provider_target("tests.providers.owner", "OwnedTarget", "tests.target.conflict")
    claimant = _provider_target(
        "tests.providers.claimant", "ClaimantTarget", "tests.target.conflict"
    )
    register_target(owner)
    with pytest.raises(ValueError, match="already owned.*cannot register"):
        register_target(claimant)


def test_authored_target_boundaries_reject_strings_and_keep_exact_instances() -> None:
    target = CudaTarget("nvidia.h200_sxm")
    module = Module("exact", (), target=target)

    assert module.target is target
    assert module.resolve_target() is target
    with pytest.raises(TypeError, match="Target instance, not a string"):
        Module("invalid", (), target="cuda")
    with pytest.raises(TypeError, match="Target instance, not a string"):
        PrimFunction("invalid", (), Sequential(()), target="cuda")


def test_authored_target_boundaries_accept_unregistered_target_instances() -> None:
    class UnregisteredTarget(Target):
        name = "tests.target.unregistered"

    target = UnregisteredTarget()
    module_value = Module("unregistered", (), target=target)
    function = PrimFunction("unregistered", (), Sequential(()), target=target)

    @module(target=target)
    class Decorated:
        def forward(self):
            return None

    assert module_value.target is target
    assert function.target is target
    assert Decorated.target is target


def test_lowering_and_codegen_keep_the_external_target_instance() -> None:
    function = RmsnormModule.entry_function()
    target = ExternalCudaTarget("nvidia.h200_sxm")
    module = Module("external", (function,), function.name, target=target)

    lowered = lower(module)

    assert lowered.target is target
    assert all(fn.target is target for fn in lowered.functions)
    assert target.get_code_generator() is CudaTarget("nvidia.h200_sxm").get_code_generator()


def test_lower_rejects_a_topology_level_unsupported_by_the_target() -> None:
    unsupported = replace(SquareCudaModel, topologies=(Topology("warp", 4),))

    with pytest.raises(ValueError, match="unsupported topology level 'warp'"):
        lower(unsupported)


@pytest.mark.parametrize(
    ("target", "topology", "extent"),
    (
        (ExternalCudaTarget("nvidia.h200_sxm"), "thread", 128),
        (AmxTarget(), "core", 1),
    ),
)
def test_public_schedule_uses_inherited_target_schedulers(
    target: Target, topology: str, extent: int
) -> None:
    scheduled = Module(
        "inherited_scheduler",
        (scheduling_gemm,),
        scheduling_gemm.name,
        target=target,
        topologies=(Topology(topology, extent),),
    )

    result = schedule(scheduled, scheduling_gemm, topology=topology)

    assert result.module is scheduled
    assert result.function is scheduling_gemm
    assert result.topology == Topology(topology, extent)
    assert isinstance(result.plan, SchedulePlan)


def test_public_schedule_overrides_refuses_and_rejects_unknown_topologies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def scheduled(target: Target, topology: str) -> Module:
        return Module(
            "custom_scheduler",
            (scheduling_gemm,),
            scheduling_gemm.name,
            target=target,
            topologies=(Topology(topology, 1),),
        )

    _CUSTOM_SOLVES.clear()
    overridden = scheduled(CustomSchedulerCudaTarget("nvidia.h200_sxm"), "custom")
    result = schedule(overridden, scheduling_gemm, topology="custom")
    assert result.plan == _CustomSchedulePlan("custom")
    assert _CUSTOM_SOLVES == ["custom"]

    solver_calls: list[str] = []

    def unexpected_thread_solver(*args):
        solver_calls.append("thread")
        raise AssertionError("refused topology reached a solver")

    monkeypatch.setattr(
        "tilefoundry.target.cuda.schedule.schedule_thread",
        unexpected_thread_solver,
    )
    refused = scheduled(RefusingCudaTarget("nvidia.h200_sxm"), "thread")
    with pytest.raises(ScheduleError) as refusal:
        schedule(refused, scheduling_gemm, topology="thread")
    assert str(refusal.value) == (
        "schedule: RefusingCudaTarget (tests.target.refusing_cuda): no scheduler for 'thread'"
    )
    assert solver_calls == []

    unknown = scheduled(CustomSchedulerCudaTarget("nvidia.h200_sxm"), "unknown")
    with pytest.raises(ScheduleError) as unknown_error:
        schedule(unknown, scheduling_gemm, topology="unknown")
    assert str(unknown_error.value) == (
        "schedule: CustomSchedulerCudaTarget "
        "(tests.target.custom_scheduler_cuda): no scheduler for 'unknown'"
    )
    assert solver_calls == []


def test_program_topologies_use_target_resource_facts() -> None:
    """A declared extent is validated against the level's own resource fact.

    A declared extent is validated against the level's own resource fact: a
    grid may be far wider than the machine's SMs and a block may not exceed the
    threads one supports. A symbolic extent is explicit but deferred until the
    program's dimensions are resolved.
    """
    target = CudaTarget("nvidia.h200_sxm")
    target.validate_program_topology(Topology("cta", 132))
    target.validate_program_topology(Topology("cta", 310_000))
    target.validate_program_topology(Topology("thread", 1024))
    target.validate_program_topology(Topology("cta", DimVar("ctas", 1, 310_001)))
    ExternalCudaTarget("nvidia.h200_sxm").validate_program_topology(Topology("thread", 1024))
    with pytest.raises(ValueError, match="must be positive"):
        target.validate_program_topology(Topology("cta", 0))
    with pytest.raises(ValueError, match="1 <= extent <= 1024") as error:
        target.validate_program_topology(Topology("thread", 1025))
    assert str(error.value) == (
        "CudaTarget (cuda): topology 'thread' extent 1025 must satisfy 1 <= extent <= 1024"
    )
    assert len(str(error.value)) < 120


def test_group_functions_by_target_fact_matching() -> None:
    """CUDA functions must agree on Target facts before grouping.

    CUDA functions must agree on Target facts before grouping; CPU
    functions are exempt from the CUDA fact-matching.
    """
    body = Sequential(body=())
    first = PrimFunction(name="first", params=(), body=body, target=CudaTarget("nvidia.h200_sxm"))
    second = PrimFunction(
        name="second",
        params=(),
        body=body,
        target=CudaTarget(
            "nvidia.h200_sxm",
            architecture=replace(CudaTarget.hardware.resolve(SM90_ID).value, name="sm_90_alt"),
        ),
    )
    with pytest.raises(ValueError, match="mixes unequal device Targets") as error:
        group_functions_by_target(Module(name="mixed", functions=(first, second), entry="first"))
    assert "CudaTarget (cuda)" in str(error.value)
    assert "architecture=" not in str(error.value)
    assert "device=" not in str(error.value)
    assert len(str(error.value)) < 300

    host = PrimFunction(name="host", params=(), body=body, target=CpuTarget())
    groups = group_functions_by_target(Module(name="mixed", functions=(first, host), entry="host"))
    assert tuple(fn.name for fn in groups[first.target]) == ("first",)
    assert tuple(fn.name for fn in groups[host.target]) == ("host",)


def test_build_rejects_unequal_device_targets_before_emitting() -> None:
    emitted: list[object] = []

    def unexpected_emit(*args):
        emitted.append(args)
        raise AssertionError("unequal device targets reached code generation")

    class FirstDeviceTarget(CudaTarget):
        name = "tests.target.first_device"

        def get_code_generator(self) -> CodeGenerator:
            return CodeGenerator(unexpected_emit)

    class SecondDeviceTarget(CudaTarget):
        name = "tests.target.second_device"

        def get_code_generator(self) -> CodeGenerator:
            return CodeGenerator(unexpected_emit)

    first_target = FirstDeviceTarget("nvidia.h200_sxm")
    second_target = SecondDeviceTarget("nvidia.h200_sxm")
    body = Sequential(body=())
    first = PrimFunction("first", (), body, target=first_target)
    second = PrimFunction("second", (), body, target=second_target)
    host = PrimFunction("host", (), body, target=CpuTarget())
    mixed = Module("mixed", (first, second, host), "host", target=first_target)

    with pytest.raises(ValueError) as error:
        build(mixed)

    assert str(error.value) == (
        "tilefoundry: module 'mixed' mixes unequal device Targets: "
        "FirstDeviceTarget (tests.target.first_device) (function 'first') vs "
        "SecondDeviceTarget (tests.target.second_device) (function 'second'); "
        "multiple device translation units are not supported"
    )
    assert emitted == []


def test_target_conflict_diagnostics_use_stable_summaries() -> None:
    function = RmsnormModule.entry_function()
    declared_target = CudaTarget("nvidia.h200_sxm")
    explicit_target = CudaTarget(
        "nvidia.h200_sxm",
        architecture=replace(CudaTarget.hardware.resolve(SM90_ID).value, name="sm_90_alt"),
    )
    module_value = Module(
        name="conflicting_targets",
        functions=(function,),
        entry=function.name,
        target=declared_target,
    )

    for invoke, operation in (
        (lambda: lower(module_value, target=explicit_target), "lower"),
        (lambda: build(module_value, target=explicit_target), "build"),
        (
            lambda: jit(
                module_value,
                target=declared_target,
                options=CompilerOptions(target=explicit_target),
            ),
            "jit",
        ),
    ):
        with pytest.raises(ValueError, match=f"tilefoundry.{operation}") as error:
            invoke()
        assert "CudaTarget (cuda)" in str(error.value)
        assert "architecture=" not in str(error.value)
        assert "device=" not in str(error.value)
        assert len(str(error.value)) < 220
