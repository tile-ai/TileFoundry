"""What a Target validates about a program, and how codegen groups by one.

What a Target validates about a program, and how codegen groups by one.

The composed hardware facts themselves -- which documents a default target
resolves, and which limits belong to the architecture rather than the device --
are asserted where those facts are installed. What is left here is what a Target
is asked during a compilation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.core.target.test_target import _provider_target
from tilefoundry import CompilerOptions, analysis, build, cli, codegen, jit, target
from tilefoundry.dsl import *
from tilefoundry.ir import core
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda
from tilefoundry_nvidia.cuda import target as cuda_target
from tilefoundry_nvidia.cuda.codegen.tir import stmts

PLACED = Path(__file__).parents[1] / "fixtures" / "placed"


class ExternalCudaTarget(cuda.Target):
    name = "tests.core.target.external_cuda"


class _NoComputeUnitRateCudaTarget(cuda.Target):
    name = "tests.core.target.no_compute_unit_rate_cuda"

    def get_facts(self, facts_type: type, query: object | None = None):
        facts = super().get_facts(facts_type, query)
        if facts_type is target.PerformanceServiceFacts:
            return replace(facts, unit_flops=())
        return facts


class ExtraTopologyCudaTarget(cuda.Target):
    name = "tests.core.target.extra_topology_cuda"
    extra_levels = (
        target.TopologyLevelFacts("custom", 1, None),
        target.TopologyLevelFacts("unknown", 1, None),
    )

    def get_facts(self, facts_type: type, query: object | None = None):
        if facts_type is target.TopologyFacts and query is None:
            inherited = super().get_facts(facts_type, query)
            return target.TopologyFacts(
                (*inherited.topologies, *self.extra_levels),
                parallel_level=inherited.parallel_level,
            )
        if facts_type is target.TopologyLevelFacts:
            for level in self.extra_levels:
                if level.name == query:
                    return level
        return super().get_facts(facts_type, query)


def test_cuda_projects_cta_rates() -> None:
    """One CTA's share of the float and memory peaks, and its own services.

    The float and bandwidth numbers are the device peaks divided among the SMs.
    The services are not a division of anything: no device-wide integer or
    predicate peak is published, so each is stated per CTA in the hardware table
    and arrives unchanged.
    """
    tgt = cuda.Target("nvidia.h200_sxm")
    device = tgt.get_facts(target.ThroughputFacts)
    services = tgt.get_facts(target.PerformanceServiceFacts)

    assert services.unit == "cta"
    assert dict(services.unit_flops) == {
        dtype: rate // tgt.device.sm_count for dtype, rate in device.peak_flops_per_second
    }
    assert dict(services.unit_bandwidth) == {
        device.bandwidth_level: (
            tgt.device.hbm_bandwidth_bytes_per_second // tgt.device.sm_count
        ),
        "smem": (
            tgt.architecture.shared_memory_bandwidth_bytes_per_clock_per_sm
            * tgt.device.sm_clock_hz
        ),
    }
    assert dict(services.unit_ops) == dict(tgt.device.service_ops_per_second)
    assert all(rate > 0 for _kind, rate in services.unit_ops)


def test_performance_refuses_missing_compute_rate() -> None:
    tgt = _NoComputeUnitRateCudaTarget("nvidia.h200_sxm")
    MoEMegaKernel = cli.load_authored_ir(f"{PLACED / 'moe_mega_kernel.py'}:MoEMegaKernel")
    subject = replace(MoEMegaKernel, target=tgt)
    function = subject.entry_function()

    with pytest.raises(analysis.Error, match=r"dtype 'f32'.*'cta'"):
        analysis.analyze(*analysis.check_program(subject, function), analysis="performance")


def test_topology_validation_uses_emission_target() -> None:
    custom = ExtraTopologyCudaTarget("nvidia.h200_sxm")

    stmts.validate_cuda_topology_levels(custom, ("custom",))
    with pytest.raises(ValueError, match=r"supports \{gpu, cta, thread, custom, unknown\}"):
        stmts.validate_cuda_topology_levels(custom, ("warp",))


def test_registration_rejects_invalid_providers() -> None:
    inherited = type(
        "InheritedTarget",
        (cuda.Target,),
        {"__module__": "tests.providers.inherited"},
    )
    empty = _provider_target("tests.providers.empty", "EmptyTarget", "")

    with pytest.raises(ValueError, match="declare a class-level"):
        target.register(inherited)
    with pytest.raises(ValueError, match="non-empty"):
        target.register(empty)
    with pytest.raises(TypeError, match="Target subclass"):
        target.register(object)
    with pytest.raises(TypeError, match="Target subclass"):
        target.register("tests.core.target.argument")

    owner = _provider_target("tests.providers.owner", "OwnedTarget", "tests.core.target.conflict")
    claimant = _provider_target(
        "tests.providers.claimant", "ClaimantTarget", "tests.core.target.conflict"
    )
    target.register(owner)
    with pytest.raises(ValueError, match="already owned.*cannot register"):
        target.register(claimant)


def test_target_rejects_strings_keeps_instances() -> None:
    tgt = cuda.Target("nvidia.h200_sxm")
    module = core.Module("exact", (), target=tgt)

    assert module.target is tgt
    assert module.resolve_target() is tgt
    with pytest.raises(TypeError, match="Target instance, not a string"):
        core.Module("invalid", (), target="cuda")
    with pytest.raises(TypeError, match="Target instance, not a string"):
        core.PrimFunction("invalid", (), core.Sequential(()), target="cuda")


def test_topology_extent_checks_resource_facts() -> None:
    """A declared extent is validated against the level's own resource fact.

    A declared extent is validated against the level's own resource fact: a
    grid may be far wider than the machine's SMs and a block may not exceed the
    threads one supports. A symbolic extent is explicit but deferred until the
    program's dimensions are resolved.
    """
    tgt = cuda.Target("nvidia.h200_sxm")
    tgt.validate_program_topology(Topology("cta", 132))
    tgt.validate_program_topology(Topology("cta", 310_000))
    tgt.validate_program_topology(Topology("thread", 1024))
    tgt.validate_program_topology(Topology("cta", DimVar("ctas", 1, 310_000)))
    ExternalCudaTarget("nvidia.h200_sxm").validate_program_topology(Topology("thread", 1024))
    with pytest.raises(ValueError, match="must be positive"):
        tgt.validate_program_topology(Topology("cta", 0))
    with pytest.raises(ValueError, match="1 <= extent <= 1024") as error:
        tgt.validate_program_topology(Topology("thread", 1025))
    assert str(error.value) == (
        "cuda.Target: topology 'thread' extent 1025 must satisfy 1 <= extent <= 1024"
    )
    assert len(str(error.value)) < 120


def test_build_rejects_mixed_device_targets() -> None:
    emitted: list[object] = []

    def unexpected_emit(*args):
        emitted.append(args)
        raise AssertionError("unequal device targets reached code generation")

    class FirstDeviceTarget(cuda.Target):
        name = "tests.core.target.first_device"

    class SecondDeviceTarget(cuda.Target):
        name = "tests.core.target.second_device"

    @target.bind(FirstDeviceTarget, codegen.LinkableModule)
    @target.bind(SecondDeviceTarget, codegen.LinkableModule)
    class UnexpectedModule(codegen.LinkableModule):
        @classmethod
        def create(cls, *args):
            return unexpected_emit(*args)

    first_target = FirstDeviceTarget("nvidia.h200_sxm")
    second_target = SecondDeviceTarget("nvidia.h200_sxm")
    body = core.Sequential(body=())
    first = core.PrimFunction("first", (), body, target=first_target)
    second = core.PrimFunction("second", (), body, target=second_target)
    host = core.PrimFunction("host", (), body, target=cpu.Target())
    mixed = core.Module("mixed", (first, second, host), "host", target=first_target)

    with pytest.raises(ValueError) as error:
        build(mixed)

    assert str(error.value) == "codegen: expected one device Target, found 2"
    assert emitted == []


def test_target_conflict_summaries_stable() -> None:
    function = cli.load_authored_ir(f"{PLACED / 'rmsnorm.py'}:RmsnormModule").entry_function()
    declared_target = cuda.Target("nvidia.h200_sxm")
    explicit_target = cuda.Target(
        "nvidia.h200_sxm",
        architecture=replace(cuda.Target.hardware.resolve(cuda_target.SM90_ID).value, name="sm_90_alt"),
    )
    module_value = core.Module(
        name="conflicting_targets",
        functions=(function,),
        entry=function.name,
        target=declared_target,
    )

    for invoke, operation in (
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
        assert "cuda.Target" in str(error.value)
        assert "architecture=" not in str(error.value)
        assert "device=" not in str(error.value)
        assert len(str(error.value)) < 220


def test_cuda_target_registered() -> None:
    assert {"cuda"} <= target.registered().keys()
