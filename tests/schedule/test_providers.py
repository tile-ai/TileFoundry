from __future__ import annotations

import pytest

from tilefoundry.ir.target import CudaTarget
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.parser import parse_module_source
from tilefoundry.providers import resolve_provider_services
from tilefoundry.providers.cuda import (
    CudaArchitectureProfile,
    CudaDeviceProfile,
    CudaFormulaCostModel,
)
from tilefoundry.providers.services import TargetScheduleProfile
from tilefoundry.schedule import (
    build_program_schedule_graph,
    build_schedule_space,
    generate_distribution_candidates,
)


def test_h200_cta_services_are_resolved_through_ioc() -> None:
    services = resolve_provider_services(
        CudaTarget(arch="sm_90", device="h200_sxm"), "cta"
    )
    assert services.get(CudaArchitectureProfile).arch == "sm_90"
    assert services.get(CudaDeviceProfile).sm_count == 132
    assert services.get(TargetScheduleProfile).max_ctas == 132
    assert isinstance(services.get(CudaFormulaCostModel), CudaFormulaCostModel)


def test_cuda_autodist_requires_a_concrete_device() -> None:
    with pytest.raises(ValueError, match=r"requires CudaTarget\(device=\.\.\.\)"):
        resolve_provider_services(CudaTarget(arch="sm_90"), "cta")


def test_unsupported_level_and_device_fail_clearly() -> None:
    with pytest.raises(ValueError, match="level 'thread'"):
        resolve_provider_services(
            CudaTarget(arch="sm_90", device="h200_sxm"), "thread"
        )
    with pytest.raises(ValueError, match="unsupported CUDA AutoDist device"):
        resolve_provider_services(
            CudaTarget(arch="sm_90", device="unknown"), "cta"
        )


def test_schedule_space_counts_subbyte_tensor_bytes() -> None:
    module = parse_module_source(
        '''from __future__ import annotations
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, tf

@module(entry="main")
class Dtypes:
    @func
    def main(
        fp4: Tensor[(8,), "f4e2m1"],
        fp8: Tensor[(8,), "fp8e4m3"],
        bf16: Tensor[(8,), "bf16"],
    ) -> Tensor[(8,), "bf16"]:
        a = tf.cast(fp4, "bf16")
        b = tf.cast(fp8, "bf16")
        c = tf.add(a, b)
        return tf.add(c, bf16)
'''
    )
    graph = build_program_schedule_graph(module)
    candidates = generate_distribution_candidates(graph, max_ctas=8)
    mesh = Mesh(Topology("cta", 8), Layout((8,), (1,)))
    space = build_schedule_space(graph, candidates, parent_mesh=mesh)
    payload_by_name = {}
    for edge in graph.edges:
        if edge.kind != "data":
            continue
        name = getattr(graph.value(edge.source).ir_value, "name", None)
        if name in {"fp4", "fp8", "bf16"}:
            payload_by_name[name] = next(
                option.payload_bytes
                for option in space.options_for_use(edge.id)
                if option.kind.value == "direct"
            )
    assert payload_by_name == {"fp4": 4, "fp8": 8, "bf16": 16}
