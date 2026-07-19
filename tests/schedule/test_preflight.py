from __future__ import annotations

from dataclasses import replace

import pytest

from tests.models.deepseek_v4_flash.moe import (
    deepseek_v4_flash_module,
    deepseek_v4_flash_moe,
)
from tests.models.qwen3_5_30b_a3b.gqa_online import gqa_online_attend
from tests.models.qwen3_5_30b_a3b.static_online import qwen_static_online
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.types.shard import Topology
from tilefoundry.parser import parse_func_source
from tilefoundry.target import CpuTarget, CudaTarget
from tilefoundry.target.cuda.preflight import CtaPreflightResult, preflight_cta


@func
def _preflight_leaf(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return tf.add(x, x)


@func(target=CudaTarget(), topologies=(Topology("cta", 4),))
def _preflight_root(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return _preflight_leaf(x)


def _replace_root_call_target(root: Function, target: Function | object) -> Function:
    assert isinstance(root.body, Call)
    return replace(root, body=replace(root.body, target=target))


def test_real_roots_pass_private_cta_preflight() -> None:
    deepseek = preflight_cta(deepseek_v4_flash_module)
    qwen = preflight_cta(qwen_static_online)
    assert isinstance(deepseek, CtaPreflightResult)
    assert deepseek.cta_count == 132
    assert qwen.cta_count == 132
    assert deepseek.reachable_functions[0] is deepseek_v4_flash_moe
    assert all(fn.target is None for fn in deepseek.reachable_functions[1:])


def test_non_cta_topologies_are_preserved_and_ignored() -> None:
    extended = replace(
        deepseek_v4_flash_moe,
        topologies=(
            Topology("cta", 132),
            Topology("gpu", 2),
            Topology("thread", 1024),
            Topology("warp", 32),
            Topology("future_level", 7),
        ),
    )
    result = preflight_cta(extended)
    assert result.cta_count == 132
    assert tuple(topology.name for topology in extended.topologies) == (
        "cta",
        "gpu",
        "thread",
        "warp",
        "future_level",
    )


@pytest.mark.parametrize(
    "topologies, pattern",
    [
        ((), "exactly one"),
        ((Topology("cta", 1), Topology("cta", 2)), "exactly one"),
        ((Topology("cta", None),), "dynamic"),
        ((Topology("cta", 133),), "supports"),
    ],
)
def test_root_cta_declarations_fail_before_traversal(topologies, pattern) -> None:
    invalid = replace(deepseek_v4_flash_moe, topologies=topologies)
    with pytest.raises(ValueError, match=pattern):
        preflight_cta(invalid)


def test_root_target_is_explicit_cuda() -> None:
    with pytest.raises(ValueError, match="no explicit CUDA"):
        preflight_cta(replace(deepseek_v4_flash_moe, target=None))
    with pytest.raises(ValueError, match="requires CudaTarget"):
        preflight_cta(replace(deepseek_v4_flash_moe, target=CpuTarget()))


def test_empty_topology_helper_inherits_without_mutation() -> None:
    result = preflight_cta(_preflight_root)
    assert result.cta_count == 4
    assert _preflight_leaf.target is None
    assert _preflight_leaf.topologies == ()


def test_helper_target_conflict_and_program_topology_fail_at_call() -> None:
    conflict = _replace_root_call_target(
        _preflight_root,
        replace(_preflight_leaf, target=CpuTarget()),
    )
    with pytest.raises(ValueError, match="conflicts.*call"):
        preflight_cta(conflict)

    topology_helper = _replace_root_call_target(
        _preflight_root,
        replace(_preflight_leaf, topologies=(Topology("cta", 1),)),
    )
    with pytest.raises(ValueError, match="program topologies.*call"):
        preflight_cta(topology_helper)


def test_recursive_and_kernel_calls_are_rejected() -> None:
    recursive = replace(_preflight_root, body=replace(_preflight_root.body, target=None))
    assert isinstance(recursive.body, Call)
    object.__setattr__(recursive.body, "target", recursive)
    with pytest.raises(ValueError, match="recursive helper call"):
        preflight_cta(recursive)

    kernel_call = Call(
        type=_preflight_root.return_type,
        target=Launch(),
        args=(),
    )
    kernel_root = replace(_preflight_root, body=kernel_call)
    with pytest.raises(ValueError, match="kernel call"):
        preflight_cta(kernel_root)


def test_nested_static_regions_pass_and_dynamic_region_fails() -> None:
    nested = parse_func_source(
        '''from __future__ import annotations
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.dsl.tf import *
from tilefoundry.target import CudaTarget
from tilefoundry.ir.types.shard import Topology

@func(target=CudaTarget(), topologies=(Topology("cta", 1),))
def nested(x: Tensor[(4, 4), "f32"]) -> Tensor[(4, 4), "f32"]:
    acc = tf.full_like(x, value=0.0)
    for i in tile(4):
        for j in tile(2):
            acc = acc + x
    return acc
'''
    )
    assert preflight_cta(nested).cta_count == 1

    dynamic = replace(gqa_online_attend.variants[0], target=CudaTarget())
    with pytest.raises(ValueError, match="GridRegion.*dynamic.*extent"):
        preflight_cta(dynamic)


def test_cuda_target_has_no_registered_cta_service() -> None:
    assert CudaTarget()._services == ()
