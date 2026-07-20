"""Private CTA planning-problem construction contract."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.models.deepseek_v4_flash.moe import (
    deepseek_v4_flash_module,
    deepseek_v4_flash_moe,
)
from tests.models.qwen3_5_30b_a3b.static_online import qwen_static_online
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shard import Topology
from tilefoundry.parser.hir_parser import parse_script
from tilefoundry.target import CudaTarget
from tilefoundry.target.cuda.planner import build_planning_problem


@func
def _planner_helper(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    return tf.add(x, x)


@func(target=CudaTarget(), topologies=(Topology("cta", 4),))
def _planner_root(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    return tf.add(_planner_helper(x), _planner_helper(x))


def test_real_fixtures_build_finite_problems() -> None:
    deepseek = build_planning_problem(deepseek_v4_flash_module, deepseek_v4_flash_moe)
    qwen = build_planning_problem(
        Module("qwen", (qwen_static_online,), "qwen_static_online"), qwen_static_online
    )

    assert deepseek.site_order == tuple(range(len(deepseek.site_order)))
    assert qwen.site_order == tuple(range(len(qwen.site_order)))
    assert any(
        type(candidate.op).__name__ == "Reshard" and candidate.site_id is not None
        for candidate in qwen.candidates.values()
    )
    assert any(
        type(candidate.op).__name__ == "Reshard" and candidate.site_id is None
        for candidate in qwen.candidates.values()
    )


def test_repeated_helper_calls_get_distinct_function_instances() -> None:
    problem = build_planning_problem(Module("m", (_planner_root,), "_planner_root"), _planner_root)
    instances = [function for _, function in problem.function_instances]
    assert len(instances) == 3
    assert instances[1] is instances[2]
    helper_paths = [
        path for path, function in problem.function_instances if function.name == "_planner_helper"
    ]
    assert len(helper_paths) == 2
    assert helper_paths[0] != helper_paths[1]
    assert len(problem.site_order) == 3


def test_grid_region_records_static_trip_count_without_unrolling() -> None:
    root = parse_script(
        '''from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.target import CudaTarget
from tilefoundry.ir.types.shard import Topology

@func(target=CudaTarget(), topologies=(Topology("cta", 1),))
def root(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    acc = tf.full_like(x, value=0.0)
    for i in range(1, 8, 2):
        acc = tf.add(acc, x)
    return acc
'''
    )
    problem = build_planning_problem(Module("m", (root,), "root"), root)
    assert len(problem.regions) == 1
    region = next(iter(problem.regions.values()))
    assert region.trip_count == 4
    assert len(region.operation_site_ids) == 1


@pytest.mark.parametrize(
    "mutator, pattern",
    [
        (lambda fn: replace(fn, target=None), "explicit CudaTarget"),
        (lambda fn: replace(fn, topologies=()), "exactly one CTA"),
        (lambda fn: replace(fn, topologies=(Topology("cta", None),)), "static CTA"),
    ],
)
def test_planning_entry_rejects_invalid_root(mutator, pattern: str) -> None:
    invalid = mutator(deepseek_v4_flash_moe)
    with pytest.raises(ValueError, match=pattern):
        build_planning_problem(Module("m", (invalid,), invalid.name), invalid)


def test_root_must_be_a_module_member() -> None:
    with pytest.raises(ValueError, match="not a member"):
        build_planning_problem(deepseek_v4_flash_module, Function.build(
            name="other", params=(), body=None, return_type=deepseek_v4_flash_moe.return_type
        ))


