from __future__ import annotations

import pytest

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types import DType, TensorType, local_type_of
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.parser.hir_parser import parse_script
from tilefoundry.target.cuda.planner import build_planning_problem


def _root_source(
    body: str,
    params: str = 'x: Tensor[(8, 16), "bf16"]',
    return_shape: str = "(8, 16)",
) -> str:
    return f'''from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.target import CudaTarget
from tilefoundry.ir.types.shard import Layout, Mesh, Topology

cta_mesh = Mesh(Topology("cta", 8), Layout((8,), (1,)))

@func(target=CudaTarget(), topologies=(Topology("cta", 8),))
def root({params}) -> Tensor[{return_shape}, "bf16"]:
{body}
'''


def test_where_fields_match_one_bucket_conjunctively() -> None:
    root = parse_script(
        _root_source(
            '    y: where(layout=(_, 8 @ cta), mesh=cta_mesh, storage="gmem") = tf.add(x, x)\n'
            "    return y",
            params='x: Tensor[(8, 8), "bf16"]',
            return_shape="(8, 8)",
        )
    )
    problem = build_planning_problem(Module("m", (root,), "root"), root)
    assert len(problem.requirements) == 1
    requirement = problem.requirements[0]
    assert len(requirement.bucket_ids) == 1
    bucket = problem.buckets[requirement.bucket_ids[0]]
    type = problem.types[bucket.type_id]
    assert isinstance(type, TensorType)
    assert type.storage is StorageKind.GMEM
    assert isinstance(type.layout, ShardLayout)
    assert type.layout.mesh == Mesh(Topology("cta", 8), Layout((8,), (1,)))
    assert type.layout.attrs == (Split(1),)


def test_non_gmem_source_fails_at_planning_boundary() -> None:
    root = parse_script(
        _root_source(
            "    return x",
            params='x: Tensor[(8, 16), "bf16", None, "rmem"]',
        )
    )
    with pytest.raises(ValueError, match="GMEM"):
        build_planning_problem(Module("m", (root,), "root"), root)


def test_local_projection_is_recursive_and_view_candidates_are_zero_time() -> None:
    mesh = Mesh(Topology("cta", 2), Layout((2,), (1,)))
    inner = ShardLayout(
        layout=Layout((2, 8), (8, 1)), attrs=(Split(0),), mesh=mesh
    )
    outer = ShardLayout(
        layout=inner, attrs=(Split(1),), mesh=mesh
    )
    projected = local_type_of(
        TensorType((2, 8), DType.f32, outer, StorageKind.GMEM)
    )
    assert projected.shape == (1, 4)

    root = parse_script(
        _root_source(
            '    y = tf.reshape(x, new_shape=(16,))\n'
            '    return tf.transpose(y, perm=(0,))\n'
        )
    )
    problem = build_planning_problem(Module("m", (root,), "root"), root)
    views = [
        candidate
        for candidate in problem.candidates.values()
        if type(candidate.op).__name__ in {"Reshape", "Transpose"}
    ]
    assert views
    assert all(candidate.duration_ns == 0 for candidate in views)
    assert all(candidate.local_cost.bytes == 0 for candidate in views)
    assert all(candidate.output_alias_input_indices == (0,) for candidate in views)
