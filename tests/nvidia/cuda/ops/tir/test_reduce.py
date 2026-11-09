"""Every reduce tier in one kernel, each forced there by its layouts alone.

See [runtime §2.6](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

from pathlib import Path

import isl
import pytest
import torch

import tilefoundry
from tilefoundry import cli, verifier
from tilefoundry.dsl import *
from tilefoundry.interfaces import schedule
from tilefoundry.ir import core, dim, expr, tir, types
from tilefoundry_nvidia import cuda

OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_CUDA = cuda.Target("nvidia.h200_sxm")
_MESH = Mesh((Topology("thread", 32),), Layout((32,), (1,)), ("t",))
_OTHER_MESH = Mesh((Topology("thread", 32),), Layout((4, 8), (8, 1)), ("w", "t"))


def _sharded(
    n: int,
    *,
    dtype=DType.f32,
    storage="rmem",
    mesh: Mesh = _MESH,
    layout=None,
):
    return TensorType.from_shape(
        (n,),
        dtype,
        storage=storage,
        layout=ShardLayout(layout or Layout((n,), (1,)), (types.Broadcast(),), mesh),
    )


def _pf(*type_list, axes: tuple = (0,)) -> core.PrimFunction:
    args = tuple(expr.Var(type=type_, name=f"a{index}") for index, type_ in enumerate(type_list))
    layout = getattr(type_list[0], "layout", None)
    mesh = layout.mesh if isinstance(layout, ShardLayout) else _MESH
    return core.PrimFunction(
        target=cuda.Target("nvidia.h200_sxm"),
        name="fn",
        params=args,
        body=core.Sequential(
            body=(
                core.MeshScope(
                    mesh,
                    expr.Var(name="thread", type=mesh),
                    core.Sequential(
                        body=(
                            core.Evaluate(
                                callable=tir.reduce.Reduce(axes=axes, keepdim=True, kind=expr.ReduceKind.MEAN),
                                args=args,
                            ),
                        )
                    ),
                ),
                core.Return(),
            )
        ),
    )


_THREADS_128 = Mesh((Topology("thread", 128),), Layout((128,), (1,)), ("t",))


def _split(shape: tuple, axis: int):
    """A source cut along *axis* over the 128-thread mesh."""
    strides = (shape[1], 1)
    return TensorType.from_shape(
        shape,
        DType.f32,
        storage="rmem",
        layout=ShardLayout(Layout(shape, strides), (Split(axis),), _THREADS_128),
    )


def _replicated(shape: tuple, storage: str = "rmem"):
    return TensorType.from_shape(
        shape,
        DType.f32,
        storage=storage,
        layout=ShardLayout(Layout(shape, types.compact_row_major(shape)), (types.Broadcast(),), _THREADS_128),
    )


@pytest.mark.parametrize(
    ("operands", "axes", "refused"),
    [
        pytest.param(
            (_sharded(8), _sharded(1), _sharded(4, storage="rmem")),
            (0,),
            r"(?s)workspace: rmem does not match smem",
            id="outside_shared_memory",
        ),
        pytest.param(
            (_split((128, 8), 0), _replicated((1, 8)), _replicated((2,), "smem")),
            (0,),
            r"workspace \(2,\) must be rank-1 with at least 4 slots",
            id="fewer_slots_than_warps",
        ),
        pytest.param(
            (_split((128, 8), 0), _replicated((1, 8)), _replicated((2, 2), "smem")),
            (0,),
            r"workspace \(2, 2\) must be rank-1",
            id="not_rank_one",
        ),
        pytest.param(
            (UnitType(), _replicated((1, 8)), _replicated((4,), "smem")),
            (0,),
            r"src: None does not match Tensor\(dtype=src_dtype",
            id="source_not_a_tensor",
        ),
        pytest.param(
            (_split((128, 8), 0), _replicated((1, 8)), UnitType()),
            (0,),
            r"workspace: None does not match Tensor\(storage=smem",
            id="workspace_not_a_tensor",
        ),
        pytest.param(
            (_split((128, 8), 0), _replicated((1, 8)), _replicated((128,), "smem")),
            (0,),
            None,
            id="split_axis_0_wider_workspace",
        ),
        pytest.param(
            (_split((8, 128), 1), _replicated((8, 1)), _replicated((4,), "smem")),
            (1,),
            None,
            id="split_axis_1",
        ),
    ],
)
def test_workspace_holds_slot_per_warp(operands, axes, refused) -> None:
    """A workspace is shared memory with one slot per warp of the source's mesh.

    What every source coordinate reaches there is those slots, all of them and
    no others, however wide the workspace was allocated: which warp posts its
    partial where is the runtime's choice, not a coordinate of the source.
    """
    function = _pf(*operands, axes=axes)
    if refused is not None:
        with pytest.raises(expr.VerifyError, match=refused):
            verifier.verify_prim_function(function)
        return
    verifier.verify_prim_function(function)
    (statement,) = (stmt for stmt in function.body.body[0].body.body if isinstance(stmt, core.Evaluate))
    type_list = tuple(arg.type for arg in statement.args)
    reached = schedule.operand_relations(statement.callable, type_list)[2].relation
    source = dim.shape_to_isl_set(tuple(operands[0].shape), {})
    assert reached.is_equal(isl.map.from_domain_and_range(source, isl.set("{ [s] : 0 <= s < 4 }")))


def test_reduce_refuses_nonfloat_dtype() -> None:
    with pytest.raises(
        expr.VerifyError,
        match=r"could not satisfy src_dtype in \{f32, f16, bf16\}",
    ):
        verifier.verify_prim_function(_pf(_sharded(8, dtype=DType.fp8e4m3), _sharded(1)))


def test_reduce_refuses_swizzled_source() -> None:
    swizzled = ComposedLayout(Swizzle(2, 2, 2), 0, Layout((8,), (1,)))
    with pytest.raises(
        expr.VerifyError,
        match=r"does not match Layout\(\(\*_\,\), \(\*_\,\)\)",
    ):
        verifier.verify_prim_function(_pf(_sharded(8, layout=swizzled), _sharded(1)))


def test_refuses_operands_on_different_meshes() -> None:
    with pytest.raises(expr.VerifyError, match=r"does not match \*mesh_shape"):
        verifier.verify_prim_function(_pf(_sharded(8), _sharded(1, mesh=_OTHER_MESH)))


_ENTRY = r"tilefoundry::ops::reduce<tilefoundry::ops::"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reduce_tiers_match_torch() -> None:
    """One compile behind four assertions, each naming the tier that failed."""
    ReduceTiers = cli.load_authored_ir(f"{OPS / 'reduce_tiers.py'}:ReduceTiers")
    rm = tilefoundry.compile(ReduceTiers, target=_CUDA)
    torch.manual_seed(0)
    a_plain = torch.randn(128, 8, dtype=torch.float32, device="cuda")
    out_plain = torch.zeros(128, dtype=torch.float32, device="cuda")
    a_warp = torch.randn(32, 4, dtype=torch.float32, device="cuda")
    out_warp = torch.zeros(1, dtype=torch.float32, device="cuda")
    a_cta = torch.randn(4, 32, 8, dtype=torch.float32, device="cuda")
    out_cta = torch.zeros(1, dtype=torch.float32, device="cuda")
    a_cross = torch.randn(4, 32, dtype=torch.float32, device="cuda")
    out_cross = torch.zeros(1, 32, dtype=torch.float32, device="cuda")
    rm(a_plain, out_plain, a_warp, out_warp, a_cta, out_cta, a_cross, out_cross)
    torch.cuda.synchronize()
    assert torch.allclose(out_plain, a_plain.mean(dim=1), rtol=1e-6, atol=1e-6), (
        "plain tier (mesh_reduced false: dst splits the axis src splits): mean "
        "divides by the reduced span alone -- 8 -- which is the only count this "
        "tier has, so a tier that had brought a mesh extent into the divisor "
        "answers 8 or 32 times small"
    )
    assert torch.allclose(out_warp, a_warp.abs().max().reshape(1), rtol=0, atol=0), (
        "intra-warp tier (warps_per_group == 1 on a flat 32-lane mesh): a fold "
        "that added the lanes' maxima instead of maxing them answers roughly 32 "
        "times large"
    )
    assert torch.allclose(out_cta, a_cta.mean().reshape(1), rtol=1e-5, atol=1e-6), (
        "intra-CTA tier (lane_reduced with warps_per_group == 4): mean's divisor "
        "is the product of all three counts the tier knows -- span 8, lanes 32, "
        "warps 4 -- so the 1024 it must divide by is exactly what the greedy "
        "warp walk this dispatch replaced got wrong"
    )
    assert torch.allclose(out_cross[0], a_cross.abs().amax(dim=0), rtol=0, atol=0), (
        "cross-warp tier (warps_per_group == 4 with lane_reduced false): absmax "
        "is what shows the fold combining with the reduction's own operator "
        "rather than adding"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cta_thread_mesh_reduces() -> None:
    """On a composed cta×thread mesh, the warp count comes from the thread level.

    The dispatch used to keep one ``is_thread`` per mesh, so a cta-scoped
    composite counted the thread level's instances as warps -- 128 here, not 4
    -- and the cross-warp tier indexed the workspace out of bounds, while the
    verifier demanded one slot per warp of the flattened mesh (16, not 4) and
    refused this program before it ran. The plan the levels state is the
    intra-CTA tier: 32 lanes and 4 warps per group, one slot per warp, and
    eight kept cells combined one at a time.
    """
    TwoLevelReduce = cli.load_authored_ir(f"{OPS / 'two_level_reduce.py'}:TwoLevelReduce")
    rm = tilefoundry.compile(TwoLevelReduce, target=_CUDA)
    torch.manual_seed(0)
    a_two = torch.randn(8, 4, 128, 4, dtype=torch.float32, device="cuda")
    out_two = torch.zeros(8, 4, 1, 1, dtype=torch.float32, device="cuda")
    rm(a_two, out_two)
    torch.cuda.synchronize()
    assert torch.allclose(out_two, a_two.mean(dim=(2, 3), keepdim=True), rtol=1e-5, atol=1e-6), (
        "intra-CTA tier on a cta×thread mesh (warps_per_group == 4 read off the "
        "thread level, 32 lanes): the mean's divisor is span 4 x lanes 32 x "
        "warps 4 = 512 either way, so what a wrong warp count shows is not a "
        "wrong divisor but a workspace read out of bounds -- a launch failure "
        "here -- or another group's partials folded into the answer"
    )
