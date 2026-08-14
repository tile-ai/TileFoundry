from __future__ import annotations

from tilefoundry.module import module
from tilefoundry import func
from tilefoundry.target import CudaTarget
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.dsl import ConstTensor, Tensor
from tilefoundry.dsl.storage import gmem, host, rmem, smem, tmem  # noqa: F401
from tilefoundry.ir.types.shard import (
    B, S, P, ComposedLayout, Layout, Mesh, ShardLayout, Topology,
)
from tilefoundry.ir.types.dim import DimVar, ceildiv

N = DimVar("N", 1, 64)

cta = Mesh((Topology("cta", 4),), Layout((4,), (1,)), names=('tile',))
cta_2 = Mesh((Topology("cta", 4),), Layout((4,), (1,)), names=('tile',))

@module(entry="calls_a_child", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 4),))
class HirModule:
    @module(entry="entry")
    class leaf:
        @func
        def helper(
            x: Tensor[(8,), "f32"]
        ) -> Tensor[(8,), "f32"]:
            v0 = mul(x, x)
            return v0

        @func
        def entry(
            x: Tensor[(8,), "f32"]
        ) -> Tensor[(8,), "f32"]:
            v0 = helper(x)
            return v0

    @module(entry="entry")
    class first:
        @func
        def helper(
            x: Tensor[(8,), "f32"]
        ) -> Tensor[(8,), "f32"]:
            v0 = mul(x, x)
            return v0

        @func
        def entry(
            x: Tensor[(8,), "f32"]
        ) -> Tensor[(8,), "f32"]:
            v0 = helper(x)
            return v0

    @module(entry="entry")
    class second:
        @func
        def helper(
            x: Tensor[(8,), "f32"]
        ) -> Tensor[(8,), "f32"]:
            v0 = mul(x, x)
            return v0

        @func
        def entry(
            x: Tensor[(8,), "f32"]
        ) -> Tensor[(8,), "f32"]:
            v0 = helper(x)
            return v0

    @module(entry="run")
    class mlp:
        @func
        def run(
            x: Tensor[(4, 8), "f32"],
            w: ConstTensor[(8, 8), "f32"]
        ) -> Tensor[(4, 8), "f32"]:
            v0 = matmul(x, w)
            return v0

    @module(entry="mid")
    class deep:
        @module(entry="run")
        class grand:
            @func
            def run(
                x: Tensor[(8,), "f32"],
                w: ConstTensor[(8,), "f32"]
            ) -> Tensor[(8,), "f32"]:
                v0 = mul(x, w)
                return v0

        @func
        def mid(
            x: Tensor[(8,), "f32"]
        ) -> Tensor[(8,), "f32"]:
            v0 = grand(x)
            return v0

    @module(entry="scale")
    class variant_leaf:
        @func
        def scale(
            x: Tensor[(N,), "f32"]
        ) -> Tensor[(N,), "f32"]:
            v0 = mul(x, x)
            return v0

    @func
    def two_bindings_under_a_reshard(
        x: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        local = reshard(x, layout=ShardLayout(
            layout=Layout((4, 2), None),
            attrs=(S(0),),
            mesh=cta,
        ), storage=rmem)
        v0 = first(local)
        v1 = second(local)
        v2 = add(v0, v1)
        return v2

    @func
    def through_the_grandchild(
        x: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        local = reshard(x, layout=ShardLayout(
            layout=Layout((4, 2), None),
            attrs=(S(0),),
            mesh=cta_2,
        ), storage=gmem)
        v0 = deep(local)
        return v0

    @func
    def carries_activations_only(
        x: Tensor[(4, 8), "f32"]
    ) -> Tensor[(4, 8), "f32"]:
        v0 = mlp(x)
        return v0

    @func
    def converts_its_weight(
        x: Tensor[(8,), "f32"],
        w: ConstTensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        v0 = add(x, w)
        return v0

    @func
    def dispatches_to_a_variant(
        x: Tensor[(N,), "f32"]
    ) -> Tensor[(N,), "f32"]:
        pass

    @dispatches_to_a_variant.specialize(DimVarRangePat("N", 1, 64))
    def _(
        x: Tensor[(N,), "f32"]
    ) -> Tensor[(N,), "f32"]:
        v0 = variant_leaf(x)
        return v0

    @func
    def uses_a_custom_op(
        a: Tensor[(8,), "f32"],
        b: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        v0 = custom_parse_addsq(a, b)
        return v0

    @func
    def calls_a_child(
        x: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        v0 = leaf(x)
        return v0
