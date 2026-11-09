"""A hand-written TIR kernel where each CTA of a launch-sized mesh squares one row."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.ir import expr
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda

_TILE = 12
_NT = DimVar("Ntile", 1, 63)


@module(entry="dyn_square_host", target=cuda.Target("nvidia.h200_sxm"))
class DynSquare:
    @prim_func(target=cuda.Target("nvidia.h200_sxm"))
    def dyn_square(a: Tensor[(_NT, _TILE), "f32"]):
        with Mesh((Topology("cta", _NT),), Layout(shape=(_NT,), strides=(1,))) as cta:
            a_view = T.tensor_view(
                T.ptr_of(a),
                layout=ShardLayout(
                    layout=Layout(shape=(_NT, _TILE), strides=(_TILE, 1)),
                    attrs=(Split(0),),
                    mesh=cta,
                ),
            )
            reg = T.alloc_tensor(
                TensorType(
                    shape=(_NT, _TILE),
                    dtype=DType.f32,
                    layout=ShardLayout(
                        layout=Layout(shape=(_NT, _TILE), strides=(_TILE, 1)),
                        attrs=(Split(0),),
                        mesh=cta,
                    ),
                    storage=StorageKind.RMEM,
                )
            )
            T.copy(a_view, reg)
            T.binary(reg, reg, reg, kind=expr.BinaryKind.MUL)
            T.copy(reg, a_view)

    @prim_func(target=cpu.Target())
    def dyn_square_host(a: Tensor[(_NT, _TILE), "f32"]):
        launch(dyn_square, a, grid=(_NT, 1, 1), block=(1, 1, 1))
