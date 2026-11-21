"""Windows each CTA takes of its input, copied into registers split over threads or into ``out``."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda

_CUDA = cuda.Target("nvidia.h200_sxm")
_ROWS = DimVar("rows", 1, 32)


@module(entry="rows_host", target=_CUDA, topologies=(Topology("cta", 4), Topology("thread", 32)))
class WindowToThreads:
    """Each CTA copies its 32 rows of ``x`` into one register per thread, then into ``out``."""

    @prim_func(target=_CUDA)
    def rows(x: Tensor[(128,), "f32"], out: Tensor[(128,), "f32"]):
        with Mesh((Topology("cta", 4),), layout=(4,), names=("m",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("row",)) as scope:
                held = T.alloc_tensor(
                    tensor_type=Tensor[(32,), "f32", ((32 @ scope.row,), (1,)) @ {cta.m @ V()}, "rmem"]
                )
                window = T.tensor_view(
                    T.ptr_of(x[cta.m * 32:cta.m * 32 + 1]), layout=(32,) @ {cta.m @ V()}, shape=(32,)
                )
                T.copy(window, held)
                stored = T.tensor_view(T.ptr_of(out[cta.m * 32:cta.m * 32 + 1]), layout=(32,), shape=(32,))
                T.copy(held, stored)

    @prim_func(target=cpu.Target())
    def rows_host(x: Tensor[(128,), "f32"], out: Tensor[(128,), "f32"]):
        launch(rows, x, out, grid=(4, 1, 1), block=(32, 1, 1))


@module(entry="columns_host", target=_CUDA, topologies=(Topology("cta", 4), Topology("thread", 32)))
class DynamicWindowToColumns:
    """Each CTA copies column ``m`` of ``x``, ``rows`` long, into column ``m`` of ``out``."""

    @prim_func(target=_CUDA)
    def columns(x: Tensor[(_ROWS, 4), "f32"], out: Tensor[(32, 4), "f32"]):
        with Mesh((Topology("cta", 4),), layout=(4,), names=("m",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("t",)) as scope:
                with Mesh(scope[:1], layout=(1,), names=("lead",)) as _lead:
                    window = T.tensor_view(
                        T.ptr_of(x[0:1, cta.m:cta.m + 1]),
                        layout=((_ROWS,), (4,)) @ {cta.m @ V()},
                        shape=(_ROWS,),
                    )
                    stored = T.tensor_view(
                        T.ptr_of(out[0:1, cta.m:cta.m + 1]), layout=((_ROWS,), (4,)), shape=(_ROWS,)
                    )
                    T.copy(window, stored)

    @prim_func(target=cpu.Target())
    def columns_host(x: Tensor[(_ROWS, 4), "f32"], out: Tensor[(32, 4), "f32"]):
        launch(columns, x, out, grid=(4, 1, 1), block=(32, 1, 1))


@module(entry="columns_host", target=_CUDA, topologies=(Topology("cta", 4), Topology("thread", 32)))
class DynamicSwizzledWindowToColumns:
    """``DynamicWindowToColumns`` reading its column through ``Swizzle(1, 2, 1)``."""

    @prim_func(target=_CUDA)
    def columns(x: Tensor[(_ROWS, 4), "f32"], out: Tensor[(32, 4), "f32"]):
        with Mesh((Topology("cta", 4),), layout=(4,), names=("m",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("t",)) as scope:
                with Mesh(scope[:1], layout=(1,), names=("lead",)) as _lead:
                    window = T.tensor_view(
                        T.ptr_of(x[0:1, cta.m:cta.m + 1]),
                        layout=(((_ROWS,), (4,)) | Swizzle(1, 2, 1)) @ {cta.m @ V()},
                        shape=(_ROWS,),
                    )
                    stored = T.tensor_view(
                        T.ptr_of(out[0:1, cta.m:cta.m + 1]), layout=((_ROWS,), (4,)), shape=(_ROWS,)
                    )
                    T.copy(window, stored)

    @prim_func(target=cpu.Target())
    def columns_host(x: Tensor[(_ROWS, 4), "f32"], out: Tensor[(32, 4), "f32"]):
        launch(columns, x, out, grid=(4, 1, 1), block=(32, 1, 1))


@module(entry="block_host", target=_CUDA, topologies=(Topology("thread", 32),))
class DynamicRowMajorBlock:
    """A (2, rows, 4) parameter, first stride ``4 * rows``, copied into the top of ``out``."""

    @prim_func(target=_CUDA)
    def block(x: Tensor[(2, _ROWS, 4), "f32"], out: Tensor[(2, 32, 4), "f32"]):
        with Mesh((Topology("thread", 32),), layout=(32,), names=("t",)) as scope:
            with Mesh(scope[:1], layout=(1,), names=("lead",)) as _lead:
                top = T.tensor_view(
                    T.ptr_of(out), layout=((2, _ROWS, 4), (128, 4, 1)), shape=(2, _ROWS, 4)
                )
                T.copy(x, top)

    @prim_func(target=cpu.Target())
    def block_host(x: Tensor[(2, _ROWS, 4), "f32"], out: Tensor[(2, 32, 4), "f32"]):
        launch(block, x, out, grid=(1, 1, 1), block=(32, 1, 1))


@module(entry="columns_host", target=_CUDA, topologies=(Topology("cta", 4), Topology("thread", 32)))
class DynamicPlainWindowToColumns:
    """``DynamicWindowToColumns`` with a plain source window."""

    @prim_func(target=_CUDA)
    def columns(x: Tensor[(_ROWS, 4), "f32"], out: Tensor[(32, 4), "f32"]):
        with Mesh((Topology("cta", 4),), layout=(4,), names=("m",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("t",)) as scope:
                with Mesh(scope[:1], layout=(1,), names=("lead",)) as _lead:
                    window = T.tensor_view(
                        T.ptr_of(x[0:1, cta.m:cta.m + 1]), layout=((_ROWS,), (4,)), shape=(_ROWS,)
                    )
                    stored = T.tensor_view(
                        T.ptr_of(out[0:1, cta.m:cta.m + 1]), layout=((_ROWS,), (4,)), shape=(_ROWS,)
                    )
                    T.copy(window, stored)

    @prim_func(target=cpu.Target())
    def columns_host(x: Tensor[(_ROWS, 4), "f32"], out: Tensor[(32, 4), "f32"]):
        launch(columns, x, out, grid=(4, 1, 1), block=(32, 1, 1))
