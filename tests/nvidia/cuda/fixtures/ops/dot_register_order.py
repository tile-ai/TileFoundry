"""Dots whose lhs holds a (2, 4) block per lane beside a (2, 4) parameter."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda

_CUDA = cuda.Target("nvidia.h200_sxm")
_BLOCKS = Layout(shape=(32, 2, 4), strides=(8, 4, 1))


@module(entry="view_host", target=_CUDA, topologies=(Topology("thread", 32),))
class BlocksInMemory:
    """Each lane reads its block of ``a`` straight from global memory."""

    @prim_func(target=_CUDA)
    def view_device(a: Tensor[(32, 2, 4), "f32"], b: Tensor[(2, 4), "f32"], c: Tensor[(32,), "f32"]):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as mesh:
            blocks = T.tensor_view(T.ptr_of(a), layout=ShardLayout(layout=_BLOCKS, attrs=(Split(0),), mesh=mesh))
            lanes = T.tensor_view(
                T.ptr_of(c), layout=ShardLayout(layout=Layout(shape=(32,), strides=(1,)), attrs=(Split(0),), mesh=mesh)
            )
            T.dot(blocks, b, lanes)

    @prim_func(target=cpu.Target())
    def view_host(a: Tensor[(32, 2, 4), "f32"], b: Tensor[(2, 4), "f32"], c: Tensor[(32,), "f32"]):
        launch(view_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))


@module(entry="odd_rows_host", target=_CUDA, topologies=(Topology("thread", 32),))
class BlocksAtOddRows:
    """Rows nine elements apart, so most lanes' blocks start off a 16 byte boundary."""

    @prim_func(target=_CUDA)
    def odd_rows_device(a: Tensor[(288,), "f32"], b: Tensor[(2, 4), "f32"], c: Tensor[(32,), "f32"]):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as mesh:
            blocks = T.tensor_view(
                T.ptr_of(a),
                layout=ShardLayout(layout=Layout(shape=(32, 2, 4), strides=(9, 4, 1)), attrs=(Split(0),), mesh=mesh),
                shape=(32, 2, 4),
            )
            lanes = T.tensor_view(
                T.ptr_of(c), layout=ShardLayout(layout=Layout(shape=(32,), strides=(1,)), attrs=(Split(0),), mesh=mesh)
            )
            T.dot(blocks, b, lanes)

    @prim_func(target=cpu.Target())
    def odd_rows_host(a: Tensor[(288,), "f32"], b: Tensor[(2, 4), "f32"], c: Tensor[(32,), "f32"]):
        launch(odd_rows_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))


@module(entry="registers_host", target=_CUDA, topologies=(Topology("thread", 32),))
class BlocksInRegisters:
    """Each lane first copies its block of ``a`` into registers."""

    @prim_func(target=_CUDA)
    def registers_device(a: Tensor[(32, 2, 4), "f32"], b: Tensor[(2, 4), "f32"], c: Tensor[(32,), "f32"]):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as mesh:
            blocks = T.tensor_view(T.ptr_of(a), layout=ShardLayout(layout=_BLOCKS, attrs=(Split(0),), mesh=mesh))
            held = T.alloc_tensor(
                tensor_type=Tensor[(32, 2, 4), "f32", ShardLayout(layout=_BLOCKS, attrs=(Split(0),), mesh=mesh), "rmem"]
            )
            T.copy(blocks, held)
            lanes = T.tensor_view(
                T.ptr_of(c), layout=ShardLayout(layout=Layout(shape=(32,), strides=(1,)), attrs=(Split(0),), mesh=mesh)
            )
            T.dot(held, b, lanes)

    @prim_func(target=cpu.Target())
    def registers_host(a: Tensor[(32, 2, 4), "f32"], b: Tensor[(2, 4), "f32"], c: Tensor[(32,), "f32"]):
        launch(registers_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))


@module(entry="swizzled_host", target=_CUDA, topologies=(Topology("thread", 32),))
class BlocksInSwizzledShared:
    """Each lane first copies its block of ``a`` into a swizzled shared tile."""

    @prim_func(target=_CUDA)
    def swizzled_device(a: Tensor[(32, 2, 4), "f32"], b: Tensor[(2, 4), "f32"], c: Tensor[(32,), "f32"]):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as mesh:
            blocks = T.tensor_view(T.ptr_of(a), layout=ShardLayout(layout=_BLOCKS, attrs=(Split(0),), mesh=mesh))
            staged = T.alloc_tensor(
                tensor_type=Tensor[
                    (32, 2, 4),
                    "f32",
                    ShardLayout(
                        layout=ComposedLayout(inner=Swizzle(2, 2, 3), offset=0, outer=_BLOCKS),
                        attrs=(Split(0),),
                        mesh=mesh,
                    ),
                    "smem",
                ]
            )
            T.copy(blocks, staged)
            lanes = T.tensor_view(
                T.ptr_of(c), layout=ShardLayout(layout=Layout(shape=(32,), strides=(1,)), attrs=(Split(0),), mesh=mesh)
            )
            T.dot(staged, b, lanes)

    @prim_func(target=cpu.Target())
    def swizzled_host(a: Tensor[(32, 2, 4), "f32"], b: Tensor[(2, 4), "f32"], c: Tensor[(32,), "f32"]):
        launch(swizzled_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))
