"""Hidden shape-scalar param injection for hand-written dynamic-extent kernels.

A device ``@prim_func`` that reads a dynamic tensor dim (a ``DimVar`` axis) must
declare a hidden ``<param>_shape_<axis>`` i32 scalar so codegen can plumb the
runtime extent — the same ABI the HIR→TIR lowering appends. Host entries read
shapes from their tensor args, so they must stay unpolluted; static device
kernels get no hidden scalars.
"""
from __future__ import annotations

from tilefoundry import prim_func
from tilefoundry.dsl import DimVar, T, Tensor
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.ir.types.storage import StorageKind

_TILE = 12
_NT = DimVar("Ntile", 1, 64)


def _shape_scalars(pf) -> list[str]:
    return [p.name for p in pf.params if "_shape_" in p.name]


def test_device_dynamic_dim_injects_shape_scalar() -> None:
    @prim_func(target="cuda")
    def dev(a: Tensor[(_NT, _TILE), "f32"]):
        with Mesh(Topology("cta", None), Layout(shape=(None,), strides=(1,))) as cta:
            a_view = T.tensor_view(
                a,
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

    assert _shape_scalars(dev) == ["a_shape_0"]
    scalar = next(p for p in dev.params if p.name == "a_shape_0")
    assert isinstance(scalar.type, TensorType)
    assert scalar.type.shape == ()
    assert scalar.type.dtype is DType.i32


def test_host_entry_not_polluted() -> None:
    """A ``cpu`` host entry carrying the same DimVar param gets no hidden scalar
    — it reads the shape from its tensor argument at launch time."""

    @prim_func(target="cuda")
    def dev(a: Tensor[(_NT, _TILE), "f32"]):
        with Mesh(Topology("cta", None), Layout(shape=(None,), strides=(1,))) as cta:
            a_view = T.tensor_view(
                a,
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

    @prim_func(target="cpu")
    def host(a: Tensor[(_NT, _TILE), "f32"]):
        launch(dev, a, grid=(_NT, 1, 1), block=(1, 1, 1))  # noqa: F821

    assert _shape_scalars(host) == []
    assert [p.name for p in host.params] == ["a"]


def test_static_device_kernel_has_no_shape_scalar() -> None:
    """A device kernel with only static dims gets no hidden scalars."""

    @prim_func(target="cuda")
    def dev(a: Tensor[(16, 8), "f32"]):
        with Mesh(Topology("thread", 8), Layout(shape=(8,), strides=(1,))) as t:
            a_view = T.tensor_view(
                a,
                layout=ShardLayout(
                    layout=Layout(shape=(16, 8), strides=(8, 1)),
                    attrs=(Split(0),),
                    mesh=t,
                ),
            )
            reg = T.alloc_tensor(
                TensorType(
                    shape=(16, 8),
                    dtype=DType.f32,
                    layout=ShardLayout(
                        layout=Layout(shape=(16, 8), strides=(8, 1)),
                        attrs=(Split(0),),
                        mesh=t,
                    ),
                    storage=StorageKind.RMEM,
                )
            )
            T.copy(a_view, reg)

    assert _shape_scalars(dev) == []
