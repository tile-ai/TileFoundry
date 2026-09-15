"""Host lowering checks for entries containing several CUDA launches."""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tilefoundry import module, prim_func
from tilefoundry.codegen.context_builder import build_codegen_context
from tilefoundry.codegen.cpu.context import CpuCodegenContext
from tilefoundry.codegen.cpu.module import emit_host_module
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types import CallableType, DType, TensorType, UnitType
from tilefoundry.ir.types.shard import B, Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="host", target=CudaTarget("nvidia.h200_sxm"))
class _TwoCopyLaunches:
    """Two concrete kernels used by the end-to-end multi-launch check."""

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def first(x: Tensor[(1, 128), "f32"], out: Tensor[(1, 128), "f32"]):
        with Mesh((Topology("thread", 1),), Layout((1,), (1,))) as thread:
            view = T.tensor_view(
                x,
                layout=ShardLayout(
                    layout=Layout((1, 128), (128, 1)),
                    attrs=(B(),),
                    mesh=Mesh((Topology("thread", 1),), Layout((1,), (1,))),
                ),
            )
            out_view = T.tensor_view(
                out,
                layout=ShardLayout(
                    layout=Layout((1, 128), (128, 1)),
                    attrs=(B(),),
                    mesh=Mesh((Topology("thread", 1),), Layout((1,), (1,))),
                ),
            )
            T.copy(view, out_view)
            T.sync(thread)

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def second(x: Tensor[(1, 128), "f32"], out: Tensor[(1, 128), "f32"]):
        with Mesh((Topology("thread", 1),), Layout((1,), (1,))) as thread:
            view = T.tensor_view(
                x,
                layout=ShardLayout(
                    layout=Layout((1, 128), (128, 1)),
                    attrs=(B(),),
                    mesh=Mesh((Topology("thread", 1),), Layout((1,), (1,))),
                ),
            )
            out_view = T.tensor_view(
                out,
                layout=ShardLayout(
                    layout=Layout((1, 128), (128, 1)),
                    attrs=(B(),),
                    mesh=Mesh((Topology("thread", 1),), Layout((1,), (1,))),
                ),
            )
            T.copy(view, out_view)
            T.sync(thread)

    @prim_func(target=CpuTarget())
    def host(
        x0: Tensor[(1, 128), "f32"],
        out0: Tensor[(1, 128), "f32"],
        x1: Tensor[(1, 128), "f32"],
        out1: Tensor[(1, 128), "f32"],
    ):
        launch(first, x0, out0, grid=(1, 1, 1), block=(1, 1, 1))  # noqa: F821
        launch(second, x1, out1, grid=(1, 1, 1), block=(1, 1, 1))  # noqa: F821


def _tensor(name: str, shape=(8,)) -> Var:
    return Var(
        name=name,
        type=TensorType(
            shape=shape,
            dtype=DType.f32,
            layout=None,
            storage=StorageKind.GMEM,
        ),
    )


def _device(name: str) -> PrimFunction:
    return PrimFunction(
        name=name,
        params=(_tensor("a"),),
        body=Sequential(body=()),
        target=CudaTarget("nvidia.h200_sxm"),
    )


def _launch(device: PrimFunction, *args: Var, block: int = 32) -> Evaluate:
    ref = SymbolRef(
        name=device.name,
        type=CallableType(return_type=UnitType(), parameters=tuple(p.type for p in device.params)),
    )
    i64 = TensorType.scalar(DType.i64, storage=StorageKind.RMEM)
    one = Constant(type=i64, value=1)
    block_expr = Constant(type=i64, value=block)
    return Evaluate(
        callable=Launch(),
        args=(ref, one, one, one, block_expr, one, one, *args),
    )


def _module(devices, launches, params) -> tuple[Module, PrimFunction]:
    host = PrimFunction(
        name="host",
        params=tuple(params),
        body=Sequential(body=tuple(launches)),
        target=CpuTarget(),
    )
    return Module(name="multi", functions=(*devices, host), entry="host"), host


def _source(devices, launches, params) -> str:
    mod, host = _module(devices, launches, params)
    root = build_codegen_context(mod)
    group = next(group for group in root.groups if host in group.functions)
    view = root.for_group(group)
    ctx = CpuCodegenContext(
        symbols=view.symbols,
        target=host.target,
        resolved_launches=view.resolved_launches,
    )
    return emit_host_module(mod, (host,), host.target, ctx).source


def test_two_launches_keep_order_and_each_block_size() -> None:
    a, b = _tensor("a"), _tensor("b")
    first, second = _device("first"), _device("second")
    source = _source(
        (first, second),
        (_launch(first, a, block=32), _launch(second, b, block=64)),
        (a, b),
    )
    assert source.index("tilefoundry_first_launch") < source.index("tilefoundry_second_launch")
    assert "1, 32, 0" in source
    assert "1, 64, 0" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_two_launches_compile_and_write_both_outputs() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(1, 128, device="cuda")
    x1 = torch.randn(1, 128, device="cuda")
    out0 = torch.empty_like(x0)
    out1 = torch.empty_like(x1)
    runtime = tilefoundry.compile(_TwoCopyLaunches, target=CudaTarget("nvidia.h200_sxm"))
    runtime(x0, out0, x1, out1)
    torch.cuda.synchronize()
    torch.testing.assert_close(out0, x0, rtol=0, atol=0)
    torch.testing.assert_close(out1, x1, rtol=0, atol=0)


def test_same_entry_parameter_may_feed_two_launches() -> None:
    a = _tensor("a")
    first, second = _device("first"), _device("second")
    source = _source((first, second), (_launch(first, a), _launch(second, a)), (a,))
    assert source.count("a.data_ptr()") == 2


def test_a_launch_arg_the_entry_does_not_declare_is_rejected() -> None:
    device = _device("device")
    outside = _tensor("outside")
    with pytest.raises(ValueError, match="is not a parameter of entry"):
        _source((device,), (_launch(device, outside),), (_tensor("a"),))


def test_same_device_function_can_be_launched_twice() -> None:
    a = _tensor("a")
    device = _device("device")
    source = _source((device,), (_launch(device, a, block=32), _launch(device, a, block=64)), (a,))
    assert source.count('extern "C" void tilefoundry_device_launch') == 1
    assert source.count("tilefoundry_device_launch(a.data_ptr()") == 2


def test_a_parameter_named_like_an_extent_is_still_a_parameter() -> None:
    """An open axis is stated by the type, so no name is reserved for one."""
    a, looks_hidden = _tensor("a"), _tensor("a_shape_0")
    device = PrimFunction(
        name="device",
        params=(a, looks_hidden),
        body=Sequential(body=()),
        target=CudaTarget("nvidia.h200_sxm"),
    )
    source = _source((device,), (_launch(device, a, looks_hidden),), (a, looks_hidden))
    assert "tvm::ffi::Tensor a_shape_0" in source
    assert "tilefoundry_device_launch(a.data_ptr(), a_shape_0.data_ptr()" in source


def test_single_launch_path_remains_available() -> None:
    a = _tensor("a")
    device = _device("device")
    source = _source((device,), (_launch(device, a),), (a,))
    assert "tilefoundry_device_launch" in source
