"""Host lowering checks for entries containing several CUDA launches."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tilefoundry import cli, codegen
from tilefoundry.dsl import *
from tilefoundry.ir import core, expr, tir
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda
from tilefoundry_nvidia.cuda import codegen as cuda_codegen

CODEGEN = Path(__file__).parents[1] / "fixtures" / "codegen"


def _tensor(name: str, shape=(8,)) -> expr.Var:
    return expr.Var(
        name=name,
        type=TensorType(
            shape=shape,
            dtype=DType.f32,
            layout=None,
            storage=StorageKind.GMEM,
        ),
    )


def _device(name: str) -> core.PrimFunction:
    return core.PrimFunction(
        name=name,
        params=(_tensor("a"),),
        body=core.Sequential(body=()),
        target=cuda.Target("nvidia.h200_sxm"),
    )


def _launch(device: core.PrimFunction, *args: expr.Var, block: int = 32) -> core.Evaluate:
    ref = core.SymbolRef(
        name=device.name,
        type=CallableType(return_type=UnitType(), parameters=tuple(p.type for p in device.params)),
    )
    i64 = TensorType.from_scalar(DType.i64, storage=StorageKind.RMEM)
    one = expr.Constant(type=i64, value=1)
    block_expr = expr.Constant(type=i64, value=block)
    return core.Evaluate(
        callable=tir.launch.Launch(),
        args=(ref, one, one, one, block_expr, one, one, *args),
    )


def _module(devices, launches, params) -> tuple[core.Module, core.PrimFunction]:
    host = core.PrimFunction(
        name="host",
        params=tuple(params),
        body=core.Sequential(body=tuple(launches)),
        target=cpu.Target(),
    )
    return core.Module(name="multi", functions=(*devices, host), entry="host"), host


def _source(devices, launches, params) -> str:
    mod, host = _module(devices, launches, params)
    root = codegen.build_context(mod)
    group = next(group for group in root.groups if host in group.functions)
    view = root.for_group(group)
    return host.target.resolve(codegen.LinkableModule, mod, (host,), host.target.resolve(codegen.EmitContext, view)).source


def test_launches_keep_order_and_blocks() -> None:
    a, b = _tensor("a"), _tensor("b")
    first, second = _device("first"), _device("second")
    source = _source(
        (first, second),
        (_launch(first, a, block=32), _launch(second, b, block=64)),
        (a, b),
    )
    mod, _ = _module((first, second), (), ())
    assert source.index(cuda_codegen.launcher_symbol(codegen.mangle(mod, first))) < source.index(
        cuda_codegen.launcher_symbol(codegen.mangle(mod, second))
    )
    assert "1, 32, 0" in source
    assert "1, 64, 0" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_two_launches_write_outputs() -> None:
    """Both compiled objects retain their callable and library after registration override."""
    _TwoCopyLaunches = cli.load_authored_ir(f"{CODEGEN / 'two_copy_launches.py'}:_TwoCopyLaunches")
    torch.manual_seed(0)
    x0 = torch.randn(1, 128, device="cuda")
    x1 = torch.randn(1, 128, device="cuda")
    out0 = torch.empty_like(x0)
    out1 = torch.empty_like(x1)
    runtimes = [
        tilefoundry.compile(_TwoCopyLaunches, target=cuda.Target("nvidia.h200_sxm"))
        for _ in range(2)
    ]
    for runtime in runtimes:
        out0.fill_(float("nan"))
        out1.fill_(float("nan"))
        runtime(x0, out0, x1, out1)
        torch.cuda.synchronize()
        torch.testing.assert_close(out0, x0, rtol=0, atol=0)
        torch.testing.assert_close(out1, x1, rtol=0, atol=0)


def test_entry_parameter_feeds_two_launches() -> None:
    a = _tensor("a")
    first, second = _device("first"), _device("second")
    source = _source((first, second), (_launch(first, a), _launch(second, a)), (a,))
    assert source.count("a.data_ptr()") == 2


def test_undeclared_launch_arg_rejected() -> None:
    device = _device("device")
    outside = _tensor("outside")
    with pytest.raises(ValueError, match="is not a parameter of entry"):
        _source((device,), (_launch(device, outside),), (_tensor("a"),))


def test_device_function_launches_twice() -> None:
    a = _tensor("a")
    device = _device("device")
    source = _source((device,), (_launch(device, a, block=32), _launch(device, a, block=64)), (a,))
    mod, _ = _module((device,), (), ())
    symbol = cuda_codegen.launcher_symbol(codegen.mangle(mod, device))
    assert source.count(f'extern "C" const char* {symbol}') == 1
    assert source.count(f"{symbol}(a.data_ptr()") == 2


def test_extent_like_name_stays_parameter() -> None:
    """An open axis is stated by the type, so no name is reserved for one."""
    a, looks_hidden = _tensor("a"), _tensor("a_shape_0")
    device = core.PrimFunction(
        name="device",
        params=(a, looks_hidden),
        body=core.Sequential(body=()),
        target=cuda.Target("nvidia.h200_sxm"),
    )
    source = _source((device,), (_launch(device, a, looks_hidden),), (a, looks_hidden))
    assert "tvm::ffi::Tensor a_shape_0" in source
    mod, _ = _module((device,), (), ())
    symbol = cuda_codegen.launcher_symbol(codegen.mangle(mod, device))
    assert f"{symbol}(a.data_ptr(), a_shape_0.data_ptr()" in source


def test_single_launch_path_remains_available() -> None:
    a = _tensor("a")
    device = _device("device")
    source = _source((device,), (_launch(device, a),), (a,))
    assert cuda_codegen.launcher_symbol(codegen.mangle(_module((device,), (), ())[0], device)) in source
