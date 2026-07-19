"""Runtime wide-load fast path inside ``copy()``.

A gmem→rmem shard copy whose per-thread fragment is a static, contiguous run of
at least 128 bits is loaded as 128-bit vectors; a sub-128-bit fragment (or a
non-gmem source / unaligned / strided source) falls back to the element loop.
The fast path is selected purely from the operands — no new IR / DSL surface —
and produces results identical to the scalar path.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 — trigger emitter autodiscovery
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.ir.types.storage import StorageKind


# 4 f32 = 128 bits -> the vector path; each thread owns one contiguous,
# 16B-aligned row.
@module(entry="wide_host")
class WideLoad:
    @prim_func(target="cuda")
    def wide_device(a: Tensor[(128, 4), "f32"], b: Tensor[(128, 4), "f32"]):
        with Mesh(Topology("thread", 128), Layout(shape=(128,), strides=(1,)), ("t",)) as m:
            a_view = T.tensor_view(
                a, layout=ShardLayout(layout=Layout(shape=(128, 4), strides=(4, 1)), attrs=(Split(0),), mesh=m)
            )
            b_view = T.tensor_view(
                b, layout=ShardLayout(layout=Layout(shape=(128, 4), strides=(4, 1)), attrs=(Split(0),), mesh=m)
            )
            reg = T.alloc_tensor(
                TensorType(
                    shape=(128, 4),
                    dtype=DType.f32,
                    layout=ShardLayout(layout=Layout(shape=(128, 4), strides=(4, 1)), attrs=(Split(0),), mesh=m),
                    storage=StorageKind.RMEM,
                )
            )
            T.copy(a_view, reg)     # gmem -> rmem: 128-bit vector load
            T.copy(reg, b_view)     # rmem -> gmem

    @prim_func(target="cpu")
    def wide_host(a: Tensor[(128, 4), "f32"], b: Tensor[(128, 4), "f32"]):
        launch(wide_device, a, b, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821


# 2 f32 = 64 bits -> sub-128-bit, so the fast path is not selected and the copy
# falls back to the scalar element loop.
@module(entry="narrow_host")
class NarrowLoad:
    @prim_func(target="cuda")
    def narrow_device(a: Tensor[(128, 2), "f32"], b: Tensor[(128, 2), "f32"]):
        with Mesh(Topology("thread", 128), Layout(shape=(128,), strides=(1,)), ("t",)) as m:
            a_view = T.tensor_view(
                a, layout=ShardLayout(layout=Layout(shape=(128, 2), strides=(2, 1)), attrs=(Split(0),), mesh=m)
            )
            b_view = T.tensor_view(
                b, layout=ShardLayout(layout=Layout(shape=(128, 2), strides=(2, 1)), attrs=(Split(0),), mesh=m)
            )
            reg = T.alloc_tensor(
                TensorType(
                    shape=(128, 2),
                    dtype=DType.f32,
                    layout=ShardLayout(layout=Layout(shape=(128, 2), strides=(2, 1)), attrs=(Split(0),), mesh=m),
                    storage=StorageKind.RMEM,
                )
            )
            T.copy(a_view, reg)     # gmem -> rmem: scalar fallback (64-bit)
            T.copy(reg, b_view)     # rmem -> gmem

    @prim_func(target="cpu")
    def narrow_host(a: Tensor[(128, 2), "f32"], b: Tensor[(128, 2), "f32"]):
        launch(narrow_device, a, b, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "cls,cols", [(WideLoad, 4), (NarrowLoad, 2)], ids=["wide_128b", "narrow_64b_fallback"]
)
def test_wide_load_roundtrip_matches(cls, cols):
    """Both the 128-bit vector path and the sub-128-bit scalar fallback copy the
    input through the rmem fragment unchanged."""
    rm = tilefoundry.compile(cls, target="cuda")
    torch.manual_seed(0)
    a = torch.randn(128, cols, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    rm(a, b)
    torch.cuda.synchronize()
    assert torch.allclose(b, a, rtol=0, atol=0)


def _device_sass(cls) -> str:
    """Compile a module's CUDA device source to a cubin and return its SASS."""
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.linker import (  # noqa: PLC0415
        _DEFAULT_CUTLASS_INCLUDE,
        _DEFAULT_INCLUDE,
    )
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    src = emit_cuda_module(
        group_functions_by_target(tilefoundry.lower(cls, target="cuda"))["cuda"]
    ).source
    with tempfile.TemporaryDirectory() as d:
        cu, cubin = Path(d) / "device.cu", Path(d) / "device.cubin"
        cu.write_text(src)
        subprocess.run(
            ["nvcc", "-std=c++17", "-arch=sm_80", "-cubin",
             "-Wno-deprecated-gpu-targets", "-DTILEFOUNDRY_TARGET_CUDA",
             "-I", str(_DEFAULT_INCLUDE), "-I", str(_DEFAULT_CUTLASS_INCLUDE),
             str(cu), "-o", str(cubin)],
            check=True, capture_output=True, text=True,
        )
        return subprocess.run(
            ["cuobjdump", "-sass", str(cubin)],
            check=True, capture_output=True, text=True,
        ).stdout


@pytest.mark.skipif(
    not (shutil.which("nvcc") and shutil.which("cuobjdump")),
    reason="requires nvcc + cuobjdump",
)
def test_wide_load_emits_128bit_load_only_when_qualified():
    """SASS proof that the fast path is actually emitted: the 128-bit fragment
    lowers to a `LDG.E.128` vector load, while the 64-bit fragment does not (it
    stays on the scalar element loop)."""
    assert "LDG.E.128" in _device_sass(WideLoad)
    assert "LDG.E.128" not in _device_sass(NarrowLoad)
