"""CUDA device linkable-module emitter (split pipeline).

Emits the device ``.cu`` translation unit: the ``__global__`` kernels (identical
to the single-source path) plus, per kernel, an ``extern "C"`` launch shim that
performs the ``<<<grid, block, smem, stream>>>`` launch. Grid / block / smem /
stream arrive as plain C ABI arguments from the host module, so all CUDA types
stay inside this translation unit.
"""

from __future__ import annotations

from tilefoundry.codegen.linkable import LinkableModule
from tilefoundry.codegen.registry import CodeGenerator
from tilefoundry.codegen.signature import (
    TensorSignature,
)
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.target import Target


def _shim_ctype(fields):
    """A shim takes everything through the C ABI: a pointer or an integer."""

    def ctype(param: TensorSignature) -> str:
        if fields.param_kinds[param.name] == "tensor":
            return "void*"
        return "long long"

    return ctype


def _kernel_ctype(fields):
    """A kernel takes its buffers typed, so the body can index them."""

    def ctype(param: TensorSignature) -> str:
        if fields.param_kinds[param.name] != "tensor":
            return "int"
        return f"{fields.param_cpp_types[param.name]}*"

    return ctype


def emit_cuda_module(
    module: Module, cuda_fns: tuple[PrimFunction, ...], target: Target
) -> LinkableModule:
    """Emit the device ``.cu`` linkable module for *cuda_fns*."""
    raise NotImplementedError(
        "emit_cuda_module: the device emitter is being rebuilt. What was here "
        "precomputed a bag of fields per function and rendered a template from "
        "it; the visitor that walks the body is the emitter, and the call "
        "conventions are signatures, so neither needs the bag."
    )


CUDA_CODE_GENERATOR = CodeGenerator(emit_cuda_module)


__all__ = ["CUDA_CODE_GENERATOR", "emit_cuda_module"]
