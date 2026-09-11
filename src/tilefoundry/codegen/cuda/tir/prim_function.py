"""Emit a TIR primitive function's CUDA kernel and host wrapper.

The wrapper accepts TVM tensors, extracts pointers and hidden integer shape
scalars, and derives launch dimensions from mesh scopes. A specialization entry
containing only dispatch emits no global kernel; its host wrapper selects and
calls variant wrappers because CUDA kernels cannot call host code.
See [codegen §1](docs/spec/codegen.md#1-pipeline).
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext
from tilefoundry.codegen.cuda.tir.memory.tensor_view import render_shard_layout_value
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.shard_layout import ShardLayout


def _param_wrapper(name: str, total: int, cpp_type: str) -> str:
    layout = f"cute::make_layout(cute::Shape<cute::Int<{total}>>{{}})"
    return f"auto {name}_tensor = cute::make_tensor(cute::make_gmem_ptr({name}), {layout});"


def _param_cpp_types(params: tuple, ctx: CodegenContext) -> dict[str, str]:
    """Map each param name → CUDA C++ type from its TensorType dtype."""
    result: dict[str, str] = {}
    for p in params:
        ty = p.type
        if isinstance(ty, TensorType):
            result[p.name] = ctx.dtype_to_cpp(ty.dtype.name)
        else:
            result[p.name] = "float"
    return result


def _param_wrapper_shard(
    name: str, total: int, shard_layout: ShardLayout, dim_var_runtime=None
) -> str:
    """Emit make_shard_tensor wrapping a kernel param with ShardLayout value."""
    global_layout = f"cute::make_layout(cute::Shape<cute::Int<{total}>>{{}})"
    tensor_ref = f"{name}_tensor"
    preamble, shard_value = render_shard_layout_value(
        tensor_ref, shard_layout, dim_var_runtime, None, ctx
    )
    wrapper = (
        f"auto {tensor_ref} = tilefoundry::make_shard_tensor("
        f"cute::make_tensor(cute::make_gmem_ptr({name}), {global_layout}), "
        f"{global_layout}, {shard_value});"
    )
    return "\n".join([*preamble, wrapper])
