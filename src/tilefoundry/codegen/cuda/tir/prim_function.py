"""Emitter for ``tir.PrimFunction`` — produces the `__global__` kernel plus
a ``tvm::ffi::Tensor``-parameterised host wrapper.

Host entry signature and launch config follow ``docs/spec/codegen.md §6``:
- Wrapper parameters are ``tvm::ffi::Tensor``; raw pointers are extracted
  via ``.data_ptr<float>()`` before the kernel launch.
- Grid / block dims are derived from the outermost ``MeshScope`` topologies
  (``cta`` → grid, ``thread`` → block).

Shape-scalar params (rank-0 ``i32`` tensors named ``<src>_shape_<axis>``)
are passed-by-value ``int`` kernel scalars. They are NOT user-facing — the
host wrapper extracts them from the corresponding tensor argument's shape
before the kernel launch.

If the PrimFunction body is a single ``DispatchCall`` (the entry of a
specialization group), no ``__global__`` kernel is emitted; the host
wrapper alone holds the dispatch if-chain that forwards to mangled
host wrappers. CUDA forbids calling host code from ``__global__`` —
host-side dispatch is the simplest legal lowering.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.codegen.cuda.templates import render
from tilefoundry.codegen.cuda.tir.memory.tensor_view import render_shard_layout_value
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import (
    is_hidden_shape_scalar as _is_hidden_shape_scalar,
)
from tilefoundry.ir.tir.shape import (
    parse_shape_var_name as _parse_shape_param_name,
)
from tilefoundry.ir.tir.shape import (
    shape_var_name,
)
from tilefoundry.ir.tir.stmts import Return, Sequential
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shape_helpers import upper_bound
from tilefoundry.ir.types.shard.shard_layout import ShardLayout


def _total(shape) -> int:
    if not shape:
        return 1
    # ``upper_bound`` reduces a ``DimVar`` axis to its maximum runtime value
    # (``hi - 1``, since the envelope ``[lo, hi)`` is half-open) so the
    # kernel-param cute layout stays large enough for any runtime shape that
    # flows through the dispatch.
    # The actual runtime extent is plumbed via the separate
    # ``<param>_shape_<axis>`` kernel scalars produced by the dispatch
    # lowering.
    return reduce(mul, (upper_bound(s) for s in shape), 1)


def _internal_wrapper_symbol(kernel_name: str) -> str:
    """Map a user-facing kernel name to its internal C++ wrapper symbol.

    The user-facing name may be ``main`` (collides with ``::main``) or
    a mangled variant like ``main$S$1_4`` (``$`` is a GCC extension,
    not portable). The internal symbol is always a plain C++ identifier
    so the generated source compiles under strict toolchains.
    """
    return "__tilefoundry_" + kernel_name.replace("$", "__") + "_host"


def _param_wrapper(name: str, total: int, cpp_type: str) -> str:
    layout = f"cute::make_layout(cute::Shape<cute::Int<{total}>>{{}})"
    return (
        f"auto {name}_tensor = cute::make_tensor("
        f"cute::make_gmem_ptr({name}), {layout});"
    )


def _param_cpp_types(params: tuple, ctx: CodegenContext) -> dict[str, str]:
    """Map each param name → CUDA C++ type from its TensorType dtype."""
    result: dict[str, str] = {}
    for p in params:
        ty = p.type
        if isinstance(ty, TensorType):
            result[p.name] = ctx.dtype_to_cpp(ty.dtype.name)
        else:
            result[p.name] = "float"  # fallback
    return result


def _param_wrapper_shard(
    name: str, total: int, shard_layout: ShardLayout, dim_var_runtime=None
) -> str:
    """Emit make_shard_tensor wrapping a kernel param with ShardLayout value."""
    global_layout = f"cute::make_layout(cute::Shape<cute::Int<{total}>>{{}})"
    tensor_ref = f"{name}_tensor"
    preamble, shard_value = render_shard_layout_value(
        tensor_ref, shard_layout, dim_var_runtime
    )
    wrapper = (
        f"auto {tensor_ref} = tilefoundry::make_shard_tensor("
        f"cute::make_tensor(cute::make_gmem_ptr({name}), {global_layout}), "
        f"{global_layout}, {shard_value});"
    )
    return "\n".join([*preamble, wrapper])


def _collect_mesh_dims(body: Sequential) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return the ``(grid, block)`` launch config for ``body``.

    Forwards to ``_derive_launch_config`` in
    ``tilefoundry.codegen.cuda.emit`` so the multi-topology walk is not
    duplicated here.
    """
    # noqa cycle: emit.py auto-discovers this module via importlib, so
    # importing emit at top-level would be a real circular import.
    from tilefoundry.codegen.cuda.emit import _derive_launch_config  # noqa: PLC0415
    # Device-fragment codegen accepts a launch-provided (dynamic) CTA extent:
    # the grid comes from the host launch, so a dynamic cta is reported as
    # ``grid=(None, 1, 1)``. Static-grid callers (auto-entry, single-source,
    # dispatch host) check for ``None`` and error at their own site.
    return _derive_launch_config(body)


def _is_dispatch_entry(node: PrimFunction) -> bool:
    """A dispatch entry PrimFunction's body is ``Sequential`` whose
    statements consist of one ``DispatchCall`` plus an optional trailing
    ``Return``. The HIR→TIR lowering produces exactly this shape for
    overload-group entries.
    """
    body = node.body
    if not isinstance(body, Sequential):
        return False
    return any(isinstance(s, DispatchCall) for s in body.body)


def _is_dispatch_entry_shape(node: PrimFunction) -> bool:
    """Narrow dispatch-entry recognition: the body is exactly
    ``Sequential((DispatchCall,))`` or ``Sequential((DispatchCall, Return))``.

    Stricter than :func:`_is_dispatch_entry` (which matches any body that
    merely contains a ``DispatchCall``); used where a host-only dispatch entry
    must be recognized precisely rather than guessed."""
    body = node.body
    if not isinstance(body, Sequential):
        return False
    stmts = body.body
    if not stmts or not isinstance(stmts[0], DispatchCall):
        return False
    if len(stmts) == 1:
        return True
    return len(stmts) == 2 and isinstance(stmts[1], Return)


def _has_nested_dispatch(node: PrimFunction) -> bool:
    """Walk *node*'s body and return True if any nested stmt is a
    ``DispatchCall``. Used to guard non-entry PrimFunctions, where
    a nested dispatch would require host-side dispatch from inside a
    ``__global__`` kernel — not supported in v0.
    """
    body = node.body
    if not isinstance(body, Sequential):
        return False
    for stmt in body.body:
        if isinstance(stmt, DispatchCall):
            return True
    return False


@dataclass(frozen=True)
class _KernelFields:
    """Per-PrimFunction codegen pieces shared by the single-source kernel
    emitter and the split device / host fragment emitters."""

    kernel_name: str
    internal_wrapper_name: str
    params: tuple
    param_cpp_types: dict
    param_kinds: dict  # name -> "tensor" | "hidden_scalar" | "user_scalar"
    kernel_params_sig: str
    wrapper_params_sig: str
    user_params: tuple
    launch_args: str
    param_wrappers: list
    wrapper_locals: list
    kernel_body: str
    grid: tuple
    block: tuple
    entry_host_only: bool


def _compute_kernel_fields(node: PrimFunction, ctx: CodegenContext) -> _KernelFields:
    for p in node.params:
        ctx.register_kernel_param(p)

    # Register DimVar -> runtime kernel-scalar expression for every
    # DimVar that appears in a tensor param's shape. Downstream emitters
    # (arith / fill / copy / alloc) use this mapping to size runtime
    # iteration counts from the live shape scalars instead of the
    # compile-time envelope upper bound.
    ctx._dim_var_runtime = {}
    for p in node.params:
        ty = p.type
        if not isinstance(ty, TensorType):
            continue
        for axis, dim in enumerate(ty.shape):
            if isinstance(dim, DimVar) and dim.name not in ctx._dim_var_runtime:
                ctx._dim_var_runtime[dim.name] = shape_var_name(p.name, axis)

    # Named-barrier ids are per-kernel: reset before walking this body.
    ctx.reset_barrier_ids()

    entry_host_only = _is_dispatch_entry(node)
    if not entry_host_only and _has_nested_dispatch(node):
        # v0 restriction: a DispatchCall inside a non-entry PrimFunction
        # body would have to dispatch from a ``__global__`` kernel into
        # host wrappers — which CUDA forbids. Only dispatch at the entry
        # is supported until device-callable variant dispatch lands.
        raise NotImplementedError(
            f"PrimFunction emitter: v0 nested dispatch inside a specialized "
            f"kernel is not yet supported (function {node.name!r})."
        )

    # Capture stmt-body emission into a fresh buffer — Python walker still
    # drives per-stmt emission; the resulting string becomes a template var.
    def _emit_body(inner: CodegenContext) -> None:
        inner.emit_node(node.body)

    body = ctx.capture(_emit_body)

    param_cpp_types = _param_cpp_types(node.params, ctx)
    hidden_shape = tuple(
        p for p in node.params if _is_hidden_shape_scalar(p, node.params)
    )
    hidden_names = {p.name for p in hidden_shape}
    user_params = tuple(p for p in node.params if p.name not in hidden_names)

    # Kernel signature: hidden shape scalars are pass-by-value ``int``;
    # rank-0 i32 user params are also pass-by-value ``int``;
    # other params are typed raw pointers.
    def _is_user_scalar(p) -> bool:
        return (
            p.name not in hidden_names
            and isinstance(p.type, TensorType)
            and not p.type.shape
        )

    def _kind(p) -> str:
        if p.name in hidden_names:
            return "hidden_scalar"
        if _is_user_scalar(p):
            return "user_scalar"
        return "tensor"

    param_kinds = {p.name: _kind(p) for p in node.params}

    def _kernel_sig_token(p) -> str:
        if param_kinds[p.name] != "tensor":
            return f"int {p.name}"
        return f"{param_cpp_types[p.name]}* {p.name}"

    kernel_params_sig = ", ".join(_kernel_sig_token(p) for p in node.params)

    # Host wrapper: every user-facing param surfaces here. Rank-0 i32
    # user params come in as ``int`` (pass-by-value, matching the
    # shape-scalar convention). Tensor params come in as ``tvm::ffi::Tensor``.
    def _wrapper_param_token(p) -> str:
        if param_kinds[p.name] == "user_scalar":
            return f"int {p.name}"
        return f"tvm::ffi::Tensor {p.name}"

    wrapper_params_sig = ", ".join(_wrapper_param_token(p) for p in user_params)
    wrapper_locals = []
    for p in hidden_shape:
        parsed = _parse_shape_param_name(p.name)
        # _is_hidden_shape_scalar guarantees this parses; assert for safety.
        assert parsed is not None
        base, axis = parsed
        wrapper_locals.append(
            f"int {p.name} = static_cast<int>({base}.shape()[{axis}]);"
        )

    # Launch args: tensor params extract a raw pointer; scalar params
    # (hidden or user-declared) are forwarded by name as plain ``int``.
    def _launch_arg(p) -> str:
        if param_kinds[p.name] != "tensor":
            return p.name
        return f"static_cast<{param_cpp_types[p.name]}*>({p.name}.data_ptr())"

    launch_args = ", ".join(_launch_arg(p) for p in node.params)

    # cute tensor wrappers are only meaningful for tensor params; scalar
    # params are plain ``int`` and stay as-is in the kernel body.
    buffer_params = tuple(
        p for p in user_params if param_kinds[p.name] == "tensor"
    )
    param_wrappers = []
    for p in buffer_params:
        total = _total(p.type.shape)
        sl = getattr(p.type, "layout", None)
        if isinstance(sl, ShardLayout):
            param_wrappers.append(
                _param_wrapper_shard(p.name, total, sl, ctx._dim_var_runtime)
            )
        else:
            param_wrappers.append(
                _param_wrapper(p.name, total, param_cpp_types[p.name])
            )

    grid, block = _collect_mesh_dims(node.body)

    return _KernelFields(
        kernel_name=node.name,
        internal_wrapper_name=_internal_wrapper_symbol(node.name),
        params=node.params,
        param_cpp_types=param_cpp_types,
        param_kinds=param_kinds,
        kernel_params_sig=kernel_params_sig,
        wrapper_params_sig=wrapper_params_sig,
        user_params=user_params,
        launch_args=launch_args,
        param_wrappers=param_wrappers,
        wrapper_locals=wrapper_locals,
        kernel_body=body,
        grid=grid,
        block=block,
        entry_host_only=entry_host_only,
    )


@register_codegen_cuda(PrimFunction)
def _emit(node: PrimFunction, ctx: CodegenContext) -> None:
    f = _compute_kernel_fields(node, ctx)
    text = render(
        "kernel.cu.j2",
        kernel_name=f.kernel_name,
        internal_wrapper_name=f.internal_wrapper_name,
        kernel_params_sig=f.kernel_params_sig,
        wrapper_params_sig=f.wrapper_params_sig,
        launch_args=f.launch_args,
        param_wrappers=f.param_wrappers,
        wrapper_locals=f.wrapper_locals,
        kernel_body=f.kernel_body,
        grid_x=f.grid[0],
        grid_y=f.grid[1],
        grid_z=f.grid[2],
        block_x=f.block[0],
        block_y=f.block[1],
        block_z=f.block[2],
        entry_host_only=f.entry_host_only,
    )
    for line in text.rstrip("\n").split("\n"):
        ctx._lines.append(line)
