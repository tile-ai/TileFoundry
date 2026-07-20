"""Shared codegen context for the CUDA target.

A lightweight source builder with backend-specific helpers (dtype mapping,
etc.). Emitter registration lives in tilefoundry.visitor_registry (the
canonical ``codegen_cuda_registry``); this module re-exports the decorator
+ a convenience ``lookup`` for back-compat.
"""

from __future__ import annotations

from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.ir.types import UnitType
from tilefoundry.visitor_registry.registries import (
    codegen_cuda_registry,
    register_codegen_cuda,
)

_CUDA_CPP: dict[str, str] = {
    "f32": "float",
    "bf16": "__nv_bfloat16",
    "f16": "half",
    "fp8e4m3": "__nv_fp8_e4m3",
    "i32": "int",
    "i64": "long long",
}


def topology_scope_str(name: str) -> str:
    """Map a topology level name to its C++ ``tilefoundry::TopologyScope``
    enumerator. Loud on an unknown level rather than silently defaulting."""
    scopes = {
        "cta": "tilefoundry::TopologyScope::cta",
        "warp": "tilefoundry::TopologyScope::warp",
        "thread": "tilefoundry::TopologyScope::thread",
    }
    try:
        return scopes[name]
    except KeyError:
        raise ValueError(
            f"unknown topology level {name!r}; expected one of {sorted(scopes)}"
        ) from None


def lookup(node_type: type):
    return codegen_cuda_registry.lookup(node_type)


class CodegenContext:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self._indent = 0
        self._var_names: dict[int, str] = {}
        self._counter = 0
        self._kernel_param_ids: set[int] = set()
        self._mesh_aliases: dict[int, str] = {}  # id(mesh) → alias + type_string
        # DimVar name → C++ runtime expression (typically a
        # ``<param>_shape_<axis>`` kernel scalar identifier). Populated
        # by the PrimFunction emitter before the body is walked so
        # downstream emitters (binary / unary / fill / copy) can size
        # their iteration counts at runtime when a DimVar appears.
        self._dim_var_runtime: dict[str, str] = {}
        # Named-barrier ids for sub-CTA ``T.sync``. Allocated implicitly per
        # kernel (reset at each PrimFunction body); id 0 is reserved for the
        # implicit whole-CTA barrier, so sub-CTA syncs draw from 1..15.
        self._next_barrier_id = 1

    def reset_barrier_ids(self) -> None:
        """Reset the named-barrier id counter at the start of a kernel body."""
        self._next_barrier_id = 1

    def alloc_barrier_id(self) -> int:
        """Allocate the next named-barrier id for a sub-CTA sync in this kernel.

        Hardware exposes ids 0..15; id 0 is reserved for the whole-CTA barrier,
        so 1..15 are available. Each emitted ``bar.sync`` draws a fresh id; a
        sync op node emits once, so a loop body reuses its id. Raises when a
        single kernel needs more distinct named barriers than the hardware has."""
        bid = self._next_barrier_id
        if bid > 15:
            raise ValueError(
                "T.sync: too many distinct named barriers in one kernel "
                "(hardware supports ids 1..15 for sub-CTA sync)"
            )
        self._next_barrier_id = bid + 1
        return bid

    def dtype_to_cpp(self, dtype_name: str) -> str:
        t = _CUDA_CPP.get(dtype_name)
        if t is None:
            raise ValueError(
                f"unsupported dtype for CUDA codegen: {dtype_name!r}"
            )
        return t

    def register_kernel_param(self, var) -> None:
        """Called by PrimFunction emitter before emitting the body — binds
        the param's original name and marks it as a pointer (float*)."""
        key = id(var)
        self._var_names[key] = var.name
        self._kernel_param_ids.add(key)

    def is_kernel_param(self, var) -> bool:
        return id(var) in self._kernel_param_ids

    def emit(self, line: str) -> None:
        self._lines.append("  " * self._indent + line)

    def blank(self) -> None:
        self._lines.append("")

    def indent(self) -> None:
        self._indent += 1

    def dedent(self) -> None:
        self._indent -= 1

    def name_for(self, var) -> str:
        key = id(var)
        if key in self._var_names:
            return self._var_names[key]
        self._counter += 1
        n = f"{var.name}_{self._counter}"
        self._var_names[key] = n
        return n

    def source(self) -> str:
        return "\n".join(self._lines) + "\n"

    def capture(self, fn) -> str:
        """Run ``fn(ctx)`` with ``ctx._lines`` temporarily swapped for a
        fresh buffer, returning the text emitted during the call. Indent
        and symbol tables are preserved across the swap; only the output
        buffer is isolated. Use this to render a stmt sub-sequence into a
        string that a Jinja template can splice in."""
        saved_lines = self._lines
        self._lines = []
        try:
            fn(self)
            return "\n".join(self._lines)
        finally:
            self._lines = saved_lines

    def emit_node(self, node) -> None:
        # Effect-ful Op invocation in Stmt position: Evaluate(op, args).
        # Dispatch on the Op class; the codegen handler ABI is Call-based, so
        # feed it a Call built from the Op and its args.
        if isinstance(node, Evaluate):
            op = node.callable
            op_cls = type(op)
            fn = lookup(op_cls)
            if fn is None:
                raise RuntimeError(
                    f"no @register_codegen_cuda for Op {op_cls.__name__}"
                )
            call = Call(type=UnitType(), target=op, args=node.args)
            fn(call, self)
            return
        fn = lookup(type(node))
        if fn is None:
            raise RuntimeError(
                f"no @register_codegen_cuda for {type(node).__name__}"
            )
        fn(node, self)


__all__ = ["CodegenContext", "register_codegen_cuda", "lookup"]
