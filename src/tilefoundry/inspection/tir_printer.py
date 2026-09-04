"""Canonical Python DSL printer for TIR PrimFunction and Module values."""

from __future__ import annotations

import enum
import re

from tilefoundry.inspection._python_render import mesh_str, tensor_annotation
from tilefoundry.ir.core import Call, Constant, Expr, Op, Tuple, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import (
    Abort,
    Evaluate,
    For,
    If,
    LetStmt,
    MeshScope,
    Return,
    Sequential,
    While,
)
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types import DType, TensorType


def _name(expr: Expr) -> str:
    return expr.name if isinstance(expr, Var) else f"v{id(expr) % 10000}"


def _expr(expr: object) -> str:
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Constant):
        return repr(expr.value)
    if isinstance(expr, SymbolRef):
        return expr.name
    if isinstance(expr, Op):
        name = getattr(getattr(expr, "_op_schema", None), "name", type(expr).__name__.lower())
        return f"T.{name}"
    if isinstance(expr, Tuple):
        values = ", ".join(_expr(x) for x in expr.elements)
        if len(expr.elements) == 1:
            values += ","
        return f"({values})"
    if isinstance(expr, Call):
        target = expr.target
        op_name = getattr(getattr(target, "_op_schema", None), "name", None)
        if op_name is None:
            op_name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(target).__name__).lower()
        args = [_expr(x) for x in expr.args]
        for p in type(target).params():
            if p.kind != "attribute":
                continue
            value = getattr(target, p.name, None)
            if value is None or p.name == "kind" and isinstance(value, enum.Enum):
                if p.name == "kind":
                    continue
                continue
            if isinstance(value, DType):
                value = repr(value.name)
            elif isinstance(value, enum.Enum):
                value = repr(value.value)
            elif isinstance(value, TensorType):
                value = tensor_annotation(value)
            else:
                value = repr(value)
            args.append(f"{p.name}={value}")
        return f"T.{op_name}({', '.join(args)})"
    return repr(expr)


def _emit_stmt(stmt, indent: str, lines: list[str]) -> None:
    if isinstance(stmt, Sequential):
        for child in stmt.body:
            _emit_stmt(child, indent, lines)
    elif isinstance(stmt, LetStmt):
        lines.append(f"{indent}{stmt.var.name} = {_expr(stmt.value)}")
        _emit_stmt(stmt.body, indent, lines)
    elif isinstance(stmt, Evaluate):
        lines.append(f"{indent}{_expr(stmt.callable)}({_join_args(stmt.args)})")
    elif isinstance(stmt, MeshScope):
        lines.append(f"{indent}with {mesh_str(stmt.mesh)} as {stmt.binding.name}:")
        _emit_stmt(stmt.body, indent + "    ", lines)
    elif isinstance(stmt, For):
        lines.append(f"{indent}for {stmt.induction_var.name} in range({_expr(stmt.start)}, {_expr(stmt.stop)}, {_expr(stmt.step)}):")
        _emit_stmt(stmt.body, indent + "    ", lines)
    elif isinstance(stmt, If):
        lines.append(f"{indent}if {_expr(stmt.cond)}:")
        _emit_stmt(stmt.then_body, indent + "    ", lines)
        if stmt.else_body.body:
            lines.append(f"{indent}else:")
            _emit_stmt(stmt.else_body, indent + "    ", lines)
    elif isinstance(stmt, While):
        lines.append(f"{indent}while {_expr(stmt.cond)}:")
        _emit_stmt(stmt.body, indent + "    ", lines)
    elif isinstance(stmt, Return):
        lines.append(f"{indent}return")
    elif isinstance(stmt, Abort):
        lines.append(f"{indent}abort({stmt.message!r})")
    elif isinstance(stmt, DispatchCall):
        raise NotImplementedError("TIR printer has no canonical form for DispatchCall")
    else:
        raise NotImplementedError(f"TIR printer has no form for {type(stmt).__name__}")


def _join_args(args) -> str:
    return ", ".join(_expr(arg) for arg in args)


def tir_function_to_python(fn: PrimFunction, *, options=None) -> str:
    target = fn.target.to_python()
    lines = ["from __future__ import annotations", ""]
    lines.extend(target.imports)
    lines.extend([
        "from tilefoundry import prim_func",
        "from tilefoundry.dsl import T, Tensor",
        "from tilefoundry.ir.types import DType, TensorType",
        "from tilefoundry.ir.types.shard import B, P, S, Layout, Mesh, ShardLayout, Split, Topology",
        "from tilefoundry.ir.types.storage import StorageKind",
        "",
    ])
    lines.append("@prim_func(target=" + target.text + ")")
    params = ", ".join(
        f"{p.name}: {tensor_annotation(p.type) if isinstance(p.type, TensorType) else repr(p.type)}"
        for p in fn.params
    )
    lines.append(f"def {fn.name}({params}):")
    body: list[str] = []
    _emit_stmt(fn.body, "    ", body)
    lines.extend(body or ["    pass"])
    return "\n".join(lines) + "\n"


def tir_module_to_python(mod: Module, module_name: str | None = None, *, options=None) -> str:
    name = module_name or mod.name
    lines = ["from __future__ import annotations", ""]
    if mod.target is not None:
        target = mod.target.to_python()
        lines.extend(target.imports)
    for fn in mod.functions:
        if isinstance(fn, PrimFunction):
            lines.extend(fn.target.to_python().imports)
    lines.extend([
        "from tilefoundry import module, prim_func",
        "from tilefoundry.dsl import T, Tensor",
        "from tilefoundry.ir.types import DType, TensorType",
        "from tilefoundry.ir.types.shard import B, P, S, Layout, Mesh, ShardLayout, Split, Topology",
        "from tilefoundry.ir.types.storage import StorageKind",
        "",
    ])
    entry = f'entry="{mod.entry}"' if mod.entry is not None else ""
    lines.append(f"@module({entry})")
    lines.append(f"class {name}:")
    for fn in mod.functions:
        if not isinstance(fn, PrimFunction):
            raise TypeError(f"TIR printer cannot serialize {type(fn).__name__}")
        rendered_all = tir_function_to_python(fn, options=options).splitlines()
        rendered = rendered_all[rendered_all.index(next(line for line in rendered_all if line.startswith("@prim_func"))) :]
        lines.extend("    " + line if line else line for line in rendered)
    return "\n".join(lines) + "\n"
