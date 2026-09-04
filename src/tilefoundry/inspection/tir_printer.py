"""Canonical Python DSL printer for TIR PrimFunction and Module values."""

from __future__ import annotations

import enum
import re

from tilefoundry.inspection._python_render import (
    mesh_str,
    pattern_ctor,
    shard_layout_str,
    tensor_annotation,
)
from tilefoundry.ir.core import Call, Constant, Op, Tuple, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
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
from tilefoundry.ir.types.shard import ShardLayout


def _expr(expr: object) -> str:
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Constant):
        return repr(expr.value)
    if isinstance(expr, SymbolRef):
        return expr.name
    if isinstance(expr, ShapeOf):
        return f"shape_of({expr.param.name}, axis={expr.axis})"
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
            if value is None:
                continue
            if isinstance(value, DType):
                value = repr(value.name)
            elif isinstance(value, MmaAtom):
                value = f"T.cuda.mma.atom(op=T.cuda.mma.{value.op.name})"
            elif isinstance(value, enum.Enum):
                value = f"{type(value).__name__}.{value.name}"
            elif isinstance(value, TensorType):
                value = tensor_annotation(value)
            elif isinstance(value, ShardLayout):
                value = shard_layout_str(value)
            elif isinstance(value, (str, int, float, bool, tuple)):
                value = repr(value)
            else:
                raise NotImplementedError(f"TIR printer has no canonical attribute form for {type(value).__name__}")
            args.append(f"{p.name}={value}")
        return f"T.{op_name}({', '.join(args)})"
    raise NotImplementedError(f"TIR printer has no canonical expression form for {type(expr).__name__}")


def _emit_stmt(stmt, indent: str, lines: list[str]) -> None:
    if isinstance(stmt, Sequential):
        for child in stmt.body:
            _emit_stmt(child, indent, lines)
    elif isinstance(stmt, LetStmt):
        lines.append(f"{indent}{stmt.var.name} = {_expr(stmt.value)}")
        _emit_stmt(stmt.body, indent, lines)
    elif isinstance(stmt, Evaluate):
        if type(stmt.callable).__name__ == "Launch":
            callee, grid = stmt.args[0], stmt.args[1:4]
            block = stmt.args[4:7]
            forwarded = stmt.args[7:]
            lines.append(
                f"{indent}launch({_expr(callee)}, {_join_args(forwarded)}, "
                f"grid={_expr(Tuple(type=grid[0].type, elements=tuple(grid)))}, "
                f"block={_expr(Tuple(type=block[0].type, elements=tuple(block)))})  # noqa: F821"
            )
        else:
            target = stmt.callable
            args = list(stmt.args)
            attrs = []
            if isinstance(target, Op):
                for p in type(target).params():
                    if p.kind == "attribute":
                        value = getattr(target, p.name, None)
                        if value is not None:
                            if isinstance(value, enum.Enum):
                                rendered = f"{type(value).__name__}.{value.name}"
                            elif isinstance(value, MmaAtom):
                                rendered = f"T.cuda.mma.atom(op=T.cuda.mma.{value.op.name})"
                            elif isinstance(value, (str, int, float, bool, tuple)):
                                rendered = repr(value)
                            else:
                                raise NotImplementedError(f"TIR printer has no canonical attribute form for {type(value).__name__}")
                            attrs.append(f"{p.name}={rendered}")
            lines.append(f"{indent}{_expr(target)}({', '.join([_expr(a) for a in args] + attrs)})")
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
        cases = []
        for patterns, call in zip(stmt.case_patterns, stmt.case_calls):
            pats = ", ".join(pattern_ctor(pattern) for pattern in patterns)
            args = _join_args(call.args)
            cases.append(f"(({pats}), {call.callable.name}, ({args}))")
        fallback: list[str] = []
        _emit_stmt(stmt.fallback, "", fallback)
        lines.append(
            f"{indent}dispatch_call({stmt.callee_name!r}, "
            f"subjects=({_join_args(stmt.subjects)}), cases=({', '.join(cases)}), "
            f"fallback=({'; '.join(fallback)}))"
        )
    else:
        raise NotImplementedError(f"TIR printer has no form for {type(stmt).__name__}")


def _join_args(args) -> str:
    return ", ".join(_expr(arg) for arg in args)


def _function_block(fn: PrimFunction) -> list[str]:
    target = fn.target.to_python()
    lines = ["@prim_func(target=" + target.text + ")"]
    params = ", ".join(
        f"{p.name}: {tensor_annotation(p.type) if isinstance(p.type, TensorType) else repr(p.type)}"
        for p in fn.params
    )
    lines.append(f"def {fn.name}({params}):")
    body: list[str] = []
    _emit_stmt(fn.body, "    ", body)
    lines.extend(body or ["    pass"])
    return lines


def _imports(text: str, targets: set[str], *, module: bool) -> list[str]:
    names = ["module"] if module else []
    names.append("prim_func")
    lines = [f"from tilefoundry import {', '.join(names)}"]
    if "T." in text:
        lines.append("from tilefoundry.dsl import T, Tensor")
    else:
        lines.append("from tilefoundry.dsl import Tensor")
    shard_names = [name for name in ("Layout", "Mesh", "S", "ShardLayout", "Topology") if re.search(rf"\b{name}\b", text)]
    if shard_names:
        lines.append(f"from tilefoundry.ir.types.shard import {', '.join(shard_names)}")
    if targets:
        lines.append(f"from tilefoundry.target import {', '.join(sorted(targets))}")
    return lines


def tir_function_to_python(fn: PrimFunction, *, options=None) -> str:
    lines = _function_block(fn)
    text = "\n".join(lines)
    targets = {type(fn.target).__name__}
    lines = ["from __future__ import annotations", "", *_imports(text, targets, module=False), "", "", *lines]
    return "\n".join(lines) + "\n"


def tir_module_to_python(mod: Module, module_name: str | None = None, *, options=None) -> str:
    name = module_name or mod.name
    lines: list[str] = []
    kwargs = []
    if mod.entry is not None:
        kwargs.append(f'entry="{mod.entry}"')
    if mod.target is not None:
        kwargs.append(f"target={mod.target.to_python().text}")
    if mod.topologies is not None:
        rendered = ", ".join(f'Topology("{t.name}", {t.size!r})' for t in mod.topologies)
        kwargs.append(f"topologies=({rendered},)" if rendered else "topologies=()")
    lines.append(f"@module({', '.join(kwargs)})")
    lines.append(f"class {name}:")
    blocks: list[list[str]] = []
    for child in mod.modules:
        child_source = tir_module_to_python(child, child.name, options=options).splitlines()
        class_at = next(i for i, line in enumerate(child_source) if line.startswith("@module"))
        blocks.append(child_source[class_at:])
    for fn in mod.functions:
        if not isinstance(fn, PrimFunction):
            if isinstance(fn, HirFunction):
                raise NotImplementedError("mixed HIR/TIR module printing is not yet supported")
            raise TypeError(f"TIR printer cannot serialize {type(fn).__name__}")
        blocks.append(_function_block(fn))
    for index, block in enumerate(blocks):
        if index:
            lines.append("")
        lines.extend("    " + line if line else line for line in block)
    text = "\n".join(lines)
    targets = {type(fn.target).__name__ for fn in mod.functions if isinstance(fn, PrimFunction)}
    if mod.target is not None:
        targets.add(type(mod.target).__name__)
    header = ["from __future__ import annotations", "", *_imports(text, targets, module=True), "", ""]
    return "\n".join(header + lines) + "\n"
