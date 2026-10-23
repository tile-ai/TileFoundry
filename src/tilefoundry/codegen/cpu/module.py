"""Assemble CPU host entries through registered statement writers."""

from __future__ import annotations

from tilefoundry.codegen.cpu.context import CpuCodegenContext
from tilefoundry.codegen.cpu.templates import render
from tilefoundry.codegen.linkable_function import LinkableFunction
from tilefoundry.codegen.linkable_module import LinkableModule
from tilefoundry.codegen.registry import CodeGenerator
from tilefoundry.codegen.signature import CallableSignature
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.target import Target


def _host_entry(
    entry: PrimFunction,
    signature: CallableSignature,
    body_lines: list[str],
    ctx: CpuCodegenContext,
) -> LinkableFunction:
    """*entry* in both of its positions, written from the one signature it is called by."""
    return LinkableFunction(
        name=entry.name,
        declaration=f"void {signature.name}({ctx.parameters(signature)});",
        definition=render(
            "cpu_entry.cpp.j2",
            host_symbol=signature.name,
            params=ctx.parameters(signature),
            body_lines=body_lines,
            entry_name=ctx.codegen_context.symbols.by_function[id(entry)].export_name,
        ),
    )


def emit_host_module(
    module: Module,
    functions: tuple[PrimFunction, ...],
    target: Target,
    ctx: CpuCodegenContext,
) -> LinkableModule:
    """Emit the host ``.cpp`` linkable module for a CPU *entry*.

    Registered statement writers supply foreign declarations and body lines.
    The FFI export is written with each definition.
    """
    if not functions:
        raise ValueError("emit_host_module: expected at least one CPU host entry")
    if ctx.codegen_context is None:
        raise ValueError("host emission requires a CodegenContext")
    declarations: dict[str, str] = {}
    emitted = []
    for entry in functions:
        body_lines: list[str] = []
        if not isinstance(entry.body, Sequential) or not entry.body.body:
            raise ValueError(f"emit_host_module: entry {entry.name!r} body must contain statements")
        for stmt in entry.body.body:
            if not isinstance(stmt, Evaluate):
                raise ValueError(f"emit_host_module: expected Evaluate, got {type(stmt).__name__}")
            declaration, lines = ctx.handler_for(type(stmt.callable))(entry, stmt, ctx)
            declarations[declaration] = declaration
            body_lines += lines
        emitted.append(_host_entry(entry, ctx.signature_of(entry), body_lines, ctx))
    return LinkableModule(
        target="cpu",
        language="cpp",
        preamble=render("cpu_preamble.cpp.j2", launcher_decls=list(declarations.values())),
        functions=tuple(emitted),
    )


CPU_CODE_GENERATOR = CodeGenerator(emit_host_module, emits=frozenset({"full", "skeleton"}))


__all__ = ["emit_host_module"]
