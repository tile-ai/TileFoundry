"""Assemble CPU host entries through registered statement writers."""

from __future__ import annotations

from tilefoundry import target
from tilefoundry.analysis import communication
from tilefoundry.codegen import linkable_function, linkable_module, signature
from tilefoundry.ir import core, types
from tilefoundry.target import cpu

from .context import Context
from .templates import render


def host_entry(
    entry: core.PrimFunction,
    sig: signature.CallableSignature,
    body_lines: list[str],
    ctx: Context,
) -> linkable_function.LinkableFunction:
    """*entry* in both of its positions, written from the one signature it is called by."""
    return linkable_function.LinkableFunction(
        name=entry.name,
        declaration=f"void {sig.name}({ctx.parameters(sig)});",
        definition=render(
            "cpu_entry.cpp.j2",
            host_symbol=sig.name,
            params=ctx.parameters(sig),
            body_lines=body_lines,
            entry_name=ctx.codegen_context.symbols.by_function[id(entry)].export_name,
        ),
    )


@target.bind(cpu.Target, linkable_module.LinkableModule)
class HostLinkableModule(linkable_module.LinkableModule):
    """The single translation unit containing the selected CPU host entries."""

    emits = frozenset({"full", "skeleton"})

    @classmethod
    def create(cls, tgt: target.Target, module: core.Module, functions: tuple[core.PrimFunction, ...], ctx):
        """Assemble the selected entries through registered statement writers."""
        if ctx.codegen_context is None:
            raise ValueError("host emission requires a codegen.Context")
        emit = ctx.codegen_context.options.emit
        if emit not in cls.emits:
            raise target.UnsupportedCapabilityError(
                f"{target.summary(tgt)}: code generator does not support emit={emit!r}"
            )
        if not functions:
            raise ValueError("emit_host_module: expected at least one CPU host entry")
        declarations: dict[str, str] = {}
        emitted = []
        for entry in functions:
            body_lines: list[str] = []
            if not isinstance(entry.body, core.Sequential) or not entry.body.body:
                raise ValueError(f"emit_host_module: entry {entry.name!r} body must contain statements")
            statements = entry.body.body
            if any(isinstance(param.type, types.CommunicationHandleType) for param in entry.params):
                report = communication.check_communication(entry)
                if report.hazards:
                    raise ValueError(f"unsafe communication buffer lifetimes: {report.hazards}")
                statements = tuple(communication.host_statements(entry.body))
                declaration, lines = ctx.handler_for(types.CommunicationHandleType)(entry, None, ctx)
                declarations[declaration] = declaration
                body_lines += lines
            for stmt in statements:
                if not isinstance(stmt, core.Evaluate):
                    raise ValueError(f"emit_host_module: expected Evaluate, got {type(stmt).__name__}")
                declaration, lines = ctx.handler_for(type(stmt.callable))(entry, stmt, ctx)
                declarations[declaration] = declaration
                body_lines += lines
            emitted.append(host_entry(entry, ctx.signature_of(entry), body_lines, ctx))
        return cls(
            target="cpu",
            language="cpp",
            preamble=render("cpu_preamble.cpp.j2", launcher_decls=list(declarations.values())),
            functions=tuple(emitted),
        )

    def link(self, project: "CMakeProject") -> None:
        """Keep all selected CPU host entries in the scaffold's host.cpp."""
        project.add_source("host.cpp", self.source)


__all__ = ["HostLinkableModule"]
