"""Build the compile-wide code-generation context before source emission."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from tilefoundry.codegen.cuda.abi import check_parameter_names
from tilefoundry.codegen.options import CodegenOptions
from tilefoundry.codegen.signature import (
    LAUNCH_ABI,
    CallableSignature,
    called_as,
    program_id_params,
)
from tilefoundry.ir.core.module import Module, resolve_symbol
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.target import CpuTarget, Target
from tilefoundry.visitor_registry.contexts import ValuationContext


def _function_paths(module: Module, prefix: tuple[str, ...] = ()):
    """Root-relative paths for every declared function, using instance slot keys."""
    for call_path, frame in list_call_paths(module).items():
        yield (*prefix, *call_path.split(".")), frame.function


def function_path(root: Module, fn) -> tuple[str, ...]:
    """Use definition call paths for prototypes and their specialization labels."""
    from tilefoundry.ir.hir.specialize import display_name  # noqa: PLC0415

    matches = []
    for path, function in _function_paths(root):
        if function is fn:
            matches.append((root.name, *path))
        for index, variant in enumerate(function.variants or ()):
            if variant is fn:
                matches.append((root.name, *path, display_name(variant) or f"v{index}"))
    if len(matches) != 1:
        raise ValueError(f"codegen: function {fn.name!r} has no unique path in {root.name!r}")
    return matches[0]


def mangle(root: Module, fn) -> str:
    """Encode identifiers by length and decimal indices with a distinct I tag."""
    path = function_path(root, fn)
    for segment in path:
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+", segment) is None:
            raise ValueError(f"codegen: invalid path segment {segment!r} in {'.'.join(path)!r}")
    return "tilefoundry_" + "".join(
        f"I{len(segment)}_{segment}" if segment.isdecimal() else f"{len(segment)}{segment}"
        for segment in path
    )


def export_name(root: Module, fn) -> str:
    """The dotted TVM FFI registration name corresponding to a call path."""
    return ".".join(function_path(root, fn))


@dataclass(frozen=True)
class FunctionSymbol:
    """Resolved IR identity and generated ABI names for one callable function."""

    function: PrimFunction | Function
    symbol: str
    export_name: str
    owner: Module
    target: Target
    callable: CallableSignature


@dataclass(frozen=True)
class FunctionSymbolTable:
    """Separate IR and generated-name namespaces shared by every emitter."""

    by_function: Mapping[int, FunctionSymbol]

    @property
    def callables(self) -> Mapping[int, CallableSignature]:
        return MappingProxyType(
            {identity: symbol.callable for identity, symbol in self.by_function.items()}
        )


@dataclass(frozen=True)
class LaunchEdge:
    """A resolved host-to-device Launch and its domain-owned callable ABI."""

    launch: Evaluate
    caller: PrimFunction
    callee: PrimFunction | Function
    symbol: FunctionSymbol
    callable: CallableSignature
    grid_x: object
    block_x: object


@dataclass(frozen=True)
class EmissionGroup:
    owner: Module
    target: Target
    functions: tuple[PrimFunction | Function, ...]


@dataclass(frozen=True)
class EmitContext:
    """The compile facts required by one target/topology emission group."""

    codegen: CodegenContext
    owner: Module
    target: Target
    functions: tuple[PrimFunction | Function, ...]
    symbols: Mapping[int, CallableSignature]
    launches: Mapping[int, tuple[tuple[object, object, object], tuple[object, object, object]]]
    resolved_launches: Mapping[int, PrimFunction | Function]


@dataclass(frozen=True)
class CodegenContext:
    """Complete compile-wide resolution performed once before emission."""

    module: Module
    options: CodegenOptions
    functions: tuple[PrimFunction | Function, ...]
    ownership: Mapping[int, Module]
    targets: Mapping[int, Target]
    symbols: FunctionSymbolTable
    launch_edges: tuple[LaunchEdge, ...]
    topology_domains: Mapping[int, ValuationContext]
    launch_callables: Mapping[int, CallableSignature]
    groups: tuple[EmissionGroup, ...]

    def for_group(self, group: EmissionGroup) -> EmitContext:
        identities = {id(fn) for fn in group.functions}
        symbols = {
            identity: symbol.callable
            for identity, symbol in self.symbols.by_function.items()
            if identity in identities or symbol.target != group.target
        }
        launches = {
            id(edge.callee): ((edge.grid_x, 1, 1), (edge.block_x, 1, 1))
            for edge in self.launch_edges
            if id(edge.callee) in identities
        }
        for edge in self.launch_edges:
            if id(edge.callee) in identities or edge.caller in group.functions:
                symbols[id(edge.callee)] = edge.callable
        resolved_launches = {
            id(edge.launch): edge.callee
            for edge in self.launch_edges
            if edge.caller in group.functions
        }
        return EmitContext(
            codegen=self,
            owner=group.owner,
            target=group.target,
            functions=group.functions,
            symbols=MappingProxyType(symbols),
            launches=MappingProxyType(launches),
            resolved_launches=MappingProxyType(resolved_launches),
        )


def _domains(root: Module):
    """Group definitions by their first call path's effective topology declaration."""
    rows = {}
    for frame in list_call_paths(root).values():
        key = (id(frame.topologies), frame.target)
        row = rows.setdefault(key, (frame, []))
        row[1].append(frame.function)
    return tuple((frame, tuple(functions)) for frame, functions in rows.values())


def hir_domains(root: Module):
    """Coalesce first-call-path contexts with equal effective hierarchy."""
    rows = []
    for frame, functions in _domains(root):
        row = next((row for row in rows if row[0].topologies == frame.topologies and row[0].target == frame.target), None)
        if row is None:
            rows.append((frame, list(functions)))
        else:
            row[1].extend(functions)
    return tuple((frame, tuple(functions)) for frame, functions in rows)


def _product(values: tuple[object, ...]) -> object:
    result: object = 1
    for value in values:
        if value is None:
            raise ValueError(
                "codegen: topology extent cannot be None; use a DimVar-backed "
                "CPU-computable launch expression"
            )
        result = result * value
    return result


def _launch_extents(frame: ValuationContext) -> tuple[object, object]:
    """Derive one-dimensional grid/block extents from declared topologies."""
    ctas = tuple(
        topology.size for topology in frame.topologies if topology.name == "cta"
    )
    threads = tuple(
        topology.size for topology in frame.topologies if topology.name == "thread"
    )
    return _product(ctas), _product(threads)


def _launch_statements(function: PrimFunction | Function) -> tuple[Evaluate, ...]:
    body = function.body
    if not isinstance(body, Sequential):
        return ()
    return tuple(
        stmt
        for stmt in body.body
        if isinstance(stmt, Evaluate) and isinstance(stmt.callable, Launch)
    )


def build_codegen_context(
    module: Module, options: CodegenOptions = CodegenOptions()
) -> CodegenContext:
    """Resolve functions, symbols, Launch edges, domains, and groups once.

    An HIR function enters codegen only as the callee a Launch names, and takes
    the arguments that Launch binds; Python wraps any other one.
    """
    functions = tuple(frame.function for frame in list_call_paths(module).values())
    for function in functions:
        if isinstance(function, Function) and options.emit != "skeleton":
            raise ValueError(
                f"codegen: HIR function {export_name(module, function)!r} requires "
                'emit="skeleton" or lowering to TIR first'
            )
    selected = tuple((path, frame.function) for path, frame in list_call_paths(module).items() if isinstance(frame.function, Function))
    contexts = {
        id(each): (path, frame)
        for path, frame in list_call_paths(module).items()
        for each in (frame.function, *frame.function.variants)
    }
    launched = {
        id(
            resolve_symbol(contexts[id(caller)][1].module, stmt.args[0].name, nested=stmt.args[0].nested)
        ): stmt.args[7:]
        for caller in functions
        for stmt in _launch_statements(caller)
    }
    launch_args = {
        id(each): launched[id(function)]
        for _selector, function in selected
        if id(function) in launched
        for each in (function, *function.variants)
    }
    functions = tuple(
        function
        for function in functions
        if not isinstance(function, Function) or id(function) in launch_args
    )
    expanded_list: list[PrimFunction] = []
    seen_expanded: set[int] = set()
    for function in functions:
        for variant in function.variants or (function,):
            if id(variant) not in seen_expanded:
                seen_expanded.add(id(variant))
                expanded_list.append(variant)
    expanded = tuple(expanded_list)
    identities = {id(fn): fn for fn in (*functions, *expanded)}
    ownership: dict[int, Module] = {}
    targets: dict[int, Target] = {}
    for function in identities.values():
        _path, frame = contexts[id(function)]
        owner = frame.module
        target = function.target if isinstance(function, PrimFunction) else frame.target
        if target is None:
            raise ValueError(f"codegen: function {function.name!r} has no resolved Target")
        ownership[id(function)] = owner
        targets[id(function)] = target

    device_targets = tuple(
        dict.fromkeys(
            targets[id(function)]
            for function in identities.values()
            if not isinstance(targets[id(function)], CpuTarget)
        )
    )
    if len(device_targets) != 1:
        raise ValueError(f"codegen: expected one device Target, found {len(device_targets)}")
    program_ids = program_id_params(module, device_targets[0])
    by_function: dict[int, FunctionSymbol] = {}
    device_functions = tuple(
        fn for fn in identities.values() if not isinstance(targets[id(fn)], CpuTarget)
    )
    cpu_functions = tuple(
        fn for fn in identities.values() if isinstance(targets[id(fn)], CpuTarget)
    )
    for function in (*device_functions, *cpu_functions):
        launches = _launch_statements(function)
        launched = (
            by_function.get(id(resolve_symbol(
                ownership[id(function)], launches[0].args[0].name,
                nested=launches[0].args[0].nested,
            ))) if len(launches) == 1 else None
        )
        if launched is not None and isinstance(launched.function, Function):
            mangled = launched.symbol
            registered_name = launched.export_name
        else:
            mangled = mangle(module, function)
            registered_name = export_name(module, function)
        signature = called_as(
            function,
            mangled,
            program_ids,
            target=targets[id(function)],
            params=launch_args.get(id(function)),
        )
        check_parameter_names(registered_name, signature.params)
        symbol = FunctionSymbol(
            function=function,
            symbol=mangled,
            export_name=registered_name,
            owner=ownership[id(function)],
            target=targets[id(function)],
            callable=signature,
        )
        by_function[id(function)] = symbol

    table = FunctionSymbolTable(
        MappingProxyType(by_function),
    )
    edges: list[LaunchEdge] = []
    launch_callables: dict[int, CallableSignature] = {}
    for caller in functions:
        for stmt in _launch_statements(caller):
            reference = stmt.args[0]
            callee = resolve_symbol(contexts[id(caller)][1].module, reference.name, nested=reference.nested)
            symbol = by_function[id(callee)]
            grid_x, block_x = _launch_extents(contexts[id(callee)][1])
            callable_signature = replace(symbol.callable, trailing=LAUNCH_ABI)
            launch_callables[id(stmt)] = callable_signature
            edges.append(
                LaunchEdge(
                    stmt,
                    caller,
                    symbol.function,
                    symbol,
                    callable_signature,
                    grid_x,
                    block_x,
                )
            )

    domain_rows = tuple(
        (frame, tuple(function for function in owned if id(function) in identities))
        for frame, owned in (
            hir_domains(module)
            if any(isinstance(fn, Function) for fn in functions)
            else _domains(module)
        )
    )
    topology_domains = {id(function): frame for frame, owned in domain_rows for function in owned}
    groups: list[EmissionGroup] = []
    for frame, owned in domain_rows:
        by_target: dict[Target, list[PrimFunction]] = {}
        for function in owned:
            by_target.setdefault(targets[id(function)], []).append(function)
        for target, grouped in by_target.items():
            if not isinstance(target, CpuTarget) and not any(
                edge.callee in grouped for edge in edges
            ):
                missing = ", ".join(function.name for function in grouped)
                raise ValueError(f"codegen: no Launch states geometry for {missing}")
            groups.append(EmissionGroup(frame.module, target, tuple(grouped)))

    return CodegenContext(
        module=module,
        options=options,
        functions=functions,
        ownership=MappingProxyType(ownership),
        targets=MappingProxyType(targets),
        symbols=table,
        launch_edges=tuple(edges),
        topology_domains=MappingProxyType(topology_domains),
        launch_callables=MappingProxyType(launch_callables),
        groups=tuple(groups),
    )


__all__ = [
    "CodegenContext",
    "EmissionGroup",
    "EmitContext",
    "FunctionSymbol",
    "FunctionSymbolTable",
    "LaunchEdge",
    "build_codegen_context",
    "function_path",
    "mangle",
    "export_name",
]
