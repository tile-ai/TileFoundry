"""Build the compile-wide code-generation context before source emission."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from tilefoundry import target
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, tir
from tilefoundry.target import cpu

from .options import Options
from .signature import LAUNCH_ABI, CallableSignature, called_as, program_id_params


def function_paths(module: core.Module, prefix: tuple[str, ...] = ()):
    """Root-relative paths for every declared function, using instance slot keys."""
    for ref, frame in valuation.collect_functions(module).items():
        yield (*prefix, *ref.path, ref.name), frame.function


def function_path(root: core.Module, fn) -> tuple[str, ...]:
    """Use definition call paths for prototypes and their specialization labels."""
    matches = []
    for path, function in function_paths(root):
        if function is fn:
            matches.append((root.name, *path))
        for index, variant in enumerate(function.variants or ()):
            if variant is fn:
                matches.append((root.name, *path, valuation.display_name(variant) or f"v{index}"))
    if len(matches) != 1:
        raise ValueError(f"codegen: function {fn.name!r} has no unique path in {root.name!r}")
    return matches[0]


def mangle(root: core.Module, fn) -> str:
    """Encode identifiers by length and decimal indices with a distinct I tag."""
    path = function_path(root, fn)
    for segment in path:
        if isinstance(segment, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", segment) is None:
            raise ValueError(f"codegen: invalid path segment {segment!r} in {'.'.join(map(str, path))!r}")
    return "tilefoundry_" + "".join(
        f"I{len(str(segment))}_{segment}" if isinstance(segment, int) else f"{len(segment)}{segment}"
        for segment in path
    )


def export_name(root: core.Module, fn) -> str:
    """The dotted TVM FFI registration name corresponding to a call path."""
    return ".".join(map(str, function_path(root, fn)))


@dataclass(frozen=True)
class FunctionSymbol:
    """Resolved IR identity and generated ABI names for one callable function."""

    function: core.PrimFunction | core.Function
    symbol: str
    export_name: str
    owner: core.Module
    target: target.Target
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

    launch: core.Evaluate
    caller: core.PrimFunction
    callee: core.PrimFunction | core.Function
    symbol: FunctionSymbol
    callable: CallableSignature
    grid_x: object
    block_x: object


@dataclass(frozen=True)
class EmissionGroup:
    owner: core.Module
    target: target.Target
    functions: tuple[core.PrimFunction | core.Function, ...]


@dataclass(frozen=True)
class EmitContext:
    """The compile facts required by one target/topology emission group."""

    codegen: Context
    owner: core.Module
    target: target.Target
    functions: tuple[core.PrimFunction | core.Function, ...]
    symbols: Mapping[int, CallableSignature]
    launches: Mapping[int, tuple[tuple[object, object, object], tuple[object, object, object]]]
    resolved_launches: Mapping[int, core.PrimFunction | core.Function]


@dataclass(frozen=True)
class Context:
    """Complete compile-wide resolution performed once before emission."""

    module: core.Module
    options: Options
    functions: tuple[core.PrimFunction | core.Function, ...]
    ownership: Mapping[int, core.Module]
    targets: Mapping[int, target.Target]
    symbols: FunctionSymbolTable
    launch_edges: tuple[LaunchEdge, ...]
    topology_domains: Mapping[int, valuation.Context]
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


def domains(root: core.Module):
    """Group definitions by their first call path's effective topology declaration."""
    rows = {}
    for frame in valuation.collect_functions(root).values():
        key = (id(frame.topologies), frame.target)
        row = rows.setdefault(key, (frame, []))
        row[1].append(frame.function)
    return tuple((frame, tuple(functions)) for frame, functions in rows.values())


def hir_domains(root: core.Module):
    """Coalesce first-call-path contexts with equal effective hierarchy."""
    rows = []
    for frame, functions in domains(root):
        row = next((row for row in rows if row[0].topologies == frame.topologies and row[0].target == frame.target), None)
        if row is None:
            rows.append((frame, list(functions)))
        else:
            row[1].extend(functions)
    return tuple((frame, tuple(functions)) for frame, functions in rows)


def product_of(values: tuple[object, ...]) -> object:
    result: object = 1
    for value in values:
        if value is None:
            raise ValueError(
                "codegen: topology extent cannot be None; use a DimVar-backed "
                "CPU-computable launch expression"
            )
        result = result * value
    return result


def launch_extents(frame: valuation.Context) -> tuple[object, object]:
    """Derive one-dimensional grid/block extents from declared topologies."""
    ctas = tuple(
        topology.size for topology in frame.topologies if topology.name == "cta"
    )
    threads = tuple(
        topology.size for topology in frame.topologies if topology.name == "thread"
    )
    return product_of(ctas), product_of(threads)


class LaunchCollector(core.StmtVisitor):
    def __init__(self):
        self.launches = []

    def visit_Evaluate(self, stmt):
        if isinstance(stmt.callable, tir.launch.Launch):
            self.launches.append(stmt)


def launch_statements(function: core.PrimFunction | core.Function) -> tuple[core.Evaluate, ...]:
    if not isinstance(function, core.PrimFunction):
        return ()
    collector = LaunchCollector()
    collector.visit(function.body)
    return tuple(collector.launches)


def build_context(
    module: core.Module, options: Options = Options()
) -> Context:
    """Resolve functions, symbols, Launch edges, domains, and groups once.

    An HIR function enters codegen only as the callee a Launch names, and takes
    the arguments that Launch binds; Python wraps any other one.
    """
    functions = tuple(frame.function for frame in valuation.collect_functions(module).values())
    for function in functions:
        if isinstance(function, core.Function) and options.emit != "skeleton":
            raise ValueError(
                f"codegen: HIR function {export_name(module, function)!r} requires "
                'emit="skeleton" or lowering to TIR first'
            )
    selected = tuple((path, frame.function) for path, frame in valuation.collect_functions(module).items() if isinstance(frame.function, core.Function))
    contexts = {
        id(each): (path, frame)
        for path, frame in valuation.collect_functions(module).items()
        for each in (frame.function, *frame.function.variants)
    }
    launched = {
        id(
            core.resolve(contexts[id(caller)][1].module, stmt.args[0])[1]
        ): stmt.args[7:]
        for caller in functions
        for stmt in launch_statements(caller)
    }
    launch_args = {
        id(each): launched[id(function)]
        for _, function in selected
        if id(function) in launched
        for each in (function, *function.variants)
    }
    functions = tuple(
        function
        for function in functions
        if not isinstance(function, core.Function) or id(function) in launch_args
    )
    expanded_list: list[core.PrimFunction] = []
    seen_expanded: set[int] = set()
    for function in functions:
        for variant in function.variants or (function,):
            if id(variant) not in seen_expanded:
                seen_expanded.add(id(variant))
                expanded_list.append(variant)
    expanded = tuple(expanded_list)
    identities = {id(fn): fn for fn in (*functions, *expanded)}
    ownership: dict[int, core.Module] = {}
    targets: dict[int, target.Target] = {}
    for function in identities.values():
        _, frame = contexts[id(function)]
        owner = frame.module
        tgt = function.target if isinstance(function, core.PrimFunction) else frame.target
        if tgt is None:
            raise ValueError(f"codegen: function {function.name!r} has no resolved Target")
        ownership[id(function)] = owner
        targets[id(function)] = tgt

    device_targets = tuple(
        dict.fromkeys(
            targets[id(function)]
            for function in identities.values()
            if not isinstance(targets[id(function)], cpu.Target)
        )
    )
    if not device_targets and module.target is not None and not isinstance(module.target, cpu.Target):
        device_targets = (module.target,)
    if len(device_targets) != 1:
        raise ValueError(f"codegen: expected one device Target, found {len(device_targets)}")
    program_ids = program_id_params(module, device_targets[0])
    by_function: dict[int, FunctionSymbol] = {}
    device_functions = tuple(
        fn for fn in identities.values() if not isinstance(targets[id(fn)], cpu.Target)
    )
    cpu_functions = tuple(
        fn for fn in identities.values() if isinstance(targets[id(fn)], cpu.Target)
    )
    for function in (*device_functions, *cpu_functions):
        launches = launch_statements(function)
        launched = (
            by_function.get(id(core.resolve(ownership[id(function)], launches[0].args[0])[1]))
            if len(launches) == 1 else None
        )
        if launched is not None and isinstance(launched.function, core.Function):
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
        for stmt in launch_statements(caller):
            callee = core.resolve(contexts[id(caller)][1].module, stmt.args[0])[1]
            symbol = by_function[id(callee)]
            grid_x, block_x = launch_extents(contexts[id(callee)][1])
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
            if any(isinstance(fn, core.Function) for fn in functions)
            else domains(module)
        )
    )
    topology_domains = {id(function): frame for frame, owned in domain_rows for function in owned}
    groups: list[EmissionGroup] = []
    for frame, owned in domain_rows:
        by_target: dict[target.Target, list[core.PrimFunction]] = {}
        for function in owned:
            by_target.setdefault(targets[id(function)], []).append(function)
        for tgt, grouped in by_target.items():
            if not isinstance(tgt, cpu.Target) and not any(
                edge.callee in grouped for edge in edges
            ):
                missing = ", ".join(function.name for function in grouped)
                raise ValueError(f"codegen: no Launch states geometry for {missing}")
            groups.append(EmissionGroup(frame.module, tgt, tuple(grouped)))

    if not device_functions:
        groups.append(EmissionGroup(module, device_targets[0], ()))

    return Context(
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
    "Context",
    "EmissionGroup",
    "EmitContext",
    "FunctionSymbol",
    "FunctionSymbolTable",
    "LaunchEdge",
    "build_context",
    "function_path",
    "mangle",
    "export_name",
]
