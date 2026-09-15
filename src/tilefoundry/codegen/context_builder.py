"""Build the compile-wide code-generation context before source emission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from tilefoundry.codegen.signature import (
    LAUNCH_ABI,
    CallableSignature,
    called_as,
    program_id_params,
)
from tilefoundry.ir.core.module import Module, module_functions, owning_module, subtree
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.target import CpuTarget, Target


@dataclass(frozen=True)
class FunctionSymbol:
    """Resolved IR identity and generated ABI names for one callable function."""

    function: PrimFunction
    ir_name: str
    kernel_name: str | None
    shim_name: str | None
    host_name: str | None
    owner: Module
    target: Target
    callable: CallableSignature


@dataclass(frozen=True)
class FunctionSymbolTable:
    """Separate IR and generated-name namespaces shared by every emitter."""

    by_function: Mapping[int, FunctionSymbol]
    by_ir_name: Mapping[str, FunctionSymbol]
    by_kernel_name: Mapping[str, FunctionSymbol]

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
    callee: PrimFunction
    symbol: FunctionSymbol
    callable: CallableSignature
    grid_x: object
    block_x: object


@dataclass(frozen=True)
class EmissionGroup:
    owner: Module
    target: Target
    functions: tuple[PrimFunction, ...]


@dataclass(frozen=True)
class EmitContext:
    """The compile facts required by one target/topology emission group."""

    codegen: CodegenContext
    owner: Module
    target: Target
    functions: tuple[PrimFunction, ...]
    symbols: Mapping[int, CallableSignature]
    launches: Mapping[int, tuple[tuple[object, object, object], tuple[object, object, object]]]
    resolved_launches: Mapping[int, PrimFunction]


@dataclass(frozen=True)
class CodegenContext:
    """Complete compile-wide resolution performed once before emission."""

    module: Module
    functions: tuple[PrimFunction, ...]
    ownership: Mapping[int, Module]
    targets: Mapping[int, Target]
    symbols: FunctionSymbolTable
    launch_edges: tuple[LaunchEdge, ...]
    topology_domains: Mapping[int, Module]
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


def _domain_functions(node: Module) -> tuple[PrimFunction, ...]:
    inherited = tuple(
        function
        for child in node.modules
        if child.topologies is None
        for function in _domain_functions(child)
    )
    return (*node.functions, *inherited)


def _domains(root: Module) -> tuple[tuple[Module, tuple[PrimFunction, ...]], ...]:
    domains = [(root, _domain_functions(root))]
    for node in subtree(root):
        domains.extend(
            (child, _domain_functions(child))
            for child in node.modules
            if child.topologies is not None
        )
    return tuple(domains)


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


def _launch_extents(owner: Module) -> tuple[object, object]:
    """Derive one-dimensional grid/block extents from declared topologies."""
    ctas = tuple(
        topology.size for topology in owner.effective_topologies() if topology.name == "cta"
    )
    threads = tuple(
        topology.size for topology in owner.effective_topologies() if topology.name == "thread"
    )
    return _product(ctas), _product(threads)


def _launch_statements(function: PrimFunction) -> tuple[Evaluate, ...]:
    body = function.body
    if not isinstance(body, Sequential):
        return ()
    return tuple(
        stmt
        for stmt in body.body
        if isinstance(stmt, Evaluate) and isinstance(stmt.callable, Launch)
    )


def build_codegen_context(module: Module) -> CodegenContext:
    """Resolve functions, symbols, Launch edges, domains, and groups once."""
    functions = tuple(module_functions(module))
    if not all(isinstance(function, PrimFunction) for function in functions):
        bad = next(function for function in functions if not isinstance(function, PrimFunction))
        raise TypeError(
            f"codegen expects PrimFunction values, got {type(bad).__name__} {bad.name!r}"
        )
    expanded_list: list[PrimFunction] = []
    seen_expanded: set[int] = set()
    for function in functions:
        for variant in function.variants or (function,):
            if id(variant) not in seen_expanded:
                seen_expanded.add(id(variant))
                expanded_list.append(variant)
    expanded = tuple(expanded_list)
    ownership: dict[int, Module] = {}
    targets: dict[int, Target] = {}
    for function in (*functions, *expanded):
        owner = owning_module(module, function)
        if owner is None:
            raise ValueError(f"codegen: function {function.name!r} has no unique owning Module")
        if function.target is None:
            raise ValueError(f"codegen: function {function.name!r} has no resolved Target")
        ownership[id(function)] = owner
        targets[id(function)] = function.target

    device_targets = tuple(
        dict.fromkeys(
            function.target
            for function in (*functions, *expanded)
            if not isinstance(function.target, CpuTarget)
        )
    )
    if len(device_targets) != 1:
        raise ValueError(f"codegen: expected one device Target, found {len(device_targets)}")
    program_ids = program_id_params(module, device_targets[0])
    by_function: dict[int, FunctionSymbol] = {}
    by_ir_name: dict[str, FunctionSymbol] = {}
    by_kernel_name: dict[str, FunctionSymbol] = {}
    for function in functions:
        signature = called_as(function, program_ids)
        is_device = not isinstance(function.target, CpuTarget)
        symbol = FunctionSymbol(
            function=function,
            ir_name=function.name,
            kernel_name=function.name if is_device else None,
            shim_name=signature.name if is_device else None,
            host_name=signature.name if not is_device else None,
            owner=ownership[id(function)],
            target=function.target,
            callable=signature,
        )
        if function.name in by_ir_name:
            raise ValueError(f"codegen: duplicate IR function name {function.name!r}")
        by_function[id(function)] = symbol
        by_ir_name[function.name] = symbol
        if symbol.kernel_name is not None:
            if symbol.kernel_name in by_kernel_name:
                raise ValueError(f"codegen: duplicate kernel symbol {symbol.kernel_name!r}")
            by_kernel_name[symbol.kernel_name] = symbol

    table = FunctionSymbolTable(
        MappingProxyType(by_function),
        MappingProxyType(by_ir_name),
        MappingProxyType(by_kernel_name),
    )
    edges: list[LaunchEdge] = []
    launch_callables: dict[int, CallableSignature] = {}
    for caller in functions:
        for stmt in _launch_statements(caller):
            reference = stmt.args[0]
            symbol = by_ir_name.get(reference.name)
            if symbol is None:
                raise ValueError(
                    f"codegen: launch from {caller.name!r} names unknown function {reference.name!r}"
                )
            grid_x, block_x = _launch_extents(symbol.owner)
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

    domain_rows = _domains(module)
    topology_domains = {id(function): owner for owner, owned in domain_rows for function in owned}
    groups: list[EmissionGroup] = []
    for owner, owned in domain_rows:
        by_target: dict[Target, list[PrimFunction]] = {}
        for function in owned:
            by_target.setdefault(function.target, []).append(function)
        for target, grouped in by_target.items():
            if not isinstance(target, CpuTarget) and not any(
                edge.callee in grouped for edge in edges
            ):
                missing = ", ".join(function.name for function in grouped)
                raise ValueError(f"codegen: no Launch states geometry for {missing}")
            groups.append(EmissionGroup(owner, target, tuple(grouped)))

    return CodegenContext(
        module=module,
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
]
