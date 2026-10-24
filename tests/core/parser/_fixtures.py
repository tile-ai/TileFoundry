"""Tests and helpers for the owning target."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry.cli.source import load_namespace
from tilefoundry.ir.core import Call, SourceSpanMetadata, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.visitor import collect_exprs


def _functions_defined_by(namespace: dict[str, object]) -> tuple[Function, ...]:
    found: list[Function] = []
    seen: set[int] = set()

    def add(function: Function) -> None:
        if id(function) not in seen:
            seen.add(id(function))
            found.append(function)

    for value in namespace.values():
        if isinstance(value, Function):
            add(value)
        elif isinstance(value, Module):
            for module_value in (value, *(frame.module for frame in list_call_paths(value).values())):
                for function in module_value.functions:
                    if isinstance(function, Function):
                        add(function)
    return tuple(found)


def check_parsed_call_sources(source: Path) -> None:
    """Every parser-authored Call reachable from fixture source has a source span."""
    if "diagnostics" in source.parts:
        pytest.skip("diagnostic fixtures are expected to refuse parsing")
    namespace, _ = load_namespace(str(source))
    for function in _functions_defined_by(namespace):
        for expr in collect_exprs(function.body):
            if isinstance(expr, Call):
                assert get_metadata(expr, SourceSpanMetadata) is not None
