"""Shared parser, printer, source-span, and constant-parameter corpus checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.core._corpus import Case, Problems, reload
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, SourceSpanMetadata, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.visitor_registry.verify import verify_module


@dataclass(frozen=True)
class ParserCase(Case):
    canonical: bool = False
    golden: Path | None = None
    prints: tuple[str, ...] = ()
    same_as: Path | None = None


@dataclass(frozen=True)
class Parsed:
    ir: Module | None
    printed: str | None
    rebuilt: Module | None


def _functions(ir):
    return tuple(frame.function for frame in list_call_paths(ir).values())


def require_call_spans(ir) -> None:
    for function in _functions(ir):
        if isinstance(function, Function):
            for expr in collect_exprs(function.body):
                if isinstance(expr, Call):
                    assert get_metadata(expr, SourceSpanMetadata) is not None


def _reparse(ir, printed):
    return reload(printed, ir.name)


def _const_parameters(ir, printed, rebuilt):
    original = [param.is_const for fn in _functions(ir) for param in fn.params]
    if rebuilt is not ir:
        restored = [param.is_const for fn in _functions(rebuilt) for param in fn.params]
        assert restored == original
    if any(original):
        assert "ConstTensor[" in printed


def _verify_tir(ir):
    if any(isinstance(fn, PrimFunction) for fn in _functions(ir)):
        verify_module(ir)


def _same_print(printed, path):
    other = Case(path=path).load()
    assert as_script(other) == printed


def check_parser_case(case: ParserCase) -> None:
    problems = Problems(case)
    ir = problems.run("parse", case.load)
    problems.run("source spans", require_call_spans, ir)
    printed = problems.run("print", as_script, ir)
    if case.canonical:
        problems.expect("canonical text", lambda text: text == case.path.read_text(), printed)
        rebuilt = ir
    else:
        rebuilt = problems.run("reparse", _reparse, ir, printed)
        reprinted = problems.run("reprint", as_script, rebuilt)
        problems.expect("fixed point", lambda before, after: before == after, printed, reprinted)
    problems.run("constant parameters", _const_parameters, ir, printed, rebuilt)
    problems.run("verify TIR", _verify_tir, rebuilt)
    if case.golden is not None:
        problems.expect("golden text", lambda text: text == case.golden.read_text(), printed)
    for fragment in case.prints:
        problems.expect(f"contains {fragment!r}", lambda text: fragment in text, printed)
    if case.same_as is not None:
        problems.run("same print", _same_print, printed, case.same_as)
    state = Parsed(ir, printed, rebuilt)
    for hook in case.hooks:
        problems.run(hook.__name__, hook, state, problems)
    problems.dump()


def unique_parser_cases(cases):
    """Refuse colliding ids after directory cases and named bindings are combined."""
    found = set()
    for case in cases:
        if case.id in found:
            raise ValueError(f"duplicate parser case id: {case.id}")
        found.add(case.id)
    return cases
