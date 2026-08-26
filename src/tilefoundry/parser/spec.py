"""Generate the private parser grammar and constraint reference."""

from __future__ import annotations

import argparse
import difflib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ast_pattern import (
    AstNodePattern,
    BindPattern,
    BranchPattern,
    CapturePattern,
    ChildPattern,
    ChoicePattern,
    ConditionPattern,
    ElementPattern,
    FieldPattern,
    LazyPattern,
    LiteralPattern,
    ModuleBuildContext,
    OptionalPattern,
    PredicatePattern,
    ReferencePattern,
    RepeatPattern,
    SequencePattern,
)
from .grammar_render import render_grammar
from .pattern_nodes import FunctionPattern

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, order=True)
class RuleRow:
    owner: str
    situation: str
    rule: str
    statement: str
    source: str


def _source(rule: object) -> str:
    filename = inspect.getsourcefile(type(rule))
    if filename is None:
        return "<unknown>"
    path = Path(filename).resolve()
    try:
        return path.relative_to(_REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _row(owner: str, situation: str, rule: object) -> RuleRow:
    return RuleRow(
        owner=owner,
        situation=situation,
        rule=type(rule).__name__,
        statement=rule.STATEMENT,
        source=_source(rule),
    )


class _RuleVisitor:
    def __init__(self) -> None:
        self._seen_elements: set[tuple[str, str]] = set()
        self._rows: set[RuleRow] = set()

    def visit(self, pattern: object, situation: str) -> None:
        if isinstance(pattern, ElementPattern):
            name = pattern.element_name
            if not name:
                raise TypeError(f"{type(pattern).__name__} has no element_name")
            key = (name, situation)
            if key in self._seen_elements:
                return
            self._seen_elements.add(key)
            self._rows.update(_row(name, situation, rule) for rule in pattern.RULES)
            if pattern.syntax is None:
                raise TypeError(f"{type(pattern).__name__} has no executable syntax")
            self.visit(pattern.syntax, situation)
            return
        if isinstance(pattern, LazyPattern):
            self.visit(pattern.pattern, situation)
            return
        if isinstance(pattern, ChildPattern):
            self.visit(pattern.pattern, pattern.situation)
            return
        if isinstance(pattern, AstNodePattern):
            for part in pattern.parts:
                self.visit(part, situation)
            return
        if isinstance(pattern, (ChoicePattern, SequencePattern)):
            for item in pattern.patterns:
                self.visit(item, situation)
            return
        if isinstance(
            pattern,
            (
                BindPattern,
                BranchPattern,
                ConditionPattern,
                FieldPattern,
                OptionalPattern,
                RepeatPattern,
            ),
        ):
            self.visit(pattern.pattern, situation)
            return
        if isinstance(
            pattern,
            (CapturePattern, LiteralPattern, PredicatePattern, ReferencePattern),
        ):
            return
        raise TypeError(f"unsupported executable pattern {type(pattern).__name__}")

    def rows(self) -> tuple[RuleRow, ...]:
        return tuple(sorted(self._rows))


def _collect_rule_rows(root: ElementPattern[Any]) -> tuple[RuleRow, ...]:
    visitor = _RuleVisitor()
    visitor.visit(root, "function")
    return visitor.rows()


def _collect_module_rule_rows() -> tuple[RuleRow, ...]:
    rows = [
        _row("module", "module_function", rule)
        for rule in ModuleBuildContext.FUNCTION_RULES
    ]
    rows.extend(
        _row("module", "module_finalization", rule)
        for rule in ModuleBuildContext.FINALIZATION_RULES
    )
    return tuple(rows)


def _merge_rule_rows(rows: tuple[RuleRow, ...]) -> tuple[RuleRow, ...]:
    """Render one row per owning element and rule, retaining every situation."""
    situations: dict[tuple[str, str, str, str], set[str]] = {}
    for row in rows:
        key = (row.owner, row.rule, row.statement, row.source)
        situations.setdefault(key, set()).add(row.situation)
    return tuple(
        sorted(
            RuleRow(
                owner=owner,
                situation=", ".join(sorted(rule_situations)),
                rule=rule,
                statement=statement,
                source=source,
            )
            for (owner, rule, statement, source), rule_situations in situations.items()
        )
    )


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_spec_content() -> str:
    root = FunctionPattern()
    rows = _merge_rule_rows((*_collect_rule_rows(root), *_collect_module_rule_rows()))
    lines = [
        "# Parser Grammar and Constraints",
        "",
        "```ebnf",
        render_grammar(root),
        "```",
        "",
        "| Owner | Situation | Rule | Statement | Source |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        "| "
        + " | ".join(
            _escape_cell(value)
            for value in (
                row.owner,
                row.situation,
                row.rule,
                row.statement,
                row.source,
            )
        )
        + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


_GRAMMAR_START = "<!-- parser-grammar:start -->"
_GRAMMAR_END = "<!-- parser-grammar:end -->"
_CONSTRAINTS_START = "<!-- parser-constraints:start -->"
_CONSTRAINTS_END = "<!-- parser-constraints:end -->"


def _replace_generated_section(document: str, start: str, end: str, content: str) -> str:
    """Replace exactly one marked generated section in *document*."""
    if document.count(start) != 1 or document.count(end) != 1:
        raise ValueError(f"parser spec must contain exactly one {start!r} and {end!r}")
    if document.index(start) > document.index(end):
        raise ValueError(f"parser spec marker {start!r} must precede {end!r}")
    prefix, remainder = document.split(start, 1)
    _old, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{content.rstrip()}\n{end}{suffix}"


def render_parser_document(document: str) -> str:
    """Update only the marked generated sections of a Parser Spec document."""
    generated = render_spec_content().removeprefix("# Parser Grammar and Constraints\n\n")
    grammar, rules = generated.split("| Owner | Situation | Rule | Statement | Source |", 1)
    rule_rows = rules.split("| --- | --- | --- | --- | --- |", 1)[1].lstrip()
    updated = _replace_generated_section(document, _GRAMMAR_START, _GRAMMAR_END, grammar)
    return _replace_generated_section(
        updated,
        _CONSTRAINTS_START,
        _CONSTRAINTS_END,
        "| Owner | Situation | Rule | Statement | Source |\n"
        "| --- | --- | --- | --- | --- |\n" + rule_rows,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--write", type=Path, metavar="PATH")
    output.add_argument("--check", type=Path, metavar="PATH")
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.write is not None:
        document = args.write.read_text()
        args.write.write_text(render_parser_document(document))
        return 0
    if args.check is not None:
        actual = args.check.read_text() if args.check.exists() else ""
        expected = render_parser_document(actual)
        if actual == expected:
            return 0
        sys.stderr.writelines(
            difflib.unified_diff(
                actual.splitlines(keepends=True),
                expected.splitlines(keepends=True),
                fromfile=str(args.check),
                tofile="generated parser spec",
            )
        )
        return 1
    sys.stdout.write(render_spec_content())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
