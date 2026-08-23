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


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_spec_content() -> str:
    root = FunctionPattern()
    rows = (*_collect_rule_rows(root), *_collect_module_rule_rows())
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


def render_parser_document() -> str:
    """Render the checked three-section Parser Spec document."""
    generated = render_spec_content().removeprefix("# Parser Grammar and Constraints\n\n")
    return f'''# TileFoundry Spec - Parser

The Parser accepts authored Python functions and produces HIR or TIR through one typed API.

## 1. Public API

`@module` executes its Python class body and finalizes the collected Functions, child Modules,
and ordinary methods. `@func` produces an HIR Function; `@prim_func` produces a TIR PrimFunction.
`specialize` and `converter` register variants and weight converters on an existing HIR Function.

```python
def parse_function(
    fn: FunctionType, context: FuncParserContext
) -> hir.Function | tir.PrimFunction: ...
```

`FuncParserContext` carries the dialect, Function role, closure, topology scope, target, and
optional base/key for one parse. `FunctionRole` is `ROOT`, `VARIANT`, or `CONVERTER`.
`ParseError` is the single authored-source diagnostic type and includes source location and
recursive parse situation. These are the only public parser symbols.

## 2. Syntax and Rules

### 2.1 Syntax

<!-- parser-grammar:start -->
{generated.split('| Owner | Situation | Rule | Statement | Source |', 1)[0].rstrip()}
<!-- parser-grammar:end -->

### 2.2 Rules

<!-- parser-constraints:start -->
| Owner | Situation | Rule | Statement | Source |
| --- | --- | --- | --- | --- |
{generated.split('| Owner | Situation | Rule | Statement | Source |', 1)[1].split('| --- | --- | --- | --- | --- |', 1)[1].lstrip()}
<!-- parser-constraints:end -->

## 3. Implementation Overview

| Component | Responsibility |
| --- | --- |
| Parser API and Context | Receives authored Functions and carries dialect, role, scope, and recursion inputs. |
| Executable Pattern Graph | Composes concrete AST elements into the Function root pattern. |
| Match and Construction | Matches recursively into `AstMatch`, then constructs owner values on return. |
| Ordered Rules | Validates and normalizes each owner value after construction. |
| Module Build | Lets Python execute the class body, records Functions, and finalizes the Module. |
| Pattern Visitor | Traverses the same graph to render this section's generated grammar and constraints. |

```mermaid
classDiagram
    ParserAPI --> FuncParserContext
    ParserAPI --> FunctionPattern
    AstPattern <|.. Element
    Element o-- AstPattern
    Element o-- AstRule
    AstPattern --> AstMatch
    PatternVisitor ..> AstPattern
    ParserAPI ..> ModuleBuild
```

```mermaid
flowchart TD
    API["parse_function(fn, context)"] --> AST["Extract FunctionDef AST"]
    AST --> ROOT["FunctionPattern.match"]
    ROOT --> TREE["AstMatch tree"]
    TREE --> BACKWARD["construct children, then apply Rules"]
    BACKWARD --> FUNCTION["HIR Function / TIR PrimFunction"]
    FUNCTION --> MODULE{{"Module authoring context?"}}
    MODULE -->|yes| FINALIZE["registration / finalization"]
    MODULE -->|no| RETURN["return standalone result"]
```

Pattern combinators serve both runtime matching and Spec traversal. `AstMatch` separates syntax
matching from object construction, while each Rule reads the recursive context after its owner
value exists. Module class control flow remains Python execution; no Module AST grammar exists.
'''


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--write", type=Path, metavar="PATH")
    output.add_argument("--check", type=Path, metavar="PATH")
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    generated = render_spec_content()
    if args.write is not None:
        args.write.write_text(render_parser_document())
        return 0
    if args.check is not None:
        expected = render_parser_document()
        actual = args.check.read_text() if args.check.exists() else ""
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
    sys.stdout.write(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
