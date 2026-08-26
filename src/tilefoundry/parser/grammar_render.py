"""Render the executable Pattern graph as deterministic surface EBNF."""

# ruff: noqa: PLC0415

from __future__ import annotations

import ast
import dataclasses
import textwrap
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
    OptionalPattern,
    PredicatePattern,
    ReferencePattern,
    RepeatPattern,
    SequencePattern,
)


def _grammar_name(value: str) -> str:
    return value.replace("_", "-").replace(" ", "-").lower()


@dataclasses.dataclass(frozen=True)
class _Expr:
    text: str
    alternatives: tuple[str, ...] = ()

    def flat(self) -> str:
        if self.alternatives:
            return "(" + " | ".join(self.alternatives) + ")"
        return self.text


def _text(value: str) -> _Expr:
    return _Expr(value)


def _concat(*parts: _Expr | None) -> _Expr:
    return _text(" ".join(part.flat() for part in parts if part and part.flat()))


def _choice(*parts: _Expr) -> _Expr:
    alternatives: list[str] = []
    for part in parts:
        values = part.alternatives or (part.text,)
        for value in values:
            if value and value not in alternatives:
                alternatives.append(value)
    if len(alternatives) == 1:
        return _text(alternatives[0])
    return _Expr("", tuple(alternatives))


def _terminal(value: str) -> _Expr:
    return _text(repr(value))


def _optional(value: _Expr) -> _Expr:
    return _text(f"({value.flat()})?")


def _delimited(left: str, value: _Expr, right: str) -> _Expr:
    return _concat(_terminal(left), value, _terminal(right))


def _separated(items: tuple[_Expr, ...], separator: _Expr) -> _Expr:
    if not items:
        return _text("")
    output = items[0]
    for item in items[1:]:
        output = _concat(output, separator, item)
    return output


def _repeated(item: _Expr, minimum: int, separator: _Expr) -> _Expr:
    required = _concat(item, _text(f"({separator.flat()} {item.flat()})*"))
    return required if minimum else _optional(required)


class RenderVisitor:
    """Project executable AST patterns into readable Python authoring EBNF."""

    def __init__(self, *, line_width: int = 100):
        self.line_width = line_width
        self._seen: set[str] = set()
        self._productions: list[tuple[str, _Expr]] = []

    def _element(self, pattern: ElementPattern[Any]) -> _Expr:
        name = pattern.element_name
        if not name:
            raise TypeError(f"{type(pattern).__name__} has no explicit element_name")
        grammar_name = _grammar_name(name)
        if name in self._seen:
            return _text(grammar_name)
        self._seen.add(name)
        if pattern.syntax is None:
            raise TypeError(f"{type(pattern).__name__} has no executable syntax")
        rhs = self.visit(pattern.syntax)
        self._productions.append((grammar_name, rhs))
        return _text(grammar_name)

    def _list_pattern(self, pattern: object) -> _Expr:
        comma = _terminal(",")
        if isinstance(pattern, RepeatPattern):
            return _repeated(self.visit(pattern.pattern), pattern.minimum, comma)
        if isinstance(pattern, SequencePattern):
            return _separated(tuple(self.visit(item) for item in pattern.patterns), comma)
        if isinstance(pattern, ChoicePattern):
            return _choice(*(self._list_pattern(item) for item in pattern.patterns))
        return self.visit(pattern)

    @staticmethod
    def _fields(pattern: AstNodePattern) -> dict[str, object]:
        return {part.name: part.pattern for part in pattern.parts if isinstance(part, FieldPattern)}

    def _field(self, fields: dict[str, object], name: str, fallback: str) -> _Expr:
        pattern = fields.get(name)
        return _text(fallback) if pattern is None else self.visit(pattern)

    def _optional_field(self, fields: dict[str, object], name: str, fallback: str) -> _Expr:
        pattern = fields.get(name)
        if isinstance(pattern, OptionalPattern):
            pattern = pattern.pattern
        return _text(fallback) if pattern is None else self.visit(pattern)

    def _ast_node(self, pattern: AstNodePattern) -> _Expr:
        node_type = pattern.node_type
        for part in pattern.parts:
            self.visit(part)
        fields = self._fields(pattern)

        if node_type is ast.Constant:
            predicates = [part for part in pattern.parts if isinstance(part, PredicatePattern)]
            if predicates:
                return self.visit(predicates[-1])
            value = fields.get("value")
            if isinstance(value, CapturePattern) or value is None:
                return _choice(
                    _text("None"),
                    _text("Ellipsis"),
                    _text("boolean-literal"),
                    _text("integer-literal"),
                    _text("float-literal"),
                    _text("complex-literal"),
                    _text("string-literal"),
                    _text("bytes-literal"),
                )
            return self.visit(value)
        if node_type is ast.Name:
            identifier = fields.get("id")
            if isinstance(identifier, (LiteralPattern, ChoicePattern)):
                return self.visit(identifier)
            return _text("identifier")
        if node_type is ast.Attribute:
            value = fields.get("value")
            if isinstance(value, AstNodePattern) and value.node_type is ast.Name:
                return _concat(
                    _text("identifier"),
                    _terminal("."),
                    _text("identifier"),
                )
            return _concat(
                _text("primary"),
                _terminal("."),
                _text("identifier"),
            )
        if node_type in {ast.Tuple, ast.List, ast.Set}:
            delimiters = {
                ast.Tuple: ("(", ")"),
                ast.List: ("[", "]"),
                ast.Set: ("{", "}"),
            }
            left, right = delimiters[node_type]
            values_pattern = fields.get("elts")
            if values_pattern is None:
                structural = [
                    part
                    for part in pattern.parts
                    if not isinstance(part, (CapturePattern, FieldPattern, PredicatePattern))
                ]
                if structural:
                    return self.visit(structural[-1])
                values = _text("")
            else:
                values = self._list_pattern(values_pattern)
            return _delimited(left, values, right)
        if node_type is ast.Dict:
            key_pattern = fields.get("keys", LiteralPattern())
            value_pattern = fields.get("values", LiteralPattern())
            key = (
                self.visit(key_pattern.pattern)
                if isinstance(key_pattern, RepeatPattern)
                else self.visit(key_pattern)
            )
            value = (
                self.visit(value_pattern.pattern)
                if isinstance(value_pattern, RepeatPattern)
                else self.visit(value_pattern)
            )
            entry = _concat(key, _terminal(":"), value)
            return _delimited("{", _repeated(entry, 0, _terminal(",")), "}")
        if node_type is ast.BinOp:
            return _concat(
                self._field(fields, "left", "expression"),
                self._field(fields, "op", "binary-op"),
                self._field(fields, "right", "expression"),
            )
        if node_type is ast.UnaryOp:
            return _concat(
                self._field(fields, "op", "unary-op"),
                self._field(fields, "operand", "expression"),
            )
        if node_type is ast.Compare:
            return _concat(
                self._field(fields, "left", "expression"),
                self._list_pattern(fields.get("ops", SequencePattern())),
                self._list_pattern(fields.get("comparators", SequencePattern())),
            )
        if node_type is ast.BoolOp:
            values = fields.get("values", SequencePattern())
            if isinstance(values, SequencePattern) and len(values.patterns) == 2:
                return _concat(
                    self.visit(values.patterns[0]),
                    self._field(fields, "op", "boolean-op"),
                    self.visit(values.patterns[1]),
                )
            return self._list_pattern(values)
        if node_type is ast.Call:
            args = fields.get("args")
            keywords = fields.get("keywords")
            if not isinstance(args, RepeatPattern) or not isinstance(keywords, RepeatPattern):
                arguments = _concat(
                    self._list_pattern(args) if args is not None else None,
                    self._list_pattern(keywords) if keywords is not None else None,
                )
                return _concat(
                    self._field(fields, "func", "callee"),
                    _delimited("(", arguments, ")"),
                )
            argument_patterns: list[_Expr] = []
            minimum = 0
            for name in ("args", "keywords"):
                values = fields.get(name)
                if isinstance(values, RepeatPattern):
                    argument_patterns.append(self.visit(values.pattern))
                    minimum = max(minimum, values.minimum)
            arguments = (
                _repeated(_choice(*argument_patterns), minimum, _terminal(","))
                if argument_patterns
                else _text("")
            )
            return _concat(
                self._field(fields, "func", "callee"),
                _delimited("(", arguments, ")"),
            )
        if node_type is ast.keyword:
            return _concat(
                self._field(fields, "arg", "name"),
                _terminal("="),
                self._field(fields, "value", "expression"),
            )
        if node_type is ast.Slice:
            lower = self._field(fields, "lower", "")
            upper = self._field(fields, "upper", "")
            step = self._optional_field(fields, "step", "")
            return _concat(
                lower,
                _terminal(":"),
                upper,
                _optional(_concat(_terminal(":"), step)),
            )
        if node_type is ast.Subscript:
            return _concat(
                self._field(fields, "value", "expression"),
                _delimited("[", self._field(fields, "slice", "expression"), "]"),
            )
        if node_type is ast.arguments:
            return self._list_pattern(fields.get("args", SequencePattern()))
        if node_type is ast.arg:
            return _concat(
                _text("name"),
                _terminal(":"),
                self._field(fields, "annotation", "type-annotation"),
            )
        if node_type is ast.Assign:
            return _concat(
                self._list_pattern(fields.get("targets", SequencePattern())),
                _terminal("="),
                self._field(fields, "value", "expression"),
            )
        if node_type is ast.AnnAssign:
            value = self._optional_field(fields, "value", "expression")
            return _concat(
                self._field(fields, "target", "name"),
                _terminal(":"),
                self._field(fields, "annotation", "type-annotation"),
                _optional(_concat(_terminal("="), value)),
            )
        if node_type is ast.Return:
            return _concat(_terminal("return"), self._field(fields, "value", ""))
        if node_type is ast.Expr:
            return self._field(fields, "value", "expression")
        if node_type is ast.Pass:
            return _terminal("pass")
        if node_type is ast.For:
            return _concat(
                _terminal("for"),
                self._field(fields, "target", "name"),
                _terminal("in"),
                self._field(fields, "iter", "expression"),
                _terminal(":"),
                self._field(fields, "body", "block"),
            )
        if node_type is ast.withitem:
            alias = _concat(
                _terminal("as"),
                self._field(fields, "optional_vars", "name"),
            )
            return _concat(
                self._field(fields, "context_expr", "expression"),
                _optional(alias),
            )
        if node_type is ast.With:
            return _concat(
                _terminal("with"),
                self._list_pattern(fields.get("items", SequencePattern())),
                _terminal(":"),
                self._field(fields, "body", "block"),
            )
        if node_type is ast.Module:
            body = fields.get("body", SequencePattern())
            if isinstance(body, RepeatPattern):
                return _repeated(self.visit(body.pattern), body.minimum, _text("newline"))
            return self._list_pattern(body)
        if node_type is ast.FunctionDef:
            returns = self._optional_field(fields, "returns", "return-type")
            return _concat(
                _terminal("def"),
                _text("name"),
                _delimited("(", self._field(fields, "args", "signature"), ")"),
                _optional(_concat(_terminal("->"), returns)),
                _terminal(":"),
                self._field(fields, "body", "block"),
            )
        if node_type is ast.MatMult:
            return _terminal("@")
        operator = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**",
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.And: "and",
            ast.Or: "or",
            ast.UAdd: "+",
            ast.USub: "-",
            ast.Not: "not",
        }.get(node_type)
        if operator is not None:
            return _terminal(operator)
        if node_type is ast.Load:
            return _text("")
        if node_type in {ast.expr, ast.stmt}:
            structural = [
                part
                for part in pattern.parts
                if not isinstance(part, (CapturePattern, PredicatePattern))
            ]
            if structural:
                return self.visit(structural[-1])
            predicates = [part for part in pattern.parts if isinstance(part, PredicatePattern)]
            if predicates:
                return self.visit(predicates[-1])
            return _text("expression" if node_type is ast.expr else "statement")

        parts = tuple(self.visit(part) for part in pattern.parts)
        return _concat(_text(_grammar_name(node_type.__name__)), *parts)

    def visit(self, pattern: Any) -> _Expr:
        if isinstance(pattern, ElementPattern):
            return self._element(pattern)
        if isinstance(pattern, AstNodePattern):
            return self._ast_node(pattern)
        if isinstance(pattern, FieldPattern):
            return self.visit(pattern.pattern)
        if isinstance(pattern, LiteralPattern):
            if pattern.value_type is not None:
                types = (
                    pattern.value_type
                    if isinstance(pattern.value_type, tuple)
                    else (pattern.value_type,)
                )
                names = {
                    bool: "boolean-literal",
                    int: "integer-literal",
                    float: "float-literal",
                    str: "string-literal",
                }
                return _choice(
                    *(_text(names.get(item, f"{item.__name__}-literal")) for item in types)
                )
            if pattern.value is dataclasses.MISSING:
                return _text("literal")
            return _text(repr(pattern.value))
        if isinstance(pattern, ReferencePattern):
            return _text("primary")
        if isinstance(pattern, SequencePattern):
            return _separated(tuple(self.visit(item) for item in pattern.patterns), _terminal(","))
        if isinstance(pattern, ChoicePattern):
            return _choice(*(self.visit(item) for item in pattern.patterns))
        if isinstance(pattern, ConditionPattern):
            return self.visit(pattern.pattern)
        if isinstance(pattern, OptionalPattern):
            return _optional(self.visit(pattern.pattern))
        if isinstance(pattern, RepeatPattern):
            return _repeated(self.visit(pattern.pattern), pattern.minimum, _terminal(","))
        if isinstance(pattern, (ChildPattern, BranchPattern, BindPattern, LazyPattern)):
            return self.visit(pattern.pattern)
        if isinstance(pattern, PredicatePattern):
            return _text(_grammar_name(pattern.label))
        if isinstance(pattern, CapturePattern):
            return _text(_grammar_name(pattern.name))
        raise TypeError(f"unsupported executable pattern {type(pattern).__name__}")

    def _format_production(self, name: str, rhs: _Expr, width: int) -> list[str]:
        prefix = f"{name:<{width}} ::= "
        continuation = " " * (width + 5)
        if rhs.alternatives:
            lines: list[str] = []
            for index, alternative in enumerate(rhs.alternatives):
                marker = "" if index == 0 else "| "
                lines.extend(
                    textwrap.wrap(
                        marker + alternative,
                        width=self.line_width,
                        initial_indent=prefix if index == 0 else continuation,
                        subsequent_indent=continuation + "  ",
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                )
            return lines
        return textwrap.wrap(
            rhs.text,
            width=self.line_width,
            initial_indent=prefix,
            subsequent_indent=continuation,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [prefix.rstrip()]

    def render(self, root: ElementPattern[Any]) -> str:
        root_name = root.element_name
        if not root_name:
            raise TypeError(f"{type(root).__name__} has no explicit element_name")
        self._element(root)
        width = max(len(name) for name, _ in self._productions)
        lines = [
            f"; root: {_grammar_name(root_name)}",
            '; literal: Python ast.Constant syntax, e.g. 1, "bf16", or None',
            "; name: Python variable name; primary: name/attribute base for calls and subscripts",
            "; expression: Python syntax composed from literals, names, primaries, and operators",
            "; runtime-expression: expression lowered to a TileFoundry IR Expr",
        ]
        for name, rhs in self._productions:
            lines.extend(self._format_production(name, rhs, width))
        return "\n".join(lines)


def render_grammar(root: ElementPattern[Any] | None = None) -> str:
    if root is None:
        from .pattern_nodes import FunctionPattern

        root = FunctionPattern()
    return RenderVisitor().render(root)


__all__ = ["RenderVisitor", "render_grammar"]
