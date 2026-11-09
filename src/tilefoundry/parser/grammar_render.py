"""Render the executable Pattern graph as deterministic surface EBNF."""

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
from .pattern_nodes import FunctionPattern


def to_grammar_name(value: str) -> str:
    return value.replace("_", "-").replace(" ", "-").lower()


@dataclasses.dataclass(frozen=True)
class Fragment:
    text: str
    alternatives: tuple[str, ...] = ()

    def flat(self) -> str:
        if self.alternatives:
            return "(" + " | ".join(self.alternatives) + ")"
        return self.text


def plain(value: str) -> Fragment:
    return Fragment(value)


def concat(*parts: Fragment | None) -> Fragment:
    return plain(" ".join(part.flat() for part in parts if part and part.flat()))


def choice(*parts: Fragment) -> Fragment:
    alternatives: list[str] = []
    for part in parts:
        values = part.alternatives or (part.text,)
        for value in values:
            if value and value not in alternatives:
                alternatives.append(value)
    if len(alternatives) == 1:
        return plain(alternatives[0])
    return Fragment("", tuple(alternatives))


def terminal(value: str) -> Fragment:
    return plain(repr(value))


def optional(value: Fragment) -> Fragment:
    return plain(f"({value.flat()})?")


def delimited(left: str, value: Fragment, right: str) -> Fragment:
    return concat(terminal(left), value, terminal(right))


def separated(items: tuple[Fragment, ...], separator: Fragment) -> Fragment:
    if not items:
        return plain("")
    output = items[0]
    for item in items[1:]:
        output = concat(output, separator, item)
    return output


def repeated(item: Fragment, minimum: int, separator: Fragment) -> Fragment:
    required = concat(item, plain(f"({separator.flat()} {item.flat()})*"))
    return required if minimum else optional(required)


def field_patterns(pattern: AstNodePattern) -> dict[str, object]:
    return {part.name: part.pattern for part in pattern.parts if isinstance(part, FieldPattern)}


class RenderVisitor:
    """Project executable AST patterns into readable Python authoring EBNF."""

    def __init__(self, *, line_width: int = 100):
        self.line_width = line_width
        self.seen: set[str] = set()
        self.seen_named: set[str] = set()
        self.productions: list[tuple[str, Fragment]] = []

    def named_pattern(self, pattern: Any) -> Fragment | None:
        name = pattern.grammar_name
        if not isinstance(name, str):
            return None
        grammar_name = to_grammar_name(name)
        if grammar_name not in self.seen_named:
            self.seen_named.add(grammar_name)
            self.productions.append((grammar_name, self.ast_node(pattern, allow_named=False)))
        return plain(grammar_name)

    def element(self, pattern: ElementPattern[Any]) -> Fragment:
        name = pattern.element_name
        if not name:
            raise TypeError(f"{type(pattern).__name__} has no explicit element_name")
        grammar_name = to_grammar_name(name)
        if name in self.seen:
            return plain(grammar_name)
        self.seen.add(name)
        if pattern.syntax is None:
            raise TypeError(f"{type(pattern).__name__} has no executable syntax")
        rhs = self.visit(pattern.syntax)
        for pattern_class in getattr(pattern, "ATTRIBUTE_PATTERNS", ()):
            self.visit(pattern_class())
        self.productions.append((grammar_name, rhs))
        return plain(grammar_name)

    def list_pattern(self, pattern: object) -> Fragment:
        comma = terminal(",")
        if isinstance(pattern, RepeatPattern):
            return repeated(self.visit(pattern.pattern), pattern.minimum, comma)
        if isinstance(pattern, SequencePattern):
            return separated(tuple(self.visit(item) for item in pattern.patterns), comma)
        if isinstance(pattern, ChoicePattern):
            return choice(*(self.list_pattern(item) for item in pattern.patterns))
        return self.visit(pattern)

    def field(self, fields: dict[str, object], name: str, fallback: str) -> Fragment:
        pattern = fields.get(name)
        return plain(fallback) if pattern is None else self.visit(pattern)

    def optional_field(self, fields: dict[str, object], name: str, fallback: str) -> Fragment:
        pattern = fields.get(name)
        if isinstance(pattern, OptionalPattern):
            pattern = pattern.pattern
        return plain(fallback) if pattern is None else self.visit(pattern)

    def ast_node(self, pattern: AstNodePattern, *, allow_named: bool = True) -> Fragment:
        if allow_named:
            named = self.named_pattern(pattern)
            if named is not None:
                return named
        node_type = pattern.node_type
        for part in pattern.parts:
            self.visit(part)
        fields = field_patterns(pattern)

        if node_type is ast.Constant:
            predicates = [part for part in pattern.parts if isinstance(part, PredicatePattern)]
            if predicates:
                return self.visit(predicates[-1])
            value = fields.get("value")
            if isinstance(value, CapturePattern) or value is None:
                return choice(
                    plain("None"),
                    plain("Ellipsis"),
                    plain("boolean-literal"),
                    plain("integer-literal"),
                    plain("float-literal"),
                    plain("complex-literal"),
                    plain("string-literal"),
                    plain("bytes-literal"),
                )
            return self.visit(value)
        if node_type is ast.Name:
            identifier = fields.get("id")
            if isinstance(identifier, (LiteralPattern, ChoicePattern)):
                return self.visit(identifier)
            return plain("identifier")
        if node_type is ast.Attribute:
            value = fields.get("value")
            fallback = "identifier" if isinstance(value, AstNodePattern) and value.node_type is ast.Name else "primary"
            attr = fields.get("attr")
            return concat(
                self.visit(value) if value is not None else plain(fallback),
                terminal("."),
                self.visit(attr)
                if isinstance(attr, (LiteralPattern, ChoicePattern))
                else plain("identifier"),
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
                values = plain("")
            else:
                values = self.list_pattern(values_pattern)
            return delimited(left, values, right)
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
            entry = concat(key, terminal(":"), value)
            return delimited("{", repeated(entry, 0, terminal(",")), "}")
        if node_type is ast.BinOp:
            return concat(
                self.field(fields, "left", "expression"),
                self.field(fields, "op", "binary-op"),
                self.field(fields, "right", "expression"),
            )
        if node_type is ast.UnaryOp:
            return concat(
                self.field(fields, "op", "unary-op"),
                self.field(fields, "operand", "expression"),
            )
        if node_type is ast.Compare:
            return concat(
                self.field(fields, "left", "expression"),
                self.list_pattern(fields.get("ops", SequencePattern())),
                self.list_pattern(fields.get("comparators", SequencePattern())),
            )
        if node_type is ast.BoolOp:
            values = fields.get("values", SequencePattern())
            if isinstance(values, SequencePattern) and len(values.patterns) == 2:
                return concat(
                    self.visit(values.patterns[0]),
                    self.field(fields, "op", "boolean-op"),
                    self.visit(values.patterns[1]),
                )
            return self.list_pattern(values)
        if node_type is ast.Call:
            args = fields.get("args")
            keywords = fields.get("keywords")
            if not isinstance(args, RepeatPattern) or not isinstance(keywords, RepeatPattern):
                arguments = concat(
                    self.list_pattern(args) if args is not None else None,
                    self.list_pattern(keywords) if keywords is not None else None,
                )
                return concat(
                    self.field(fields, "func", "callee"),
                    delimited("(", arguments, ")"),
                )
            argument_patterns: list[Fragment] = []
            minimum = 0
            for name in ("args", "keywords"):
                values = fields.get(name)
                if isinstance(values, RepeatPattern):
                    argument_patterns.append(self.visit(values.pattern))
                    minimum = max(minimum, values.minimum)
            arguments = (
                repeated(choice(*argument_patterns), minimum, terminal(","))
                if argument_patterns
                else plain("")
            )
            return concat(
                self.field(fields, "func", "callee"),
                delimited("(", arguments, ")"),
            )
        if node_type is ast.keyword:
            return concat(
                self.field(fields, "arg", "name"),
                terminal("="),
                self.field(fields, "value", "expression"),
            )
        if node_type is ast.Slice:
            lower = self.field(fields, "lower", "")
            upper = self.field(fields, "upper", "")
            step = self.optional_field(fields, "step", "")
            return concat(
                lower,
                terminal(":"),
                upper,
                optional(concat(terminal(":"), step)),
            )
        if node_type is ast.Subscript:
            return concat(
                self.field(fields, "value", "expression"),
                delimited("[", self.field(fields, "slice", "expression"), "]"),
            )
        if node_type is ast.arguments:
            return self.list_pattern(fields.get("args", SequencePattern()))
        if node_type is ast.arg:
            return concat(
                plain("name"),
                terminal(":"),
                self.field(fields, "annotation", "type-annotation"),
            )
        if node_type is ast.Assign:
            return concat(
                self.list_pattern(fields.get("targets", SequencePattern())),
                terminal("="),
                self.field(fields, "value", "expression"),
            )
        if node_type is ast.AnnAssign:
            value = self.optional_field(fields, "value", "expression")
            return concat(
                self.field(fields, "target", "name"),
                terminal(":"),
                self.field(fields, "annotation", "type-annotation"),
                optional(concat(terminal("="), value)),
            )
        if node_type is ast.Return:
            return concat(terminal("return"), self.field(fields, "value", ""))
        if node_type is ast.Expr:
            return self.field(fields, "value", "expression")
        if node_type is ast.Pass:
            return terminal("pass")
        if node_type is ast.For:
            return concat(
                terminal("for"),
                self.field(fields, "target", "name"),
                terminal("in"),
                self.field(fields, "iter", "expression"),
                terminal(":"),
                self.field(fields, "body", "block"),
            )
        if node_type is ast.withitem:
            alias = concat(
                terminal("as"),
                self.field(fields, "optional_vars", "name"),
            )
            return concat(
                self.field(fields, "context_expr", "expression"),
                optional(alias),
            )
        if node_type is ast.With:
            return concat(
                terminal("with"),
                self.list_pattern(fields.get("items", SequencePattern())),
                terminal(":"),
                self.field(fields, "body", "block"),
            )
        if node_type is ast.Module:
            body = fields.get("body", SequencePattern())
            if isinstance(body, RepeatPattern):
                return repeated(self.visit(body.pattern), body.minimum, plain("newline"))
            return self.list_pattern(body)
        if node_type is ast.FunctionDef:
            returns = self.optional_field(fields, "returns", "return-type")
            return concat(
                terminal("def"),
                plain("name"),
                delimited("(", self.field(fields, "args", "signature"), ")"),
                optional(concat(terminal("->"), returns)),
                terminal(":"),
                self.field(fields, "body", "block"),
            )
        if node_type is ast.MatMult:
            return terminal("@")
        operator = {
            ast.Add: "+",
            ast.BitOr: "|",
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
            return terminal(operator)
        if node_type is ast.Load:
            return plain("")
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
            return plain("expression" if node_type is ast.expr else "statement")

        parts = tuple(self.visit(part) for part in pattern.parts)
        return concat(plain(to_grammar_name(node_type.__name__)), *parts)

    def visit(self, pattern: Any) -> Fragment:
        if isinstance(pattern, ElementPattern):
            return self.element(pattern)
        if isinstance(pattern, AstNodePattern):
            return self.ast_node(pattern)
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
                return choice(
                    *(plain(names.get(item, f"{item.__name__}-literal")) for item in types)
                )
            if pattern.value is dataclasses.MISSING:
                return plain("literal")
            return plain(repr(pattern.value))
        if isinstance(pattern, ReferencePattern):
            return plain("primary")
        if isinstance(pattern, SequencePattern):
            return separated(tuple(self.visit(item) for item in pattern.patterns), terminal(","))
        if isinstance(pattern, ChoicePattern):
            return choice(*(self.visit(item) for item in pattern.patterns))
        if isinstance(pattern, ConditionPattern):
            return self.visit(pattern.pattern)
        if isinstance(pattern, OptionalPattern):
            return optional(self.visit(pattern.pattern))
        if isinstance(pattern, RepeatPattern):
            return repeated(self.visit(pattern.pattern), pattern.minimum, terminal(","))
        if isinstance(pattern, (ChildPattern, BranchPattern, BindPattern, LazyPattern)):
            return self.visit(pattern.pattern)
        if isinstance(pattern, PredicatePattern):
            return plain(to_grammar_name(pattern.label))
        if isinstance(pattern, CapturePattern):
            return plain(to_grammar_name(pattern.name))
        raise TypeError(f"unsupported executable pattern {type(pattern).__name__}")

    def format_production(self, name: str, rhs: Fragment, width: int) -> list[str]:
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
        self.element(root)
        width = max(len(name) for name, _ in self.productions)
        lines = [
            f"; root: {to_grammar_name(root_name)}",
            '; literal: Python ast.Constant syntax, e.g. 1, "bf16", or None',
            "; name: Python variable name; primary: name/attribute base for calls and subscripts",
            "; expression: Python syntax composed from literals, names, primaries, and operators",
            "; runtime-expression: expression lowered to a TileFoundry IR Expr",
        ]
        for name, rhs in self.productions:
            lines.extend(self.format_production(name, rhs, width))
        return "\n".join(lines)


def render_grammar(root: ElementPattern[Any] | None = None) -> str:
    if root is None:
        root = FunctionPattern()
    return RenderVisitor().render(root)


__all__ = ["RenderVisitor", "render_grammar"]
