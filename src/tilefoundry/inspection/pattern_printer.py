"""Human-readable operation-declaration patterns and match refusals."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from types import MappingProxyType

from tilefoundry import services
from tilefoundry.ir import clause, expr, pattern, types

from .printer_base import PythonPrinter

UNNAMED = "_"
ARRANGEMENT = "every arrangement"
BINARY_OPERATORS = MappingProxyType({
    **{op: (symbol, 1) for op, symbol in (
        ("eq", "=="), ("ne", "!="), ("lt", "<"), ("le", "<="), ("gt", ">"), ("ge", ">="),
    )},
    "and": ("&", 3),
    "or": ("|", 2),
    "add": ("+", 4),
    "sub": ("-", 4),
    "mul": ("*", 5),
    "floordiv": ("//", 5),
    "mod": ("%", 5),
})


def with_predicates(written: str, rules: tuple[str, ...]) -> str:
    if not rules:
        return written
    return "\n".join((written, "  predicates:", *(
        f"    {line}" for rule in rules for line in rule.splitlines()
    )))


def written_tuple(items: tuple[str, ...]) -> str:
    written = ", ".join(items)
    return f"({written},)" if len(items) == 1 else f"({written})"


class PatternPrinter:
    """Render pattern trees without a Python print context."""

    def written(self, node, name: str = UNNAMED) -> str:
        """Write the value shape admitted by *pattern*."""
        if not isinstance(node, pattern.Pattern):
            return self.written_value(node)
        return self.dispatch("visit", node, name)

    def rules(self, node, name: str = UNNAMED) -> tuple[str, ...]:
        """Write the ordered, de-duplicated conditions imposed by *pattern*."""
        if not isinstance(node, pattern.Pattern):
            return ()
        lines = tuple(
            dict.fromkeys(
                (
                    *self.dispatch("rules", node, name),
                    *self.rules_of(node.predicates, name, predicates=True),
                )
            )
        )
        return lines

    def described(self, node, name: str = UNNAMED) -> str:
        """Write a value shape followed by its predicate lines, when present."""
        written, rules = self.written(node, name), self.rules(node, name)
        if rules and all("\n" not in line for line in (written, *rules)):
            return written + " where " + "; ".join(rules)
        return with_predicates(written, rules)


    def alternatives(self, node, bindings=()) -> tuple:
        """Flatten declaration branches into ``(bindings, leaf)`` pairs."""
        if not isinstance(node, pattern.Pattern):
            return ((tuple(bindings), node),)
        return self.dispatch("alternatives", node, tuple(bindings))

    def refusal_reason(self, refusal, parameter=None) -> str:
        """The failed node's reason, optionally named by its instruction operand."""
        if refusal is None:
            reason = "the pattern did not match"
        else:
            node = refusal.pattern
            if isinstance(node, pattern.Formula):
                failed = pattern.failing((node,), refusal.bindings)
                if failed is not None:
                    node = failed
            reason = (
                self.dispatch("refused", node, refusal.subject)
                if isinstance(node, pattern.Pattern)
                else self.refused_Pattern(node, refusal.subject)
            )
        return reason if parameter is None else f"{parameter}: {reason}"

    def match_data(self, paths, reason="") -> dict:
        """Merge ordered match paths without separating parameters from internal captures."""
        merged = {}
        for path in paths:
            for name, value in path:
                written = (
                    self.subject_value(value)
                    if isinstance(value, float) else getattr(value, "name", str(value))
                )
                values = merged.setdefault(name, [])
                if written not in values:
                    values.append(written)
        return {
            "path": [{"name": name, "value": "|".join(values)} for name, values in merged.items()],
            "reason": reason,
        }

    def match_text(self, identifier, data) -> str:
        """Write an instruction and its ordered path, followed by a refusal reason."""
        path = ", ".join(f"{item['name']}={item['value']}" for item in data["path"])
        first = identifier + (f"  {path}" if path else "")
        reason = data["reason"]
        return first if not reason else f"{first}\n  {reason}"

    def refusal(self, refusal) -> str:
        """Render structural evidence with the same path/reason form as instruction reports."""
        path = () if refusal is None else tuple(refusal.bindings.items())
        data = self.match_data((path,), self.refusal_reason(refusal))
        return self.match_text("", data).lstrip()

    def subject_value(self, value) -> str:
        if isinstance(value, (expr.DType, expr.StorageKind)):
            return self.written_value(value)
        return PythonPrinter().print(value)

    def refused_Pattern(self, node, subject) -> str:
        return f"{self.subject_value(subject)} does not match {self.written(node)}"

    def refused_Predicate(self, node, subject) -> str:
        return "could not satisfy " + "; ".join(self.rules(node))

    def refused_Switch(self, node, subject) -> str:
        options = ", ".join(self.written_value(value) for value, _ in node.branches)
        return (
            f"{node.param} has no matching branch for "
            f"{self.subject_value(subject)}; branches: {options}"
        )

    def declaration(self, op_type) -> dict[str, tuple[str, ...]]:
        """Render the declaration directly, retaining symbolic conditions."""
        sections = []
        parameters = tuple(getattr(op_type, "parameters", ())) or expr.collect_param_defs(op_type)
        parameters_by_name = {param.name: param for param in parameters}
        if parameters:
            sections.append(("parameters", tuple((param.name, param.pattern) for param in parameters)))
        roles = tuple(
            (name, getattr(op_type, name))
            for name in ("C", "A", "B")
            if isinstance(getattr(op_type, name, None), pattern.Pattern)
        )
        if roles:
            sections.append(("operands", roles))
            sections.append(
                (
                    "attributes",
                    (("execution mesh", pattern.declared_execution_mesh(op_type)),),
                )
            )
        if not roles and parameters:
            operands = tuple(
                (param.name, param.pattern)
                for param in parameters
                if param.kind == "input"
            )
            attributes = tuple(
                (param.name, param.pattern)
                for param in parameters
                if param.kind == "attribute"
            )
            sections = [
                (heading, items)
                for heading, items in (("operands", operands), ("attributes", attributes))
                if items
            ]

        rendered = {}
        for heading, items in sections:
            lines = []
            width = max((len(name) for name, _ in items), default=0)
            for name, node in items:
                parameter = parameters_by_name.get(name)
                description = "any value" if node is None else self.described(node, name)
                qualifier = ""
                if parameter is not None:
                    annotation = parameter.annotation
                    if isinstance(annotation, type) and issubclass(annotation, Enum):
                        if node is None or isinstance(node, pattern.Wildcard):
                            qualifier += " in {" + ", ".join(
                                self.written_value(value) for value in annotation
                            ) + "}"
                    elif annotation is not object:
                        qualifier += f" (type {getattr(annotation, '__name__', annotation)})"
                    if parameter.optional:
                        qualifier += " or None"
                described = description.splitlines()
                described[0] += qualifier
                default = ""
                if parameter is not None and parameter.has_default:
                    default = f" (default {self.written_value(parameter.default)})"
                prefix = f"    {name.ljust(width)}  "
                lines.append(prefix + described[0] + default)
                continuation = " " * len(prefix)
                lines.extend(continuation + line for line in described[1:])
            rendered[heading] = tuple(lines)
        return rendered

    def dispatch(self, operation: str, node: pattern.Pattern, argument):
        for cls in type(node).__mro__:
            visitor = getattr(self, f"{operation}_{cls.__name__}", None)
            if visitor is not None:
                return visitor(node, argument)
        raise NotImplementedError(
            f"{type(node).__name__} has no PatternPrinter {operation} visitor"
        )

    def written_dim(self, value) -> str:
        if isinstance(value, expr.DimVar):
            return value.name
        if expr.is_dim_op_call(value):
            left, right = (self.written_dim(arg) for arg in value.args)
            if isinstance(value.target, expr.DimFloorDiv):
                return f"{left}/{right}"
            if isinstance(value.target, expr.DimMul):
                return f"{left}*{right}"
            return f"({left} {type(value.target).__name__} {right})"
        return str(getattr(value, "value", value))

    def written_value(self, value) -> str:
        if isinstance(value, float):
            return self.subject_value(value)
        if isinstance(value, types.Broadcast):
            return "B()"
        if isinstance(value, expr.DType):
            return value.name
        if isinstance(value, types.Split):
            return f"S({value.axis})"
        if isinstance(value, expr.StorageKind):
            return value.name.lower()
        if isinstance(value, Enum):
            try:
                return f"{expr.get_registered_name(value)}.{value.name}"
            except ValueError:
                return f"{type(value).__name__}.{value.name}"
        if isinstance(value, tuple):
            return written_tuple(tuple(self.written_value(item) for item in value))
        if isinstance(value, list):
            return written_tuple(tuple(self.written_value(item) for item in value))
        if isinstance(value, expr.DimVar) or expr.is_dim_op_call(value):
            return self.written_dim(value)
        return UNNAMED if clause.is_layout_wildcard(value) else str(value)


    def written_grouped(self, modes) -> str:
        if isinstance(modes, tuple):
            return written_tuple(tuple(self.written_grouped(mode) for mode in modes))
        return self.written(modes)

    def rules_of(self, values, name: str = UNNAMED, *, predicates=False) -> tuple[str, ...]:
        lines = []
        for value in values:
            if not predicates:
                lines.extend(self.rules(value, name))
            elif isinstance(value, pattern.Pattern):
                lines.extend(self.dispatch("predicate", value, name))
            else:
                lines.append(f"{name} == {self.written_value(value)}")
        return tuple(lines)

    def predicate_Pattern(self, node, name) -> tuple[str, ...]:
        return self.rules(node, name)

    def predicate_Wildcard(self, node, name) -> tuple[str, ...]:
        binding = (f"{name} matches {node.name or 'any value'}",) if node.name != name else ()
        return (*binding, *self.rules(node, name))

    def visit_Wildcard(self, node, name) -> str:
        return "any value" if node.name == name else node.name or name

    def rules_Wildcard(self, node, name) -> tuple[str, ...]:
        return ()

    def visit_Star(self, node, name) -> str:
        return f"*{self.written(node.pattern, name)}"

    def rules_Star(self, node, name) -> tuple[str, ...]:
        return self.rules(node.pattern, name)

    def visit_Or(self, node, name) -> str:
        return "one of {" + "; ".join(
            self.described(item, name) for item in node.patterns
        ) + "}"

    def rules_Or(self, node, name) -> tuple[str, ...]:
        return ()

    def predicate_Or(self, node, name) -> tuple[str, ...]:
        return (self.described(node, name),)

    def visit_And(self, node, name) -> str:
        return " and ".join(self.described(part, name) for part in node.parts)

    def rules_And(self, node, name) -> tuple[str, ...]:
        return ()

    def visit_Sequence(self, node, name) -> str:
        return written_tuple(tuple(self.written(item, name) for item in node.patterns))

    def rules_Sequence(self, node, name) -> tuple[str, ...]:
        return self.rules_of(node.patterns, name)

    def visit_Range(self, node, name) -> str:
        return name

    def rules_Range(self, node, name) -> tuple[str, ...]:
        if node.lo is None:
            return (f"{name} <= {node.hi}",)
        if node.hi is None:
            return (f"{node.lo} <= {name}",)
        return (f"{node.lo} <= {name} <= {node.hi}",)

    def visit_Switch(self, node, name) -> str:
        branches = []
        for value, branch in node.branches:
            lines = self.described(branch, name).splitlines()
            branches.append(f"  {node.param}={self.written_value(value)}: {lines[0]}")
            branches.extend(f"    {line}" for line in lines[1:])
        return "\n".join((f"switch {node.param}:", *branches))

    def rules_Switch(self, node, name) -> tuple[str, ...]:
        return ()

    def predicate_Switch(self, node, name) -> tuple[str, ...]:
        return (self.described(node, name),)

    def visit_Layout(self, node, name) -> str:
        if node.shape is None and node.strides is None:
            return "layout"
        shape = name if node.shape is None else self.written_grouped(tuple(node.shape))
        strides = (
            name if node.strides is None else self.written_grouped(tuple(node.strides))
        )
        return f"Layout({shape}, {strides})"

    def rules_Layout(self, node, name) -> tuple[str, ...]:
        return self.rules_of(node.positions())

    def visit_Swizzle(self, node, name) -> str:
        return (
            f"Swizzle({self.written(node.bits)}, {self.written(node.base)}, "
            f"{self.written(node.shift)})"
        )

    def rules_Swizzle(self, node, name) -> tuple[str, ...]:
        return self.rules_of((node.bits, node.base, node.shift))

    def visit_ComposedLayout(self, node, name) -> str:
        return (
            f"ComposedLayout({self.written(node.inner)}, "
            f"{self.written(node.offset)}, {self.written(node.outer)})"
        )

    def rules_ComposedLayout(self, node, name) -> tuple[str, ...]:
        return self.rules_of((node.inner, node.offset, node.outer))

    def visit_Mesh(self, node, name) -> str:
        topologies = (
            repr(node.topologies)
            if isinstance(node.topologies, tuple)
            else self.written(node.topologies)
        )
        return f"Mesh({topologies}, {self.written(node.layout)})"

    def rules_Mesh(self, node, name) -> tuple[str, ...]:
        return self.rules(node.layout)

    def visit_Ranked(self, node, name) -> str:
        return name

    def rules_Ranked(self, node, name) -> tuple[str, ...]:
        return (f"{name} has rank > 0",)

    def visit_Tensor(self, node, name) -> str:
        fields = []
        for field in ("shape", "dtype", "storage", "layout"):
            value = getattr(node, field)
            if value is None:
                continue
            written = (
                written_tuple(tuple(
                    self.described(item, f"{name}.shape[{index}]")
                    for index, item in enumerate(value)
                ))
                if field == "shape" else self.described(value, f"{name}.{field}")
            )
            lines = written.splitlines()
            fields.append(f"  {field}={lines[0]}")
            fields.extend(f"    {line}" for line in lines[1:])
        if node.layout is None:
            fields.append("  layout=any arrangement")
        if all("\n" not in line for line in fields) and all(not line.startswith("    ") for line in fields):
            return "Tensor(" + ", ".join(line.strip() for line in fields) + ")"
        return "\n".join(("Tensor:", *fields))

    def rules_Tensor(self, node, name) -> tuple[str, ...]:
        return ()

    def visit_ShardLayout(self, node, name) -> str:
        attrs = (
            written_tuple(tuple(self.written_value(attr) for attr in node.attrs))
            if isinstance(node.attrs, tuple)
            else self.written(node.attrs)
        )
        return (
            f"ShardLayout({self.written(node.layout)}, {attrs}, "
            f"{self.written(node.mesh)})"
        )

    def rules_ShardLayout(self, node, name) -> tuple[str, ...]:
        return self.rules_of((node.layout, node.attrs, node.mesh))

    def visit_AtomPattern(self, node, name) -> str:
        return "one of " + ", ".join(expr.get_registered_name(item) for item in node.declarations)

    def rules_AtomPattern(self, node, name) -> tuple[str, ...]:
        return ()

    def visit_FromAtom(self, node, name) -> str:
        return f"the {node.role} operand of its atom"

    def rules_FromAtom(self, node, name) -> tuple[str, ...]:
        return ()

    def visit_Predicate(self, node, name) -> str:
        return name

    def rules_Predicate(self, node, name) -> tuple[str, ...]:
        raise NotImplementedError(f"{type(node).__name__} has no PatternPrinter rules visitor")

    def visit_Formula(self, node, name) -> str:
        return name

    def rules_Formula(self, node, name) -> tuple[str, ...]:
        return (self.written_expression(node),)

    def visit_Forward(self, node, name) -> str:
        return name

    def rules_Forward(self, node, name) -> tuple[str, ...]:
        subject = "each top-level mode" if node.per_mode else ARRANGEMENT
        return (f"{subject} has no backward step",)

    def visit_Injective(self, node, name) -> str:
        return name

    def rules_Injective(self, node, name) -> tuple[str, ...]:
        subject = "each top-level mode" if node.per_mode else ARRANGEMENT
        return (f"{subject} reaches each of its own slots exactly once",)

    def visit_WholeVectors(self, node, name) -> str:
        return name

    def rules_WholeVectors(self, node, name) -> tuple[str, ...]:
        return (
            "every run: each tile axis's modes walked fastest first, contiguous ones "
            "joined; the run at step 1 and every other step a whole number of vectors",
            *self.rules(node.width),
        )

    def visit_PlainArrangement(self, node, name) -> str:
        return name

    def rules_PlainArrangement(self, node, name) -> tuple[str, ...]:
        return (
            "every plain arrangement is a static strided layout with no swizzle and zero "
            "offset, and, where it is sharded, one every participant holds whole",
        )

    def visit_BoxDims(self, node, name) -> str:
        return name

    def rules_BoxDims(self, node, name) -> tuple[str, ...]:
        reading = (
            "every box: each tile axis's modes, contiguous ones joined up to "
            f"{node.limit} elements, one dim each, in increasing step"
        )
        return (reading, *self.rules_of(node.dims))

    def visit_TensorMap(self, node, name) -> str:
        return name

    def rules_TensorMap(self, node, name) -> tuple[str, ...]:
        return (
            "every tensormap: one dim per mode of the tile, the mode at step 1 first",
            *self.rules_of(node.steps),
        )

    def written_expression(self, value, parent=0) -> str:
        if isinstance(value, float):
            return self.subject_value(value)
        name = type(value).__name__
        if name == "Formula":
            if value.op == "not":
                return f"~({self.written_expression(value.args[0])})"
            if value.op == "forall":
                term, token, body = value.args
                return (
                    f"all({self.written_expression(body)} for x{token} "
                    f"in {self.written_expression(term)})"
                )
            if value.op == "in":
                values = ", ".join(self.written_value(item) for item in value.args[1])
                return f"{self.written_expression(value.args[0])} in {{{values}}}"
            return self.written_binary(value, parent)
        if name == "Term":
            if value.op == "constant":
                return self.written_expression(value.args[0])
            if value.op in {"variable", "bits"}:
                return (
                    value.args[0]
                    if value.op == "variable"
                    else f"Bits({value.args[0]!r})"
                )
            if value.op == "bound":
                return f"x{value.args[0]}"
            if value.op == "element":
                return f"{self.written_value(value.args[0])}[{self.written_expression(value.args[1])}]"
            if value.op in {"sum", "count"}:
                return f"{value.op.title()}({self.written_expression(value.args[0])})"
            return self.written_binary(value, parent)
        return self.written_value(value) if isinstance(value, (expr.DType, expr.StorageKind, Enum, tuple)) else repr(value)

    def written_binary(self, value, parent):
        symbol, precedence = BINARY_OPERATORS[value.op]
        text = (
            f"{self.written_expression(value.args[0], precedence + (precedence == 1))} {symbol} "
            f"{self.written_expression(value.args[1], precedence + 1)}"
        )
        return f"({text})" if precedence < parent else text

    def alternatives_Pattern(self, node, bindings) -> tuple:
        return ((bindings, node),)

    def alternatives_OrPattern(self, node, bindings) -> tuple:
        if not any(isinstance(item, pattern.Pattern) for item in node.patterns):
            return ((bindings, node),)
        return tuple(
            held
            for alternative in node.patterns
            for held in self.alternatives(alternative, bindings)
        )

    def alternatives_SwitchPattern(self, node, bindings) -> tuple:
        known = dict(bindings)
        if node.param in known:
            branch = next(
                (branch for value, branch in node.branches if value == known[node.param]), None
            )
            return () if branch is None else self.alternatives(branch, bindings)
        return tuple(
            held
            for value, branch in node.branches
            for held in self.alternatives(branch, (*bindings, (node.param, value)))
        )

    def alternatives_TensorPattern(self, node, bindings) -> tuple:

        states = ((bindings, node),)
        for field in ("dtype", "storage"):
            states = tuple(
                (held, replace(tensor, **{field: alternative}))
                for bound, tensor in states
                for held, alternative in self.alternatives(getattr(tensor, field), bound)
            )
        if node.shape is not None:
            for index in range(len(node.shape)):
                states = tuple(
                    (
                        held,
                        replace(
                            tensor,
                            shape=(*tensor.shape[:index], alternative, *tensor.shape[index + 1 :]),
                        ),
                    )
                    for bound, tensor in states
                    for held, alternative in self.alternatives(tensor.shape[index], bound)
                )
        return tuple((bound, pattern.selected(tensor, dict(bound))) for bound, tensor in states)

    def alternatives_ShardLayoutPattern(self, node, bindings) -> tuple:
        return tuple(
            (held, replace(node, layout=alternative))
            for held, alternative in self.alternatives(node.layout, bindings)
        )

__all__ = ["PatternPrinter", "describe_refusal", "refusal_reason"]


def describe_refusal(instruction: str, path: tuple, reason: object) -> str:
    """Text of a schedule refusal of *instruction* at the binding *path*."""
    printer = PatternPrinter()
    return printer.match_text(instruction, printer.match_data((path,), reason)).lstrip()


def refusal_reason(refusal: object, parameter: str | None) -> str:
    """The reason a failed pattern match gives for *parameter*."""
    return PatternPrinter().refusal_reason(refusal, parameter)


services.registry.register(services.describe_refusal, describe_refusal)
services.registry.register(services.refusal_reason, refusal_reason)

