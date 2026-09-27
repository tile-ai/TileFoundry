"""Human-readable operation-declaration patterns and match refusals."""

from __future__ import annotations

from enum import Enum

from tilefoundry.ir.clause.layout import is_layout_wildcard
from tilefoundry.ir.core.param_def import collect_param_defs
from tilefoundry.ir.pattern.pattern import Pattern
from tilefoundry.ir.types import Broadcast, DType, Split
from tilefoundry.ir.types.dim import DimFloorDiv, DimMul, DimVar, is_dim_op_call

_UNNAMED = "_"
_ARRANGEMENT = "every arrangement"


class PatternPrinter:
    """Render pattern trees without a Python print context."""

    def written(self, pattern, name: str = _UNNAMED) -> str:
        """Write the value shape admitted by *pattern*."""
        if not isinstance(pattern, Pattern):
            return self._written_value(pattern)
        return self._dispatch("visit", pattern, name)

    def rules(self, pattern, name: str = _UNNAMED) -> tuple[str, ...]:
        """Write the ordered, de-duplicated conditions imposed by *pattern*."""
        if not isinstance(pattern, Pattern):
            return ()
        lines = tuple(dict.fromkeys(self._dispatch("rules", pattern, name)))
        formulas = frozenset(
            self._written_expression(formula) for formula in self._formulas(pattern)
        )
        return tuple(
            sorted(
                lines,
                key=lambda line: not any(
                    line == formula or line.startswith(f"{formula} where ")
                    for formula in formulas
                ),
            )
        )

    def described(self, pattern, name: str = _UNNAMED) -> str:
        """Write a value shape followed by its predicate lines, when present."""
        return self._described(self.written(pattern, name), self.rules(pattern, name))

    @staticmethod
    def _described(written: str, rules: tuple[str, ...]) -> str:
        if not rules:
            return written
        return "\n".join((written, "  predicates:", *(f"    {rule}" for rule in rules)))

    def alternatives(self, pattern, bindings=()) -> tuple:
        """Flatten declaration branches into ``(bindings, leaf)`` pairs."""
        if not isinstance(pattern, Pattern):
            return ((tuple(bindings), pattern),)
        return self._dispatch("alternatives", pattern, tuple(bindings))

    def refusal(self, refusal) -> str:
        """Render one structural match refusal at a consumer boundary."""
        if refusal is None:
            return "the pattern did not match"
        rules = self.rules(refusal.pattern)
        if rules:
            reason = f"could not satisfy {rules[0]}"
        else:
            reason = (
                f"{refusal.subject!r} does not match "
                f"{self.written(refusal.pattern)}"
            )
        bindings = self._written_bindings(refusal.bindings.items())
        return reason if not bindings else f"{reason} ({bindings})"

    def declaration(self, op_type) -> str:
        """Render all parameter and operand patterns declared by *op_type*."""
        title = getattr(op_type, "reference_name", "") or getattr(
            getattr(op_type, "_op_schema", None), "name", op_type.__name__
        )
        sections = []
        parameters = tuple(getattr(op_type, "parameters", ())) or collect_param_defs(op_type)
        parameters_by_name = {param.name: param for param in parameters}
        if parameters:
            sections.append(("parameters", tuple((param.name, param.pattern) for param in parameters)))
        roles = tuple(
            (name, getattr(op_type, name))
            for name in ("C", "A", "B")
            if isinstance(getattr(op_type, name, None), Pattern)
        )
        if roles:
            sections.append(("operands", roles))
            scope_pattern = getattr(op_type, "scope_pattern", None)
            if callable(scope_pattern):
                sections.append(("attributes", (("scope", scope_pattern()),)))
        if not roles and parameters:
            operands = tuple(
                (param.name, param.pattern)
                for param in parameters
                if param.kind == "input" and param.pattern is not None
            )
            attributes = tuple(
                (param.name, param.pattern)
                for param in parameters
                if param.kind == "attribute" and param.pattern is not None
            )
            sections = [
                (heading, items)
                for heading, items in (("operands", operands), ("attributes", attributes))
                if items
            ]

        lines = [title]
        for heading, items in sections:
            lines.append(f"  {heading}")
            width = max((len(name) for name, _ in items), default=0)
            for name, pattern in items:
                described = self._declared(pattern, name).splitlines()
                parameter = parameters_by_name.get(name)
                if parameter is not None and parameter.has_default:
                    described[0] += f" (default {self._written_value(parameter.default)})"
                prefix = f"    {name.ljust(width)}  "
                lines.append(prefix + described[0])
                lines.extend(" " * len(prefix) + line for line in described[1:])
        return "\n".join(lines)

    def _declared(self, pattern, name: str) -> str:
        if not any(cls.__name__ == "SwitchPattern" for cls in type(pattern).__mro__):
            return self.described(pattern, name)
        alternatives = tuple(
            (bindings, alternative, self.rules(alternative, name))
            for bindings, alternative in self.alternatives(pattern)
        )
        if not alternatives:
            return self.described(pattern, name)
        common = tuple(
            rule
            for rule in alternatives[0][2]
            if all(rule in rules for _, _, rules in alternatives[1:])
        )
        held = []
        for bindings, alternative, rules in alternatives:
            written = self.written(alternative, name)
            specific = tuple(rule for rule in rules if rule not in common)
            if written == name and len(specific) == 1:
                written, specific = specific[0], ()
            held.append(
                (
                    self._written_bindings(bindings),
                    self._described(written, specific),
                )
            )
        return self._described(self._aligned_alternatives(tuple(held)), common)

    def _dispatch(self, operation: str, pattern: Pattern, argument):
        for cls in type(pattern).__mro__:
            visitor = getattr(self, f"{operation}_{cls.__name__}", None)
            if visitor is not None:
                return visitor(pattern, argument)
        raise NotImplementedError(
            f"{type(pattern).__name__} has no PatternPrinter {operation} visitor"
        )

    @staticmethod
    def _named(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, Enum):
            return f"{type(value).__name__}.{value.name}"
        return getattr(value, "name", str(value)).lower()

    @staticmethod
    def _field_name(value) -> str | None:
        return None if value is None else getattr(value, "name", str(value)).lower()

    def _written_dim(self, value) -> str:
        if isinstance(value, DimVar):
            return value.name
        if is_dim_op_call(value):
            left, right = (self._written_dim(arg) for arg in value.args)
            if isinstance(value.target, DimFloorDiv):
                return f"{left}/{right}"
            if isinstance(value.target, DimMul):
                return f"{left}*{right}"
            return f"({left} {type(value.target).__name__} {right})"
        return str(getattr(value, "value", value))

    def _written_value(self, value) -> str:
        if isinstance(value, Broadcast):
            return "B()"
        if isinstance(value, DType):
            return value.name
        if isinstance(value, Split):
            return f"S({value.axis})"
        if isinstance(value, Enum):
            return f"{type(value).__name__}.{value.name}"
        if isinstance(value, tuple):
            return self._written_tuple(tuple(self._written_value(item) for item in value))
        if isinstance(value, list):
            return self._written_tuple(tuple(self._written_value(item) for item in value))
        if isinstance(value, DimVar) or is_dim_op_call(value):
            return self._written_dim(value)
        return _UNNAMED if is_layout_wildcard(value) else str(value)

    @staticmethod
    def _written_tuple(items: tuple[str, ...]) -> str:
        written = ", ".join(items)
        return f"({written},)" if len(items) == 1 else f"({written})"

    def _written_grouped(self, modes) -> str:
        if isinstance(modes, tuple):
            return self._written_tuple(tuple(self._written_grouped(mode) for mode in modes))
        return self.written(modes)

    def _written_bindings(self, bindings) -> str:
        return ", ".join(
            f"{name}={getattr(value, 'name', str(value))}" for name, value in bindings
        )

    def _written_alternatives(self, items, name: str = _UNNAMED) -> str:
        held = tuple(
            (self._written_bindings(bindings), self.written(alternative, name))
            for bindings, alternative in items
        )
        return self._aligned_alternatives(held)

    @staticmethod
    def _aligned_alternatives(held) -> str:
        width = max((len(label) for label, _ in held), default=0)
        lines = []
        for label, written in held:
            first, *rest = written.splitlines() or ("",)
            lines.append(first if not width else f"{label.ljust(width)}  {first}")
            lines.extend(
                line if not width else f"{' ' * (width + 2)}{line}" for line in rest
            )
        return "\n".join(lines)

    def _rules_of(self, values, name: str = _UNNAMED) -> tuple[str, ...]:
        return tuple(
            line
            for value in values
            if isinstance(value, Pattern)
            for line in self.rules(value, name)
        )

    def _alternative_rules(self, pattern, name: str = _UNNAMED) -> tuple[str, ...]:
        alternatives, common = self._partitioned_alternatives(pattern, name)
        if not alternatives:
            return ()
        specific = tuple(
            self._qualified_rule(rule, bindings)
            for bindings, _, rules in alternatives
            for rule in rules
            if rule not in common
        )
        return (*common, *specific)

    def _partitioned_alternatives(self, pattern, name: str = _UNNAMED) -> tuple:
        alternatives = tuple(
            (bindings, alternative, self.rules(alternative, name))
            for bindings, alternative in self.alternatives(pattern)
        )
        if not alternatives:
            return (), ()
        common = tuple(
            rule
            for rule in alternatives[0][2]
            if all(rule in rules for _, _, rules in alternatives[1:])
        )
        return alternatives, common

    def _qualified_rule(self, rule: str, bindings) -> str:
        label = self._written_bindings(bindings)
        if not label:
            return rule
        return f"{rule}, {label}" if " where " in rule else f"{rule} where {label}"

    @staticmethod
    def _ordered_predicates(predicates) -> tuple:
        formulas = tuple(p for p in predicates if type(p).__name__ == "Formula")
        hand_written = tuple(p for p in predicates if type(p).__name__ != "Formula")
        return formulas + hand_written

    def _formulas(self, value) -> tuple:
        if type(value).__name__ == "Formula":
            return (value,)
        if isinstance(value, Pattern):
            return tuple(
                formula
                for field_value in vars(value).values()
                for formula in self._formulas(field_value)
            )
        if isinstance(value, tuple):
            return tuple(
                formula for item in value for formula in self._formulas(item)
            )
        return ()

    def visit_WildcardPattern(self, pattern, name) -> str:
        return pattern.name or name

    def rules_WildcardPattern(self, pattern, name) -> tuple[str, ...]:
        return ()

    def visit_StarPattern(self, pattern, name) -> str:
        return f"*{self.written(pattern.pattern, name)}"

    def rules_StarPattern(self, pattern, name) -> tuple[str, ...]:
        return self.rules(pattern.pattern, name)

    def visit_OrPattern(self, pattern, name) -> str:
        if not any(isinstance(item, Pattern) for item in pattern.patterns):
            return name
        return self._written_alternatives(self.alternatives(pattern), name)

    def rules_OrPattern(self, pattern, name) -> tuple[str, ...]:
        if not any(isinstance(item, Pattern) for item in pattern.patterns):
            values = ", ".join(self._named(value) for value in pattern.patterns)
            return (f"{name} in {{{values}}}",)
        return self._alternative_rules(pattern, name)

    def visit_AndPattern(self, pattern, name) -> str:
        return name

    def rules_AndPattern(self, pattern, name) -> tuple[str, ...]:
        return self._rules_of(pattern.parts, name)

    def visit_SequencePattern(self, pattern, name) -> str:
        return self._written_tuple(tuple(self.written(item, name) for item in pattern.patterns))

    def rules_SequencePattern(self, pattern, name) -> tuple[str, ...]:
        return self._rules_of(pattern.patterns, name)

    def visit_RangePattern(self, pattern, name) -> str:
        return name

    def rules_RangePattern(self, pattern, name) -> tuple[str, ...]:
        if pattern.lo is None:
            return (f"{name} <= {pattern.hi}",)
        if pattern.hi is None:
            return (f"{pattern.lo} <= {name}",)
        return (f"{pattern.lo} <= {name} <= {pattern.hi}",)

    def visit_SwitchPattern(self, pattern, name) -> str:
        return self._written_alternatives(self.alternatives(pattern), name)

    def rules_SwitchPattern(self, pattern, name) -> tuple[str, ...]:
        return self._alternative_rules(pattern, name)

    def visit_LayoutPattern(self, pattern, name) -> str:
        if pattern.shape is None and pattern.strides is None:
            return "layout"
        shape = name if pattern.shape is None else self._written_grouped(tuple(pattern.shape))
        strides = (
            name if pattern.strides is None else self._written_grouped(tuple(pattern.strides))
        )
        return f"Layout({shape}, {strides})"

    def rules_LayoutPattern(self, pattern, name) -> tuple[str, ...]:
        return (
            *self._rules_of(pattern.positions()),
            *self._rules_of(self._ordered_predicates(pattern.predicates)),
        )

    def visit_SwizzlePattern(self, pattern, name) -> str:
        return (
            f"Swizzle({self.written(pattern.bits)}, {self.written(pattern.base)}, "
            f"{self.written(pattern.shift)})"
        )

    def rules_SwizzlePattern(self, pattern, name) -> tuple[str, ...]:
        return self._rules_of((pattern.bits, pattern.base, pattern.shift))

    def visit_ComposedLayoutPattern(self, pattern, name) -> str:
        return (
            f"ComposedLayout({self.written(pattern.inner)}, "
            f"{self.written(pattern.offset)}, {self.written(pattern.outer)})"
        )

    def rules_ComposedLayoutPattern(self, pattern, name) -> tuple[str, ...]:
        return (
            *self._rules_of((pattern.inner, pattern.offset, pattern.outer)),
            *self._rules_of(self._ordered_predicates(pattern.predicates)),
        )

    def visit_MeshPattern(self, pattern, name) -> str:
        return f"Mesh({pattern.topologies!r}, {self.written(pattern.layout)})"

    def rules_MeshPattern(self, pattern, name) -> tuple[str, ...]:
        return self.rules(pattern.layout)

    def visit_ScalarPattern(self, pattern, name) -> str:
        return "scalar"

    def rules_ScalarPattern(self, pattern, name) -> tuple[str, ...]:
        return ()

    def visit_TensorPattern(self, pattern, name) -> str:
        stated = []
        if pattern.shape is not None:
            stated.append(
                "shape=" + self._written_tuple(tuple(self.written(item) for item in pattern.shape))
            )
        if pattern.dtype is not None:
            stated.append(f"dtype={self._field_name(pattern.dtype)}")
        if pattern.storage is not None:
            stated.append(f"storage={self._field_name(pattern.storage)}")
        head = " ".join(stated) if stated else "any tensor"
        if pattern.layout is None:
            return f"{head}, in any arrangement"
        alternatives, common = self._partitioned_alternatives(pattern.layout)
        label = f"{len(alternatives)} arrangement{'' if len(alternatives) == 1 else 's'}:"
        items = tuple(
            (
                self._written_bindings(bindings),
                self._described(
                    self.written(alternative),
                    tuple(rule for rule in rules if rule not in common),
                ),
            )
            for bindings, alternative, rules in alternatives
        )
        written = self._aligned_alternatives(items).splitlines()
        return "\n".join((f"{head}, held in {label}", *(f"  {line}" for line in written)))

    def rules_TensorPattern(self, pattern, name) -> tuple[str, ...]:
        values = (
            pattern.dtype,
            pattern.storage,
            *(pattern.shape or ()),
        )
        layout_rules = (
            ()
            if pattern.layout is None
            else self._partitioned_alternatives(pattern.layout)[1]
        )
        return (
            *self._rules_of(values),
            *layout_rules,
            *self._rules_of(self._ordered_predicates(pattern.predicates)),
        )

    def visit_ShardLayoutPattern(self, pattern, name) -> str:
        attrs = (
            self._written_tuple(tuple(self._written_value(attr) for attr in pattern.attrs))
            if isinstance(pattern.attrs, tuple)
            else self.written(pattern.attrs)
        )
        return (
            f"ShardLayout({self.written(pattern.layout)}, {attrs}, "
            f"{self.written(pattern.mesh)})"
        )

    def rules_ShardLayoutPattern(self, pattern, name) -> tuple[str, ...]:
        return (
            *self._alternative_rules(pattern),
            *self._rules_of(self._ordered_predicates(pattern.predicates)),
        )

    def visit_AtomPattern(self, pattern, name) -> str:
        return "one of " + ", ".join(item.reference_name for item in pattern.declarations)

    def rules_AtomPattern(self, pattern, name) -> tuple[str, ...]:
        return ()

    def visit_FromAtom(self, pattern, name) -> str:
        return f"the {pattern.role} operand of its atom"

    def rules_FromAtom(self, pattern, name) -> tuple[str, ...]:
        return ()

    def visit_Predicate(self, pattern, name) -> str:
        return name

    def rules_Predicate(self, pattern, name) -> tuple[str, ...]:
        raise NotImplementedError(f"{type(pattern).__name__} has no PatternPrinter rules visitor")

    def visit_Formula(self, pattern, name) -> str:
        return name

    def rules_Formula(self, pattern, name) -> tuple[str, ...]:
        return (self._written_expression(pattern),)

    def visit_Forward(self, pattern, name) -> str:
        return name

    def rules_Forward(self, pattern, name) -> tuple[str, ...]:
        subject = "each top-level mode" if pattern.per_mode else _ARRANGEMENT
        return (f"{subject} has no backward step",)

    def visit_Injective(self, pattern, name) -> str:
        return name

    def rules_Injective(self, pattern, name) -> tuple[str, ...]:
        subject = "each top-level mode" if pattern.per_mode else _ARRANGEMENT
        return (f"{subject} reaches each of its own slots exactly once",)

    def visit_WholeVectors(self, pattern, name) -> str:
        return name

    def rules_WholeVectors(self, pattern, name) -> tuple[str, ...]:
        return (
            "every run: each tile axis's modes walked fastest first, contiguous ones "
            "joined; the run at step 1 and every other step a whole number of vectors",
            *self.rules(pattern.width),
        )

    def visit_PlainArrangement(self, pattern, name) -> str:
        return name

    def rules_PlainArrangement(self, pattern, name) -> tuple[str, ...]:
        return (
            "every plain arrangement is a static strided layout with no swizzle and zero "
            "offset, and, where it is sharded, one every participant holds whole",
        )

    def visit_BoxDims(self, pattern, name) -> str:
        return name

    def rules_BoxDims(self, pattern, name) -> tuple[str, ...]:
        reading = (
            "every box: each tile axis's modes, contiguous ones joined up to "
            f"{pattern.limit} elements, one dim each, in increasing step"
        )
        return (reading, *self._rules_of(pattern.dims))

    def visit_TensorMap(self, pattern, name) -> str:
        return name

    def rules_TensorMap(self, pattern, name) -> tuple[str, ...]:
        return (
            "every tensormap: one dim per mode of the tile, the mode at step 1 first",
            *self._rules_of(pattern.steps),
        )

    def _written_expression(self, value) -> str:
        name = type(value).__name__
        if name == "Formula":
            if value.op == "not":
                return f"~({self._written_expression(value.args[0])})"
            if value.op == "forall":
                term, token, body = value.args
                return (
                    f"all({self._written_expression(body)} for x{token} "
                    f"in {self._written_expression(term)})"
                )
            if value.op == "in":
                values = ", ".join(self._written_value(item) for item in value.args[1])
                return f"{self._written_expression(value.args[0])} in {{{values}}}"
            symbol = {
                "eq": "==",
                "ne": "!=",
                "lt": "<",
                "le": "<=",
                "gt": ">",
                "ge": ">=",
                "and": "&",
                "or": "|",
            }[value.op]
            return (
                f"{self._written_expression(value.args[0])} {symbol} "
                f"{self._written_expression(value.args[1])}"
            )
        if name == "Term":
            if value.op == "constant":
                return repr(value.args[0])
            if value.op in {"variable", "bits"}:
                return (
                    value.args[0]
                    if value.op == "variable"
                    else f"Bits({value.args[0]!r})"
                )
            if value.op == "bound":
                return f"x{value.args[0]}"
            if value.op == "element":
                return f"{value.args[0]!r}[{self._written_expression(value.args[1])}]"
            if value.op in {"sum", "count"}:
                return f"{value.op.title()}({self._written_expression(value.args[0])})"
            symbol = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "floordiv": "//",
                "mod": "%",
            }[value.op]
            return (
                f"{self._written_expression(value.args[0])} {symbol} "
                f"{self._written_expression(value.args[1])}"
            )
        return repr(value)

    def alternatives_Pattern(self, pattern, bindings) -> tuple:
        return ((bindings, pattern),)

    def alternatives_OrPattern(self, pattern, bindings) -> tuple:
        if not any(isinstance(item, Pattern) for item in pattern.patterns):
            return ((bindings, pattern),)
        return tuple(
            held
            for alternative in pattern.patterns
            for held in self.alternatives(alternative, bindings)
        )

    def alternatives_SwitchPattern(self, pattern, bindings) -> tuple:
        return tuple(
            held
            for value, branch in pattern.branches
            for held in self.alternatives(branch, (*bindings, (pattern.param, value)))
        )

    def alternatives_ShardLayoutPattern(self, pattern, bindings) -> tuple:
        return self.alternatives(pattern.layout, bindings)


__all__ = ["PatternPrinter"]
