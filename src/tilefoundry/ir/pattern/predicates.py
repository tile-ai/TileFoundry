"""Named computed predicates for operation-declaration layout patterns."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import count
from typing import Callable, Iterable

from tilefoundry.ir.types import Broadcast, ComposedLayout, Layout, ShardLayout, Swizzle
from tilefoundry.ir.types.int_tuple import flatten
from tilefoundry.ir.types.layout_algebra import coalesce, is_inverse_projectable

from .match import (
    ARRANGEMENT,
    UNNAMED_PLACE,
    matched,
    relations_of,
    written_place,
    written_tuple,
)
from .pattern import Pattern, SequencePattern, WildcardPattern


class Unknown:
    def __bool__(self) -> bool:
        return True


UNKNOWN = Unknown()
_MISSING = object()
_TOKENS = count()


def _same(left, right) -> bool:
    """Compare expression structure without invoking overloaded equality."""
    if type(left) is not type(right):
        return False
    if is_dataclass(left):
        return all(
            _same(getattr(left, item.name), getattr(right, item.name)) for item in fields(left)
        )
    if isinstance(left, tuple):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _as_term(value) -> Term:
    if isinstance(value, WildcardPattern):
        return variable(value.name)
    return value if isinstance(value, Term) else Term("constant", (value,))


@dataclass(frozen=True, eq=False)
class Term:
    """An integer expression over named pattern bindings."""

    op: str
    args: tuple

    def _binary(self, op: str, other) -> Term:
        return Term(op, (self, _as_term(other)))

    def _reflected(self, op: str, other) -> Term:
        return Term(op, (_as_term(other), self))

    def __add__(self, other):
        return self._binary("add", other)

    def __radd__(self, other):
        return self._reflected("add", other)

    def __sub__(self, other):
        return self._binary("sub", other)

    def __rsub__(self, other):
        return self._reflected("sub", other)

    def __mul__(self, other):
        return self._binary("mul", other)

    def __rmul__(self, other):
        return self._reflected("mul", other)

    def __floordiv__(self, other):
        return self._binary("floordiv", other)

    def __rfloordiv__(self, other):
        return self._reflected("floordiv", other)

    def __mod__(self, other):
        return self._binary("mod", other)

    def __rmod__(self, other):
        return self._reflected("mod", other)

    def _compare(self, op: str, other) -> Formula:
        return Formula(op, (self, _as_term(other)))

    def __eq__(self, other):
        return self._compare("eq", other)

    def __ne__(self, other):
        return self._compare("ne", other)

    def __lt__(self, other):
        return self._compare("lt", other)

    def __le__(self, other):
        return self._compare("le", other)

    def __gt__(self, other):
        return self._compare("gt", other)

    def __ge__(self, other):
        return self._compare("ge", other)


def variable(name: str | None) -> Term:
    if not name:
        raise TypeError("an unnamed WildcardPattern cannot be used in a formula")
    return Term("variable", (name,))


@dataclass(frozen=True)
class Predicate(Pattern):
    """A computed condition over a subject and the matcher's bindings."""

    @staticmethod
    def arrangement(subject) -> Layout | None:
        """Read the static strided layout beneath shard and composition wrappers."""
        if isinstance(subject, ShardLayout):
            if not all(isinstance(attr, Broadcast) for attr in subject.attrs):
                return None
            subject = subject.layout
        if isinstance(subject, ComposedLayout):
            if subject.inner is not None and not isinstance(subject.inner, Swizzle):
                return None
            subject = subject.outer
        if not isinstance(subject, Layout) or subject.strides is None:
            return None
        extents = tuple(flatten(subject.shape))
        strides = tuple(flatten(subject.strides))
        if any(type(number) is not int for number in (*extents, *strides)):
            return None
        if any(extent <= 0 for extent in extents):
            return None
        return subject

    def holds(self, subject, bindings: dict) -> bool | None:
        """Return true, false, or None while required bindings are unknown."""
        raise NotImplementedError

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        raise NotImplementedError

    def relations(self) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class Formula(Predicate):
    """A Boolean expression evaluated against pattern bindings."""

    op: str
    args: tuple

    def __eq__(self, other) -> bool:
        return isinstance(other, Formula) and self.op == other.op and _same(self.args, other.args)

    def __ne__(self, other) -> bool:
        return not self == other

    def __and__(self, other):
        if not isinstance(other, Formula):
            return NotImplemented
        return Formula("and", (self, other))

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        if not isinstance(other, Formula):
            return NotImplemented
        return Formula("or", (self, other))

    def __ror__(self, other):
        return self.__or__(other)

    def __invert__(self):
        return Formula("not", (self,))

    def __bool__(self) -> bool:
        if self.op == "eq":
            return _same(*self.args)
        if self.op == "ne":
            return not _same(*self.args)
        raise TypeError(
            "Formula has no Python truth value; combine conditions with & / |, "
            "and use In(term, values) instead of Python 'in' or chained comparisons"
        )

    def holds(self, subject, bindings: dict) -> bool | None:
        return evaluate(self, bindings)

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return _written(self)

    def relations(self) -> tuple[str, ...]:
        return (self.describe(),)


def Bits(name: str) -> Term:
    """Read the bit width of the dtype bound to *name*."""
    return Term("bits", (name,))


@dataclass(frozen=True)
class Table:
    values: tuple

    def __init__(self, values: Iterable):
        object.__setattr__(self, "values", tuple(values))

    def __getitem__(self, index) -> Term:
        return Term("element", (self.values, _as_term(index)))


def In(term, values: Iterable) -> Formula:
    return Formula("in", (_as_term(term), tuple(values)))


def ForAll(term, build: Callable[[Term], Formula]) -> Formula:
    token = next(_TOKENS)
    body = build(Term("bound", (token,)))
    if not isinstance(body, Formula):
        raise TypeError("ForAll body must produce a Formula")
    return Formula("forall", (_as_term(term), token, body))


def Sum(term) -> Term:
    return Term("sum", (_as_term(term),))


def Count(term) -> Term:
    return Term("count", (_as_term(term),))


def _value(term: Term, env: dict, local: dict | None = None):
    local = {} if local is None else local
    if term.op == "constant":
        return term.args[0]
    if term.op == "variable":
        return env.get(term.args[0], _MISSING)
    if term.op == "bound":
        return local.get(term.args[0], _MISSING)
    if term.op == "bits":
        value = env.get(term.args[0], _MISSING)
        return _MISSING if value is _MISSING else getattr(value, "bit_width", _MISSING)
    if term.op == "element":
        values, index = term.args
        index = _value(index, env, local)
        return _MISSING if index is _MISSING else values[index]
    if term.op in {"sum", "count"}:
        value = _value(term.args[0], env, local)
        if value is _MISSING:
            return _MISSING
        if not isinstance(value, tuple):
            raise TypeError(f"{term.op.title()} requires a tuple binding")
        return sum(value) if term.op == "sum" else len(value)
    left = _value(term.args[0], env, local)
    right = _value(term.args[1], env, local)
    if left is _MISSING or right is _MISSING:
        return _MISSING
    return {
        "add": lambda: left + right,
        "sub": lambda: left - right,
        "mul": lambda: left * right,
        "floordiv": lambda: left // right,
        "mod": lambda: left % right,
    }[term.op]()


def _truth(formula: Formula, env: dict, local: dict | None = None) -> bool | None:
    local = {} if local is None else local
    if formula.op in {"and", "or"}:
        left = _truth(formula.args[0], env, local)
        right = _truth(formula.args[1], env, local)
        if formula.op == "and":
            return False if False in (left, right) else None if None in (left, right) else True
        return True if True in (left, right) else None if None in (left, right) else False
    if formula.op == "not":
        value = _truth(formula.args[0], env, local)
        return None if value is None else not value
    if formula.op == "forall":
        term, token, body = formula.args
        values = _value(term, env, local)
        if values is _MISSING:
            raise TypeError(
                "ForAll requires a tuple whose length is known after structural matching"
            )
        if not isinstance(values, tuple):
            raise TypeError("ForAll requires a tuple binding")
        found = tuple(_truth(body, env, {**local, token: value}) for value in values)
        return False if False in found else None if None in found else True
    if formula.op == "in":
        value = _value(formula.args[0], env, local)
        return None if value is _MISSING else value in formula.args[1]
    left = _value(formula.args[0], env, local)
    right = _value(formula.args[1], env, local)
    if left is _MISSING or right is _MISSING:
        return None
    return {
        "eq": lambda: left == right,
        "ne": lambda: left != right,
        "lt": lambda: left < right,
        "le": lambda: left <= right,
        "gt": lambda: left > right,
        "ge": lambda: left >= right,
    }[formula.op]()


def evaluate(formula: Formula, env) -> bool | None:
    """Evaluate *formula*, returning None when a named value is unbound."""
    return _truth(formula, dict(env or {}))


class _CpBuilder:
    """Compile the unresolved portion of formulas into one CP-SAT model."""

    LIMIT = 2**31 - 1

    def __init__(self, model, env):
        self.model = model
        self.env = dict(env)
        self.variables = {}
        self.serial = count()

    def fresh(self, prefix: str):
        return self.model.new_int_var(-self.LIMIT, self.LIMIT, f"_{prefix}{next(self.serial)}")

    def term(self, term: Term, local=None):
        local = {} if local is None else local
        known = _value(term, self.env, local)
        if known is not _MISSING:
            if type(known) is not int:
                raise TypeError(f"CP-SAT arithmetic requires integers, got {known!r}")
            return known
        if term.op == "variable":
            name = term.args[0]
            if name not in self.variables:
                self.variables[name] = self.model.new_int_var(-self.LIMIT, self.LIMIT, name)
            return self.variables[name]
        if term.op == "bits":
            raise TypeError(f"Bits({term.args[0]!r}) requires its dtype binding")
        if term.op == "bound":
            raise TypeError("ForAll requires its tuple binding before CP-SAT solving")
        if term.op == "element":
            values, index = term.args
            target = self.fresh("element")
            self.model.add_element(self.term(index, local), values, target)
            return target
        if term.op in {"sum", "count"}:
            raise TypeError(f"{term.op.title()} requires its tuple binding before CP-SAT solving")
        left = self.term(term.args[0], local)
        right = self.term(term.args[1], local)
        if term.op == "add":
            return left + right
        if term.op == "sub":
            return left - right
        target = self.fresh(term.op)
        if term.op == "mul":
            self.model.add_multiplication_equality(target, (left, right))
        elif term.op == "floordiv":
            self.model.add_division_equality(target, left, right)
        elif term.op == "mod":
            self.model.add_modulo_equality(target, left, right)
        else:
            raise ValueError(f"unknown Term operation {term.op!r}")
        return target

    def formula(self, formula: Formula, local=None):
        local = {} if local is None else local
        known = _truth(formula, self.env, local)
        if known is not None:
            return self.model.new_constant(known)
        if formula.op == "not":
            return self.formula(formula.args[0], local).Not()
        if formula.op in {"and", "or"}:
            parts = tuple(self.formula(part, local) for part in formula.args)
            result = self.model.new_bool_var(f"_{formula.op}{next(self.serial)}")
            if formula.op == "and":
                self.model.add_bool_and(parts).only_enforce_if(result)
                self.model.add_bool_or(tuple(part.Not() for part in parts)).only_enforce_if(
                    result.Not()
                )
            else:
                self.model.add_bool_or(parts).only_enforce_if(result)
                self.model.add_bool_and(tuple(part.Not() for part in parts)).only_enforce_if(
                    result.Not()
                )
            return result
        if formula.op == "forall":
            term, token, body = formula.args
            values = _value(term, self.env, local)
            if values is _MISSING:
                raise TypeError(
                    "ForAll requires a tuple whose length is known after structural matching"
                )
            if not isinstance(values, tuple):
                raise TypeError("ForAll requires a tuple binding")
            parts = tuple(self.formula(body, {**local, token: value}) for value in values)
            result = self.model.new_bool_var(f"_forall{next(self.serial)}")
            self.model.add_bool_and(parts).only_enforce_if(result)
            self.model.add_bool_or(tuple(part.Not() for part in parts)).only_enforce_if(
                result.Not()
            )
            return result
        result = self.model.new_bool_var(f"_{formula.op}{next(self.serial)}")
        if formula.op == "in":
            value = self.term(formula.args[0], local)
            rows = tuple((item,) for item in formula.args[1])
            self.model.add_allowed_assignments((value,), rows).only_enforce_if(result)
            self.model.add_forbidden_assignments((value,), rows).only_enforce_if(result.Not())
            return result
        left = self.term(formula.args[0], local)
        right = self.term(formula.args[1], local)
        relation = {
            "eq": lambda: left == right,
            "ne": lambda: left != right,
            "lt": lambda: left < right,
            "le": lambda: left <= right,
            "gt": lambda: left > right,
            "ge": lambda: left >= right,
        }[formula.op]
        opposite = {
            "eq": lambda: left != right,
            "ne": lambda: left == right,
            "lt": lambda: left >= right,
            "le": lambda: left > right,
            "gt": lambda: left <= right,
            "ge": lambda: left < right,
        }[formula.op]
        self.model.add(relation()).only_enforce_if(result)
        self.model.add(opposite()).only_enforce_if(result.Not())
        return result


def solve(formulas, env) -> dict | None | Unknown:
    """Solve unresolved formulas, or return UNKNOWN without treating it as failure."""
    formulas = tuple(formulas)
    held = dict(env or {})
    evaluated = tuple(evaluate(formula, held) for formula in formulas)
    if False in evaluated:
        return None
    unresolved = tuple(
        formula for formula, value in zip(formulas, evaluated) if value is None
    )
    if not unresolved:
        return held

    from ortools.sat.python import cp_model  # noqa: PLC0415 - only build CP-SAT on demand

    model = cp_model.CpModel()
    builder = _CpBuilder(model, held)
    for formula in unresolved:
        model.add(builder.formula(formula) == 1)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    if status == cp_model.UNKNOWN:
        return UNKNOWN
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return None
    held.update({name: solver.value(value) for name, value in builder.variables.items()})
    return held


def failing(formulas, env) -> Formula | None:
    """Return the first false conjunct, if evaluation can identify one."""
    for formula in formulas:
        if formula.op == "and":
            found = failing(formula.args, env)
            if found is not None:
                return found
        elif evaluate(formula, env) is False:
            return formula
    return None


def _written(value) -> str:
    if isinstance(value, Formula):
        if value.op == "not":
            return f"~({_written(value.args[0])})"
        if value.op == "forall":
            term, token, body = value.args
            return f"all({_written(body)} for x{token} in {_written(term)})"
        if value.op == "in":
            return f"{_written(value.args[0])} in {value.args[1]!r}"
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
        return f"{_written(value.args[0])} {symbol} {_written(value.args[1])}"
    if isinstance(value, Term):
        if value.op == "constant":
            return repr(value.args[0])
        if value.op in {"variable", "bits"}:
            return value.args[0] if value.op == "variable" else f"Bits({value.args[0]!r})"
        if value.op == "bound":
            return f"x{value.args[0]}"
        if value.op == "element":
            return f"{value.args[0]!r}[{_written(value.args[1])}]"
        if value.op in {"sum", "count"}:
            return f"{value.op.title()}({_written(value.args[0])})"
        symbol = {"add": "+", "sub": "-", "mul": "*", "floordiv": "//", "mod": "%"}[value.op]
        return f"{_written(value.args[0])} {symbol} {_written(value.args[1])}"
    return repr(value)


VECTOR_READING = (
    "every run: each tile axis's modes walked fastest first, contiguous ones joined; "
    "the run at step 1 and every other step a whole number of vectors"
)
TENSORMAP_READING = "every tensormap: one dim per mode of the tile, the mode at step 1 first"


def _arrangements(layout: Layout, per_mode: bool) -> tuple[Layout, ...]:
    if per_mode:
        return tuple(
            Layout(tuple(flatten(shape)), tuple(flatten(steps)))
            for shape, steps in zip(layout.shape, layout.strides)
        )
    return (Layout(tuple(flatten(layout.shape)), tuple(flatten(layout.strides))),)


@dataclass(frozen=True)
class Forward(Predicate):
    """Require nonnegative steps, across the whole layout or per top-level mode."""

    per_mode: bool = False

    def holds(self, subject, bindings: dict) -> bool:
        arrangement = self.arrangement(subject)
        if arrangement is None:
            return False
        return all(
            all(step >= 0 for step in flatten(part.strides))
            for part in _arrangements(arrangement, self.per_mode)
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        subject = "each top-level mode" if self.per_mode else ARRANGEMENT
        return f"{subject} with no backward step"

    def relations(self) -> tuple[str, ...]:
        subject = "each top-level mode" if self.per_mode else ARRANGEMENT
        return (f"{subject} has no backward step",)


@dataclass(frozen=True)
class Injective(Predicate):
    """Require every slot to be reached once, across the layout or per mode."""

    per_mode: bool = False

    def holds(self, subject, bindings: dict) -> bool:
        arrangement = self.arrangement(subject)
        if arrangement is None:
            return False
        return all(
            is_inverse_projectable(part) for part in _arrangements(arrangement, self.per_mode)
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        subject = "each top-level mode" if self.per_mode else ARRANGEMENT
        return f"{subject} reaching each of its own slots exactly once"

    def relations(self) -> tuple[str, ...]:
        subject = "each top-level mode" if self.per_mode else ARRANGEMENT
        return (f"{subject} reaches each of its own slots exactly once",)


def _reverse_group(group):
    if isinstance(group, tuple):
        return tuple(_reverse_group(mode) for mode in reversed(group))
    return group


def _row_major_groups_for_cute(layout: Layout) -> Layout:
    """Reverse modes within each tile axis for CuTe's mode-0-fast algebra."""
    return Layout(
        shape=tuple(_reverse_group(group) for group in layout.shape),
        strides=tuple(_reverse_group(group) for group in layout.strides),
    )


def _vector_widths(layout, element_bits: int, widths: tuple[int, ...]) -> tuple[int, ...]:
    """Every requested byte width that divides every run in an arrangement."""
    if not widths:
        return ()
    widest = widths[-1]
    stated = layout.layout if isinstance(layout, ShardLayout) else layout
    inner = getattr(stated, "inner", None)
    if inner is not None and hasattr(inner, "base"):
        widest = min(widest, 1 << inner.base)
    held = Predicate.arrangement(layout)
    if held is None:
        return ()
    grouped = coalesce(
        _row_major_groups_for_cute(held),
        (0,) * len(held.shape),
    )
    runs = tuple(
        sorted(
            zip(flatten(grouped.shape), flatten(grouped.strides)),
            key=lambda run: run[1],
        )
    )
    unit = [extent for extent, step in runs if step == 1]
    if len(unit) != 1:
        return ()
    counted = (unit[0], *(step for _, step in runs if step != 1))
    return tuple(
        width
        for width in widths
        if width <= widest and all(value * element_bits % (width * 8) == 0 for value in counted)
    )


@dataclass(frozen=True)
class WholeVectors(Predicate):
    """Require whole vectors at one of the requested byte widths."""

    width: WildcardPattern
    dtype: str
    widths: tuple[int, ...]

    def available_widths(self, subject, captures) -> tuple[int, ...]:
        bits = getattr(dict(captures or {}).get(self.dtype), "bit_width", None)
        layout = subject.layout if isinstance(subject, ShardLayout) else subject
        return () if type(bits) is not int else _vector_widths(layout, bits, self.widths)

    def holds(self, subject, bindings: dict) -> bool:
        widths = self.available_widths(subject, bindings)
        if not widths:
            return False
        if self.width.name in bindings:
            return bindings[self.width.name] in widths
        return matched(self.width, widths[-1], bindings) is not None

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return f"vectors of {self.width.name} bytes"

    def relations(self) -> tuple[str, ...]:
        return (VECTOR_READING, *relations_of((self.width,)))


@dataclass(frozen=True)
class PlainArrangement(Predicate):
    """Require an arrangement with no transform and no nonzero offset."""

    @staticmethod
    def _stated(subject):
        return subject.layout if isinstance(subject, ShardLayout) else subject

    def holds(self, subject, bindings: dict) -> bool:
        stated = self._stated(subject)
        if isinstance(stated, ComposedLayout) and (stated.inner is not None or stated.offset != 0):
            return False
        return self.arrangement(subject) is not None

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return "a plain arrangement with no transform or offset"

    def relations(self) -> tuple[str, ...]:
        return ("every plain arrangement has no transform or nonzero offset",)


@dataclass(frozen=True)
class Run:
    """One contiguous run of modes from one logical tile axis."""

    extent: int
    step: int
    axis: int
    mode: int


def box_runs(
    layout: Layout,
    element_bits: int,
    span: int | None,
    limit: int,
) -> tuple[Run, ...]:
    """Read box runs by tile axis, ordered by increasing step."""
    runs: list[Run] = []
    for axis, (extents, steps) in enumerate(zip(layout.shape, layout.strides)):
        modes = tuple(enumerate(zip(flatten(extents), flatten(steps))))
        for mode, (extent, step) in reversed(modes):
            if extent == 1:
                continue
            last = runs[-1] if runs and runs[-1].axis == axis else None
            joined = None if last is None else last.extent * extent
            if (
                last is not None
                and step == last.step * last.extent
                and joined <= limit
                and not (span is not None and last.step == 1 and joined * element_bits > span * 8)
            ):
                runs[-1] = replace(last, extent=joined)
            else:
                runs.append(Run(extent, step, axis, mode))
    return tuple(sorted(runs, key=lambda run: run.step))


@dataclass(frozen=True)
class BoxDims(Predicate):
    """Require one box dimension per contiguous run of tile-axis modes."""

    dims: tuple
    dtype: str
    limit: int
    span: int | None = None

    def reading(self, subject, captures) -> tuple[tuple | None, str | None]:
        layout = self.arrangement(subject)
        if layout is None:
            return None, f"{subject!r} is no static strided arrangement"
        width = getattr(captures.get(self.dtype), "bit_width", None)
        if type(width) is not int:
            return None, f"the element it arranges is not bound as {self.dtype}"
        runs = box_runs(layout, width, self.span, self.limit)
        if not runs:
            return None, "it holds one element, which is no box"
        if len(runs) > len(self.dims):
            return None, (
                f"it is {len(runs)} runs of modes, and a box has at most {len(self.dims)} dims"
            )
        extents = tuple(run.extent for run in runs)
        extents += (1,) * (len(self.dims) - len(extents))
        if runs[0].step != 1:
            return extents, (
                f"its smallest step is {runs[0].step}, and a box lays its dim 0 at step 1"
            )
        if self.span is not None and (self.span * 8) % width:
            return extents, f"a {self.span}-byte row is no whole number of {self.dtype}"
        expected = runs[0].extent if self.span is None else self.span * 8 // width
        for index, run in enumerate(runs[1:], 1):
            if run.step != expected:
                return extents, (
                    f"its dim {index} steps {run.step} where a box lays it at "
                    f"{expected} ({self.laid()})"
                )
            expected *= run.extent
        return extents, None

    def laid(self) -> str:
        rows = "" if self.span is None else f", rows {self.span} B apart"
        return f"dim 0 fastest{rows}"

    def holds(self, subject, bindings: dict) -> bool:
        extents, unlaid = self.reading(subject, bindings)
        return (
            extents is not None
            and unlaid is None
            and matched(SequencePattern(*self.dims), extents, bindings) is not None
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        dims = written_tuple(tuple(written_place(place) for place in self.dims))
        return f"box {dims}, {self.laid()}"

    def relations(self) -> tuple[str, ...]:
        reading = (
            "every box: each tile axis's modes, contiguous ones joined up to "
            f"{self.limit} elements, one dim each, in increasing step"
        )
        return (reading, *relations_of(self.dims))


@dataclass(frozen=True)
class TensorMap(Predicate):
    """Require one tensor-map dimension per nontrivial tile mode."""

    steps: tuple
    dtype: str

    def reading(self, subject) -> tuple[tuple | None, str | None]:
        layout = self.arrangement(subject)
        if layout is None:
            return None, f"{subject!r} is no static strided tensor a tensormap describes"
        modes = [
            (extent, step)
            for extents, steps in zip(layout.shape, layout.strides)
            for extent, step in zip(flatten(extents), flatten(steps))
            if extent > 1
        ]
        unit = [mode for mode in modes if mode[1] == 1]
        if len(unit) != 1:
            return None, (
                f"{len(unit)} of its modes step 1, and a tensormap's dim 0 is its one "
                "contiguous mode"
            )
        if len(modes) > len(self.steps) + 1:
            return None, (
                f"it is {len(modes)} modes, and a tensormap has at most {len(self.steps) + 1} dims"
            )
        others = tuple(step for _, step in modes if step != 1)
        return others + (0,) * (len(self.steps) - len(others)), None

    def holds(self, subject, bindings: dict) -> bool:
        steps, _ = self.reading(subject)
        return (
            steps is not None and matched(SequencePattern(*self.steps), steps, bindings) is not None
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        steps = written_tuple(("1", *(written_place(place) for place in self.steps)))
        return f"tensormap at {steps}"

    def relations(self) -> tuple[str, ...]:
        return (TENSORMAP_READING, *relations_of(self.steps))


__all__ = [
    "Bits",
    "BoxDims",
    "Count",
    "ForAll",
    "Formula",
    "Forward",
    "In",
    "Injective",
    "PlainArrangement",
    "Predicate",
    "Sum",
    "Table",
    "Term",
    "TensorMap",
    "Unknown",
    "WholeVectors",
    "evaluate",
    "failing",
    "solve",
]
