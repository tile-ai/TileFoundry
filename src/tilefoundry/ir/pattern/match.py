"""Shared matching, resolution, and rendering helpers for IR patterns."""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.clause.layout import is_layout_wildcard
from tilefoundry.ir.types import (
    ComposedLayout,
    Layout,
    Mesh,
    ShardLayout,
    Swizzle,
    TensorType,
    make_mesh,
)
from tilefoundry.ir.types.dim import DimFloorDiv, DimMul, DimVar, is_dim_op_call
from tilefoundry.ir.types.int_tuple import congruent
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.mesh import separate
from tilefoundry.ir.types.substitute import DimSubstitutionError, substitute_shape_dim

UNNAMED_PLACE = "_"
ARRANGEMENT = "every arrangement"
OPAQUE = object()


class Unknown:
    """A truthy result when the solver cannot prove success or failure."""

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True)
class Match:
    """The bindings produced by a successful match."""

    captures: dict = field(default_factory=dict)


class _Bindings(dict):
    """A binding dict that keeps nested ``matched`` calls on its owner."""

    matcher: PatternMatcher


class PatternMatcher:
    """Match one pattern tree while retaining bindings and match evidence."""

    def __init__(self, bindings=None):
        self.bindings: _Bindings = _Bindings(bindings or {})
        self.bindings.matcher = self
        self.memo: dict[int, object] = {}
        self.pending: list[object] = []
        self._refusal: str | None = None
        self._depth = 0

    def snapshot(self):
        """Save all transactional matcher state."""
        return (
            dict(self.bindings),
            dict(self.memo),
            list(self.pending),
            self._refusal,
        )

    def restore(self, saved) -> None:
        """Restore state saved by :meth:`snapshot`."""
        bindings, memo, pending, refusal = saved
        self.bindings.clear()
        self.bindings.update(bindings)
        self.memo.clear()
        self.memo.update(memo)
        self.pending[:] = pending
        self._refusal = refusal

    def match(self, pattern, subject) -> bool:
        """Match *pattern* against *subject* transactionally."""
        root = self._depth == 0
        if root:
            self._refusal = None
        saved = self.snapshot()
        self._depth += 1
        try:
            found = self._match(pattern, subject)
        except Exception:
            self.restore(saved)
            raise
        finally:
            self._depth -= 1
        if found:
            if isinstance(pattern, _pattern_type()):
                self.memo.setdefault(id(pattern), subject)
            return True
        refusal = self._refusal or f"{subject!r} does not match {pattern!r}"
        self.restore(saved)
        self._refusal = refusal
        return False

    def solve(self) -> bool | Unknown:
        """Finish formulas that still had unbound names during structural matching."""
        if not self.pending:
            return True
        from .predicates import UNKNOWN, failing, solve  # noqa: PLC0415 - protocol cycle

        formulas = tuple(self.pending)
        solved = solve(formulas, self.bindings)
        predicate = failing(formulas, self.bindings)
        if predicate is None:
            predicate = formulas[0]
        bindings = written_bindings(self.bindings.items())
        if solved is UNKNOWN:
            reason = f"CP-SAT returned unknown for {written_place(predicate)}"
            self._refusal = reason if not bindings else f"{reason} ({bindings})"
            return UNKNOWN
        if solved is not None:
            self.bindings.update(solved)
            self.pending.clear()
            return True
        reason = f"could not satisfy {written_place(predicate)}"
        self._refusal = reason if not bindings else f"{reason} ({bindings})"
        return False

    def refusal(self) -> str:
        """Explain the first failed node in the most recent match."""
        return self._refusal or "the pattern did not match"

    def _fail(self, pattern, subject, detail: str | None = None) -> bool:
        if self._refusal is None:
            reason = detail or f"{subject!r} does not match {written_place(pattern)}"
            bindings = written_bindings(self.bindings.items())
            self._refusal = reason if not bindings else f"{reason} ({bindings})"
        return False

    def _match(self, pattern, subject) -> bool:
        if pattern is None:
            return True
        if isinstance(pattern, DimVar):
            if pattern.name in self.bindings:
                return self.bindings[pattern.name] == subject or self._fail(pattern, subject)
            if type(subject) is not int or not pattern.lo <= subject < pattern.hi:
                return self._fail(pattern, subject)
            self.bindings[pattern.name] = subject
            return True
        if is_dim_op_call(pattern):
            found = evaluated(pattern, self.bindings)
            return (found is not None and found == subject) or self._fail(pattern, subject)
        if not isinstance(pattern, _pattern_type()):
            return pattern == subject or self._fail(pattern, subject)
        for cls in type(pattern).__mro__:
            visitor = getattr(self, f"visit_{cls.__name__}", None)
            if visitor is not None:
                return visitor(pattern, subject)
        return self._fail(pattern, subject)

    def visit_Pattern(self, pattern, subject) -> bool:
        raise NotImplementedError(f"{type(pattern).__name__} has no visitor")

    def visit_WildcardPattern(self, pattern, subject) -> bool:
        if pattern.name:
            if pattern.name in self.bindings:
                return self.bindings[pattern.name] == subject or self._fail(pattern, subject)
            self.bindings[pattern.name] = subject
        return True

    def _captured_names(self, value) -> tuple[str, ...]:
        from .pattern import Pattern, WildcardPattern  # noqa: PLC0415 - pattern protocol cycle

        if isinstance(value, WildcardPattern):
            name = getattr(value, "name", None)
            return () if not name else (name,)
        if isinstance(value, Pattern):
            return tuple(
                name for field in vars(value).values() for name in self._captured_names(field)
            )
        if isinstance(value, tuple):
            return tuple(name for item in value for name in self._captured_names(item))
        return ()

    def _match_repeated(self, owner, patterns, subjects) -> bool:
        names = tuple(dict.fromkeys(self._captured_names(patterns)))
        prior = {
            name: self.bindings.pop(name) for name in names if name in self.bindings
        }
        captured = {name: [] for name in names}
        for values in subjects:
            if not all(
                self.match(pattern, value) for pattern, value in zip(patterns, values)
            ):
                return False
            for name in names:
                if name not in self.bindings:
                    return self._fail(
                        owner,
                        subjects,
                        f"StarPattern did not bind {name!r} in every matched mode",
                    )
                captured[name].append(self.bindings.pop(name))
        for name, values in captured.items():
            held = tuple(values)
            if name in prior and prior[name] != held:
                return self._fail(owner, subjects)
            self.bindings[name] = prior.get(name, held)
        return True

    def visit_StarPattern(self, pattern, subject) -> bool:
        if not isinstance(subject, (tuple, list)):
            return self._fail(pattern, subject)
        return self._match_repeated(
            pattern,
            (pattern.pattern,),
            tuple((value,) for value in subject),
        )

    def visit_OrPattern(self, pattern, subject) -> bool:
        saved = self.snapshot()
        first_refusal = None
        for alternative in pattern.patterns:
            self.restore(saved)
            if self.match(alternative, subject):
                self.memo[id(pattern)] = alternative
                return True
            first_refusal = first_refusal or self._refusal
        self.restore(saved)
        self._refusal = first_refusal
        return self._fail(pattern, subject)

    def visit_AndPattern(self, pattern, subject) -> bool:
        return all(self.match(part, subject) for part in pattern.parts)

    def visit_SequencePattern(self, pattern, subject) -> bool:
        if not isinstance(subject, (tuple, list)) or len(subject) != len(pattern.patterns):
            return self._fail(pattern, subject)
        return all(self.match(place, value) for place, value in zip(pattern.patterns, subject))

    def visit_ConstraintPattern(self, pattern, subject) -> bool:
        return all(self.match(part, subject) for part in pattern.patterns)

    def visit_MultipleOfPattern(self, pattern, subject) -> bool:
        return (
            type(subject) is int and subject % pattern.unit == 0
        ) or self._fail(pattern, subject)

    def visit_RangePattern(self, pattern, subject) -> bool:
        found = not isinstance(subject, bool) and isinstance(subject, int)
        found = found and (pattern.lo is None or subject >= pattern.lo)
        found = found and (pattern.hi is None or subject <= pattern.hi)
        return found or self._fail(pattern, subject)

    def visit_OneOfPattern(self, pattern, subject) -> bool:
        return any(subject == value for value in pattern.values) or self._fail(pattern, subject)

    def visit_AttrPattern(self, pattern, subject) -> bool:
        if not hasattr(subject, pattern.attr):
            return self._fail(pattern, subject)
        return self.match(pattern.pattern, getattr(subject, pattern.attr))

    def visit_BitsPattern(self, pattern, subject) -> bool:
        width = getattr(self.bindings.get(pattern.dtype), "bit_width", None)
        if type(subject) is not int or type(width) is not int:
            return self._fail(pattern, subject)
        return self.match(pattern.pattern, subject * width)

    def visit_SwitchPattern(self, pattern, subject) -> bool:
        if pattern.param in self.bindings:
            wanted = self.bindings[pattern.param]
            branch = next((item for value, item in pattern.branches if value == wanted), None)
            return self._fail(pattern, subject) if branch is None else self.match(branch, subject)
        saved = self.snapshot()
        first_refusal = None
        for value, branch in pattern.branches:
            self.restore(saved)
            self.bindings[pattern.param] = value
            if self.match(branch, subject):
                return True
            first_refusal = first_refusal or self._refusal
        self.restore(saved)
        self._refusal = first_refusal
        return self._fail(pattern, subject)

    def visit_GuardPattern(self, pattern, subject) -> bool:
        value = evaluated(pattern.symbol, self.bindings)
        if value is None or not self.match(pattern.condition, value):
            return self._fail(pattern, subject)
        return self.match(pattern.pattern, subject)

    def _match_mode_trees(
        self,
        owner,
        pattern_shape,
        pattern_strides,
        subject_shape,
        subject_strides,
    ) -> bool:
        from .pattern import StarPattern  # noqa: PLC0415 - pattern protocol cycle

        if isinstance(pattern_shape, tuple):
            if not isinstance(subject_shape, tuple) or not isinstance(subject_strides, tuple):
                return self._fail(owner, (subject_shape, subject_strides))
            star = next(
                (
                    index
                    for index, item in enumerate(pattern_shape)
                    if isinstance(item, StarPattern)
                ),
                None,
            )
            minimum = len(pattern_shape) - (star is not None)
            if len(subject_shape) != len(subject_strides) or (
                star is None and len(subject_shape) != minimum
            ):
                return self._fail(owner, (subject_shape, subject_strides))
            if star is not None and len(subject_shape) < minimum:
                return self._fail(owner, (subject_shape, subject_strides))

            before = len(pattern_shape) if star is None else star
            for index in range(before):
                if not self._match_mode_trees(
                    owner,
                    pattern_shape[index],
                    pattern_strides[index],
                    subject_shape[index],
                    subject_strides[index],
                ):
                    return False
            if star is None:
                return True

            after = len(pattern_shape) - star - 1
            repeated_end = len(subject_shape) - after
            shape_star = pattern_shape[star]
            stride_star = pattern_strides[star]
            repeated = tuple(
                zip(
                    subject_shape[star:repeated_end],
                    subject_strides[star:repeated_end],
                )
            )
            if not self._match_repeated(
                shape_star,
                (shape_star.pattern, stride_star.pattern),
                repeated,
            ):
                return False
            self.memo[id(shape_star)] = tuple(subject_shape[star:repeated_end])
            self.memo[id(stride_star)] = tuple(subject_strides[star:repeated_end])
            for offset in range(after):
                pattern_index = star + 1 + offset
                subject_index = repeated_end + offset
                if not self._match_mode_trees(
                    owner,
                    pattern_shape[pattern_index],
                    pattern_strides[pattern_index],
                    subject_shape[subject_index],
                    subject_strides[subject_index],
                ):
                    return False
            return True

        if isinstance(subject_shape, tuple) or isinstance(subject_strides, tuple):
            return self._fail(owner, (subject_shape, subject_strides))
        return self.match(pattern_shape, subject_shape) and self.match(
            pattern_strides, subject_strides
        )

    def visit_LayoutPattern(self, pattern, subject) -> bool:
        structural = subject
        if (
            isinstance(structural, ComposedLayout)
            and structural.inner is None
            and structural.offset == 0
        ):
            structural = structural.outer
        if pattern.shape is not None or pattern.strides is not None:
            if not isinstance(structural, Layout) or structural.strides is None:
                return self._fail(pattern, subject)
            extents = tuple(flatten(structural.shape))
            strides = tuple(flatten(structural.strides))
            if any(type(number) is not int for number in (*extents, *strides)):
                return self._fail(pattern, subject)
            if any(number <= 0 for number in extents):
                return self._fail(pattern, subject)
            if pattern.shape is not None and pattern.strides is not None:
                if not self._match_mode_trees(
                    pattern,
                    pattern.shape,
                    pattern.strides,
                    structural.shape,
                    structural.strides,
                ):
                    return False
            else:
                places = pattern.positions()
                profile, actual, values = (
                    (pattern.shape, structural.shape, extents)
                    if pattern.shape is not None
                    else (pattern.strides, structural.strides, strides)
                )
                if not congruent(profile, actual) or not all(
                    self.match(place, value) for place, value in zip(places, values)
                ):
                    return False
        return all(self.match(predicate, subject) for predicate in pattern.predicates)

    def visit_SwizzlePattern(self, pattern, subject) -> bool:
        if subject is None:
            return self.match(pattern.bits, 0)
        if not isinstance(subject, Swizzle):
            return self._fail(pattern, subject)
        return all(
            self.match(place, value)
            for place, value in (
                (pattern.bits, subject.bits),
                (pattern.base, subject.base),
                (pattern.shift, subject.shift),
            )
        )

    def visit_ComposedLayoutPattern(self, pattern, subject) -> bool:
        stated = subject
        if isinstance(subject, Layout):
            subject = ComposedLayout(None, 0, subject)
        if not isinstance(subject, ComposedLayout):
            return self._fail(pattern, subject)
        return all(
            self.match(place, value)
            for place, value in (
                (pattern.inner, subject.inner),
                (pattern.offset, subject.offset),
                (pattern.outer, subject.outer),
            )
        ) and all(self.match(predicate, stated) for predicate in pattern.predicates)

    def visit_MeshPattern(self, pattern, subject) -> bool:
        if not isinstance(subject, Mesh):
            return self._fail(pattern, subject)
        picked = tuple(
            level
            for level in separate(subject)
            if getattr(level.topologies[0], "name", level.topologies[0]) in pattern.topologies
        )
        found = tuple(getattr(level.topologies[0], "name", level.topologies[0]) for level in picked)
        if len(picked) != len(pattern.topologies) or set(found) != set(pattern.topologies):
            return self._fail(pattern, subject)
        return self.match(pattern.layout, make_mesh(*picked).layout)

    def visit_ScalarPattern(self, pattern, subject) -> bool:
        return (
            isinstance(subject, TensorType) and subject.shape == ()
        ) or self._fail(pattern, subject)

    def visit_TensorPattern(self, pattern, subject) -> bool:
        if not isinstance(subject, TensorType) or subject.shape == ():
            return self._fail(pattern, subject, f"{subject!r} is no tensor")
        if pattern.shape is not None and (
            len(pattern.shape) != len(subject.shape)
            or not all(self.match(place, value) for place, value in zip(pattern.shape, subject.shape))
        ):
            return False
        for place, value in ((pattern.dtype, subject.dtype), (pattern.storage, subject.storage)):
            if not self.match(place, value):
                return False
        return (
            (pattern.layout is None or self.match(pattern.layout, subject.layout))
            and all(self.match(predicate, subject) for predicate in pattern.predicates)
        )

    def visit_ShardLayoutPattern(self, pattern, subject) -> bool:
        if not isinstance(subject, ShardLayout):
            return self._fail(pattern, subject)
        subject_names = tuple(
            getattr(topology, "name", topology) for topology in subject.mesh.topologies
        )
        if subject_names != pattern.mesh.topologies:
            return self._fail(pattern, subject)
        return all(
            self.match(place, value)
            for place, value in (
                (pattern.layout, subject.layout),
                (pattern.attrs, subject.attrs),
                (pattern.mesh, subject.mesh),
            )
        ) and all(self.match(predicate, subject) for predicate in pattern.predicates)

    def visit_Predicate(self, pattern, subject) -> bool:
        held = pattern.holds(subject, self.bindings)
        if held is None:
            self.pending.append(pattern)
            return True
        return held or self._fail(pattern, subject)

    def visit_AtomPattern(self, pattern, subject) -> bool:
        if not isinstance(subject, pattern.declarations):
            return self._fail(pattern, subject)
        for name, value in subject.bindings.items():
            if name in self.bindings and self.bindings[name] != value:
                return self._fail(pattern, subject)
            self.bindings[name] = value
        return True

    def visit_FromAtom(self, pattern, subject) -> bool:
        raise TypeError(
            f"the {pattern.role} operand is read against a call's atom; ask read_on(op)"
        )


class _PatternAlternatives:
    """Flatten the declaration alternatives represented by a pattern tree."""

    def alternatives(self, pattern, bindings=()) -> tuple:
        if not isinstance(pattern, _pattern_type()):
            return ((tuple(bindings), pattern),)
        for cls in type(pattern).__mro__:
            visitor = getattr(self, f"visit_{cls.__name__}", None)
            if visitor is not None:
                return visitor(pattern, bindings)
        raise NotImplementedError(f"{type(pattern).__name__} has no alternatives visitor")

    def visit_Pattern(self, pattern, bindings) -> tuple:
        return ((tuple(bindings), pattern),)

    def visit_OrPattern(self, pattern, bindings) -> tuple:
        return tuple(
            held
            for alternative in pattern.patterns
            for held in self.alternatives(alternative, bindings)
        )

    def visit_StarPattern(self, pattern, bindings) -> tuple:
        return ((tuple(bindings), pattern),)

    def visit_SwitchPattern(self, pattern, bindings) -> tuple:
        return tuple(
            held
            for value, branch in pattern.branches
            for held in self.alternatives(branch, (*bindings, (pattern.param, value)))
        )

    def visit_ShardLayoutPattern(self, pattern, bindings) -> tuple:
        return self.alternatives(pattern.layout, bindings)


def _pattern_type():
    from .pattern import Pattern  # noqa: PLC0415 - pattern protocol cycle

    return Pattern


def _named(value):
    """One enumerated field as an author writes it, or None when unstated."""
    return None if value is None else getattr(value, "name", str(value)).lower()


def _extents(bindings) -> dict:
    return {name: value for name, value in dict(bindings or {}).items() if type(value) is int}


def evaluated(value, captures):
    """Evaluate one symbolic dimension, or return None while it is unresolved."""
    try:
        held = substitute_shape_dim(value, _extents(captures))
    except DimSubstitutionError:
        return None
    return held if type(held) is int else None


def is_symbolic(value) -> bool:
    return isinstance(value, DimVar) or is_dim_op_call(value)


def written_dim(value) -> str:
    if isinstance(value, DimVar):
        return value.name
    if is_dim_op_call(value):
        left, right = (written_dim(arg) for arg in value.args)
        if isinstance(value.target, DimFloorDiv):
            return f"{left}/{right}"
        if isinstance(value.target, DimMul):
            return f"{left}*{right}"
        return f"({left} {type(value.target).__name__} {right})"
    return str(getattr(value, "value", value))


def written_binding(value) -> str:
    return getattr(value, "name", str(value))


def written_bindings(bindings) -> str:
    return ", ".join(f"{name}={written_binding(value)}" for name, value in bindings)


def written_tuple(items) -> str:
    written = ", ".join(items)
    return f"({written},)" if len(items) == 1 else f"({written})"


def written_grouped(modes) -> str:
    if isinstance(modes, tuple):
        return written_tuple(tuple(written_grouped(mode) for mode in modes))
    return written_place(modes)


def written_place(value, name: str = UNNAMED_PLACE) -> str:
    if isinstance(value, _pattern_type()):
        return value.describe(name)
    if is_symbolic(value):
        return written_dim(value)
    return UNNAMED_PLACE if is_layout_wildcard(value) else str(value)


def written_field(value) -> str | None:
    if isinstance(value, _pattern_type()):
        return value.describe()
    return _named(value)


def written_alternatives(items, name: str = UNNAMED_PLACE) -> str:
    held = tuple(
        (written_bindings(bindings), written_place(alternative, name))
        for bindings, alternative in items
    )
    width = max((len(label) for label, _ in held), default=0)
    lines = []
    for label, written in held:
        first, *rest = written.splitlines() or ("",)
        lines.append(first if not width else f"{label.ljust(width)}  {first}")
        lines.extend(line if not width else f"{' ' * (width + 2)}{line}" for line in rest)
    return "\n".join(lines)


def alternatives_of(pattern, bindings=()) -> tuple:
    return _PatternAlternatives().alternatives(pattern, bindings)


def matched(pattern, subject, captures=None) -> Match | None:
    """Match a nested pattern, symbolic dimension, wildcard, or fixed value."""
    owner = getattr(captures, "matcher", None)
    if isinstance(owner, PatternMatcher):
        return Match(dict(owner.bindings)) if owner.match(pattern, subject) else None
    if pattern is None:
        return Match(dict(captures or {}))
    held = Match(dict(captures or {}))
    if isinstance(pattern, DimVar):
        if pattern.name in held.captures:
            return held if held.captures[pattern.name] == subject else None
        if type(subject) is not int or not pattern.lo <= subject < pattern.hi:
            return None
        return Match({**held.captures, pattern.name: subject})
    if is_dim_op_call(pattern):
        found = evaluated(pattern, held.captures)
        return held if found is not None and found == subject else None
    if isinstance(pattern, _pattern_type()):
        return pattern.match(subject, held.captures)
    return held if pattern == subject else None


def relations_of(values) -> tuple[str, ...]:
    lines: dict[str, None] = {}
    for value in values:
        if isinstance(value, _pattern_type()):
            lines.update(dict.fromkeys(value.relations()))
    return tuple(lines)


def between_rules(op_type) -> tuple:
    return tuple(getattr(op_type, "between", ()))


def refusals_between(op_type, operands: dict) -> tuple[str, ...]:
    return tuple(
        rule.refused(operands) for rule in between_rules(op_type) if not rule.holds(operands)
    )


__all__ = [
    "ARRANGEMENT",
    "Match",
    "PatternMatcher",
    "OPAQUE",
    "UNNAMED_PLACE",
    "_named",
    "alternatives_of",
    "between_rules",
    "evaluated",
    "is_symbolic",
    "matched",
    "refusals_between",
    "relations_of",
    "written_alternatives",
    "written_binding",
    "written_bindings",
    "written_dim",
    "written_field",
    "written_grouped",
    "written_place",
    "written_tuple",
]
