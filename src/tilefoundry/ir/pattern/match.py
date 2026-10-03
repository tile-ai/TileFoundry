"""Shared matching helpers for IR patterns."""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.types import (
    ComposedLayout,
    Layout,
    Mesh,
    ShardLayout,
    StorageKind,
    Swizzle,
    TensorType,
    make_mesh,
)
from tilefoundry.ir.types.dim import DimVar, is_dim_op_call
from tilefoundry.ir.types.int_tuple import congruent
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.mesh import separate
from tilefoundry.ir.types.substitute import DimSubstitutionError, substitute_shape_dim

OPAQUE = object()


class Unknown:
    """A truthy result when the solver cannot prove success or failure."""

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True)
class Match:
    """The bindings produced by a successful match."""

    captures: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Refusal:
    """The first pattern node rejected during one match attempt."""

    pattern: object
    subject: object
    bindings: dict


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
        self.refusal: Refusal | None = None
        self._depth = 0

    def snapshot(self):
        """Save all transactional matcher state."""
        return (
            dict(self.bindings),
            dict(self.memo),
            list(self.pending),
            self.refusal,
        )

    def restore(self, saved) -> None:
        """Restore state saved by :meth:`snapshot`."""
        bindings, memo, pending, refusal = saved
        self.bindings.clear()
        self.bindings.update(bindings)
        self.memo.clear()
        self.memo.update(memo)
        self.pending[:] = pending
        self.refusal = refusal

    def match(self, pattern, subject) -> bool:
        """Match *pattern* against *subject* transactionally."""
        root = self._depth == 0
        if root:
            self.refusal = None
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
        refusal = self.refusal or Refusal(pattern, subject, dict(self.bindings))
        self.restore(saved)
        self.refusal = refusal
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
        subject = self.memo.get(id(predicate), self.memo.get(id(formulas[0])))
        if solved is UNKNOWN:
            self.refusal = Refusal(predicate, subject, dict(self.bindings))
            return UNKNOWN
        if solved is not None:
            self.bindings.update(solved)
            self.pending.clear()
            return True
        self.refusal = Refusal(predicate, subject, dict(self.bindings))
        return False

    def _fail(self, pattern, subject) -> bool:
        if self.refusal is None:
            self.refusal = Refusal(pattern, subject, dict(self.bindings))
        return False

    def _match(self, pattern, subject) -> bool:
        if pattern is None:
            return True
        if isinstance(pattern, DimVar):
            if pattern.name in self.bindings:
                return self.bindings[pattern.name] == subject or self._fail(pattern, subject)
            if type(subject) is not int or not pattern.lo <= subject <= pattern.hi:
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
                return visitor(pattern, subject) and all(
                    self.match(predicate, subject) for predicate in pattern.predicates
                )
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
                    return self._fail(owner, subjects)
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
            first_refusal = first_refusal or self.refusal
        self.restore(saved)
        self.refusal = first_refusal
        return self._fail(pattern, subject)

    def visit_AndPattern(self, pattern, subject) -> bool:
        return all(self.match(part, subject) for part in pattern.parts)

    def visit_SequencePattern(self, pattern, subject) -> bool:
        if not isinstance(subject, (tuple, list)) or len(subject) != len(pattern.patterns):
            return self._fail(pattern, subject)
        return all(self.match(place, value) for place, value in zip(pattern.patterns, subject))

    def visit_RangePattern(self, pattern, subject) -> bool:
        found = not isinstance(subject, bool) and isinstance(subject, int)
        found = found and (pattern.lo is None or subject >= pattern.lo)
        found = found and (pattern.hi is None or subject <= pattern.hi)
        return found or self._fail(pattern, subject)

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
            first_refusal = first_refusal or self.refusal
        self.restore(saved)
        self.refusal = first_refusal
        return self._fail(pattern, subject)

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
        return True

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
        )

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

    def visit_Ranked(self, pattern, subject) -> bool:
        return (
            isinstance(subject, TensorType) and len(subject.shape) > 0
        ) or self._fail(pattern, subject)

    def visit_TensorPattern(self, pattern, subject) -> bool:
        if not isinstance(subject, TensorType):
            return self._fail(pattern, subject)
        if pattern.shape is not None and (
            len(pattern.shape) != len(subject.shape)
            or not all(self.match(place, value) for place, value in zip(pattern.shape, subject.shape))
        ):
            return False
        if not self.match(pattern.dtype, subject.dtype):
            return False
        if not (
            subject.storage is StorageKind.UMAT and isinstance(pattern.storage, StorageKind)
        ) and not self.match(pattern.storage, subject.storage):
            return False
        return pattern.layout is None or self.match(pattern.layout, subject.layout)

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
        )

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
        if type(subject) is not int or not pattern.lo <= subject <= pattern.hi:
            return None
        return Match({**held.captures, pattern.name: subject})
    if is_dim_op_call(pattern):
        found = evaluated(pattern, held.captures)
        return held if found is not None and found == subject else None
    if isinstance(pattern, _pattern_type()):
        return pattern.match(subject, held.captures)
    return held if pattern == subject else None


def between_rules(op_type) -> tuple:
    return tuple(getattr(op_type, "between", ()))


def refusals_between(op_type, operands: dict) -> tuple[str, ...]:
    return tuple(
        rule.refused(operands) for rule in between_rules(op_type) if not rule.holds(operands)
    )


__all__ = [
    "Match",
    "PatternMatcher",
    "Refusal",
    "OPAQUE",
    "between_rules",
    "evaluated",
    "is_symbolic",
    "matched",
    "refusals_between",
]
