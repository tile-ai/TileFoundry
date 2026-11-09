"""The ``Pattern`` base and the matcher that reads it, with the shared matching helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import ClassVar

from tilefoundry.ir import dim, expr, types
from tilefoundry.ir.types import int_tuple

from .solve import UNKNOWN, Unknown, failing, solve

UNREADABLE = object()
"""What a pattern's ``tensor_layout`` returns when it cannot read a tensor's layout."""

OPAQUE = object()


@dataclass(frozen=True)
class Match:
    """The bindings produced by a successful match."""

    captures: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Refusal:
    """The pattern node rejected at this match boundary."""

    pattern: object
    subject: object
    bindings: dict


@dataclass(frozen=True)
class Pattern:
    """Base class for a predicate that returns bindings or ``None``."""

    predicates: tuple["Predicate", ...] = field(default_factory=tuple, kw_only=True)
    repeats: ClassVar[bool] = False

    def replace(self, **changes):
        """A copy of this value with *changes* applied to its fields.

        ``dataclasses.replace`` cannot rebuild a value with ``init=False`` fields or with
        its own ``__init__``.
        """
        result = copy.copy(self)
        vars(result).update(changes)
        return result

    def match(self, subject, captures=None) -> Match | None:
        held = Matcher(dict(captures or {}))
        return Match(dict(held.bindings)) if held.match(self, subject) and held.solve() is True else None

    def captured(self) -> tuple[str, ...]:
        """The names this pattern binds, in field order."""
        return tuple(name for field in vars(self).values() for name in captured_names(field))

    def project(self, subject, bindings: dict):
        """The subject as this pattern reads it; a pattern that reads it whole returns it."""
        return subject

    def tensor_layout(self, subject: types.TensorType, matcher):
        """The layout of a tensor as this pattern reads it, or ``UNREADABLE``."""
        return subject.layout


def captured_names(value) -> tuple[str, ...]:
    """The names bound by *value*: a pattern, or the patterns in a tuple."""
    if isinstance(value, Pattern):
        return value.captured()
    if isinstance(value, tuple):
        return tuple(name for item in value for name in captured_names(item))
    return ()


class Bindings(dict):
    """A binding dict that keeps nested ``matched`` calls on its owner."""

    def __init__(self, matcher: Matcher, initial: dict) -> None:
        super().__init__(initial)
        self.matcher = matcher


class Matcher:
    """Match one pattern tree while retaining bindings and match evidence."""

    def __init__(self, bindings=None, *, owners=None, mesh=None):
        self.owners = owners if owners is not None else expr.StorageOwners((), {})
        self.mesh = mesh
        self.bindings: Bindings = Bindings(self, bindings or {})
        self.memo: dict[int, object] = {}
        self.pending: list[object] = []
        self.refusal: Refusal | None = None
        self.depth = 0

    def use_mesh(self, mesh) -> None:
        """Read tensors that carry no shard layout over *mesh*."""
        self.mesh = mesh

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
        root = self.depth == 0
        if root:
            self.refusal = None
        saved = self.snapshot()
        self.depth += 1
        try:
            try:
                if isinstance(pattern, Pattern):
                    subject = pattern.project(subject, self.bindings)
            except ValueError:
                found = self.fail(pattern, subject)
            else:
                found = self.match_one(pattern, subject)
        except Exception:
            self.restore(saved)
            raise
        finally:
            self.depth -= 1
        if found:
            if isinstance(pattern, Pattern):
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

    def fail(self, pattern, subject) -> bool:
        if self.refusal is None:
            self.refusal = Refusal(pattern, subject, dict(self.bindings))
        return False

    def match_one(self, pattern, subject) -> bool:
        """Match one node without a transaction."""
        if pattern is None:
            return True
        if isinstance(pattern, expr.DimVar):
            if pattern.name in self.bindings:
                return self.bindings[pattern.name] == subject or self.fail(pattern, subject)
            if type(subject) is not int or not pattern.lo <= subject <= pattern.hi:
                return self.fail(pattern, subject)
            self.bindings[pattern.name] = subject
            return True
        if expr.is_dim_op_call(pattern):
            found = evaluated(pattern, self.bindings)
            return (found is not None and found == subject) or self.fail(pattern, subject)
        if not isinstance(pattern, Pattern):
            return pattern == subject or self.fail(pattern, subject)
        for cls in type(pattern).__mro__:
            visitor = getattr(self, f"visit_{cls.__name__}", None)
            if visitor is not None:
                return visitor(pattern, subject) and all(
                    self.match(predicate, subject) for predicate in pattern.predicates
                )
        return self.fail(pattern, subject)

    def visit_Pattern(self, pattern, subject) -> bool:
        raise NotImplementedError(f"{type(pattern).__name__} has no visitor")

    def visit_Wildcard(self, pattern, subject) -> bool:
        if pattern.name:
            if pattern.name in self.bindings:
                return self.bindings[pattern.name] == subject or self.fail(pattern, subject)
            self.bindings[pattern.name] = subject
        return True

    def match_repeated(self, owner, patterns, subjects) -> bool:
        names = tuple(dict.fromkeys(captured_names(patterns)))
        prior = {name: self.bindings.pop(name) for name in names if name in self.bindings}
        captured = {name: [] for name in names}
        for values in subjects:
            if not all(self.match(pattern, value) for pattern, value in zip(patterns, values)):
                return False
            for name in names:
                if name not in self.bindings:
                    return self.fail(owner, subjects)
                captured[name].append(self.bindings.pop(name))
        for name, values in captured.items():
            held = tuple(values)
            if name in prior and prior[name] != held:
                return self.fail(owner, subjects)
            self.bindings[name] = prior.get(name, held)
        return True

    def visit_Star(self, pattern, subject) -> bool:
        if not isinstance(subject, (tuple, list)):
            return self.fail(pattern, subject)
        return self.match_repeated(
            pattern,
            (pattern.pattern,),
            tuple((value,) for value in subject),
        )

    def visit_Or(self, pattern, subject) -> bool:
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
        return self.fail(pattern, subject)

    def visit_And(self, pattern, subject) -> bool:
        return all(self.match(part, subject) for part in pattern.parts)

    def visit_Sequence(self, pattern, subject) -> bool:
        if not isinstance(subject, (tuple, list)) or len(subject) != len(pattern.patterns):
            return self.fail(pattern, subject)
        return all(self.match(place, value) for place, value in zip(pattern.patterns, subject))

    def visit_Range(self, pattern, subject) -> bool:
        found = not isinstance(subject, bool) and isinstance(subject, int)
        found = found and (pattern.lo is None or subject >= pattern.lo)
        found = found and (pattern.hi is None or subject <= pattern.hi)
        return found or self.fail(pattern, subject)

    def visit_Switch(self, pattern, subject) -> bool:
        if pattern.param in self.bindings:
            wanted = self.bindings[pattern.param]
            branch = next((item for value, item in pattern.branches if value == wanted), None)
            return self.fail(pattern, subject) if branch is None else self.match(branch, subject)
        saved = self.snapshot()
        for value, branch in pattern.branches:
            self.restore(saved)
            self.bindings[pattern.param] = value
            if self.match(branch, subject):
                return True
        self.restore(saved)
        return self.fail(pattern, subject)

    def match_mode_trees(
        self,
        owner,
        pattern_shape,
        pattern_strides,
        subject_shape,
        subject_strides,
    ) -> bool:
        if isinstance(pattern_shape, tuple):
            if not isinstance(subject_shape, tuple) or not isinstance(subject_strides, tuple):
                return self.fail(owner, (subject_shape, subject_strides))
            star = next(
                (
                    index
                    for index, item in enumerate(pattern_shape)
                    if isinstance(item, Pattern) and item.repeats
                ),
                None,
            )
            minimum = len(pattern_shape) - (star is not None)
            if len(subject_shape) != len(subject_strides) or (
                star is None and len(subject_shape) != minimum
            ):
                return self.fail(owner, (subject_shape, subject_strides))
            if star is not None and len(subject_shape) < minimum:
                return self.fail(owner, (subject_shape, subject_strides))

            before = len(pattern_shape) if star is None else star
            for index in range(before):
                if not self.match_mode_trees(
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
            if not self.match_repeated(
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
                if not self.match_mode_trees(
                    owner,
                    pattern_shape[pattern_index],
                    pattern_strides[pattern_index],
                    subject_shape[subject_index],
                    subject_strides[subject_index],
                ):
                    return False
            return True

        if isinstance(subject_shape, tuple) or isinstance(subject_strides, tuple):
            return self.fail(owner, (subject_shape, subject_strides))
        if not self.match(pattern_shape, subject_shape):
            return False
        return subject_shape == 1 or self.match(pattern_strides, subject_strides)

    def visit_Layout(self, pattern, subject) -> bool:
        structural = subject
        if (
            isinstance(structural, types.ComposedLayout)
            and structural.inner is None
            and structural.offset == 0
        ):
            structural = structural.outer
        if pattern.shape is not None or pattern.strides is not None:
            if not isinstance(structural, types.Layout):
                return self.fail(pattern, subject)
            extents = tuple(types.flatten(structural.shape))
            strides = tuple(types.flatten(structural.strides))
            if any(type(number) is not int for number in (*extents, *strides)):
                return self.fail(pattern, subject)
            if any(number <= 0 for number in extents):
                return self.fail(pattern, subject)
            if pattern.shape is not None and pattern.strides is not None:
                if not self.match_mode_trees(
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
                if not int_tuple.congruent(profile, actual) or not all(
                    self.match(place, value) for place, value in zip(places, values)
                ):
                    return False
        return True

    def visit_Swizzle(self, pattern, subject) -> bool:
        if subject is None:
            return self.match(pattern.bits, 0)
        if not isinstance(subject, types.Swizzle):
            return self.fail(pattern, subject)
        return all(
            self.match(place, value)
            for place, value in (
                (pattern.bits, subject.bits),
                (pattern.base, subject.base),
                (pattern.shift, subject.shift),
            )
        )

    def visit_ComposedLayout(self, pattern, subject) -> bool:
        if isinstance(subject, types.Layout):
            subject = types.ComposedLayout(None, 0, subject)
        if not isinstance(subject, types.ComposedLayout):
            return self.fail(pattern, subject)
        return all(
            self.match(place, value)
            for place, value in (
                (pattern.inner, subject.inner),
                (pattern.offset, subject.offset),
                (pattern.outer, subject.outer),
            )
        )

    def visit_Mesh(self, pattern, subject) -> bool:
        if not isinstance(subject, types.Mesh):
            return self.fail(pattern, subject)
        levels = types.separate(subject)
        if not isinstance(pattern.topologies, tuple):
            if not levels:
                return self.fail(pattern, subject)
            selected = levels[-1]
            names = tuple(getattr(level, "name", level) for level in selected.topologies)
            return self.match(pattern.topologies, names) and self.match(
                pattern.layout, selected.layout
            )
        picked = tuple(
            level
            for level in levels
            if getattr(level.topologies[0], "name", level.topologies[0]) in pattern.topologies
        )
        found = tuple(getattr(level.topologies[0], "name", level.topologies[0]) for level in picked)
        if len(picked) != len(pattern.topologies) or set(found) != set(pattern.topologies):
            return self.fail(pattern, subject)
        return self.match(pattern.layout, types.Mesh.from_meshes(*picked).layout)

    def visit_Tensor(self, pattern, subject) -> bool:
        if not isinstance(subject, types.TensorType):
            return self.fail(pattern, subject)
        if not self.match(pattern.dtype, subject.dtype):
            return False
        if not (
            subject.storage is expr.StorageKind.UMAT and isinstance(pattern.storage, expr.StorageKind)
        ) and not self.match(pattern.storage, subject.storage):
            return False
        if pattern.shape is not None and (
            len(pattern.shape) != len(subject.shape)
            or not all(
                self.match(place, value) for place, value in zip(pattern.shape, subject.shape)
            )
        ):
            return False

        layout = subject.layout
        if isinstance(pattern.layout, Pattern):
            layout = pattern.layout.tensor_layout(subject, self)
            if layout is UNREADABLE:
                return self.fail(pattern.layout, subject.layout)
        return pattern.layout is None or self.match(pattern.layout, layout)

    def visit_ShardLayout(self, pattern, subject) -> bool:
        if not isinstance(subject, types.ShardLayout):
            return self.fail(pattern, subject)
        subject_names = tuple(
            getattr(topology, "name", topology) for topology in subject.mesh.topologies
        )
        if not self.match(pattern.mesh.topologies, subject_names):
            return self.fail(pattern, subject)
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
        return held or self.fail(pattern, subject)

    def visit_AtomPattern(self, pattern, subject) -> bool:
        if not isinstance(subject, pattern.declarations):
            return self.fail(pattern, subject)
        for name, value in subject.bindings.items():
            if name in self.bindings and self.bindings[name] != value:
                return self.fail(pattern, subject)
            self.bindings[name] = value
        return True

    def visit_FromAtom(self, pattern, subject) -> bool:
        raise TypeError(
            f"the {pattern.role} operand is read against a call's atom; ask read_on(op)"
        )


def named(value):
    """One enumerated field as an author writes it, or None when unstated."""
    return None if value is None else getattr(value, "name", str(value)).lower()


def extents(bindings) -> dict:
    return {name: value for name, value in dict(bindings or {}).items() if type(value) is int}


def evaluated(value, captures):
    """Evaluate one symbolic dimension, or return None while it is unresolved."""
    try:
        held = dim.substitute_shape_entry(value, extents(captures))
    except dim.SubstitutionError:
        return None
    return held if type(held) is int else None


def is_symbolic(value) -> bool:
    return isinstance(value, expr.DimVar) or expr.is_dim_op_call(value)


def matched(pattern, subject, captures=None) -> Match | None:
    """Match a nested pattern, symbolic dimension, wildcard, or fixed value."""
    owner = getattr(captures, "matcher", None)
    if isinstance(owner, Matcher):
        return Match(dict(owner.bindings)) if owner.match(pattern, subject) else None
    if pattern is None:
        return Match(dict(captures or {}))
    held = Match(dict(captures or {}))
    if isinstance(pattern, expr.DimVar):
        if pattern.name in held.captures:
            return held if held.captures[pattern.name] == subject else None
        if type(subject) is not int or not pattern.lo <= subject <= pattern.hi:
            return None
        return Match({**held.captures, pattern.name: subject})
    if expr.is_dim_op_call(pattern):
        found = evaluated(pattern, held.captures)
        return held if found is not None and found == subject else None
    if isinstance(pattern, Pattern):
        return pattern.match(subject, held.captures)
    return held if pattern == subject else None


def between_rules(op_type) -> tuple:
    return tuple(getattr(op_type, "between", ()))


def refusals_between(op_type, operands: dict) -> tuple[str, ...]:
    return tuple(
        rule.refused(operands) for rule in between_rules(op_type) if not rule.holds(operands)
    )


__all__ = [
    "UNREADABLE",
    "OPAQUE",
    "Match",
    "Matcher",
    "Pattern",
    "Refusal",
    "between_rules",
    "captured_names",
    "evaluated",
    "is_symbolic",
    "matched",
    "named",
    "refusals_between",
]
