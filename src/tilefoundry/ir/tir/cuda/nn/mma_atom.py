"""Declarative CUDA MMA atoms and operand patterns."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.pattern import (
    ABSENT,
    CapturePattern,
    ComposedLayoutPattern,
    Match,
    MeshPattern,
    MultipleOfPattern,
    OneOfPattern,
    OrPattern,
    Pattern,
    alternatives_of,
    arrangement_pattern,
    matched,
    resolved,
)
from tilefoundry.ir.pattern.match import written_binding, written_bindings, written_place
from tilefoundry.ir.types import ComposedLayout, Layout, Mesh
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.layout_algebra import coalesce
from tilefoundry.ir.types.mesh import levels, starts


class MmaAtom:
    """One instruction declaration; an instance binds its authored parameters."""

    namespace: str
    scope: Mesh
    capability: str
    C: object
    A: object
    B: object

    parameters: tuple[ParamDef, ...] = ()
    reference_name: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.parameters = tuple(value for value in vars(cls).values() if isinstance(value, ParamDef))
        cls.reference_name = f"{cls.namespace}.{cls.__name__}"

    def __init__(self, *, mesh: Mesh | None = None, **bindings):
        unknown = set(bindings) - {param.name for param in self.parameters}
        if unknown:
            raise ValueError(
                f"{self.reference_name} takes no parameter {', '.join(sorted(unknown))}"
            )
        held = {}
        for param in self.parameters:
            value = bindings[param.name] if param.name in bindings else self._implied(param, held)
            if matched(param.pattern, value, held) is None:
                raise ValueError(
                    f"{self.reference_name}: {param.name}={written_binding(value)} is not one "
                    f"it takes{self._where(held)}; it takes {param.name} "
                    f"{written_place(resolved(param.pattern, held), param.name)}"
                )
            held[param.name] = value
        self.bindings = held
        self.mesh = mesh
        self._roles: dict[str, object] = {}

    @classmethod
    def _implied(cls, param: ParamDef, held: dict):
        left = resolved(param.pattern, held)
        if not isinstance(left, Pattern) and left is not ABSENT:
            return left
        if param.has_default:
            return param.default
        raise ValueError(f"{cls.reference_name} needs {param.name}")

    @staticmethod
    def _where(held: dict) -> str:
        return "" if not held else f" where {written_bindings(held.items())}"

    @classmethod
    def bindings_for(cls, reads) -> tuple[dict, ...]:
        """Every parameter binding the operand types leave possible."""
        choices = []
        for param in cls.parameters:
            if isinstance(param.pattern, OneOfPattern):
                choices.append([(param.name, value) for value in param.pattern.values])
                continue
            if param.has_default:
                continue
            extent = cls._extent_of(reads, param.name) or 0
            choices.append(
                [
                    (param.name, value)
                    for value in range(1, extent + 1)
                    if extent % value == 0 and matched(param.pattern, value) is not None
                ]
            )
        return tuple(dict(one) for one in itertools.product(*choices))

    @staticmethod
    def _extent_of(reads, name: str) -> int | None:
        for pattern, held in reads:
            for _, tensor in alternatives_of(pattern):
                for extent, value in zip(
                    getattr(tensor, "shape", None) or (), getattr(held, "shape", ())
                ):
                    if isinstance(extent, DimVar) and extent.name == name and type(value) is int:
                        return value
        return None

    @classmethod
    def role_of(cls, role: str, bindings=()):
        return resolved(getattr(cls, role), dict(bindings))

    def role(self, role: str):
        held = self._roles.get(role)
        if held is None:
            held = self._roles[role] = self.role_of(role, self.bindings)
        return held

    @property
    def shape_mnk(self) -> tuple[int, int, int]:
        (m, n), (_, k) = tuple(self.role("C").shape), tuple(self.role("A").shape)
        return m, n, k

    @property
    def dtype_a(self):
        return self.role("A").dtype

    @property
    def dtype_b(self):
        return self.role("B").dtype

    @property
    def dtype_c(self):
        return self.role("C").dtype

    @property
    def required_scope(self) -> Mesh:
        return self.scope

    @classmethod
    def scope_pattern(cls) -> MeshPattern:
        topology, = cls.scope.topologies
        size = topology.size
        bare = arrangement_pattern(cls.scope.layout, per_mode=True)
        sliced = ComposedLayoutPattern(
            offset=CapturePattern("p0", MultipleOfPattern(size)),
            outer=arrangement_pattern(cls.scope.layout, per_mode=True),
        )
        return MeshPattern((topology.name,), OrPattern(sliced, bare))

    def on(self, mesh: Mesh) -> MmaAtom:
        return type(self)(mesh=mesh, **self.bindings)

    def written(self, mesh: str | None = None) -> str:
        stated, held = [], {}
        for param in self.parameters:
            value = self.bindings[param.name]
            try:
                implied = self._implied(param, held)
            except ValueError:
                implied = None
            if implied is None or implied != value:
                stated.append(f"{param.name}={self.written_value(value)}")
            held[param.name] = value
        if mesh is not None:
            stated.append(f"mesh={mesh}")
        return f"{self.reference_name}({', '.join(stated)})"

    @classmethod
    def written_value(cls, value) -> str:
        if type(value) is int:
            return str(value)
        return f"{cls.namespace}.{type(value).__name__}.{value.name}"

    @property
    def reference(self) -> str:
        return self.written()

    def describe(self) -> str:
        return self.reference

    def __eq__(self, other):
        return (
            type(other) is type(self)
            and other.bindings == self.bindings
            and other.mesh == self.mesh
        )

    def __hash__(self):
        return hash((type(self), tuple(self.bindings.items()), self.mesh))

    def __repr__(self):
        return self.written(None if self.mesh is None else repr(self.mesh))


@dataclass(frozen=True, init=False)
class AtomPattern(Pattern):
    """An instance of any declared atom class."""

    declarations: tuple[type[MmaAtom], ...]

    def __init__(self, *declarations):
        object.__setattr__(self, "declarations", tuple(declarations))

    def match(self, subject, captures=None):
        if not isinstance(subject, self.declarations):
            return None
        return Match({**dict(captures or {}), **subject.bindings})

    def describe(self, name: str = "_") -> str:
        return "one of " + ", ".join(held.reference_name for held in self.declarations)

    def resolve(self, bindings):
        return self


@dataclass(frozen=True)
class FromAtom(Pattern):
    """An operand pattern read from one role of the call's atom."""

    role: str

    def __post_init__(self):
        if self.role not in ("A", "B", "C"):
            raise ValueError("an MMA operand reads role A, B or C of its atom")

    def read_on(self, op):
        return op.atom.role_of(self.role, getattr(op.atom, "bindings", ()))

    def match(self, subject, captures=None):
        raise TypeError(f"the {self.role} operand is read against a call's atom; ask read_on(op)")

    def describe(self, name: str = "_") -> str:
        return f"the {self.role} operand of its atom"


def read_on(pattern, op):
    held = getattr(pattern, "read_on", None)
    return pattern if held is None else held(op)


def _reversed_modes(modes):
    if isinstance(modes, tuple):
        return tuple(_reversed_modes(mode) for mode in reversed(modes))
    return modes


def _reverse(layout: Layout) -> Layout:
    return Layout(
        _reversed_modes(tuple(layout.shape)),
        _reversed_modes(tuple(layout.strides)),
    )


def physical_frames_match(left: Mesh, right: Mesh) -> bool:
    """Compare affine offsets and ordered participant lanes."""
    if left.topologies != right.topologies:
        return False

    def frame(mesh):
        if isinstance(mesh.layout, ComposedLayout) and mesh.layout.inner is not None:
            return None
        try:
            arranged = levels(mesh)
        except (IndexError, TypeError, ValueError):
            return None
        if any(layout.strides is None for layout in arranged):
            return None
        return starts(mesh), tuple(coalesce(_reverse(layout)) for layout in arranged)

    a, b = frame(left), frame(right)
    return a is not None and b is not None and a == b


__all__ = [
    "AtomPattern",
    "FromAtom",
    "MmaAtom",
    "physical_frames_match",
    "read_on",
]
