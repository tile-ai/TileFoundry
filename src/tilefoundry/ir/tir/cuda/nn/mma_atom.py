"""Declarative CUDA MMA atoms and operand patterns."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.pattern import (
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    Pattern,
    SwitchPattern,
    WildcardPattern,
    matched,
)
from tilefoundry.ir.pattern import (
    predicates as P,
)
from tilefoundry.ir.types import ComposedLayout, Layout, Mesh
from tilefoundry.ir.types.layout_algebra import coalesce
from tilefoundry.ir.types.mesh import levels, starts

_MISS = object()


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
                    f"{self.reference_name}: {param.name}={value!r} is not one "
                    f"it takes{self._where(held)}; it takes {param.name} "
                    f"{param.pattern!r}"
                )
            held[param.name] = value
        self.bindings = held
        self.mesh = mesh

    @classmethod
    def _implied(cls, param: ParamDef, held: dict):
        pattern = param.pattern
        while isinstance(pattern, SwitchPattern) and pattern.param in held:
            pattern = dict(pattern.branches).get(held[pattern.param], _MISS)
            if pattern is _MISS:
                break
        if pattern is not _MISS and not isinstance(pattern, Pattern):
            return pattern
        if param.has_default:
            return param.default
        raise ValueError(f"{cls.reference_name} needs {param.name}")

    @staticmethod
    def _where(held: dict) -> str:
        return "" if not held else f" where {held!r}"

    def role(self, role: str):
        return getattr(type(self), role)

    @property
    def required_scope(self) -> Mesh:
        return self.scope

    @classmethod
    def scope_pattern(cls) -> MeshPattern:
        topology, = cls.scope.topologies
        size = topology.size
        per_mode = (P.Forward(per_mode=True), P.Injective(per_mode=True))
        layout = ComposedLayoutPattern(
            inner=None,
            offset=WildcardPattern("p0"),
            outer=LayoutPattern.from_layout(cls.scope.layout, predicates=per_mode),
            predicates=(WildcardPattern("p0") % size == 0,),
        )
        return MeshPattern((topology.name,), layout)

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


@dataclass(frozen=True)
class FromAtom(Pattern):
    """An operand pattern read from one role of the call's atom."""

    role: str

    def __post_init__(self):
        if self.role not in ("A", "B", "C"):
            raise ValueError("an MMA operand reads role A, B or C of its atom")

    def read_on(self, op):
        return getattr(type(op.atom), self.role)


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
]
