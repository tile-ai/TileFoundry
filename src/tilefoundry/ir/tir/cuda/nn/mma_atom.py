"""Declarative CUDA MMA atoms and operand patterns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum

from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.pattern import (
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    Pattern,
    PatternMatcher,
    ShardLayoutPattern,
    SwitchPattern,
    TensorPattern,
    WildcardPattern,
    matched,
)
from tilefoundry.ir.pattern import (
    predicates as P,
)
from tilefoundry.ir.pattern.utils import declared_shape, matched_row_issues, selected_pattern
from tilefoundry.ir.types import ComposedLayout, DType, Layout, Mesh, ShardLayout, TensorType
from tilefoundry.ir.types.layout_algebra import coalesce
from tilefoundry.ir.types.mesh import levels, starts
from tilefoundry.ir.types.utils import tile_view_layout

_MISS = object()


def execution_mesh_pattern(execution_mesh: Mesh) -> MeshPattern:
    """Any aligned run matching an instruction's execution mesh."""
    (topology,) = execution_mesh.topologies
    per_mode = (P.Forward(per_mode=True), P.Injective(per_mode=True))
    layout = ComposedLayoutPattern(
        inner=None,
        offset=WildcardPattern("p0"),
        outer=LayoutPattern.from_layout(execution_mesh.layout, predicates=per_mode),
        predicates=(WildcardPattern("p0") % topology.size == 0,),
    )
    return MeshPattern((topology.name,), layout)


class MmaAtom:
    """One instruction declaration; an instance binds its authored parameters."""

    namespace: str
    execution_mesh: Mesh
    capability: str
    resource: str
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
            if param.annotation is DType and isinstance(value, str):
                value = DType.from_name(value)
            if matched(param.pattern, value, held) is None:
                shown = value.name if isinstance(value, DType) else repr(value)
                raise ValueError(
                    f"{self.reference_name}: {param.name}={shown} is not one "
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

    def operand_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Return the declared C, A, and B shapes for one atom issue."""
        shapes = tuple(
            declared_shape(selected_pattern(self.role(role), self.bindings), self.bindings)
            for role in ("C", "A", "B")
        )
        if any(shape is None for shape in shapes):
            raise ValueError(f"{self.reference_name} does not declare fixed operand shapes")
        return shapes

    def operand_tiles(
        self,
        whole_types: tuple[TensorType, ...],
        frame: Mesh,
        axes: tuple[tuple[int, ...], ...],
        *,
        repeat: tuple[int, ...],
    ) -> tuple[tuple[TensorType, ...], tuple[tuple[int, int] | None, ...]]:
        """Return the operand tiles and declared adjacent-issue properties."""
        shapes = self.operand_shapes()
        patterns = tuple(
            selected_pattern(self.role(role), self.bindings) for role in ("C", "A", "B")
        )
        tiles = []
        for whole, shape, pattern, mapped in zip(whole_types, shapes, patterns, axes, strict=True):
            layout = tile_view_layout(
                whole, shape, counts=tuple(repeat[axis] for axis in mapped), participant=frame
            )
            if (
                isinstance(pattern, TensorPattern)
                and isinstance(pattern.layout, ShardLayoutPattern)
                and not isinstance(layout, ShardLayout)
            ):
                layout = ShardLayout(layout, pattern.layout.attrs, frame)
            if isinstance(layout, ShardLayout):
                layout = replace(layout, mesh=frame)
            tiles.append(TensorType(shape, whole.dtype, layout, whole.storage))
        matcher = PatternMatcher(self.bindings)
        if not all(
            matcher.match(pattern, tile)
            for pattern, tile in zip(patterns, tiles, strict=True)
        ) or not matcher.solve():
            raise ValueError(f"{self.reference_name} tile violates its operand declaration")
        rows = tuple(
            None
            if (row := matched_row_issues(pattern, matcher)) is None
            else (mapped[row[0]], row[1])
            for pattern, mapped in zip(patterns, axes, strict=True)
        )
        return tuple(tiles), rows

    @property
    def required_execution_mesh(self) -> Mesh:
        return self.execution_mesh

    @classmethod
    def execution_mesh_pattern(cls) -> MeshPattern:
        return execution_mesh_pattern(cls.execution_mesh)

    def on(self, mesh: Mesh) -> MmaAtom:
        return type(self)(mesh=mesh, **self.bindings)

    def stated_bindings(self) -> tuple[tuple[str, object], ...]:
        """The bindings a reader must see: those earlier ones do not imply."""
        stated, held = [], {}
        for param in self.parameters:
            value = self.bindings[param.name]
            try:
                implied = self._implied(param, held)
            except ValueError:
                implied = None
            if implied is None or implied != value:
                stated.append((param.name, value))
            held[param.name] = value
        return tuple(stated)

    @classmethod
    def configurations(
        cls,
        *,
        defaulted: bool,
        values: Callable[[ParamDef], tuple] | None = None,
    ) -> tuple[dict, ...]:
        """Every binding of this declaration's parameters that its patterns admit.

        Enum and DType parameters take every member their pattern admits;
        ``values`` supplies the others, and without it they stay unbound.
        ``defaulted`` also varies parameters that have a default. Shared by
        ``schedule candidates`` and HIR typeinfer.
        """
        states: tuple[dict, ...] = ({},)
        for param in cls.parameters:
            if param.has_default and not defaulted:
                continue
            options = cls._options(param, values)
            if options is None:
                continue
            held = []
            for state in states:
                for value in options:
                    matcher = PatternMatcher(state)
                    if matcher.match(param.pattern, value) and matcher.solve():
                        held.append({**state, param.name: value})
            states = tuple(held)
        return states

    @staticmethod
    def _options(param: ParamDef, values) -> tuple | None:
        annotation = param.annotation
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return tuple(annotation)
        if annotation is DType:
            return tuple(DType._members().values())
        return None if values is None else values(param)

    def __eq__(self, other):
        return (
            type(other) is type(self)
            and other.bindings == self.bindings
            and other.mesh == self.mesh
        )

    def __hash__(self):
        return hash((type(self), tuple(self.bindings.items()), self.mesh))

    def __repr__(self):
        from tilefoundry.inspection.printer_base import PythonPrinter  # noqa: PLC0415

        return PythonPrinter().atom_reference(self)


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
    "execution_mesh_pattern",
]
