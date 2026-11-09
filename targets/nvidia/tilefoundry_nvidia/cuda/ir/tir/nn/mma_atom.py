"""Declarative CUDA MMA atoms and operand patterns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial

from tilefoundry.ir import core, expr, pattern, types
from tilefoundry.ir.types import layout_algebra

MISS = object()


def held_text(held: dict) -> str:
    return "" if not held else f" where {held!r}"


def execution_mesh_pattern(execution_mesh: types.Mesh, *, groups=False) -> pattern.Mesh:
    """Any aligned run matching an instruction's execution mesh."""
    (topology,) = execution_mesh.topologies
    per_mode = (pattern.Forward(per_mode=True), pattern.Injective(per_mode=True))
    outer = pattern.Layout.from_layout(execution_mesh.layout, predicates=per_mode)
    if groups:
        outer = pattern.Layout(
            ((pattern.Star(pattern.Wildcard("g")), *outer.shape[0]),),
            ((pattern.Star(pattern.Wildcard("gs")), *outer.strides[0]),),
        )
    layout = pattern.ComposedLayout(
        inner=None,
        offset=pattern.Wildcard("p0"),
        outer=outer,
        predicates=(pattern.Wildcard("p0") % topology.size == 0,)
        + ((pattern.Forward(), pattern.Injective()) if groups else ()),
    )
    return pattern.Mesh((topology.name,), layout)


def rmem_fragment(shape: tuple, dtype: expr.DType, held: types.ShardLayout) -> pattern.Tensor:
    """A warp MMA operand held in registers as *held* lays it out over its mesh."""
    return pattern.Tensor(
        shape=shape,
        dtype=dtype,
        storage=expr.StorageKind.RMEM,
        layout=pattern.ShardLayout(
            layout=pattern.Layout.from_layout(held.layout),
            attrs=held.attrs,
            mesh=execution_mesh_pattern(held.mesh),
        ),
    )


class MmaAtom(core.Printable):
    """One instruction declaration; an instance binds its authored parameters."""

    namespace: str
    execution_mesh: types.Mesh
    capability: str
    resource: str
    C: object
    A: object
    B: object

    roles = ("C", "A", "B")

    parameters: tuple[expr.ParamDef, ...] = ()

    def __init_subclass__(cls, **kwargs):
        """Collect the parameters, and let a declared mesh state the pattern that matches it."""
        super().__init_subclass__(**kwargs)
        if "execution_mesh" in vars(cls):
            cls.required_execution_mesh = cls.execution_mesh
            cls.execution_mesh_pattern = staticmethod(partial(execution_mesh_pattern, cls.execution_mesh))
        cls.parameters = tuple(value for value in vars(cls).values() if isinstance(value, expr.ParamDef))
        expr.register_type(dialect="T", namespace=cls.namespace.removeprefix("T."))(cls)

    def __init__(self, *, mesh: types.Mesh | None = None, **bindings):
        unknown = set(bindings) - {param.name for param in self.parameters}
        if unknown:
            raise ValueError(
                f"{expr.get_registered_name(self)} takes no parameter {', '.join(sorted(unknown))}"
            )
        held = {}
        for param in self.parameters:
            value = bindings[param.name] if param.name in bindings else self.implied(param, held)
            if param.annotation is expr.DType and isinstance(value, str):
                value = expr.DType.from_name(value)
            if pattern.matched(param.pattern, value, held) is None:
                shown = value.name if isinstance(value, expr.DType) else repr(value)
                raise ValueError(
                    f"{expr.get_registered_name(self)}: {param.name}={shown} is not one "
                    f"it takes{held_text(held)}; it takes {param.name} "
                    f"{param.pattern!r}"
                )
            held[param.name] = value
        self.bindings = held
        self.mesh = mesh

    def implied(self, param: expr.ParamDef, held: dict):
        declared = param.pattern
        while isinstance(declared, pattern.Switch) and declared.param in held:
            declared = dict(declared.branches).get(held[declared.param], MISS)
            if declared is MISS:
                break
        if declared is not MISS and not isinstance(declared, pattern.Pattern):
            return declared
        if param.has_default:
            return param.default
        raise ValueError(f"{expr.get_registered_name(self)} needs {param.name}")

    def role(self, role: str):
        return getattr(type(self), role)

    def match_view(self, param_name: str, type_: types.TensorType) -> types.TensorType:
        """Return a temporary pattern view; projection and emission keep the original type."""
        return type_

    def operand_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Return the declared C, A, and B shapes for one atom issue."""
        shapes = tuple(
            pattern.declared_shape(pattern.selected(self.role(role), self.bindings), self.bindings)
            for role in self.roles
        )
        if any(shape is None for shape in shapes):
            raise ValueError(f"{expr.get_registered_name(self)} does not declare fixed operand shapes")
        return shapes

    def on(self, mesh: types.Mesh) -> MmaAtom:
        return type(self)(mesh=mesh, **self.bindings)

    def print(self, printer: core.PrinterBase, ctx=None) -> str:
        """This atom as importable DSL source, written with *printer*.

        The atom states its own name, every binding in parameter order and an
        Enum's registered type name; every other value, the mesh included, is
        printed by *printer* in the same import context.
        """
        stated = [
            f"{name}={self.printed_value(value, printer, ctx)}"
            for name, value in self.bindings.items()
        ]
        if self.mesh is not None:
            stated.append(f"mesh={printer.print(self.mesh, ctx)}")
        return f"{expr.get_registered_name(self)}({', '.join(stated)})"

    def printed_value(self, value, printer: core.PrinterBase, ctx) -> str:
        if isinstance(value, Enum):
            return f"{expr.get_registered_name(value)}.{value.name}"
        return printer.print(value, ctx)

    def __eq__(self, other):
        return (
            type(other) is type(self)
            and other.bindings == self.bindings
            and other.mesh == self.mesh
        )

    def __hash__(self):
        return hash((type(self), tuple(self.bindings.items()), self.mesh))

    def __repr__(self):
        stated = [f"{name}={self.repr_value(value)}" for name, value in self.bindings.items()]
        if self.mesh is not None:
            stated.append(f"mesh={self.mesh!r}")
        return f"{expr.get_registered_name(self)}({', '.join(stated)})"

    def repr_value(self, value) -> str:
        if isinstance(value, Enum):
            return f"{expr.get_registered_name(value)}.{value.name}"
        if isinstance(value, expr.DType):
            return repr(value.name)
        return repr(value)


@dataclass(frozen=True, init=False)
class AtomPattern(pattern.Pattern):
    """An instance of any declared atom class."""

    declarations: tuple[type[MmaAtom], ...]

    def __init__(self, *declarations):
        object.__setattr__(self, "declarations", tuple(declarations))
        object.__setattr__(self, "predicates", ())


@dataclass(frozen=True)
class FromAtom(pattern.Pattern):
    """An operand pattern read from one role of the call's atom."""

    role: str

    def __post_init__(self):
        if self.role not in ("A", "B", "C"):
            raise ValueError("an MMA operand reads role A, B or C of its atom")

    def read_on(self, op):
        declaration = op.atom if isinstance(op.atom, type) else type(op.atom)
        return getattr(declaration, self.role)


def reversed_modes(modes):
    if isinstance(modes, tuple):
        return tuple(reversed_modes(mode) for mode in reversed(modes))
    return modes


def reverse(layout: types.Layout) -> types.Layout:
    return types.Layout(
        reversed_modes(tuple(layout.shape)),
        reversed_modes(tuple(layout.strides)),
    )


def participant_frame(mesh):
    if isinstance(mesh.layout, types.ComposedLayout) and mesh.layout.inner is not None:
        return None

    try:
        arranged = types.levels(mesh)
    except (IndexError, TypeError, ValueError):
        return None
    return types.starts(mesh), tuple(layout_algebra.coalesce(reverse(layout)) for layout in arranged)


def physical_frames_match(left: types.Mesh, right: types.Mesh) -> bool:
    """Compare affine offsets and ordered participant lanes."""
    if left.topologies != right.topologies:
        return False
    a, b = participant_frame(left), participant_frame(right)
    return a is not None and b is not None and a == b


__all__ = [
    "AtomPattern",
    "FromAtom",
    "MmaAtom",
    "physical_frames_match",
    "execution_mesh_pattern",
    "rmem_fragment",
]
