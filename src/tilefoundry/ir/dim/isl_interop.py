"""Interoperation between dimension and shape IR values and isl.

Dimension definitions stay in :mod:`tilefoundry.ir.expr.dim` and construction in
:mod:`tilefoundry.ir.dim.simplify`;
pure isl operations stay in :mod:`tilefoundry.utils.isl_utils`. This module owns
the boundary between those layers: rendering, decoding, normalization, value
ranges, and shape domains.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType

import isl

from tilefoundry import services
from tilefoundry.ir import expr, types
from tilefoundry.ir.types import int_tuple

from .mesh_coord import MeshCoord
from .simplify import i64_const

__all__ = [
    "IslParamValues",
    "isl_to_dimension",
    "layout_to_isl_map",
    "normalize",
    "normalize_entries",
    "shape_to_isl_set",
    "dimension_to_isl_pw_aff",
    "value_range",
]

IslParamValues = dict[str, expr.Expr]
"""isl parameter name -> the IR value it stands for; the one dictionary every conversion shares."""


class NameCounter:
    """The numbers that make isl parameter names unique within the process."""

    def __init__(self) -> None:
        """Start counting at zero."""
        self.numbers = itertools.count()

    def next(self) -> int:
        """The next unused number."""
        return next(self.numbers)


counter = NameCounter()

INTEGER_BINARY_DIM_OP = MappingProxyType(
    {
        expr.BinaryKind.ADD: expr.DimAdd,
        expr.BinaryKind.SUB: expr.DimSub,
        expr.BinaryKind.MUL: expr.DimMul,
        expr.BinaryKind.FLOOR_DIV: expr.DimFloorDiv,
        expr.BinaryKind.MOD: expr.DimMod,
        expr.BinaryKind.MIN: expr.DimMin,
        expr.BinaryKind.MAX: expr.DimMax,
    }
)


def fresh_name(values: IslParamValues, taken: set[str]) -> str:
    """A parameter name no value in *values* and no coordinate in *taken* has.

    The name carries no meaning, so it is a fixed prefix and a number rather than
    anything read off the value: what a parameter stands for is in *values*.
    """
    while True:
        name = f"p{counter.next()}"
        if name not in values and name not in taken:
            return name


def leaf_bound(value) -> tuple[int, int] | None:
    """The half-open range a leaf states for itself, if it states one."""
    if isinstance(value, expr.DimVar):
        return value.lo, value.hi + 1
    stored = expr.get_metadata(value, expr.RangeMetadata) if isinstance(value, expr.Expr) else None
    return None if stored is None else (stored.lo, stored.hi)


def named_bound(value) -> tuple[int, int] | None:
    """The range of a value some earlier conversion already named."""
    try:
        return value_range(value)
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        return None


def pw_aff_of(node: str, params: dict[str, tuple[int, int] | None]) -> isl.pw_aff:
    prefix = f"[{', '.join(params)}] -> " if params else ""
    return isl.pw_aff(prefix + f"{{ [{node}] }}")


def bound_of(
    node: str,
    params: dict[str, tuple[int, int] | None],
) -> tuple[int, int] | None:
    """Return finite half-open bounds for one rendered expression, if available."""
    pw_aff = pw_aff_of(node, params)
    constraints = [
        f"{lo} <= {name} <= {hi - 1}"
        for name, bound in params.items()
        if bound is not None
        for lo, hi in (bound,)
    ]
    if constraints:
        prefix = f"[{', '.join(params)}] -> "
        context = isl.set(prefix + f"{{ : {' and '.join(constraints)} }}")
        pw_aff = pw_aff.intersect_params(context)
    minimum = pw_aff.min_val()
    maximum = pw_aff.max_val()
    if not minimum.is_int() or not maximum.is_int():
        return None
    return int(minimum.num_si()), int(maximum.num_si()) + 1


def constant_value_of(
    node: str,
    params: dict[str, tuple[int, int] | None],
) -> int | None:
    """Return an integer only when *expr* is constant without parameter bounds."""
    pw_aff = pw_aff_of(node, params)
    minimum = pw_aff.min_val()
    maximum = pw_aff.max_val()
    if not minimum.is_int() or not maximum.is_int():
        return None
    lo = int(minimum.num_si())
    return lo if lo == int(maximum.num_si()) else None


def interval_mul(
    a_bounds: tuple[int, int] | None,
    b_bounds: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if a_bounds is None or b_bounds is None:
        return None
    alo, ahi = a_bounds
    blo, bhi = b_bounds
    corners = (
        alo * blo,
        alo * (bhi - 1),
        (ahi - 1) * blo,
        (ahi - 1) * (bhi - 1),
    )
    return min(corners), max(corners) + 1


def dim_op_type(dim: expr.Call) -> type | None:
    op = type(dim.target)
    if op in expr.DIM_OP_TYPES:
        return op
    kind = getattr(dim.target, "kind", None)
    if not (
        isinstance(dim.type, types.TensorType)
        and dim.type.shape == ()
        and isinstance(dim.type.dtype, expr.IntegerDType)
        and isinstance(kind, expr.BinaryKind)
    ):
        return None
    return INTEGER_BINARY_DIM_OP.get(kind)


class Renderer:
    """Render a dimension expression as an isl expression string, naming its leaves.

    Each leaf that is not an integer becomes a parameter (or a coordinate when it
    is one of *coords*) with a name and, where the leaf states one, a range.
    """

    def __init__(self, values: IslParamValues, coords: Mapping[int, str], param_name=None) -> None:
        """Start with the parameters *values* and the coordinates *coords* already named."""
        self.memo: dict[int, tuple[object, str]] = {}
        self.values = values
        self.param_name = param_name
        self.coord_names = set(coords.values())
        self.known = {id(value): name for name, value in values.items()}
        self.known.update(coords)
        self.bounds: dict[str, tuple[int, int] | None] = {}

    def visit(self, dim) -> str:
        """The rendering of *dim*, computed once per node."""
        hit = self.memo.get(id(dim))
        if hit is not None:
            return hit[1]
        result = self.render(dim)
        self.memo[id(dim)] = (dim, result)
        return result

    def render(self, dim) -> str:
        """Render one node by its class."""
        if isinstance(dim, expr.Constant):
            return str(int(dim.value))
        if isinstance(dim, (expr.DimVar, expr.Var)):
            return self.bind(dim)
        if isinstance(dim, expr.Call):
            op = dim_op_type(dim)
            if op is None:
                return self.bind(dim)
            return getattr(self, f"visit_{op.__name__}")(dim)
        if isinstance(dim, bool):
            raise TypeError("ShapeDim must not be bool")
        if isinstance(dim, int):
            return str(dim)
        return self.bind(dim)

    def bind(self, value) -> str:
        """The name *value* has here, naming it into ``values`` if it has none."""
        name = self.known.get(id(value))
        if name is None:
            name = (
                self.param_name(value)
                if self.param_name
                else fresh_name(self.values, self.coord_names)
            )
            self.known[id(value)] = name
            self.values[name] = value
            self.bounds[name] = leaf_bound(value)
        elif name not in self.coord_names and name not in self.bounds:
            self.bounds[name] = named_bound(value)
        return name

    def range_params(self) -> dict[str, tuple[int, int] | None]:
        """Include coordinates as unbounded names while probing affine ranges."""
        return {**dict.fromkeys(self.coord_names), **self.bounds}

    @contextmanager
    def speculative_state(self) -> Iterator[Callable[[], None]]:
        """Roll back every name and range bound while rendering, unless the caller commits."""
        values = self.values.copy()
        known = self.known.copy()
        bounds = self.bounds.copy()
        memo = self.memo.copy()
        committed = False

        def commit() -> None:
            nonlocal committed
            committed = True

        try:
            yield commit
        finally:
            if not committed:
                for state, saved in (
                    (self.values, values),
                    (self.known, known),
                    (self.bounds, bounds),
                    (self.memo, memo),
                ):
                    state.clear()
                    state.update(saved)

    def render_operands(self, dim: expr.Call) -> tuple[str, str]:
        """The renderings of both operands."""
        a, b = dim.args
        return self.visit(a), self.visit(b)

    def visit_DimAdd(self, dim: expr.Call) -> str:
        """``(a + b)``."""
        sa, sb = self.render_operands(dim)
        return f"({sa} + {sb})"

    def visit_DimSub(self, dim: expr.Call) -> str:
        """``(a - b)``."""
        sa, sb = self.render_operands(dim)
        return f"({sa} - {sb})"

    def visit_DimMul(self, dim: expr.Call) -> str:
        """``(a * b)`` for a constant operand, else one parameter bounded by interval arithmetic."""
        with self.speculative_state() as commit:
            sa, sb = self.render_operands(dim)
            range_params = self.range_params()
            ca = constant_value_of(sa, range_params)
            cb = constant_value_of(sb, range_params)
            if ca is not None or cb is not None:
                commit()
                return f"({ca if ca is not None else sa} * {cb if cb is not None else sb})"
            bound = interval_mul(
                bound_of(sa, range_params),
                bound_of(sb, range_params),
            )
        name = self.bind(dim)
        if self.bounds[name] is None:
            self.bounds[name] = bound
        return name

    def divided(self, dim: expr.Call, op: str) -> tuple[str, int]:
        """The rendered dividend and the divisor's value, when it is constant.

        The divisor may be any expression with one value, as for ``DimMul``.
        """
        with self.speculative_state() as commit:
            sa, sb = self.render_operands(dim)
            divisor = constant_value_of(sb, self.range_params())
            if divisor is not None:
                commit()
                return sa, divisor
        raise NotImplementedError(f"{op} by a symbolic divisor has no isl representation")

    def visit_DimFloorDiv(self, dim: expr.Call) -> str:
        """``floor(a/b)`` for a divisor with one constant value."""
        sa, divisor = self.divided(dim, "DimFloorDiv")
        return f"floor({sa}/{divisor})"

    def visit_DimMod(self, dim: expr.Call) -> str:
        """``(a mod b)`` for a divisor with one constant value."""
        sa, divisor = self.divided(dim, "DimMod")
        return f"({sa} mod {divisor})"

    def visit_DimMax(self, dim: expr.Call) -> str:
        """``max(a, b)``."""
        sa, sb = self.render_operands(dim)
        return f"max({sa}, {sb})"

    def visit_DimMin(self, dim: expr.Call) -> str:
        """``min(a, b)``."""
        sa, sb = self.render_operands(dim)
        return f"min({sa}, {sb})"


def render(
    dim, values: IslParamValues, coords: Mapping[int, str], *, bounded: bool, param_name=None
):
    """*dim* as a piecewise affine over *coords*, with or without its parameters' ranges."""
    clash = set(values) & set(coords.values())
    if clash:
        raise ValueError(f"{sorted(clash)} name both a parameter and a coordinate")
    visitor = Renderer(values, coords, param_name)
    node = visitor.visit(dim)
    params = visitor.bounds
    prefix = f"[{', '.join(params)}] -> " if params else ""
    dims = ", ".join(dict.fromkeys(coords.values()))
    body = f"{{ [{dims}] -> [{node}] }}" if coords else f"{{ [{node}] }}"
    pw_aff = isl.pw_aff(prefix + body)
    constraints = [
        f"{bound[0]} <= {name} < {bound[1]}" for name, bound in params.items() if bound is not None
    ]
    if bounded and constraints:
        pw_aff = pw_aff.intersect_params(isl.set(f"{prefix}{{ : {' and '.join(constraints)} }}"))
    return pw_aff


def dimension_to_isl_pw_aff(
    dim, values: IslParamValues, *, coords: Mapping[int, str] | None = None, param_name=None
) -> "isl.pw_aff":
    """*dim* as an isl piecewise affine over *coords*; each new leaf is named into *values*.

    *coords* maps ``id`` of a value that is a coordinate of the space -- a loop's
    induction variable, say -- to that dimension's name, and the result is a
    function on those dimensions in that order. Every other leaf is a parameter:
    one already in *values* keeps its name, by identity, and a new one gets a name
    no other value or coordinate has. What a parameter's value may be is stated
    as a constraint on the result rather than kept on the side.
    """
    return render(dim, values, coords or {}, bounded=True, param_name=param_name)


def raw_dim_call(op_cls, args: tuple):
    scalar = types.TensorType.from_umat_scalar()

    def wrap(value):
        if isinstance(value, bool):
            raise TypeError("bool is not a ShapeDim")
        if isinstance(value, int):
            return expr.Constant(type=scalar, value=value)
        return value

    return expr.Call(type=scalar, target=op_cls(), args=tuple(wrap(arg) for arg in args))


def decode_isl_expr(node, values: IslParamValues):
    if isinstance(node, isl.ast_expr_int):
        return int(node.val().num_si())
    if isinstance(node, isl.ast_expr_id):
        name = node.id().name()
        if name not in values:
            raise ValueError(f"isl identifier {name!r} has no known ShapeDim")
        return values[name]
    if isinstance(node, isl.ast_expr_op):
        op = node.op_type()
        kinds = isl.ast_expr_op_type
        if op == kinds.MINUS:
            return raw_dim_call(expr.DimSub, (0, decode_isl_expr(node.op_arg(0), values)))
        a = decode_isl_expr(node.op_arg(0), values)
        b = decode_isl_expr(node.op_arg(1), values)
        if op == kinds.ADD:
            return raw_dim_call(expr.DimAdd, (a, b))
        if op == kinds.SUB:
            return raw_dim_call(expr.DimSub, (a, b))
        if op == kinds.MUL:
            return raw_dim_call(expr.DimMul, (a, b))
        if op in (kinds.DIV, kinds.PDIV_Q, kinds.FDIV_Q):
            return raw_dim_call(expr.DimFloorDiv, (a, b))
        if op == kinds.PDIV_R:
            return raw_dim_call(expr.DimMod, (a, b))
        if op == kinds.MAX:
            return raw_dim_call(expr.DimMax, (a, b))
        if op == kinds.MIN:
            return raw_dim_call(expr.DimMin, (a, b))
        raise NotImplementedError(f"ast_expr op {op!r} has no ShapeDim decoding")
    raise NotImplementedError(f"unsupported ast_expr type {type(node).__name__}")


def isl_to_dimension(pw_aff: "isl.pw_aff", values: IslParamValues):
    """Decode *pw_aff* into a ShapeDim, reading each parameter's value from *values*."""
    build = isl.ast_build.from_context(pw_aff.domain_space().universe_set())
    return decode_isl_expr(build.expr_from(pw_aff), values)


def normalize(value):
    """Return the sole isl affine normal form for one dimension value."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, expr.DimVar):
        return value
    if not isinstance(value, (expr.Constant, expr.Var, expr.Call)):
        return value
    try:
        values: IslParamValues = {}
        normalized = isl_to_dimension(render(value, values, {}, bounded=False), values)
        return value if normalized == value else normalized
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        return value


def normalize_entries(value):
    """Normalize dimension leaves in a tuple, preserving unchanged objects.

    A dimension is an int, a ``DimVar``, an integer ``Constant`` or a ``Call`` the
    renderer reads as dimension arithmetic: a dimension operation, or an integer
    scalar ``Binary``. Any other value is returned as it is.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, tuple):
        entries = tuple(normalize_entries(entry) for entry in value)
        return value if all(a is b for a, b in zip(entries, value)) else entries
    if (
        isinstance(value, expr.DimVar)
        or (
            isinstance(value, expr.Constant)
            and isinstance(value.value, int)
            and not isinstance(value.value, bool)
        )
        or (isinstance(value, expr.Call) and dim_op_type(value) is not None)
    ):
        return normalize(value)
    return value


def value_range(dim) -> tuple[int, int] | None:
    """Return conservative half-open value bounds ``[lo, hi)`` for *dim*."""
    stored = expr.get_metadata(dim, expr.RangeMetadata) if isinstance(dim, expr.Expr) else None
    if stored is not None:
        return stored.lo, stored.hi
    visitor = Renderer({}, {})
    node = visitor.visit(dim)
    return bound_of(node, visitor.bounds)


def shape_to_isl_set(shape: tuple, values: IslParamValues) -> "isl.set":
    """The coordinates a value of *shape* has, ``0 <= d_i < shape[i]``.

    A number is an extent; a DimVar is a parameter bounded by its envelope; any
    other ``Call`` is one opaque parameter for the whole extent, bounded by
    ``dim_range`` where it has one and unconstrained where it does not -- a
    consumer that needs a bounded set refuses that parameter itself. Parameters
    are named into *values* by identity, so one object is one parameter.
    """
    dims = [f"d{i}" for i in range(len(shape))]
    clash = set(values) & set(dims)
    if clash:
        raise ValueError(f"{sorted(clash)} name both a parameter and a coordinate")
    known = {id(value): name for name, value in values.items()}
    names: dict[str, tuple[int, int] | None] = {}

    def bind(extent, bound) -> str:
        name = known.get(id(extent))
        if name is None:
            name = fresh_name(values, set(dims))
            known[id(extent)] = name
            values[name] = extent
        names.setdefault(name, bound)
        return name

    constraints: list[str] = []
    for i, extent in enumerate(shape):
        if isinstance(extent, bool):
            raise TypeError("ShapeDim must not be bool")
        if isinstance(extent, int):
            constraints.append(f"0 <= d{i} < {extent}")
        elif isinstance(extent, expr.Constant):
            constraints.append(f"0 <= d{i} < {int(extent.value)}")
        elif isinstance(extent, expr.DimVar):
            constraints.append(f"0 <= d{i} < {bind(extent, leaf_bound(extent))}")
        elif isinstance(extent, expr.Call):
            constraints.append(f"0 <= d{i} < {bind(extent, value_range(extent))}")
        else:
            raise TypeError(f"unsupported ShapeDim {type(extent).__name__}")

    constraints += [
        f"{bound[0]} <= {name} < {bound[1]}" for name, bound in names.items() if bound is not None
    ]
    prefix = f"[{', '.join(names)}] -> " if names else ""
    if not shape:
        return isl.set(prefix + "{ [] }")
    return isl.set(prefix + f"{{ [{', '.join(dims)}] : {' and '.join(constraints)} }}")


def mesh_coordinate(mesh, axis: int, values: IslParamValues) -> str:
    """The parameter standing for this unit's coordinate on one mesh axis."""
    for name, value in values.items():
        if (
            isinstance(value, expr.Call)
            and isinstance(value.target, MeshCoord)
            and value.target.mesh is mesh
            and expr.static_dim_value(value.args[0]) == axis
        ):
            return name
    coordinate = expr.Call(
        type=types.TensorType.from_umat_scalar(), target=MeshCoord(mesh=mesh), args=(i64_const(axis),)
    )
    name = fresh_name(values, set())
    values[name] = coordinate
    return name


def layout_to_isl_map(
    shape: tuple,
    layout: types.LayoutBase,
    values: IslParamValues,
    *,
    divided: Callable[[types.ShardLayout], Collection[int]],
) -> "isl.map":
    """Where each logical coordinate of a *shape* value sits among one unit's positions.

    The coordinates regroup row-major onto the flattened layout positions
    ([semantic-analysis §3.1](docs/spec/semantic-analysis.md#31-logical-shape-to-layout-domain)).
    A position cut by the mesh axes ``divided(layer)`` names, layer by layer
    from the innermost, is split into digits -- cutting axes outermost first,
    then the residual: a divided digit is that unit's ``MeshCoord``, a parameter
    named into *values*, and the rest is the position within.
    """
    layers = [types.shard_layout_of(layout) or layout]
    if not isinstance(layers[0], (types.Layout, types.ShardLayout)):
        raise TypeError(f"layout_to_isl_map places a Layout or ShardLayout, not {type(layout).__name__}")
    while isinstance(layers[-1], types.ShardLayout) and (inner := types.shard_layout_of(layers[-1].layout)) is not None:
        layers.append(inner)
    extents = tuple(types.flatten(layers[0].shape))
    coords = [f"c{axis}" for axis in range(len(shape))]
    clash = set(values) & set(coords)
    if clash:
        raise ValueError(f"{sorted(clash)} name both a parameter and a coordinate")
    domain = ", ".join(coords)
    sizes = [expr.static_dim_value(extent) for extent in (*shape, *extents)]
    if None not in sizes:
        logical, held = math.prod(sizes[: len(shape)]), math.prod(sizes[len(shape) :])
        if logical != held:
            raise ValueError(
                f"shape {shape} and layout positions {extents} differ in size, "
                f"{logical} against {held}"
            )
        if logical == 0:
            positions = ", ".join(f"p{index}" for index in range(len(extents)))
            return isl.map(f"{{ [{domain}] -> [{positions}] : 1 = 0 }}")
    axes = int_tuple.mode_axes(int_tuple.compatible_profile(layers[0].shape, tuple(shape)))
    image = ["0"] * len(extents)
    for axis, coord in enumerate(coords):
        positions = [p for p, target in enumerate(axes)
                     if target == axis and expr.static_dim_value(extents[p]) != 1]
        inner = 1
        for position in reversed(positions):
            term = coord if inner == 1 else f"floor({coord}/{inner})"
            if position == positions[0]:
                image[position] = term
                break
            extent = expr.static_dim_value(extents[position])
            if extent is None:
                raise ValueError(
                    f"layout position {position} has extent {extents[position]!r}; "
                    "dividing a coordinate by it is not affine"
                )
            image[position] = f"({term}) mod {extent}"
            inner *= extent
    cutting: dict[int, list[tuple[types.ShardLayout, int, bool]]] = {}
    for layer in reversed(layers):
        if not isinstance(layer, types.ShardLayout):
            continue
        mesh_extents = types.flatten(layer.mesh.layout).shape
        cuts = divided(layer)
        for mesh_axis, attr in enumerate(layer.attrs):
            if not isinstance(attr, types.Split):
                continue
            if not 0 <= attr.axis < len(extents) or mesh_axis >= len(mesh_extents):
                raise ValueError(
                    f"{layer!r} splits a layout position or mesh axis it does not have"
                )
            cutting.setdefault(attr.axis, []).append((layer, mesh_axis, mesh_axis in cuts))
    params: list[str] = []
    guards: list[str] = []
    for position, cuts in cutting.items():
        if not any(cut for *_, cut in cuts):
            continue
        whole = expr.static_dim_value(extents[position])
        parts = [expr.static_dim_value(types.flatten(layer.mesh.layout).shape[a]) for layer, a, _ in cuts]
        if whole is None or any(part is None or part <= 0 for part in parts):
            raise ValueError(
                f"layout position {position} of {shape} is split by mesh axes; placing "
                "one unit needs the position and every mesh extent cutting it static"
            )
        if whole % math.prod(parts):
            raise ValueError(
                f"layout position {position} extent {whole} is not divisible by its mesh "
                f"extents {tuple(parts)}"
            )
        residual = whole // math.prod(parts)
        coordinate = image[position]
        stride = whole
        kept: list[tuple[str, int]] = []
        for order, ((layer, mesh_axis, cut), part) in enumerate(zip(cuts, parts)):
            stride //= part
            digit = f"floor(({coordinate})/{stride})"
            if order:
                digit = f"({digit}) mod {part}"
            if cut:
                name = mesh_coordinate(layer.mesh, mesh_axis, values)
                if name not in params:
                    params.append(name)
                    guards.append(f"0 <= {name} < {part}")
                guards.append(f"{digit} = {name}")
            else:
                kept.append((digit, part))
        terms = [f"({coordinate}) mod {residual}"] if residual != 1 else []
        scale = residual
        for digit, part in reversed(kept):
            terms.append(digit if scale == 1 else f"{scale} * ({digit})")
            scale *= part
        image[position] = " + ".join(reversed(terms)) or "0"
    prefix = f"[{', '.join(params)}] -> " if params else ""
    where = f" : {' and '.join(guards)}" if guards else ""
    return isl.map(f"{prefix}{{ [{domain}] -> [{', '.join(image)}]{where} }}")


services.registry.register(services.normalize_dim, normalize_entries)
services.registry.register(services.dim_range, value_range)
