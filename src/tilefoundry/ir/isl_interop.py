"""Interoperation between dimension and shape IR values and isl.

Dimension definitions and construction stay in :mod:`tilefoundry.ir.types.dim`;
pure isl operations stay in :mod:`tilefoundry.utils.isl_utils`. This module owns
the boundary between those layers: rendering, decoding, normalization, value
ranges, and shape domains.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager

import isl

from tilefoundry.ir.core.expr import Call, Constant, Expr, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.metadata import RangeMetadata, get_metadata

from .types.dim import (
    _DIM_OP_TYPES,
    DimAdd,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
    simplify_dim,
)
from .types.dtype import IntegerDType
from .types.int_tuple import flatten as flatten_tuple
from .types.layout import LayoutBase, flatten
from .types.shard_layout import ShardLayout, Split, shard_layout_of
from .types.tensor_type import TensorType
from .types.utils import i64_const, static_dim_value

IslParamValues = dict[str, Expr]
"""isl parameter name -> the IR value it stands for; the one dictionary every conversion shares."""

_COUNTER = itertools.count()

_INTEGER_BINARY_DIM_OP = {
    BinaryKind.ADD: DimAdd,
    BinaryKind.SUB: DimSub,
    BinaryKind.MUL: DimMul,
    BinaryKind.FLOOR_DIV: DimFloorDiv,
    BinaryKind.MOD: DimMod,
    BinaryKind.MIN: DimMin,
    BinaryKind.MAX: DimMax,
}


def _is_const(node) -> bool:
    if isinstance(node, bool):
        return False
    return isinstance(node, int) or isinstance(node, Constant)


def _fresh_name(values: IslParamValues, taken: set[str]) -> str:
    """A parameter name no value in *values* and no coordinate in *taken* has.

    The name carries no meaning, so it is a fixed prefix and a number rather than
    anything read off the value: what a parameter stands for is in *values*.
    """
    while True:
        name = f"p{next(_COUNTER)}"
        if name not in values and name not in taken:
            return name


def _leaf_bound(value) -> tuple[int, int] | None:
    """The half-open range a leaf states for itself, if it states one."""
    if isinstance(value, DimVar):
        return value.lo, value.hi + 1
    stored = get_metadata(value, RangeMetadata) if isinstance(value, Expr) else None
    return None if stored is None else (stored.lo, stored.hi)


def _named_bound(value) -> tuple[int, int] | None:
    """The range of a value some earlier conversion already named."""
    try:
        return dim_range(value)
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        return None


def _pw_aff(expr: str, params: dict[str, tuple[int, int] | None]) -> isl.pw_aff:
    prefix = f"[{', '.join(params)}] -> " if params else ""
    return isl.pw_aff(prefix + f"{{ [{expr}] }}")


def _bound_of(
    expr: str,
    params: dict[str, tuple[int, int] | None],
) -> tuple[int, int] | None:
    """Return finite half-open bounds for one rendered expression, if available."""
    pw_aff = _pw_aff(expr, params)
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


def _constant_value_of(
    expr: str,
    params: dict[str, tuple[int, int] | None],
) -> int | None:
    """Return an integer only when *expr* is constant without parameter bounds."""
    pw_aff = _pw_aff(expr, params)
    minimum = pw_aff.min_val()
    maximum = pw_aff.max_val()
    if not minimum.is_int() or not maximum.is_int():
        return None
    lo = int(minimum.num_si())
    return lo if lo == int(maximum.num_si()) else None


def _interval_mul(
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


def _dim_op_type(dim: Call) -> type | None:
    op = type(dim.target)
    if op in _DIM_OP_TYPES:
        return op
    kind = getattr(dim.target, "kind", None)
    if not (
        isinstance(dim.type, TensorType)
        and dim.type.shape == ()
        and isinstance(dim.type.dtype, IntegerDType)
        and isinstance(kind, BinaryKind)
    ):
        return None
    return _INTEGER_BINARY_DIM_OP.get(kind)


_DIM_VISITOR_TYPE = None


def _dim_visitor_type():
    global _DIM_VISITOR_TYPE
    if _DIM_VISITOR_TYPE is None:
        from tilefoundry.ir.visitor import ExprVisitor  # noqa: PLC0415

        class _DimVisitor(ExprVisitor[str]):
            def __init__(self, values: IslParamValues, coords: Mapping[int, str]) -> None:
                super().__init__()
                self.values = values
                self.coord_names = set(coords.values())
                self.known = {id(value): name for name, value in values.items()}
                self.known.update(coords)
                self.bounds: dict[str, tuple[int, int] | None] = {}

            def bind(self, value) -> str:
                """The name *value* has here, naming it into ``values`` if it has none."""
                name = self.known.get(id(value))
                if name is None:
                    name = _fresh_name(self.values, self.coord_names)
                    self.known[id(value)] = name
                    self.values[name] = value
                    self.bounds[name] = _leaf_bound(value)
                elif name not in self.coord_names and name not in self.bounds:
                    self.bounds[name] = _named_bound(value)
                return name

            def _range_params(self) -> dict[str, tuple[int, int] | None]:
                """Include coordinates as unbounded names while probing affine ranges."""
                return {**dict.fromkeys(self.coord_names), **self.bounds}

            @contextmanager
            def _speculative_state(self) -> Iterator[Callable[[], None]]:
                values = self.values.copy()
                known = self.known.copy()
                bounds = self.bounds.copy()
                memo = self._memo.copy()
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
                            (self._memo, memo),
                        ):
                            state.clear()
                            state.update(saved)

            def _render_operands(self, dim: Call, ctx) -> tuple[str, str]:
                a, b = dim.args
                return self.visit(a, ctx), self.visit(b, ctx)

            def visit_Constant(self, dim: Constant, ctx=None) -> str:
                return str(int(dim.value))

            def visit_DimVar(self, dim: DimVar, ctx=None) -> str:
                return self.bind(dim)

            def visit_Var(self, dim: Var, ctx=None) -> str:
                return self.bind(dim)

            def visit_Call(self, dim: Call, ctx=None) -> str:
                op = _dim_op_type(dim)
                if op is None:
                    return self.bind(dim)
                return getattr(self, f"visit_{op.__name__}")(dim, ctx)

            def visit_DimAdd(self, dim: Call, ctx=None) -> str:
                sa, sb = self._render_operands(dim, ctx)
                return f"({sa} + {sb})"

            def visit_DimSub(self, dim: Call, ctx=None) -> str:
                sa, sb = self._render_operands(dim, ctx)
                return f"({sa} - {sb})"

            def visit_DimMul(self, dim: Call, ctx=None) -> str:
                with self._speculative_state() as commit:
                    sa, sb = self._render_operands(dim, ctx)
                    range_params = self._range_params()
                    ca = _constant_value_of(sa, range_params)
                    cb = _constant_value_of(sb, range_params)
                    if ca is not None or cb is not None:
                        commit()
                        return f"({ca if ca is not None else sa} * {cb if cb is not None else sb})"
                    bound = _interval_mul(
                        _bound_of(sa, range_params),
                        _bound_of(sb, range_params),
                    )
                name = self.bind(dim)
                if self.bounds[name] is None:
                    self.bounds[name] = bound
                return name

            def visit_DimFloorDiv(self, dim: Call, ctx=None) -> str:
                _, b = dim.args
                if not _is_const(b):
                    raise NotImplementedError(
                        "DimFloorDiv by a symbolic divisor has no isl representation"
                    )
                sa, sb = self._render_operands(dim, ctx)
                return f"floor({sa}/{sb})"

            def visit_DimMod(self, dim: Call, ctx=None) -> str:
                _, b = dim.args
                if not _is_const(b):
                    raise NotImplementedError(
                        "DimMod by a symbolic divisor has no isl representation"
                    )
                sa, sb = self._render_operands(dim, ctx)
                return f"({sa} mod {sb})"

            def visit_DimMax(self, dim: Call, ctx=None) -> str:
                sa, sb = self._render_operands(dim, ctx)
                return f"max({sa}, {sb})"

            def visit_DimMin(self, dim: Call, ctx=None) -> str:
                sa, sb = self._render_operands(dim, ctx)
                return f"min({sa}, {sb})"

            def default_visit(self, value, ctx=None) -> str:
                if isinstance(value, bool):
                    raise TypeError("ShapeDim must not be bool")
                if isinstance(value, int):
                    return str(value)
                return self.bind(value)

        _DIM_VISITOR_TYPE = _DimVisitor
    return _DIM_VISITOR_TYPE


def _render(dim, values: IslParamValues, coords: Mapping[int, str], *, bounded: bool):
    """*dim* as a piecewise affine over *coords*, with or without its parameters' ranges."""
    clash = set(values) & set(coords.values())
    if clash:
        raise ValueError(f"{sorted(clash)} name both a parameter and a coordinate")
    visitor = _dim_visitor_type()(values, coords)
    expr = visitor.visit(dim)
    params = visitor.bounds
    prefix = f"[{', '.join(params)}] -> " if params else ""
    dims = ", ".join(dict.fromkeys(coords.values()))
    body = f"{{ [{dims}] -> [{expr}] }}" if coords else f"{{ [{expr}] }}"
    pw_aff = isl.pw_aff(prefix + body)
    constraints = [
        f"{bound[0]} <= {name} < {bound[1]}" for name, bound in params.items() if bound is not None
    ]
    if bounded and constraints:
        pw_aff = pw_aff.intersect_params(isl.set(f"{prefix}{{ : {' and '.join(constraints)} }}"))
    return pw_aff


def dim_to_isl_pw_aff(
    dim, values: IslParamValues, *, coords: Mapping[int, str] | None = None
) -> "isl.pw_aff":
    """*dim* as an isl piecewise affine over *coords*; each new leaf is named into *values*.

    *coords* maps ``id`` of a value that is a coordinate of the space -- a loop's
    induction variable, say -- to that dimension's name, and the result is a
    function on those dimensions in that order. Every other leaf is a parameter:
    one already in *values* keeps its name, by identity, and a new one gets a name
    no other value or coordinate has. What a parameter's value may be is stated
    as a constraint on the result rather than kept on the side.
    """
    return _render(dim, values, coords or {}, bounded=True)


def _raw_dim_call(op_cls, args: tuple):
    scalar = TensorType.umat_scalar()

    def wrap(value):
        if isinstance(value, bool):
            raise TypeError("bool is not a ShapeDim")
        if isinstance(value, int):
            return Constant(type=scalar, value=value)
        return value

    return Call(type=scalar, target=op_cls(), args=tuple(wrap(arg) for arg in args))


def _visit_isl_expr(expr, values: IslParamValues):
    if isinstance(expr, isl.ast_expr_int):
        return int(expr.val().num_si())
    if isinstance(expr, isl.ast_expr_id):
        name = expr.id().name()
        if name not in values:
            raise ValueError(f"isl identifier {name!r} has no known ShapeDim")
        return values[name]
    if isinstance(expr, isl.ast_expr_op):
        op = expr.op_type()
        Op = isl.ast_expr_op_type
        if op == Op.MINUS:
            return _raw_dim_call(DimSub, (0, _visit_isl_expr(expr.op_arg(0), values)))
        a = _visit_isl_expr(expr.op_arg(0), values)
        b = _visit_isl_expr(expr.op_arg(1), values)
        if op == Op.ADD:
            return _raw_dim_call(DimAdd, (a, b))
        if op == Op.SUB:
            return _raw_dim_call(DimSub, (a, b))
        if op == Op.MUL:
            return _raw_dim_call(DimMul, (a, b))
        if op in (Op.DIV, Op.PDIV_Q, Op.FDIV_Q):
            return _raw_dim_call(DimFloorDiv, (a, b))
        if op == Op.PDIV_R:
            return _raw_dim_call(DimMod, (a, b))
        if op == Op.MAX:
            return _raw_dim_call(DimMax, (a, b))
        if op == Op.MIN:
            return _raw_dim_call(DimMin, (a, b))
        raise NotImplementedError(f"ast_expr op {op!r} has no ShapeDim decoding")
    raise NotImplementedError(f"unsupported ast_expr type {type(expr).__name__}")


def isl_to_dim(pw_aff: "isl.pw_aff", values: IslParamValues):
    """Decode *pw_aff* into a ShapeDim, reading each parameter's value from *values*."""
    build = isl.ast_build.from_context(pw_aff.domain_space().universe_set())
    return _visit_isl_expr(build.expr_from(pw_aff), values)


def normalize_dim(value):
    """Return the sole isl affine normal form for one dimension value."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, DimVar):
        return value
    if not isinstance(value, (Constant, Var, Call)):
        return value
    try:
        values: IslParamValues = {}
        normalized = isl_to_dim(_render(value, values, {}, bounded=False), values)
        return value if normalized == value else normalized
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        return value


def normalize_dim_entries(value):
    """Normalize dimension leaves in a tuple, preserving unchanged objects."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, tuple):
        entries = tuple(normalize_dim_entries(entry) for entry in value)
        return value if all(a is b for a, b in zip(entries, value)) else entries
    if (
        isinstance(value, DimVar)
        or (
            isinstance(value, Constant)
            and isinstance(value.value, int)
            and not isinstance(value.value, bool)
        )
        or (isinstance(value, Call) and isinstance(value.target, _DIM_OP_TYPES))
    ):
        return normalize_dim(value)
    return value


def dim_range(dim) -> tuple[int, int] | None:
    """Return conservative half-open value bounds ``[lo, hi)`` for *dim*."""
    stored = get_metadata(dim, RangeMetadata) if isinstance(dim, Expr) else None
    if stored is not None:
        return stored.lo, stored.hi
    visitor = _dim_visitor_type()({}, {})
    expr = visitor.visit(dim)
    return _bound_of(expr, visitor.bounds)


def dim_at_most(a, b) -> bool:
    """Prove ``a <= b`` from the conservative range of their difference."""
    if isinstance(a, int) and isinstance(b, int):
        return a <= b
    bounds = dim_range(b - a)
    return bounds is not None and bounds[0] >= 0


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
            name = _fresh_name(values, set(dims))
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
        elif isinstance(extent, Constant):
            constraints.append(f"0 <= d{i} < {int(extent.value)}")
        elif isinstance(extent, DimVar):
            constraints.append(f"0 <= d{i} < {bind(extent, _leaf_bound(extent))}")
        elif isinstance(extent, Call):
            constraints.append(f"0 <= d{i} < {bind(extent, dim_range(extent))}")
        else:
            raise TypeError(f"unsupported ShapeDim {type(extent).__name__}")

    constraints += [
        f"{bound[0]} <= {name} < {bound[1]}" for name, bound in names.items() if bound is not None
    ]
    prefix = f"[{', '.join(names)}] -> " if names else ""
    if not shape:
        return isl.set(prefix + "{ [] }")
    return isl.set(prefix + f"{{ [{', '.join(dims)}] : {' and '.join(constraints)} }}")


def _same_extent(a, b) -> bool:
    """Whether two extents are provably equal."""
    if a is b or a == b:
        return True
    if static_dim_value(a) is not None and static_dim_value(b) is not None:
        return False
    try:
        return normalize_dim(simplify_dim(DimSub, (a, b))) == 0
    except (TypeError, ValueError, NotImplementedError):
        return False


def _congruent_groups(shape: tuple, extents: tuple) -> list[list[int]] | None:
    """Which layout positions each logical axis owns, when they line up in order.

    Each logical axis takes the next positions whose product is provably its
    extent. Anything else -- an axis spread over a boundary, or positions left
    over that hold more than one -- is not congruent, and None says so.
    """
    groups: list[list[int]] = []
    position = 0
    for extent in shape:
        group: list[int] = []
        product = 1
        while not _same_extent(product, extent):
            if position >= len(extents):
                return None
            held = extents[position]
            product = held if product == 1 else simplify_dim(DimMul, (product, held))
            if static_dim_value(product) is not None and static_dim_value(extent) is not None:
                product = static_dim_value(product)
                if product > static_dim_value(extent):
                    return None
            group.append(position)
            position += 1
        groups.append(group)
    if any(static_dim_value(extent) != 1 for extent in extents[position:]):
        return None
    return groups


def _regrouped(shape: tuple, extents: tuple, coords: list[str]) -> list[str]:
    """Each layout position's coordinate, by the row-major regroup of *coords*.

    A position of extent 1 is 0. The outermost of the rest is not reduced
    modulo its extent, so a coordinate past the value is not folded back onto
    one inside it; every other divisor and modulus has to be a number.
    """
    image = ["0"] * len(extents)
    groups = _congruent_groups(shape, extents)
    if groups is not None:
        runs = list(zip(coords, groups, strict=True))
    else:
        whole = [static_dim_value(extent) for extent in shape]
        if None in whole or any(static_dim_value(extent) is None for extent in extents):
            raise ValueError(
                f"shape {shape} regroups onto layout positions {extents} across axes, "
                "which needs static extents on both sides"
            )
        terms = []
        stride = 1
        for coord, extent in reversed(list(zip(coords, whole))):
            terms.append(coord if stride == 1 else f"{stride} * {coord}")
            stride *= extent
        runs = [(f"({' + '.join(reversed(terms)) or '0'})", list(range(len(extents))))]
    for coord, group in runs:
        held = [position for position in group if static_dim_value(extents[position]) != 1]
        inner = 1
        for position in reversed(held):
            term = coord if inner == 1 else f"floor({coord}/{inner})"
            if position == held[0]:
                image[position] = term
                break
            extent = static_dim_value(extents[position])
            if extent is None:
                raise ValueError(
                    f"layout position {position} has extent {extents[position]!r} and a "
                    "position outside it; dividing a coordinate by it is not affine"
                )
            image[position] = f"({term}) mod {extent}"
            inner *= extent
    return image


def _mesh_coordinate(mesh, axis: int, values: IslParamValues) -> str:
    """The parameter standing for this unit's coordinate on one mesh axis."""
    from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord  # noqa: PLC0415 - cycle guard

    for name, value in values.items():
        if (
            isinstance(value, Call)
            and isinstance(value.target, MeshCoord)
            and value.target.mesh is mesh
            and static_dim_value(value.args[0]) == axis
        ):
            return name
    coordinate = Call(
        type=TensorType.umat_scalar(), target=MeshCoord(mesh=mesh), args=(i64_const(axis),)
    )
    name = _fresh_name(values, set())
    values[name] = coordinate
    return name


def layout_to_isl_map(
    shape: tuple,
    layout: LayoutBase,
    values: IslParamValues,
    *,
    divided: Callable[[ShardLayout], Collection[int]],
) -> "isl.map":
    """Where each logical coordinate of a *shape* value sits among one unit's positions.

    The coordinates regroup row-major onto the flattened layout positions
    ([semantic-analysis §3.1](docs/spec/semantic-analysis.md#31-logical-shape-to-layout-domain)).
    A position cut by the mesh axes ``divided(layer)`` names, layer by layer
    from the innermost, is split into digits -- cutting axes outermost first,
    then the residual: a divided digit is that unit's ``MeshCoord``, a parameter
    named into *values*, and the rest is the position within.
    """
    layers = [shard_layout_of(layout)]
    if layers[0] is None:
        raise TypeError(f"layout_to_isl_map places a ShardLayout, not {type(layout).__name__}")
    while (inner := shard_layout_of(layers[-1].layout)) is not None:
        layers.append(inner)
    extents = tuple(flatten_tuple(layers[0].shape))
    coords = [f"c{axis}" for axis in range(len(shape))]
    clash = set(values) & set(coords)
    if clash:
        raise ValueError(f"{sorted(clash)} name both a parameter and a coordinate")
    domain = ", ".join(coords)
    sizes = [static_dim_value(extent) for extent in (*shape, *extents)]
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
    image = _regrouped(tuple(shape), extents, coords)
    cutting: dict[int, list[tuple[ShardLayout, int, bool]]] = {}
    for layer in reversed(layers):
        mesh_extents = flatten(layer.mesh.layout).shape
        cuts = divided(layer)
        for mesh_axis, attr in enumerate(layer.attrs):
            if not isinstance(attr, Split):
                continue
            if not 0 <= attr.axis < len(extents) or mesh_axis >= len(mesh_extents):
                raise ValueError(
                    f"{layer!r} splits a layout position or mesh axis it does not have"
                )
            cutting.setdefault(attr.axis, []).append((layer, mesh_axis, mesh_axis in cuts))
    params: list[str] = []
    guards: list[str] = []
    for position, cuts in cutting.items():
        if not any(cut for _layer, _axis, cut in cuts):
            continue
        whole = static_dim_value(extents[position])
        parts = [static_dim_value(flatten(layer.mesh.layout).shape[a]) for layer, a, _ in cuts]
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
                name = _mesh_coordinate(layer.mesh, mesh_axis, values)
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


__all__ = [
    "IslParamValues",
    "dim_at_most",
    "dim_range",
    "dim_to_isl_pw_aff",
    "isl_to_dim",
    "layout_to_isl_map",
    "normalize_dim",
    "normalize_dim_entries",
    "shape_to_isl_set",
]
