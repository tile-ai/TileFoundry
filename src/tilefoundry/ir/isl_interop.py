"""Interoperation between dimension and shape IR values and isl.

Dimension definitions and construction stay in :mod:`tilefoundry.ir.types.dim`;
pure isl operations stay in :mod:`tilefoundry.utils.isl_utils`. This module owns
the boundary between those layers: rendering, decoding, normalization, value
ranges, and shape domains.
"""

from __future__ import annotations

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
)
from .types.dtype import IntegerDType
from .types.tensor_type import TensorType

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


def _bind_param(
    value,
    params: dict[str, tuple[int, int] | None],
    param_map: dict[str, object] | None,
    identities: dict[int, str] | None,
) -> str:
    if isinstance(value, DimVar):
        name = value.name
        bound = (value.lo, value.hi)
        previous = params.get(name)
        if previous is not None and previous != bound:
            raise ValueError(f"DimVar {name!r} used with conflicting bounds {previous} vs {bound}")
    else:
        stored = get_metadata(value, RangeMetadata) if isinstance(value, Expr) else None
        if identities is None:
            raise TypeError(f"unsupported ShapeDim {type(value).__name__}")
        key = id(value)
        known = identities.get(key)
        if known is not None:
            return known
        index = len(identities)
        name = f"__tf_runtime_{index}"
        while name in params:
            index += 1
            name = f"__tf_runtime_{index}"
        identities[key] = name
        bound = (stored.lo, stored.hi) if stored is not None else None

    params[name] = bound
    if param_map is not None:
        previous_value = param_map.get(name)
        if previous_value is not None and previous_value is not value:
            raise ValueError(f"isl parameter {name!r} maps to multiple dimension values")
        param_map[name] = value
    return name


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


_DIM_VISITOR_TYPE = None


def _dim_visitor_type():
    global _DIM_VISITOR_TYPE
    if _DIM_VISITOR_TYPE is None:
        from tilefoundry.ir.visitor import ExprVisitor  # noqa: PLC0415

        class _DimVisitor(ExprVisitor[str]):
            def __init__(self, params, param_map, identities) -> None:
                super().__init__()
                self.params = params
                self.param_map = param_map
                self.identities = identities

            def _snapshot(self):
                return (
                    self.params.copy(),
                    None if self.param_map is None else self.param_map.copy(),
                    None if self.identities is None else self.identities.copy(),
                )

            def _restore(self, snapshot) -> None:
                params, param_map, identities = snapshot
                self.params.clear()
                self.params.update(params)
                if self.param_map is not None:
                    self.param_map.clear()
                    self.param_map.update(param_map)
                if self.identities is not None:
                    self.identities.clear()
                    self.identities.update(identities)

            def visit_Constant(self, dim: Constant, ctx=None) -> str:
                return str(int(dim.value))

            def visit_DimVar(self, dim: DimVar, ctx=None) -> str:
                return _bind_param(dim, self.params, self.param_map, self.identities)

            def visit_Var(self, dim: Var, ctx=None) -> str:
                return _bind_param(dim, self.params, self.param_map, self.identities)

            def visit_Call(self, dim: Call, ctx=None) -> str:
                op = type(dim.target)
                if op not in _DIM_OP_TYPES:
                    kind = getattr(dim.target, "kind", None)
                    if not (
                        isinstance(dim.type, TensorType)
                        and dim.type.shape == ()
                        and isinstance(dim.type.dtype, IntegerDType)
                        and isinstance(kind, BinaryKind)
                        and kind in _INTEGER_BINARY_DIM_OP
                    ):
                        return _bind_param(dim, self.params, self.param_map, self.identities)
                    op = _INTEGER_BINARY_DIM_OP[kind]
                a, b = dim.args
                if op in (DimFloorDiv, DimMod) and not _is_const(b):
                    raise NotImplementedError(
                        f"{op.__name__} by a symbolic divisor has no isl representation"
                    )
                snapshot = self._snapshot() if op is DimMul else None
                sa = self.visit(a, ctx)
                sb = self.visit(b, ctx)
                if op is DimMul:
                    ca = _constant_value_of(sa, self.params)
                    cb = _constant_value_of(sb, self.params)
                    if ca is not None or cb is not None:
                        return f"({ca if ca is not None else sa} * {cb if cb is not None else sb})"
                    bound = _interval_mul(
                        _bound_of(sa, self.params),
                        _bound_of(sb, self.params),
                    )
                    self._restore(snapshot)
                    name = _bind_param(dim, self.params, self.param_map, self.identities)
                    if self.params[name] is None:
                        self.params[name] = bound
                    return name
                if op is DimAdd:
                    return f"({sa} + {sb})"
                if op is DimSub:
                    return f"({sa} - {sb})"
                if op is DimFloorDiv:
                    return f"floor({sa}/{sb})"
                if op is DimMod:
                    return f"({sa} mod {sb})"
                if op is DimMax:
                    return f"max({sa}, {sb})"
                if op is DimMin:
                    return f"min({sa}, {sb})"
                raise AssertionError(f"unhandled dim op {op.__name__}")

            def default_visit(self, value, ctx=None) -> str:
                if isinstance(value, bool):
                    raise TypeError("ShapeDim must not be bool")
                if isinstance(value, int):
                    return str(value)
                if self.identities is None:
                    raise TypeError(f"unsupported ShapeDim {type(value).__name__}")
                return _bind_param(value, self.params, self.param_map, self.identities)

        _DIM_VISITOR_TYPE = _DimVisitor
    return _DIM_VISITOR_TYPE


def dim_to_isl_expr(
    dim,
    params: dict[str, tuple[int, int] | None],
    *,
    param_map: dict[str, object] | None = None,
    identities: dict[int, str] | None = None,
) -> str:
    """Render *dim* as an isl expression and register its leaf parameters."""
    return _dim_visitor_type()(params, param_map, identities).visit(dim)


def _raw_dim_call(op_cls, args: tuple):
    scalar = TensorType.umat_scalar()

    def wrap(value):
        if isinstance(value, bool):
            raise TypeError("bool is not a ShapeDim")
        if isinstance(value, int):
            return Constant(type=scalar, value=value)
        return value

    return Call(type=scalar, target=op_cls(), args=tuple(wrap(arg) for arg in args))


def _visit_isl_expr(expr, param_map: dict[str, object]):
    if isinstance(expr, isl.ast_expr_int):
        return int(expr.val().num_si())
    if isinstance(expr, isl.ast_expr_id):
        name = expr.id().name()
        if name not in param_map:
            raise ValueError(f"isl identifier {name!r} has no known ShapeDim")
        return param_map[name]
    if isinstance(expr, isl.ast_expr_op):
        op = expr.op_type()
        Op = isl.ast_expr_op_type
        if op == Op.MINUS:
            return _raw_dim_call(DimSub, (0, _visit_isl_expr(expr.op_arg(0), param_map)))
        a = _visit_isl_expr(expr.op_arg(0), param_map)
        b = _visit_isl_expr(expr.op_arg(1), param_map)
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


def isl_to_dim(pw_aff: "isl.pw_aff", param_map: dict[str, object]):
    """Decode *pw_aff* into a ShapeDim using *param_map* for identifiers."""
    build = isl.ast_build.from_context(pw_aff.domain_space().universe_set())
    return _visit_isl_expr(build.expr_from(pw_aff), param_map)


def normalize_dim(value):
    """Return the sole isl affine normal form for one dimension value."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, DimVar):
        return value
    if not isinstance(value, (Constant, Var, Call)):
        return value
    try:
        params: dict[str, tuple[int, int] | None] = {}
        param_map: dict[str, object] = {}
        expr = dim_to_isl_expr(
            value,
            params,
            param_map=param_map,
            identities={},
        )
        prefix = f"[{', '.join(params)}] -> " if params else ""
        normalized = isl_to_dim(isl.pw_aff(prefix + f"{{ [{expr}] }}"), param_map)
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
    params: dict[str, tuple[int, int] | None] = {}
    expr = _dim_visitor_type()(params, None, {}).visit(dim)
    return _bound_of(expr, params)


def shape_to_isl_domain(extents: tuple) -> tuple[isl.set, dict[str, object]]:
    """Build an iteration domain and its isl-parameter ShapeDim map.

    A ``Call`` without a value range becomes an unconstrained parameter. Consumers
    that require a bounded domain must reject that parameter explicitly.
    """
    param_map: dict[str, object] = {}
    bounds: dict[str, tuple[int, int] | None] = {}
    seen: dict = {}
    names: list[str] = []

    def bind(name: str, dim, bound: tuple[int, int] | None) -> None:
        previous = bounds.get(name)
        if previous is not None and previous != bound:
            raise ValueError(
                f"isl parameter {name!r} used with conflicting bounds {previous} vs {bound}"
            )
        if name not in bounds:
            names.append(name)
        bounds[name] = bound
        param_map[name] = dim

    dims = [f"d{i}" for i in range(len(extents))]
    constraints: list[str] = []
    for i, extent in enumerate(extents):
        if isinstance(extent, bool):
            raise TypeError("ShapeDim must not be bool")
        if isinstance(extent, int):
            constraints.append(f"0 <= d{i} < {extent}")
        elif isinstance(extent, Constant):
            constraints.append(f"0 <= d{i} < {int(extent.value)}")
        elif isinstance(extent, DimVar):
            bind(extent.name, extent, (extent.lo, extent.hi))
            constraints.append(f"0 <= d{i} < {extent.name}")
        elif isinstance(extent, Call):
            name = seen.get(extent)
            if name is None:
                name = f"D{i}"
                seen[extent] = name
            bind(name, extent, dim_range(extent))
            constraints.append(f"0 <= d{i} < {name}")
        else:
            raise TypeError(f"unsupported ShapeDim {type(extent).__name__}")

    constraints += [
        f"{bound[0]} <= {name} < {bound[1]}"
        for name in names
        if (bound := bounds[name]) is not None
    ]
    prefix = f"[{', '.join(names)}] -> " if names else ""
    if not dims:
        return isl.set(prefix + "{ [] }"), param_map
    body = f"{{ [{', '.join(dims)}] : {' and '.join(constraints)} }}"
    return isl.set(prefix + body), param_map


def index_set(shape: tuple) -> isl.set | None:
    """Return the coordinate set for a non-negative literal shape."""
    if any(
        not isinstance(extent, int) or isinstance(extent, bool) or extent < 0 for extent in shape
    ):
        return None
    domain, _ = shape_to_isl_domain(shape)
    return domain


__all__ = [
    "dim_to_isl_expr",
    "dim_range",
    "index_set",
    "isl_to_dim",
    "normalize_dim",
    "normalize_dim_entries",
    "shape_to_isl_domain",
]
