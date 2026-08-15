"""The single ShapeDim <-> isl bridge and normalization authority."""

from __future__ import annotations

import isl

from tilefoundry.ir.core.expr import Call, Constant, Var

from .dim import (
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
from .tensor_type import TensorType


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
            raise ValueError(
                f"DimVar {name!r} used with conflicting bounds {previous} vs {bound}"
            )
    elif identities is not None:
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
        bound = None
    else:
        raise TypeError(f"unsupported ShapeDim {type(value).__name__}")

    params[name] = bound
    if param_map is not None:
        previous_value = param_map.get(name)
        if previous_value is not None and previous_value is not value:
            raise ValueError(f"isl parameter {name!r} maps to multiple dimension values")
        param_map[name] = value
    return name


_RANGE_EXPR_VISITOR_TYPE = None
_DIM_RANGE_VISITOR_TYPE = None


def _range_expr_visitor_type():
    global _RANGE_EXPR_VISITOR_TYPE
    if _RANGE_EXPR_VISITOR_TYPE is None:
        from tilefoundry.ir.visitor import ExprVisitor  # noqa: PLC0415

        class _RangeExprVisitor(ExprVisitor[str]):
            def __init__(self, params, param_map, identities) -> None:
                super().__init__()
                self.params = params
                self.param_map = param_map
                self.identities = identities

            def visit_Constant(self, dim: Constant) -> str:
                return str(int(dim.value))

            def visit_DimVar(self, dim: DimVar) -> str:
                return _bind_param(dim, self.params, self.param_map, self.identities)

            def visit_Var(self, dim: Var) -> str:
                return _bind_param(dim, self.params, self.param_map, self.identities)

            def visit_Call(self, dim: Call) -> str:
                op = type(dim.target)
                if op not in _DIM_OP_TYPES:
                    return _bind_param(dim, self.params, self.param_map, self.identities)
                a, b = dim.args
                if op is DimMul and not (_is_const(a) or _is_const(b)):
                    name = _bind_param(dim, self.params, self.param_map, self.identities)
                    if self.params[name] is None:
                        self.params[name] = dim_range(dim)
                    return name
                if op in (DimFloorDiv, DimMod) and not _is_const(b):
                    raise NotImplementedError(
                        f"{op.__name__} by a symbolic divisor has no isl representation"
                    )
                sa = self.visit(a)
                sb = self.visit(b)
                if op is DimAdd:
                    return f"({sa} + {sb})"
                if op is DimSub:
                    return f"({sa} - {sb})"
                if op is DimMul:
                    return f"({sa} * {sb})"
                if op is DimFloorDiv:
                    return f"floor({sa}/{sb})"
                if op is DimMod:
                    return f"({sa} mod {sb})"
                if op is DimMax:
                    return f"max({sa}, {sb})"
                if op is DimMin:
                    return f"min({sa}, {sb})"
                raise AssertionError(f"unhandled dim op {op.__name__}")

            def default_visit(self, value) -> str:
                if isinstance(value, bool):
                    raise TypeError("ShapeDim must not be bool")
                if isinstance(value, int):
                    return str(value)
                if self.identities is None:
                    raise TypeError(f"unsupported ShapeDim {type(value).__name__}")
                return _bind_param(value, self.params, self.param_map, self.identities)

        _RANGE_EXPR_VISITOR_TYPE = _RangeExprVisitor
    return _RANGE_EXPR_VISITOR_TYPE


def _dim_range_visitor_type():
    global _DIM_RANGE_VISITOR_TYPE
    if _DIM_RANGE_VISITOR_TYPE is None:
        from tilefoundry.ir.visitor import ExprVisitor  # noqa: PLC0415

        class _DimRangeVisitor(ExprVisitor[tuple[int, int]]):
            def visit_Constant(self, value: Constant) -> tuple[int, int]:
                number = int(value.value)
                return number, number + 1

            def visit_DimVar(self, value: DimVar) -> tuple[int, int]:
                return value.lo, value.hi

            def visit_Call(self, value: Call) -> tuple[int, int]:
                if type(value.target) is DimMul:
                    a, b = value.args
                    if not (_is_const(a) or _is_const(b)):
                        alo, ahi = self.visit(a)
                        blo, bhi = self.visit(b)
                        corners = (
                            alo * blo,
                            alo * (bhi - 1),
                            (ahi - 1) * blo,
                            (ahi - 1) * (bhi - 1),
                        )
                        return min(corners), max(corners) + 1
                params: dict[str, tuple[int, int] | None] = {}
                expr = _range_expr(value, params)
                prefix = f"[{', '.join(params)}] -> " if params else ""
                pw_aff = isl.pw_aff(prefix + f"{{ [{expr}] }}")
                if params:
                    bounds = " and ".join(
                        f"{lo} <= {name} <= {hi - 1}"
                        for name, bound in params.items()
                        for lo, hi in (bound,)
                    )
                    pw_aff = pw_aff.intersect_params(isl.set(prefix + f"{{ : {bounds} }}"))
                return int(pw_aff.min_val().num_si()), int(pw_aff.max_val().num_si()) + 1

            def default_visit(self, value) -> tuple[int, int]:
                if isinstance(value, bool):
                    raise TypeError("ShapeDim must not be bool")
                if isinstance(value, int):
                    return value, value + 1
                raise TypeError(f"unsupported ShapeDim {type(value).__name__}")

        _DIM_RANGE_VISITOR_TYPE = _DimRangeVisitor
    return _DIM_RANGE_VISITOR_TYPE


def _range_expr(
    dim,
    params: dict[str, tuple[int, int] | None],
    *,
    param_map: dict[str, object] | None = None,
    identities: dict[int, str] | None = None,
) -> str:
    return _range_expr_visitor_type()(params, param_map, identities).visit(dim)


def _raw_dim_call(op_cls, args: tuple):
    scalar = TensorType.umat_scalar()

    def wrap(value):
        if isinstance(value, bool):
            raise TypeError("bool is not a ShapeDim")
        if isinstance(value, int):
            return Constant(type=scalar, value=value)
        return value

    return Call(type=scalar, target=op_cls(), args=tuple(wrap(arg) for arg in args))


def _visit(expr, param_map: dict[str, object]):
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
            return _raw_dim_call(DimSub, (0, _visit(expr.op_arg(0), param_map)))
        a = _visit(expr.op_arg(0), param_map)
        b = _visit(expr.op_arg(1), param_map)
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


def to_dim(pw_aff: "isl.pw_aff", param_map: dict[str, object]):
    """Decode *pw_aff* into a ShapeDim using *param_map* for identifiers."""
    build = isl.ast_build.from_context(pw_aff.domain_space().universe_set())
    return _visit(build.expr_from(pw_aff), param_map)


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
        expr = _range_expr(
            value,
            params,
            param_map=param_map,
            identities={},
        )
        prefix = f"[{', '.join(params)}] -> " if params else ""
        normalized = to_dim(isl.pw_aff(prefix + f"{{ [{expr}] }}"), param_map)
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
    if isinstance(value, DimVar) or (
        isinstance(value, Constant)
        and isinstance(value.value, int)
        and not isinstance(value.value, bool)
    ) or (
        isinstance(value, Call) and isinstance(value.target, _DIM_OP_TYPES)
    ):
        return normalize_dim(value)
    return value


def dim_range(dim) -> tuple[int, int]:
    """Return conservative half-open value bounds ``[lo, hi)`` for *dim*."""
    return _dim_range_visitor_type()().visit(dim)


def to_domain(extents: tuple) -> tuple:
    """Build a bounded iteration domain and its isl-parameter ShapeDim map."""
    param_map: dict[str, object] = {}
    bounds: dict[str, tuple[int, int]] = {}
    seen: dict = {}
    names: list[str] = []

    def bind(name: str, dim, lo: int, hi: int) -> None:
        bound = (lo, hi)
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
            bind(extent.name, extent, extent.lo, extent.hi)
            constraints.append(f"0 <= d{i} < {extent.name}")
        elif isinstance(extent, Call):
            name = seen.get(extent)
            if name is None:
                name = f"D{i}"
                seen[extent] = name
            lo, hi = dim_range(extent)
            bind(name, extent, lo, hi)
            constraints.append(f"0 <= d{i} < {name}")
        else:
            raise TypeError(f"unsupported ShapeDim {type(extent).__name__}")

    constraints += [f"{bounds[name][0]} <= {name} < {bounds[name][1]}" for name in names]
    prefix = f"[{', '.join(names)}] -> " if names else ""
    if not dims:
        return isl.set(prefix + "{ [] }"), param_map
    body = f"{{ [{', '.join(dims)}] : {' and '.join(constraints)} }}"
    return isl.set(prefix + body), param_map


__all__ = [
    "dim_range",
    "normalize_dim",
    "normalize_dim_entries",
    "to_dim",
    "to_domain",
]
