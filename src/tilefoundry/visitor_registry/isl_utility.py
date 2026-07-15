"""ShapeDim <-> isl bridge for relation-derived shape inference.

A composite ``ShapeDim``'s arithmetic never enters the domain carried by
``AccessRelationResult`` — only its ``dim_range`` does, as the bound of a
freshly minted isl parameter (``to_domain``). Recovery reads that domain
back through isl's own ``ast_build`` (``to_dim``).
"""
from __future__ import annotations

import isl

from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.types.dim import (
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


def _is_const(node) -> bool:
    if isinstance(node, bool):
        return False
    return isinstance(node, int) or isinstance(node, Constant)


def _range_expr(dim, params: dict) -> str:
    if isinstance(dim, bool):
        raise TypeError("ShapeDim must not be bool")
    if isinstance(dim, int):
        return str(dim)
    if isinstance(dim, Constant):
        return str(int(dim.value))
    if isinstance(dim, DimVar):
        bound = (dim.lo, dim.hi)
        prev = params.get(dim.name)
        if prev is not None and prev != bound:
            raise ValueError(
                f"DimVar {dim.name!r} used with conflicting bounds {prev} vs {bound}"
            )
        params[dim.name] = bound
        return dim.name
    if isinstance(dim, Call):
        op = type(dim.target)
        a, b = dim.args
        if op is DimMul and not (_is_const(a) or _is_const(b)):
            # a * b for two non-constant terms is not affine -- isl cannot
            # represent it; bind it to a synthetic parameter instead.
            name = f"_t{len(params)}"
            params[name] = dim_range(dim)
            return name
        if op in (DimFloorDiv, DimMod) and not _is_const(b):
            raise NotImplementedError(
                f"{op.__name__} by a symbolic divisor has no isl representation"
            )
        sa, sb = _range_expr(a, params), _range_expr(b, params)
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
        raise NotImplementedError(f"dim op {op.__name__} has no isl representation")
    raise TypeError(f"unsupported ShapeDim {type(dim).__name__}")


def dim_range(dim) -> tuple[int, int]:
    """Half-open value bounds ``[lo, hi)`` of *dim*: build its isl value
    expression, bind every identifier to its own bound, and read the range
    back from isl. The one case isl cannot express -- a product of two
    non-constant terms -- falls back to interval arithmetic."""
    if isinstance(dim, bool):
        raise TypeError("ShapeDim must not be bool")
    if isinstance(dim, int):
        return (dim, dim + 1)
    if isinstance(dim, Constant):
        v = int(dim.value)
        return (v, v + 1)
    if isinstance(dim, DimVar):
        return (dim.lo, dim.hi)
    if isinstance(dim, Call) and type(dim.target) is DimMul:
        a, b = dim.args
        if not (_is_const(a) or _is_const(b)):
            alo, ahi = dim_range(a)
            blo, bhi = dim_range(b)
            corners = (alo * blo, alo * (bhi - 1), (ahi - 1) * blo, (ahi - 1) * (bhi - 1))
            return (min(corners), max(corners) + 1)
    params: dict = {}
    expr = _range_expr(dim, params)
    prefix = f"[{', '.join(params)}] -> " if params else ""
    pa = isl.pw_aff(prefix + f"{{ [{expr}] }}")
    if params:
        bounds = " and ".join(f"{lo} <= {n} <= {hi - 1}" for n, (lo, hi) in params.items())
        pa = pa.intersect_params(isl.set(prefix + f"{{ : {bounds} }}"))
    return (int(pa.min_val().num_si()), int(pa.max_val().num_si()) + 1)


def to_domain(extents: tuple) -> tuple:
    """Bounded iteration domain ``{ [d0, ..., dn] : 0 <= di < extent_i }``.

    A static extent is an inline constraint; a bare ``DimVar`` is a
    same-name isl parameter bound to its own ``[lo, hi)``; any other
    ``ShapeDim`` mints an opaque parameter bound to ``dim_range(extent)``,
    deduped by canonical expression across axes. Returns ``(domain,
    param_map)`` where ``param_map`` resolves each isl parameter name back
    to its ``ShapeDim`` -- this call's own data, not shared across calls.
    """
    param_map: dict = {}
    bounds: dict[str, tuple[int, int]] = {}
    seen: dict = {}
    names: list[str] = []

    def _bind(name: str, dim, lo: int, hi: int) -> None:
        bound = (lo, hi)
        prev = bounds.get(name)
        if prev is not None and prev != bound:
            raise ValueError(
                f"isl parameter {name!r} used with conflicting bounds {prev} vs {bound}"
            )
        if name not in bounds:
            names.append(name)
        bounds[name] = bound
        param_map[name] = dim

    dims = [f"d{i}" for i in range(len(extents))]
    constraints: list[str] = []
    for i, ext in enumerate(extents):
        if isinstance(ext, bool):
            raise TypeError("ShapeDim must not be bool")
        if isinstance(ext, int):
            constraints.append(f"0 <= d{i} < {ext}")
        elif isinstance(ext, Constant):
            constraints.append(f"0 <= d{i} < {int(ext.value)}")
        elif isinstance(ext, DimVar):
            _bind(ext.name, ext, ext.lo, ext.hi)
            constraints.append(f"0 <= d{i} < {ext.name}")
        elif isinstance(ext, Call):
            name = seen.get(ext)
            if name is None:
                name = f"D{i}"
                seen[ext] = name
            lo, hi = dim_range(ext)
            _bind(name, ext, lo, hi)
            constraints.append(f"0 <= d{i} < {name}")
        else:
            raise TypeError(f"unsupported ShapeDim {type(ext).__name__}")

    constraints += [f"{bounds[name][0]} <= {name} < {bounds[name][1]}" for name in names]
    prefix = f"[{', '.join(names)}] -> " if names else ""
    if not dims:
        return isl.set(prefix + "{ [] }"), param_map
    body = f"{{ [{', '.join(dims)}] : {' and '.join(constraints)} }}"
    return isl.set(prefix + body), param_map


def _visit(expr, param_map: dict):
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
            return simplify_dim(DimSub, (0, _visit(expr.op_arg(0), param_map)))
        a = _visit(expr.op_arg(0), param_map)
        b = _visit(expr.op_arg(1), param_map)
        if op == Op.ADD:
            return simplify_dim(DimAdd, (a, b))
        if op == Op.SUB:
            return simplify_dim(DimSub, (a, b))
        if op == Op.MUL:
            return simplify_dim(DimMul, (a, b))
        if op in (Op.DIV, Op.PDIV_Q, Op.FDIV_Q):
            return simplify_dim(DimFloorDiv, (a, b))
        if op == Op.PDIV_R:
            return simplify_dim(DimMod, (a, b))
        if op == Op.MAX:
            return simplify_dim(DimMax, (a, b))
        if op == Op.MIN:
            return simplify_dim(DimMin, (a, b))
        raise NotImplementedError(f"ast_expr op {op!r} has no ShapeDim decoding")
    raise NotImplementedError(f"unsupported ast_expr type {type(expr).__name__}")


def to_dim(pw_aff: "isl.pw_aff", param_map: dict):
    """Decode *pw_aff* into a ``ShapeDim`` via ``ast_build.expr_from`` plus
    a generic ``ast_expr`` visitor. *param_map* resolves each isl
    identifier the expression bottoms out on (from a prior ``to_domain``)."""
    build = isl.ast_build.from_context(pw_aff.domain_space().universe_set())
    return _visit(build.expr_from(pw_aff), param_map)


__all__ = ["dim_range", "to_domain", "to_dim"]
