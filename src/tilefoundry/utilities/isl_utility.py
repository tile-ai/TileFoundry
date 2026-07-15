"""DimExpr <-> isl bridge (port of nncase's ``Utilities/ISLUtility.cs``).

Encode (``to_domain``): a composite ``ShapeDim``'s arithmetic structure never
enters isl — only its ``dim_range`` enters, as the bound of a freshly minted
opaque isl parameter. The parameter name -> original ``ShapeDim`` mapping
(``ParamMap``) threads through the caller's own return value; nothing here is
module state.

Decode (``to_dim``): isl's own ``ast_build`` turns a ``pw_aff`` into an
``ast_expr`` tree (resolving any internal div/mod/piecewise structure isl
itself introduced), which a generic visitor walks back into a ``ShapeDim`` —
never a hand-written pattern match against a specific ``DimExpr`` op.
"""
from __future__ import annotations

from typing import NamedTuple

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

ParamMap = dict  # isl parameter name (str) -> the ShapeDim it stands for.


class DomainBuild(NamedTuple):
    domain: "isl.set"
    param_map: ParamMap


def dim_range(dim) -> tuple[int, int]:
    """Half-open value bounds ``[lo, hi)`` of *dim*, by interval arithmetic
    over every ``DimExpr`` kind (``DimVar``, ``DimAdd``/``DimSub``/``DimMul``/
    ``DimFloorDiv``/``DimMod``/``DimMax``/``DimMin``, constants)."""
    if isinstance(dim, bool):
        raise TypeError("ShapeDim must not be bool")
    if isinstance(dim, int):
        return (dim, dim + 1)
    if isinstance(dim, Constant):
        v = int(dim.value)
        return (v, v + 1)
    if isinstance(dim, DimVar):
        return (dim.lo, dim.hi)
    if isinstance(dim, Call):
        op = type(dim.target)
        alo, ahi = dim_range(dim.args[0])
        blo, bhi = dim_range(dim.args[1])
        if op is DimAdd:
            return (alo + blo, ahi + bhi - 1)
        if op is DimSub:
            return (alo - (bhi - 1), ahi - blo)
        if op is DimMul:
            corners = (alo * blo, alo * (bhi - 1), (ahi - 1) * blo, (ahi - 1) * (bhi - 1))
            return (min(corners), max(corners) + 1)
        if op is DimFloorDiv:
            if blo <= 0:
                raise ValueError("DimFloorDiv divisor range must be positive")
            corners = (alo // blo, alo // (bhi - 1), (ahi - 1) // blo, (ahi - 1) // (bhi - 1))
            return (min(corners), max(corners) + 1)
        if op is DimMod:
            if blo <= 0:
                raise ValueError("DimMod divisor range must be positive")
            return (0, bhi - 1)
        if op is DimMax:
            return (max(alo, blo), max(ahi, bhi))
        if op is DimMin:
            return (min(alo, blo), min(ahi, bhi))
        raise NotImplementedError(f"dim op {op.__name__} has no known value range")
    raise TypeError(f"unsupported ShapeDim {type(dim).__name__}")


def to_domain(extents: tuple) -> DomainBuild:
    """Bounded iteration domain ``{ [d0, ..., dn] : 0 <= di < extent_i }``.

    A static extent is an inline constraint. A bare ``DimVar`` is a same-name
    isl parameter bound to its own ``[lo, hi)``. Any other ``DimExpr`` mints
    an opaque parameter bound to ``dim_range(extent)`` — its arithmetic
    structure never enters isl, only its value range does. The same
    canonical expression reused across axes binds to one parameter. Returns
    the domain together with the per-call parameter name -> ``ShapeDim`` map
    (nothing here is retained between calls).
    """
    param_map: ParamMap = {}
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
        return DomainBuild(isl.set(prefix + "{ [] }"), param_map)
    body = f"{{ [{', '.join(dims)}] : {' and '.join(constraints)} }}"
    return DomainBuild(isl.set(prefix + body), param_map)


def _visit(expr, param_map: ParamMap):
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


def to_dim(pw_aff: "isl.pw_aff", param_map: ParamMap):
    """Decode *pw_aff* back into a ``ShapeDim`` via ``ast_build.expr_from``
    plus a generic ``ast_expr`` visitor — never a hand-written pattern match
    against a ``DimExpr`` op. *param_map* resolves each isl identifier the
    expression bottoms out on (as returned by a prior ``to_domain`` call)."""
    build = isl.ast_build.from_context(pw_aff.domain_space().universe_set())
    return _visit(build.expr_from(pw_aff), param_map)


__all__ = ["ParamMap", "DomainBuild", "dim_range", "to_domain", "to_dim"]
