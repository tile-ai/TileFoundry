"""Forward access-relation construction helpers (input-type driven).

These build the bounded iteration ``domain`` (an ``isl.set``) for an op from
its iteration extents. A static extent becomes a constant constraint
(``0 <= d < N``); a ``DimVar`` becomes an isl parameter constrained to its
half-open ``[lo, hi)`` range; affine dim-arithmetic extents translate to isl affine
expressions over those parameters. The result feeds
``AccessRelationResult.domain``; the relation never carries a tensor shape.
"""
from __future__ import annotations

import isl

from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.types.dim import DimAdd, DimFloorDiv, DimMul, DimSub, DimVar, simplify_dim


def _is_const(node) -> bool:
    """A static (non-symbolic) dim operand."""
    if isinstance(node, bool):
        return False
    return isinstance(node, int) or isinstance(node, Constant)


# Opaque isl parameter registry for ``DimFloorDiv`` extents. The same
# canonicalized expression always binds to the same isl parameter name
# (keyed table): ``_floordiv_param_name`` mints it on first sight,
# ``_floordiv_param_expr`` is its reverse, consulted by the recovery side so
# an opaque parameter resolves back to the original ``DimExpr``.
_floordiv_param_name: dict[object, str] = {}
_floordiv_param_expr: dict[str, object] = {}


def _affine_bounds(dim) -> tuple[int, int]:
    """Half-open value bounds ``[lo, hi)`` of *dim*.

    Used to derive an opaque isl parameter's bound from a ``DimFloorDiv``
    dividend. Accepts the same affine op set ``_dim_to_isl`` renders
    (``DimVar`` / ``Constant`` / ``int`` leaves, ``DimAdd`` / ``DimSub``,
    ``DimMul`` with a constant operand); anything else raises.
    """
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
        if op in (DimAdd, DimSub):
            alo, ahi = _affine_bounds(dim.args[0])
            blo, bhi = _affine_bounds(dim.args[1])
            if op is DimAdd:
                return (alo + blo, ahi + bhi - 1)
            return (alo - bhi + 1, ahi - blo)
        if op is DimMul:
            lhs, rhs = dim.args
            if not (_is_const(lhs) or _is_const(rhs)):
                raise NotImplementedError(
                    "DimMul of two symbolic dims is not affine-expressible as an isl extent"
                )
            const, other = (lhs, rhs) if _is_const(lhs) else (rhs, lhs)
            c = int(const.value) if isinstance(const, Constant) else int(const)
            olo, ohi = _affine_bounds(other)
            if c >= 0:
                return (olo * c, (ohi - 1) * c + 1)
            return ((ohi - 1) * c, olo * c + 1)
        raise NotImplementedError(
            f"dim op {op.__name__} is not affine-expressible as an isl extent bound"
        )
    raise TypeError(f"unsupported ShapeDim {type(dim).__name__}")


def _floordiv_to_isl_param(dim, params: dict[str, tuple[int, int]]) -> str:
    """Bind a ``DimFloorDiv`` extent to an opaque isl parameter.

    The divisor must be a positive constant — a symbolic divisor is not
    affine-expressible and still raises ``NotImplementedError``. The
    canonicalized expression is the registry key (see
    ``_floordiv_param_name``), and the parameter's bound is derived from the
    dividend's own value bounds: ``P in [lo, hi)`` binds ``P // c`` to
    ``[lo // c, (hi - 1) // c + 1)``.
    """
    dividend, divisor = dim.args
    if not _is_const(divisor):
        raise NotImplementedError(
            "DimFloorDiv by a symbolic divisor is not affine-expressible as an isl extent"
        )
    c = int(divisor.value) if isinstance(divisor, Constant) else int(divisor)
    if c <= 0:
        raise ValueError(f"DimFloorDiv divisor must be positive, got {c}")
    canon = simplify_dim(DimFloorDiv, (dividend, divisor))
    name = _floordiv_param_name.get(canon)
    if name is None:
        name = f"__floordiv{len(_floordiv_param_name)}"
        _floordiv_param_name[canon] = name
        _floordiv_param_expr[name] = canon
    lo, hi = _affine_bounds(dividend)
    bound = (lo // c, (hi - 1) // c + 1)
    prev = params.get(name)
    if prev is not None and prev != bound:
        raise ValueError(
            f"isl parameter {name!r} used with conflicting bounds {prev} vs {bound}"
        )
    params[name] = bound
    return name


def _dim_to_isl(dim, params: dict[str, tuple[int, int]]) -> str:
    """Render *dim* as an isl expression string, recording any ``DimVar`` it
    uses in *params* (name → ``(lo, hi)``).

    Affine extents are expressible directly: ``+`` / ``-``, and ``*`` with at
    least one constant operand. A ``DimFloorDiv`` by a constant binds to a
    fresh opaque isl parameter (see ``_floordiv_to_isl_param``). A symbol×symbol
    product (or any other dim op) raises ``NotImplementedError`` so it never
    reaches libisl as a non-affine string. A ``DimVar`` (or opaque parameter)
    reused under conflicting bounds raises ``ValueError``.
    """
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
        if op in (DimAdd, DimSub):
            a = _dim_to_isl(dim.args[0], params)
            b = _dim_to_isl(dim.args[1], params)
            return f"({a} {'+' if op is DimAdd else '-'} {b})"
        if op is DimMul:
            lhs, rhs = dim.args
            if not (_is_const(lhs) or _is_const(rhs)):
                raise NotImplementedError(
                    "DimMul of two symbolic dims is not affine-expressible as an isl extent"
                )
            a = _dim_to_isl(lhs, params)
            b = _dim_to_isl(rhs, params)
            return f"({a} * {b})"
        if op is DimFloorDiv:
            return _floordiv_to_isl_param(dim, params)
        raise NotImplementedError(
            f"dim op {op.__name__} is not affine-expressible as an isl extent"
        )
    raise TypeError(f"unsupported ShapeDim {type(dim).__name__}")


def build_domain(extents: tuple) -> "isl.set":
    """Bounded iteration domain ``{ [d0, ..., dn] : 0 <= di < extent_i }``.

    Static extents are constant constraints; ``DimVar`` extents are isl
    parameters carrying their half-open ``[lo, hi)`` bound. A rank-0 op gives ``{ [] }``.
    """
    params: dict[str, tuple[int, int]] = {}
    dims = [f"d{i}" for i in range(len(extents))]
    constraints = [
        f"0 <= d{i} < {_dim_to_isl(ext, params)}" for i, ext in enumerate(extents)
    ]
    constraints += [f"{lo} <= {name} < {hi}" for name, (lo, hi) in params.items()]
    prefix = f"[{', '.join(params)}] -> " if params else ""
    if not dims:
        return isl.set(prefix + "{ [] }")
    body = f"{{ [{', '.join(dims)}] : {' and '.join(constraints)} }}"
    return isl.set(prefix + body)


def _extent_of_domain_dim(domain: "isl.set", d: int, dimvars: dict):
    """Extent of domain dim *d* = ``dim_max(d) + 1``. Static → ``int``;
    otherwise isl has normalized the extent to ``const + Σ coeff_i·param_i``,
    which is rebuilt into a ``ShapeDim`` by resolving each param name back to
    its ``DimExpr`` (via *dimvars*, or the opaque ``DimFloorDiv`` registry)
    and folding the sum through ``simplify_dim``. Anything isl could not
    normalize to this form (piecewise max, a div) or a param with no known
    ``DimExpr`` is not a recoverable ShapeDim → raise (fail closed)."""
    pieces: list = []
    domain.dim_max(d).foreach_piece(lambda _s, a: pieces.append(a))
    if len(pieces) != 1:
        raise ValueError(f"domain dim {d}: dim_max is piecewise; cannot recover extent")
    aff = pieces[0]
    if aff.dim(isl.dim_type.DIV):
        raise ValueError(f"domain dim {d}: dim_max involves a div; cannot recover extent")
    n_par = aff.dim(isl.dim_type.PARAM)
    params = [
        (aff.get_dim_name(isl.dim_type.PARAM, i),
         int(aff.get_coefficient_val(isl.dim_type.PARAM, i).num_si()))
        for i in range(n_par)
    ]
    size_const = int(aff.get_constant_val().num_si()) + 1
    nonzero = [(name, c) for name, c in params if c != 0]
    if not nonzero:
        return size_const
    terms: list = []
    for name, coeff in nonzero:
        base = dimvars.get(name, _floordiv_param_expr.get(name))
        if base is None:
            raise ValueError(f"domain dim {d}: cannot recover ShapeDim from extent {aff}")
        terms.append(base if coeff == 1 else simplify_dim(DimMul, (coeff, base)))
    if size_const != 0:
        terms.insert(0, size_const)
    result = terms[0]
    for term in terms[1:]:
        result = simplify_dim(DimAdd, (result, term))
    return result


def _collect_dimvars(dim, dimvars: dict) -> None:
    """Record every ``DimVar`` reachable from *dim* into *dimvars* (name →
    ``DimVar``), recursing into a derived dim expression's operands so a
    ``DimVar`` nested inside it (not just a bare top-level shape entry) is
    still resolvable by parameter name."""
    if isinstance(dim, DimVar):
        dimvars[dim.name] = dim
    elif isinstance(dim, Call):
        for arg in dim.args:
            _collect_dimvars(arg, dimvars)


def shape_from_relation(input_types: tuple, relation) -> tuple:
    """Derive the output shape from the relation's output map + bounded domain.

    Each output map result axis is a pure projection of a domain dim (its
    extent becomes the output dim) or a constant (a size-1 output axis). The
    domain (built forward from the input types, dynamic dims as isl params) is
    the single source of bounds; ``DimVar``s are recovered by parameter name.
    A non-projection / non-constant result axis, or an extent that does not
    resolve to a ShapeDim, fails closed.
    """
    domain = relation.domain
    output_map = relation.maps[-1]
    ma = output_map.as_pw_multi_aff().as_multi_aff()
    n_out = ma.dim(isl.dim_type.OUT)
    n_in = ma.dim(isl.dim_type.IN)
    dimvars: dict = {}
    for t in input_types:
        for dim in t.shape:
            _collect_dimvars(dim, dimvars)
    shape: list = []
    for o in range(n_out):
        aff = ma.get_at(o)
        used = [
            (j, int(aff.get_coefficient_val(isl.dim_type.IN, j).num_si()))
            for j in range(n_in)
            if int(aff.get_coefficient_val(isl.dim_type.IN, j).num_si()) != 0
        ]
        if not used:
            shape.append(1)  # constant result: a size-1 output axis
        elif len(used) == 1 and used[0][1] == 1:
            shape.append(_extent_of_domain_dim(domain, used[0][0], dimvars))
        else:
            raise ValueError(
                f"output axis {o} is not a pure projection or constant; "
                "cannot infer shape"
            )
    return tuple(shape)


def validate_output_map_arity(output_map: "isl.map", output_shape: tuple) -> None:
    """Check the output access map's range rank matches the claimed output
    shape rank. The relation carries no shape, so this is the consistency
    point between the relation and the typeinfer-side output shape."""
    n_out = output_map.dim(isl.dim_type.OUT)
    if n_out != len(output_shape):
        raise ValueError(
            f"output map range rank {n_out} != output shape rank {len(output_shape)}"
        )


__all__ = ["build_domain", "shape_from_relation", "validate_output_map_arity"]
