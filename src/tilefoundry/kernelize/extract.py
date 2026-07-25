"""``extract(hir) -> TileGraph`` -- lift an HIR ``Function`` body into a
polyhedral (isl) representation: one statement per compute op, tiled to a
fixed V1 tile size, with reads/writes/deps ready to feed
``isl.schedule_constraints`` (see ``schedule_tree.py``).

Algorithm (docstring mirrors the design so the "why" travels with the code):

1. Walk ``hir.body`` with the analyzer's own ``_postorder`` (SSA-DAG
   postorder -- dependencies before dependents) and keep the ``Call``
   nodes whose target is a compute op (not a nested HIR ``Function``,
   not a structural node). Each such ``Call`` becomes one statement.
2. For each statement, get its access relations via
   ``access_relation.build_relation`` (the *forward*, input-type-driven
   registry) when the op has one registered. ``build_relation`` returns
   the relation at *element* granularity (``AccessRelationResult.domain``
   ranges over real tensor extents, static or ``DimVar``-parametrised);
   V1 tiles it down to a small fixed tile size (``DEFAULT_TILE_SIZE``) by
   re-deriving the domain from the tile-count extents (``_tile_domain``)
   and reusing each access map's own formula at the new granularity --
   valid because ``build_relation``'s maps are pure per-axis
   selections/broadcasts with no bounds of their own (see
   ``_retile_map``). Any isl parameter used is resolved back to its
   ``ShapeDim`` in the returned ``TileGraph.params``.
3. An op with no registered forward relation (``build_relation`` returns
   ``None``) has no generic path -- V1 special-cases ``RMSNorm`` only
   (see ``_rmsnorm_access``), and raises a clear, actionable
   ``NotImplementedError`` for anything else (never silently guesses).
4. Union every statement's domain/reads/writes into one
   ``isl.union_set``/``isl.union_map`` pair, then auto-infer ``deps`` by
   feeding an initial (topological-postorder) execution order into
   ``isl.union_access_info(...).compute_flow()`` -- the exact technique
   validated in ``m1_deps_probe.py``.

See the module docstring in ``tile_graph.py`` for why ``TileGraph``'s
shape departs from ``docs/spec/tilegraph.md``'s single-domain sketch.
"""
from __future__ import annotations

from dataclasses import dataclass

import isl

from tilefoundry.analysis.analyzer import _postorder
from tilefoundry.ir.core import Call, TypeInferContext, Var, binding_name
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.visitor_registry.access_relation import AccessRelationResult, build_relation
from tilefoundry.visitor_registry.isl_utility import to_domain

from .tile_graph import TileGraph, TileUnit

# V1 has no tile-pin annotation reader yet (`where(...)` is future work --
# see the module docstring); every domain axis is tiled by this one fixed
# size (ceiling division, see `_tile_extents`).
DEFAULT_TILE_SIZE = 32

# Cosmetic-only: short isl tuple names for the two ops this V1 slice
# exercises, matching the m1_deps_probe.py / PoC 09 reference output
# 1:1 so a reviewer can diff by eye. Any other op falls back to its bare
# class name (still a valid isl identifier); a name that collides with
# another statement in the same extraction gets a numeric suffix
# (`_assign_statement_names`), so this table never causes an ambiguity,
# it only picks a nicer name for the common single-occurrence case.
_STATEMENT_ABBREV = {"MatMul": "MM", "RMSNorm": "RN"}


class ExtractError(NotImplementedError):
    """A construct `extract` does not (yet) support -- always raised with
    a specific, actionable message; V1 never silently guesses."""


@dataclass(frozen=True)
class _StatementAccess:
    """Extraction-internal: one statement's tiled domain + tuple-named
    access maps, before they are unioned into the returned ``TileGraph``."""

    name: str
    domain: "isl.set"
    op: Call
    reads: tuple["isl.map", ...]
    writes: tuple["isl.map", ...]
    params: dict


def _buffer_namer():
    """Stable ``id(expr) -> isl tuple name`` assignment, shared across the
    whole extraction so the same SSA value (e.g. matmul's ``h`` output,
    read again by rms_norm) gets the same buffer name everywhere it is
    referenced. Prefers the authored name (``Var.name`` for a parameter,
    ``binding_name`` -- the parser-attached SSA let-binding -- for a Call
    result) and de-dupes any accidental collision with a numeric suffix.
    """
    seen: dict[int, str] = {}
    used: set[str] = set()

    def name_for(expr) -> str:
        key = id(expr)
        cached = seen.get(key)
        if cached is not None:
            return cached
        if isinstance(expr, Var):
            base = expr.name
        else:
            base = binding_name(expr) or f"t{key}"
        candidate = base
        suffix = 0
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        used.add(candidate)
        seen[key] = candidate
        return candidate

    return name_for


def _assign_statement_names(ops: list[object]) -> list[str]:
    """One isl tuple name per statement, in call order: the bare (possibly
    abbreviated) op name when it is unique in this extraction, else
    suffixed ``NAME0``, ``NAME1``, ... in first-seen order."""
    bases = [_STATEMENT_ABBREV.get(type(op).__name__, type(op).__name__) for op in ops]
    counts: dict[str, int] = {}
    for base in bases:
        counts[base] = counts.get(base, 0) + 1
    next_index: dict[str, int] = {}
    names = []
    for base in bases:
        if counts[base] == 1:
            names.append(base)
            continue
        index = next_index.get(base, 0)
        names.append(f"{base}{index}")
        next_index[base] = index + 1
    return names


@dataclass(frozen=True)
class _ParamAxis:
    """Marks a domain axis whose extent is the isl parameter ``param``,
    not a static int (see ``_domain_extents``)."""

    param: str


def _axis_param_name(domain: "isl.set", axis: int) -> "str | None":
    """The isl parameter axis ``axis``'s upper bound depends on, or
    ``None`` if it's a plain constant. ``dim_max``'s pw_aff mentions
    every param in the domain regardless (its piece's validity condition
    leaks in), so this checks each piece's own ``aff`` instead."""
    mx = domain.dim_max(axis)
    if mx.is_cst():
        return None
    nparams = domain.dim(isl.dim_type.PARAM)
    pieces: list = []
    mx.foreach_piece(lambda s, aff: pieces.append(aff))
    names = {
        domain.get_dim_name(isl.dim_type.PARAM, p)
        for aff in pieces
        for p in range(nparams)
        if aff.involves_dims(isl.dim_type.PARAM, p, 1)
    }
    if len(names) != 1:
        raise ExtractError(
            f"kernelize.extract: domain axis {axis}'s bound depends on "
            f"{len(names)} isl parameters {sorted(names)!r} -- V1 "
            "dynamic-shape tiling only supports a bound tied to exactly "
            "one parameter per axis"
        )
    return next(iter(names))


def _domain_extents(domain: "isl.set") -> tuple:
    """Read ``domain``'s per-axis bounds back: a static axis as a plain
    ``[lo, hi]`` int extent, a dynamic axis as a ``_ParamAxis`` marker.
    Only SET axes are read -- a PARAM dim is never treated as one."""
    rank = domain.dim(isl.dim_type.SET)
    extents: list = []
    for i in range(rank):
        name = _axis_param_name(domain, i)
        if name is not None:
            extents.append(_ParamAxis(name))
            continue
        lo = int(domain.dim_min_val(i).num_si())
        hi = int(domain.dim_max_val(i).num_si())
        extents.append(hi - lo + 1)
    return tuple(extents)


def _tile_extents(extents: tuple, tile_size: int) -> tuple:
    tiled = []
    for axis, extent in enumerate(extents):
        if isinstance(extent, _ParamAxis):
            tiled.append(extent)
            continue
        if not isinstance(extent, int) or isinstance(extent, bool):
            raise ExtractError(
                f"kernelize.extract: domain axis {axis} has a non-static "
                f"extent {extent!r} ({type(extent).__name__}) -- V1 only "
                "tiles a plain int extent or a dynamic-shape axis "
                "(_domain_extents' _ParamAxis marker)"
            )
        tiled.append(-(-extent // tile_size))  # ceildiv
    return tuple(tiled)


def _tile_domain(
    extents: tuple, tile_size: int, elem_domain: "isl.set", stmt_name: str,
) -> "isl.set":
    """``extents`` (``_tile_extents``'s output) -> the tuple-named
    tile-count domain. All-static defers to ``to_domain`` unchanged; a
    ``_ParamAxis`` gets a ``tile_size*d_i < param`` bound instead of a
    fixed count, bounded via ``elem_domain``'s own parameter range."""
    if not any(isinstance(e, _ParamAxis) for e in extents):
        domain, _ = to_domain(extents)
        return domain.set_tuple_name(stmt_name)

    dims = [f"d{i}" for i in range(len(extents))]
    constraints = [
        f"{tile_size}*d{i} < {e.param} and d{i} >= 0"
        if isinstance(e, _ParamAxis)
        else f"0 <= d{i} < {e}"
        for i, e in enumerate(extents)
    ]
    params = sorted({e.param for e in extents if isinstance(e, _ParamAxis)})
    prefix = f"[{', '.join(params)}] -> "
    body = f"{{ [{', '.join(dims)}] : {' and '.join(constraints)} }}"
    domain = isl.set(prefix + body).intersect_params(elem_domain.params())
    return domain.set_tuple_name(stmt_name)


def _collect_params(extents: tuple, param_map: dict, stmt_name: str) -> dict:
    """This statement's ``TileGraph.params`` contribution: each dynamic
    axis's isl parameter resolved back to its ``ShapeDim`` via
    ``param_map``."""
    out: dict = {}
    for e in extents:
        if not isinstance(e, _ParamAxis):
            continue
        dim = param_map.get(e.param)
        if dim is None:
            raise ExtractError(
                f"kernelize.extract: statement {stmt_name!r} has a dynamic "
                f"axis bound to isl parameter {e.param!r}, but its access "
                "relation's param_map has no entry for it"
            )
        out[e.param] = dim
    return out


def _retile_map(m: "isl.map", stmt_name: str, tile_domain: "isl.set", buffer_name: str) -> "isl.map":
    """Reuse an *element*-granularity access map's own formula at *tile*
    granularity.

    ``build_relation``'s maps (e.g. matmul's ``{ [d0,d1,d2] -> [d0,d2] }``)
    are pure per-axis selections/broadcasts with no bounds of their own --
    the bounds live entirely in the paired ``domain``. isl requires a
    map's ``IN`` tuple identity (name *and* param space) to match the set
    it is intersected with, so the statement name has to be stamped on
    *before* ``intersect_domain`` -- reversing that order raises
    ``isl.Error: incompatible spaces`` (confirmed against isl-python
    0.1.8 while building this module). Restricting the *same* formula to
    a differently-bounded (tile-count instead of element-count) domain of
    equal rank is exactly "op iteration space div tile size".
    """
    return (
        m.set_tuple_name(isl.dim_type.IN, stmt_name)
        .intersect_domain(tile_domain)
        .set_tuple_name(isl.dim_type.OUT, buffer_name)
    )


def _registered_access(
    call: Call, stmt_name: str, tile_size: int, result: AccessRelationResult, namer,
) -> _StatementAccess:
    """Statement extraction for an op with a registered forward relation
    (``access_relation.build_relation`` returned non-``None``, e.g. MatMul).
    """
    tile_extents = _tile_extents(_domain_extents(result.domain), tile_size)
    tile_domain = _tile_domain(tile_extents, tile_size, result.domain, stmt_name)
    params = _collect_params(tile_extents, result.param_map, stmt_name)

    n_inputs = len(call.args)
    output_maps = result.maps[n_inputs:]
    if not output_maps:
        raise ExtractError(
            f"kernelize.extract: {type(call.target).__name__} build_relation "
            "produced no output map(s); a compute-op statement must write "
            "at least one value"
        )

    reads: list["isl.map"] = []
    writes: list["isl.map"] = []
    for i, arg in enumerate(call.args):
        reads.append(_retile_map(result.maps[i], stmt_name, tile_domain, namer(arg)))

    n_outputs = len(output_maps)
    for out_idx, raw_map in enumerate(output_maps):
        out_buf = namer(call) if n_outputs == 1 else f"{namer(call)}_{out_idx}"
        tiled = _retile_map(raw_map, stmt_name, tile_domain, out_buf)
        writes.append(tiled)
        # An output map that is not injective means two distinct domain
        # points (e.g. two different k-steps of a reduction) write the
        # *same* output cell -- only sound as a read-modify-write
        # accumulation, so the write is also a read (this is what lets
        # isl's compute_flow discover the matmul k-carry automatically,
        # exactly like m1_deps_probe.py's hand-written `MM -> h` read).
        # An injective output map (the common elementwise/projection
        # case) is pure write, no self-read.
        if not tiled.is_injective():
            reads.append(tiled)

    return _StatementAccess(
        name=stmt_name, domain=tile_domain, op=call,
        reads=tuple(reads), writes=tuple(writes), params=params,
    )


def _rmsnorm_access(call: Call, stmt_name: str, tile_size: int, namer) -> _StatementAccess:
    """V1 fallback for ``RMSNorm``: it has no upstream-registered forward
    relation (``build_relation`` returns ``None`` -- see the task report),
    so this hand-derives the same shape a real registration would.

    RMSNorm reduces over the tensor's last axis entirely *inside* one
    statement instance (V1 does not tile the reduction axis -- the
    statement reads/writes the whole row), so its domain is only the
    *batch* axes (``x.shape[:-1]``), tiled; the reduction axis is an
    existentially-quantified range dim on the access maps, matching
    m1_deps_probe.py's ``RN[i] -> h[i,j] : 0<=j<Tj`` shape. A batch axis
    may itself be dynamic -- it goes through the same pipeline
    ``_registered_access`` does; the reduction/weight axes stay
    static-only.
    """
    x, weight = call.args
    x_shape = x.type.shape
    w_shape = weight.type.shape
    if len(x_shape) < 1 or len(w_shape) != 1:
        raise ExtractError(
            "kernelize.extract: RMSNorm V1 fallback expects rank>=1 x and "
            f"rank-1 weight, got x.shape={x_shape} weight.shape={w_shape}"
        )

    elem_batch_domain, batch_param_map = to_domain(x_shape[:-1])
    batch_extents = _tile_extents(_domain_extents(elem_batch_domain), tile_size)
    reduce_tiles = _tile_extents((x_shape[-1],), tile_size)[0]
    weight_tiles = _tile_extents((w_shape[0],), tile_size)[0]

    domain = _tile_domain(batch_extents, tile_size, elem_batch_domain, stmt_name)
    params = _collect_params(batch_extents, batch_param_map, stmt_name)

    batch_dims = [f"d{i}" for i in range(len(batch_extents))]
    src = f"[{', '.join(batch_dims)}]"
    row = ", ".join(batch_dims + ["j"])

    read_x = isl.map(f"{{ {src} -> [{row}] : 0 <= j < {reduce_tiles} }}")
    read_w = isl.map(f"{{ {src} -> [j] : 0 <= j < {weight_tiles} }}")
    write_y = isl.map(f"{{ {src} -> [{row}] : 0 <= j < {reduce_tiles} }}")

    reads = (
        _retile_map(read_x, stmt_name, domain, namer(x)),
        _retile_map(read_w, stmt_name, domain, namer(weight)),
    )
    writes = (_retile_map(write_y, stmt_name, domain, namer(call)),)
    return _StatementAccess(
        name=stmt_name, domain=domain, op=call, reads=reads, writes=writes, params=params,
    )


def _extract_statement(call: Call, stmt_name: str, tile_size: int, namer) -> _StatementAccess:
    input_types = tuple(arg.type for arg in call.args)
    result = build_relation(call, input_types, TypeInferContext())
    if result is not None:
        return _registered_access(call, stmt_name, tile_size, result, namer)
    if isinstance(call.target, RMSNorm):
        return _rmsnorm_access(call, stmt_name, tile_size, namer)
    raise ExtractError(
        f"kernelize.extract: op {type(call.target).__name__!r} has no "
        "registered forward type_relation (access_relation.build_relation "
        "returned None) and no V1 fallback in kernelize.extract. Register "
        "one via tilefoundry.visitor_registry.access_relation."
        "register_type_relation(...), or extend the V1 fallback table in "
        "this module -- do not guess an access pattern silently."
    )


def _initial_schedule(accesses: list[_StatementAccess]) -> "isl.union_map":
    """A total execution order sufficient to seed ``compute_flow`` --
    *not* the final schedule (``schedule_tree.py`` computes that for
    real). Encodes exactly m1_deps_probe.py step 4's two ingredients,
    concatenated as ``[stage, *own_dims, 0-pad]``: postorder call index
    as a leading stage tie-break, then each statement's own iteration
    dims in declared order. m1_deps_probe.py instead interleaves the
    shared outer dim before the stage marker (``[i, stage, j, k]``) to
    model a fused per-i loop nest; this flat ``[stage, i, j, k]`` form is
    simpler to derive generically (no cross-statement dim-correspondence
    detection) and was confirmed (via an isl-python probe reproducing
    this exact program) to yield the identical must-dependence result --
    schedule order only ever tie-breaks *same-memory-location* accesses,
    and here those only ever come from matching statement instances
    either way.
    """
    common_rank = 1 + max((a.domain.dim(isl.dim_type.SET) for a in accesses), default=0)
    sched = isl.union_map("{}")
    for stage, acc in enumerate(accesses):
        rank = acc.domain.dim(isl.dim_type.SET)
        dims = [f"d{i}" for i in range(rank)]
        src = f"[{', '.join(dims)}]"
        tail = dims + ["0"] * (common_rank - 1 - rank)
        dst = f"[{stage}, {', '.join(tail)}]" if tail else f"[{stage}]"
        m = isl.map(f"{{ {src} -> {dst} }}").set_tuple_name(isl.dim_type.IN, acc.name)
        sched = sched.union(m.intersect_domain(acc.domain))
    return sched


def extract(hir: Function, *, tile_size: int = DEFAULT_TILE_SIZE) -> TileGraph:
    """Lift ``hir``'s body into a :class:`TileGraph`: one statement per
    compute op (tiled to ``tile_size``), with ``deps`` auto-inferred from
    ``reads``/``writes`` (see module docstring for the full algorithm)."""
    if hir.body is None:
        raise ExtractError(
            f"kernelize.extract: hir Function {hir.name!r} has no body "
            "(a dispatch prototype cannot be extracted)"
        )

    order = _postorder(hir.body)
    if any(isinstance(e, GridRegionExpr) for e in order):
        raise ExtractError(
            "kernelize.extract: GridRegionExpr (looped) bodies are not "
            "supported in V1 -- only a flat SSA-DAG of compute-op Calls is"
        )

    compute_calls: list[Call] = []
    for e in order:
        if not isinstance(e, Call):
            continue
        if isinstance(e.target, Function):
            raise ExtractError(
                "kernelize.extract: nested HIR Function calls are not "
                "supported in V1 -- extract one flattened Function body at "
                "a time"
            )
        compute_calls.append(e)
    if not compute_calls:
        raise ExtractError(
            f"kernelize.extract: hir Function {hir.name!r} body has no "
            "compute ops to extract"
        )

    stmt_names = _assign_statement_names([call.target for call in compute_calls])
    namer = _buffer_namer()
    accesses = [
        _extract_statement(call, name, tile_size, namer)
        for call, name in zip(compute_calls, stmt_names)
    ]

    domain = isl.union_set("{}")
    reads = isl.union_map("{}")
    writes = isl.union_map("{}")
    params: dict = {}
    for acc in accesses:
        domain = domain.union(acc.domain)
        for m in acc.reads:
            reads = reads.union(m)
        for m in acc.writes:
            writes = writes.union(m)
        for name, dim in acc.params.items():
            prev = params.get(name)
            if prev is not None and prev != dim:
                raise ExtractError(
                    f"kernelize.extract: isl parameter {name!r} resolves to "
                    f"conflicting ShapeDims across statements: {prev!r} vs "
                    f"{dim!r}"
                )
            params[name] = dim

    schedule_map = _initial_schedule(accesses)
    info = isl.union_access_info(reads).set_must_source(writes).set_schedule_map(schedule_map)
    deps = info.compute_flow().get_must_dependence()

    units = tuple(TileUnit(name=acc.name, op=acc.op) for acc in accesses)
    return TileGraph(
        domain=domain, deps=deps, reads=reads, writes=writes, units=units, params=params,
    )


__all__ = ["extract", "ExtractError", "DEFAULT_TILE_SIZE"]
