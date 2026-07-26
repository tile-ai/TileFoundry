"""``extract(hir) -> TileGraph`` -- lift an HIR ``Function`` body into a
polyhedral (isl) representation: one statement per compute op, at element
granularity, with reads/writes/deps ready for the schedule layer to build a
tree over (see ``schedule/kernel_schedule.py``).

Algorithm (docstring mirrors the design so the "why" travels with the code):

1. Walk ``hir.body`` with the analyzer's own ``_postorder`` (SSA-DAG
   postorder -- dependencies before dependents) and keep the ``Call``
   nodes whose target is a compute op (not a structural/view node, e.g.
   ``TupleGetItem`` or the zero-op ``Reshape``, folded into every
   consumer's access map instead -- see ``_buffer_namer``). A ``Call``
   whose target is a nested HIR ``Function`` is penetrated instead of
   rejected: ``_walk_calls`` binds its params to the caller's own
   (already-resolved) argument expressions and recurses into its body,
   prefixing every statement/buffer that body contributes with the
   callee name plus a per-call-site index. Each compute ``Call`` becomes
   one statement -- or several, for an op whose outputs cannot share one
   domain (``RoPE``'s GQA q/k, see ``_rope_access``).
2. For each statement, narrow every argument's type to its local (per-shard)
   shape when it carries a ``ShardLayout`` (``_local_type``: each mesh
   ``Split``'s *tensor* axis divided by its mesh extent, tensor rank
   preserved -- centralized here, once, so every registered relation is
   sharding-aware without knowing sharding exists), then get the
   statement's access relations via ``access_relation.build_relation``
   (the *forward*, input-type-driven registry) over those local types.
   ``build_relation`` returns the relation at *element* granularity
   (``AccessRelationResult.domain`` ranges over the, already-local, tensor
   extents, static or ``DimVar``-parametrised); extract only stamps the
   statement's tuple name onto it and reuses each access map's own
   formula unchanged (see ``_bind_map``) -- no retiling happens here. Any
   isl parameter used is resolved back to its ``ShapeDim`` via the
   relation's own ``param_map`` into the returned ``TileGraph.params``.
3. An op with no registered forward relation (``build_relation`` returns
   ``None``) raises a clear, actionable ``NotImplementedError`` -- extract
   never silently guesses an access pattern. ``RoPE`` *is* registered but
   is also special-cased (``_rope_access``): its relation is single-domain
   (one value input + its own cos/sin/pos), so extract calls it once per
   branch (q, k) rather than through the generic multi-domain-per-statement
   path.
4. Union every statement's domain/reads/writes into one
   ``isl.union_set``/``isl.union_map`` pair, then auto-infer ``deps`` by
   feeding an initial (topological-postorder) execution order into
   ``isl.union_access_info(...).compute_flow()`` -- the exact technique
   validated in ``m1_deps_probe.py``. ``parallel_dims`` then falls out of
   ``domain`` + ``deps`` alone (``_parallel_dims``), so which dimensions are
   dependence-free is measured here rather than asked of a scheduler.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass, field

import isl

from tilefoundry.ir.core import Call, Tuple, TypeInferContext, Var, binding_name
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.nn.rope import RoPE
from tilefoundry.ir.hir.tensor.reshape import Reshape, flat_reshape_map
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, split_target_axes
from tilefoundry.visitor_registry.access_relation import AccessRelationResult, build_relation

from .analyzer import _postorder


@dataclass(frozen=True)
class TileUnit:
    """One statement's identity.

    ``name`` is the isl tuple name shared by this statement's pieces of
    ``TileGraph.domain``/``reads``/``writes``/``deps`` (e.g. ``"MM"``).
    ``op`` is the HIR ``Call`` (op@site) that produced this statement --
    the call node itself, not just its bare ``Op``, so a consumer can
    still recover ``op.target`` / ``op.args`` / ``op.type``.
    """

    name: str
    op: object


@dataclass(frozen=True)
class TileGraph:
    """Polyhedral model of one HIR ``Function`` body, plus its schedule.

    ``domain``/``reads``/``writes`` are unions of per-``TileUnit`` pieces,
    one tuple name per producing statement (domain/reads/writes) or
    accessed buffer (reads/writes range) -- a *union* domain, not the
    single ``isl.set`` ``docs/spec/tilegraph.md`` sketches, because the
    schedule tree needs one named tuple per statement.
    ``deps`` is the auto-inferred RAW must-dependence relation between
    statement instances (see :func:`extract`). ``params`` resolves any
    dynamic-shape isl parameter name appearing in ``domain`` back to its
    ``ShapeDim``.

    ``buffer_dtypes`` resolves each buffer tuple name back to the HIR
    ``DType`` its elements carry, so a byte count over an access relation
    needs no second walk of the HIR. ``parallel_dims`` is per statement,
    per own domain dimension, whether that dimension carries no dependence
    (see :func:`_parallel_dims`) -- a measured fact, not a scheduler's
    output, so the schedule layer reads it rather than deriving it.

    ``tree``/``ring``/``decisions`` start empty and fill in as the same
    object flows through the schedule layer's own stages. Both dict
    fields are plain fields, never an isl mark payload -- isl marks are
    process-global C state, not a place for a Python object to live.
    """

    domain: "isl.union_set"
    deps: "isl.union_map"
    reads: "isl.union_map"
    writes: "isl.union_map"
    units: tuple[TileUnit, ...]
    params: dict
    buffer_dtypes: dict = field(default_factory=dict)
    parallel_dims: dict = field(default_factory=dict)
    tree: "isl.schedule | None" = None
    ring: dict = field(default_factory=dict)
    decisions: dict | None = None


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
    """Extraction-internal: one statement's domain + tuple-named access
    maps, before they are unioned into the returned ``TileGraph``."""

    name: str
    domain: "isl.set"
    op: Call
    reads: tuple["isl.map", ...]
    writes: tuple["isl.map", ...]
    params: dict
    dtypes: dict


def _buffer_namer():
    """Stable ``id(expr) -> isl tuple name`` assignment, shared across the
    whole extraction so the same SSA value (e.g. matmul's ``h`` output,
    read again by rms_norm) gets the same buffer name everywhere it is
    referenced. Prefers the authored name (``Var.name`` for a parameter,
    ``binding_name`` -- the parser-attached SSA let-binding -- for a Call
    result) and de-dupes any accidental collision with a numeric suffix. A
    ``TupleGetItem`` has no buffer of its own -- it resolves to its source
    call's name plus the same ``_{index}`` suffix a multi-output statement
    writes under (``_registered_access``, ``_rope_access``), so a
    downstream read lines up with the write without minting a fresh name.
    ``Reshape`` is the same passthrough (also chases a chain), but its
    source has a different coordinate space -- the returned callable also
    carries that recomposition as ``namer.pierce(m, expr)`` (``_read_map``).
    ``prefix`` (only ever passed for a statement's own output, never a
    read) tags a fresh name with the penetrated helper it came from, so
    two call sites of the same helper never collide.
    """
    seen: dict[int, str] = {}
    used: set[str] = set()
    reshape_source: dict[int, "isl.map | None"] = {}
    anonymous = itertools.count()

    def name_for(expr, prefix: str = "") -> str:
        key = id(expr)
        cached = seen.get(key)
        if cached is not None:
            return cached
        if isinstance(expr, Call) and isinstance(expr.target, TupleGetItem):
            name = f"{name_for(expr.args[0])}_{expr.target.index}"
            seen[key] = name
            return name
        if isinstance(expr, Call) and isinstance(expr.target, Reshape):
            name = name_for(expr.args[0])
            seen[key] = name
            return name
        if isinstance(expr, Var):
            base = expr.name
        else:
            # An unbound intermediate (a bare `return op(...)`) has no authored
            # name; number it in visit order rather than by `id`, which would
            # put a memory address in the skeleton and change run to run.
            base = binding_name(expr) or f"t{next(anonymous)}"
        base = f"{prefix}{base}"
        candidate = base
        suffix = 0
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        used.add(candidate)
        seen[key] = candidate
        return candidate

    def source_map(expr) -> "isl.map | None":
        """``expr``'s own coords -> its ultimate non-reshape source's coords,
        composed hop-by-hop through a chain of reshapes; ``None`` if ``expr``
        is not (transitively) a reshape output."""
        key = id(expr)
        if key in reshape_source:
            return reshape_source[key]
        result = None
        if isinstance(expr, Call) and isinstance(expr.target, Reshape):
            x = expr.args[0]
            hop = flat_reshape_map(x.type.shape, expr.type.shape)
            upstream = source_map(x)
            result = hop if upstream is None else hop.apply_range(upstream)
        reshape_source[key] = result
        return result

    def pierce(m: "isl.map", expr) -> "isl.map":
        resh = source_map(expr)
        return m if resh is None else m.apply_range(resh)

    name_for.pierce = pierce
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


def _local_type(ty: TensorType) -> TensorType:
    """*ty* narrowed to its per-shard local shape when ``ty.layout`` is a
    ``ShardLayout``, else *ty* unchanged.

    ``shard_layout_local_shape`` divides *layout* positions, not tensor
    axes: ``canonical_shard_layout`` (the real ``make_shard_tensor_type``
    path) factors a split tensor axis into extra layout positions, so that
    helper's result can outrank the tensor -- an access relation built from
    it would then carry more dims than the buffer it reads/writes, and
    ``compute_flow`` silently drops the dependence rather than raising.
    ``split_target_axes`` instead names, per mesh axis, which *tensor* axis
    a ``Split`` targets; dividing that axis by its mesh extent keeps rank
    intact. A ``Partial``/``Broadcast``/``Dynamic`` mesh axis consumes no
    tensor axis. Centralized here (not per-relation) so every registered
    ``type_relation`` is sharding-aware for free.
    """
    layout = ty.layout
    if not isinstance(layout, ShardLayout):
        return ty
    targets = split_target_axes(layout, ty.shape)
    mesh_extents = layout.mesh.layout.shape
    local = list(ty.shape)
    for mesh_axis, tensor_axis in enumerate(targets):
        if tensor_axis is None:
            continue  # Partial / Broadcast / Dynamic: no tensor axis consumed
        extent = mesh_extents[mesh_axis]
        if extent is None:
            local[tensor_axis] = 1  # launch-provided CTA split: one shard, one slice
            continue
        size = local[tensor_axis]
        if not isinstance(size, int) or isinstance(size, bool):
            raise ExtractError(
                f"extract: tensor axis {tensor_axis} is Split-sharded "
                f"but its extent {size!r} is not a static int -- cannot divide "
                "a dynamic axis by a mesh extent"
            )
        if size % extent != 0:
            raise ExtractError(
                f"extract: tensor axis {tensor_axis} (extent {size}) "
                f"is not evenly divisible by its mesh extent {extent}"
            )
        local[tensor_axis] = size // extent
    return TensorType(shape=tuple(local), dtype=ty.dtype, layout=None, storage=ty.storage)


def _bind_map(m: "isl.map", stmt_name: str, domain: "isl.set", buffer_name: str) -> "isl.map":
    """Stamp one ``build_relation`` access map (element-granularity, no
    bounds of its own -- the bounds live entirely in the paired ``domain``)
    with its statement/buffer tuple names and restrict it to ``domain``.
    isl requires a map's ``IN`` tuple identity (name *and* param space) to
    match the set it is intersected with, so the statement name has to be
    stamped on *before* ``intersect_domain`` -- reversing that order raises
    ``isl.Error: incompatible spaces`` (confirmed against isl-python 0.1.8
    while building this module).
    """
    return (
        m.set_tuple_name(isl.dim_type.IN, stmt_name)
        .intersect_domain(domain)
        .set_tuple_name(isl.dim_type.OUT, buffer_name)
    )


def _read_map(m: "isl.map", stmt_name: str, domain: "isl.set", arg, namer) -> "isl.map":
    """A read access for input ``arg``, pierced through any reshape view
    (``namer.pierce``) before binding -- the view-fold's landing point for
    every op's input side (an op's own output buffer never needs piercing,
    a reshape output is never written to)."""
    return _bind_map(namer.pierce(m, arg), stmt_name, domain, namer(arg))


def _out_dtype(call: Call, out_idx: int):
    """The dtype of ``call``'s ``out_idx``-th output: a multi-output op
    (``RoPE``) types its result as a ``TupleType``, a single-output one as
    the ``TensorType`` itself."""
    ty = call.type
    if isinstance(ty, TupleType) and out_idx < len(ty.fields):
        return getattr(ty.fields[out_idx], "dtype", None)
    return getattr(ty, "dtype", None)


def _registered_access(
    call: Call, stmt_name: str, result: AccessRelationResult, namer, prefix: str,
) -> _StatementAccess:
    """Statement extraction for an op with a registered forward relation
    (``access_relation.build_relation`` returned non-``None``, e.g. MatMul,
    RMSNorm)."""
    domain = result.domain.set_tuple_name(stmt_name)

    n_inputs = len(call.args)
    output_maps = result.maps[n_inputs:]
    if not output_maps:
        raise ExtractError(
            f"extract: {type(call.target).__name__} build_relation "
            "produced no output map(s); a compute-op statement must write "
            "at least one value"
        )

    reads: list["isl.map"] = []
    writes: list["isl.map"] = []
    dtypes: dict = {}
    for i, arg in enumerate(call.args):
        m = _read_map(result.maps[i], stmt_name, domain, arg, namer)
        reads.append(m)
        dtypes[m.get_tuple_name(isl.dim_type.OUT)] = getattr(arg.type, "dtype", None)

    n_outputs = len(output_maps)
    for out_idx, raw_map in enumerate(output_maps):
        out_buf = namer(call, prefix) if n_outputs == 1 else f"{namer(call, prefix)}_{out_idx}"
        bound = _bind_map(raw_map, stmt_name, domain, out_buf)
        writes.append(bound)
        dtypes[out_buf] = _out_dtype(call, out_idx)
        # An output map that is not injective means two distinct domain
        # points (e.g. two different k-steps of a reduction) write the
        # *same* output cell -- only sound as a read-modify-write
        # accumulation, so the write is also a read (this is what lets
        # isl's compute_flow discover the matmul k-carry automatically,
        # exactly like m1_deps_probe.py's hand-written `MM -> h` read).
        # An injective output map (the common elementwise/projection
        # case) is pure write, no self-read.
        if not bound.is_injective():
            reads.append(bound)

    return _StatementAccess(
        name=stmt_name, domain=domain, op=call,
        reads=tuple(reads), writes=tuple(writes), params=result.param_map, dtypes=dtypes,
    )


def _rope_branch(
    call: Call, x, cos_cache, sin_cache, pos_ids, out_idx: int,
    stmt_name: str, namer, prefix: str,
) -> _StatementAccess:
    """One RoPE branch (q or k): calls the registered relation with *x*
    standing in for both value-input slots, keeping only *x*'s own reads
    and its ``out_idx`` output."""
    x_ty, cos_ty, sin_ty, pos_ty = (
        _local_type(t.type) for t in (x, cos_cache, sin_cache, pos_ids)
    )
    result = build_relation(call, (x_ty, x_ty, cos_ty, sin_ty, pos_ty), TypeInferContext())
    domain = result.domain.set_tuple_name(stmt_name)

    reads = [
        _read_map(result.maps[0], stmt_name, domain, x, namer),
        _read_map(result.maps[2], stmt_name, domain, cos_cache, namer),
        _read_map(result.maps[3], stmt_name, domain, sin_cache, namer),
        _read_map(result.maps[4], stmt_name, domain, pos_ids, namer),
    ]
    out_buf = f"{namer(call, prefix)}_{out_idx}"
    write = _bind_map(result.maps[5 + out_idx], stmt_name, domain, out_buf)
    if not write.is_injective():
        reads.append(write)
    dtypes = {m.get_tuple_name(isl.dim_type.OUT): getattr(t.type, "dtype", None)
              for m, t in zip(reads, (x, cos_cache, sin_cache, pos_ids))}
    dtypes[out_buf] = _out_dtype(call, out_idx)
    return _StatementAccess(
        name=stmt_name, domain=domain, op=call,
        reads=tuple(reads), writes=(write,), params=result.param_map, dtypes=dtypes,
    )


def _rope_access(call: Call, stmt_name: str, namer, prefix: str) -> list[_StatementAccess]:
    """RoPE -> two statements, one per value input: GQA's Hq != Hkv means
    q_rope/k_rope cannot share one domain (see the task report's path A).
    Output buffers use the same ``_{out_idx}`` suffix ``_registered_access``
    uses for a same-domain multi-output op, so ``_buffer_namer``'s
    ``TupleGetItem`` passthrough resolves a downstream read to the matching
    branch's write.
    """
    q, k, cos_cache, sin_cache, pos_ids = call.args
    return [
        _rope_branch(call, q, cos_cache, sin_cache, pos_ids, 0, f"{stmt_name}_q", namer, prefix),
        _rope_branch(call, k, cos_cache, sin_cache, pos_ids, 1, f"{stmt_name}_k", namer, prefix),
    ]


def _extract_statement(call: Call, stmt_name: str, namer, prefix: str) -> list[_StatementAccess]:
    if isinstance(call.target, RoPE):
        return _rope_access(call, stmt_name, namer, prefix)
    input_types = tuple(_local_type(arg.type) for arg in call.args)
    result = build_relation(call, input_types, TypeInferContext())
    if result is not None:
        return [_registered_access(call, stmt_name, result, namer, prefix)]
    raise ExtractError(
        f"extract: op {type(call.target).__name__!r} has no "
        "registered forward type_relation (access_relation.build_relation "
        "returned None) -- register one via tilefoundry.visitor_registry."
        "access_relation.register_type_relation(...); extract has no "
        "per-op fallback."
    )


def _initial_schedule(accesses: list[_StatementAccess]) -> "isl.union_map":
    """A total execution order sufficient to seed ``compute_flow`` --
    *not* the final schedule (``kernel_schedule.py`` builds that).
    Encodes exactly m1_deps_probe.py step 4's two ingredients,
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


def _parallel_dims(domain: "isl.union_set", deps: "isl.union_map") -> dict[str, tuple[bool, ...]]:
    """Per statement, per own domain dimension, whether that dimension is
    free of dependence -- the fact ``coincident`` names in isl.

    Only a statement's *self*-dependence can constrain its own loop
    dimensions: the schedule layer sequences statements, so every
    cross-statement dependence is already satisfied by that order. A
    dimension is parallel when every self-dependence has distance 0 there
    (a matmul's k, which accumulates, is the one that is not).
    """
    sets: list["isl.set"] = []
    domain.foreach_set(sets.append)
    out: dict[str, tuple[bool, ...]] = {}
    for s in sets:
        own = s.to_union_set()
        rank = s.dim(isl.dim_type.SET)
        self_deps = deps.intersect_domain(own).intersect_range(own)
        if self_deps.is_empty():
            out[s.get_tuple_name()] = (True,) * rank
            continue
        pieces: list["isl.set"] = []
        self_deps.deltas().foreach_set(pieces.append)
        out[s.get_tuple_name()] = tuple(
            all(p.dim_min_val(d).is_zero() and p.dim_max_val(d).is_zero() for p in pieces)
            for d in range(rank)
        )
    return out


def _resolve(expr, table: dict[int, object]):
    """``expr`` if it is not (transitively) bound in ``table``, else the
    expression it resolves to -- a penetrated callee's own param Var
    bound to the caller's argument, or a penetrated wrapper Call aliased
    to whatever its body ultimately resolves to."""
    return table.get(id(expr), expr)


def _bind_dim_vars(params: tuple[Var, ...], args: tuple, callee_name: str) -> None:
    """Mirrors ``evaluator.interpreter._bind_dim_vars`` at the type level:
    each ``DimVar`` in a callee param's declared shape binds to the
    caller argument's ``ShapeDim`` at that axis. A conflicting bind for
    the same name is an actionable error naming the callee -- defense in
    depth, since ``hir.function.elaborate`` already requires a call's
    argument shapes to equal the callee's own declared ones exactly."""
    binding: dict[str, object] = {}
    for p, a in zip(params, args):
        p_shape = getattr(p.type, "shape", None)
        a_shape = getattr(a.type, "shape", None)
        if p_shape is None or a_shape is None:
            continue
        for axis, dim in enumerate(p_shape):
            if not isinstance(dim, DimVar) or axis >= len(a_shape):
                continue
            bound = a_shape[axis]
            prev = binding.get(dim.name)
            if prev is not None and prev != bound:
                raise ExtractError(
                    f"extract: call to {callee_name!r}: DimVar "
                    f"{dim.name!r} binds to conflicting shapes {prev!r} vs {bound!r}"
                )
            binding[dim.name] = bound


@dataclass(frozen=True)
class _Gathered:
    """One compute-op ``Call``, args already resolved against every
    enclosing penetrated call's argument substitution, ready for
    ``_extract_statement``. ``prefix`` is its owning scope's call-site
    tag (empty at the top level); ``stmt_name`` already carries it."""

    call: Call
    stmt_name: str
    prefix: str


def _maybe_replace_args(e: Call, resolved_args: tuple) -> Call:
    if all(r is a for r, a in zip(resolved_args, e.args)):
        return e
    return dataclasses.replace(e, args=resolved_args)


def _walk_calls(
    body, prefix: str, active: tuple[int, ...],
    site_counter: dict[str, int], table: dict[int, object],
) -> list["_Gathered"]:
    """Postorder over one function body, penetrating every ``Call``
    whose target is a ``Function``: bind its params to the caller's own
    (already-resolved) argument expressions in ``table``, recurse into
    its body under a callee-name-plus-call-site-index-prefixed scope,
    and splice the resulting statements in at the call's own position --
    then alias the wrapper call to whatever its body resolved to, so a
    sibling statement reading the wrapper's result finds the real
    producer. Every other ``Call``/``Tuple`` node gets its own
    args/elements resolved through ``table`` and registered there too,
    so a later reference (in this or an enclosing scope) sees the
    resolved form. Raises on self-recursion, a dispatch prototype or an
    arity mismatch, naming the offending callee.
    """
    order = _postorder(body)
    if any(isinstance(e, GridRegionExpr) for e in order):
        raise ExtractError(
            "extract: GridRegionExpr (looped) bodies are not "
            "supported in V1 -- only a flat SSA-DAG of compute-op Calls is"
        )

    own_targets: list[object] = []
    pending: list[object] = []  # _Gathered (nested, done) | Call (own, needs a name)
    for e in order:
        if isinstance(e, Tuple):
            resolved_elems = tuple(_resolve(x, table) for x in e.elements)
            same = all(r is x for r, x in zip(resolved_elems, e.elements))
            table[id(e)] = e if same else dataclasses.replace(e, elements=resolved_elems)
            continue
        if not isinstance(e, Call):
            continue

        target = e.target
        resolved_args = tuple(_resolve(a, table) for a in e.args)

        if isinstance(target, Function):
            callee = target
            if id(callee) in active:
                raise ExtractError(
                    f"extract: self-recursive call to {callee.name!r} "
                    "-- extract cannot unroll a function that (transitively) "
                    "calls itself"
                )
            if callee.variants or callee.body is None:
                raise ExtractError(
                    f"extract: {callee.name!r} is a dispatch "
                    "prototype (has variants / no body) -- extract has no "
                    "runtime shape to pick a variant statically"
                )
            if len(resolved_args) != len(callee.params):
                raise ExtractError(
                    f"extract: call to {callee.name!r} expects "
                    f"{len(callee.params)} arg(s), got {len(resolved_args)}"
                )
            for p, a in zip(callee.params, resolved_args):
                table[id(p)] = a
            _bind_dim_vars(callee.params, resolved_args, callee.name)
            idx = site_counter.get(callee.name, 0)
            site_counter[callee.name] = idx + 1
            nested = _walk_calls(
                callee.body, f"{prefix}{callee.name}{idx}_",
                active + (id(callee),), site_counter, table,
            )
            pending.extend(nested)
            table[id(e)] = _resolve(callee.body, table)
            continue

        if isinstance(target, (TupleGetItem, Reshape)):
            table[id(e)] = _maybe_replace_args(e, resolved_args)
            continue  # structural view, not a statement of its own

        resolved = _maybe_replace_args(e, resolved_args)
        table[id(e)] = resolved
        own_targets.append(target)
        pending.append(resolved)

    own_names = iter(_assign_statement_names(own_targets))
    gathered: list[_Gathered] = []
    for item in pending:
        if isinstance(item, _Gathered):
            gathered.append(item)
        else:
            gathered.append(
                _Gathered(call=item, stmt_name=f"{prefix}{next(own_names)}", prefix=prefix)
            )
    return gathered


def extract(hir: Function) -> TileGraph:
    """Lift ``hir``'s body into a :class:`TileGraph`: one statement per
    compute op at element granularity, penetrating every nested ``@func``
    call, with ``deps`` auto-inferred from ``reads``/``writes`` (see
    module docstring for the full algorithm)."""
    if hir.body is None:
        raise ExtractError(
            f"extract: hir Function {hir.name!r} has no body "
            "(a dispatch prototype cannot be extracted)"
        )

    gathered = _walk_calls(hir.body, "", (id(hir),), {}, {})
    if not gathered:
        raise ExtractError(
            f"extract: hir Function {hir.name!r} body has no "
            "compute ops to extract"
        )

    namer = _buffer_namer()
    accesses: list[_StatementAccess] = []
    for g in gathered:
        accesses.extend(_extract_statement(g.call, g.stmt_name, namer, g.prefix))

    domain = isl.union_set("{}")
    reads = isl.union_map("{}")
    writes = isl.union_map("{}")
    params: dict = {}
    buffer_dtypes: dict = {}
    for acc in accesses:
        domain = domain.union(acc.domain)
        for m in acc.reads:
            reads = reads.union(m)
        for m in acc.writes:
            writes = writes.union(m)
        for buf, dtype in acc.dtypes.items():
            if dtype is not None:
                buffer_dtypes.setdefault(buf, dtype)
        for name, dim in acc.params.items():
            prev = params.get(name)
            if prev is not None and prev != dim:
                raise ExtractError(
                    f"extract: isl parameter {name!r} resolves to "
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
        buffer_dtypes=buffer_dtypes, parallel_dims=_parallel_dims(domain, deps),
    )


# ---------------------------------------------------------------------------
# Facts over a time space: extents, per-statement dims, dependence distance,
# access footprint. Every one takes the time relation as data -- the schedule
# layer owns the tree these are read off, this layer only measures.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisExtent:
    """One buffer dimension's extent inside one time-space box:
    ``constant`` times the tile extent of every time dimension in ``axes``.
    ``axes`` is empty for a dimension no time dimension reaches (the whole
    of it is touched, as in ``RN[i] -> weight[j]``)."""

    axes: tuple[int, ...]
    constant: int


@dataclass(frozen=True)
class AccessFootprint:
    """One (statement, buffer) access sized per buffer dimension, so the
    element count of one time-space box is the product over ``dims``.

    Exact when every buffer dimension moves with at most one time
    dimension, an upper bound otherwise (a diagonal access ``b[t0 + t1]``
    visits fewer than ``tile[0] * tile[1]`` elements).
    """

    statement: str
    buffer: str
    is_read: bool
    dims: tuple[AxisExtent, ...]
    elem_bytes: int


def _as_map(value) -> "isl.map":
    """The single ``isl.map`` in ``value``, which isl-python returns as a
    ``union_map`` when a ``map`` is composed with one."""
    if not hasattr(value, "foreach_map"):
        return value
    maps: list["isl.map"] = []
    value.foreach_map(maps.append)
    if len(maps) != 1:
        raise ExtractError(f"expected a single map, got {len(maps)}: {value}")
    return maps[0]


def _only_out_dim(m: "isl.map", pos: int) -> "isl.map":
    n_out = m.dim(isl.dim_type.OUT)
    return m.project_out(isl.dim_type.OUT, pos + 1, n_out - pos - 1).project_out(
        isl.dim_type.OUT, 0, pos
    )


def _travels_with(m: "isl.map", pos: int) -> tuple[int, ...]:
    """The input dimensions output dimension ``pos`` of ``m`` moves with.
    The map's own domain bounds mention every input dimension, so they are
    dropped first -- only the constraints that tie ``pos`` to an input can
    answer this."""
    coupled = _only_out_dim(m.drop_constraints_not_involving_dims(isl.dim_type.OUT, pos, 1), pos)
    return tuple(
        i for i in range(m.dim(isl.dim_type.IN))
        if coupled.involves_dims(isl.dim_type.IN, i, 1)
    )


def _static_extent(s: "isl.set", pos: int, what: str) -> tuple[int, int]:
    lo, hi = s.dim_min_val(pos), s.dim_max_val(pos)
    if not (lo.is_int() and hi.is_int()):
        raise ExtractError(
            f"{what}: dimension {pos} of {s} is not statically bounded "
            "-- a parametric extent has no integer tile count"
        )
    return int(lo.num_si()), int(hi.num_si())


def time_extents(tg: TileGraph, time_map: "isl.union_map") -> tuple[int, ...]:
    """Per-dimension extent of ``time_map``'s range over ``tg.domain``.
    Raises unless every dimension starts at 0, since a tile index counts
    from the origin."""
    sets: list["isl.set"] = []
    time_map.intersect_domain(tg.domain).range().foreach_set(sets.append)
    if len(sets) != 1:
        raise ExtractError(
            f"time_extents: expected one time space, got {len(sets)} -- "
            "every statement must share the band's own range space"
        )
    box = sets[0]
    extents = []
    for i in range(box.dim(isl.dim_type.SET)):
        lo, hi = _static_extent(box, i, "time_extents")
        if lo != 0:
            raise ExtractError(
                f"time_extents: time dimension {i} starts at {lo}, not 0 -- "
                "tile counting assumes an origin-based extent"
            )
        extents.append(hi + 1)
    return tuple(extents)


def statement_time_dims(tg: TileGraph, time_map: "isl.union_map") -> dict[str, tuple[int, ...]]:
    """Per statement, one entry per time dimension: the statement's own
    domain dimension that dimension travels with, or ``-1`` when it is
    constant there (``RN[d0] -> [d0, 63, 127]`` gives ``(0, -1, -1)``).
    Raises on a skewed time dimension, which no per-axis tile size can
    describe."""
    maps: list["isl.map"] = []
    time_map.foreach_map(maps.append)
    out: dict[str, tuple[int, ...]] = {}
    for m in maps:
        name = m.get_tuple_name(isl.dim_type.IN)
        row = []
        for pos in range(m.dim(isl.dim_type.OUT)):
            involved = _travels_with(m, pos)
            if len(involved) > 1:
                raise ExtractError(
                    f"statement_time_dims: time dimension {pos} of statement "
                    f"{name!r} mixes domain dimensions {involved} ({m}) -- "
                    "a skewed band has no per-axis tile size"
                )
            row.append(involved[0] if involved else -1)
        out[name] = tuple(row)
    return out


def _buffers_by_statement(um: "isl.union_map") -> dict[str, set[str]]:
    maps: list["isl.map"] = []
    um.foreach_map(maps.append)
    out: dict[str, set[str]] = {}
    for m in maps:
        stmt = m.get_tuple_name(isl.dim_type.IN)
        out.setdefault(stmt, set()).add(m.get_tuple_name(isl.dim_type.OUT))
    return out


def carried_distances(
    tg: TileGraph, time_map: "isl.union_map", n_dims: int
) -> dict[str, tuple[int, ...]]:
    """Per buffer, the largest dependence distance isl reports along each
    time dimension. A flow dependence ``a -> b`` is attributed to every
    buffer ``a`` writes and ``b`` reads, which for a RAW must-dependence is
    exactly the memory it travels through."""
    written = _buffers_by_statement(tg.writes)
    read = _buffers_by_statement(tg.reads)
    names = {buf for bufs in (*written.values(), *read.values()) for buf in bufs}
    distances: dict[str, list[int]] = {buf: [0] * n_dims for buf in names}
    deps: list["isl.map"] = []
    tg.deps.foreach_map(deps.append)
    for dep in deps:
        carriers = written.get(dep.get_tuple_name(isl.dim_type.IN), set()) & read.get(
            dep.get_tuple_name(isl.dim_type.OUT), set()
        )
        if not carriers:
            continue
        pieces: list["isl.set"] = []
        dep.apply_domain(time_map).apply_range(time_map).deltas().foreach_set(pieces.append)
        for piece in pieces:
            for i in range(n_dims):
                lo, hi = _static_extent(piece, i, "carried_distances")
                reach = max(abs(lo), abs(hi))
                for buf in carriers:
                    distances[buf][i] = max(distances[buf][i], reach)
    return {buf: tuple(dims) for buf, dims in distances.items()}


def access_footprints(tg: TileGraph, time_map: "isl.union_map") -> tuple[AccessFootprint, ...]:
    """Every read and write of ``tg``, expressed against ``time_map``'s
    range so a tile size per time dimension sizes it (see
    :class:`AccessFootprint`)."""
    out: list[AccessFootprint] = []
    for um, is_read in ((tg.reads, True), (tg.writes, False)):
        maps: list["isl.map"] = []
        um.foreach_map(maps.append)
        for m in maps:
            stmt = m.get_tuple_name(isl.dim_type.IN)
            buf = m.get_tuple_name(isl.dim_type.OUT)
            dtype = tg.buffer_dtypes.get(buf)
            if dtype is None:
                raise ExtractError(
                    f"access_footprints: buffer {buf!r} has no recorded dtype "
                    "-- extract must resolve every accessed buffer's element type"
                )
            timed = _as_map(m.apply_domain(time_map))
            dims = []
            for pos in range(timed.dim(isl.dim_type.OUT)):
                involved = _travels_with(timed, pos)
                if involved:
                    dims.append(AxisExtent(axes=involved, constant=1))
                    continue
                lo, hi = _static_extent(
                    _only_out_dim(timed, pos).range(), 0, f"access_footprints[{buf}]"
                )
                dims.append(AxisExtent(axes=(), constant=hi - lo + 1))
            out.append(
                AccessFootprint(
                    statement=stmt, buffer=buf, is_read=is_read, dims=tuple(dims),
                    elem_bytes=math.ceil(dtype.bit_width / 8),
                )
            )
    return tuple(out)


__all__ = [
    "AccessFootprint",
    "AxisExtent",
    "ExtractError",
    "TileGraph",
    "TileUnit",
    "access_footprints",
    "carried_distances",
    "extract",
    "statement_time_dims",
    "time_extents",
]
