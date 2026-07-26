"""``emit_scaffold(tg) -> (Skeleton, Swimlane, list[HoleContract])`` --
turn ``tg``'s isl schedule tree (``tg.tree``, filled in by
``compute_schedule()``) into something an agent (or a human) can fill in:
a holed C-like loop skeleton, a human-readable Mermaid swimlane, and one
``HoleContract`` per statement. ``TileGraph`` carries the model
(``reads``/``writes``/``units``) and the schedule tree together, so this
is the single argument the whole render needs.

BufferAccess reuse note (flagged): ``src/tilefoundry/ir/tir/memory/
tensor_view.py`` already defines a ``TensorView``, but it is a TIR ``Op``
-- constructed via ``ParamDef``/``register_op``/``register_typeinfer`` and
driven through the HIR->TIR pass machinery, wrapping a real memory
``Expr`` plus a ``Layout``/``ShardLayout``. This render step lives
*alongside* that path, never importing it; and structurally, ``TileGraph``
does not expose a buffer-name -> HIR-value table (that mapping is private
to ``analysis.poly``'s own namer), so there is no memory ``Expr`` here to
wrap even if we did import it. What a ``HoleContract`` actually needs -- one
buffer's polyhedral access footprint at one statement, i.e. an
``isl.map`` from statement coordinates to buffer elements -- is a
different concept from a TIR slice/layout view besides sharing a name.
V1 therefore falls back to a minimal local ``BufferAccess`` dataclass
(``tensor_name``/``index_map``/``dtype``), with ``dtype`` best-effort
recovered from HIR type info (see ``_dtype_table``).

Skeleton rendering note: the validated hook probe
(``m2_hook_probe.py``) flagged ``isl.id.alloc`` as the wrong constructor.
The right one is the positional ``isl.id(name, user)`` (confirmed by
probing ``dir(isl.id)``/``help``); but probing further showed
``ast_node.set_annotation``/``get_annotation`` round-tripping through
``ast_build.node_from`` is unreliable once a non-``None`` Python ``user``
payload is attached (``ann.name()`` raises ``AttributeError`` on the
walked-back node, even though the same id's ``.name()`` works fine right
after construction) -- so this module does not use isl node annotations at
all. Instead ``at_each_domain`` reads each statement call's structure
natively via ``ast_expr_op_call.op_arg()``/``op_n_arg()`` (statement name +
coordinate sub-exprs, each rendered with their own ``to_C_str()`` --
never hand-parsed from text), and the *substitution* of the naked call for
the decorated hole call is a plain, cursor-ordered string splice against
``ast.to_C_str()``'s final text (isl's own printer has no notion of a
custom call-with-comments-and-extra-args shape, so there is no native isl
API for that part regardless of the annotation question).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import isl

from tilefoundry.analysis.poly import TileGraph, TileUnit
from tilefoundry.ir.core import Var, binding_name


class EmitScaffoldError(RuntimeError):
    """A construct `emit_scaffold` does not (yet) support, or a ``tg``
    consistency precondition that did not hold -- always raised with
    a specific, actionable message; V1 never silently guesses."""


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Skeleton:
    """A holed, C-like loop-nest skeleton: isl ``ast_build`` codegen (PoC
    11) with every naked statement call (``MM(c0, c1, c2);``) rendered as
    a ``HOLE_<name>(...)`` call instead (see ``_build_skeleton``).
    ``holes`` names every hole in ``text``, in first-appearance order.
    """

    text: str
    holes: tuple[str, ...]


@dataclass(frozen=True)
class Swimlane:
    """A human-readable Mermaid ``gantt`` rendering of the schedule: one
    section (swimlane) per statement, minimally unrolled (prologue + a
    handful of steady-state iterations + epilogue, see
    ``_illustrative_instances``) rather than the full iteration count."""

    text: str


@dataclass(frozen=True)
class BufferAccess:
    """V1 fallback view of one buffer touched by one statement (see the
    module docstring's "BufferAccess reuse note" for why the existing TIR
    ``TensorView`` is not reused here). ``index_map`` is the ``isl.map``
    (from ``TileGraph.reads``/``writes``) taking this statement's
    coordinates to elements of buffer ``tensor_name``; ``dtype`` is
    best-effort HIR dtype recovery (``None`` if it could not be resolved,
    e.g. an unbound intermediate -- see ``_dtype_table``)."""

    tensor_name: str
    index_map: "isl.map"
    dtype: object | None


@dataclass(frozen=True)
class HoleContract:
    """One per statement: what a hole must compute, as a pure function
    ``(inputs, coords) -> output`` -- no side effects, no indexing/sync (the
    skeleton already carries those). ``op_ref`` is ``TileUnit.op`` (the HIR
    ``Call``) so M3 can fill the hole and diff it against the HIR
    Evaluator's own op subgraph result; M2 itself never runs it."""

    name: str
    op_ref: object
    inputs: tuple[BufferAccess, ...]
    output: BufferAccess
    coords: tuple[str, ...]


# ---------------------------------------------------------------------------
# Shared: buffer name -> dtype / per-statement access maps
# ---------------------------------------------------------------------------


def _dtype_table(units: tuple[TileUnit, ...]) -> dict[str, object]:
    """buffer name -> HIR dtype, reconstructed by re-applying poly.py's
    own naming rule (``Var.name`` / ``binding_name``) to every HIR value any
    statement touches: each ``op.args[i]`` and each statement's own output
    (``binding_name(op)``). ``TileGraph`` does not expose
    ``poly._buffer_namer``'s ``id(expr) -> name`` table, so this is a
    best-effort re-derivation of the common (no rename-collision) case, not
    a byte-for-byte reuse. Two documented V1 gaps: (1) a real
    ``poly.py`` naming collision (its numeric-suffix fallback) is not
    reproduced here -- last-write-wins on the colliding key; (2) an
    unbound intermediate (no source-level name, ``binding_name`` is
    ``None``) is skipped, so its buffer keeps ``dtype=None`` downstream.
    Neither gap is hit by today's MM/RN extraction."""
    table: dict[str, object] = {}
    for unit in units:
        call = unit.op
        out_name = binding_name(call)
        if out_name is not None:
            table[out_name] = getattr(call.type, "dtype", None)
        for arg in call.args:
            name = arg.name if isinstance(arg, Var) else binding_name(arg)
            if name is not None:
                table[name] = getattr(arg.type, "dtype", None)
    return table


def _by_buf(union_map: "isl.union_map", stmt_name: str) -> dict[str, "isl.map"]:
    """Decompose ``union_map`` (``tg.reads`` or ``tg.writes``) into the
    per-buffer maps whose ``IN`` tuple is ``stmt_name``, keyed by ``OUT``
    tuple (buffer) name."""
    maps: list["isl.map"] = []
    union_map.foreach_map(maps.append)
    return {
        m.get_tuple_name(isl.dim_type.OUT): m
        for m in maps
        if m.get_tuple_name(isl.dim_type.IN) == stmt_name
    }


def _ordered_inputs(
    unit: TileUnit, read_by_buf: dict[str, "isl.map"], dtype_table: dict[str, object]
) -> tuple[BufferAccess, ...]:
    """Reads of ``unit``, as ``BufferAccess``es ordered to match
    ``unit.op.args`` (the natural, human-readable source-call order) --
    e.g. MM's ``(x, w)`` before its own read-modify-write self-read on
    ``h``. Any read not reachable from ``op.args`` (that self-read: the
    output buffer, read again because its write map is not injective --
    see ``poly._registered_access``) is appended after, sorted by
    buffer name for determinism, since ``foreach_map``'s own union-map
    iteration order is not guaranteed stable."""
    ordered: list[str] = []
    used: set[str] = set()
    for arg in unit.op.args:
        name = arg.name if isinstance(arg, Var) else binding_name(arg)
        if name in read_by_buf and name not in used:
            ordered.append(name)
            used.add(name)
    for name in sorted(read_by_buf):
        if name not in used:
            ordered.append(name)
            used.add(name)
    return tuple(
        BufferAccess(tensor_name=name, index_map=read_by_buf[name], dtype=dtype_table.get(name))
        for name in ordered
    )


def _output_view(
    unit: TileUnit, write_by_buf: dict[str, "isl.map"], dtype_table: dict[str, object]
) -> BufferAccess:
    if len(write_by_buf) != 1:
        raise EmitScaffoldError(
            f"emit_scaffold: statement {unit.name!r} writes {len(write_by_buf)} "
            "buffers -- HoleContract.output is singular in V1 (a real "
            "multi-output statement is future work, see poly.py's "
            "n_outputs support in _registered_access)"
        )
    ((name, m),) = write_by_buf.items()
    return BufferAccess(tensor_name=name, index_map=m, dtype=dtype_table.get(name))


# ---------------------------------------------------------------------------
# ① Skeleton
# ---------------------------------------------------------------------------


def _call_coords(expr: "isl.ast_expr") -> tuple[str, tuple[str, ...]]:
    """Decode an isl ``ast_expr_op_call`` (e.g. ``MM(c0, c1, c2)``, from a
    domain leaf's ``node.get_expr()``) into ``(stmt_name, coord_texts)``.
    ``op_arg(0)`` is always the statement-name id; every following
    ``op_arg`` (there are ``op_n_arg() - 1`` of them) is a coordinate,
    rendered via its own ``to_C_str()`` -- not hand-parsed from text --
    so a non-trivial affine coordinate (e.g. a future strip-mined
    schedule) would still print correctly, not just the bare
    iterator-id case this V1 schedule always produces."""
    name = expr.op_arg(0).id().name()
    coords = tuple(expr.op_arg(i).to_C_str() for i in range(1, expr.op_n_arg()))
    return name, coords


def _ring_ref(buf_name: str, coords: tuple[str, ...], ring: dict) -> str:
    """Render one buffer reference for a hole call. ``compute_schedule()``
    always leaves ``tg.ring`` at ``{}``; ``solve_resources()`` is the producer,
    so this falls through to the bare buffer name until that has run --
    then indexes the buffer by its innermost coordinate mod ``N``."""
    n = ring.get(buf_name)
    if not n or n <= 1:
        return buf_name
    return f"{buf_name}[{coords[-1]} % {n}]"


def _render_hole_call(
    hole_name: str, in_refs: tuple[str, ...], out_ref: str, coords: tuple[str, ...]
) -> str:
    """``HOLE_<name>(/*in*/ a, b, /*out*/ c, /*coords*/ c0, c1, c2);`` --
    the hole's *inputs* (all reads, including any RMW self-read on the
    output buffer -- included honestly rather than silently dropped, see
    ``_ordered_inputs``), its *output*, and the raw schedule coordinates
    it is parametrised by, each behind its own comment marker."""
    sections = [
        "/*in*/ " + ", ".join(in_refs),
        "/*out*/ " + out_ref,
        "/*coords*/ " + ", ".join(coords),
    ]
    return f"{hole_name}({', '.join(sections)});"


def _line_indent(text: str, pos: int) -> str:
    """Leading whitespace of the line containing ``text[pos]``."""
    start = text.rfind("\n", 0, pos) + 1
    return text[start:pos]


def _wrap_hole_stmt(indent: str, hole_line: str) -> str:
    """Wrap a hole call + its conservative sync placeholder in their own
    brace block, so the substitution is always a single valid C-like
    statement regardless of whether isl's printer had originally left the
    replaced call brace-less (a single-statement loop body, e.g. the
    innermost ``MM`` loop, prints with no braces at all)."""
    inner = indent + "  "
    return f"{{\n{inner}{hole_line}\n{inner}// barrier\n{indent}}}\n"


def _build_skeleton(
    tg: TileGraph, dtype_table: dict[str, object]
) -> tuple[Skeleton, dict[str, HoleContract]]:
    """isl ``ast_build`` codegen (PoC 11) over ``tg.tree``, with an
    ``at_each_domain`` hook (validated in ``m2_hook_probe.py``) that
    records each statement's hole-call replacement text in visit order,
    then splices those replacements into the final ``to_C_str()`` text.
    Also builds each statement's ``HoleContract`` along the way (first
    occurrence only -- one contract per *statement*, not per call site)."""
    if tg.tree is None:
        raise EmitScaffoldError(
            "emit_scaffold: tg.tree is None -- call compute_schedule(tg) "
            "before emit_scaffold(tg)"
        )
    units_by_name = {u.name: u for u in tg.units}
    contracts: dict[str, HoleContract] = {}
    occurrences: list[tuple[str, str]] = []  # (original call text, wrapped replacement)

    def at_each_domain(node, build):
        expr = node.get_expr()
        stmt_name, coords = _call_coords(expr)

        contract = contracts.get(stmt_name)
        if contract is None:
            unit = units_by_name.get(stmt_name)
            if unit is None:
                raise EmitScaffoldError(
                    f"emit_scaffold: schedule tree statement {stmt_name!r} has no "
                    "matching TileUnit in tg.units -- tg.tree and tg.units must come "
                    "from the same extract()/compute_schedule() pipeline run"
                )
            read_by_buf = _by_buf(tg.reads, stmt_name)
            write_by_buf = _by_buf(tg.writes, stmt_name)
            inputs = _ordered_inputs(unit, read_by_buf, dtype_table)
            output = _output_view(unit, write_by_buf, dtype_table)
            contract = HoleContract(
                name=f"HOLE_{stmt_name}",
                op_ref=unit.op,
                inputs=inputs,
                output=output,
                coords=coords,
            )
            contracts[stmt_name] = contract

        in_refs = tuple(_ring_ref(v.tensor_name, coords, tg.ring) for v in contract.inputs)
        out_ref = _ring_ref(contract.output.tensor_name, coords, tg.ring)
        hole_line = _render_hole_call(contract.name, in_refs, out_ref, coords)
        occurrences.append((node.to_C_str(), hole_line))
        return node

    build = isl.ast_build.from_context(isl.set("{ : }")).set_at_each_domain(at_each_domain)
    ast = build.node_from(tg.tree)
    text = ast.to_C_str()

    rendered: list[str] = []
    cursor = 0
    for original, hole_line in occurrences:
        idx = text.index(original, cursor)
        indent = _line_indent(text, idx)
        rendered.append(text[cursor:idx])
        rendered.append(_wrap_hole_stmt(indent, hole_line))
        cursor = idx + len(original)
    rendered.append(text[cursor:])

    holes = tuple(contract.name for contract in contracts.values())
    return Skeleton(text="".join(rendered), holes=holes), contracts


# ---------------------------------------------------------------------------
# ② Swimlane
# ---------------------------------------------------------------------------


def _statement_extents(tg: TileGraph, stmt_name: str) -> tuple[int, ...]:
    """Per-axis ``[lo, hi]`` extent of ``stmt_name``'s own domain piece
    (same ``dim_min_val``/``dim_max_val`` technique, applied to the one
    ``isl.set`` in ``tg.domain`` whose tuple name matches)."""
    sets: list["isl.set"] = []
    tg.domain.foreach_set(sets.append)
    for s in sets:
        if s.get_tuple_name() == stmt_name:
            rank = s.dim(isl.dim_type.SET)
            return tuple(
                int(s.dim_max_val(i).num_si()) - int(s.dim_min_val(i).num_si()) + 1
                for i in range(rank)
            )
    raise EmitScaffoldError(f"emit_scaffold: no domain set found for statement {stmt_name!r}")


def _illustrative_instances(
    extents: tuple[int, ...],
) -> tuple[list[tuple[int, ...]], int]:
    """Minimal loop unrolling for the swimlane: the first instance
    (prologue) + up to ``depth + 1`` following instances (steady-state) +
    the last instance (epilogue) -- never the full ``K``-instance unroll.
    Returns ``(shown, n_collapsed)``; ``n_collapsed == 0`` when the whole
    domain already fits in prologue+steady+epilogue (nothing to elide)."""
    depth = len(extents)
    all_coords = list(itertools.product(*(range(e) for e in extents)))
    total = len(all_coords)
    head_n = min(1 + (depth + 1), total)
    head = all_coords[:head_n]
    tail = all_coords[-1:] if total > head_n else []
    shown = head + [c for c in tail if c not in head]
    return shown, total - len(shown)


def _swimlane_lines(tg: TileGraph, contracts: dict[str, HoleContract]) -> list[str]:
    lines = [
        "```mermaid",
        "gantt",
        "    title tilefoundry scaffold -- statement swimlanes",
        "    dateFormat  X",
        "    axisFormat  %s",
    ]
    tick = 0
    for stmt_name in contracts:
        extents = _statement_extents(tg, stmt_name)
        shown, collapsed = _illustrative_instances(extents)
        lines.append(f"    section {stmt_name}")
        n_shown = len(shown)
        for idx, coord in enumerate(shown):
            is_last = idx == n_shown - 1
            if collapsed and is_last:
                lines.append(f"    … ×{collapsed} 省略 :{tick}, 1d")
                tick += 1
            if not collapsed:
                role = ""
            elif idx == 0:
                role = " (prologue)"
            elif is_last:
                role = " (epilogue)"
            else:
                role = " (steady)"
            label = f"{stmt_name}({', '.join(str(c) for c in coord)}){role}"
            lines.append(f"    {label} :{tick}, 1d")
            tick += 1
    lines.append("```")
    return lines


def _build_swimlane(tg: TileGraph, contracts: dict[str, HoleContract]) -> Swimlane:
    return Swimlane(text="\n".join(_swimlane_lines(tg, contracts)))


# ---------------------------------------------------------------------------
# emit_scaffold
# ---------------------------------------------------------------------------


def emit_scaffold(tg: TileGraph) -> tuple[Skeleton, Swimlane, list[HoleContract]]:
    """Render ``tg`` (carrying its isl schedule tree, from
    ``compute_schedule(tg)``) into a holed skeleton + a human swimlane + one
    ``HoleContract`` per statement. See the module docstring for the
    ``BufferAccess``-reuse decision."""
    dtype_table = _dtype_table(tg.units)
    skeleton, contracts = _build_skeleton(tg, dtype_table)
    swimlane = _build_swimlane(tg, contracts)
    return skeleton, swimlane, list(contracts.values())


__all__ = [
    "Skeleton",
    "Swimlane",
    "BufferAccess",
    "HoleContract",
    "EmitScaffoldError",
    "emit_scaffold",
]
