"""HIR / Module → ``graphviz.Digraph`` visitor (no intermediate model)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import graphviz

from tilefoundry.inspection.python_printer import (
    _DIM_FUNC_OPS,
    _DIM_INFIX_OPS,
    _collect_meshes,
    _mesh_name_map,
    _op_display_name,
    _shard_layout_str,
    _shard_layout_surface_str,
    _tensor_annotation,
    shape_entry_str,
    shard_compact_inline,
)
from tilefoundry.ir.core import Tuple as HirTuple
from tilefoundry.ir.core.expr import Call, Constant, Var
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.ir.types.tensor_type import TensorType, TupleType
from tilefoundry.ir.visitor import ExprVisitor

from .htmltable import Cell, Span, Table
from .palette import (
    DIMVAR_COLOR,
    HAIR,
    INK,
    MUTED,
    PAPER,
    depth_border,
    depth_fill,
    exprkind_color,
    storage_color,
)


def _renderable_functions(root) -> list[tuple[str, "HirFunction"]]:
    """The ``(label, function)`` units to draw.

    The ``(label, function)`` units to draw. A dispatch prototype (body
    ``None``) is expanded to its variants — each labelled by its canonical
    specialization signature — so the graph shows the executable bodies, never a
    bodyless prototype.
    """
    from tilefoundry.ir.core.module import Module  # noqa: PLC0415 — avoid import cycle
    from tilefoundry.ir.hir.function import (  # noqa: PLC0415
        Function as _HirFunction,
    )
    from tilefoundry.ir.hir.specialize import canonical_specialization_signature  # noqa: PLC0415

    funcs = root.functions if isinstance(root, Module) else [root]
    out: list[tuple[str, _HirFunction]] = []
    for fn in funcs:
        if isinstance(fn, _HirFunction) and getattr(fn, "variants", ()):
            for v in fn.variants:
                sig = canonical_specialization_signature(v.specializations)
                out.append((f"{v.name}${sig}", v))
        else:
            out.append((fn.name, fn))
    return out


def _collect_view_meshes(root) -> dict[int, "Mesh"]:
    """Collect unique ``Mesh`` objects referenced anywhere in *root*.

    Collect unique ``Mesh`` objects referenced anywhere in *root* — params,
    return type, every node's result type, and ``Reshard`` layout attrs.

    The viewer renders shard sugar on intermediate node result types too, so it
    needs a wider mesh-name map than the printer's param/return/Reshard scan:
    built on the shared ``python_printer._collect_meshes`` with
    ``include_node_types=True``.
    """
    meshes: dict[int, Mesh] = {}
    for _label, fn in _renderable_functions(root):
        meshes.update(_collect_meshes(fn, include_node_types=True))
    return meshes


@dataclass
class DetailRef:
    """Minimal click-lookup reference. NOT a graph/IR model.

    The detail-panel endpoint formats ``{kind, name, params, returns, attrs}``
    on demand from ``hir_expr``; nothing is pre-baked here.
    """
    hir_expr: Any
    kind: str
    call_path: tuple[str, ...]
    region_visual_id: str | None = None
    param_index: int | None = None


@dataclass
class DetailIndex:
    """Detail lookup index only. ``dict[visual_id, DetailRef]``.

    Carries the per-view ``mesh_name_map`` so the on-demand detail endpoint can
    render shard sugar with the same stable mesh names as the graph.
    """
    entries: dict[str, DetailRef] = field(default_factory=dict)
    mesh_name_map: dict[int, str] = field(default_factory=dict)

    def add(self, visual_id: str, ref: DetailRef) -> None:


        if visual_id in self.entries:
            raise ValueError(f"DetailIndex: duplicate visual_id {visual_id!r}")
        self.entries[visual_id] = ref

    def get(self, visual_id: str) -> DetailRef | None:
        return self.entries.get(visual_id)


_CONST_DTYPE_SUFFIX: dict[str, str] = {
    "f32": "f", "f16": "h", "bf16": "bf16", "fp8e4m3": "fp8",
}


def _format_constant(c: Constant) -> str:
    """Compact constant rendering (ported from the old viewer's pretty view).

    Compact constant rendering (ported from the old viewer's pretty
    view). Scalars: ``const(1)`` / ``const(1.0f)``; sequences:
    ``const([1.0f, 2.0f, ...])`` truncated to the first 8 elements.
    """
    ty = getattr(c, "type", None)
    suffix = (
        _CONST_DTYPE_SUFFIX.get(ty.dtype.name, "")
        if isinstance(ty, TensorType) and isinstance(ty.dtype, DType) else ""
    )

    def _fmt(v) -> str:
        if isinstance(v, bool):
            return repr(v)
        if isinstance(v, float):
            return f"{v}{suffix}"
        return repr(v)

    val = c.value
    if isinstance(ty, TensorType) and ty.shape == ():
        return f"const({_fmt(val)})"
    try:
        items = list(val)
    except TypeError:
        return f"const({_fmt(val)})"
    head = ", ".join(_fmt(v) for v in items[:8])
    tail = ", ..." if len(items) > 8 else ""
    return f"const([{head}{tail}])"


def _shard_layout_text(sl: ShardLayout, mesh_name_map: dict[int, str] | None) -> str:
    """Shard layout text.

    Render a bare ``ShardLayout`` attr (e.g. a ``Reshard`` layout) through the
    canonical sugar core, falling back to the verbose ``ShardLayout(...)`` form
    when the mesh is unnamed or the layout is not sugar-expressible.
    """
    mesh_name = mesh_name_map.get(id(sl.mesh)) if mesh_name_map else None
    if mesh_name and getattr(sl.mesh, "names", None):
        mesh_unique = mesh_name_map is not None and len(mesh_name_map) == 1
        sugar = _shard_layout_surface_str(sl, mesh_name=mesh_name, mesh_unique=mesh_unique)
        if sugar is not None:
            return sugar
    return _shard_layout_str(sl)


def _pretty_attr_value(value, *, full: bool = False, mesh_name_map: dict[int, str] | None = None) -> str:
    """Readable rendering of an Op attribute value.

    Readable rendering of an Op attribute value — pretty constants /
    tuples / lists / types instead of raw ``repr``. ``full`` selects the
    canonical type form (detail panel) vs the compact one (graph label).
    """
    if isinstance(value, Constant):
        return _format_constant(value)
    if isinstance(value, DType):
        return value.name
    if isinstance(value, (TensorType, TupleType)):
        return (
            type_to_canonical_pretty(value, mesh_name_map=mesh_name_map)
            if full
            else type_to_compact_pretty(value, mesh_name_map=mesh_name_map)
        )
    if isinstance(value, ShardLayout):
        return _shard_layout_text(value, mesh_name_map)
    if isinstance(value, tuple):
        inner = ", ".join(_pretty_attr_value(v, full=full, mesh_name_map=mesh_name_map) for v in value)
        return f"({inner}{',' if len(value) == 1 else ''})"
    if isinstance(value, list):
        return "[" + ", ".join(_pretty_attr_value(v, full=full, mesh_name_map=mesh_name_map) for v in value) + "]"
    if isinstance(value, str):
        return value
    return repr(value)


def _op_attributes(
    target, *, full: bool = False, mesh_name_map: dict[int, str] | None = None
) -> list[tuple[str, str]]:
    """Op attributes.

    The Op's non-input (attribute) params as ``(name, pretty-value)``
    pairs — e.g. ``("axis", "2")`` / ``("begin", "(const(0), ...)")``.
    ``full`` selects canonical (detail) vs compact (graph) type text.
    Empty when the Op has no attributes or doesn't expose ``params()``.
    """
    try:
        pdefs = type(target).params()
    except (AttributeError, TypeError):
        return []
    out = []
    for p in pdefs:
        if getattr(p, "kind", None) == "attribute":
            out.append((
                p.name,
                _pretty_attr_value(getattr(target, p.name, None), full=full, mesh_name_map=mesh_name_map),
            ))
    return out


class _DimSpanVisitor(ExprVisitor[list[Span]]):
    def visit_DimVar(self, dim: DimVar, ctx=None) -> list[Span]:
        return [Span(text=dim.name, color=DIMVAR_COLOR, bold=True)]

    def visit_Constant(self, dim: Constant, ctx=None) -> list[Span]:
        return [Span(text=str(dim.value))]

    def visit_Call(self, dim: Call, ctx=None) -> list[Span]:
        target = dim.target
        for op_cls, sym in _DIM_INFIX_OPS.items():
            if isinstance(target, op_cls):
                spans = list(self.visit(dim.args[0], ctx))
                spans.append(Span(text=f" {sym} "))
                spans.extend(self.visit(dim.args[1], ctx))
                return spans
        for op_cls, fname in _DIM_FUNC_OPS.items():
            if isinstance(target, op_cls):
                spans = [Span(text=f"{fname}(")]
                for i, arg in enumerate(dim.args):
                    if i:
                        spans.append(Span(text=", "))
                    spans.extend(self.visit(arg, ctx))
                spans.append(Span(text=")"))
                return spans
        return [Span(text=shape_entry_str(dim))]

    def default_visit(self, dim, ctx=None) -> list[Span]:
        return [Span(text=shape_entry_str(dim))]


def _format_dim(dim) -> list[Span]:
    """Render a shape dimension as inline viewer spans.

    Share arithmetic syntax with the canonical Python printer. Dimension
    variables use one token-class color while their text and detail identify the
    symbol; integers remain plain.
    See [inspection §2.3](docs/spec/inspection.md#23-dsl-text-forms).
    """
    return _DimSpanVisitor().visit(dim)


def _shard_inline(ty: TensorType, mesh_name_map: dict[int, str] | None):
    """Compact shard decomposition for *ty*.

    Compact shard decomposition for *ty*: ``(split_ref_by_tensor_axis,
    partials)`` or ``None`` when there is no named sugar-expressible shard
    layout (caller renders the plain shape).
    """
    layout = getattr(ty, "layout", None)
    if not isinstance(layout, ShardLayout):
        return None
    mesh_name = mesh_name_map.get(id(layout.mesh)) if mesh_name_map else None
    if not (mesh_name and getattr(layout.mesh, "names", None)):
        return None
    return shard_compact_inline(layout, mesh_name, ty.shape)


def _compact_type_spans(ty, mesh_name_map: dict[int, str] | None = None) -> list[Span]:
    """Render compact graph-label types as colored inline spans.

    Tint dimension variables and storage, inline shard splits on tensor axes,
    and append partial states. Layouts that cannot be inlined use canonical
    annotation text. ``type_to_compact_pretty`` joins spans as plain text.
    See [inspection §2.3](docs/spec/inspection.md#23-dsl-text-forms).
    """
    if isinstance(ty, TensorType):
        inline = _shard_inline(ty, mesh_name_map)
        if isinstance(ty.layout, ShardLayout) and inline is None:

            return [Span(text=type_to_canonical_pretty(ty, mesh_name_map=mesh_name_map))]
        split_ref, partials = inline if inline is not None else ({}, [])
        dtype = ty.dtype.name if hasattr(ty.dtype, "name") else str(ty.dtype)
        spans: list[Span] = [Span(text=f"{dtype}[")]
        for i, d in enumerate(ty.shape):
            if i:
                spans.append(Span(text=", "))
            spans.extend(_format_dim(d))
            if i in split_ref:
                spans.append(Span(text=f" @ {split_ref[i]}"))
        spans.append(Span(text="]"))
        if partials:
            spans.append(Span(text=" {" + ", ".join(partials) + "}"))
        storage = getattr(ty, "storage", None)
        if storage:
            spans.append(Span(text=" @"))
            spans.append(Span(text=str(storage), color=storage_color(str(storage)), bold=True))
        return spans
    if isinstance(ty, TupleType):


        spans: list[Span] = [Span(text="⟨")]
        for i, sub in enumerate(ty.fields):
            if i:
                spans.append(Span(text=", "))
            spans.extend(_compact_type_spans(sub, mesh_name_map))
        spans.append(Span(text="⟩"))
        return spans
    return [Span(text=str(ty))]


def type_to_compact_pretty(ty, mesh_name_map: dict[int, str] | None = None) -> str:
    """Render [inspection §2.3](docs/spec/inspection.md#23-dsl-text-forms) compact text."""
    return "".join(s.text for s in _compact_type_spans(ty, mesh_name_map))


def type_to_canonical_pretty(ty, mesh_name_map: dict[int, str] | None = None) -> str:
    """Render [inspection §2.3](docs/spec/inspection.md#23-dsl-text-forms) canonical text."""
    if isinstance(ty, TensorType):
        return _tensor_annotation(ty, mesh_name_map=mesh_name_map)
    if isinstance(ty, TupleType):
        return "(" + ", ".join(type_to_canonical_pretty(f, mesh_name_map) for f in ty.fields) + ")"
    return str(ty)


def _returns_of(ty, mesh_name_map: dict[int, str] | None = None) -> list[dict]:
    if isinstance(ty, TupleType):
        return [
            {"idx": i, "type": type_to_canonical_pretty(f, mesh_name_map)}
            for i, f in enumerate(ty.fields)
        ]
    if ty is None:
        return []
    return [{"idx": 0, "type": type_to_canonical_pretty(ty, mesh_name_map)}]


def format_detail(
    visual_id: str, ref: "DetailRef", mesh_name_map: dict[int, str] | None = None
) -> dict:
    """Format a detail-panel payload from a ``DetailRef`` on demand.

    Format a detail-panel payload from a ``DetailRef`` on demand (no
    pre-baked JSON in the index). Shape:
    ``{id, kind, name, params:[{name,type}], returns:[{idx,type}], attrs:[{key,value}]}``.

    ``mesh_name_map`` (the per-view map carried on ``DetailIndex``) lets the
    canonical type text render shard sugar with the same stable mesh names as
    the graph.
    """
    expr = ref.hir_expr
    name = ref.kind
    params: list[dict] = []
    attrs: list[dict] = []
    returns: list[dict] = []
    mm = mesh_name_map

    if isinstance(expr, HirFunction):
        name = expr.name
        params = [{"name": p.name, "type": type_to_canonical_pretty(p.type, mm)} for p in expr.params]
        returns = _returns_of(expr.return_type, mm)
    elif isinstance(expr, Var):
        name = expr.name
        returns = _returns_of(expr.type, mm)
    elif isinstance(expr, Constant):
        name = _format_constant(expr)
        returns = _returns_of(expr.type, mm)
    elif isinstance(expr, HirTuple):
        name = "Tuple"
        params = [{"name": f"e{i}", "type": type_to_canonical_pretty(el.type, mm)}
                  for i, el in enumerate(expr.elements)]
        returns = _returns_of(expr.type, mm)
    elif isinstance(expr, Call):
        tgt = expr.target
        if isinstance(tgt, HirFunction):
            name = tgt.name
            pnames = [p.name for p in tgt.params]
        else:
            name = _op_display_name(tgt)
            try:
                pnames = [p.name for p in type(tgt).params() if p.kind == "input"]
            except (AttributeError, TypeError):
                pnames = []
            attrs = [{"key": k, "value": v} for k, v in _op_attributes(tgt, full=True, mesh_name_map=mm)]
        params = [
            {"name": pnames[i] if i < len(pnames) else f"in{i}",
             "type": type_to_canonical_pretty(a.type, mm)}
            for i, a in enumerate(expr.args)
        ]
        returns = _returns_of(expr.type, mm)
    else:
        returns = _returns_of(getattr(expr, "type", None), mm)

    return {"id": visual_id, "kind": ref.kind, "name": name,
            "params": params, "returns": returns, "attrs": attrs}


class ViewerBuilder:
    """Walk an HIR ``Function`` or ``Module`` and emit a typed DOT graph.

    ``collapsed`` is a set of ``region_visual_id`` strings. A collapsed
    region renders as a single compact stand-in node; an expanded
    region renders as a ``subgraph cluster_<region_visual_id>`` wrapping
    its child nodes.
    """

    def __init__(self, root, collapsed: set[str] | None = None) -> None:
        self.root = root
        self.collapsed = set(collapsed or ())


        self.mesh_name_map = _mesh_name_map(_collect_view_meshes(root))
        self.index = DetailIndex(mesh_name_map=self.mesh_name_map)




        self._call_outputs: dict[int, list[str]] = {}

    def build(self) -> graphviz.Digraph:
        from tilefoundry.ir.core.module import Module  # noqa: PLC0415 — avoid import cycle
        g = graphviz.Digraph(name=self._root_name(), strict=False)
        g.attr(rankdir="TB", bgcolor="#eef4ec", compound="true", newrank="true")
        g.attr("node", shape="plain", margin="0")
        g.attr("edge", color="#555555", arrowsize="0.58", penwidth="1.0")

        if isinstance(self.root, (Module, HirFunction)):



            for i, (label, fn) in enumerate(_renderable_functions(self.root)):
                self._emit_function_region(g, fn, call_path=(label,), local_idx=i)
        else:
            raise TypeError(f"ViewerBuilder root must be HirFunction or Module, got {type(self.root).__name__}")

        return g

    def _root_name(self) -> str:
        if isinstance(self.root, HirFunction):
            return self.root.name
        return "module"

    def _visual_id(self, call_path: tuple[str, ...], local: str) -> str:
        return "__".join(call_path) + f"__{local}"

    @staticmethod
    def _output_arity(ty) -> int:
        """Number of result slots a value of type ``ty`` exposes.

        Number of result slots a value of type ``ty`` exposes — one per
        ``TupleType`` field, else a single output.
        """
        return len(ty.fields) if isinstance(ty, TupleType) else 1

    @staticmethod
    def _output_marker_row(slot: int, width: int) -> Cell:
        """Bottom ``out<i>`` marker row appended to a region's real return producer node.

        Bottom ``out<i>`` marker row appended to a region's real return
        producer node (instead of a separate anchor node).
        """
        return Cell(
            text=f"▼ out{slot}", colspan=(width if width > 1 else None),
            bgcolor="#e7efe1", color=MUTED, align="CENTER", font_size=10,
        )

    @staticmethod
    def _out_port_cells(n_out: int, width: int) -> list[Cell]:
        """Out port cells.

        Quiet ``:out<i>`` anchor cells (one per result slot) whose
        colspans sum to ``width`` so the row never goes ragged (ragged
        HTML-table rows make ``dot`` warn).
        """
        cells = []
        for i in range(n_out):
            if n_out <= width:
                base, rem = divmod(width, n_out)
                colspan = base + (1 if i >= n_out - rem else 0)
            else:
                colspan = 1
            cells.append(
                Cell(text=f"out{i}", port=f"out{i}", bgcolor="#f7f8f3",
                     color=MUTED, align="CENTER", font_size=10, colspan=colspan)
            )
        return cells




    def _emit_function_region(
        self,
        g: graphviz.Digraph,
        fn: HirFunction,
        *,
        call_path: tuple[str, ...],
        local_idx: int,
        call_args: tuple | None = None,
    ) -> tuple[str, list[str]]:
        """Emit a function region and return its title node and output refs.

        The collapsed stand-in owns output ports because it is the producer.
        Expanded regions expose actual body producers; routing them through the
        header would visually introduce a self-dependency.
        """
        region_vid = self._visual_id(call_path, f"r{local_idx}")
        node_vid = self._visual_id(call_path, f"fn{local_idx}")
        is_collapsed = region_vid in self.collapsed
        n_out = self._output_arity(fn.return_type)
        depth = (len(call_path) - 1) // 2


        self._emit_function_node(g, fn, node_vid, region_vid, call_path, collapsed=is_collapsed)
        self.index.add(
            node_vid,
            DetailRef(
                hir_expr=fn,
                kind="Function",
                call_path=call_path,
                region_visual_id=region_vid,
            ),
        )

        if is_collapsed:

            return node_vid, [f"{node_vid}:out{i}" for i in range(n_out)]



        cluster_name = f"cluster_{region_vid}"
        with g.subgraph(name=cluster_name) as sub:
            sub.attr(
                label="", style="rounded,filled",
                fillcolor=depth_fill(depth), color=depth_border(depth),
                penwidth="1.0",
            )
            visited: dict[int, str] = {}
            for i, param in enumerate(fn.params):





                visited[id(param)] = f"{node_vid}:pout{i}"
            inner_local_counter = [0]





            elements = fn.body.elements if isinstance(fn.body, HirTuple) else (fn.body,)
            outputs: list[str] = []
            direct_output_nodes: list[str] = []
            for slot, elem in enumerate(elements):
                ref = self._walk_expr(
                    sub, elem, call_path=call_path, visited=visited,
                    local_counter=inner_local_counter, output_slot=slot,
                )
                outputs.append(ref)



                if not (isinstance(elem, Call) and isinstance(elem.target, HirFunction)):
                    direct_output_nodes.append(ref.split(":")[0])

            if len(direct_output_nodes) > 1:
                with sub.subgraph() as rank:
                    rank.attr(rank="same")
                    for nid in direct_output_nodes:
                        rank.node(nid)

        return node_vid, outputs

    def _emit_function_node(
        self,
        g: graphviz.Digraph,
        fn: HirFunction,
        node_vid: str,
        region_vid: str,
        call_path: tuple[str, ...],
        *,
        collapsed: bool,
    ) -> None:
        """Title-row node.

        Title-row node. Toggle port + clickable title cell + per-param
        ports. When expanded the params span two rows — row 1 ``:pin<i>``
        (the external caller connects) and row 2 ``:pout<i>`` (the body
        reads the param) — so one port is never both an external sink and
        an internal source (untangles the edges).
        Collapsed has no body, so it shows only the ``:pin<i>`` row.
        """
        icon = "▶" if collapsed else "▼"
        n_params = len(fn.params)
        two_row = (not collapsed) and n_params > 0
        span = 2 if two_row else None

        title = Table(cellpadding=6, bgcolor=PAPER, color=HAIR)

        title.add_row(
            Cell(
                text=icon, port="toggle", href="javascript:void(0)",
                title=f"toggle:{region_vid}", bgcolor="#f7f8f3",
                width=22, align="CENTER", rowspan=span,
            ),
            Cell(
                text=f"fn {fn.name}", href="javascript:void(0)",
                title=f"expr:{node_vid}", bgcolor=exprkind_color("Function"),
                color="#ffffff", bold=True, font_size=14, cellpadding=8,
                rowspan=span,
            ),
            *[
                Cell(
                    text=p.name, port=f"pin{i}", href="javascript:void(0)",
                    title=f"expr:{node_vid}__p{i}", bgcolor="#f7f8f3",
                    color=INK, bold=True, align="CENTER", cellpadding=6,
                )
                for i, p in enumerate(fn.params)
            ],
        )

        if two_row:
            title.add_row(
                *[
                    Cell(text="▾", port=f"pout{i}", bgcolor="#f1f3ec",
                         color=MUTED, align="CENTER", font_size=10)
                    for i in range(n_params)
                ]
            )


        for i, p in enumerate(fn.params):
            self.index.add(
                f"{node_vid}__p{i}",
                DetailRef(
                    hir_expr=p, kind="Param", call_path=call_path,
                    region_visual_id=region_vid, param_index=i,
                ),
            )


        width = 2 + n_params
        title.add_row(
            Cell(
                spans=tuple(_compact_type_spans(fn.return_type, self.mesh_name_map)),
                colspan=width, color=INK, bold=True, font_size=12,
            )
        )





        if collapsed:
            title.add_row(*self._out_port_cells(self._output_arity(fn.return_type), width))

        g.node(node_vid, label=title.to_html())




    def _walk_expr(
        self,
        g: graphviz.Digraph,
        expr,
        *,
        call_path: tuple[str, ...],
        visited: dict[int, str],
        local_counter: list[int],
        output_slot: int | None = None,
    ) -> str:
        """Emit ``expr`` (and its dependencies) into ``g``.

        Emit ``expr`` (and its dependencies) into ``g``. Returns the
        DOT id (possibly with ``:port`` suffix) that an outer Call can
        attach an edge to.

        ``output_slot`` marks this expr as the function's ``i``-th return
        value: an ``out<i>`` marker row is appended to its node.
        """
        key = id(expr)
        if key in visited:
            return visited[key]

        if isinstance(expr, HirTuple):
            return self._emit_tuple(
                g, expr, call_path=call_path, visited=visited, local_counter=local_counter
            )

        if isinstance(expr, Var):

            local = f"v{local_counter[0]}"
            local_counter[0] += 1
            vid = self._visual_id(call_path, local)
            self.index.add(vid, DetailRef(hir_expr=expr, kind="Var", call_path=call_path))
            tbl = Table(cellpadding=4, bgcolor=PAPER, color=HAIR)
            tbl.add_row(
                Cell(text=f"Var {expr.name}", href="javascript:void(0)", title=f"expr:{vid}",
                     bgcolor=exprkind_color("Var"), color="#ffffff", bold=True)
            )
            tbl.add_row(Cell(spans=tuple(_compact_type_spans(expr.type, self.mesh_name_map)), color=MUTED, font_size=11))
            if output_slot is not None:
                tbl.add_row(self._output_marker_row(output_slot, 1))
            g.node(vid, label=tbl.to_html())
            visited[key] = vid
            return vid

        if isinstance(expr, Constant):
            local = f"c{local_counter[0]}"
            local_counter[0] += 1
            vid = self._visual_id(call_path, local)
            self.index.add(vid, DetailRef(hir_expr=expr, kind="Constant", call_path=call_path))
            tbl = Table(cellpadding=4, bgcolor=PAPER, color=HAIR)
            tbl.add_row(
                Cell(text=_format_constant(expr), href="javascript:void(0)", title=f"expr:{vid}",
                     bgcolor=exprkind_color("Constant"), color="#ffffff", bold=True)
            )
            tbl.add_row(Cell(spans=tuple(_compact_type_spans(expr.type, self.mesh_name_map)), color=MUTED, font_size=11))
            if output_slot is not None:
                tbl.add_row(self._output_marker_row(output_slot, 1))
            g.node(vid, label=tbl.to_html())
            visited[key] = vid
            return vid

        if isinstance(expr, Call):
            return self._emit_call(
                g, expr, call_path=call_path, visited=visited,
                local_counter=local_counter, output_slot=output_slot,
            )


        local = f"x{local_counter[0]}"
        local_counter[0] += 1
        vid = self._visual_id(call_path, local)
        self.index.add(vid, DetailRef(hir_expr=expr, kind=type(expr).__name__, call_path=call_path))
        tbl = Table()
        tbl.add_row(
            Cell(text=type(expr).__name__, href="javascript:void(0)", title=f"expr:{vid}",
                 bgcolor=MUTED, color="#ffffff", bold=True)
        )
        if output_slot is not None:
            tbl.add_row(self._output_marker_row(output_slot, 1))
        g.node(vid, label=tbl.to_html())
        visited[key] = vid
        return vid

    def _emit_call(
        self,
        g: graphviz.Digraph,
        call: Call,
        *,
        call_path: tuple[str, ...],
        visited: dict[int, str],
        local_counter: list[int],
        output_slot: int | None = None,
    ) -> str:



        if isinstance(call.target, HirFunction):
            inner_idx = local_counter[0]
            local_counter[0] += 1
            inner_path = call_path + (call.target.name, str(inner_idx))
            inner_vid, outputs = self._emit_function_region(
                g, call.target, call_path=inner_path, local_idx=inner_idx,
                call_args=call.args,
            )


            for i, arg in enumerate(call.args):
                src = self._walk_expr(
                    g, arg, call_path=call_path, visited=visited, local_counter=local_counter
                )
                g.edge(src, f"{inner_vid}:pin{i}")





            self._call_outputs[id(call)] = outputs
            result_ref = outputs[0]
            visited[id(call)] = result_ref
            return result_ref


        local = f"c{local_counter[0]}"
        local_counter[0] += 1
        vid = self._visual_id(call_path, local)
        op_label = _op_display_name(call.target)
        self.index.add(vid, DetailRef(hir_expr=call, kind=op_label, call_path=call_path))

        tbl = Table(cellpadding=6, bgcolor=PAPER, color=HAIR)
        cells = [
            Cell(
                text=op_label,
                href="javascript:void(0)",
                title=f"expr:{vid}",
                bgcolor=exprkind_color("Call"),
                color="#ffffff",
                bold=True,
                cellpadding=8,
            )
        ]



        try:
            input_param_names = [p.name for p in type(call.target).params() if p.kind == "input"]
        except (AttributeError, TypeError):
            input_param_names = []
        for i in range(len(call.args)):
            label = input_param_names[i] if i < len(input_param_names) else f"in{i}"
            cells.append(
                Cell(
                    text=label,
                    port=f"in{i}",
                    href="javascript:void(0)",
                    title=f"expr:{vid}__a{i}",
                    bgcolor="#f7f8f3",
                    color=INK,
                    bold=True,
                    align="CENTER",
                    cellpadding=4,
                )
            )
        tbl.add_row(*cells)
        width = 1 + len(call.args)




        for key, val in _op_attributes(call.target, mesh_name_map=self.mesh_name_map):
            tbl.add_row(
                Cell(text=f"{key}: {val}", colspan=width, color=MUTED,
                     font_size=11, align="LEFT")
            )
        tbl.add_row(
            Cell(
                spans=tuple(_compact_type_spans(call.type, self.mesh_name_map)),
                colspan=width,
                color=INK,
                bold=True,
                font_size=12,
            )
        )




        n_out = self._output_arity(call.type)
        if n_out > 1:
            tbl.add_row(*self._out_port_cells(n_out, width))


        if output_slot is not None:
            tbl.add_row(self._output_marker_row(output_slot, width))
        g.node(vid, label=tbl.to_html())
        visited[id(call)] = vid




        tuple_index = (
            call.target.index if _op_display_name(call.target) == "TupleGetItem" else None
        )
        for i, arg in enumerate(call.args):
            src = self._walk_expr(
                g, arg, call_path=call_path, visited=visited, local_counter=local_counter
            )
            if tuple_index is not None and i == 0:





                outs = self._call_outputs.get(id(arg))
                if outs is not None:
                    src = outs[tuple_index]
                elif ":" not in src:
                    src = f"{src}:out{tuple_index}"
            g.edge(src, f"{vid}:in{i}")

        return vid

    def _emit_tuple(
        self,
        g: graphviz.Digraph,
        tup: HirTuple,
        *,
        call_path: tuple[str, ...],
        visited: dict[int, str],
        local_counter: list[int],
    ) -> str:
        """Emit a value-form ``Tuple`` bundler.

        Emit a value-form ``Tuple`` bundler. Each element flows into an
        ``:in<i>`` port; a consuming ``TupleGetItem`` reads field ``i`` from
        the matching ``:out<i>`` port. (A ``return (...)`` Tuple is handled
        directly by ``_emit_function_region`` and never reaches here.)
        """
        local = f"t{local_counter[0]}"
        local_counter[0] += 1
        vid = self._visual_id(call_path, local)
        self.index.add(vid, DetailRef(hir_expr=tup, kind="Tuple", call_path=call_path))

        n = len(tup.elements)
        tbl = Table(cellpadding=6, bgcolor=PAPER, color=HAIR)
        tbl.add_row(
            Cell(
                text="Tuple",
                href="javascript:void(0)",
                title=f"expr:{vid}",
                bgcolor=exprkind_color("Tuple"),
                color="#ffffff",
                bold=True,
                colspan=max(1, n),
                cellpadding=8,
            )
        )
        tbl.add_row(
            *[
                Cell(text=f"in{i}", port=f"in{i}", bgcolor="#f7f8f3", color=INK,
                     align="CENTER", font_size=10)
                for i in range(n)
            ]
        )
        tbl.add_row(
            *[
                Cell(text=f"out{i}", port=f"out{i}", bgcolor="#f7f8f3", color=MUTED,
                     align="CENTER", font_size=10)
                for i in range(n)
            ]
        )
        g.node(vid, label=tbl.to_html())
        visited[id(tup)] = vid

        for i, elem in enumerate(tup.elements):
            src = self._walk_expr(
                g, elem, call_path=call_path, visited=visited, local_counter=local_counter
            )
            g.edge(src, f"{vid}:in{i}")

        return vid


__all__ = [
    "DetailIndex", "DetailRef", "ViewerBuilder",
    "format_detail", "type_to_compact_pretty", "type_to_canonical_pretty",
]
